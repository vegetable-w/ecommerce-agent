"""チャット画面のストリーミング出口(SSE)。

- POST /api/chat : フロントエンドの主入口

05 章でこのエンドポイントは graph の astream 入口になった(spec §7 / D2)。
1 章の SessionStore による履歴保持はもう使わない。会話の State は checkpointer が
thread_id(= conversation_id)ごとに持つので、履歴を組み立てる責務はここには無い。

ここが担うのは「イベント dict → SSE フレーム」の変換と、例外 → エラーフレームの
対応付けだけ。何を流して何を流さないかの判断は app/graph/runtime.py にある。
"""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: str) -> AsyncIterator[str]:
    # "event: error\n" と "data: {...}\n\n" の2回に分けて yield するが、
    # 連結すると `event: error\ndata: {...}\n\n` となり SSE の1フレームとして正しい。
    yield "event: error\n"
    yield _sse({"message": message})


@router.post("/api/chat")
async def chat(req: ChatRequest):
    async def event_stream() -> AsyncIterator[str]:
        try:
            # runtime.stream_turn は属性経由で呼ぶこと(from ... import stream_turn に
            # しない)。テストの monkeypatch.setattr(runtime, "stream_turn", ...) が
            # 効かなくなり、本物の上流 LLM と本番相当の DB へ流れ落ちる。
            async for ev in runtime.stream_turn(req.user_id, req.message, req.conversation_id):
                if ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "citations":
                    # 引用本文にも選択肢の draft にも改行と日本語が入りうる。
                    # json.dumps がそれを \n へエスケープするので、SSE のフレーム区切り
                    # ("\n\n")とは衝突しない。本文を生のまま流す「簡略化」をしないこと
                    # (1 フレームが割れてフロントエンドの JSON.parse が両方失敗し、
                    # 引用や選択肢が丸ごと消える)。
                    yield _sse({"event": "citations", "items": ev["items"]})
                elif ev["type"] == "actions":
                    yield _sse({"event": "actions", "items": ev["items"]})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except runtime.ConversationNotFound:
            # 注意: この例外はジェネレータの内側、つまり既に HTTP 200 と
            # text/event-stream ヘッダを送出した後に発生する。従って 404 にはできず、
            # エラーフレームとして返すのが唯一の選択肢になる。
            for f in _sse_error("会話が見つかりません"):
                yield f
            return
        except SQLAlchemyError:
            logger.exception("データベースエラー user_id=%s", req.user_id)
            for f in _sse_error("データベースを一時的に利用できません。しばらくしてからもう一度お試しください"):
                yield f
            return
        except Exception:
            # 既知かつ意図的な不整合(02 章でレビュー承認済み): ここは上流障害だけでなく
            # こちら側のバグも捕まえるが、文言は一律「上流モデルを一時的に利用できません」に
            # なる。/api/agent 側では同じ状況を 500 と 502 に区別しているので、その原則とは
            # 食い違っている。それでもこうしているのは、ここが既に HTTP 200 とヘッダを
            # 送出した後であり、ステータスコードで区別する手段が物理的に残っていないため。
            logger.exception("graph ストリーミング失敗 user_id=%s", req.user_id)
            for f in _sse_error("上流モデルを一時的に利用できません。しばらくしてからもう一度お試しください"):
                yield f
            return
        # 意図的な差異: エラーで終わったときは [DONE] を送らずに return する。
        # static/index.html の解析ループは reader.read() の done(= HTTP ストリームが
        # 閉じたこと)で終了し、[DONE] は continue で読み飛ばすだけなので、省いても
        # フロントエンドの挙動は変わらない。正常終了とエラー終了を「[DONE] が来たか」で
        # 区別できる分、下流には扱いやすい。
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # リバースプロキシによるバッファリング対策
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
