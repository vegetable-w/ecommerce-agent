"""runtime のイベントを SSE のフレームへ写す共通層。

SSE で答える入口は 2 つある。POST /api/chat(発話の 1 turn)と
POST /api/actions/resume(注文を選んだ後の再開)で、**流れるイベントは同じ**。
再開の後には Agent の回答がそのまま続くので、画面は選択の前後で描画を分けずに済む。

だから変換も 1 か所に置く。2 か所に書くと、新しいイベントを足したときに片方だけを
直す事故が起きて、「回答は出るが出典は出ない」のような半端な壊れ方をする。

何を流して何を流さないかの判断は app/graph/runtime.py にある。ここが担うのは
「イベント dict → SSE フレーム」の変換と、例外 → エラーフレームの対応付けだけ。
"""

import json
import logging
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime

logger = logging.getLogger(__name__)

_NOT_FOUND_MSG = "会話が見つかりません"
_DB_DOWN_MSG = "データベースを一時的に利用できません。しばらくしてからもう一度お試しください"
_UPSTREAM_MSG = "上流モデルを一時的に利用できません。しばらくしてからもう一度お試しください"


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _error_frames(message: str) -> AsyncIterator[str]:
    # "event: error\n" と "data: {...}\n\n" の2回に分けて yield するが、
    # 連結すると `event: error\ndata: {...}\n\n` となり SSE の1フレームとして正しい。
    yield "event: error\n"
    yield sse({"message": message})


def frame(ev: dict) -> str | None:
    """runtime のイベント 1 つを SSE のフレームにする。知らない種類は None。

    知らないイベントで turn ごと落とすより、画面が解釈できる分だけ届く方がよい。

    引用本文にも選択肢の draft にも注文の商品名にも、改行と日本語が入りうる。
    json.dumps がそれを \\n へエスケープするので、SSE のフレーム区切り("\\n\\n")とは
    衝突しない。本文を生のまま流す「簡略化」をしないこと(1 フレームが割れて
    フロントエンドの JSON.parse が両方失敗し、引用や選択肢が丸ごと消える)。
    """
    kind = ev.get("type")
    if kind == "tool":
        return sse({"event": "tool", "name": ev["name"]})
    if kind == "delta":
        return sse({"delta": ev["text"]})
    if kind == "citations":
        return sse({"event": "citations", "items": ev["items"]})
    if kind == "actions":
        return sse({"event": "actions", "items": ev["items"]})
    if kind == "interrupt":
        # 選択待ちで止まった。この後 done は来ないので、画面が会話 ID を知る手立ては
        # このフレームしかない(初回ターンで中断されると /api/actions/resume を
        # 叩けなくなる)。kind は中断の種類で、画面はこれで描き分ける。
        return sse({"event": "interrupt", "kind": ev["kind"], "orders": ev["orders"],
                    "conversation_id": ev["conversation_id"]})
    if kind == "done":
        return sse({"event": "done", "conversation_id": ev["conversation_id"]})
    return None


async def event_stream(events: AsyncIterator[dict], *, context: str) -> AsyncIterator[str]:
    """イベントの列を SSE のフレームの列にする。例外はエラーフレームへ写す。

    context はログ用の識別子(user_id / conversation_id)。例外そのものは detail へ
    載せない(接続文字列や SQL が画面まで届く)。

    **中断で終わった場合も [DONE] を出す。** 役割が違うため:
    [DONE] は「この HTTP レスポンスは終わり」、done イベントは「turn が完結した」。
    選択待ちは完結していないので done は来ないが、レスポンスは閉じる必要がある
    (閉じないと画面は待ち続ける)。画面は interrupt を見たら一覧を描いて、
    ユーザーの選択を /api/actions/resume へ送ればよい。
    """
    try:
        async for ev in events:
            f = frame(ev)
            if f is not None:
                yield f
    except runtime.ConversationNotFound:
        # 注意: この例外はジェネレータの内側、つまり既に HTTP 200 と
        # text/event-stream ヘッダを送出した後に発生する。従って 404 にはできず、
        # エラーフレームとして返すのが唯一の選択肢になる。
        for f in _error_frames(_NOT_FOUND_MSG):
            yield f
        return
    except SQLAlchemyError:
        logger.exception("データベースエラー %s", context)
        for f in _error_frames(_DB_DOWN_MSG):
            yield f
        return
    except Exception:
        # 既知かつ意図的な不整合(02 章でレビュー承認済み): ここは上流障害だけでなく
        # こちら側のバグも捕まえるが、文言は一律「上流モデルを一時的に利用できません」に
        # なる。/api/agent 側では同じ状況を 500 と 502 に区別しているので、その原則とは
        # 食い違っている。それでもこうしているのは、ここが既に HTTP 200 とヘッダを
        # 送出した後であり、ステータスコードで区別する手段が物理的に残っていないため。
        logger.exception("graph ストリーミング失敗 %s", context)
        for f in _error_frames(_UPSTREAM_MSG):
            yield f
        return
    # 意図的な差異: エラーで終わったときは [DONE] を送らずに return する。
    # static/index.html の解析ループは reader.read() の done(= HTTP ストリームが
    # 閉じたこと)で終了し、[DONE] は continue で読み飛ばすだけなので、省いても
    # フロントエンドの挙動は変わらない。正常終了とエラー終了を「[DONE] が来たか」で
    # 区別できる分、下流には扱いやすい。
    yield "data: [DONE]\n\n"


def stream_response(events: AsyncIterator[dict], *, context: str) -> StreamingResponse:
    """SSE の StreamingResponse。2 つの入口が同じヘッダで返すようにする。"""
    return StreamingResponse(
        event_stream(events, context=context),
        media_type="text/event-stream",
        # リバースプロキシによるバッファリング対策
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
