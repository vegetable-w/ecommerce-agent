"""エージェントの HTTP 出口。

- POST /api/agent/stream : フロントエンドの主入口(SSE)
- POST /api/agent        : プログラム/テスト用の入口(非ストリーミング JSON)

オーケストレーション本体は app/core/agent.py にあり、ここは「イベント dict → SSE フレーム」
「AgentResult → JSON」の変換と例外 → エラー表現のマッピングだけを担当する。
"""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.core import agent
from app.core.llm import get_chat_model
from app.schemas.agent import AgentRequest, AgentResponse, ToolCallView, ToolResultView

logger = logging.getLogger(__name__)
router = APIRouter()


def get_model() -> BaseChatModel:
    return get_chat_model(streaming=True)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: str) -> AsyncIterator[str]:
    # "event: error\n" と "data: {...}\n\n" の2回に分けて yield するが、
    # 連結すると `event: error\ndata: {...}\n\n` となり SSE の1フレームとして正しい
    # (app/api/chat.py と同じ形)。
    yield "event: error\n"
    yield _sse({"message": message})


@router.post("/api/agent/stream")
async def agent_stream(req: AgentRequest, model: BaseChatModel = Depends(get_model)):
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev in agent.stream_agent_turn(
                req.user_id, req.message, req.conversation_id, model=model
            ):
                if ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "citations":
                    # 引用本文には改行が入りうる。json.dumps がそれを \n へ
                    # エスケープするので、SSE のフレーム区切り("\n\n")とは衝突しない。
                    # 本文を生のまま流す「簡略化」をしないこと(1 フレームが割れて
                    # フロントエンドの JSON.parse が両方失敗し、引用が丸ごと消える)。
                    yield _sse({"event": "citations", "items": ev["items"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except agent.ConversationNotFound:
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
            # 既知かつ意図的な不整合(レビュー承認済み): ここは上流障害だけでなく
            # こちら側のバグも捕まえるが、文言は一律「上流モデルを一時的に利用できません」に
            # なる。/api/agent 側では同じ状況を「サーバー側の欠陥はサーバーエラーとして出す」
            # 方針で 500 と 502 に区別しているので、その原則とは食い違っている。
            # それでもこうしているのは、ここが既に HTTP 200 とヘッダを送出した後であり、
            # ステータスコードで区別する手段が物理的に残っていないため。文言を増やすより、
            # 下流には「エラーで終わった」とだけ伝え、詳細は logger.exception に委ねる。
            logger.exception("エージェントオーケストレーション失敗 user_id=%s", req.user_id)
            for f in _sse_error("上流モデルを一時的に利用できません。しばらくしてからもう一度お試しください"):
                yield f
            return
        # 意図的な差異(「簡略化」して chapter 1 に揃えないこと): /api/chat はエラーフレームの
        # 後にも data: [DONE] を送るが、こちらは送らずに return する。static/index.html の
        # 解析ループは reader.read() の done(= HTTP ストリームが閉じたこと)で終了し、
        # [DONE] は continue で読み飛ばすだけ、error 分岐は自前で後始末をするため、
        # 省いてもフロントエンドの挙動は変わらない。正常終了とエラー終了を
        # 「[DONE] が来たかどうか」で区別できる分、こちらの方が下流には扱いやすい。
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # リバースプロキシによるバッファリング対策(chapter 1 の /api/chat と同じ理由)
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/agent", response_model=AgentResponse)
async def run_agent(req: AgentRequest) -> AgentResponse:
    # agent.run_agent_turn と属性経由で呼ぶこと(from ... import run_agent_turn にしない)。
    # テストの monkeypatch.setattr(agent, "run_agent_turn", ...) が効かなくなるため。
    #
    # 注意(テストを書く人向けの落とし穴): このハンドラは model を Depends で受け取らず、
    # run_agent_turn の内部で get_chat_model() を組み立てる。したがって
    # app.dependency_overrides[agent_api.get_model] はこちらには一切効かない
    # (効くのは /api/agent/stream だけ)。両方のエンドポイントがフェイクになったつもりで
    # このエンドポイントを叩くと、本物の上流 LLM を呼び、本番相当の support DB へ
    # 書き込む。このエンドポイントを差し替えたいときは
    # monkeypatch.setattr(agent, "run_agent_turn", ...) を使うこと。
    try:
        result = await agent.run_agent_turn(req.user_id, req.message, req.conversation_id)
    except agent.ConversationNotFound:
        raise HTTPException(status_code=404, detail="会話が見つかりません")
    except SQLAlchemyError:
        logger.exception("データベースエラー user_id=%s", req.user_id)
        raise HTTPException(
            status_code=503,
            detail="データベースを一時的に利用できません。しばらくしてからもう一度お試しください",
        )
    except Exception:
        logger.exception("agent オーケストレーション失敗 user_id=%s", req.user_id)
        raise HTTPException(
            status_code=502,
            detail="上流モデルを一時的に利用できません。しばらくしてからもう一度お試しください",
        )

    # `.get(...) or ""` は意図的な防御。tc["id"] への「簡略化」をしないこと:
    # langchain_core の ToolCall は id: str | None を許容しており、id を省略する
    # OpenAI 互換ゲートウェイ経由では id=None が AgentResult.tool_calls まで届きうる。
    # ToolCallView.id は必須の str なので、素朴な添字アクセスは ValidationError になり、
    # DB 行もチケットも回答も正しく作られた「成功したターン」が日本語メッセージのない
    # 素の 500 として捨てられる。
    #
    # 応答の組み立ては上の try とは別の except で包む。run_agent_turn の失敗(上流障害・
    # DB 障害)と、組み立て段階の ValidationError(= こちら側の欠陥)は別物であり、
    # 後者を except Exception に吸わせて 502「上流モデルを一時的に利用できません」に
    # するのは嘘になる(上流は正常で、決定的に同じ結果になる再試行を促してしまう)。
    # app/api/extract.py が解析失敗を 502 ではなく 500 とした判断と同じ。
    # ただし「500 にする」ことと「日本語の説明を返す」ことは両立するので、素の
    # Internal Server Error に落とさず、detail 付きの 500 として返す。
    # 下の .text 化(ブロック形式対応)と id=None 防御により実際には到達しないはずで、
    # 到達しないことこそが狙い。最後の砦として残す。
    try:
        return AgentResponse(
            conversation_id=result.conversation_id,
            answer=result.answer,
            tool_calls=[
                ToolCallView(
                    id=tc.get("id") or "", name=tc.get("name") or "", args=tc.get("args") or {}
                )
                for tc in result.tool_calls
            ],
            tool_results=[
                ToolResultView(
                    tool_call_id=r.tool_call_id,
                    name=r.name,
                    ok=r.ok,
                    # .content ではなく .text を使う(app/api/chat.py と
                    # app/core/agent.py の3か所で既に標準化されている作法)。
                    # ToolMessage.content は str だけでなくブロック形式(list[dict])も
                    # ありうる。.content のままだと ToolResultView.content: str に
                    # 対して ValidationError になり、DB 行もチケットも回答も
                    # 正しく永続化された「成功したターン」が 500 で捨てられる。
                    # .text は TextAccessor(str のサブクラス)を返すので pydantic の
                    # str フィールドにそのまま入り、type="text" のブロックだけを
                    # 連結して reasoning などの非公開ブロックを落とす。
                    content=r.tool_message.text,
                )
                for r in result.tool_runs
            ],
        )
    except ValidationError:
        logger.exception("応答組み立てに失敗 user_id=%s", req.user_id)
        raise HTTPException(status_code=500, detail="応答の生成に失敗しました")
