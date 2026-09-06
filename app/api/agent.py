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
    # なお、この return は意図的に try の外に置いている。中に入れると想定外の形状に
    # よる ValidationError が except Exception に吸われて 502「上流モデルを一時的に
    # 利用できません」になるが、それは嘘であり(上流ではなくこちらの応答組み立ての欠陥)、
    # 決定的に同じ結果になる再試行をクライアントに促してしまう。app/api/extract.py の
    # 解析失敗を 502 ではなく 500 とした判断と同じ理由で、サーバー側の欠陥は
    # サーバーエラーとして出す。
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
                content=r.tool_message.content,
            )
            for r in result.tool_runs
        ],
    )
