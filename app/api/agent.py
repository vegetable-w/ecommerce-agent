"""エージェントの HTTP 出口。

- POST /api/agent/stream : フロントエンドの主入口(SSE)
- POST /api/agent        : プログラム/テスト用の入口(非ストリーミング JSON)

/api/agent は 05 章で LangGraph の ainvoke 入口になった(spec §7 / D2)。graph の最終
State を JSON へ写し、例外をエラー表現へ対応付けるだけを担当する。
/api/agent/stream はまだ旧オーケストレーション(app/core/agent.py)を使う。
"""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.core import agent
from app.core.llm import get_chat_model
from app.graph import runtime
from app.graph.nodes import resolve_answer
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
            # なる。/api/agent 側では同じ状況を 500 と 502 に区別しているので、その原則とは
            # 食い違っている。それでもこうしているのは、ここが既に HTTP 200 とヘッダを
            # 送出した後であり、ステータスコードで区別する手段が物理的に残っていないため。
            logger.exception("エージェントオーケストレーション失敗 user_id=%s", req.user_id)
            for f in _sse_error("上流モデルを一時的に利用できません。しばらくしてからもう一度お試しください"):
                yield f
            return
        # 意図的な差異: エラー終了では [DONE] を送らずに return する。static/index.html の
        # 解析ループは reader.read() の done(= HTTP ストリームが閉じたこと)で終了し、
        # [DONE] は continue で読み飛ばすだけなので、省いてもフロントエンドの挙動は変わらない。
        # 正常終了とエラー終了を「[DONE] が来たかどうか」で区別できる分、下流には扱いやすい。
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # リバースプロキシによるバッファリング対策
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _views_from_state(state) -> tuple[list[ToolCallView], list[ToolResultView]]:
    """最終 State の messages から tool の呼び出しと結果を組み直す。

    評価と単体テストで「モデルが何を選んだか」を見るために要る。ストリーミング側は
    tool 名だけを流せば足りるが、こちらは引数と結果まで欲しいので messages を辿る。

    `.get("id") or ""` は意図的な防御(tc["id"] への「簡略化」をしないこと):
    langchain_core の ToolCall は id: str | None を許容しており、id を省略する
    OpenAI 互換ゲートウェイ経由では id=None がここまで届きうる。ToolCallView.id は
    必須の str なので、素朴な添字アクセスは ValidationError になり、DB 行も回答も
    正しく作られた「成功したターン」が日本語メッセージのない素の 500 として捨てられる。

    content は .content ではなく .text を使う(app/api/chat.py と app/graph/nodes.py で
    既に標準化されている作法)。ToolMessage.content は str だけでなくブロック形式
    (list[dict])もありうるため、.content のままだと ToolResultView.content: str に
    対して ValidationError になる。.text は type="text" のブロックだけを連結し、
    reasoning などの非公開ブロックを落とす。
    """
    calls, results = [], []
    for m in state.get("messages", []):
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                calls.append(
                    ToolCallView(
                        id=tc.get("id") or "", name=tc.get("name") or "", args=tc.get("args") or {}
                    )
                )
        elif isinstance(m, ToolMessage):
            results.append(
                ToolResultView(
                    tool_call_id=m.tool_call_id,
                    name=m.name or "",
                    # ToolNode も app/tools/infra.py も、失敗した tool は status="error" の
                    # ToolMessage にする。ここを常に True にすると、評価は失敗した呼び出しを
                    # 成功として数える。
                    ok=(m.status != "error"),
                    content=m.text,
                )
            )
    return calls, results


@router.post("/api/agent", response_model=AgentResponse)
async def run_agent(req: AgentRequest) -> AgentResponse:
    # runtime.run_turn は属性経由で呼ぶこと(from ... import run_turn にしない)。
    # テストの monkeypatch.setattr(runtime, "run_turn", ...) が効かなくなるため。
    #
    # 注意(テストを書く人向けの落とし穴): このハンドラはモデルを Depends で受け取らない。
    # graph の中の node が get_chat_model() を組み立てるので、差し替えたいときは
    # runtime.run_turn か runtime.get_graph を monkeypatch すること。何も差し替えずに
    # 叩くと本物の上流 LLM を呼び、本番相当の support DB へ書き込む。
    try:
        out = await runtime.run_turn(req.user_id, req.message, req.conversation_id)
    except runtime.ConversationNotFound:
        raise HTTPException(status_code=404, detail="会話が見つかりません")
    except SQLAlchemyError:
        logger.exception("データベースエラー user_id=%s", req.user_id)
        raise HTTPException(
            status_code=503,
            detail="データベースを一時的に利用できません。しばらくしてからもう一度お試しください",
        )
    except Exception:
        logger.exception("graph オーケストレーション失敗 user_id=%s", req.user_id)
        raise HTTPException(
            status_code=502,
            detail="上流モデルを一時的に利用できません。しばらくしてからもう一度お試しください",
        )

    # 応答の組み立ては上の try とは別の except で包む。run_turn の失敗(上流障害・DB 障害)と、
    # 組み立て段階の ValidationError(= こちら側の欠陥)は別物であり、後者を except Exception に
    # 吸わせて 502「上流モデルを一時的に利用できません」にするのは嘘になる(上流は正常で、
    # 決定的に同じ結果になる再試行を促してしまう)。app/api/extract.py が解析失敗を 502 ではなく
    # 500 とした判断と同じ。ただし「500 にする」ことと「日本語の説明を返す」ことは両立するので、
    # 素の Internal Server Error に落とさず detail 付きの 500 として返す。
    # 上の防御により実際には到達しないはずで、到達しないことこそが狙い。最後の砦として残す。
    try:
        state = out["state"]
        calls, results = _views_from_state(state)
        return AgentResponse(
            conversation_id=out["conversation_id"],
            answer=resolve_answer(state),
            tool_calls=calls,
            tool_results=results,
            # `or []` は None 対策。suggested_actions に reducer は無く、node が None を
            # 書き戻すと key はあるが値が None の State になりうる。
            suggested_actions=state.get("suggested_actions") or [],
        )
    except ValidationError:
        logger.exception("応答組み立てに失敗 user_id=%s", req.user_id)
        raise HTTPException(status_code=500, detail="応答の生成に失敗しました")
