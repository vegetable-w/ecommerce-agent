"""エージェントの非ストリーミング出口。

- POST /api/agent : 評価・テスト用の入口(非ストリーミング JSON)

05 章でオーケストレーションは LangGraph へ移った(spec §7 / D2)。graph の最終 State を
JSON へ写し、例外をエラー表現へ対応付けるだけを担当する。

**旧 /api/agent/stream は削除した。** entry は /api/chat(astream)と /api/agent
(ainvoke)の 2 つだけという spec の決定に従う。ストリーミングは /api/chat が同じ形の
イベント(tool / citations / delta / actions / done)を出すので、フロントエンド
(static/index.html)はそちらを叩く。
"""

import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.graph.nodes import resolve_answer
from app.schemas.agent import AgentRequest, AgentResponse, ToolCallView, ToolResultView

logger = logging.getLogger(__name__)
router = APIRouter()


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
