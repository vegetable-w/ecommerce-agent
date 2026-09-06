"""エージェントのオーケストレーション。共有コア(_prepare_turn) + 2つの出口。

契約(ハード制約): turn1 だけが bind_tools を使う。収束処理(非ストリーミング/ストリーミングとも)は
未 bind の model を使うため、モデルは物理的に1ターンでツールを呼んだ後、二度目のツール呼び出しは
できない。
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import settings
from app.core.llm import get_chat_model
from app.core.memory import trim_history
from app.core.prompts import AGENT_PROMPT
from app.db import repository
from app.db.models import Message
from app.tools.infra import ToolRun, execute_tool_call
from app.tools.registry import get_all_tools


class ConversationNotFound(Exception):
    pass


@dataclass
class AgentResult:
    conversation_id: int
    answer: str
    tool_calls: list[dict]
    tool_runs: list[ToolRun]


def _text(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    # content が block list の場合はテキスト部分を連結
    return "".join(p.get("text", "") for p in msg.content if isinstance(p, dict))


def _build_history(rows: list[Message]) -> list[BaseMessage]:
    """ターン間コンテキスト: user と「content が空ではなく tool_calls を持たない assistant」だけを
    再利用する。tool_calls を持つ assistant(preamble を含む場合がある)は次ターンへ渡さず、
    後続会話を汚染しない。

    システムプロンプトは trim 対象の history リストにはあえて含めず、trim 後に AGENT_PROMPT で
    付け直す。理由: trim_history は trim_messages を strategy="last", start_on="human" で
    呼んでおり、include_system を渡していない(デフォルト False)。そのため SystemMessage を
    先頭に入れた history をそのまま渡すと、trim 時に無音で捨てられる(実測で確認済み:
    max_tokens=10000 のような十分な予算でも消える。トークン圧迫ではなく include_system=False
    が原因)。system は history と別枠で扱い、AGENT_PROMPT.format_messages が
    trim 後の history の前に必ず付け直す。history リストの中に system を戻す「簡略化」は
    同じ理由で再発するので行わないこと。
    """
    history: list[BaseMessage] = []
    for m in rows:
        if m.role == "user":
            history.append(HumanMessage(m.content or ""))
        elif m.role == "assistant" and m.content and not m.tool_calls:
            history.append(AIMessage(m.content))
    trimmed = trim_history(history, max_tokens=settings.token_budget)
    return AGENT_PROMPT.format_messages(history=trimmed)


async def _prepare_turn(user_id, message, conversation_id, model):
    """共通オーケストレーション前半: 会話ID → user 保存 → 履歴構築 → turn1 bind_tools でツール決定 →
    assistant 保存 → ツールがあれば並列実行して tool 保存。(cid, messages, ai, runs) を返す。
    """
    if conversation_id is None:
        conversation_id = await repository.create_conversation(user_id)
    elif await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)

    # 申し送り事項C(同一セッション内の同時実行): 同じ conversation_id へ複数リクエストが
    # 重なった場合、ここでの「user 保存 → list_messages 読み出し」がリクエスト間でインターリーブ
    # しうる。何かが失われたり壊れたりはしないが、DB への保存順は完了順に従うだけになり、
    # どちらの返信も相手のターンを見ないまま進む。本章は「1セッション1クライアント」を前提と
    # しており、ロック(SELECT ... FOR UPDATE 等)は追加しない既知の制限として扱う。
    await repository.append_message(conversation_id, "user", content=message)
    rows = await repository.list_messages(conversation_id)
    messages = _build_history(rows)

    # 申し送り事項A(ターン途中の部分状態): repository の各関数は自分の session を開いて
    # 即 commit するため、直前の「user 保存」は既に確定している。ここから下の ainvoke が
    # 例外を送出すると、assistant を保存しないままこの関数を抜け、user メッセージだけが
    # 永久に残る「返信のない会話」になる。ターン全体を包む transaction は無い
    # (Redis も LangGraph も本章の範囲外)。データ破損ではなく、孤児 user メッセージは
    # 次ターンで通常の履歴として再生されるだけなので、本章では許容する既知の制限として扱う。
    ai: AIMessage = await model.bind_tools(get_all_tools()).ainvoke(messages)
    await repository.append_message(
        conversation_id,
        "assistant",
        content=_text(ai) or None,
        tool_calls=ai.tool_calls or None,
    )

    runs: list[ToolRun] = []
    if ai.tool_calls:
        runs = await asyncio.gather(
            *(execute_tool_call(tc, conversation_id) for tc in ai.tool_calls)
        )
        for r in runs:
            await repository.append_message(
                conversation_id,
                "tool",
                content=r.tool_message.content,
                tool_call_id=r.tool_call_id,
            )
    return conversation_id, messages, ai, runs


async def run_agent_turn(user_id, message, conversation_id, model=None) -> AgentResult:
    """非ストリーミング出口: ツール履歴 + 最終回答を一括返却(/api/agent、eval、単体テスト用)。"""
    model = model or get_chat_model()
    conversation_id, messages, ai, runs = await _prepare_turn(
        user_id, message, conversation_id, model
    )
    if not ai.tool_calls:
        return AgentResult(conversation_id, _text(ai), [], [])
    final: AIMessage = await model.ainvoke(  # 収束: bind_tools しない(1ターン制約)
        [*messages, ai, *(r.tool_message for r in runs)]
    )
    await repository.append_message(conversation_id, "assistant", content=_text(final))
    return AgentResult(conversation_id, _text(final), ai.tool_calls, runs)


async def stream_agent_turn(
    user_id, message, conversation_id, model=None
) -> AsyncIterator[dict]:
    """ストリーミング出口: /api/agent/stream が SSE に変換するイベント dict を生成。
    収束は astream、bind しない。

    既知のトレードオフ: ツールを使わない場合は完成済み回答を一括送信し、タイプライター表示には
    ならない。turn1 ではツールを使うか判断するため非ストリーミングで完全な結果が必要であり、
    ツールなしと判断した時点で回答もすでに完成しているため。ツールを使う場合の最終回答は
    引き続き token 単位でストリーミングする。
    """
    model = model or get_chat_model(streaming=True)
    conversation_id, messages, ai, runs = await _prepare_turn(
        user_id, message, conversation_id, model
    )

    if not ai.tool_calls:
        # ツールなし: turn1 のテキストがそのまま最終回答(assistant は保存済み)。一括出力する
        yield {"type": "delta", "text": _text(ai)}
        yield {"type": "done", "conversation_id": conversation_id}
        return

    for tc in ai.tool_calls:
        yield {"type": "tool", "name": tc.get("name") or ""}

    chunks: list[str] = []
    async for chunk in model.astream(  # 収束: bind_tools しない(1ターン制約)
        [*messages, ai, *(r.tool_message for r in runs)]
    ):
        text = chunk.content if isinstance(chunk.content, str) else _text(chunk)
        if not text:
            continue
        chunks.append(text)
        yield {"type": "delta", "text": text}

    await repository.append_message(conversation_id, "assistant", content="".join(chunks))
    yield {"type": "done", "conversation_id": conversation_id}
