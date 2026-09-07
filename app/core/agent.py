"""エージェントのオーケストレーション。共有コア(_prepare_turn) + 2つの出口。

契約(ハード制約): turn1 だけが bind_tools を使う。収束処理(非ストリーミング/ストリーミングとも)は
未 bind の model を使うため、モデルは物理的に1ターンでツールを呼んだ後、二度目のツール呼び出しは
できない。
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

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
    # 回答が引用した根拠(query_faq が返した順番と番号のまま)。ツールを使わないターンや
    # 根拠不足で断ったターンでは空のまま。既定値を持たせるのは、既存の 4 引数での
    # 組み立てを壊さないため。
    citations: list[dict] = field(default_factory=list)


def _exception_to_tool_run(tool_call: dict, exc: BaseException) -> ToolRun:
    """execute_tool_call は「例外を外へ漏らさない」契約(Task 9)だが、防御的多重化として
    asyncio.gather(return_exceptions=True) が拾った例外もここで同じ ToolRun 形状へ変換する。
    こうしておくと、gather の1タスクが失敗しても他の tool タスクは最後まで await され切り、
    (create_ticket のような書き込み系ツールが)tool 行の記録なしに副作用だけ残す事態を防げる。
    """
    # id=None も "unknown" に落とす(app/tools/infra.py の execute_tool_call と同じ理由:
    # id は存在するが None の場合があり、`.get(..., default)` では防げない)。
    name = tool_call.get("name", "")
    tc_id = tool_call.get("id") or "unknown"
    return ToolRun(
        tool_call_id=tc_id,
        name=name,
        ok=False,
        tool_message=ToolMessage(
            content="ツール実行失敗: 予期しないエラー",
            tool_call_id=tc_id,
            name=name,
            status="error",
        ),
    )


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


def _faq_result(runs) -> dict | None:
    """runs の中の query_faq の結果を dict として取り出す。無ければ None。

    ツールが失敗した場合 content は JSON ではなく日本語のエラー文になる。ここは
    「引用を送るか / 低信頼プールへ積むか」を決めるだけの補助なので、解釈できない
    ものは静かに None にして、ターン本体(回答生成)を巻き込まない。
    """
    for r in runs:
        if getattr(r, "name", None) != "query_faq":
            continue
        try:
            parsed = json.loads(r.tool_message.content)
        except (ValueError, TypeError):
            return None
        # json.loads は数値や文字列も通す。dict でなければ後段の .get で落ちる
        return parsed if isinstance(parsed, dict) else None
    return None


async def _record_refusal(conversation_id: int, message: str, faq: dict) -> None:
    """根拠不足で断ったターンを低信頼プールへ積む。raw_question はユーザーの原文。

    source は DDL の ENUM に合わせる。query_faq が付けなかった場合に self_check へ
    倒すのは、ENUM に無い値を書いて DB エラーにする方が害が大きいため。
    """
    await repository.insert_low_confidence(
        conversation_id, message, faq.get("source") or "self_check", faq.get("reason")
    )


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
        content=ai.text or None,
        tool_calls=ai.tool_calls or None,
    )

    runs: list[ToolRun] = []
    if ai.tool_calls:
        # return_exceptions=True: execute_tool_call 自体は例外を漏らさない契約だが、
        # 防御的多重化としてここでも拾う。指定しないと gather は最初の例外で await を打ち切り、
        # 兄弟タスク(create_ticket など副作用のある書き込み系ツール)が tool 行の記録なしに
        # 実行され続けてしまう(実測で確認済みの問題)。
        results = await asyncio.gather(
            *(execute_tool_call(tc, conversation_id) for tc in ai.tool_calls),
            return_exceptions=True,
        )
        runs = [
            r if isinstance(r, ToolRun) else _exception_to_tool_run(tc, r)
            for tc, r in zip(ai.tool_calls, results)
        ]
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
        return AgentResult(conversation_id, ai.text, [], [])

    faq = _faq_result(runs)
    citations: list[dict] = []
    if faq is not None:
        if faq.get("sufficient"):
            citations = faq.get("citations") or []
        else:
            await _record_refusal(conversation_id, message, faq)

    final: AIMessage = await model.ainvoke(  # 収束: bind_tools しない(1ターン制約)
        [*messages, ai, *(r.tool_message for r in runs)]
    )
    await repository.append_message(conversation_id, "assistant", content=final.text)
    return AgentResult(conversation_id, final.text, ai.tool_calls, runs, citations)


async def stream_agent_turn(
    user_id, message, conversation_id, model=None
) -> AsyncIterator[dict]:
    """ストリーミング出口: SSE に変換するイベント dict を生成。収束は astream、bind しない。

    05 章でストリーミングの入口は graph(app/graph/runtime.py の stream_turn)へ移り、
    /api/agent/stream は削除した。この関数はもう HTTP からは呼ばれていない。

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
        yield {"type": "delta", "text": ai.text}
        yield {"type": "done", "conversation_id": conversation_id}
        return

    for tc in ai.tool_calls:
        yield {"type": "tool", "name": tc.get("name") or ""}

    # 回答本文より前に引用を送る。フロントエンドは [n] を描き始める時点で引用元を
    # 持っていないと、クリックできる注釈にできない。
    faq = _faq_result(runs)
    if faq is not None:
        if faq.get("sufficient"):
            yield {"type": "citations", "items": faq.get("citations") or []}
        else:
            await _record_refusal(conversation_id, message, faq)

    chunks: list[str] = []
    async for chunk in model.astream(  # 収束: bind_tools しない(1ターン制約)
        [*messages, ai, *(r.tool_message for r in runs)]
    ):
        # .contentは文字列/ブロック形式のどちらもあり得るため、両対応の.textプロパティを使う
        # (app/api/chat.py と同じ理由: 独自の連結ロジックを持つと reasoning ブロックの
        # 混入など細部がずれる)
        text = chunk.text
        if not text:
            continue
        chunks.append(text)
        yield {"type": "delta", "text": text}

    await repository.append_message(conversation_id, "assistant", content="".join(chunks))
    yield {"type": "done", "conversation_id": conversation_id}
