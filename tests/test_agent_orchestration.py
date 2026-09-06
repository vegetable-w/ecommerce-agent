"""app.core.agent の非ストリーミング出口(run_agent_turn)と共通コア(_prepare_turn)のテスト。

DB を触るため、モジュール先頭に loop_scope="session" の asyncio マーカーが必要
(tests/conftest.py の説明、および pyproject.toml のコメント参照)。
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from app.core import agent
from app.core.prompts import AGENT_SYSTEM
from app.db import repository as repo

pytestmark = pytest.mark.asyncio(loop_scope="session")


class FakeModel:
    """turn1 は scripted[0] を返す(tool_calls を含んでもよい)。非ストリーミング収束は scripted[1]。
    ストリーミング収束は stream_tokens を順番に AIMessageChunk として yield する。
    bind_calls は bind_tools の呼び出し回数を記録する(収束時 bind なし → 常に 1 の想定)。"""

    def __init__(self, scripted, stream_tokens=None):
        self._scripted = list(scripted)
        self._stream_tokens = list(stream_tokens or [])
        self.bind_calls = 0
        self.invoke_messages = []
        self.astream_messages = []

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def ainvoke(self, messages):
        self.invoke_messages.append(list(messages))
        return self._scripted.pop(0)

    async def astream(self, messages):
        self.astream_messages.append(list(messages))
        for t in self._stream_tokens:
            yield AIMessageChunk(content=t)


async def test_no_tool_calls_returns_direct_answer(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="こんにちは。どのようなご用件でしょうか?")])
    res = await agent.run_agent_turn("u1", "こんにちは", None, model=model)
    assert res.answer.startswith("こんにちは") and res.tool_calls == []
    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant"]


async def test_tool_call_flow_executes_and_converges(db_session_factory, db_clean):
    first = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
    )
    model = FakeModel([first, AIMessage(content="注文 1001 は現在輸送中です。")])
    res = await agent.run_agent_turn("u1", "注文 1001 は今どこですか", None, model=model)
    assert res.answer == "注文 1001 は現在輸送中です。"
    assert res.tool_calls[0]["name"] == "query_logistics"
    assert res.tool_runs[0].ok is True
    assert model.bind_calls == 1  # 収束時は bind しない
    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[2].tool_call_id == "c1"


async def test_continue_conversation_replays_only_final_answers(db_session_factory, db_clean):
    m1 = FakeModel(
        [
            AIMessage(
                content="確認します。",
                tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
            ),
            AIMessage(content="注文 1001 は集荷済みです。"),
        ]
    )
    cid = (await agent.run_agent_turn("u1", "注文1001は今どこですか", None, model=m1)).conversation_id
    m2 = FakeModel([AIMessage(content="ほかにお手伝いできることはありますか?")])
    await agent.run_agent_turn("u1", "ありがとう", cid, model=m2)
    ai_contents = [m.content for m in m2.invoke_messages[0] if isinstance(m, AIMessage)]
    assert "注文 1001 は集荷済みです。" in ai_contents  # 最終回答は再利用
    assert "確認します。" not in ai_contents  # tool-calling preamble は次ターンへ渡さない


async def test_unknown_conversation_raises(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    with pytest.raises(agent.ConversationNotFound):
        await agent.run_agent_turn("u1", "hi", 999999, model=model)


# --- 申し送り事項B: list_messages が存在しない/空の conversation_id に対して [] を返す契約。
# _build_history は新規会話でこれに依存するため、明示的にテストで固定する。


async def test_list_messages_returns_empty_for_nonexistent_conversation(db_session_factory, db_clean):
    assert await repo.list_messages(999999) == []


async def test_list_messages_returns_empty_for_new_conversation(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    assert await repo.list_messages(cid) == []


# --- システムプロンプトの生存確認(調査で発覚した trim_history の include_system=False 問題への回帰テスト)。
# _build_history は SystemMessage を trim 対象の history リストに含めず、AGENT_PROMPT で
# trim 後に付与する。この2本のテストは「モデルへ実際に渡る先頭メッセージが system prompt であること」
# を短い会話・trim が発生するほど長い会話の両方で固定する。


async def test_system_prompt_survives_into_model_messages(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="こんにちは")])
    await agent.run_agent_turn("u1", "こんにちは", None, model=model)
    first_call_messages = model.invoke_messages[0]
    assert isinstance(first_call_messages[0], SystemMessage)
    assert first_call_messages[0].content == AGENT_SYSTEM


async def test_system_prompt_survives_when_history_is_trimmed(monkeypatch):
    # token_budget を極端に小さくして、trim_history が実際に history を間引くことを強制する。
    monkeypatch.setattr(agent.settings, "token_budget", 5)
    rows = [
        SimpleNamespace(role="user", content=f"注文について質問その{i}です" * 5, tool_calls=None)
        for i in range(20)
    ]
    result = agent._build_history(rows)
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == AGENT_SYSTEM
    # 実際に間引かれたこと(system 1件 + 全20件がそのまま残ったのでは検証にならない)を確認する
    assert len(result) < len(rows) + 1
