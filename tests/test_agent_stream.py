"""app.core.agent のストリーミング出口(stream_agent_turn)のテスト。

DB を触るため、モジュール先頭に loop_scope="session" の asyncio マーカーが必要
(tests/conftest.py の説明、および pyproject.toml のコメント参照)。
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.core import agent
from app.db import repository as repo
from tests.test_agent_orchestration import FakeModel

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_stream_with_tools_emits_tool_then_deltas_then_done(db_session_factory, db_clean):
    first = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"tracking_no": "JP213502378238"}, "id": "c1"}],
    )
    model = FakeModel([first], stream_tokens=["注文", "1001", "は輸送中です。"])
    events = [ev async for ev in agent.stream_agent_turn("u1", "注文1001は今どこですか", None, model=model)]

    assert events[0] == {"type": "tool", "name": "query_logistics"}
    deltas = [e for e in events if e["type"] == "delta"]
    assert [d["text"] for d in deltas] == ["注文", "1001", "は輸送中です。"]
    assert len(deltas) > 1
    assert events[-1]["type"] == "done"
    conversation_id = events[-1]["conversation_id"]
    assert isinstance(conversation_id, int)

    # イベント順序: tool -> delta* -> done
    types = [e["type"] for e in events]
    assert types == ["tool"] + ["delta"] * len(deltas) + ["done"]

    assert model.bind_calls == 1  # 収束(astream)側は bind しない

    # レビュー指摘: astream 収束呼び出しにもツール結果が実際に渡っていることを確認する。
    # ここを確認しないと、[*messages, ai, *(r.tool_message for r in runs)] からツール結果が
    # 落ちても、FakeModel は stream_tokens を無条件に yield するだけなので気づけない。
    convergence_messages = model.astream_messages[0]
    tool_msgs = [m for m in convergence_messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"

    msgs = await repo.list_messages(conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[2].tool_call_id == "c1"
    assert msgs[3].content == "注文1001は輸送中です。"


async def test_stream_without_tools_emits_single_delta_then_done(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="こんにちは。どのようなご用件でしょうか?")])
    events = [ev async for ev in agent.stream_agent_turn("u1", "こんにちは", None, model=model)]

    assert [e["type"] for e in events] == ["delta", "done"]
    assert not any(e["type"] == "tool" for e in events)
    assert events[0]["text"] == "こんにちは。どのようなご用件でしょうか?"
    conversation_id = events[-1]["conversation_id"]

    assert model.astream_messages == []  # ツールなし分岐は astream を一切呼ばない

    msgs = await repo.list_messages(conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant"]


async def test_stream_without_tools_delta_excludes_reasoning_blocks(db_session_factory, db_clean):
    """ツールなし分岐の delta も .text 経由(app/core/agent.py の ai.text)であり、
    content が block list のとき reasoning など text 以外のブロックを混入させない。
    chapter 1 の Critical バグ(自前の isinstance 連結に戻す)への回帰テスト。"""
    content = [
        {"type": "text", "text": "ご案内します"},
        {"type": "reasoning", "text": "内部思考"},
    ]
    model = FakeModel([AIMessage(content=content)])
    events = [ev async for ev in agent.stream_agent_turn("u1", "こんにちは", None, model=model)]
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["ご案内します"]


async def test_stream_with_tools_delta_excludes_reasoning_blocks(db_session_factory, db_clean):
    """収束(astream)側の chunk.text も同じ回帰テスト対象。FakeModel.astream は
    stream_tokens の各要素をそのまま AIMessageChunk(content=...) にするので、
    要素として block list を渡せば chunk.content が block 形式のケースを再現できる。"""
    first = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"tracking_no": "JP213502378238"}, "id": "c1"}],
    )
    block_chunk = [
        {"type": "text", "text": "配送中です"},
        {"type": "reasoning", "text": "非公開の内部推論"},
    ]
    model = FakeModel([first], stream_tokens=[block_chunk])
    events = [
        ev async for ev in agent.stream_agent_turn("u1", "注文1001は今どこですか", None, model=model)
    ]
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["配送中です"]


async def test_stream_new_conversation_returns_fresh_id(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="はい、承知しました。")])
    events = [ev async for ev in agent.stream_agent_turn("u1", "こんにちは", None, model=model)]
    conversation_id = events[-1]["conversation_id"]
    assert isinstance(conversation_id, int)
    conv = await repo.get_conversation(conversation_id)
    assert conv is not None


async def test_stream_unknown_conversation_raises(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    with pytest.raises(agent.ConversationNotFound):
        async for _ in agent.stream_agent_turn("u1", "hi", 999999, model=model):
            pass
