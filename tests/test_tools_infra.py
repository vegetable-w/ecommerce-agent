"""app.tools.infra.execute_tool_call のテスト。

DB を触る test_execute_tool_call_real_create_ticket_injects_conversation_id だけが
tests/conftest.py の注記どおり pytest.mark.asyncio(loop_scope="session") を必要とする。
他のテストは registry.get_tool を monkeypatch したフェイクツールのみを使い DB に触れない
同期不要の async 関数なので、モジュール全体に pytestmark を付けると
(tests/test_tool_ticket.py と同じ理由で) 無関係なテストにまで影響が及ぶのを避けるため、
モジュールレベルの pytestmark ではなく該当テストにだけ直接マークを付ける。
"""

import asyncio
import json

import pytest

from app.db.models import Conversation, Ticket
from app.tools import infra, registry


def test_registry_has_five_tools():
    names = {t.name for t in registry.get_all_tools()}
    assert names == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}


async def test_execute_unknown_tool_returns_error_run():
    run = await infra.execute_tool_call({"name": "nope", "args": {}, "id": "c1"}, conversation_id=1)
    assert run.ok is False and run.tool_call_id == "c1"
    assert "不明なツール" in run.tool_message.content


async def test_timeout_then_error(monkeypatch):
    async def slow(_args):
        await asyncio.sleep(1)

    fake = type("T", (), {"name": "query_order", "ainvoke": staticmethod(slow)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake)
    run = await infra.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        conversation_id=1,
        timeout=0.05,
        max_retries=1,
    )
    assert run.ok is False and "失敗" in run.tool_message.content


async def test_retry_succeeds_on_second_attempt(monkeypatch):
    calls = {"n": 0}

    async def flaky(_args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("一時的なエラー")
        return {"ok": True}

    fake = type("T", (), {"name": "query_faq", "ainvoke": staticmethod(flaky)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake)
    run = await infra.execute_tool_call(
        {"name": "query_faq", "args": {"keyword": "x"}, "id": "c1"},
        conversation_id=1,
        timeout=1.0,
        max_retries=2,
    )
    assert run.ok is True and calls["n"] == 2


async def test_create_ticket_not_retried(monkeypatch):
    calls = {"n": 0}

    async def always_fail(_args):
        calls["n"] += 1
        raise RuntimeError("boom")

    fake = type("T", (), {"name": "create_ticket", "ainvoke": staticmethod(always_fail)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake)
    run = await infra.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x", "ticket_type": "after_sales"}, "id": "c1"},
        conversation_id=1,
        timeout=1.0,
        max_retries=2,
    )
    assert run.ok is False and calls["n"] == 1  # 書き込み系ツールは retry しない


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_tool_call_real_create_ticket_injects_conversation_id(db_session_factory, db_clean):
    """monkeypatch なしで本物の create_ticket を通す。

    conversation_id を args に含めず(モデルはこのフィールドを見えないので)呼び出し、
    execute_tool_call が registry.INJECT_CONVERSATION 経由で注入した値が実際に
    DB へ書き込まれることを確認する。ここが今回のタスクで最も壊れやすい経路
    (InjectedToolArg を実際の LangChain 実行機構経由で満たす)。
    """
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        cid = conv.id

    run = await infra.execute_tool_call(
        {
            "name": "create_ticket",
            "args": {"description": "商品が破損しているので返品したい", "ticket_type": "after_sales"},
            "id": "c1",
        },
        conversation_id=cid,
    )

    assert run.ok is True
    payload = json.loads(run.tool_message.content)
    assert "ticket_no" in payload and "status_label" in payload

    async with db_session_factory() as s:
        t = await s.get(Ticket, payload["ticket_no"])
        assert t is not None
        assert t.conversation_id == cid
