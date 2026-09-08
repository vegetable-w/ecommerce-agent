"""app.tools.infra.execute_tool_call のテスト。

DB を触る test_execute_tool_call_real_create_ticket_injects_conversation_id だけが
tests/conftest.py の注記どおり pytest.mark.asyncio(loop_scope="session") を必要とする。
他のテスト(フェイクツールのみを使うもの、および DB 不要の read-only な実ツール
query_order を通すもの)は DB に触れない同期不要の async 関数なので、モジュール全体に
pytestmark を付けると(tests/test_tool_ticket.py と同じ理由で)無関係なテストにまで
影響が及ぶのを避けるため、モジュールレベルの pytestmark ではなく該当テストにだけ
直接マークを付ける。
"""

import asyncio
import json

import pytest

from app.db.models import Conversation, Ticket
from app.tools import infra, registry


def test_registry_has_five_tools():
    # 08: query_logistics は built-in から外し、配送状況の照会は MCP 側が担う
    names = {t.name for t in registry.get_all_tools()}
    assert names == {"query_order", "query_product", "query_faq",
                     "create_ticket", "submit_refund"}


async def test_execute_unknown_tool_returns_error_run():
    run = await infra.execute_tool_call({"name": "nope", "args": {}, "id": "c1"}, conversation_id=1)
    assert run.ok is False and run.tool_call_id == "c1"
    assert "不明なツール" in run.tool_message.content


# --- execute_tool_call は「例外を外へ漏らさない」契約(Task 9)を持つ。しかし id/name を
# 素朴な dict 添字アクセスで読んでいたため、壊れた tool_call(モデル/上流由来で欠落しうる)を
# 渡すと KeyError がこの契約を破って外へ漏れていた(実測で確認済み)。この2本はその回帰テスト。


async def test_execute_tool_call_missing_id_returns_error_run_instead_of_raising():
    run = await infra.execute_tool_call({"name": "query_order"}, conversation_id=1)
    assert run.ok is False
    assert run.tool_call_id == "unknown"


async def test_execute_tool_call_missing_name_returns_error_run_instead_of_raising():
    run = await infra.execute_tool_call({"id": "c1"}, conversation_id=1)
    assert run.ok is False
    assert run.tool_call_id == "c1"
    assert "不明なツール" in run.tool_message.content


async def test_execute_tool_call_none_id_returns_error_run_instead_of_raising():
    # id はキーとして存在するが値が None(キー欠落とは別の穴): `.get("id", "unknown")` の
    # ような既定値付き取得では防げず、後段の ToolMessage(tool_call_id=None) が
    # ValidationError で落ちる。`.get("id") or "unknown"` でないと防げない。
    run = await infra.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1001"}, "id": None}, conversation_id=1
    )
    assert run.tool_call_id == "unknown"


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


async def test_execute_tool_call_real_query_order_success():
    """monkeypatch なしで本物の query_order を通す (DB 不要の read-only ツール)。

    Task 11 は AIMessage.tool_calls の id で ToolMessage を突き合わせるので、
    ToolRun.tool_call_id と ToolMessage.tool_call_id / .name が呼び出し側の
    入力と一致し続けることを、フェイクではなく実ツール経由で保証する。
    """
    run = await infra.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1001"}, "id": "call-abc"},
        conversation_id=1,
    )
    assert run.ok is True
    assert run.tool_call_id == "call-abc"
    assert run.tool_message.tool_call_id == "call-abc"
    assert run.tool_message.name == "query_order"

    payload = json.loads(run.tool_message.content)
    assert isinstance(payload, dict)
    assert payload.keys() >= {"order_id", "status", "amount", "created_at", "product"}


async def test_validation_error_is_not_retried(monkeypatch):
    """引数が不正で pydantic.ValidationError になるケースは、同じ引数で再試行しても
    必ず同じ結果になるので retry 予算を消費せず 1 回で確定させる。

    本物の query_order ツールのクラスに attempt 数を数える ainvoke を被せて計測する
    (ok is False だけでは「retry したが毎回失敗した」場合と区別できないため、
    実際の呼び出し回数を数える)。
    """
    tool_cls = type(registry.get_tool("query_order"))
    original_ainvoke = tool_cls.ainvoke
    calls = {"n": 0}

    async def counting_ainvoke(self, *args, **kwargs):
        calls["n"] += 1
        return await original_ainvoke(self, *args, **kwargs)

    monkeypatch.setattr(tool_cls, "ainvoke", counting_ainvoke)

    run = await infra.execute_tool_call(
        {"name": "query_order", "args": {}, "id": "c1"},  # 必須の order_id を欠く -> ValidationError
        conversation_id=1,
        max_retries=2,
    )
    assert run.ok is False
    assert calls["n"] == 1  # retry されていないことを回数で確認


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
