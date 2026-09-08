"""engine を**本物の built-in ツール**で通すテスト(旧 tests/test_tools_infra.py から移設)。

tests/test_tool_engine.py はフェイクのツールだけで engine の分岐を網羅するが、
それだけでは「registry が組み立てた本物の ToolSpec を engine がそのまま実行できるか」
が誰も見ていない状態になる。infra.py を畳むにあたって、その 2 つの守りをここへ移した:

1. 壊れた tool_call(id / name の欠落、id が None)でも例外を外へ漏らさない。
2. 本物の query_order / create_ticket を engine 経由で通せる(create_ticket は
   conversation_id の注入が実際の DB まで届く)。

監査の書き込みは既定でフェイクへ差し替える(autouse)。engine は必ず監査を書きに行くので、
差し替えを忘れると単体テストが本番相当の DB へ 1 行書いてしまう。
DB を触るのは create_ticket のテストだけで、そこだけ tests/conftest.py の注記どおり
session スコープのループへ載せる。
"""

import json

import pytest

from app.db.models import Conversation, Ticket
from app.tools import engine, registry


@pytest.fixture(autouse=True)
def audits(monkeypatch):
    """insert_tool_audit を記録するだけのフェイクへ差し替える(本番 DB を守る安全装置)。"""
    rows = []

    async def fake_audit(**kw):
        rows.append(kw)

    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)
    return rows


def _builtin_specs() -> dict:
    return {s.name: s for s in registry.builtin_specs()}


# --- 壊れた tool_call を渡しても例外を漏らさない -------------------------------
#
# execute_tool_call は「必ず ToolRun を返す」契約を持つ。素朴な dict の添字アクセスで
# id / name を読んでいた頃は、欠けた tool_call(モデルや上流由来で実際に欠けうる)で
# KeyError がこの契約を破って外へ漏れていた。id が None の形も別の穴で、
# `.get("id", "unknown")` のような既定値付き取得では防げない(キーは在って値が None なので
# 既定値が使われず、後段の ToolMessage(tool_call_id=None) が ValidationError で落ちる)。


async def test_execute_tool_call_missing_id_returns_error_run_instead_of_raising():
    run = await engine.execute_tool_call({"name": "query_order"}, 1, _builtin_specs())
    assert run.ok is False                       # order_id を欠くので検証で止まる
    assert run.status == engine.STATUS_VALIDATION_BLOCKED
    assert run.tool_message.tool_call_id == ""   # 元の呼び出しに id が無い以上、作らない


async def test_execute_tool_call_missing_name_returns_error_run_instead_of_raising():
    run = await engine.execute_tool_call({"id": "c1"}, 1, _builtin_specs())
    assert run.ok is False and run.tool_call_id == "c1"
    assert "未知のツール" in run.tool_message.content
    assert run.tool_message.name == "unknown"    # 名前が無くても ToolMessage は組み立てる


async def test_execute_tool_call_none_id_returns_error_run_instead_of_raising():
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1001"}, "id": None}, 1, _builtin_specs()
    )
    assert run.tool_call_id == ""                # None のまま ToolMessage へ渡さない
    assert run.tool_message.tool_call_id == ""


# --- 本物の built-in ツールを engine 経由で通す --------------------------------


async def test_execute_tool_call_real_query_order_success(audits):
    """フェイクではない本物の query_order を registry の ToolSpec 経由で通す。

    11 章は AIMessage.tool_calls の id で ToolMessage を突き合わせるので、
    ToolRun.tool_call_id と ToolMessage.tool_call_id / .name が呼び出し側の入力と
    一致し続けることを、本物のツールで保証する。
    """
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1001"}, "id": "call-abc"},
        1,
        _builtin_specs(),
    )
    assert run.ok is True and run.status == engine.STATUS_SUCCESS
    assert run.tool_call_id == "call-abc"
    assert run.tool_message.tool_call_id == "call-abc"
    assert run.tool_message.name == "query_order"

    payload = json.loads(run.tool_message.content)
    assert isinstance(payload, dict)
    assert payload.keys() >= {"order_id", "status", "amount", "created_at", "product"}
    assert audits[-1]["status"] == "success" and audits[-1]["tool_source"] == "builtin"


async def test_real_query_order_with_bad_args_is_blocked_before_the_tool_runs(audits):
    """必須の引数を欠くと、ツールへ届く前に JSON Schema の検証で止まること。

    旧 infra は pydantic の ValidationError を捕まえて「再試行しない」を担保していた。
    engine では検証が実行の前に来るので、そもそも 1 回も呼ばれない。
    """
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {}, "id": "c1"}, 1, _builtin_specs()
    )
    assert run.ok is False and run.status == engine.STATUS_VALIDATION_BLOCKED
    assert run.retry_count == 0
    assert audits[-1]["status"] == "validation_blocked"


async def test_write_tool_from_the_registry_needs_confirmation(audits):
    """本物の create_ticket も、確認の印なしでは実行されないこと。"""
    run = await engine.execute_tool_call(
        {"name": "create_ticket",
         "args": {"description": "商品が破損している", "ticket_type": "after_sales"},
         "id": "c1"},
        1,
        _builtin_specs(),
    )
    assert run.ok is False and run.status == engine.STATUS_PERMISSION_DENIED


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_tool_call_real_create_ticket_injects_conversation_id(
    db_session_factory, db_clean, audits
):
    """本物の create_ticket を通し、注入した conversation_id が DB まで届くこと。

    conversation_id は args に含めずに呼ぶ(モデルにはこのフィールドが見えない)。
    engine が ToolSpec.inject_conversation を見て検証の後に注入した値が、実際に
    tickets へ書かれることを確かめる。InjectedToolArg を本物の LangChain の
    実行機構経由で満たす経路で、ここが今回いちばん壊れやすい。
    """
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        cid = conv.id

    run = await engine.execute_tool_call(
        {"name": "create_ticket",
         "args": {"description": "商品が破損しているので返品したい", "ticket_type": "after_sales"},
         "id": "c1"},
        cid,
        _builtin_specs(),
        confirmed=True,          # 書き込みは確認の印がないと権限ゲートで止まる
    )

    assert run.ok is True
    payload = json.loads(run.tool_message.content)
    assert "ticket_no" in payload and "status_label" in payload

    async with db_session_factory() as s:
        t = await s.get(Ticket, payload["ticket_no"])
        assert t is not None
        assert t.conversation_id == cid

    assert audits[-1]["conversation_id"] == cid and audits[-1]["status"] == "success"
