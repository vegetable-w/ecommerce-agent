"""08 章 tool audit の書き込み経路のテスト。

status は DB の英語識別子(success / failed / timeout / validation_blocked /
permission_denied)で書く。日本語の表示ラベルは app/core/labels.py の
TOOL_AUDIT_STATUS だけが持つ(このリポジトリの規約: DB / ORM / ツール引数は
英語識別子で統一する)。
"""

import pytest
from sqlalchemy import select

from app.db import repository
from app.db.models import ToolAuditLog

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_insert_tool_audit_minimal(db_session_factory):
    await repository.insert_tool_audit(
        conversation_id=None, tool_call_id=None, tool_name="query_order",
        tool_source="builtin", mcp_server=None, arguments={"order_id": "1001"},
        result_summary="{}", status="success", error_message=None,
        retry_count=0, duration_ms=12,
    )
    async with db_session_factory() as s:
        row = (await s.execute(select(ToolAuditLog))).scalars().one()
    assert row.tool_name == "query_order" and row.status == "success"
    assert row.conversation_id is None          # conversation context がなくても null 可
    assert row.arguments == {"order_id": "1001"}


async def test_insert_tool_audit_all_statuses(db_session_factory):
    statuses = ["success", "failed", "timeout", "validation_blocked", "permission_denied"]
    for st in statuses:
        await repository.insert_tool_audit(
            conversation_id=1, tool_call_id="tc-1", tool_name="create_ticket",
            tool_source="mcp", mcp_server="logistics", arguments=None,
            result_summary=None, status=st, error_message="理由",
            retry_count=2, duration_ms=None,
        )
    async with db_session_factory() as s:
        rows = (await s.execute(select(ToolAuditLog))).scalars().all()
    assert {r.status for r in rows} == set(statuses)


async def test_insert_tool_audit_keeps_conversation_id_without_fk(db_session_factory):
    """存在しない conversation_id でも audit は残る。

    tool_audit_logs は conversations への FK を持たない(sql/08-ddl.sql)。
    audit は「何が起きたか」の記録であり、参照整合性制約でその書き込み自体を
    落としてしまうと、いちばん知りたい異常時の記録が消える。
    """
    await repository.insert_tool_audit(
        conversation_id=999999, tool_call_id=None, tool_name="query_order",
        tool_source="builtin", mcp_server=None, arguments=None,
        result_summary=None, status="failed", error_message="upstream error",
        retry_count=1, duration_ms=3,
    )
    async with db_session_factory() as s:
        row = (await s.execute(select(ToolAuditLog))).scalars().one()
    assert row.conversation_id == 999999

