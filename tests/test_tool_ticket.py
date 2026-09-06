"""app.tools.business.create_ticket のテスト。

DB を触るため、tests/conftest.py の注記どおりモジュール先頭で
pytest.mark.asyncio(loop_scope="session") を指定する必要がある。
"""

import pytest

from app.db.models import Conversation, Ticket
from app.tools.business import create_ticket

pytestmark = pytest.mark.asyncio(loop_scope="session")


def test_conversation_id_hidden_from_model_schema():
    schema = create_ticket.get_input_schema().model_json_schema()
    props = schema.get("properties", {})
    assert "description" in props and "ticket_type" in props
    assert "conversation_id" not in props        # 重要:モデルから会話主キーを隠す


async def test_create_ticket_injects_conversation_id_and_writes(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        cid = conv.id
    r = await create_ticket.ainvoke(
        {"description": "商品が破損しているので返品したい", "ticket_type": "after_sales", "conversation_id": cid}
    )
    assert r["ticket_no"].startswith("T")
    assert r["status"] == "escalated"
    assert r["status_label"] == "オペレーター対応"
    async with db_session_factory() as s:
        t = await s.get(Ticket, r["ticket_no"])
        assert t is not None and t.conversation_id == cid
        assert t.ticket_type == "after_sales" and t.status == "pending"
        conv = await s.get(Conversation, cid)
        assert conv.status == "escalated"
