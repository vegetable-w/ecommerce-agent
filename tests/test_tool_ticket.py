"""app.tools.business.create_ticket のテスト。

DB を触る test_create_ticket_injects_conversation_id_and_writes だけが
tests/conftest.py の注記どおり pytest.mark.asyncio(loop_scope="session") を必要とする。
test_conversation_id_hidden_from_model_schema は DB を触らない同期関数なので、
モジュール全体に pytestmark を付けるとこの関数にまで asyncio マークが付いてしまい
(同期関数への asyncio マークは無意味) PytestWarning が出る。そのため、
モジュールレベルの pytestmark ではなく該当テストにだけ直接マークを付ける。
"""

import pytest

from app.db.models import Conversation, Ticket
from app.tools.business import create_ticket


def test_conversation_id_hidden_from_model_schema():
    # get_input_schema() ではなく tool_call_schema を見る: get_input_schema() は
    # args_schema をそのまま返すだけで、InjectedToolArg 付きの引数も
    # (include_injected=True で構築されるため) 含んだままになる。
    # LangChain がモデルへ公開するスキーマ (bind_tools / convert_to_openai_tool が
    # 実際に使うもの) は tool_call_schema の方で、これが InjectedToolArg を除外する。
    schema = create_ticket.tool_call_schema.model_json_schema()
    props = schema.get("properties", {})
    assert "description" in props and "ticket_type" in props
    assert "conversation_id" not in props        # 重要:モデルから会話主キーを隠す


@pytest.mark.asyncio(loop_scope="session")
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
