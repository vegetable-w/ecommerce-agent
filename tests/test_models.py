import re

import pytest
from sqlalchemy import text

from app.db.models import Conversation, Faq, Message, Ticket

# _test_engine はセッションスコープのイベントループ上で作成される(tests/conftest.py 参照)。
# このモジュールのテストも同じループで動かさないと asyncmy のコネクションが別ループに
# 紐付いたままになり "attached to a different loop" の RuntimeError になる。
pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_conversation_defaults_and_autoincrement(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        assert conv.id is not None  # BIGINT AUTO_INCREMENT
        assert conv.status == "in_progress"  # DB DEFAULT を読み戻す
        assert conv.created_at is not None


async def test_message_json_tool_calls_roundtrip(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.flush()
        msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content=None,
            tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}],
        )
        s.add(msg)
        await s.commit()
        got = await s.get(Message, msg.id)
        assert got.tool_calls[0]["name"] == "query_order"  # JSON round-trip
        assert got.role == "assistant"


# --- 追加検証: ORM のメタデータが実際の support_test スキーマと一致していること ---
#
# 上記2テストは conversations/messages の一部カラムしか触れないため、faq や tickets の
# カラム名のタイプミス、あるいは ENUM 値の不一致(日本語が紛れ込む等)を検出できない。
# ここでは information_schema を直接参照し、ORM 側にハードコードした「期待値の第二の
# コピー」を作らずに、DB が報告する実際のスキーマと ORM 定義を突き合わせる。

_ORM_MODELS = {
    "conversations": Conversation,
    "messages": Message,
    "faq": Faq,
    "tickets": Ticket,
}

# (table, column) -> ORM column の宣言から拾う ENUM チェック対象
_ENUM_COLUMNS = [
    ("conversations", "status"),
    ("messages", "role"),
    ("tickets", "ticket_type"),
    ("tickets", "status"),
]


async def test_orm_columns_match_live_schema_for_all_tables(_test_engine, db_clean):
    async with _test_engine.connect() as conn:
        for table_name, model in _ORM_MODELS.items():
            rows = (
                await conn.execute(
                    text(
                        "SELECT COLUMN_NAME FROM information_schema.columns "
                        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"
                    ),
                    {"t": table_name},
                )
            ).scalars().all()
            db_columns = set(rows)
            orm_columns = {c.name for c in model.__table__.columns}
            assert orm_columns == db_columns, (
                f"{table_name}: ORM columns {orm_columns} != DB columns {db_columns}"
            )


async def test_orm_enum_values_match_live_schema(_test_engine, db_clean):
    async with _test_engine.connect() as conn:
        for table_name, column_name in _ENUM_COLUMNS:
            column_type = (
                await conn.execute(
                    text(
                        "SELECT COLUMN_TYPE FROM information_schema.columns "
                        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t "
                        "AND COLUMN_NAME = :c"
                    ),
                    {"t": table_name, "c": column_name},
                )
            ).scalar_one()
            # COLUMN_TYPE は "enum('a','b','c')" の形式で返る
            db_values = tuple(re.findall(r"'([^']*)'", column_type))

            orm_column = _ORM_MODELS[table_name].__table__.columns[column_name]
            orm_values = tuple(orm_column.type.enums)

            assert orm_values == db_values, (
                f"{table_name}.{column_name}: ORM enum {orm_values} != DB enum {db_values}"
            )
