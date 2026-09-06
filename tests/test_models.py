import re
from datetime import datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

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


async def test_message_with_unknown_conversation_id_is_rejected_by_fk(
    db_session_factory, db_clean
):
    """fk_messages_conversation が実際に効いていることを確認する。

    db_clean は truncate の前後で SET FOREIGN_KEY_CHECKS=0/1 を切り替えている
    (tests/conftest.py 参照)。この ON/OFF がもし何らかの理由で漏れて OFF のまま
    残ると、存在しない conversation_id を持つ孤児行が静かに書き込めてしまい、
    どのテストもそれに気づけない。ここで明示的に「制約が効いている」ことを
    確認しておく。
    """
    async with db_session_factory() as s:
        orphan = Message(
            conversation_id=999_999_999,  # conversations に存在しない ID
            role="user",
            content="orphan message",
        )
        s.add(orphan)
        with pytest.raises(IntegrityError):
            await s.commit()


async def test_conversation_updated_at_bumps_on_update(
    db_session_factory, db_clean, _test_engine
):
    """updated_at の ON UPDATE CURRENT_TIMESTAMP が実際に発火することを確認する。

    MySQL の DATETIME はここでは秒精度しかないため、「更新前後で同じ秒内に commit した
    場合、onupdate が効いていなくても新旧の値がたまたま一致して見える」という誤検知が
    起こり得る。単純に「更新前 <= 更新後」を秒精度のまま比べると、onupdate が壊れて
    いても(=値が更新されず更新前の値のままでも)たまたま偽陽性で通ってしまう。

    sleep(1) で秒境界をまたぐのは実行時間を犠牲にする上に境界付近でなお微妙に不安定
    なので、ここでは sleep せずに「更新前の updated_at を明示的に大昔の日時へ書き換えて
    から ORM 経由で1回 UPDATE する」ことで確実な分離を作る。onupdate が効いていれば
    UPDATE 後の updated_at は現在時刻(西暦2000年よりずっと後)になり、効いていなければ
    書き換えた大昔の日時のまま変わらない。これにより秒精度に関係なく `>`(等号なし)で
    判定でき、壊れている場合は確実に失敗する。
    """
    backdated = datetime(2000, 1, 1, 0, 0, 0)

    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        conv_id = conv.id

    async with _test_engine.begin() as conn:
        await conn.execute(
            text("UPDATE conversations SET updated_at = :ts WHERE id = :id"),
            {"ts": backdated, "id": conv_id},
        )

    async with db_session_factory() as s:
        conv = await s.get(Conversation, conv_id)
        conv.status = "escalated"
        await s.commit()
        assert conv.updated_at > backdated


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
