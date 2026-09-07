"""低信頼質問プールへの投入。回答拒否パスの副作用なので、実際に行が残ることを見る。"""

import pytest
from sqlalchemy import text

from app.db import repository

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_insert_low_confidence_persists_the_row(db_session_factory):
    """戻り値の id だけでなく、その id で実際に読み戻せることまで確かめる。

    ここが黙って no-op になっても上位(Task 11 の回答拒否パス)は何も気づかない。
    質問プールは 09 章のデータフライホイールの入口なので、
    「拒否はしたが 1 件も溜まっていない」状態を検知できるのはこのテストだけ。
    """
    cid = await repository.create_conversation("u1")
    lid = await repository.insert_low_confidence(
        cid, "送料って結局いくら？", "self_check", "根拠不足"
    )
    assert lid > 0

    async with db_session_factory() as s:
        row = (
            await s.execute(
                text(
                    "SELECT conversation_id, raw_question, source, reason "
                    "FROM low_confidence_questions WHERE id=:i"
                ),
                {"i": lid},
            )
        ).first()
    assert row is not None
    assert row.conversation_id == cid
    assert row.raw_question == "送料って結局いくら？"
    assert row.source == "self_check"
    assert row.reason == "根拠不足"


async def test_insert_low_confidence_accepts_null_conversation(db_session_factory):
    """会話に紐づかない経路(評価スクリプトなど)からも積めること。"""
    lid = await repository.insert_low_confidence(
        None, "conversation のない質問", "retrieval_low_conf", None
    )
    async with db_session_factory() as s:
        row = (
            await s.execute(
                text(
                    "SELECT conversation_id, source, reason "
                    "FROM low_confidence_questions WHERE id=:i"
                ),
                {"i": lid},
            )
        ).first()
    assert row.conversation_id is None
    assert row.source == "retrieval_low_conf"
    assert row.reason is None


async def test_insert_low_confidence_survives_unknown_conversation(db_session_factory):
    """存在しない conversation_id でも例外にしない。

    conversation_id は conversations への FK なので、素直に INSERT すると
    IntegrityError になる。呼び出し元(Task 11)は「根拠が足りないので断る」という
    正常系の途中でここを呼ぶため、ここで例外を投げると穏当な回答拒否が 500 に化ける。
    プールにとって本当に必要なのは質問文なので、会話への紐付けだけを捨てて残す。
    """
    lid = await repository.insert_low_confidence(
        999_999_999, "存在しない会話からの質問", "retrieval_low_conf", "top=0.010"
    )
    assert lid > 0

    async with db_session_factory() as s:
        row = (
            await s.execute(
                text(
                    "SELECT conversation_id, raw_question FROM low_confidence_questions "
                    "WHERE id=:i"
                ),
                {"i": lid},
            )
        ).first()
    assert row.conversation_id is None
    assert row.raw_question == "存在しない会話からの質問"


async def test_insert_low_confidence_accepts_every_source_value(db_session_factory):
    """DDL の ENUM 3 値がすべて通ること(値がずれると MySQL 側で空文字に化ける)。"""
    for src in ("retrieval_low_conf", "self_check", "user_feedback"):
        lid = await repository.insert_low_confidence(None, f"{src} の質問", src, None)
        async with db_session_factory() as s:
            got = (
                await s.execute(
                    text("SELECT source FROM low_confidence_questions WHERE id=:i"),
                    {"i": lid},
                )
            ).scalar()
        assert got == src
