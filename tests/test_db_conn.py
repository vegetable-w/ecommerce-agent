import pytest
from sqlalchemy import text

# _test_engine はセッションスコープのイベントループ上で作成される(pyproject.toml の
# asyncio_default_fixture_loop_scope="session" 参照)。このモジュールのテストが既定の
# function スコープのループで実行されると、asyncmy のコネクションが別ループに紐付いた
# ままになり "attached to a different loop" の RuntimeError になる。DB に触れるテスト
# モジュールは必ずこのマーカーでテスト自体もセッションループに揃えること。
pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_test_db_reachable_and_tables_exist(_test_engine, db_clean):
    async with _test_engine.connect() as conn:
        rows = (await conn.execute(text("SHOW TABLES"))).scalars().all()
    assert {"conversations", "messages", "faq", "tickets"} <= set(rows)


async def test_leaky_write_without_requesting_db_clean(_test_engine):
    """わざと db_clean を要求せず1行書き込む。次のテストが db_clean を要求していれば、
    db_clean の前方truncateがこの行を消してくれるはず(後方truncateだけだと、
    このテストは自分の後始末をしないので、次のテストの先頭にこの行が漏れて見える)。
    """
    async with _test_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO conversations (user_id, status) VALUES ('leaky-user', 'in_progress')")
        )


async def test_db_clean_starts_empty_even_after_a_leaky_prior_test(_test_engine, db_clean):
    async with _test_engine.connect() as conn:
        count = (
            await conn.execute(text("SELECT COUNT(*) FROM conversations"))
        ).scalar_one()
    assert count == 0
