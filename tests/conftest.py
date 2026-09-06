import pathlib

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings

_DDL = pathlib.Path(__file__).resolve().parent.parent / "sql" / "02-ddl.sql"
_TABLES = ["messages", "tickets", "conversations", "faq"]  # 子 → 親の順


def _split_sql_statements(sql: str) -> list[str]:
    """`;` 単純split は使えない: 各カラムの COMMENT '...' 内に英文の説明があり、
    そこに句読点としての ';' がリテラルに含まれるため(例: 'Message content; may be null...')。
    シングルクォート文字列の内側の ';' と、`--` 行コメントを正しく無視して分割する。
    """
    stmts = []
    buf: list[str] = []
    in_string = False
    i = 0
    n = len(sql)
    while i < n:
        c = sql[i]
        if in_string:
            buf.append(c)
            if c == "'":
                in_string = False
            i += 1
            continue
        if c == "'":
            in_string = True
            buf.append(c)
            i += 1
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            j = sql.find("\n", i)
            if j == -1:
                break
            i = j + 1
            continue
        if c == ";":
            stmts.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return [s for s in stmts if s]


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _test_engine():
    server_url = settings.test_database_url.rsplit("/", 1)[0]
    admin = create_async_engine(server_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text("DROP DATABASE IF EXISTS support_test"))
        await conn.execute(text("CREATE DATABASE support_test CHARACTER SET utf8mb4"))
    await admin.dispose()

    engine = create_async_engine(settings.test_database_url, pool_pre_ping=True)
    stmts = _split_sql_statements(_DDL.read_text(encoding="utf-8"))
    async with engine.begin() as conn:
        for s in stmts:
            if s.lstrip().upper().startswith("CREATE TABLE"):
                await conn.execute(text(s))
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(loop_scope="session")
async def db_session_factory(_test_engine, monkeypatch):
    """repository が利用する async_session をテスト DB 用 factory に差し替える。"""
    factory = async_sessionmaker(_test_engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.base.async_session", factory)
    return factory


@pytest_asyncio.fixture(loop_scope="session")
async def db_clean(_test_engine):
    """各テスト後に 4 テーブルを空にする。"""
    yield
    async with _test_engine.begin() as conn:
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in _TABLES:
            await conn.execute(text(f"TRUNCATE TABLE {t}"))
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
