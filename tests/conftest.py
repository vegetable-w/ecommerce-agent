"""DB を触るテストの共通fixture。

Note: DB を触るテストモジュールには、モジュール先頭で
`pytestmark = pytest.mark.asyncio(loop_scope="session")` が必要(tests/test_db_conn.py 参照)。
_test_engine はセッションスコープのイベントループ上に作られるため、そのモジュールのテスト自体も
同じループで動かさないと、asyncmy のコネクションが別ループに紐付いて
"attached to a different loop" の RuntimeError になる。今後 DB テストを追加するモジュールでも
同様にこのマーカーを付けること(pyproject.toml の asyncio_default_fixture_loop_scope の
コメントも参照)。
"""

import pathlib

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings

_DDL_FILES = [
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "02-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "03-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "04-ddl.sql",
    # 06 は ALTER のみ(tickets.ticket_type へ refund を足す)。CREATE TABLE より後に
    # 置くこと。ここに入れ忘れると support_test の ENUM だけ古いままになり、
    # ORM と実スキーマを突き合わせる tests/test_models.py が落ちる。
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "06-ticket-type.sql",
    # 07 は 2 本。conversations へ列を足す方が先で、分段要約の表はその後。
    # 順序は file の並びで保証している(CREATE より前に ALTER が来ないこと)。
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "07-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "07-layers.sql",
    # 08 は tool_audit_logs の CREATE TABLE。FK を持たないので順序の制約はない。
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "08-ddl.sql",
]
# 削除順: 子 → 親。knowledge_chunks の自己参照 FK は FOREIGN_KEY_CHECKS=0 で吸収する。
# low_confidence_questions は conversations への FK を持つので conversations より先に置く。
# faith_cases は FK を持たないため末尾でよい。
# tool_audit_logs も FK を持たないが、conversation_id で会話を指すので会話より先に消す。
# conversation_summaries は conversations への FK を持たないが、会話より先に消す
# (会話が消えたのに要約の断片だけ残ると、次のテストが前のテストの断片を読む)。
_TABLES = ["low_confidence_questions", "conversation_summaries", "messages", "tickets",
           "tool_audit_logs", "conversations", "faq", "qa_extraction_staging",
           "knowledge_chunks", "faith_cases"]


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
    stmts = []
    for ddl in _DDL_FILES:
        stmts += _split_sql_statements(ddl.read_text(encoding="utf-8"))
    async with engine.begin() as conn:
        for s in stmts:
            # SET NAMES などのセッション設定は engine 側で済んでいるので流さない。
            # 実行するのはスキーマを作る文だけに絞る。
            if s.lstrip().upper().startswith(("CREATE TABLE", "ALTER TABLE")):
                await conn.execute(text(s))
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(loop_scope="session")
async def db_clean(_test_engine):
    """各テストの前後に _TABLES の全テーブルを空にする。

    後片付け(yield 後)だけだと、db_clean を要求しなかった直前のテストが書き込みを
    残した場合に、db_clean を正しく要求した次のテストがその汚れをテスト開始時点で
    引き継いでしまう(汚れは「片付け忘れたテスト」ではなく「次のテスト」の失敗として
    現れ、原因調査を誤らせる)。前方でも truncate することで、このfixtureを要求した
    テストは他のテストの後始末忘れに関わらず常に空の状態から始まることを保証する。
    """

    async def _truncate() -> None:
        async with _test_engine.begin() as conn:
            await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
            for t in _TABLES:
                await conn.execute(text(f"TRUNCATE TABLE {t}"))
            await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))

    await _truncate()
    yield
    await _truncate()


@pytest_asyncio.fixture(loop_scope="session")
async def db_session_factory(_test_engine, db_clean, monkeypatch):
    """repository が利用する async_session をテスト DB 用 factory に差し替える。

    db_clean に依存させることで、このfixtureを要求したテストには必ず前後の
    クリーンアップが付いてくる。db_session_factory だけを要求して db_clean を
    要求し忘れる、という抜け道を構造的になくすため。
    """
    factory = async_sessionmaker(_test_engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.base.async_session", factory)
    return factory
