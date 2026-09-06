"""03 章で追加した ORM モデルの DB ラウンドトリップ。

_test_engine はセッションスコープのイベントループ上で作成される(tests/conftest.py 参照)。
このモジュールのテストも同じループで動かさないと asyncmy のコネクションが別ループに
紐付いたままになり "attached to a different loop" の RuntimeError になる。
"""

import pytest
from sqlalchemy import select

from app.db.models import KnowledgeChunk, QaExtractionStaging

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_knowledge_chunk_roundtrip(db_session_factory):
    async with db_session_factory() as s:
        row = KnowledgeChunk(category="返品ポリシー", questions="送料について", answer="9900円以上で送料無料")
        s.add(row)
        await s.commit()
        await s.refresh(row)
        assert row.id is not None
        assert row.vectorize_status == "pending"
        assert row.is_key_clause == 0


async def test_staging_roundtrip(db_session_factory):
    async with db_session_factory() as s:
        row = QaExtractionStaging(batch_no="b1", question="送料はいくらですか", answer="9900円以上で無料")
        s.add(row)
        await s.commit()
        got = (await s.execute(select(QaExtractionStaging))).scalars().all()
        assert len(got) == 1
        assert got[0].status == "extracted"
