"""app.tools.business.query_faq のテスト。

DB を触るため、tests/conftest.py の注記どおりモジュール先頭で
pytest.mark.asyncio(loop_scope="session") を指定する必要がある。
"""

import pytest

from app.db.models import Faq
from app.tools.business import query_faq

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_query_faq_hit(db_session_factory, db_clean):
    async with db_session_factory() as s:
        s.add(Faq(question="返品ポリシー", answer="7日間の返品に対応", category="アフターサービス"))
        await s.commit()
    r = await query_faq.ainvoke({"keyword": "返品"})
    assert r["hits"] and r["hits"][0]["question"] == "返品ポリシー"


async def test_query_faq_miss_returns_message(db_session_factory, db_clean):
    async with db_session_factory() as s:
        s.add(Faq(question="返品ポリシー", answer="7日間の返品に対応", category="アフターサービス"))
        await s.commit()
    r = await query_faq.ainvoke({"keyword": "靴"})
    assert r["hits"] == [] and "見つかりません" in r["message"]
