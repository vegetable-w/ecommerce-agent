"""app.tools.business.query_faq のテスト。

DB を触るため、tests/conftest.py の注記どおりモジュール先頭で
pytest.mark.asyncio(loop_scope="session") を指定する必要がある。
"""

import pytest
from pydantic import ValidationError

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


async def test_query_faq_empty_keyword_raises_validation_error():
    """Empty keyword should raise ValidationError, not return all rows."""
    with pytest.raises(ValidationError):
        await query_faq.ainvoke({"keyword": ""})


async def test_query_faq_whitespace_only_raises_validation_error():
    """Whitespace-only keyword should raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await query_faq.ainvoke({"keyword": "   "})
    assert "空白のみの値は許可されない" in str(exc_info.value)
