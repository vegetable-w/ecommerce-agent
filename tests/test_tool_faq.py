"""app.tools.business.query_faq のテスト。

検索の実体はベクトル検索(app.core.retrieval.search_knowledge)なので、埋め込み上流と
Milvus を呼ばないようにモックする。DB には触れないため、conftest.py の
pytest.mark.asyncio(loop_scope="session") は不要。
"""

import pytest
from pydantic import ValidationError

from app.tools.business import query_faq


def _stub_search(monkeypatch, hits):
    """search_knowledge を差し替え、渡された keyword を記録して返す。"""
    seen = {}

    async def fake_search(keyword, **kw):
        seen["keyword"] = keyword
        return hits

    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    return seen


async def test_query_faq_hit(monkeypatch):
    seen = _stub_search(monkeypatch, [
        {"id": 1, "score": 0.72, "question": "返品ポリシー", "answer": "7日間の返品に対応"},
    ])
    r = await query_faq.ainvoke({"keyword": "返品"})
    assert r["hits"] and r["hits"][0]["question"] == "返品ポリシー"
    # キーワードが検索へそのまま渡ること(ツールが問い合わせ文を握り潰さない)
    assert seen["keyword"] == "返品"


async def test_query_faq_miss_returns_message(monkeypatch):
    _stub_search(monkeypatch, [])
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
