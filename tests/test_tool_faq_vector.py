from app.tools.business import query_faq


async def test_query_faq_maps_hits_to_contract(monkeypatch):
    async def fake_search(keyword, **kw):
        return [{"id": 1, "score": 0.8,
                 "question": "送料はどう計算されますか", "answer": "3,000円以上で送料無料"}]
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "送料はいくらですか"})
    assert out == {"hits": [{"question": "送料はどう計算されますか", "answer": "3,000円以上で送料無料"}]}


async def test_query_faq_empty_returns_message(monkeypatch):
    async def fake_search(keyword, **kw):
        return []
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "無関係な質問"})
    assert out["hits"] == []
    assert "見つかりませんでした" in out["message"]
