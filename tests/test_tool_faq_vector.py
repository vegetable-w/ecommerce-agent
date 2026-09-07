"""query_faq が検索結果を「ツールの戻り値」へ写す部分の契約。

03 章では hits をそのまま返していた。04 章では番号付き evidence + citations になり、
根拠が無ければ回答拒否の指示を返す。ここでは写し replace の形だけを固定する
(ゲートの分岐は tests/test_query_faq_rag.py)。
"""

import pytest

from app.core.prompts import RAG_CITATION_NOTICE
from app.tools.business import query_faq


@pytest.fixture(autouse=True)
def _stub_pipeline(monkeypatch):
    async def fake_understand(q, **kw):
        return {"standard": q, "expanded": []}

    async def fake_check(q, texts, **kw):
        return {"useful": True, "reason": "十分"}

    def _boom(*a, **kw):
        raise AssertionError("本物の上流モデルが呼ばれた")

    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    monkeypatch.setattr("app.core.llm.get_chat_model", _boom)


async def test_query_faq_maps_hits_to_citations(monkeypatch):
    async def fake_search(query, **kw):
        return [{"id": 7, "rerank_score": 0.8, "question": "送料はどう計算されますか",
                 "answer": "3,000円以上で送料無料", "section_path": "送料ポリシー",
                 "content_type": "faq", "category": "送料"}]
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "送料はいくらですか"})
    assert out == {
        "sufficient": True,
        "notice": RAG_CITATION_NOTICE,
        "evidence": "[1] 送料はどう計算されますか: 3,000円以上で送料無料",
        "citations": [{"n": 1, "id": 7, "section_path": "送料ポリシー",
                       "question": "送料はどう計算されますか",
                       "answer": "3,000円以上で送料無料", "content_type": "faq"}],
    }


async def test_sufficient_result_tells_the_model_to_cite(monkeypatch):
    """収束生成が読む AGENT_SYSTEM(02 章、凍結)には引用ルールが無い。

    引用の指示を tool の戻り値本文に載せないと、回答に [n] が一切現れず、
    受け入れ基準 3(引用番号から原文へ辿れる)とクリック可能な引用が成立しない。
    """
    async def fake_search(query, **kw):
        return [{"id": 7, "rerank_score": 0.8, "question": "q", "answer": "a",
                 "section_path": "p", "content_type": "faq", "category": "c"}]
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "送料"})
    assert "[1] [2] の形式" in out["notice"]
    assert "evidence に書かれていないことは" in out["notice"]


async def test_query_faq_empty_returns_refusal(monkeypatch):
    async def fake_search(query, **kw):
        return []
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "無関係な質問"})
    assert out["sufficient"] is False
    assert out["citations"] == []
    assert out["notice"]
