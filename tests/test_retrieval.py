import uuid

import pytest

from app.core import retrieval
from app.kb import milvus_client


@pytest.fixture()
def milvus(monkeypatch):
    monkeypatch.setattr(milvus_client, "COLLECTION", "test_" + uuid.uuid4().hex[:12])
    assert milvus_client.COLLECTION.startswith("test_")
    c = milvus_client.get_client()
    milvus_client.ensure_collection(c)
    v_hit = [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)
    v_other = [0.0, 1.0, 0.0] + [0.0] * (milvus_client.DIM - 3)
    milvus_client.upsert_vectors(c, [
        {"id": 1, "vector": v_hit, "question": "送料はどう計算されますか", "answer": "3,000円以上で送料無料"},
        {"id": 2, "vector": v_other, "question": "発送までの日数", "answer": "48時間以内"},
    ])
    yield c
    milvus_client.drop_collection(c)


async def _fake_embed(_query):
    return [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)


async def test_search_returns_top_hits_above_threshold(monkeypatch, milvus):
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", _fake_embed)
    hits = await retrieval.search_knowledge("送料はいくらですか", top_k=1, min_score=0.5, client=milvus)
    assert hits and hits[0]["question"] == "送料はどう計算されますか"


async def test_below_threshold_filtered(monkeypatch, milvus):
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", _fake_embed)
    hits = await retrieval.search_knowledge("送料", top_k=2, min_score=1.0001, client=milvus)
    assert hits == []
