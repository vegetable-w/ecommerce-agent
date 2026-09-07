import uuid

import pytest

from app.config import settings
from app.core import retrieval
from app.kb import milvus_client


@pytest.fixture()
def milvus():
    """固有名の一時 collection を作り、(client, name) を返す。

    04 章で milvus_client の各関数が collection を引数に取るようになったので、
    モジュール定数の差し替えではなく名前を渡して隔離する。"""
    name = "test_" + uuid.uuid4().hex[:12]
    assert name.startswith("test_")
    c = milvus_client.get_client(uri=settings.milvus_uri)
    milvus_client.ensure_collection(c, collection=name)
    v_hit = [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)
    v_other = [0.0, 1.0, 0.0] + [0.0] * (milvus_client.DIM - 3)
    milvus_client.upsert_vectors(c, [
        {"id": 1, "dense": v_hit, "text": "送料 3,000円以上で送料無料",
         "question": "送料はどう計算されますか", "answer": "3,000円以上で送料無料",
         "section_path": "配送ポリシー / 送料", "content_type": "policy", "category": "送料"},
        {"id": 2, "dense": v_other, "text": "発送 48時間以内に発送します",
         "question": "発送までの日数", "answer": "48時間以内",
         "section_path": "配送ポリシー / 発送", "content_type": "policy", "category": "発送"},
    ], collection=name)
    yield c, name
    milvus_client.drop(c, name)


async def _fake_embed(_query):
    return [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)


async def test_search_returns_top_hits_above_threshold(monkeypatch, milvus):
    client, coll = milvus
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", _fake_embed)
    hits = await retrieval.search_knowledge(
        "送料はいくらですか", top_k=1, min_score=0.5, client=client, collection=coll)
    assert hits and hits[0]["question"] == "送料はどう計算されますか"


async def test_below_threshold_filtered(monkeypatch, milvus):
    client, coll = milvus
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", _fake_embed)
    hits = await retrieval.search_knowledge(
        "送料", top_k=2, min_score=1.0001, client=client, collection=coll)
    assert hits == []
