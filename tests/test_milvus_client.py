import uuid

import pytest

from app.kb import milvus_client as mc


@pytest.fixture()
def client(monkeypatch):
    """テストごとに固有名の collection へ差し替え、後片付けする。
    standalone は 1 サーバ共有なので、Lite の tmp_path のようには隔離できない。"""
    monkeypatch.setattr(mc, "COLLECTION", "test_" + uuid.uuid4().hex[:12])
    # 差し替えが効かないまま本番の "knowledge" を汚す事故を、書き込み前に止める
    assert mc.COLLECTION.startswith("test_")
    c = mc.get_client()
    mc.ensure_collection(c)
    yield c
    mc.drop_collection(c)


def _vec(seed: float) -> list[float]:
    # 1024 次元、方向を区別できるベクトルを作る
    v = [0.0] * mc.DIM
    v[0] = seed
    v[1] = 1.0 - seed
    return v


def test_upsert_then_search_returns_fields(client):
    mc.upsert_vectors(client, [
        {"id": 1, "vector": _vec(1.0), "question": "送料はどう計算されますか", "answer": "9900円以上で送料無料"},
        {"id": 2, "vector": _vec(0.0), "question": "発送までの日数", "answer": "48時間以内に発送"},
    ])
    hits = mc.search(client, _vec(0.98), top_k=1)
    assert len(hits) == 1
    assert hits[0]["id"] == 1
    assert hits[0]["question"] == "送料はどう計算されますか"
    assert hits[0]["answer"] == "9900円以上で送料無料"
    assert isinstance(hits[0]["score"], float)


def test_upsert_is_idempotent_by_pk(client):
    row = {"id": 1, "vector": _vec(1.0), "question": "q", "answer": "a"}
    mc.upsert_vectors(client, [row])
    mc.upsert_vectors(client, [row])  # 同じ id で書き直す
    assert mc.count(client) == 1
