"""milvus_client の基本(upsert とフィールドの往復、PK 冪等性)。

03 章はモジュール定数 mc.COLLECTION を monkeypatch して隔離していたが、04 章で
各関数が collection を引数に取るようになったため、一時 collection の名前を
明示的に渡す方式に変えた。定数の差し替えは、渡し忘れた呼び出しが 1 つでもあると
本番の knowledge に書いてしまうので、引数の方が事故に強い。
"""

import uuid

import pytest

from app.config import settings
from app.kb import milvus_client as mc


@pytest.fixture()
def coll():
    """テストごとに固有名の collection を作り、後片付けする。
    standalone は 1 サーバ共有なので、Lite の tmp_path のようには隔離できない。"""
    name = "test_" + uuid.uuid4().hex[:12]
    assert name.startswith("test_"), "本番 collection を汚す事故を書き込み前に止める"
    c = mc.get_client(uri=settings.milvus_uri)
    mc.ensure_collection(c, collection=name)
    yield c, name
    mc.drop(c, name)


def _vec(seed: float) -> list[float]:
    # 1024 次元、方向を区別できるベクトルを作る
    v = [0.0] * mc.DIM
    v[0] = seed
    v[1] = 1.0 - seed
    return v


def _row(i: int, seed: float, q: str, a: str) -> dict:
    return {"id": i, "dense": _vec(seed), "text": f"{q} {a}",
            "question": q, "answer": a, "section_path": f"配送ポリシー / {q}",
            "content_type": "policy", "category": "配送"}


def test_upsert_then_search_returns_fields(coll):
    client, name = coll
    mc.upsert_vectors(client, [
        _row(1, 1.0, "送料はどう計算されますか", "9900円以上で送料無料"),
        _row(2, 0.0, "発送までの日数", "48時間以内に発送"),
    ], collection=name)
    hits = mc.dense_search(client, _vec(0.98), top_k=1, collection=name)
    assert len(hits) == 1
    assert hits[0]["id"] == 1
    assert hits[0]["question"] == "送料はどう計算されますか"
    assert hits[0]["answer"] == "9900円以上で送料無料"
    assert hits[0]["section_path"] == "配送ポリシー / 送料はどう計算されますか"
    assert hits[0]["content_type"] == "policy"
    assert hits[0]["category"] == "配送"
    assert isinstance(hits[0]["score"], float)


def test_upsert_is_idempotent_by_pk(coll):
    client, name = coll
    row = _row(1, 1.0, "q", "a")
    mc.upsert_vectors(client, [row], collection=name)
    mc.upsert_vectors(client, [row], collection=name)  # 同じ id で書き直す
    assert mc.count(client, collection=name) == 1
