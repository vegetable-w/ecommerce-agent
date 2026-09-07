"""04 章の検索: 4 戦略の分岐、リランクの足切り、head/tail 配置。

戦略が本当に別経路を通ることは、実際の Milvus 上の一時 collection で確かめる
(モックだけだと全部同じ経路へ落ちても気付けない)。埋め込みとリランクの上流は呼ばない。
"""

import uuid

import pytest

from app.config import settings
from app.core import retrieval
from app.kb import milvus_client as mc


def _vec(i: int) -> list[float]:
    v = [0.0] * mc.DIM
    v[i] = 1.0
    return v


# ---------------------------------------------------------------------------
# head/tail 配置
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("items, expected", [
    ([], []),
    (["a"], ["a"]),
    (["a", "b"], ["a", "b"]),
    (["a", "b", "c"], ["a", "c", "b"]),
    (["a", "b", "c", "d"], ["a", "c", "d", "b"]),
])
def test_arrange_head_tail_small_cases(items, expected):
    assert retrieval.arrange_head_tail(items) == expected


def test_arrange_head_tail_ten_items():
    """関連度降順の 10 件 → 1 位が先頭、2 位が末尾、残りは順序を保って中央。"""
    src = list(range(10))
    out = retrieval.arrange_head_tail(src)
    assert out[0] == 0
    assert out[-1] == 1
    assert out[1:-1] == [2, 3, 4, 5, 6, 7, 8, 9]
    assert sorted(out) == src


def test_arrange_head_tail_does_not_mutate_input():
    src = ["a", "b", "c"]
    retrieval.arrange_head_tail(src)
    assert src == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 戦略の分岐(呼び出し経路をモックで確認)
# ---------------------------------------------------------------------------


def _hit(i: int, score: float = 0.9) -> dict:
    return {"id": i, "score": score, "question": f"q{i}", "answer": f"a{i}",
            "section_path": f"p{i}", "content_type": "faq", "category": "送料"}


@pytest.fixture()
def routes(monkeypatch):
    """3 つの検索経路を記録する差し替え。どれが呼ばれたかを called に残す。"""
    called: dict = {}

    async def fake_embed(_q):
        return [0.1] * mc.DIM

    def fake_dense(client, vector, top_k, category=None, collection=mc.COLLECTION):
        called["dense"] = {"top_k": top_k, "category": category}
        return [_hit(1)]

    def fake_bm25(client, text, top_k, category=None, collection=mc.COLLECTION):
        called["bm25"] = {"text": text, "top_k": top_k, "category": category}
        return [_hit(2)]

    def fake_hybrid(client, vector, text, top_k, recall=50, category=None,
                    collection=mc.COLLECTION):
        called["hybrid"] = {"top_k": top_k, "recall": recall, "category": category}
        return [_hit(3, 0.02), _hit(4, 0.016)]

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.core.retrieval.milvus_client.ensure_collection",
                        lambda *a, **k: None)
    monkeypatch.setattr("app.core.retrieval.milvus_client.dense_search", fake_dense)
    monkeypatch.setattr("app.core.retrieval.milvus_client.bm25_search", fake_bm25)
    monkeypatch.setattr("app.core.retrieval.milvus_client.hybrid_search", fake_hybrid)
    return called


async def test_dense_strategy_uses_dense_only(routes):
    out = await retrieval.search_knowledge("送料", strategy="dense", min_score=0.0,
                                           client=object())
    assert list(routes) == ["dense"]
    assert [h["id"] for h in out] == [1]


async def test_vector_is_an_alias_of_dense(routes):
    """03 章と計画書が使っていた名前。dense と同じ経路へ寄せる。"""
    await retrieval.search_knowledge("送料", strategy="vector", min_score=0.0, client=object())
    assert list(routes) == ["dense"]


async def test_bm25_strategy_uses_bm25_only_and_does_not_embed(routes):
    out = await retrieval.search_knowledge("送料", strategy="bm25", client=object())
    assert list(routes) == ["bm25"]
    assert routes["bm25"]["text"] == "送料"
    assert [h["id"] for h in out] == [2]


async def test_hybrid_strategy_recalls_and_truncates(routes, monkeypatch):
    async def no_rerank(*a, **k):
        raise AssertionError("hybrid はリランクを呼んではいけない")

    monkeypatch.setattr("app.core.retrieval.rerank.rerank", no_rerank)
    out = await retrieval.search_knowledge("送料", strategy="hybrid", top_k=1, client=object())
    assert list(routes) == ["hybrid"]
    assert routes["hybrid"]["recall"] == settings.recall_top_k
    assert [h["id"] for h in out] == [3]


async def test_hybrid_rerank_reorders_and_attaches_score(routes, monkeypatch):
    async def fake_rerank(q, docs, top_n=None):
        return [(1, 0.95), (0, 0.44)]  # 2 件目のほうが関連度が高い

    monkeypatch.setattr("app.core.retrieval.rerank.rerank", fake_rerank)
    out = await retrieval.search_knowledge("送料", strategy="hybrid_rerank", client=object())
    assert list(routes) == ["hybrid"]
    assert [h["id"] for h in out] == [4, 3]
    assert out[0]["rerank_score"] == 0.95
    # 融合スコアは上書きせずそのまま残す(尺度が別物なので両方見たい)
    assert out[0]["score"] == 0.016


async def test_unknown_strategy_raises(routes):
    with pytest.raises(ValueError, match="hybird"):
        await retrieval.search_knowledge("送料", strategy="hybird", client=object())
    assert routes == {}


async def test_category_filter_is_passed_through(routes):
    await retrieval.search_knowledge("送料", strategy="dense", category="返品",
                                     min_score=0.0, client=object())
    assert routes["dense"]["category"] == "返品"


# ---------------------------------------------------------------------------
# リランクの足切りと縮退
# ---------------------------------------------------------------------------


async def test_rerank_gate_drops_hits_below_threshold(routes, monkeypatch):
    """rerank_min_score を下回る証拠は返さない。ここが 04 章の足切りの本体。"""
    async def fake_rerank(q, docs, top_n=None):
        return [(0, 0.91), (1, 0.05)]

    monkeypatch.setattr("app.core.retrieval.rerank.rerank", fake_rerank)
    monkeypatch.setattr(settings, "rerank_min_score", 0.3)
    out = await retrieval.search_knowledge("送料", strategy="hybrid_rerank", client=object())
    assert [h["id"] for h in out] == [3]


async def test_rerank_gate_keeps_hits_on_the_threshold(routes, monkeypatch):
    async def fake_rerank(q, docs, top_n=None):
        return [(0, 0.3)]

    monkeypatch.setattr("app.core.retrieval.rerank.rerank", fake_rerank)
    monkeypatch.setattr(settings, "rerank_min_score", 0.3)
    out = await retrieval.search_knowledge("送料", strategy="hybrid_rerank", client=object())
    assert [h["id"] for h in out] == [3]


async def test_rerank_gate_can_drop_everything(routes, monkeypatch):
    async def fake_rerank(q, docs, top_n=None):
        return [(0, 0.01), (1, 0.0)]

    monkeypatch.setattr("app.core.retrieval.rerank.rerank", fake_rerank)
    out = await retrieval.search_knowledge("無関係な質問", strategy="hybrid_rerank",
                                           client=object())
    assert out == []


async def test_rerank_failure_degrades_to_hybrid_order(routes, monkeypatch):
    """リランク上流が落ちた場合([] が返る)は、ハイブリッドの並びで返す。"""
    async def dead_rerank(q, docs, top_n=None):
        return []

    monkeypatch.setattr("app.core.retrieval.rerank.rerank", dead_rerank)
    out = await retrieval.search_knowledge("送料", strategy="hybrid_rerank", client=object())
    assert [h["id"] for h in out] == [3, 4]
    assert "rerank_score" not in out[0]


async def test_hybrid_rerank_with_no_recall_returns_empty(monkeypatch):
    async def fake_embed(_q):
        return [0.1] * mc.DIM

    async def boom(*a, **k):
        raise AssertionError("recall が空ならリランクを呼ぶ必要はない")

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.core.retrieval.milvus_client.ensure_collection",
                        lambda *a, **k: None)
    monkeypatch.setattr("app.core.retrieval.milvus_client.hybrid_search",
                        lambda *a, **k: [])
    monkeypatch.setattr("app.core.retrieval.rerank.rerank", boom)
    assert await retrieval.search_knowledge("x", strategy="hybrid_rerank", client=object()) == []


# ---------------------------------------------------------------------------
# 実際の Milvus で 4 戦略が別々の結果を返すことを確認する
# ---------------------------------------------------------------------------


@pytest.fixture()
def corpus():
    """dense と BM25 が別の行に当たるように作った 3 行の一時 collection。"""
    client = mc.get_client(uri=settings.milvus_uri)
    name = "test_ret_" + uuid.uuid4().hex[:12]
    assert name.startswith("test_ret_"), "本番 collection を汚す事故を書き込み前に止める"
    mc.ensure_collection(client, collection=name)
    mc.upsert_vectors(client, [
        {"id": 1, "dense": _vec(0), "text": "配送 発送は翌営業日に行います",
         "question": "発送はいつですか", "answer": "翌営業日に発送します",
         "section_path": "配送ポリシー", "content_type": "policy", "category": "配送"},
        {"id": 2, "dense": _vec(1),
         "text": "ロボット掃除機 Max 型番 EC-RV300 稼働時間 約210分",
         "question": "EC-RV300 の稼働時間は", "answer": "満充電で約210分稼働します",
         "section_path": "商品仕様", "content_type": "manual", "category": "商品仕様"},
        {"id": 3, "dense": _vec(2), "text": "返品 商品到着後7日以内であれば返品できます",
         "question": "返品はできますか", "answer": "到着後7日以内なら返品できます",
         "section_path": "返品ポリシー", "content_type": "policy", "category": "返品"},
    ], collection=name)
    yield client, name
    mc.drop(client, name)


async def test_four_strategies_return_different_results(corpus, monkeypatch):
    client, coll = corpus

    async def fake_embed(_q):
        return _vec(0)          # dense は必ず id=1 に当たる

    async def fake_rerank(q, docs, top_n=None):
        # 型番の行だけを高スコアにする。内容で選ぶので recall の並び順に依存しない
        i = next(n for n, d in enumerate(docs) if "EC-RV300" in d)
        return [(i, 0.97)]

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.core.retrieval.rerank.rerank", fake_rerank)

    kw = {"client": client, "collection": coll}
    dense = await retrieval.search_knowledge("返品", strategy="dense", top_k=1,
                                             min_score=0.5, **kw)
    bm25 = await retrieval.search_knowledge("返品", strategy="bm25", top_k=1, **kw)
    hybrid = await retrieval.search_knowledge("返品", strategy="hybrid", top_k=2, **kw)
    rr = await retrieval.search_knowledge("返品", strategy="hybrid_rerank", **kw)

    assert [h["id"] for h in dense] == [1]          # ベクトルだけを見た結果
    assert [h["id"] for h in bm25] == [3]           # 語だけを見た結果
    assert {h["id"] for h in hybrid} == {1, 3}      # 両方を融合した結果
    assert [h["id"] for h in rr] == [2]             # リランクが選び直した結果
    assert rr[0]["rerank_score"] == 0.97
