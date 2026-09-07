"""実際の Milvus Standalone に対して、dense / BM25 / ハイブリッドの 3 経路を確かめる。

Standalone は 1 サーバ共有なので、Lite の tmp_path のようにファイルで隔離できない。
各テストは一意名の test_* collection を作り、teardown で必ず drop する。
本番の knowledge collection には一切触れない。
"""

import uuid

import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema

from app.config import settings
from app.kb import milvus_client as mc


def _vec(i: int) -> list[float]:
    """i 番目の軸だけが立った 1024 次元。互いに直交するので dense の当たり外れを作れる。"""
    v = [0.0] * mc.DIM
    v[i] = 1.0
    return v


def _name() -> str:
    return "test_kb_" + uuid.uuid4().hex[:12]


@pytest.fixture()
def coll():
    """3 行入りの一時 collection。(client, name) を返す。"""
    client = mc.get_client(uri=settings.milvus_uri)
    name = _name()
    assert name.startswith("test_kb_"), "本番 collection を汚す事故を書き込み前に止める"
    mc.ensure_collection(client, collection=name)
    mc.upsert_vectors(client, [
        {"id": 1, "dense": _vec(0),
         "text": "送料 1回の注文金額が3,000円以上の場合は送料無料 3,000円未満は500円",
         "question": "送料はいくらですか", "answer": "3,000円以上のご注文は送料無料です",
         "section_path": "配送ポリシー / 送料", "content_type": "policy", "category": "送料"},
        {"id": 2, "dense": _vec(1),
         "text": "ロボット掃除機 Max 型番 EC-RV300 稼働時間 約210分 吸引力 8,000Pa",
         "question": "EC-RV300 の稼働時間は", "answer": "満充電で約210分稼働します",
         "section_path": "商品仕様 / ロボット掃除機", "content_type": "manual",
         "category": "商品仕様"},
        {"id": 3, "dense": _vec(2),
         "text": "返品 商品到着後7日以内であれば返品を受け付けます",
         "question": "返品はできますか", "answer": "到着後7日以内なら返品できます",
         "section_path": "返品ポリシー", "content_type": "policy", "category": "返品"},
    ], collection=name)
    yield client, name
    mc.drop(client, name)


# ---------------------------------------------------------------------------
# BM25 単路
# ---------------------------------------------------------------------------


def test_bm25_hits_model_number_and_excludes_unrelated(coll):
    """型番のキーワードが正しい行に当たり、無関係な行は返らないこと。

    受け入れ基準 2 の土台。icu tokenizer は EC-RV300 を割らないので型番が語として残る
    (lindera は ec/rv/300 に割ってしまい、ここが崩れる)。
    top_k=3 で全行を要求してもなお id=2 だけが返ることを見ることで、
    「実は BM25 ではなく全件を返しているだけ」を排除する。
    """
    client, name = coll
    hits = mc.bm25_search(client, "EC-RV300 の稼働時間", top_k=3, collection=name)
    assert [h["id"] for h in hits] == [2]
    assert hits[0]["section_path"] == "商品仕様 / ロボット掃除機"
    assert hits[0]["content_type"] == "manual"


def test_bm25_search_is_available_immediately_after_upsert(coll):
    """consistency_level=Strong により、upsert 直後に flush 無しでも BM25 が当たる。

    既定の Bounded では 0 件になる(03 章で実測)。ここが落ちたら
    ensure_collection の consistency_level が外れたと思ってよい。
    """
    client, name = coll
    extra_id = 99
    mc.upsert_vectors(client, [
        {"id": extra_id, "dense": _vec(3),
         "text": "保証 メーカー保証は購入日から2年間です",
         "question": "保証期間は", "answer": "購入日から2年間です",
         "section_path": "保証ポリシー", "content_type": "policy", "category": "保証"},
    ], collection=name)
    hits = mc.bm25_search(client, "メーカー保証 2年間", top_k=3, collection=name)
    assert extra_id in [h["id"] for h in hits]


# ---------------------------------------------------------------------------
# dense 単路
# ---------------------------------------------------------------------------


def test_dense_search_scores_are_similarity_not_distance(coll):
    """Standalone の COSINE distance は類似度そのもの(大きいほど近い)。

    03 章の Lite は 1−similarity を返していた。ここを取り違えると min_score による
    足切りの向きが逆になり、正解だけが捨てられる。
    """
    client, name = coll
    hits = mc.dense_search(client, _vec(1), top_k=3, collection=name)
    assert hits[0]["id"] == 2
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert hits[0]["score"] > hits[-1]["score"]


# ---------------------------------------------------------------------------
# ハイブリッド
# ---------------------------------------------------------------------------


def test_hybrid_returns_scored_hits_with_metadata(coll):
    client, name = coll
    hits = mc.hybrid_search(client, _vec(0), "送料", top_k=2, recall=10, collection=name)
    assert hits
    assert {"id", "score", "question", "answer",
            "section_path", "content_type", "category"} <= hits[0].keys()
    assert hits[0]["id"] == 1
    assert isinstance(hits[0]["score"], float)


def test_hybrid_recalls_what_dense_alone_cannot(coll):
    """BM25 経路が実際に効いていること。

    recall=1 に絞ることで、dense 経路が拾えるのは query ベクトルと同じ向きの id=1 だけになる。
    それでも id=2 が返るなら、それは BM25 経路が拾った以外にありえない。
    hybrid_search から sparse の AnnSearchRequest を外すと、このテストだけが落ちる
    (= ハイブリッドが黙って dense 単路に退化する事故を検出できる)。
    """
    client, name = coll
    hits = mc.hybrid_search(client, _vec(0), "EC-RV300", top_k=5, recall=1, collection=name)
    ids = {h["id"] for h in hits}
    assert 1 in ids, "dense 経路が効いていない"
    assert 2 in ids, "BM25 経路が効いていない(dense だけでは id=2 は届かない)"


# ---------------------------------------------------------------------------
# メタデータ絞り込み
# ---------------------------------------------------------------------------


def test_bm25_category_filter_excludes_other_categories(coll):
    """category を指定すると、キーワードが当たる他カテゴリの行が実際に消えること。

    絞り込み無しでは id=2 が最上位に来るクエリを使う。_cat_expr が空文字を返すように
    壊すと、ここが id=2 を拾って落ちる。
    """
    client, name = coll
    unfiltered = mc.bm25_search(client, "EC-RV300 送料", top_k=5, collection=name)
    assert {h["id"] for h in unfiltered} == {1, 2}, "前提: 絞り込み無しなら両方当たる"

    hits = mc.bm25_search(client, "EC-RV300 送料", top_k=5, category="送料", collection=name)
    assert hits, "絞り込みで全部消えてしまった"
    assert all(h["category"] == "送料" for h in hits)
    assert 2 not in {h["id"] for h in hits}


def test_hybrid_category_filter_excludes_other_categories(coll):
    """ハイブリッド側の絞り込みは AnnSearchRequest の expr という別経路なので、別に固定する。"""
    client, name = coll
    unfiltered = mc.hybrid_search(client, _vec(1), "EC-RV300 送料", top_k=5, recall=10,
                                  collection=name)
    assert {1, 2} <= {h["id"] for h in unfiltered}, "前提: 絞り込み無しなら両方当たる"

    hits = mc.hybrid_search(client, _vec(1), "EC-RV300 送料", top_k=5, recall=10,
                            category="送料", collection=name)
    assert hits
    assert all(h["category"] == "送料" for h in hits)
    assert 2 not in {h["id"] for h in hits}


# ---------------------------------------------------------------------------
# collection の管理
# ---------------------------------------------------------------------------


def test_upsert_is_idempotent_by_pk(coll):
    client, name = coll
    row = {"id": 1, "dense": _vec(0), "text": "送料", "question": "q", "answer": "a",
           "section_path": "s", "content_type": "policy", "category": "送料"}
    mc.upsert_vectors(client, [row], collection=name)
    mc.upsert_vectors(client, [row], collection=name)
    assert mc.count(client, collection=name) == 3


def test_ensure_collection_rejects_incompatible_schema():
    """03 章の schema(vector/question/answer のみ)が残っていたら、その場で止まること。

    素通りさせると search の奥で `field section_path not exist` という原因の分かりにくい
    例外になる(実測)。運用者に何をすべきかを伝えられるのは、ここだけ。
    """
    client = mc.get_client(uri=settings.milvus_uri)
    name = _name()
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=mc.DIM),
        FieldSchema("question", DataType.VARCHAR, max_length=2048),
        FieldSchema("answer", DataType.VARCHAR, max_length=8192),
    ])
    client.create_collection(name, schema=schema, consistency_level="Strong")
    try:
        with pytest.raises(RuntimeError) as exc:
            mc.ensure_collection(client, collection=name)
        msg = str(exc.value)
        assert "sparse" in msg and "section_path" in msg, "不足フィールドを示していない"
        assert "再構築" in msg, "運用者への指示が無い"
    finally:
        mc.drop(client, name)


def test_ensure_collection_is_idempotent_for_current_schema(coll):
    client, name = coll
    mc.ensure_collection(client, collection=name)  # 2 回目でも例外にならない
    assert mc.count(client, collection=name) == 3


def test_drop_is_safe_on_missing_collection():
    client = mc.get_client(uri=settings.milvus_uri)
    mc.drop(client, _name())  # 例外にならない
