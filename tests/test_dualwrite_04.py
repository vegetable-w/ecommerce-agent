"""04 章のベクトル化が、BM25 用の text と絞り込み用のメタデータを実際に書くこと。

test_dualwrite.py が担うのは「落ちても chunk を失わない」という再実行性。
こちらは「何を Milvus へ書いているか」だけを見る。埋め込みは mock するが、
Milvus は本物の Standalone に一時 collection を作って書き、検索して読み戻す
(row を組み立てた時点で満足すると、フィールド名の綴り違いを取り逃がす)。
"""

import uuid

import pytest

from app.config import settings
from app.kb import dualwrite
from app.kb import milvus_client as mc
from app.kb.documents import Chunk

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture()
def coll():
    name = "test_kb_" + uuid.uuid4().hex[:12]
    assert name.startswith("test_kb_")
    client = mc.get_client(uri=settings.milvus_uri)
    mc.ensure_collection(client, collection=name)
    yield client, name
    mc.drop(client, name)


async def test_vectorize_writes_bm25_text_and_metadata(db_session_factory, coll, monkeypatch):
    client, name = coll

    async def fake_embed(texts):
        return [[0.05] * mc.DIM for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", fake_embed)

    await dualwrite.write_pending([
        Chunk(category="送料", questions="送料はいくらですか",
              answer="3,000円以上のご注文は送料無料です",
              section_path="配送ポリシー / 送料", content_type="policy", is_key_clause=1),
        Chunk(category="商品仕様", questions="EC-RV300 の稼働時間は",
              answer="満充電で約210分稼働します",
              section_path="商品仕様 / ロボット掃除機", content_type="manual", is_key_clause=0),
    ])

    done = await dualwrite.vectorize_pending(client, collection=name)
    assert done == 2

    # 型番は dense には乗らない(全 chunk が同じベクトル)。当たるなら BM25 の text 経由
    hits = mc.bm25_search(client, "EC-RV300 の稼働時間", top_k=3, collection=name)
    assert [h["question"] for h in hits] == ["EC-RV300 の稼働時間は"]
    assert hits[0]["section_path"] == "商品仕様 / ロボット掃除機"
    assert hits[0]["content_type"] == "manual"
    assert hits[0]["category"] == "商品仕様"

    # category は絞り込みに使えるところまで書けていること
    filtered = mc.bm25_search(client, "EC-RV300 送料", top_k=5, category="送料", collection=name)
    assert filtered and all(h["category"] == "送料" for h in filtered)
    assert all(h["content_type"] == "policy" for h in filtered)


async def test_vectorize_text_covers_category_and_answer(db_session_factory, coll, monkeypatch):
    """BM25 の対象は questions だけではなく category + questions + answer の連結であること。

    answer にしか出てこない語で引けることで確かめる。text を questions だけにすると、
    回答本文の語で引けなくなり、ハイブリッドの BM25 側が質問文の言い換えにしか効かなくなる。
    """
    client, name = coll

    async def fake_embed(texts):
        return [[0.05] * mc.DIM for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", fake_embed)
    await dualwrite.write_pending([
        Chunk(category="返品", questions="返品したいのですが",
              answer="未開封であれば到着後7日以内に返品を受け付けます",
              section_path="返品ポリシー", content_type="policy", is_key_clause=1),
    ])
    await dualwrite.vectorize_pending(client, collection=name)

    assert [h["id"] for h in mc.bm25_search(client, "未開封", top_k=3, collection=name)]
    assert [h["id"] for h in mc.bm25_search(client, "返品", top_k=3, collection=name)]


async def test_vectorize_defaults_missing_metadata_to_empty_string(
    db_session_factory, coll, monkeypatch
):
    """section_path が NULL の chunk でも落ちないこと(採掘由来の chunk は持たないことがある)。

    Milvus の VARCHAR は None を受け取らないので、None のまま渡すと upsert が例外になる。
    """
    client, name = coll

    async def fake_embed(texts):
        return [[0.05] * mc.DIM for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", fake_embed)
    await dualwrite.write_pending([
        Chunk(category="その他", questions="営業時間は", answer="平日10時から18時です",
              section_path=None, content_type="mined", is_key_clause=0),
    ])
    assert await dualwrite.vectorize_pending(client, collection=name) == 1
    hits = mc.bm25_search(client, "営業時間", top_k=1, collection=name)
    assert hits and hits[0]["section_path"] == ""
