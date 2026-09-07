import uuid

import pytest

from app.config import settings
from app.db import repository
from app.kb import dualwrite, milvus_client
from app.kb.documents import Chunk

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _chunk(q, a):
    return Chunk(category="c", questions=q, answer=a, section_path="c / " + q, content_type="policy")


@pytest.fixture()
def milvus():
    """固有名の一時 collection を作り、(client, name) を返す。

    04 章で vectorize_pending が collection を引数に取るようになったので、定数の
    差し替えではなく名前を渡して隔離する。standalone は 1 サーバ共有で、定数の
    差し替えは渡し忘れた経路から本番の knowledge に書いてしまう。"""
    name = "test_" + uuid.uuid4().hex[:12]
    assert name.startswith("test_")
    c = milvus_client.get_client(uri=settings.milvus_uri)
    milvus_client.ensure_collection(c, collection=name)
    yield c, name
    milvus_client.drop(c, name)


async def test_write_pending_links_neighbors(db_session_factory):
    ids = await dualwrite.write_pending([_chunk("q1", "a1"), _chunk("q2", "a2"), _chunk("q3", "a3")])
    assert len(ids) == 3
    by_id = {c.id: c for c in await repository.list_pending_chunks()}
    assert by_id[ids[1]].prev_chunk_id == ids[0]
    assert by_id[ids[1]].next_chunk_id == ids[2]
    assert by_id[ids[0]].prev_chunk_id is None


async def test_vectorize_resumes_after_crash(db_session_factory, milvus, monkeypatch):
    client, coll = milvus
    await dualwrite.write_pending([_chunk(f"q{i}", f"a{i}") for i in range(4)])

    calls = {"n": 0}

    async def flaky_embed(texts):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("埋め込みサービス中断")
        return [[float(i), 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3) for i, _ in enumerate(texts)]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", flaky_embed)
    with pytest.raises(RuntimeError):
        await dualwrite.vectorize_pending(client, batch_size=2, collection=coll)
    assert await repository.count_chunks_by_status("done") == 2
    assert await repository.count_chunks_by_status("pending") == 2

    async def ok_embed(texts):
        return [[9.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3) for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", ok_embed)
    done_now = await dualwrite.vectorize_pending(client, batch_size=2, collection=coll)
    assert done_now == 2
    assert await repository.count_chunks_by_status("pending") == 0
    assert await repository.count_chunks_by_status("done") == 4
    assert milvus_client.count(client, collection=coll) == 4  # 重複も欠落もない


async def test_short_embedding_response_leaves_batch_recoverable(
    db_session_factory, milvus, monkeypatch
):
    """上流が入力より少ないベクトルを返しても、chunk を失わないこと。

    zip は黙って短い方へ切り詰めるので、番人が無いとベクトルを受け取れなかった chunk が
    status=done かつ実体の無い vector_id を持ち、再実行でも pending として拾われず、
    検索から永久に消える。ここでは全件 pending のまま残ることを固定する。
    """
    client, coll = milvus
    await dualwrite.write_pending([_chunk(f"q{i}", f"a{i}") for i in range(3)])

    async def short_embed(texts):
        # 3 件要求されたのに 2 件しか返さない上流
        return [[1.0, 0.0, 0.0] + [0.0] * (milvus_client.DIM - 3) for _ in texts[:-1]]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", short_embed)
    with pytest.raises(RuntimeError, match="件数"):
        await dualwrite.vectorize_pending(client, batch_size=3, collection=coll)

    # 部分的に done 化されていない = 再実行で全件回収できる
    assert await repository.count_chunks_by_status("done") == 0
    assert await repository.count_chunks_by_status("pending") == 3
    assert milvus_client.count(client, collection=coll) == 0

    async def ok_embed(texts):
        return [[1.0, 0.0, 0.0] + [0.0] * (milvus_client.DIM - 3) for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", ok_embed)
    assert await dualwrite.vectorize_pending(client, batch_size=3, collection=coll) == 3
    assert await repository.count_chunks_by_status("pending") == 0
    assert milvus_client.count(client, collection=coll) == 3
