import uuid

import pytest

from app.db import repository
from app.kb import dualwrite, milvus_client
from app.kb.documents import Chunk

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _chunk(q, a):
    return Chunk(category="c", questions=q, answer=a, section_path="c / " + q, content_type="policy")


@pytest.fixture()
def milvus(monkeypatch):
    """固有名の collection へ差し替える。dualwrite は milvus_client を引数なしで呼ぶので、
    定数を差し替えないと本番の knowledge collection に書いてしまう。"""
    monkeypatch.setattr(milvus_client, "COLLECTION", "test_" + uuid.uuid4().hex[:12])
    assert milvus_client.COLLECTION.startswith("test_")
    c = milvus_client.get_client()
    milvus_client.ensure_collection(c)
    yield c
    milvus_client.drop_collection(c)


async def test_write_pending_links_neighbors(db_session_factory):
    ids = await dualwrite.write_pending([_chunk("q1", "a1"), _chunk("q2", "a2"), _chunk("q3", "a3")])
    assert len(ids) == 3
    by_id = {c.id: c for c in await repository.list_pending_chunks()}
    assert by_id[ids[1]].prev_chunk_id == ids[0]
    assert by_id[ids[1]].next_chunk_id == ids[2]
    assert by_id[ids[0]].prev_chunk_id is None


async def test_vectorize_resumes_after_crash(db_session_factory, milvus, monkeypatch):
    await dualwrite.write_pending([_chunk(f"q{i}", f"a{i}") for i in range(4)])

    calls = {"n": 0}

    async def flaky_embed(texts):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("埋め込みサービス中断")
        return [[float(i), 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3) for i, _ in enumerate(texts)]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", flaky_embed)
    with pytest.raises(RuntimeError):
        await dualwrite.vectorize_pending(milvus, batch_size=2)
    assert await repository.count_chunks_by_status("done") == 2
    assert await repository.count_chunks_by_status("pending") == 2

    async def ok_embed(texts):
        return [[9.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3) for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", ok_embed)
    done_now = await dualwrite.vectorize_pending(milvus, batch_size=2)
    assert done_now == 2
    assert await repository.count_chunks_by_status("pending") == 0
    assert await repository.count_chunks_by_status("done") == 4
    assert milvus_client.count(milvus) == 4  # 重複も欠落もない
