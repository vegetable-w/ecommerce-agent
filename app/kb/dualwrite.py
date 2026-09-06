"""MySQL(正)と Milvus(索引)への二重書き込み。

書き込みを 2 段階に分ける: write_pending が MySQL へ status=pending で入れ、
vectorize_pending が pending だけを拾ってベクトル化する。埋め込み上流が落ちても
MySQL の本文は残り、再実行で残りだけを処理できる。
"""

from app.core import embeddings
from app.db import repository
from app.kb import milvus_client
from app.kb.documents import Chunk


async def write_pending(chunks: list[Chunk]) -> list[int]:
    """1つの文書の chunks を順に MySQL へ insert(status=pending)し、同一文書内で prev/next を張る。"""
    ids: list[int] = []
    for c in chunks:
        cid = await repository.insert_knowledge_chunk(
            c.category, c.questions, c.answer,
            section_path=c.section_path, content_type=c.content_type,
            is_key_clause=c.is_key_clause,
        )
        ids.append(cid)
    for i, cid in enumerate(ids):
        prev_id = ids[i - 1] if i > 0 else None
        next_id = ids[i + 1] if i < len(ids) - 1 else None
        await repository.set_chunk_neighbors(cid, prev_id, next_id)
    return ids


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


async def vectorize_pending(client, batch_size: int = 64) -> int:
    """冪等・再実行可能: pending を取得 → category+questions+answer を連結して埋め込み
    → Milvus upsert(PK=id) → vector_id 反映・status=done。
    どのバッチで落ちても、再実行時は残りの pending だけを拾う(id 単位 upsert なので重複しない)。"""
    pending = await repository.list_pending_chunks()
    done = 0
    for batch in _batches(pending, batch_size):
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await embeddings.embed_texts(texts)
        rows = [
            {"id": r.id, "vector": v, "question": r.questions, "answer": r.answer}
            for r, v in zip(batch, vectors)
        ]
        milvus_client.upsert_vectors(client, rows)
        for r in batch:
            await repository.mark_chunk_vectorized(r.id, str(r.id))
        done += len(batch)
    return done
