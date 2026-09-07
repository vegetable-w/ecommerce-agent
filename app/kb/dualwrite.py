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


async def vectorize_pending(client, batch_size: int = 64,
                            collection: str = milvus_client.COLLECTION) -> int:
    """冪等・再実行可能: pending を取得 → category+questions+answer を連結して埋め込み
    → Milvus upsert(PK=id) → vector_id 反映・status=done。

    連結した text は dense の埋め込み元であると同時に、BM25 の検索対象でもある。
    両経路が同じ文字列を見ることで、同じ chunk 群を recall できる(片方だけ別の文字列に
    すると、ハイブリッドが「2 つの別々のコーパス」を引くことになり RRF の融合が壊れる)。
    どのバッチで落ちても、再実行時は残りの pending だけを拾う(id 単位 upsert なので重複しない)。"""
    pending = await repository.list_pending_chunks()
    done = 0
    for batch in _batches(pending, batch_size):
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await embeddings.embed_texts(texts)
        # 件数が合わなければ upsert も done 化もせずに落とす。
        # zip は黙って短い方に切り詰めるため、この番人が無いと「ベクトルを受け取れなかった
        # chunk が status=done かつ実体の無い vector_id を持つ」状態になり、再実行しても
        # pending として拾われず、検索から永久に消える(実測で確認した穴)。
        # ここで例外にすればバッチ全体が pending のまま残り、再実行で回収できる。
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"埋め込みの件数が入力と一致しない(入力 {len(batch)} 件 / 返却 {len(vectors)} 件)。"
                "このバッチは pending のまま残すので、再実行で補完できる。"
            )
        rows = [
            {"id": r.id, "dense": v, "text": t,
             "question": r.questions, "answer": r.answer,
             "section_path": r.section_path or "", "content_type": r.content_type or "",
             "category": r.category or ""}
            for r, v, t in zip(batch, vectors, texts)
        ]
        milvus_client.upsert_vectors(client, rows, collection=collection)
        for r in batch:
            await repository.mark_chunk_vectorized(r.id, str(r.id))
        done += len(batch)
    if done:
        # 取り込みの区切りで growing segment を封じる。検索可能にするためではない
        # (collection は Strong なので upsert 直後から当たる)。
        milvus_client.flush(client, collection=collection)
    return done
