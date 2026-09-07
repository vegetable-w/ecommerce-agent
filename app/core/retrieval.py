from app.config import settings
from app.core import embeddings
from app.kb import milvus_client


async def search_knowledge(
    query: str, top_k: int | None = None,
    min_score: float | None = None, client=None,
    collection: str = milvus_client.COLLECTION,
) -> list[dict]:
    """dense 単路の意味検索。score は COSINE の類似度(大きいほど近い)。

    collection を引数で受け取るのは、Standalone が 1 サーバ共有でモジュール定数の
    差し替えでは隔離できないため。テストと評価は一時 collection の名前を明示的に渡す。
    """
    top_k = top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if min_score is None else min_score
    vector = await embeddings.embed_query(query)
    client = client or milvus_client.get_client()
    milvus_client.ensure_collection(client, collection=collection)
    hits = milvus_client.dense_search(client, vector, top_k, collection=collection)
    return [h for h in hits if h["score"] >= min_score]
