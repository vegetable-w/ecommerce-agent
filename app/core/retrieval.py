"""ナレッジ検索。dense / bm25 / hybrid / hybrid_rerank の 4 戦略を 1 つの入口で切り替える。

戦略を引数にしたのは、04 章の評価で同じクエリを 4 経路に流して比較するため。
既定は 03 章と同じ dense 単路のままにしてある。ハイブリッド + リランクへ切り替えるのは
query_faq 側の仕事(Task 10)で、そこから strategy="hybrid_rerank" を明示的に渡す。

**スコアの尺度は戦略ごとに別物である。** dense は COSINE の類似度、bm25 は BM25 の生スコア、
hybrid は RRF の融合値、hybrid_rerank は reranker の 0〜1 の関連度。したがって足切りの
既定値も戦略ごとに設定から取る(03 章の retrieval_min_score を rerank 側へ流用しない)。
"""

import logging

from app.config import settings
from app.core import embeddings, rerank
from app.kb import milvus_client

logger = logging.getLogger(__name__)

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
# 03 章と計画書は dense 単路を "vector" と呼んでいた。milvus_client の dense_search に
# 名前を揃えつつ、既存の呼び名も受け付ける。
_ALIASES = {"vector": "dense"}


def arrange_head_tail(items: list) -> list:
    """関連度の降順で並んだ列を、1 位を先頭・2 位を末尾・残りを順序どおり中央へ配置し直す。

    長い文脈では中央の情報が読み飛ばされやすい(lost in the middle)ため、上位 2 件を
    両端に置く。2 件以下では並べ替える意味がないのでそのまま返す。入力は破壊しない。
    """
    if len(items) <= 2:
        return list(items)
    return [items[0], *items[2:], items[1]]


def _cut(hits: list[dict], min_score: float | None, key: str = "score") -> list[dict]:
    if min_score is None:
        return hits
    return [h for h in hits if h[key] >= min_score]


async def search_knowledge(
    query: str,
    strategy: str = "dense",
    top_k: int | None = None,
    min_score: float | None = None,
    category: str | None = None,
    client=None,
    collection: str = milvus_client.COLLECTION,
) -> list[dict]:
    """最終順の hit を返す(hybrid_rerank は rerank_score を含む)。head/tail 配置は行わない。

    top_k / min_score を省略した場合の既定値は戦略ごとに異なる:
      dense  : retrieval_top_k / retrieval_min_score(COSINE の類似度)
      bm25   : retrieval_top_k / 足切りなし(BM25 の生スコアに絶対的な意味がないため)
      hybrid : rerank_top_k / 足切りなし(RRF の融合値も同じ理由)
      hybrid_rerank : rerank_top_k / rerank_min_score(reranker の 0〜1 の関連度)
    明示的に min_score を渡す場合は、その戦略の尺度に合わせた値を渡すこと。

    collection を引数で受け取るのは、Standalone が 1 サーバ共有でモジュール定数の
    差し替えでは隔離できないため。テストと評価は一時 collection の名前を明示的に渡す。
    """
    strategy = _ALIASES.get(strategy, strategy)
    if strategy not in STRATEGIES:
        raise ValueError(
            f"未知の検索戦略「{strategy}」。{' / '.join(STRATEGIES)} のいずれかを指定してください"
        )
    client = client or milvus_client.get_client()
    milvus_client.ensure_collection(client, collection=collection)

    if strategy == "dense":
        k = top_k or settings.retrieval_top_k
        cut = settings.retrieval_min_score if min_score is None else min_score
        vector = await embeddings.embed_query(query)
        hits = milvus_client.dense_search(client, vector, k, category, collection)
        return _cut(hits, cut)

    if strategy == "bm25":
        k = top_k or settings.retrieval_top_k
        # クエリは文字列のまま渡す(埋め込み上流を呼ばない経路)
        hits = milvus_client.bm25_search(client, query, k, category, collection)
        return _cut(hits, min_score)

    # hybrid / hybrid_rerank: まず recall_top_k 件まで広く拾う
    k = top_k or settings.rerank_top_k
    vector = await embeddings.embed_query(query)
    hits = milvus_client.hybrid_search(
        client, vector, query, settings.recall_top_k, settings.recall_top_k,
        category, collection)

    if strategy == "hybrid":
        return _cut(hits, min_score)[:k]

    if not hits:
        return []

    # hybrid_rerank: recall した候補を reranker で並べ替え、閾値未満は証拠として返さない
    cut = settings.rerank_min_score if min_score is None else min_score
    docs = [f"{h['question']} {h['answer']}" for h in hits]
    ranked = await rerank.rerank(query, docs, top_n=k)
    if not ranked:
        # rerank 上流が落ちた場合。問い合わせ全体を落とさず、融合検索の並びへ縮退する
        # (足切りもできないので、この経路では rerank_score を付けない)
        logger.warning("リランクできなかったためハイブリッド検索の並びで返す")
        return hits[:k]
    out = []
    for idx, score in ranked:
        if score < cut:
            continue
        h = dict(hits[idx])
        h["rerank_score"] = score
        out.append(h)
    return out
