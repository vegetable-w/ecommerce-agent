"""上流の /rerank へ直接接続する client(bge-reranker-v2-m3)。

/rerank は OpenAI protocol の機能ではなく Jina / Cohere 系の shape
(request: query + documents、response: results[].index / relevance_score)なので、
openai SDK ではなく HTTP request を手書きする。

**失敗しても例外を投げず [] を返す。** リランクは検索結果を並べ替える補助であり、
上流が 5xx を返したりタイムアウトしたりしただけで問い合わせ全体を落とすべきではない。
呼び出し側は [] を「リランクなし」と解釈して、ハイブリッド検索の並びへ縮退する
(app/core/retrieval.py)。
"""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def _url() -> str:
    """設定を読むのは呼び出し時。モジュール定数にすると import 順に依存する。"""
    return settings.rerank_base_url.rstrip("/") + "/rerank"


async def _post(url: str, json: dict, headers: dict, timeout: float) -> httpx.Response:
    async with httpx.AsyncClient() as c:
        return await c.post(url, json=json, headers=headers, timeout=timeout)


async def rerank(query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """[(元の index, relevance_score)] を score の降順で返し、top_n で truncate する。

    上流は実測では既に降順で返すが、その順序は契約ではないのでこちらで必ず並べ替える。
    docs の範囲外を指す index は捨てる(呼び出し側は hits[idx] で引き当てるため)。
    """
    if not docs:
        return []
    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": docs,
        "top_n": top_n or len(docs),
    }
    try:
        resp = await _post(
            _url(),
            payload,
            {"Authorization": f"Bearer {settings.rerank_key.get_secret_value()}"},
            settings.request_timeout,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        ranked = sorted(
            (
                (int(r["index"]), float(r["relevance_score"]))
                for r in results
                if 0 <= int(r["index"]) < len(docs)
            ),
            key=lambda x: -x[1],
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        # 上流の応答本文はログに出さない(鍵は載らないが、そもそも出す必要がない)
        logger.warning("rerank に失敗したためリランクなしで続行する: %s: %s",
                       type(exc).__name__, exc)
        return []
    return ranked[:top_n] if top_n else ranked
