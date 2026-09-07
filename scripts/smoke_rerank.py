"""スモーク: 上流の /rerank へ直接接続する。04 章の red line。

/rerank は OpenAI protocol ではなく Jina / Cohere 系の shape
(request: query + documents、response: results[].index / relevance_score)なので、
openai client は使わず HTTP request を手書きする。
実行: uv run --env-file .env python scripts/smoke_rerank.py
"""

import httpx

from app.config import settings

QUERY = "返品するとき送料は誰が払いますか"
# 3 件目が正解。1 件目は「送料」という語を共有するだけの別条項で、
# キーワードの重なりだけで並べると 1 件目が上に来る。rerank が意味で並べ替えられるかを見る。
DOCUMENTS = [
    "1回の注文金額が3,000円以上の場合は送料無料。3,000円未満の場合は500円の送料がかかる。",
    "ロボット掃除機 Max（型番 EC-RV300）の稼働時間は約210分、吸引力は8,000Pa。",
    "商品不良ではないお客様都合の返品については、返送料は購入者が負担する。",
]
EXPECT_INDEX = 2


def main() -> None:
    url = settings.rerank_base_url.rstrip("/") + "/rerank"
    payload = {
        "model": settings.rerank_model,
        "query": QUERY,
        "documents": DOCUMENTS,
        "top_n": len(DOCUMENTS),
    }
    r = httpx.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {settings.rerank_key.get_secret_value()}"},
        timeout=settings.request_timeout,
    )
    r.raise_for_status()
    results = r.json()["results"]
    assert results, "rerank が空の results を返した"

    ordered = sorted(results, key=lambda x: -x["relevance_score"])
    print(f"model={settings.rerank_model}  query={QUERY!r}")
    for rank, item in enumerate(ordered, 1):
        i = item["index"]
        print(f"  {rank}. [{item['relevance_score']:.4f}] ({i}) {DOCUMENTS[i][:44]}")

    top = ordered[0]["index"]
    assert top == EXPECT_INDEX, (
        f"返品送料の質問は {EXPECT_INDEX} 番目(購入者負担)が 1 位になるべき。実際は {top} 番目"
    )
    print(f"GO: /rerank 経路は動作し、意味で並べ替えられている")


if __name__ == "__main__":
    main()
