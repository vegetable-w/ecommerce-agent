"""埋め込みモデルを比較するための collection を作る。**既存の索引には触らない。**

`make kb-vectorize` は「pending の chunk を done にしながら 1 つの collection へ書く」
取り込みの経路なので、同じ chunk を別のモデルでもう一度埋め込むのには使えない
(2 度目は pending が空になっている)。ここは MySQL の chunk を**全件**読み、
いま設定されているモデルで埋め込んで、指定した名前の collection に入れるだけ。
MySQL 側の状態は 1 行も変えない。

埋め込む文字列は取り込みと同じ `category + questions + answer` の連結にする。
dense と BM25 が同じ文字列を見ないと、ハイブリッドが「2 つの別々のコーパス」を
引くことになって RRF の融合が壊れる(app/kb/dualwrite.py の同じ注記)。

使い方(モデルは環境変数で切り替える):

    EMBED_BASE_URL=http://127.0.0.1:8200/v1 EMBED_MODEL=cl-nagoya/ruri-v3-310m \\
    EMBED_DIM=768 EMBED_DOC_PREFIX='検索文書: ' EMBED_QUERY_PREFIX='検索クエリ: ' \\
    uv run --env-file .env python scripts/build_eval_collection.py knowledge_ruri
"""

import asyncio
import sys

from app.config import settings
from app.core import embeddings
from app.db import repository
from app.kb import milvus_client

BATCH = 32


async def main() -> int:
    if len(sys.argv) < 2:
        print("collection 名を渡してください(例: knowledge_ruri)")
        return 2
    name = sys.argv[1]
    if name == milvus_client.COLLECTION:
        print(f"「{name}」は本番の collection です。比較用には別の名前を使ってください。")
        return 2

    client = milvus_client.get_client()
    # 既にあれば作り直す。前のモデルのベクトルが混ざると比較にならない。
    if client.has_collection(name):
        print(f"既存の {name} を落として作り直します")
        client.drop_collection(name)
    milvus_client.ensure_collection(client, collection=name)

    chunks = await repository.list_all_chunks()
    print(f"chunk {len(chunks)} 件 / モデル {settings.embed_model} "
          f"({settings.embed_dim} 次元) → collection {name}")

    done = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await embeddings.embed_documents(texts)
        if len(vectors) != len(batch):
            print(f"埋め込みの件数が合いません(入力 {len(batch)} / 返却 {len(vectors)})")
            return 1
        if len(vectors[0]) != settings.embed_dim:
            print(f"次元が設定と違います(返却 {len(vectors[0])} / 設定 {settings.embed_dim})。"
                  f"EMBED_DIM を合わせてください。")
            return 1
        milvus_client.upsert_vectors(client, [
            {"id": r.id, "dense": v, "text": t,
             "question": r.questions, "answer": r.answer,
             "section_path": r.section_path or "", "content_type": r.content_type or "",
             "category": r.category or ""}
            for r, v, t in zip(batch, vectors, texts)
        ], collection=name)
        done += len(batch)
        print(f"  {done}/{len(chunks)}", flush=True)

    milvus_client.flush(client, collection=name)
    print(f"完了: {name} に {milvus_client.count(client, name)} 件")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
