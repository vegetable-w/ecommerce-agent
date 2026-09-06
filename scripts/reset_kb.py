"""CLI: ナレッジベースを初期状態へ戻す(破壊的)。

MySQL の knowledge_chunks / qa_extraction_staging を空にし、Milvus の collection を
削除する。**両方**やらないと意味がない: MySQL だけ消すと Milvus に孤児のベクトルが残り、
検索が「MySQL に無い chunk」を返し続ける。Milvus だけ消すと MySQL 側が status=done の
ままなので、kb-vectorize が pending を 0 件と判断して二度と作り直さない。

collection は truncate ではなく drop する。次回の ensure_collection が schema から
作り直すので、DIM や index の設定を変えた場合もここで揃う。

実行: uv run --env-file .env python scripts/reset_kb.py
"""

import asyncio

from app.db import repository
from app.kb import milvus_client


async def main() -> None:
    await repository.clear_knowledge()
    print("MySQL: knowledge_chunks と qa_extraction_staging を空にした")
    client = milvus_client.get_client()
    milvus_client.drop_collection(client)
    print(f"Milvus: collection {milvus_client.COLLECTION} を削除した")
    print("初期状態に戻した。再構築は make kb-build のあと make kb-vectorize")


if __name__ == "__main__":
    asyncio.run(main())
