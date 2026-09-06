"""CLI: knowledge_chunks の pending をベクトル化して Milvus へ書き込む(冪等・再実行可能)。
実行: uv run --env-file .env python scripts/vectorize_kb.py"""
import asyncio

from app.kb import dualwrite, milvus_client


async def main() -> None:
    client = milvus_client.get_client()
    milvus_client.ensure_collection(client)
    n = await dualwrite.vectorize_pending(client)
    print(f"ベクトル化 {n} 件。Milvus 現在 {milvus_client.count(client)} 件")


if __name__ == "__main__":
    asyncio.run(main())
