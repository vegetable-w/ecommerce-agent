"""CLI(再実行可能な job): 過去の会話から QA を抽出 → staging → 重複排除 → knowledge_chunks(pending)。
そのあと scripts/vectorize_kb.py でベクトル化する。
実行: uv run --env-file .env python scripts/mine_knowledge.py"""
import asyncio

from app.kb import mining


async def main() -> None:
    stats = await mining.mine()
    print(f"ナレッジ抽出: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
