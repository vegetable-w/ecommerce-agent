"""CLI: data/kb/*.md を chunk 化して knowledge_chunks へ pending として書き込む。

そのあと scripts/vectorize_kb.py でベクトル化する。
実行: uv run --env-file .env python scripts/build_kb.py
"""

import asyncio

from app.kb import documents, dualwrite, sources


async def main() -> None:
    total = 0
    for name, ctype in sources.SOURCE_TYPES.items():
        md = sources.source_path(name).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        ids = await dualwrite.write_pending(chunks)
        keys = sum(c.is_key_clause for c in chunks)
        total += len(ids)
        print(f"  {name}: {len(ids)} chunk (重要条項 {keys} 件)")
    print(f"合計 {total} chunk を pending として登録。次: scripts/vectorize_kb.py")


if __name__ == "__main__":
    asyncio.run(main())
