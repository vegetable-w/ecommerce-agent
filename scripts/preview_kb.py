"""CLI: data/kb の資料を分割した結果だけを報告する(ドライラン)。

DB にも Milvus にも埋め込み API にも触れない。分割パラメータを変えたときに
「何 chunk になるのか」「重要条項をいくつ拾うのか」を、副作用なしで確かめるための入口。

build_kb.py に --dry-run を足す形にはしなかった。書き込むスクリプトにフラグで
書き込まないモードを足すと、フラグを落とした瞬間に本番へ書く。読むだけの経路は
書くコードを一行も持たない別ファイルにしておく方が、事故の目が減る。

実行: uv run --env-file .env python scripts/preview_kb.py
"""

import asyncio

from app.kb import chunking, documents, sources


async def main() -> None:
    total = keys = tables = 0
    for name, ctype in sources.SOURCE_TYPES.items():
        path = sources.source_path(name)
        if not path.is_file():
            print(f"  {name}: 資料が見つからない ({path})")
            continue
        chunks = documents.build_chunks(path.read_text(encoding="utf-8"), content_type=ctype)
        k = sum(c.is_key_clause for c in chunks)
        t = sum(1 for c in chunks if chunking.is_table_block(c.answer))
        longest = max((len(c.answer) for c in chunks), default=0)
        print(f"  {name} [{ctype}]: {len(chunks)} chunk / 重要条項 {k} / 表 {t} / 最長 {longest} 文字")
        total, keys, tables = total + len(chunks), keys + k, tables + t
    print(f"合計 {total} chunk(重要条項 {keys} 件、表ブロック {tables} 件)")
    print("ドライラン。DB にも Milvus にも書き込んでいない。登録するなら make kb-build")


if __name__ == "__main__":
    asyncio.run(main())
