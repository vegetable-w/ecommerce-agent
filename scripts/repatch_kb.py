"""CLI: data/kb/*.md の変更を、既存の chunk へ差分だけ当てて反映する。

kb-reset → kb-build → kb-vectorize の全量作り直しとの違いは 2 つ。

1. **変わっていない chunk を再度埋め込まない。** 埋め込みは課金される外部呼び出しな
   ので、句読点を 1 つ直しただけで 47 chunk 全部を焼き直すのは無駄が大きい。
2. **id が変わらない。** Milvus の PK は chunk id そのものなので、id を保ったまま
   upsert すれば古いベクトルがその場で置き換わる。作り直すと id が振り直され、
   低信頼プールや幻覚ケース台帳が持っている chunk id が指す先を失う。

突き合わせは (content_type, section_path) で行う。section_path は文書内での位置を
表す安定した名前で、本文を書き換えても変わらない。逆に見出しを変えた節は「古い節が
消えて新しい節が増えた」と見なされる — それは実際に別の節になったということなので、
埋め込み直すのが正しい。

同じ section_path が 1 文書に複数ある場合(大きな表が split_table_rows で割れると
起きる。実測: 「よくある問い合わせの対応時間」が 2 chunk)、出現順に 1 対 1 で
対応させる。

実行: uv run --env-file .env python scripts/repatch_kb.py
"""

import argparse
import asyncio
from collections import defaultdict

from app.db import repository
from app.kb import documents, dualwrite, milvus_client, sources

# 消える chunk がこの割合を超えたら止める。data/kb を取り違えたまま走らせると
# 「全部消して数件入れ直す」になり、差分適用のつもりで全量作り直しより悪い結果になる。
_MAX_DELETE_RATIO = 0.5


# **資料から来ていない chunk の section_path。** この突き合わせは
# 「knowledge_chunks にある行は全部 data/kb のどこかの節から来ている」という前提で
# 書かれていたが、09 章でフライホイールが、03 章で会話マイニングが、資料に無い
# ナレッジを直接書き戻すようになってその前提は崩れた。除外しないと、資料側に
# 対応する節が無い以上これらは必ず「削除」と判定される。しかも件数が少ないので
# _MAX_DELETE_RATIO の歯止めにも掛からず、data/kb の誤字を 1 つ直しただけで
# レビューを通ったナレッジが MySQL からも Milvus からも黙って消える。
_NON_DOCUMENT_SECTION_PATHS = frozenset({"flywheel", "mined"})


def _from_document(row) -> bool:
    """この行が data/kb の資料に由来するか。差分の対象はこれだけ。"""
    return (row.section_path or "") not in _NON_DOCUMENT_SECTION_PATHS


def _key(content_type: str | None, section_path: str | None) -> tuple[str, str]:
    return (content_type or "", section_path or "")


def _plan(existing: list, fresh: list[tuple[str, documents.Chunk]]) -> dict:
    """既存行と新しい chunk を突き合わせて、更新 / 追加 / 削除 / 据え置きに分ける。"""
    old_by_key: dict[tuple[str, str], list] = defaultdict(list)
    for row in existing:
        if not _from_document(row):
            continue          # 書き戻し由来は差分の外。据え置きにも削除にも入れない
        old_by_key[_key(row.content_type, row.section_path)].append(row)

    new_by_key: dict[tuple[str, str], list] = defaultdict(list)
    for category, c in fresh:
        new_by_key[_key(c.content_type, c.section_path)].append((category, c))

    same, changed, added = [], [], []
    for key, items in new_by_key.items():
        rows = old_by_key.get(key, [])
        for i, (category, c) in enumerate(items):
            if i < len(rows):
                row = rows[i]
                if row.questions == c.questions and row.answer == c.answer:
                    same.append(row)
                else:
                    changed.append((row, category, c))
            else:
                added.append((category, c))

    removed = []
    for key, rows in old_by_key.items():
        keep = len(new_by_key.get(key, []))
        removed.extend(rows[keep:])
    return {"same": same, "changed": changed, "added": added, "removed": removed}


def _exceeds_delete_guard(plan: dict, existing: list) -> bool:
    """歯止めの判定。**分母は資料由来の行だけ。**

    書き戻し由来を分母に混ぜると、資料由来が全滅していても総数で薄まって
    割合が小さく出る。歯止めが守りたいのは「data/kb の指し先を取り違えた」場合
    なので、見るべきは資料由来の行が何割消えるかになる。
    """
    n_doc = sum(1 for row in existing if _from_document(row))
    if not n_doc:
        return False
    return len(plan["removed"]) / n_doc > _MAX_DELETE_RATIO


async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="data/kb の変更を差分で反映する")
    p.add_argument("--dry-run", action="store_true",
                   help="差分を出すだけで、DB にも Milvus にも書かない")
    p.add_argument("--no-vectorize", action="store_true",
                   help="pending にするところまでで止める(あとで kb-vectorize)")
    args = p.parse_args(argv)

    fresh: list[tuple[str, documents.Chunk]] = []
    for name, ctype in sources.SOURCE_TYPES.items():
        md = sources.source_path(name).read_text(encoding="utf-8")
        for c in documents.build_chunks(md, content_type=ctype):
            fresh.append((c.category, c))

    existing = await repository.list_all_chunks()
    plan = _plan(existing, fresh)
    n_same, n_chg = len(plan["same"]), len(plan["changed"])
    n_add, n_del = len(plan["added"]), len(plan["removed"])

    print(f"資料 {len(sources.SOURCE_TYPES)} 件 / 新しい chunk {len(fresh)} 件 "
          f"/ 既存 {len(existing)} 件")
    print(f"  据え置き {n_same} / 変更 {n_chg} / 追加 {n_add} / 削除 {n_del}")
    for row, _, _ in plan["changed"]:
        print(f"  [変更] {row.section_path}")
    for _, c in plan["added"]:
        print(f"  [追加] {c.section_path}")
    for row in plan["removed"]:
        print(f"  [削除] {row.section_path}")

    if _exceeds_delete_guard(plan, existing):
        n_doc = sum(1 for row in existing if _from_document(row))
        print(f"\n中止: 資料由来の {n_del}/{n_doc} 件が消える判定になった。"
              "data/kb の指し先を確かめること。"
              "本当に作り直すなら kb-reset を明示的に使う")
        return 1

    if n_chg == n_add == n_del == 0:
        print("\n差分なし。何もしない")
        return 0
    if args.dry_run:
        print("\n--dry-run のため書き込みません")
        return 0

    for row, category, c in plan["changed"]:
        await repository.update_knowledge_chunk(
            row.id, category, c.questions, c.answer,
            c.section_path, c.content_type, c.is_key_clause)
    for category, c in plan["added"]:
        await repository.insert_knowledge_chunk(
            category, c.questions, c.answer,
            section_path=c.section_path, content_type=c.content_type,
            is_key_clause=c.is_key_clause)

    client = milvus_client.get_client()
    if plan["removed"]:
        ids = [row.id for row in plan["removed"]]
        for row in plan["removed"]:
            await repository.delete_knowledge_chunk(row.id)
        # MySQL を先に消す。Milvus だけ残った行は次の repatch で拾い直せるが、
        # 逆(索引だけ消えて本文が残る)は status=done のまま検索に出てこなくなる
        n = milvus_client.delete_vectors(client, ids)
        print(f"\nMilvus から {n} 件削除")

    # 前後リンクは文書単位で張り直す。1 件ずつ繋ぐと、消した行を指したままの
    # 中間状態が残る
    after = await repository.list_all_chunks()
    by_doc: dict[str, list] = defaultdict(list)
    for row in after:
        by_doc[row.content_type or ""].append(row)
    for rows in by_doc.values():
        for i, row in enumerate(rows):
            await repository.set_chunk_neighbors(
                row.id,
                rows[i - 1].id if i > 0 else None,
                rows[i + 1].id if i < len(rows) - 1 else None)

    pending = await repository.count_chunks_by_status("pending")
    print(f"pending {pending} 件")
    if args.no_vectorize:
        print("--no-vectorize のためここで止めます。次: scripts/vectorize_kb.py")
        return 0

    done = await dualwrite.vectorize_pending(client)
    print(f"ベクトル化 {done} 件 / Milvus 合計 {milvus_client.count(client)} 件")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
