"""評価セット tests/data/eval_04.jsonl を、ナレッジベースの実物と突き合わせて検証する。

評価セットはコードではなくデータなので、壊れていても誰も落ちない。期待値がどの節にも
当たらなければ Recall は永久に 0、原文に無い key point は永久に cover されない。
どちらも「検索が悪い」ように見えるだけで、実際に悪いのは出題の方という事故になる。
それを機械的に洗い出すのがこのスクリプト。

検査するのは 5 点:

  1. id と質問文が重複していないこと
  2. bucket の集合が想定どおりで、各 bucket の問題数が揃っていること
  3. 期待した節(グループ)が、ナレッジ内の少なくとも 1 つの section に当たること
  4. expect_points が対象 section の本文に実在すること(空白を無視した部分一致)
  5. D_absent が正解を持たず、should_refuse が true であること

正となるナレッジは MySQL の knowledge_chunks。Milvus は読まない(Milvus が落ちていても
評価セットの検査はできるべきなので)。**読み取り専用**で、1 行も書き込まない。

節への当たり判定は scripts/eval_04.py の関数をそのまま使う。採点側と検証側で判定が
ずれると、「検証は通るのに Recall が 0」という最悪の食い違いが起きるため。

実行:
    PYTHONUTF8=1 uv run --env-file .env python scripts/validate_eval_04.py
    make eval-check
"""

import argparse
import asyncio
import collections
import pathlib
import sys

from app.db import repository
from scripts import eval_04 as ev


def build_sections(rows: list[tuple[str, str]]) -> dict[str, str]:
    """(section_path, answer) の一覧を section_path 単位の本文へまとめる。

    大きな表は split_table_rows で複数 chunk に割れ、同じ section_path を共有する
    (実測: after-sales-manual.md の「よくある問い合わせの対応時間」)。chunk 単位で
    照合すると、2 枚目にしか無い key point を「存在しない」と誤判定する。
    """
    out: dict[str, list[str]] = collections.defaultdict(list)
    for path, answer in rows:
        if path:
            out[path].append(answer)
    return {path: "\n".join(bodies) for path, bodies in out.items()}


def matching_sections(sections: dict[str, str], group: list[str]) -> list[str]:
    """グループ(AND 条件)に当たる section_path。採点側の is_relevant と同じ判定。"""
    return [path for path in sections
            if ev.is_relevant({"section_path": path}, group)]


# ---------------------------------------------------------------------------
# 個々の検査。どれも「エラー文のリスト」を返す(空なら合格)。
# ---------------------------------------------------------------------------


def check_no_duplicates(samples: list[dict]) -> list[str]:
    """id と質問文の重複。同じ質問を並べても問題数が増えるだけで難易度は測れない。"""
    errors = []
    seen_id: dict[str, str] = {}
    seen_query: dict[str, str] = {}
    for s in samples:
        sid = s["id"]
        if sid in seen_id:
            errors.append(f"{sid}: id が重複している")
        else:
            seen_id[sid] = sid
        q = ev.norm(s["query"])
        if q in seen_query:
            errors.append(f"{sid}: 質問が {seen_query[q]} と重複している({s['query']})")
        else:
            seen_query[q] = sid
    return errors


def check_bucket_counts(samples: list[dict], buckets: list[str] = ev.BUCKETS) -> list[str]:
    """bucket の集合と、bucket ごとの問題数の揃い。

    1 bucket あたりの目標値はここに書かない。「各 60 問」のような数を焼き付けると
    評価セットを増やすたびにこのスクリプトを直すことになる。見るのは
    「想定した bucket が過不足なくあること」と「どの bucket も同じ問題数であること」。
    """
    errors = []
    counts = collections.Counter(s["bucket"] for s in samples)
    for b in sorted(set(buckets) - set(counts)):
        errors.append(f"bucket {b} の問題が 1 問も無い(想定は {len(buckets)} bucket)")
    for b in sorted(set(counts) - set(buckets)):
        errors.append(f"未知の bucket {b} が {counts[b]} 問ある(想定は {buckets})")
    sizes = {b: counts.get(b, 0) for b in buckets}
    if len(set(sizes.values())) > 1:
        detail = ", ".join(f"{b}={n}" for b, n in sizes.items())
        errors.append(f"bucket ごとの問題数が揃っていない: {detail}")
    return errors


def check_sections_exist(samples: list[dict], sections: dict[str, str]) -> list[str]:
    """期待した節が実在すること。どこにも当たらない期待値は出題ミス。"""
    errors = []
    for s in samples:
        for i, group in enumerate(ev.expect_groups(s), 1):
            if not matching_sections(sections, group):
                errors.append(
                    f"{s['id']}: 期待した節(グループ {i}){group} に当たる section がナレッジに無い")
    return errors


def check_points_exist(samples: list[dict], sections: dict[str, str]) -> list[str]:
    """key point が対象 section の本文に実在すること。

    存在しない point は evidence coverage で永久に 0 になり、点数を静かに押し下げる。
    複数根拠の問いでは、どのグループの本文に載っていてもよい(回答は全根拠をまとめて
    書くものなので、point とグループの対応までは求めない)。
    """
    errors = []
    for s in samples:
        groups = ev.expect_groups(s)
        if not groups:
            continue
        paths = {p for g in groups for p in matching_sections(sections, g)}
        body = ev.norm("\n".join(sections[p] for p in sorted(paths)))
        for point in s["expect_points"]:
            if ev.norm(point) not in body:
                errors.append(f"{s['id']}: key point {point!r} が対象 section の本文に無い")
    return errors


def check_expectations_match_the_bucket(samples: list[dict]) -> list[str]:
    """D_absent は正解を持たず断る側、それ以外は正解を持つ側であること。"""
    errors = []
    for s in samples:
        sid = s["id"]
        has_answer = bool(s.get("expect_section") or s.get("expect_sections_all"))
        if s["bucket"] == "D_absent":
            if has_answer:
                errors.append(f"{sid}: D_absent なのに expect_section / expect_sections_all がある")
            if s.get("expect_points"):
                errors.append(f"{sid}: D_absent なのに expect_points がある")
            if s.get("should_refuse") is not True:
                errors.append(f"{sid}: D_absent なのに should_refuse が true でない")
        else:
            if not has_answer:
                errors.append(f"{sid}: 採点対象の bucket なのに正解の節が指定されていない")
            if not s.get("expect_points"):
                errors.append(f"{sid}: 採点対象の bucket なのに expect_points が空")
            if s.get("should_refuse") is not False:
                errors.append(f"{sid}: 採点対象の bucket なのに should_refuse が false でない")
    return errors


def validate(samples: list[dict], sections: dict[str, str]) -> list[str]:
    """5 つの検査をまとめて回す。最初のエラーで止めない(1 度に全部直せるように)。"""
    return [*check_no_duplicates(samples),
            *check_bucket_counts(samples),
            *check_expectations_match_the_bucket(samples),
            *check_sections_exist(samples, sections),
            *check_points_exist(samples, sections)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="04 章の評価セットを検証する(読み取り専用)")
    p.add_argument("--file", default=str(ev.EVAL_SET), help="検証する評価セット(JSONL)")
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")   # cp932 の console でも落とさない
        except (AttributeError, ValueError):
            pass

    samples = ev.load_samples(pathlib.Path(args.file))
    rows = await repository.list_chunk_sections()
    if not rows:
        raise SystemExit(
            "MySQL の knowledge_chunks が空です。"
            "先に make kb-build でナレッジを取り込んでから検証してください。")
    sections = build_sections(rows)

    print(f"評価セット {args.file} を検証: {len(samples)} 問 / "
          f"ナレッジ {len(sections)} 節({len(rows)} chunk)")
    errors = validate(samples, sections)
    for e in errors:
        print(f"  - {e}")
    print(f"\n{len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
