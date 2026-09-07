"""judge の回帰チェック: 幻覚ケース台帳の人手ラベルを正解として judge を測り直す。

make eval-rag の judge は「この回答は与えた根拠に忠実か」を判定し、忠実でないと判定した
問いを faith_cases(幻覚ケース台帳)へ積む。人はそれを 1 件ずつ見て「本当に幻覚だった」
「judge の行き過ぎで直す必要はない」と印を付ける。その印を**正解ラベル**として、judge 自身の
判定基準が甘すぎないか / 厳しすぎないかを測るのがこのスクリプト。

  resolved         人が「本当に幻覚だった」と認めた  -> judge は faithful=false と言うべき
  no_action_needed 人が「幻覚ではない」と判断した    -> judge は faithful=true  と言うべき
  unresolved       まだ誰も見ていない                -> 正解ラベルが無いので測れない(skip)

**検索も生成もやり直さない。** 台帳に残っている「そのとき judge へ渡した根拠の全件
スナップショット(citations)」と「そのとき生成された回答」をそのまま再生し、judge だけを
差し替えて同じ入力に何と答えるかを見る。だから検索や生成が変わっても結果がぶれず、
judge の変更だけを測れる。

そのため evidence の組み立ては scripts/eval_04.py の build_evidence をそのまま使う。
番号付けをここに書き直すと [n] の振り方が 2 か所になり、片方だけ直したときに
「judge の変化」を測っているつもりで「入力の違い」を測ることになる。しかも両方それらしい
文字列を出すので誰も気づけない。

台帳(MySQL)は**読むだけ**で 1 行も書き込まない。

実行:
    PYTHONUTF8=1 uv run --env-file .env python scripts/judge_check.py
    PYTHONUTF8=1 uv run --env-file .env python scripts/judge_check.py --dry-run  # 上流を呼ばない
    make judge-check
"""

import argparse
import asyncio
import collections
import sys
import unicodedata

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT
from app.db import repository
from scripts import eval_04 as ev

# 台帳を読むときの 1 ページ。全件でも評価セットの問題数(300)程度なので数回で読み切る
PAGE_SIZE = 100
# judge を並べる数。eval_04 の生成段と同じ値にして、上流の詰まり方を揃える
CONCURRENCY = 3
# 上流が空を返したときの試行回数(1 回目 + retry 1 回)
ATTEMPTS = 2

# 人手の印 -> 「judge はこう言うべき」という正解ラベル。
# unresolved はここに無い。まだ誰も見ていないので正解が存在せず、測る対象にならない。
EXPECTED_FAITHFUL = {"resolved": False, "no_action_needed": True}

# skip の理由。件数だけでなく理由も出す(黙って母数が減ると「一致率 100%」が
# 「1 件も測れなかった」の別名になっていても気づけない)
SKIP_LABELS = {
    "unresolved": "未対処(人手の正解ラベルがまだ無い)",
    "no_citations": "根拠スナップショットが無い(同じ入力を再生できない)",
}


# ---------------------------------------------------------------------------
# 入力の再生。ここが「judge の変更だけを測る」ことの土台になる
# ---------------------------------------------------------------------------


def expected_faithful(status: str | None) -> bool | None:
    """人手の印から正解ラベルへ。測れない状態は None。"""
    return EXPECTED_FAITHFUL.get(status or "")


def _n_of(citation: dict, index: int) -> int:
    """スナップショット 1 件の番号。n が無い古い行はリスト順で補完する。

    n を書き始めたのは Task 19 の途中からで、それ以前に積まれた行には無い。
    ここで落とすと過去のケースが丸ごと測れなくなるので、当時の並び順を番号とみなす。
    """
    n = citation.get("n")
    return n if isinstance(n, int) else index + 1


def restore_hits(citations: list[dict]) -> list[dict]:
    """台帳のスナップショットを、build_evidence へ渡せる検索ヒットの形へ戻す。

    番号は付け直さない。n の順に並べ替えて build_evidence(= build_citations)へ渡し、
    番号付けは向こうに任せる。ここで [n] を組み立て直すと番号付けのコードが 2 か所になる。
    """
    ordered = sorted(enumerate(citations), key=lambda pair: _n_of(pair[1], pair[0]))
    return [{"id": c.get("chunk_id"), "section_path": c.get("section_path"),
             "question": c.get("question"), "answer": c.get("answer")}
            for _, c in ordered]


def rebuild_evidence(citations: list[dict]) -> str:
    """そのとき judge へ渡した evidence を byte 単位で復元する。"""
    return ev.build_evidence(restore_hits(citations))


def select_cases(rows: list) -> tuple[list[dict], dict[str, int]]:
    """台帳の行から測れるケースだけを取り出す。落とした行は理由ごとに数えて返す。"""
    targets: list[dict] = []
    skipped = {key: 0 for key in SKIP_LABELS}
    for row in rows:
        expected = expected_faithful(row.status)
        if expected is None:
            skipped["unresolved"] += 1
            continue
        if not isinstance(row.citations, list) or not row.citations:
            skipped["no_citations"] += 1
            continue
        targets.append({
            "eval_id": row.eval_id, "bucket": row.bucket, "status": row.status,
            "expected": expected, "query": row.query, "answer": row.answer,
            "evidence": rebuild_evidence(row.citations), "judge_model": row.judge_model,
        })
    return targets, skipped


async def fetch_rows(page_size: int = PAGE_SIZE) -> list:
    """台帳の全行。読み取り専用。

    status で絞り込まずに読むのは、skip した件数と理由も出すため。DB 側で絞ると
    「何件を測らなかったのか」が最初から見えなくなる。
    """
    rows: list = []
    page = 1
    while True:
        res = await repository.list_faith_cases(page=page, size=page_size)
        rows.extend(res["rows"])
        if page >= res["pages"]:
            return rows
        page += 1


# ---------------------------------------------------------------------------
# judge の再実行
# ---------------------------------------------------------------------------


def build_chain():
    """judge は temperature=0。同じ入力に同じ判定を返してほしいので揺れは害にしかならない。"""
    model = get_chat_model(temperature=0)
    return FAITHFULNESS_PROMPT | model.with_structured_output(ev._Faithful)


async def judge_case(chain, case: dict) -> dict:
    """1 件ぶん、同じ入力を judge へ渡し直す。

    構造化出力はときどき None になる(parse に失敗した回)。その場合は 1 回だけやり直し、
    それでも駄目なら**呼び出し失敗**として記録する。失敗を不一致に数えると、上流の
    調子で数字が動いてしまい、judge の変更を測るという目的そのものが崩れる。
    """
    actual: bool | None = None
    reason = ""
    error: str | None = None
    for _ in range(ATTEMPTS):
        try:
            r = await asyncio.wait_for(
                chain.ainvoke({"evidence": case["evidence"], "answer": case["answer"]}),
                ev.CALL_TIMEOUT)
        except Exception as exc:
            r, error = None, f"{type(exc).__name__}: {exc}"
        else:
            error = None if r is not None else "judge が構造化出力を返しませんでした"
        if r is not None:
            actual = bool(r.faithful)
            reason = (r.reason or "").strip() or ev.NO_REASON
            break
    return {**case, "actual": actual, "reason": reason, "error": error,
            "agree": None if actual is None else actual == case["expected"]}


def summarize(results: list[dict]) -> dict:
    """一致率。**呼び出しに失敗したケースは母数から外す。**

    失敗を不一致として数えると、judge を 1 文字も変えていない実行同士で数字が食い違う。
    測りたいのは judge の判定基準であって上流の調子ではないので、失敗は母数の外に置き、
    件数だけを別に出す。
    """
    judged = [r for r in results if r["actual"] is not None]
    agreed = [r for r in judged if r["agree"]]
    return {
        "total": len(results),
        "judged": len(judged),
        "agreed": len(agreed),
        "mismatched": len(judged) - len(agreed),
        "failed": len(results) - len(judged),
        # 1 件も判定できなかった実行は 0% ではない(「測っていない」を None で表す)
        "rate": len(agreed) / len(judged) if judged else None,
    }


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------


def _width(s: str) -> int:
    """全角を 2 桁として数えた表示幅。ljust は文字数で詰めるので日本語の列がずれる。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _width(s))


def _clip(s: str, width: int) -> str:
    """表示幅で切り詰める。理由の全文は不一致の詳細の方に出す。"""
    out = ""
    for c in s.replace("\n", " "):
        if _width(out) + _width(c) > width:
            return out + "…"
        out += c
    return out


def _label(faithful: bool | None) -> str:
    if faithful is None:
        return "呼び出し失敗"
    return "忠実" if faithful else "幻覚"


def print_report(results: list[dict], summary: dict) -> None:
    print("\n  " + _pad("eval_id", 10) + _pad("bucket", 15) + _pad("期待", 8)
          + _pad("judge", 14) + _pad("一致", 6) + "judge の理由")
    for r in results:
        mark = "-" if r["agree"] is None else ("○" if r["agree"] else "×")
        note = r["reason"] if r["actual"] is not None else (r["error"] or "")
        print("  " + _pad(r["eval_id"], 10) + _pad(r["bucket"], 15)
              + _pad(_label(r["expected"]), 8) + _pad(_label(r["actual"]), 14)
              + _pad(mark, 6) + _clip(note, 46))

    bad = [r for r in results if r["agree"] is False]
    if bad:
        print(f"\n不一致 {len(bad)} 件の詳細")
        for r in bad:
            print(f"  - {r['eval_id']} ({r['bucket']}, 台帳の印 {r['status']})")
            print(f"      質問        : {_clip(r['query'], 90)}")
            print(f"      期待        : {_label(r['expected'])} / "
                  f"judge: {_label(r['actual'])}")
            print(f"      judge の理由: {r['reason']}")

    rate = "  --  " if summary["rate"] is None else f"{summary['rate'] * 100:.1f}%"
    print(f"\n一致率 {rate} ({summary['agreed']}/{summary['judged']})  "
          f"不一致 {summary['mismatched']} 件  "
          f"呼び出し失敗 {summary['failed']} 件(一致率の母数から除外)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="幻覚ケース台帳の人手ラベルを正解として judge を回帰テストする(台帳は読むだけ)")
    p.add_argument("--limit", type=int, default=0, help="先頭 N 件だけ測る(試走用)")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY,
                   help="judge を同時に呼ぶ数")
    p.add_argument("--dry-run", action="store_true",
                   help="対象と skip の内訳だけを出す(judge を 1 回も呼ばない)")
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")   # cp932 の console でも落とさない
        except (AttributeError, ValueError):
            pass

    rows = await fetch_rows()
    targets, skipped = select_cases(rows)
    if args.limit > 0:
        targets = targets[:args.limit]

    print(f"幻覚ケース台帳 {len(rows)} 件 / 測れるケース {len(targets)} 件")
    for key, n in skipped.items():
        print(f"  skip {n} 件: {SKIP_LABELS[key]}")
    judged_by = collections.Counter(t["judge_model"] or "(不明)" for t in targets)
    print(f"今回の judge = {settings.chat_model} (temperature=0) / "
          f"台帳の判定に使われた judge = {dict(judged_by)}")

    if args.dry_run:
        print("--dry-run のため judge は呼びません")
        return 0
    if not targets:
        # 一致率の母数が 0。人が台帳に印を付けるまでは何も測れないので、
        # 「全件一致」と紛らわしくならないようここで打ち切る
        print("\n測れるケースが 1 件もありません。台帳に人手で印を付けてから実行してください")
        return 0

    chain = build_chain()
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def guarded(case: dict) -> dict:
        async with sem:
            return await judge_case(chain, case)

    results = await asyncio.gather(*[guarded(c) for c in targets])
    results.sort(key=lambda r: r["eval_id"])
    summary = summarize(results)
    print_report(results, summary)
    return 1 if summary["mismatched"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
