"""09 評価の定期実行: 04 章の評価セットを回し、1 回分を eval_runs へ残してトレンドにする。

測るのは 4 つだけ。

  検索段 Recall@10 / MRR  hybrid_rerank、答えられる bucket(A/B/C/E)
  生成段 Faithfulness     同じ bucket。回答が渡した根拠を裏切っていないか
  生成段 refusal_rate     D bucket(ナレッジに無い問い)を断れた割合

**重点は 1 回のスコアではなくトレンド**である。絶対値は上流のモデルを替えただけでも動く。
知りたいのは「前回からどちらへ動いたか」なので、保存した run を新しい順に並べ、
前回と比べて悪い方へ動いた指標に ⚠ を付ける。

指標の計算は scripts/eval_04.py の関数をそのまま呼ぶ。同じ式をここに書き直すと、
04 章の評価とこのトレンドが少しずつ別のものを測るようになり、しかも両方それらしい
数字を出すので誰も気づけない。検索も eval_04.run_deterministic に任せる
(top_k / 足切りなし / 書き換えを通さない、という評価時の検索条件ごと借りるため)。

なお 04 章のレポートが表に出す Recall は上位 5 件(eval_04.RECALL_K)だが、ここは
K=10 まで見た Recall を残す。深さが違えば別の数字なので、キー名にも深さを埋める
(eval_04.RECALL_KEY と同じ考え方)。

## 課金

全量(300 問)を指定すると、回答の生成を 300 回、judge を 240 回上流へ投げる。
**課金が大きい。** 動作を確かめるだけなら --limit を必ず付けること。--limit は
bucket ごとに均等に取るので、先頭から N 件を切り出したときのような A_policy への
偏りは出ない。dataset_size には実際に評価した件数が入る。

実行:
    PYTHONUTF8=1 uv run --env-file .env python scripts/eval_flywheel.py --limit 10
    make eval-flywheel LIMIT=10
    make eval-flywheel TRIGGER=scheduled LIMIT=10
    make eval-flywheel                          # 全量。課金が大きい

定期実行の例(毎朝 6 時。scheduler は入れない。出力は追記して残す):
    0 6 * * * cd /path/to/ecommerce-agent && make eval-flywheel TRIGGER=scheduled >> log/eval.log 2>&1

前提: Milvus と MySQL が起動し、ナレッジベースが構築済みであること。
出力: dev-notes/09-eval-trend.txt(eval_runs が正本で、この txt は読むための控え)
"""

import argparse
import asyncio
import datetime
import pathlib
import sys

from app.core import labels
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from app.db import repository
from app.kb import milvus_client
from scripts import eval_04 as ev

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "dev-notes" / "09-eval-trend.txt"

# 04 章の 4 戦略のうち、本番が使っている 1 つだけを回す。トレンドは戦略の比較ではない。
STRATEGY = "hybrid_rerank"
# 表に出す run 数。eval_runs は増え続けるので、読む側は直近だけを見る。
TREND_LIMIT = 10
# これ以下の変化は → として扱う。小数第 3 位まで残すので、その 1/2 を境にする。
# 揺れまで ↑↓ として拾うと、毎回どれかが動いて ⚠ が意味を失う。
EPS = 0.005

# Recall の深さはキー名に埋める。読む側(トレンドの画面)がキーだけで深さを判別できる。
RECALL_KEY = f"recall_at_{ev.K}"

# 指標の向き。+1 は「上がったら良い」。4 つとも +1 だが、refusal_rate だけは
# 名前から「拒否が増えた = 悪化」と読めてしまうので、表に持たせて明示する。
# ここで測る refusal_rate は **D bucket(ナレッジに無いので断るべき問い)の拒否率**で、
# 下がることは「知らないことを答えてしまった」を意味する。だから他の 3 つと同じく
# 下がったときに ⚠ を付ける。向きを直感で決めると、ここだけ逆に付く。
DIRECTIONS: dict[str, int] = {
    RECALL_KEY: +1,
    "mrr": +1,
    "faithfulness": +1,
    "refusal_rate": +1,
}
METRIC_NAMES = list(DIRECTIONS)


# ---------------------------------------------------------------------------
# サンプルの間引き
# ---------------------------------------------------------------------------


def limited_even(samples: list[dict], limit: int) -> list[dict]:
    """先頭から limit 件。ただし bucket を 1 件ずつ回りながら取る。

    評価セットは bucket ごとにまとまって並んでいるので、素直に先頭 N 件を取ると
    A_policy だけになり、D bucket が 0 件 = 拒否率が測れない run になる。

    eval_04._limited は「各 bucket から N 件」で、こちらは「全体で N 件」。
    課金の上限を件数で押さえたいので、bucket 数を掛けた数にならない方を採る。
    """
    if limit <= 0:
        return samples
    by_bucket: dict[str, list[dict]] = {}
    for s in samples:
        by_bucket.setdefault(s["bucket"], []).append(s)
    picked: list[dict] = []
    for i in range(max(len(v) for v in by_bucket.values())):
        for rows in by_bucket.values():
            if i < len(rows):
                picked.append(rows[i])
                if len(picked) >= limit:
                    return picked
    return picked


# ---------------------------------------------------------------------------
# 1 回分の評価
# ---------------------------------------------------------------------------


async def run_retrieval(samples: list[dict], collection: str) -> tuple[dict, dict]:
    """(検索段の指標, hits)。検索そのものは eval_04.run_deterministic に任せる。"""
    _, _, hits_by = await ev.run_deterministic(samples, collection, [STRATEGY])
    records = []
    for s in samples:
        hits = hits_by[(STRATEGY, s["id"])]
        groups = ev.expect_groups(s)
        # D bucket は正解の節が無いので測る対象から外れる(None は平均に含まれない)。
        records.append({
            "bucket": s["bucket"],
            "recall": ev.recall_at_k(hits, groups, ev.K) if groups else None,
            "rr": ev.reciprocal_rank(hits, groups, ev.K) if groups else None,
        })
    return {
        RECALL_KEY: ev.aggregate(records, "recall", ev.GRADED_BUCKETS)["overall"],
        "mrr": ev.aggregate(records, "rr", ev.GRADED_BUCKETS)["overall"],
    }, hits_by


async def run_generation(samples: list[dict], hits_by: dict) -> dict:
    """生成段の指標。答えられる bucket は Faithfulness、D bucket は拒否率。

    D bucket に judge は掛けない。断り文句が根拠に忠実かを聞いても、拒否率で見たいこと
    (知らないと言えたか)は測れないうえ、1 問につき judge の呼び出しが 1 回増える。
    """
    # 回答は本番と同じ temperature、judge は 0(同じ回答には同じ判定を返してほしい)。
    # eval_04.run_generation と同じ組み方。
    answer_chain = RAG_ANSWER_PROMPT | get_chat_model()
    faith_chain = FAITHFULNESS_PROMPT | get_chat_model(
        temperature=0).with_structured_output(ev._Faithful)
    sem = asyncio.Semaphore(ev.GEN_CONCURRENCY)

    async def one(s: dict) -> dict:
        hits = hits_by[(STRATEGY, s["id"])]
        evidence = ev.build_evidence(hits)
        async with sem:
            msg = await ev._try(
                lambda: answer_chain.ainvoke({"query": s["query"], "evidence": evidence}),
                f"generate[{s['id']}]")
        answer = getattr(msg, "content", None) if msg is not None else None
        rec = {"bucket": s["bucket"], "faithful": None, "refused": None}
        if not answer:
            return rec
        if s["bucket"] not in ev.GRADED_BUCKETS:
            rec["refused"] = 1.0 if ev.looks_refused(answer) else 0.0
            return rec
        async with sem:
            r = await ev._try(
                lambda: faith_chain.ainvoke({"evidence": evidence, "answer": answer}),
                f"faithfulness[{s['id']}]")
        if r is not None:
            rec["faithful"] = 1.0 if r.faithful else 0.0
        return rec

    results = await asyncio.gather(*[one(s) for s in samples], return_exceptions=True)
    records = []
    for s, r in zip(samples, results, strict=True):
        if isinstance(r, BaseException):
            ev._ERRS.append(f"generate[{s['id']}]: {type(r).__name__}: {r}")
            r = {"bucket": s["bucket"], "faithful": None, "refused": None}
        records.append(r)
    faith = [r["faithful"] for r in records if r["faithful"] is not None]
    refused = [r["refused"] for r in records if r["refused"] is not None]
    return {
        "faithfulness": {"value": ev._mean(faith), "n": len(faith)},
        "refusal_rate": {"value": ev._mean(refused), "n": len(refused)},
    }


async def run_once(samples: list[dict], collection: str) -> dict:
    """今回の run の metrics を組み立てる。n(分母)はログにだけ出す。"""
    retrieval_m, hits_by = await run_retrieval(samples, collection)
    generation_m = await run_generation(samples, hits_by)
    nodes = {**retrieval_m, **generation_m}
    metrics: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        node = nodes[name]
        ev._log(f"  {name:16s} {ev._pct(node['value'])}  (n={node['n']})")
        metrics[name] = None if node["value"] is None else round(node["value"], 3)
    return metrics


# ---------------------------------------------------------------------------
# トレンド
# ---------------------------------------------------------------------------


def cell(name: str, value: float | None, previous: float | None) -> str:
    """値 + 前回からの向き。→ は変化なし、⚠ は「この指標にとって悪い方へ動いた」。"""
    if value is None:
        return "--"
    if previous is None:
        return f"{value:.3f}"
    delta = value - previous
    if abs(delta) <= EPS:
        return f"{value:.3f} →"
    # ↑↓ は値そのものの動き。良し悪しの判定だけ向き(DIRECTIONS)を掛ける。
    # refusal_rate は「上がったら良い」ので、上がって ⚠ が付くことはない。
    warn = "" if delta * DIRECTIONS[name] > 0 else " ⚠"
    return f"{value:.3f} {'↑' if delta > 0 else '↓'}{warn}"


def show_trend(runs: list) -> None:
    """新しい順に並べ、1 つ古い run と比べる。

    list_eval_runs は新しい順に返る。**今回の run を保存した後に読んでいる**ので
    runs[0] が今回、runs[1] が前回になる。保存する前に読むと runs[0] が前回になり、
    比較の相手が 1 つずつずれる(しかも表はもっともらしく出るので気づけない)。
    """
    ev._log(f"\n=== 評価トレンド(新しい順、直近 {len(runs)} 回)===")
    ev._log(f"{'run':>5s} {'日時':11s} {'由来':6s} {'件数':>5s}"
            + "".join(f"{n:>18s}" for n in METRIC_NAMES))
    for i, r in enumerate(runs):
        prev = runs[i + 1].metrics if i + 1 < len(runs) else None
        cells = "".join(
            f"{cell(n, r.metrics.get(n), (prev or {}).get(n)):>18s}" for n in METRIC_NAMES)
        ev._log(f"#{r.id:>4d} {r.created_at:%m-%d %H:%M} "
                f"{labels.label(labels.EVAL_TRIGGERED_BY, r.triggered_by)} "
                f"{r.dataset_size:>5d}{cells}")
    if len(runs) < 2:
        ev._log("\n前回の run がまだありません(次回から比較が出ます)。")
        return
    latest, previous = runs[0].metrics, runs[1].metrics
    drops = [n for n in METRIC_NAMES
             if latest.get(n) is not None and previous.get(n) is not None
             and (latest[n] - previous[n]) * DIRECTIONS[n] < -EPS]
    ev._log(("\n⚠ 下がった指標: " + "、".join(drops)) if drops
            else "\n前回から下がった指標はありません。")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="09 評価の定期実行とトレンド")
    p.add_argument("--triggered-by", default="manual", choices=sorted(labels.EVAL_TRIGGERED_BY),
                   help="この run の由来。cron からは scheduled")
    p.add_argument("--limit", type=int, default=0,
                   help="評価する問題数(bucket ごとに均等に取る)。0 は全量で課金が大きい")
    p.add_argument("--collection", default=milvus_client.COLLECTION)
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")   # cp932 の console でも run を落とさない
        except (AttributeError, ValueError):
            pass

    samples = limited_even(ev.load_samples(), args.limit)
    counts: dict[str, int] = {}
    for s in samples:
        counts[s["bucket"]] = counts.get(s["bucket"], 0) + 1
    client = milvus_client.get_client()
    rows = ev.check_collection(client, args.collection)

    started = datetime.datetime.now().isoformat(timespec="seconds")
    ev._log(f"=== 09 評価の定期実行({started} / 由来={args.triggered_by})===")
    ev._log(f"collection={args.collection}({rows} 件)  戦略={STRATEGY}  K={ev.K}  "
            f"サンプル {len(samples)} 問 {counts}")
    if args.limit <= 0:
        ev._log("全量です。回答の生成と judge を評価セットの数だけ上流へ投げます"
                "(課金が大きい)。動作を確かめるだけなら --limit を付けてください。")

    ev._log("\n=== 今回の run ===")
    metrics = await run_once(samples, args.collection)

    if ev._ERRS:
        ev._log(f"\n上流エラー {len(ev._ERRS)} 件(先頭 5 件)")
        for e in ev._ERRS[:5]:
            ev._log("  - " + e)

    code = 0
    if all(v is None for v in metrics.values()):
        # 1 つも測れていない run を積んでも、トレンドに空の行が増えるだけで害しかない。
        ev._log("\n測れた指標が 1 つもありません。eval_runs へは保存しません。")
        code = 1
    else:
        try:
            run_id = await repository.insert_eval_run(args.triggered_by, len(samples), metrics)
        except Exception as exc:
            ev._log(f"\n[eval_runs へ保存できませんでした] {type(exc).__name__}: {exc}")
            code = 1
        else:
            ev._log(f"\neval_runs #{run_id} として保存しました(dataset_size={len(samples)})。")
            show_trend(await repository.list_eval_runs(limit=TREND_LIMIT))

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(ev._LINES) + "\n", encoding="utf-8")
    ev._log(f"\nレポート: {_OUT.relative_to(_ROOT)}")
    return code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
