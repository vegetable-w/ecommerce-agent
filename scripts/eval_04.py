"""04 章の評価: 4 戦略 (dense / bm25 / hybrid / hybrid_rerank) を bucket 別に比較する。

3 段構成で、前段だけでも成立するように分けてある。

  Stage 1 Retrieval        : 決定的。期待した節が section_path に当たったかで Recall / MRR。
                             複数根拠 (expect_sections_all) はグループ単位の部分点で測る。
  Stage 2 Evidence Coverage: 決定的。Top-K の evidence 本文が expect_points を何割含むか。
  Stage 3 Generation       : チャット上流を使う。4 戦略の Answer Coverage、hybrid_rerank の
                             Faithfulness、D bucket の回答拒否率。

Stage 1/2 は上流を 1 つも介さない**わけではない**(dense と hybrid は埋め込みが要る)が、
同じ入力に対して同じ数字を返す。Stage 3 だけがモデルの気分に左右されるので、失敗しても
Stage 1/2 の結果は必ず書き出す(report の meta.generation_complete を false にする)。

検索クエリはユーザーの原文をそのまま使い、query_understanding による書き換えは通さない。
書き換えを挟むと同じ入力でも毎回違う検索になり Stage 1/2 の再現性が失われるため。
書き換え器そのものの質は --check-rewrite で別途測る。

同じ理由で、足切り(min_score)も掛けない。4 戦略はスコアの尺度が別物なので、
閾値を掛けた状態で並べると「検索の質の差」ではなく「閾値の当てはまりの差」を測ってしまう。

実行:
    uv run --env-file .env python scripts/eval_04.py                 # 3 stage すべて
    uv run --env-file .env python scripts/eval_04.py --no-generation # Stage 1/2 のみ
    uv run --env-file .env python scripts/eval_04.py --check-rewrite --check-selfcheck
"""

import argparse
import asyncio
import datetime
import json
import pathlib
import sys

from pydantic import BaseModel, Field

from app.config import settings
from app.core import query_understanding, retrieval, selfcheck
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from app.kb import milvus_client

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "tests" / "data" / "eval_04.jsonl"
REWRITE_SET = ROOT / "tests" / "data" / "query_rewrite_samples.jsonl"
SELFCHECK_SET = ROOT / "tests" / "data" / "selfcheck_samples.jsonl"
OUT_DIR = ROOT / "data" / "04" / "reports"

STRATEGIES = ["dense", "bm25", "hybrid", "hybrid_rerank"]
GRADED_BUCKETS = ["A_policy", "B_model", "C_colloquial", "E_multi"]
BUCKETS = [*GRADED_BUCKETS, "D_absent"]
K = 10          # 検索深度。検索も MRR もここまでを見る
RECALL_K = 5    # Recall だけはこの深さで測る。K とは別物なので連動させない
# report(JSON)に出す Recall のキー名。読む側(評価ページ / トレンド)がキーで
# 深さを判別できるように、K ではなく RECALL_K を名前へ埋める。
RECALL_KEY = f"recall_at_{RECALL_K}"
RETR_CONCURRENCY, GEN_CONCURRENCY, CALL_TIMEOUT = 8, 3, 45.0

# 足切りなしを伝える番兵(app/tools/business.py と同じ考え方)。
# 0.0 だと「rerank スコアは 0 以上」という上流の値域への暗黙の仮定になる。
UNGATED = float("-inf")

# 回答拒否の判定に使う目印。RAG_ANSWER_SYSTEM が
# 「現在、関連する情報を確認できませんでした」と書くよう指示しているので 1 つ目が本命で、
# 残りは言い回しの揺れ(語尾、丁寧さ)を拾うための保険。
REFUSAL_MARKERS = (
    "確認できませんでした",
    "確認できません",
    "情報がございません",
    "情報はございません",
    "記載がございません",
    "記載がありません",
    "お答えできません",
)


# ---------------------------------------------------------------------------
# 純粋な指標計算。ここが壊れると後続の比較がすべて無意味になるので、
# 上流にも Milvus にも触れない関数として切り出し、tests/test_eval_04.py で固定する。
# ---------------------------------------------------------------------------


def norm(s: str | None) -> str:
    """空白をすべて落とす。「3 週間」と「3週間」、表の「| 修理 |」と「|修理|」を同一視する。"""
    return "".join((s or "").split())


def is_relevant(hit: dict, expect_section: list[str]) -> bool:
    """hit の section_path が expect_section の**すべて**を含むか。

    AND にしているのは、「交換時の送料」のように単独では 2 つの節に当たってしまう語を
    ["交換ポリシー", "交換時の送料"] と重ねて 1 節に絞り込めるようにするため。
    expect_section が空(D bucket)のときは正解が存在しないので常に False。
    """
    if not expect_section:
        return False
    path = hit.get("section_path") or ""
    return all(key in path for key in expect_section)


def first_relevant_rank(hits: list[dict], expect_section: list[str]) -> int | None:
    """1 つのグループに当たる chunk が最初に現れた順位(1 始まり)。1 件も無ければ None。"""
    for i, h in enumerate(hits, 1):
        if is_relevant(h, expect_section):
            return i
    return None


def as_groups(expect: list) -> list[list[str]]:
    """期待値を「グループのリスト」へ揃える。グループ 1 つが根拠 chunk 1 つを表す。

      * list[str]        … 旧 expect_section。グループ 1 個として扱う(後方互換)
      * list[list[str]]  … expect_sections_all。そのまま

    グループの中身は expect_section と同じ AND 条件(section_path がそのすべてを含む)で、
    その条件に当たる chunk が 1 件でもあればそのグループは満たされたとみなす。
    空グループは正解を指さないので落とす。
    """
    if not expect:
        return []
    if all(isinstance(e, str) for e in expect):
        return [list(expect)]
    return [[g] if isinstance(g, str) else list(g) for g in expect if g]


def expect_groups(sample: dict) -> list[list[str]]:
    """1 問の期待値を as_groups に通して取り出す。

    両方書かれている場合は expect_sections_all を採る。複数の根拠を宣言している以上
    そちらが正で、expect_section は単一根拠時代の書き方とみなす。
    """
    return as_groups(sample.get("expect_sections_all") or sample.get("expect_section") or [])


def recall_at_k(hits: list[dict], expect: list, k: int = RECALL_K) -> float:
    """上位 k 件で満たせたグループ数 / 全グループ数。

    グループ単位の部分点。3 グループのうち 2 つ当たれば 0.667 で、全部揃って 1.0。
    単一根拠(グループ 1 個)なら従来どおり 0.0 か 1.0 の hit rate になる。
    """
    groups = as_groups(expect)
    if not groups:
        return 0.0
    top = hits[:k]
    return sum(1 for g in groups if first_relevant_rank(top, g) is not None) / len(groups)


def reciprocal_rank(hits: list[dict], expect: list, k: int = K) -> float:
    """各根拠 chunk の逆順位の**平均**。圏外のグループは 0 として平均に含める。

    「最後の 1 グループが揃った順位」の逆数にしてはいけない。それだと 2 グループの問いは
    どう頑張っても上限 1/2 になり、単一根拠の bucket と同じ表に並べた瞬間、検索が劣化した
    ように見える(指標が bucket の構造を測ってしまう)。rank 1 と rank 2 に出たなら
    (1/1 + 1/2)/2 = 0.75 であって 0.5 ではない。
    """
    groups = as_groups(expect)
    if not groups:
        return 0.0
    top = hits[:k]
    rrs = []
    for g in groups:
        rank = first_relevant_rank(top, g)
        rrs.append(1.0 / rank if rank is not None else 0.0)
    return sum(rrs) / len(rrs)


def coverage_mech(points: list[str], hits: list[dict]) -> float | None:
    """expect_points のうち、検索できた evidence 本文に実在した割合。

    空白を落としてから部分一致を取る。points が空(D bucket)なら測る対象が無いので None。
    """
    if not points:
        return None
    ev = norm("\n".join(h.get("answer") or "" for h in hits))
    return sum(1 for p in points if norm(p) in ev) / len(points)


def looks_refused(answer: str | None) -> bool:
    """回答が「ナレッジに無いので答えられない」と述べているか。

    RAG_ANSWER_SYSTEM が定型句を指示しているため機械判定で足りる。判定を上流のモデルに
    任せると、D bucket の拒否率という**最も落としてはいけない指標**が上流の気分に依存する。
    """
    text = norm(answer)
    return any(norm(m) in text for m in REFUSAL_MARKERS)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(records: list[dict], field: str, buckets: list[str]) -> dict:
    """[{"bucket":..., field: 数値 or None}] を bucket 別 + overall の平均に畳む。

    None は「測る対象が無かった」であって 0 点ではないので、平均から外す
    (0 として混ぜると、D bucket を含めた overall が実態より低く出る)。
    """
    out: dict[str, dict] = {}
    for b in [*buckets, "overall"]:
        vals = [r[field] for r in records
                if r.get(field) is not None and (b == "overall" or r["bucket"] == b)]
        out[b] = {"value": _mean(vals), "n": len(vals)}
    return out


# ---------------------------------------------------------------------------
# 入出力
# ---------------------------------------------------------------------------

_LINES: list[str] = []
_ERRS: list[str] = []


def _log(line: str = "") -> None:
    _LINES.append(line)
    try:
        print(line)
    except UnicodeEncodeError:
        # cp932 の console で日本語が出せない場合でも run を落とさない。
        # ファイルは必ず UTF-8 で書くので、内容そのものは report 側に残る。
        enc = sys.stdout.encoding or "utf-8"
        print(line.encode(enc, errors="replace").decode(enc, errors="replace"))


def load_samples(path: pathlib.Path = EVAL_SET) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path.name} の {i} 行目が JSON として読めません: {exc}") from exc
        missing = {"id", "bucket", "query", "expect_points", "should_refuse"} - set(r)
        if missing:
            raise SystemExit(f"{path.name} の {i} 行目に項目が足りません: {sorted(missing)}")
        # 正解の形式は 2 つ。単一根拠は expect_section、複数根拠は expect_sections_all。
        # どちらか一方は必ず要る。両方あるときは expect_sections_all を採る(expect_groups と
        # 同じ優先順位。片方だけ直して片方を消し忘れた、という取り違えを防ぐため)。
        if "expect_section" not in r and "expect_sections_all" not in r:
            raise SystemExit(f"{path.name} の {i} 行目に項目が足りません: "
                             "['expect_section' か 'expect_sections_all' のどちらか]")
        rows.append(r)
    if not rows:
        raise SystemExit(f"{path} に評価サンプルがありません")
    return rows


def check_collection(client, collection: str) -> int:
    """評価を始める前に collection が使える状態かを確かめ、件数を返す。

    ここで止めないと、80 問 × 4 戦略の途中で `field section_path not exist` のような
    検索の奥からの例外が出て、何が悪いのか分からないまま run が壊れる。
    """
    if not client.has_collection(collection):
        raise SystemExit(
            f"Milvus collection「{collection}」がありません。"
            "先に make kb-build と make kb-vectorize でナレッジを構築してください。"
        )
    try:
        milvus_client.ensure_collection(client, collection=collection)
    except RuntimeError as exc:
        raise SystemExit(f"評価を開始できません: {exc}") from exc
    n = milvus_client.count(client, collection)
    if n == 0:
        raise SystemExit(
            f"Milvus collection「{collection}」が空です。"
            "make kb-vectorize でベクトル化してから評価してください。"
        )
    return n


# ---------------------------------------------------------------------------
# Stage 1 + 2: 決定的
# ---------------------------------------------------------------------------


async def run_deterministic(samples: list[dict], collection: str,
                            strategies: list[str]) -> tuple[dict, dict, dict]:
    """(retrieval 指標, evidence coverage 指標, HITS) を返す。

    HITS[(strategy, id)] = hits。Stage 3 で同じ検索結果を使い回すため、検索は 1 回だけ行う。
    """
    hits_by: dict[tuple[str, str], list[dict]] = {}
    sem = asyncio.Semaphore(RETR_CONCURRENCY)

    async def one(strategy: str, s: dict) -> None:
        async with sem:
            try:
                hits = await asyncio.wait_for(
                    retrieval.search_knowledge(
                        s["query"], strategy=strategy, top_k=K,
                        min_score=UNGATED, collection=collection),
                    CALL_TIMEOUT)
            except Exception as exc:
                _ERRS.append(f"retrieval[{strategy}/{s['id']}]: {type(exc).__name__}: {exc}")
                hits = []
        hits_by[(strategy, s["id"])] = hits

    await asyncio.gather(*[one(st, s) for st in strategies for s in samples])

    retrieval_out: dict[str, dict] = {}
    coverage_out: dict[str, dict] = {}
    for st in strategies:
        records = []
        for s in samples:
            hits = hits_by[(st, s["id"])]
            groups = expect_groups(s)
            records.append({
                "id": s["id"], "bucket": s["bucket"],
                "recall": recall_at_k(hits, groups, RECALL_K) if groups else None,
                "rr": reciprocal_rank(hits, groups, K) if groups else None,
                "coverage": coverage_mech(s["expect_points"], hits[:K]),
            })
        recall_agg = aggregate(records, "recall", GRADED_BUCKETS)
        retrieval_out[st] = {
            RECALL_KEY: recall_agg,
            # 旧キー。評価ページと過去 run のトレンドが recall_at_k を読んでいるので、
            # 同じ値を旧名でも残す(読む側が新旧どちらでも拾えるようにするための移行措置)。
            "recall_at_k": recall_agg,
            "mrr": aggregate(records, "rr", GRADED_BUCKETS),
            "per_sample": [{"id": r["id"], "bucket": r["bucket"], "recall": r["recall"],
                            "rr": r["rr"]} for r in records],
        }
        coverage_out[st] = {
            "coverage": aggregate(records, "coverage", GRADED_BUCKETS),
            "per_sample": [{"id": r["id"], "bucket": r["bucket"],
                            "coverage": r["coverage"]} for r in records],
        }
    return retrieval_out, coverage_out, hits_by


# ---------------------------------------------------------------------------
# Stage 3: 生成(チャット上流)
# ---------------------------------------------------------------------------


class _Coverage(BaseModel):
    """flat field。list[bool] だと長さがずれた瞬間に対応が崩れるので、番号の集合で受け取る。"""

    covered_numbers: list[int] = Field(
        default_factory=list, description="回答が言及できている key point の番号(1 始まり)")


class _Faithful(BaseModel):
    faithful: bool = Field(description="回答の事実が evidence に裏付けられているか")
    reason: str | None = Field(default="", description="判定の根拠を 1 文で")


_COVERAGE_INSTRUCTION = (
    "次のカスタマーサポートの回答が、模範解答の key point をいくつ言い当てているかを判定してください。\n"
    "言い回しが違っても、同じ事実を述べていれば言い当てたとみなします。回答に書かれていない\n"
    "key point は含めないでください。covered_numbers に、言い当てた key point の番号だけを入れてください。\n\n"
    "key point:\n{points}\n\nカスタマーサポートの回答:\n{answer}"
)


def build_evidence(hits: list[dict]) -> str:
    """query_faq と同じ体裁の番号付き evidence。生成にも判定にもこの文字列を渡す。"""
    return "\n".join(f"[{i}] {h.get('question')}: {h.get('answer')}"
                     for i, h in enumerate(hits, 1))


async def _try(coro, label: str):
    """上流呼び出しを timeout 付きで包み、失敗しても run 全体を落とさない。"""
    try:
        return await asyncio.wait_for(coro, CALL_TIMEOUT)
    except Exception as exc:
        _ERRS.append(f"{label}: {type(exc).__name__}: {exc}")
        return None


async def _generate_one(model, judge, s: dict, hits: list[dict], strategy: str) -> dict:
    evidence = build_evidence(hits)
    chain = RAG_ANSWER_PROMPT | model
    msg = await _try(chain.ainvoke({"query": s["query"], "evidence": evidence}),
                     f"generate[{strategy}/{s['id']}]")
    answer = getattr(msg, "content", None) if msg is not None else None
    rec = {"id": s["id"], "bucket": s["bucket"], "answer": answer,
           "refused": looks_refused(answer) if answer else None,
           "answer_coverage": None}
    if answer and s["expect_points"]:
        points = "\n".join(f"{i}. {p}" for i, p in enumerate(s["expect_points"], 1))
        r = await _try(
            judge.ainvoke(_COVERAGE_INSTRUCTION.format(points=points, answer=answer)),
            f"coverage[{strategy}/{s['id']}]")
        if r is not None:
            valid = {n for n in r.covered_numbers if 1 <= n <= len(s["expect_points"])}
            rec["answer_coverage"] = len(valid) / len(s["expect_points"])
    return rec


async def run_generation(samples: list[dict], hits_by: dict, strategies: list[str]) -> dict:
    model = get_chat_model()
    judge = model.with_structured_output(_Coverage)
    faith_chain = FAITHFULNESS_PROMPT | model.with_structured_output(_Faithful)
    sem = asyncio.Semaphore(GEN_CONCURRENCY)

    async def guarded(strategy: str, s: dict) -> dict:
        async with sem:
            return await _generate_one(model, judge, s, hits_by[(strategy, s["id"])], strategy)

    out: dict[str, dict] = {}
    answers: dict[str, dict[str, str]] = {}
    for st in strategies:
        results = await asyncio.gather(*[guarded(st, s) for s in samples],
                                       return_exceptions=True)
        records = []
        for s, r in zip(samples, results, strict=True):
            if isinstance(r, BaseException):
                _ERRS.append(f"generate[{st}/{s['id']}]: {type(r).__name__}: {r}")
                r = {"id": s["id"], "bucket": s["bucket"], "answer": None, "refused": None,
                     "answer_coverage": None}
            records.append(r)
        answers[st] = {r["id"]: r["answer"] for r in records if r["answer"]}
        refusals = [r for r in records if r["bucket"] == "D_absent" and r["refused"] is not None]
        out[st] = {
            "answer_coverage": aggregate(records, "answer_coverage", GRADED_BUCKETS),
            "refusal_rate": {
                "value": _mean([1.0 if r["refused"] else 0.0 for r in refusals]),
                "n": len(refusals),
            },
            "per_sample": [{"id": r["id"], "bucket": r["bucket"],
                            "answer_coverage": r["answer_coverage"],
                            "refused": r["refused"]} for r in records],
        }

    # Faithfulness は hybrid_rerank だけ測る。4 戦略ぶん測っても「同じ生成器が
    # evidence を裏切らないか」を 4 回聞くだけで、戦略の比較にはならない。
    target = "hybrid_rerank" if "hybrid_rerank" in strategies else strategies[-1]
    faith_records: list[dict] = []

    async def faith(s: dict) -> None:
        answer = answers[target].get(s["id"])
        if not answer:
            return
        async with sem:
            r = await _try(
                faith_chain.ainvoke({"evidence": build_evidence(hits_by[(target, s["id"])]),
                                     "answer": answer}),
                f"faithfulness[{s['id']}]")
        if r is not None:
            faith_records.append({"id": s["id"], "bucket": s["bucket"],
                                  "faithful": 1.0 if r.faithful else 0.0})

    await asyncio.gather(*[faith(s) for s in samples], return_exceptions=True)
    out["faithfulness"] = {
        "strategy": target,
        "value": _mean([r["faithful"] for r in faith_records]),
        "n": len(faith_records),
        "per_sample": sorted(faith_records, key=lambda r: r["id"]),
    }
    return out


# ---------------------------------------------------------------------------
# 積み残しのラベル付きサンプル(Task 8 / Task 9 で作ったが runner が無かった)
# ---------------------------------------------------------------------------


async def run_rewrite_check() -> dict:
    """クエリ理解: 書き換え後の standard + expanded に期待キーワードが 1 つでも残るか。"""
    rows = [json.loads(ln) for ln in
            REWRITE_SET.read_text(encoding="utf-8").splitlines() if ln.strip()]
    details = []
    for r in rows:
        u = await _try(query_understanding.understand(r["query"]), f"rewrite[{r['query'][:12]}]")
        if u is None:
            details.append({"query": r["query"], "ok": None, "standard": None})
            continue
        text = norm(u["standard"] + " " + " ".join(u["expanded"]))
        ok = any(norm(w) in text for w in r["expect_any"])
        details.append({"query": r["query"], "ok": ok, "standard": u["standard"],
                        "expanded": u["expanded"]})
    graded = [d for d in details if d["ok"] is not None]
    return {"pass_rate": _mean([1.0 if d["ok"] else 0.0 for d in graded]),
            "n": len(graded), "details": details}


async def run_selfcheck_check() -> dict:
    """自己評価ゲート: ラベル済みの (query, evidence) に対する useful 判定の正解率。"""
    rows = [json.loads(ln) for ln in
            SELFCHECK_SET.read_text(encoding="utf-8").splitlines() if ln.strip()]
    details = []
    for r in rows:
        c = await _try(selfcheck.check_sufficient(r["query"], r["evidence"]),
                       f"selfcheck[{r['id']}]")
        if c is None:
            details.append({"id": r["id"], "ok": None})
            continue
        details.append({"id": r["id"], "expect": r["expect_useful"], "got": c["useful"],
                        "ok": c["useful"] == r["expect_useful"], "reason": c["reason"]})
    graded = [d for d in details if d["ok"] is not None]
    return {"accuracy": _mean([1.0 if d["ok"] else 0.0 for d in graded]),
            "n": len(graded), "details": details}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _pct(v: float | None) -> str:
    return "  --  " if v is None else f"{v * 100:5.1f}%"


def _table(title: str, per_strategy: dict, path: list[str], buckets: list[str]) -> None:
    _log(f"\n{title}")
    _log("  " + "戦略".ljust(16) + "".join(b.ljust(14) for b in [*buckets, "overall"]))
    for st, data in per_strategy.items():
        node = data
        for key in path:
            node = node.get(key, {})
        cells = "".join(_pct((node.get(b) or {}).get("value")).ljust(14)
                        for b in [*buckets, "overall"])
        _log("  " + st.ljust(16) + cells)


def write_report(report: dict, out_dir: pathlib.Path = OUT_DIR) -> dict[str, pathlib.Path]:
    """.txt(人が読む run log) / .json(評価ページ用) / .html(単体で開ける run report)。

    encoding="utf-8" を必ず明示する。Windows の既定は cp932 で、指定を落とすと
    日本語が黙って壊れる(㎡ や 〜 のような文字は書き出しの時点で例外になる)。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {ext: out_dir / f"rag_eval.{ext}" for ext in ("txt", "json", "html")}
    paths["txt"].write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    paths["html"].write_text(_html(report), encoding="utf-8")
    return paths


def _html(report: dict) -> str:
    meta = report.get("meta", {})
    body = [
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">",
        "<title>RAG 評価レポート</title>",
        "<style>body{font-family:sans-serif;margin:2rem;line-height:1.6}"
        "table{border-collapse:collapse;margin-bottom:1.5rem}"
        "th,td{border:1px solid #ccc;padding:.3rem .6rem;text-align:right}"
        "th:first-child,td:first-child{text-align:left}pre{white-space:pre-wrap}</style></head><body>",
        "<h1>RAG 評価レポート</h1>",
        f"<p>実行時刻: {meta.get('generated_at', '')} / collection: "
        f"{meta.get('collection', '')} ({meta.get('collection_rows', '')} 件) / "
        f"サンプル {meta.get('samples', '')} 問</p>",
        "<pre>" + "\n".join(_LINES).replace("&", "&amp;").replace("<", "&lt;") + "</pre>",
        "</body></html>",
    ]
    return "\n".join(body)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="04 章 4 戦略 RAG 評価")
    p.add_argument("--collection", default=milvus_client.COLLECTION)
    p.add_argument("--strategies", default=",".join(STRATEGIES),
                   help="比較する戦略をカンマ区切りで指定する")
    p.add_argument("--limit", type=int, default=0, help="各 bucket から先頭 N 問だけ使う(試走用)")
    p.add_argument("--no-generation", action="store_true", help="Stage 3 を実行しない")
    p.add_argument("--check-rewrite", action="store_true", help="クエリ理解のサンプルも評価する")
    p.add_argument("--check-selfcheck", action="store_true", help="自己評価ゲートのサンプルも評価する")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args(argv)


def _limited(samples: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return samples
    out, seen = [], {}
    for s in samples:
        seen[s["bucket"]] = seen.get(s["bucket"], 0) + 1
        if seen[s["bucket"]] <= limit:
            out.append(s)
    return out


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")   # cp932 の console でも run を落とさない
        except (AttributeError, ValueError):
            pass

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    # search_knowledge は 03 章の呼び名 "vector" も受け付ける。ここで弾かない
    known = set(retrieval.STRATEGIES) | set(retrieval._ALIASES)
    unknown = [s for s in strategies if s not in known]
    if unknown:
        raise SystemExit(f"未知の検索戦略: {unknown}。{sorted(known)} から選んでください")

    samples = _limited(load_samples(), args.limit)
    client = milvus_client.get_client()
    rows = check_collection(client, args.collection)

    counts: dict[str, int] = {}
    for s in samples:
        counts[s["bucket"]] = counts.get(s["bucket"], 0) + 1
    started = datetime.datetime.now().isoformat(timespec="seconds")
    _log(f"評価開始 {started}  collection={args.collection}({rows} 件)  "
         f"サンプル {len(samples)} 問 {counts}")
    _log(f"戦略={strategies}  K={K}(Recall は上位 {RECALL_K} 件)  "
         f"埋め込み={settings.embed_model}  "
         f"リランク={settings.rerank_model}  生成={settings.chat_model}")

    retrieval_d, coverage_d, hits_by = await run_deterministic(samples, args.collection,
                                                              strategies)
    _table("Stage 1 Retrieval  Recall@%d" % RECALL_K, retrieval_d, [RECALL_KEY],
           GRADED_BUCKETS)
    _table("Stage 1 Retrieval  MRR", retrieval_d, ["mrr"], GRADED_BUCKETS)
    _table("Stage 2 Evidence Coverage(機械的な部分一致)", coverage_d, ["coverage"],
           GRADED_BUCKETS)

    generation_d = None
    if not args.no_generation:
        try:
            generation_d = await run_generation(samples, hits_by, strategies)
        except Exception as exc:   # 生成段が丸ごと落ちても Stage 1/2 は書き出す
            _log(f"[Stage 3 は完走できませんでした] {type(exc).__name__}: {exc}")
        else:
            per_strategy = {k: v for k, v in generation_d.items() if k in strategies}
            _table("Stage 3 Generation  Answer Coverage", per_strategy, ["answer_coverage"],
                   GRADED_BUCKETS)
            _log("\nStage 3 D bucket 回答拒否率")
            for st in strategies:
                node = generation_d[st]["refusal_rate"]
                _log(f"  {st.ljust(16)}{_pct(node['value'])}  (n={node['n']})")
            f = generation_d["faithfulness"]
            _log(f"\nStage 3 Faithfulness({f['strategy']}) {_pct(f['value'])} (n={f['n']})")

    extra = {}
    if args.check_rewrite:
        extra["query_rewrite"] = await run_rewrite_check()
        _log(f"\nクエリ理解 サンプル通過率 {_pct(extra['query_rewrite']['pass_rate'])} "
             f"(n={extra['query_rewrite']['n']})")
    if args.check_selfcheck:
        extra["self_check"] = await run_selfcheck_check()
        _log(f"自己評価ゲート 正解率 {_pct(extra['self_check']['accuracy'])} "
             f"(n={extra['self_check']['n']})")

    if _ERRS:
        _log(f"\n上流エラー {len(_ERRS)} 件(先頭 10 件)")
        for e in _ERRS[:10]:
            _log("  - " + e)

    report = {
        "meta": {
            "generated_at": started,
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "collection": args.collection, "collection_rows": rows,
            "samples": len(samples), "bucket_counts": counts,
            "k": K, "recall_k": RECALL_K, "strategies": strategies,
            "embed_model": settings.embed_model, "rerank_model": settings.rerank_model,
            "chat_model": settings.chat_model,
            "generation_complete": generation_d is not None,
            "errors": _ERRS,
        },
        "retrieval": retrieval_d,
        "evidence_coverage": coverage_d,
        "generation": generation_d,
        **extra,
    }
    paths = write_report(report, pathlib.Path(args.out_dir))
    _log("\n出力: " + " / ".join(str(p) for p in paths.values()))
    paths["txt"].write_text("\n".join(_LINES) + "\n", encoding="utf-8")  # 出力行自体も残す
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
