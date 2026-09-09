"""09 確信度のしきい値を評価セットで校正する。**勘で決めないための script**。

04 章の評価セットには、答えられるはずの bucket(A/B/C)と、ナレッジに無いので
断るべき bucket(D)がある。その両方で `evidence_confidence` の分布を出し、
しきい値を動かしながら「答えられる質問を通す率 − 断るべき質問を誤って通す率」
(Youden J)が最大になる点を選ぶ。

**モデルの生成も judge も呼ばない。** 検索とリランクだけを 300 件走らせる。
しきい値は「答えるか断るか」の方針そのものなので、実データの分布を見ずに
決めると、断りすぎ(答えられるのに断る)か、通しすぎ(根拠が無いのに答える)の
どちらかへ静かに倒れる。

前提: Milvus が起動し、ナレッジベースが構築済みであること
      (make milvus-up && make kb-build && make kb-vectorize)。

使い方: make calibrate-confidence
出力  : dev-notes/09-confidence-calibration.txt
"""

import asyncio
import io
import json
import pathlib
import sys

from app.core import retrieval
from app.core.confidence import compute_evidence_confidence

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "dev-notes" / "09-confidence-calibration.txt"
ANSWERABLE = {"A_policy", "B_model", "C_colloquial"}
# 同時実行数。上流の埋め込みとリランクに投げるので、04 章の評価と同じ程度に抑える
_SEM = asyncio.Semaphore(8)
_LINES: list[str] = []


def _log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(msg)


async def _conf(sample: dict) -> tuple[str, float]:
    async with _SEM:
        hits = await retrieval.search_knowledge(sample["query"], strategy="hybrid_rerank")
    return sample["bucket"], compute_evidence_confidence(hits).score


def _dist(name: str, xs: list[float]) -> None:
    xs = sorted(xs)
    if not xs:
        return

    def p(q: float) -> float:
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    _log(f"{name:18s} n={len(xs):3d} min={xs[0]:.3f} p25={p(.25):.3f} "
         f"p50={p(.5):.3f} p75={p(.75):.3f} max={xs[-1]:.3f}")


async def main() -> int:
    path = _ROOT / "tests" / "data" / "eval_04.jsonl"
    samples = [json.loads(ln) for ln in io.open(path, encoding="utf-8") if ln.strip()]
    results = await asyncio.gather(*(_conf(s) for s in samples))
    answerable = [c for b, c in results if b in ANSWERABLE]
    absent = [c for b, c in results if b == "D_absent"]

    _log("=== evidence_confidence の分布 (hybrid_rerank) ===")
    _dist("答えられる(ABC)", answerable)
    _dist("断るべき(D)", absent)

    _log("\n=== しきい値の走査 (通過率=ABC のうち conf>=t / 誤通過率=D のうち conf>=t) ===")
    _log(f"{'t':>6s} {'通過率':>10s} {'誤通過率':>12s} {'YoudenJ':>10s}")
    best_t, best_j = 0.0, -1.0
    for i in range(5, 96):
        t = i / 100
        tpr = sum(1 for c in answerable if c >= t) / len(answerable)
        fpr = sum(1 for c in absent if c >= t) / len(absent)
        j = tpr - fpr
        if i % 5 == 0 or j > best_j:
            _log(f"{t:6.2f} {tpr:10.3f} {fpr:12.3f} {j:10.3f}")
        if j > best_j:
            best_t, best_j = t, j

    _log(f"\n推奨 evidence_confidence_threshold = {best_t:.2f} (Youden J={best_j:.3f})")
    if best_j < 0.3:
        _log("**分離が弱い(J < 0.3)。この数字をそのまま採用しない。**"
             "信号か重みの見直しが要る。")
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _log(f"レポート: {_OUT.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
