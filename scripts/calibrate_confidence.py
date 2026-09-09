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
出力  : data/09/reports/confidence_calibration.{txt,json}
        txt は端末で読むための控えで、json は /observability の画面が読む。
        **分布も Youden J も推奨しきい値もここで計算して両方へ書く。**
        画面側で走査をやり直すと、端末の結論と画面の結論が食い違う道が開く。
"""

import asyncio
import io
import json
import pathlib
import sys
from datetime import datetime, timezone

from app.core import retrieval
from app.core.confidence import compute_evidence_confidence

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / "data" / "09" / "reports"
_OUT = _OUT_DIR / "confidence_calibration.txt"
_OUT_JSON = _OUT_DIR / "confidence_calibration.json"
STRATEGY = "hybrid_rerank"
ANSWERABLE = {"A_policy", "B_model", "C_colloquial"}
# 同時実行数。上流の埋め込みとリランクに投げるので、04 章の評価と同じ程度に抑える
_SEM = asyncio.Semaphore(8)
_LINES: list[str] = []


def _log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(msg)


async def _conf(sample: dict) -> tuple[str, float]:
    async with _SEM:
        hits = await retrieval.search_knowledge(sample["query"], strategy=STRATEGY)
    return sample["bucket"], compute_evidence_confidence(hits).score


def _dist(name: str, xs: list[float]) -> dict | None:
    """分布を 1 行出し、同じ値を成果物用に返す。

    JSON 側は端末に出したのと同じ桁(小数第 3 位)で丸める。端末が 0.732 と出して
    画面が 0.7315… と出すと、同じ script の同じ実行なのに数字が 2 つあることになる。
    """
    xs = sorted(xs)
    if not xs:
        return None

    def p(q: float) -> float:
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    _log(f"{name:18s} n={len(xs):3d} min={xs[0]:.3f} p25={p(.25):.3f} "
         f"p50={p(.5):.3f} p75={p(.75):.3f} max={xs[-1]:.3f}")
    return {"name": name.strip(), "n": len(xs), "min": round(xs[0], 3),
            "p25": round(p(.25), 3), "p50": round(p(.5), 3),
            "p75": round(p(.75), 3), "max": round(xs[-1], 3)}


async def main() -> int:
    path = _ROOT / "tests" / "data" / "eval_04.jsonl"
    samples = [json.loads(ln) for ln in io.open(path, encoding="utf-8") if ln.strip()]
    results = await asyncio.gather(*(_conf(s) for s in samples))
    answerable = [c for b, c in results if b in ANSWERABLE]
    absent = [c for b, c in results if b == "D_absent"]

    _log(f"=== evidence_confidence の分布 ({STRATEGY}) ===")
    dists = [d for d in (_dist("答えられる(ABC)", answerable),
                         _dist("断るべき(D)", absent)) if d]

    _log("\n=== しきい値の走査 (通過率=ABC のうち conf>=t / 誤通過率=D のうち conf>=t) ===")
    _log(f"{'t':>6s} {'通過率':>10s} {'誤通過率':>12s} {'YoudenJ':>10s}")
    # 端末には 5 刻みと更新のあった点だけを出す(91 行を読ませても目が滑る)が、
    # 成果物には走査した点を全部残す。画面はこれを曲線として描くので間引くと折れる。
    # 出す値は同じなので、端末に出た行はそのまま JSON の中にもある。
    scan: list[dict] = []
    best_t, best_j = 0.0, -1.0
    for i in range(5, 96):
        t = i / 100
        tpr = sum(1 for c in answerable if c >= t) / len(answerable)
        fpr = sum(1 for c in absent if c >= t) / len(absent)
        j = tpr - fpr
        if i % 5 == 0 or j > best_j:
            _log(f"{t:6.2f} {tpr:10.3f} {fpr:12.3f} {j:10.3f}")
        scan.append({"t": round(t, 2), "tpr": round(tpr, 3),
                     "fpr": round(fpr, 3), "j": round(j, 3)})
        if j > best_j:
            best_t, best_j = t, j

    _log(f"\n推奨 evidence_confidence_threshold = {best_t:.2f} (Youden J={best_j:.3f})")
    weak = best_j < 0.3
    if weak:
        _log("**分離が弱い(J < 0.3)。この数字をそのまま採用しない。**"
             "信号か重みの見直しが要る。")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _OUT_JSON.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": STRATEGY,
        "distributions": dists,
        "scan": scan,
        # 推奨値の選定は上の走査だけが行う。画面は選び直さない
        "recommended_threshold": round(best_t, 2),
        "best_j": round(best_j, 3),
        "weak_separation": weak,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _log(f"レポート: {_OUT.relative_to(_ROOT)} / {_OUT_JSON.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
