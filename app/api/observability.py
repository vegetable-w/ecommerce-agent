"""可観測性ページ (/observability) の HTTP 出口。

09 章で作った 3 つのレポートを、端末に入れない運用者が読めるようにする層。

この層は **成果物を読むだけ** で、指標を 1 つも計算し直さない。app/api/rageval.py と
同じ規律で、理由も同じである。平均 token も割合も Youden J も推奨しきい値も、
`make cost-report` / `make calibrate-confidence` が計算して成果物に書いた値をそのまま
渡す。ここで割り算を書き足すと、端末の出力と画面の数字が食い違う道が開く
(しかも片方だけ直したときに、どちらが正しいか誰にも言えなくなる)。

したがって、この module が数を触る場所は 2 つだけ:

1. **結論を 1 か所で選ぶ。** 最もコストが高い intent は `top`(成果物の先頭行)。
   画面側で最大値を探させない。
2. **前回との差を取る。** トレンドの ↑↓ と ⚠ は保存済みの値どうしの引き算で、
   新しい指標を作っているわけではない。向き(DIRECTIONS)と誤差の幅(EPS)は
   scripts/eval_flywheel.py と一致させる(tests/test_observability_api.py が固定する)。

## section ごとに閉じる

トレンドの出所は MySQL の eval_runs、他の 2 つはローカルのファイルで、依存が別物である。
まとめて 1 つの try で包むと、MySQL が落ちている間はコストの内訳まで読めなくなる。
app/api/admin.py がカードごとに try を閉じるのと同じ考え方で、ここは section ごとに閉じる。

## 成果物が無い / 壊れている

500 にしない。clone 直後は必ず「無い」状態で、それは異常ではなく未実行である。
成果物は script が上書きで書くので、実行が途中で落ちれば書きかけの JSON が残る。
どちらも present=false として返し、**次に何をすればよいか**(hint)を添える。

## 時刻について

eval_runs.created_at は MySQL の CURRENT_TIMESTAMP なので **UTC** である(09 章 Task 10 の
申し送り)。ここでは変換せず DB の値をそのまま返し、UTC であることを timezone として
明示する。表示のためにこの画面だけ +9 時間すると、同じ列を出す他の画面
(/review など)と食い違うためで、時差の扱いはこの章ではなくテーブル全体で決めること。
"""

import json
import logging
import pathlib

from fastapi import APIRouter

from app.config import settings
from app.core import jobs, labels
from app.db import repository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/observability", tags=["observability"])

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# 3 つの script が書く場所。あちらが書き、ここは読むだけ。
REPORT_DIR = REPO_ROOT / "data" / "09" / "reports"
COST_JSON = REPORT_DIR / "cost_by_intent.json"
CALIBRATION_JSON = REPORT_DIR / "confidence_calibration.json"

COST_JOB = "cost-report"
TREND_JOB = "eval-flywheel"
CALIBRATION_JOB = "calibrate-confidence"

# 表に出す run 数。scripts/eval_flywheel.py の TREND_LIMIT と揃える。
TREND_LIMIT = 10
# これ以下の変化は → として扱う。scripts/eval_flywheel.py の EPS と揃える。
TREND_EPS = 0.005
# 指標と向き。+1 は「上がったら良い」。refusal_rate は D bucket(ナレッジに無い問い)を
# 断れた割合なので、下がることが悪化である(名前から逆に読めるので表に持たせる)。
TREND_METRICS: list[dict] = [
    {"key": "recall_at_10", "label": "Recall@10", "direction": 1},
    {"key": "mrr", "label": "MRR", "direction": 1},
    {"key": "faithfulness", "label": "忠実性", "direction": 1},
    {"key": "refusal_rate", "label": "ナレッジ外の拒否率", "direction": 1},
]

COST_NOT_RUN = (
    "intent 別の token 集計がまだありません。下の「コストを集計」から一度実行してください"
    "(Langfuse が起動している必要があります)。"
)
COST_BROKEN = (
    "集計レポートを読めませんでした(途中で終わった実行の書きかけと思われます)。"
    "もう一度実行してください。"
)
COST_EMPTY = "この窓には intent の分かる trace が 1 本もありませんでした。"
COST_EMPTY_HINT = (
    "集計そのものは動いています。まずチャット画面で何会話か流してから、窓の日数を"
    "見直したうえで集計し直してください。"
)
CALIBRATION_NOT_RUN = (
    "確信度の校正結果がまだありません。下の「しきい値を校正」から一度実行してください"
    "(Milvus と構築済みのナレッジベースが要ります)。"
)
CALIBRATION_BROKEN = (
    "校正レポートを読めませんでした(途中で終わった実行の書きかけと思われます)。"
    "もう一度実行してください。"
)
CALIBRATION_EMPTY = "校正は動いたものの、しきい値を走査できる結果がありませんでした。"
CALIBRATION_EMPTY_HINT = (
    "評価セットに対する検索が 1 件も返っていない可能性があります。"
    "ナレッジベースの構築状況(/kb)を確認してください。"
)
TREND_NOT_RUN = "評価の実行結果がまだ 1 件もありません。"

HINTS = {
    COST_JOB: f"make {COST_JOB} を実行すると集計されます(下のボタンからも実行できます)。",
    TREND_JOB: f"make {TREND_JOB} を実行すると 1 回分が eval_runs に記録されます"
               "(下のボタンからも実行できます)。",
    CALIBRATION_JOB: f"make {CALIBRATION_JOB} を実行すると校正されます"
                     "(下のボタンからも実行できます)。",
}


# ---------------------------------------------------------------------------
# 成果物の読み込み
# ---------------------------------------------------------------------------


def _load(path: pathlib.Path) -> tuple[dict | None, bool]:
    """(成果物, 壊れていたか) を返す。**例外は投げない。**

    「無い」と「壊れている」を区別するのは、運用者の次の行動が違うからである。
    前者はまだ実行していないだけ、後者は前回の実行が途中で死んだ可能性を示す。
    """
    if not path.is_file():
        return None, False
    try:
        # encoding は必ず指定する。既定は Windows では cp932 になり、日本語を含む
        # レポートで UnicodeDecodeError か文字化けを起こす
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("成果物を読めない path=%s %s", path, exc)
        return None, True
    if not isinstance(data, dict):
        # 書きかけの JSON がたまたま構文として通ることもある
        return None, True
    return data, False


def _rows(data: dict, key: str) -> list[dict] | None:
    """成果物の中の行の並び。list でなければ、または dict 以外が混ざれば None(壊れている)。"""
    value = data.get(key)
    if not isinstance(value, list):
        return None
    if any(not isinstance(row, dict) for row in value):
        return None
    return value


def _job(name: str) -> dict:
    """再実行ボタンが指す job の状態。registry 側が壊れても画面は出す。"""
    try:
        return jobs.status(name)
    except Exception as exc:  # pragma: no cover - registry が壊れたときの保険
        logger.exception("ジョブ状態を取得できない name=%s", name)
        return {"name": name, "error": f"取得できません({type(exc).__name__})"}


# ---------------------------------------------------------------------------
# intent 別のコスト
# ---------------------------------------------------------------------------


def cost_section() -> dict:
    """scripts/cost_by_intent.py の成果物をそのまま渡す。

    rows の中身には触らない(平均 token も割合も script が計算して書いている)。
    """
    data, broken = _load(COST_JSON)
    rows = _rows(data, "rows") if data is not None else None
    base = {"job": _job(COST_JOB), "report_path": str(COST_JSON)}
    if rows is None:
        return {
            **base, "present": False, "status": "not_run",
            "message": COST_BROKEN if (broken or data is not None) else COST_NOT_RUN,
            "hint": HINTS[COST_JOB],
            "generated_at": None, "days": None, "traces": None, "resolved": None,
            "unknown": None, "total_tokens": None, "rows": [], "top": None,
        }
    section = {
        **base, "present": True,
        "generated_at": data.get("generated_at"),
        "days": data.get("days"),
        "traces": data.get("traces"),
        "resolved": data.get("resolved"),
        "unknown": data.get("unknown"),
        "total_tokens": data.get("total_tokens"),
        "rows": rows,
        # 結論はここだけで選ぶ。成果物は token の降順で書かれているので先頭が最も高い
        "top": rows[0] if rows else None,
    }
    if not rows:
        return {**section, "status": "missing",
                "message": COST_EMPTY, "hint": COST_EMPTY_HINT}
    return {**section, "status": "ok", "message": None, "hint": None}


# ---------------------------------------------------------------------------
# 確信度の校正
# ---------------------------------------------------------------------------


def calibration_section() -> dict:
    """scripts/calibrate_confidence.py の成果物 + いま有効なしきい値。

    推奨値は成果物から読む。有効値は settings から読む。**この 2 つを比べることだけ**が
    この関数の仕事で、Youden J もしきい値も計算し直さない。
    """
    data, broken = _load(CALIBRATION_JSON)
    scan = _rows(data, "scan") if data is not None else None
    dists = _rows(data, "distributions") if data is not None else None
    active = settings.evidence_confidence_threshold
    base = {"job": _job(CALIBRATION_JOB), "report_path": str(CALIBRATION_JSON),
            "active_threshold": active}
    if scan is None or dists is None:
        return {
            **base, "present": False, "status": "not_run",
            "message": CALIBRATION_BROKEN if (broken or data is not None)
            else CALIBRATION_NOT_RUN,
            "hint": HINTS[CALIBRATION_JOB],
            "generated_at": None, "strategy": None, "distributions": [], "scan": [],
            "recommended_threshold": None, "best_j": None,
            "weak_separation": False, "in_sync": None,
        }
    recommended = data.get("recommended_threshold")
    if not isinstance(recommended, (int, float)) or isinstance(recommended, bool):
        recommended = None
    section = {
        **base, "present": True,
        "generated_at": data.get("generated_at"),
        "strategy": data.get("strategy"),
        "distributions": dists,
        "scan": scan,
        "recommended_threshold": recommended,
        "best_j": data.get("best_j"),
        "weak_separation": bool(data.get("weak_separation")),
        # 比べる相手が無ければ判定を出さない。None は False とは違う
        # (「ずれている」と「まだ分からない」を同じ印にしない)
        "in_sync": None if recommended is None else abs(recommended - active) < 1e-9,
    }
    if not scan:
        return {**section, "status": "missing",
                "message": CALIBRATION_EMPTY, "hint": CALIBRATION_EMPTY_HINT}
    return {**section, "status": "ok", "message": None, "hint": None}


# ---------------------------------------------------------------------------
# 評価トレンド
# ---------------------------------------------------------------------------


def _cell(direction: int, value, previous) -> dict:
    """1 つの指標の値と、前回からの向き。

    ↑↓ は値そのものの動きで、⚠(warn)だけが向きを掛けた良し悪しの判定である。
    ここを 1 つにまとめると、refusal_rate のように名前から逆に読める指標で必ず間違う。
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return {"value": None, "delta": None, "arrow": None, "warn": False}
    if not isinstance(previous, (int, float)) or isinstance(previous, bool):
        return {"value": value, "delta": None, "arrow": None, "warn": False}
    delta = value - previous
    if abs(delta) <= TREND_EPS:
        return {"value": value, "delta": delta, "arrow": "→", "warn": False}
    return {
        "value": value, "delta": delta,
        "arrow": "↑" if delta > 0 else "↓",
        "warn": delta * direction < 0,
    }


async def trend_section() -> dict:
    """eval_runs を新しい順に読む。**正本はテーブルで、成果物の txt ではない。**

    txt は端末で読むための控えなので、そちらを読みに行くと、画面が最後に
    レポートを書いた 1 回だけを永久に映すことになる。
    """
    base = {"job": _job(TREND_JOB), "metrics": TREND_METRICS, "timezone": "UTC"}
    try:
        runs = await repository.list_eval_runs(limit=TREND_LIMIT)
    except Exception as exc:
        logger.exception("評価トレンドを取得できない")
        # 例外メッセージ自体は出さない。SQLAlchemy の OperationalError は接続 URL を
        # 本文に含み、そこにはパスワードが入っている。型名だけなら秘密を持たない
        return {
            **base, "present": False, "status": "error",
            "message": f"評価の実行結果を取得できません({type(exc).__name__})",
            "hint": "MySQL が起動しているか確認してください。"
                    "この欄だけの失敗で、他の 2 つは通常どおり読めます。",
            "runs": [], "dropped": [],
        }

    rows = []
    for i, run in enumerate(runs):
        metrics = run.metrics if isinstance(run.metrics, dict) else {}
        # list_eval_runs は新しい順なので、1 つ後ろの行が「前回」になる
        previous = runs[i + 1].metrics if i + 1 < len(runs) else None
        previous = previous if isinstance(previous, dict) else {}
        rows.append({
            "id": run.id,
            "triggered_by": run.triggered_by,
            "triggered_by_label": labels.label(labels.EVAL_TRIGGERED_BY, run.triggered_by),
            "dataset_size": run.dataset_size,
            "created_at": run.created_at.isoformat(timespec="seconds")
            if run.created_at else None,
            "metrics": metrics,
            "cells": {
                m["key"]: _cell(m["direction"], metrics.get(m["key"]),
                                previous.get(m["key"]))
                for m in TREND_METRICS
            },
        })

    if not rows:
        return {**base, "present": False, "status": "not_run",
                "message": TREND_NOT_RUN, "hint": HINTS[TREND_JOB],
                "runs": [], "dropped": []}
    dropped = [m["key"] for m in TREND_METRICS if rows[0]["cells"][m["key"]]["warn"]]
    return {**base, "present": True, "status": "ok", "message": None, "hint": None,
            "runs": rows, "dropped": dropped}


# ---------------------------------------------------------------------------
# 画面 1 枚ぶん
# ---------------------------------------------------------------------------


async def build_overview() -> dict:
    return {
        "cost": cost_section(),
        "trend": await trend_section(),
        "calibration": calibration_section(),
    }


async def build_card() -> dict:
    """/admin のカード 1 枚ぶん。数値はこの module を通す(出所を 1 つにする)。

    /admin は「何かが落ちている最中に見る画面」なので、ここでは要約だけを返し、
    読めなかった欄は None にする。0 で埋めない(「0 件」と「読めなかった」は別物)。
    """
    overview = await build_overview()
    cost, trend, cal = overview["cost"], overview["trend"], overview["calibration"]
    top = cost["top"] or {}
    latest = trend["runs"][0] if trend["runs"] else {}
    return {
        "cost_present": cost["present"],
        "cost_status": cost["status"],
        "top_intent": top.get("intent"),
        "top_share": top.get("share"),
        "top_share_label": top.get("share_label"),
        "runs": len(trend["runs"]) if trend["status"] != "error" else None,
        "trend_status": trend["status"],
        "latest_faithfulness": (latest.get("metrics") or {}).get("faithfulness"),
        "dropped": trend["dropped"],
        "active_threshold": cal["active_threshold"],
        "recommended_threshold": cal["recommended_threshold"],
        "threshold_in_sync": cal["in_sync"],
        # 要対応 = 校正の結論が設定に反映されていない。レポートを出しただけで
        # .env を直し忘れる、が最も起きやすい取りこぼしなので、ここで名指しする
        "needs_attention": cal["in_sync"] is False,
    }


@router.get("/overview")
async def overview() -> dict:
    return await build_overview()
