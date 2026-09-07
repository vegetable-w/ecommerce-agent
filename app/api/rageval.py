"""RAG 評価ページ (/rag-eval) の HTTP 出口。

この層は **レポートを読むだけ** で、指標を 1 つも計算し直さない。

理由は費用と時間である。`make eval-rag` は 300 問 × 4 戦略で埋め込み・リランク・
生成・判定の上流をすべて実際に呼ぶ数分がかりの処理で、実費も出る。画面を開いた
だけでそれが走る作りにすると、管理画面を覗くたびに評価が動き出す。

もう 1 つの理由は真実が 2 つになることである。terminal の `make eval-rag` の出力と
この画面の数値は、同じ data/04/reports/rag_eval.json という 1 つの成果物から出て
いなければならない。ここで MRR を計算し直すと、集計方法がわずかに違うだけで
「画面では 97.6% なのに terminal では 96.1%」が起き、どちらが正しいか誰にも
言えなくなる。

したがってこのモジュールが artifact に対してすることは 2 つだけ:

1. そのまま渡す(retrieval / evidence_coverage / generation / meta)。
2. 結論として使う派生値(best)を **1 か所で** 選ぶ。画面の KPI も表のハイライトも
   この 1 つを参照する。画面側で「最大値を探す」処理を書かない。

レポートが無い / 壊れている場合は 500 ではなく `present=false` を返す。clone 直後は
必ずその状態であり、それは異常ではなく「まだ実行していない」という状態である。
"""

import json
import logging
import pathlib

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError

from app.core import jobs
from app.db import repository
from app.schemas.rageval import (
    FaithCaseListResponse,
    FaithCaseRow,
    FaithCaseStatusRequest,
    FaithCaseStatusResponse,
    FaithStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag-eval", tags=["rag-eval"])

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# scripts/eval_04.py の OUT_DIR と同じ場所。あちらが書き、ここは読むだけ。
REPORT_PATH = REPO_ROOT / "data" / "04" / "reports" / "rag_eval.json"

JOB_NAME = "eval-rag"
NOT_RUN_MESSAGE = (
    "評価レポートがまだありません。下の「RAG 評価を再実行」から一度実行してください"
    "（300 問 × 4 戦略で上流を呼ぶため数十分かかります）。"
)
BROKEN_MESSAGE = (
    "評価レポートを読めませんでした（途中で終わった実行の書きかけと思われます）。"
    "もう一度実行してください。"
)


def load_report() -> tuple[dict | None, str | None]:
    """(レポート, 読めなかった理由) を返す。例外は投げない。

    「ファイルが無い」と「JSON が壊れている」を別の文言にする。前者は未実行、
    後者は前回の実行が途中で死んだ可能性を示していて、利用者の次の行動が違う。
    """
    if not REPORT_PATH.is_file():
        return None, NOT_RUN_MESSAGE
    try:
        # encoding は必ず指定する。既定は Windows では cp932 になり、日本語を含む
        # レポートで UnicodeDecodeError か文字化けを起こす
        raw = REPORT_PATH.read_text(encoding="utf-8")
        report = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("評価レポートを読めない path=%s %s", REPORT_PATH, exc)
        return None, BROKEN_MESSAGE
    if not isinstance(report, dict) or not isinstance(report.get("retrieval"), dict):
        # 書きかけの JSON がたまたま構文として通ることもある(空の {} など)。
        # 必須の section が無いものは未実行として扱う
        return None, BROKEN_MESSAGE
    return report, None


def _value(metric) -> float | None:
    """{"value": x, "n": k} から x を取り出す。形が違えば None。"""
    if isinstance(metric, dict):
        v = metric.get("value")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _overall(section, key: str) -> float | None:
    """section[key]["overall"]["value"] を安全に取り出す。"""
    if not isinstance(section, dict):
        return None
    sub = section.get(key)
    if not isinstance(sub, dict):
        return None
    return _value(sub.get("overall"))


def pick_best(report: dict) -> dict | None:
    """overall MRR が最高の戦略を選ぶ。結論を決めるのはここ 1 か所だけ。

    戦略名は決め打ちしない。hybrid が dense を下回るような結果でも、artifact の
    数値がそう言うならそのまま結論になる(実測でそうなっている)。
    """
    retrieval = report.get("retrieval")
    if not isinstance(retrieval, dict):
        return None
    scored = [(name, _overall(sec, "mrr")) for name, sec in retrieval.items()]
    scored = [(n, v) for n, v in scored if v is not None]
    if not scored:
        return None
    name, mrr = max(scored, key=lambda pair: pair[1])

    gen = report.get("generation")
    gen_sec = gen.get(name) if isinstance(gen, dict) else None
    ev = report.get("evidence_coverage")
    ev_sec = ev.get(name) if isinstance(ev, dict) else None
    return {
        "strategy": name,
        "mrr": mrr,
        "recall_at_k": _overall_any(retrieval.get(name), "recall_at_5", "recall_at_k"),
        "evidence_coverage": _overall(ev_sec, "coverage"),
        "answer_coverage": _overall(gen_sec, "answer_coverage"),
        "refusal_rate": _value(gen_sec.get("refusal_rate")) if isinstance(gen_sec, dict) else None,
    }


def _overall_any(section, *keys: str) -> float | None:
    """先に見つかったキーの overall を返す。

    Recall のキーは recall_at_k から recall_at_5 へ変わった。以前の実行で作った
    artifact をそのまま開くこともあるので、新しい名前が無ければ古い名前を見る。
    拾えないと /admin のカードと評価画面の Recall が黙って「取得不可」になる。
    """
    for key in keys:
        v = _overall(section, key)
        if v is not None:
            return v
    return None


def _job_status() -> dict:
    """再実行ボタンが指す job の状態。job registry 側が壊れても画面は出す。"""
    try:
        return jobs.status(JOB_NAME)
    except Exception as exc:  # pragma: no cover - registry が壊れたときの保険
        logger.exception("ジョブ状態を取得できない name=%s", JOB_NAME)
        return {"name": JOB_NAME, "error": f"取得できません({type(exc).__name__})"}


def build_overview() -> dict:
    """画面 1 枚ぶんの内容。/admin のカードもここを通す(数値の出所を 1 つにする)。"""
    report, problem = load_report()
    job = _job_status()
    if report is None:
        return {
            "present": False, "message": problem, "meta": None,
            "retrieval": None, "evidence_coverage": None,
            "generation": None, "generation_done": False,
            "best": None, "job": job, "report_path": str(REPORT_PATH),
        }

    meta = report.get("meta") or {}
    generation = report.get("generation")
    generation = generation if isinstance(generation, dict) else None
    return {
        "present": True,
        "message": None,
        # 以下 3 つは artifact の値をそのまま渡す。加工も再計算もしない
        "meta": meta,
        "retrieval": report.get("retrieval"),
        "evidence_coverage": report.get("evidence_coverage"),
        "generation": generation,
        "generation_done": bool(generation) and bool(meta.get("generation_complete", True)),
        "best": pick_best(report),
        "job": job,
        "report_path": str(REPORT_PATH),
    }


@router.get("/overview")
async def overview() -> dict:
    return build_overview()


# ---------------------------------------------------------------------------
# 幻覚ケース台帳 (Task 19)
#
# 上の overview と違い、ここだけは artifact ではなく MySQL の faith_cases を読む。
# 台帳は実行をまたいで積み上がるもので、実行ごとに作り直されるレポートには置けない。
# ---------------------------------------------------------------------------


@router.get("/faith-cases", response_model=FaithCaseListResponse)
async def faith_cases(
    status: FaithStatus | None = Query(default=None, description="status で絞り込む"),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
) -> FaithCaseListResponse:
    """台帳の一覧。DB が落ちていても評価画面ごと 500 にはしない。

    レポートが無いときに /overview が present=false という「状態」を返すのと同じ考え方で、
    読めなかったことを error に載せた空の一覧として返す。台帳は評価画面に後から足した
    管理用の欄であり、MySQL が落ちている間に評価指標まで見られなくなるのは割に合わない。

    counts は読めなければ None にする。全部 0 の dict にしてはいけない
    (「台帳が空」と「集計できなかった」は画面で言うべきことが違う)。
    """
    try:
        result = await repository.list_faith_cases(status=status, page=page, size=size)
    except Exception as exc:
        logger.exception("幻覚ケース台帳を取得できない")
        return FaithCaseListResponse(
            rows=[], total=0, page=page, size=size, pages=0, status=status,
            counts=None, error=f"データベースから取得できません({type(exc).__name__})",
        )
    return FaithCaseListResponse(
        rows=[FaithCaseRow.model_validate(r) for r in result["rows"]],
        total=result["total"], page=result["page"], size=result["size"],
        pages=result["pages"], status=result["status"], counts=result["counts"],
        error=None,
    )


@router.post("/faith-cases/{case_id}/status", response_model=FaithCaseStatusResponse)
async def update_faith_case_status(
    case_id: int, req: FaithCaseStatusRequest
) -> FaithCaseStatusResponse:
    """1 件の状態を人手で変える。更新後の行をそのまま返す(画面はこれで描き直す)。

    status の綴り誤りはスキーマの Literal が先に 422 で弾く。ここで if を書き足して
    400 にし直さないこと(検査が 2 か所になる)。400 にするのは対処メモの規則違反だけ。
    """
    try:
        row = await repository.set_faith_case_status(case_id, req.status, req.resolution)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.exception("幻覚ケースの状態を更新できない id=%s", case_id)
        raise HTTPException(
            status_code=503,
            detail="データベースを一時的に利用できません。しばらくしてからもう一度お試しください",
        ) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"ケース {case_id} は見つかりません")
    return FaithCaseStatusResponse(case=FaithCaseRow.model_validate(row))
