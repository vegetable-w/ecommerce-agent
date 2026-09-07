"""/api/rag-eval/faith-cases の入出力スキーマ。

status は Literal にして pydantic に弾かせる(不正値は 422)。台帳の status は DDL の
ENUM そのものであり、そこに無い値は綴りの誤りでしかない。app/schemas/kb.py が
content_type を Literal にせず 400 で返しているのとは判断が違う点に注意: あちらは
「登録済みの資料名かどうか」という業務の検査なので、意味の誤りとして 400 に寄せている。

対処メモ(resolution)の規則はここに書かない。必須かどうかは status に依存し、
評価スクリプトなど HTTP を通らない経路からも同じ規則が要るので、
repository.set_faith_case_status に置いてある(違反は ValueError → 400)。
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FaithStatus = Literal["unresolved", "resolved", "no_action_needed"]


class FaithCaseRow(BaseModel):
    """台帳 1 件。ORM の FaithCase をそのまま写す。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    eval_id: str
    bucket: str
    query: str
    strategy: str
    answer: str
    reason: str
    # その実行でモデルへ渡した Top-K 根拠の全件。[{n, chunk_id, section_path, question,
    # answer}] で、回答中の [n] はこの n を指す。古い行には無いことがあるので None 可
    citations: list[dict] | None
    judge_model: str | None
    status: FaithStatus
    seen_count: int
    first_seen_at: datetime
    last_seen_at: datetime
    resolution: str | None
    resolved_at: datetime | None


class FaithCaseListResponse(BaseModel):
    rows: list[FaithCaseRow]
    total: int = Field(description="status フィルタ適用後の件数。ページ送りの母数")
    page: int
    size: int
    pages: int = Field(description="総ページ数。0 件のときは 0")
    status: FaithStatus | None = Field(description="適用した status フィルタ")
    # None は「DB を読めなかったので集計していない」。全部 0 の dict とは意味が違う
    counts: dict | None = Field(
        description="status ごとの全体件数 + total。フィルタにもページにも依存しない"
    )
    error: str | None = None


class FaithCaseStatusRequest(BaseModel):
    status: FaithStatus = Field(description="unresolved / resolved / no_action_needed")
    resolution: str | None = Field(
        default=None,
        description="対処メモ。resolved / no_action_needed では必須。unresolved へ戻すときは無視して消す",
    )


class FaithCaseStatusResponse(BaseModel):
    case: FaithCaseRow
