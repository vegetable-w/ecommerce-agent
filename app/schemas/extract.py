from enum import Enum

from pydantic import BaseModel, Field, field_validator

class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, description="ユーザーのアフターサービスに関する説明原文")

    # min_lengthはOpenAPIスキーマのminLengthとして表出させるために残し、
    # 空白のみの値(min_lengthを通過してしまう)はこのvalidatorで拒否する
    @field_validator("text")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("空白のみの値は許可されない")
        return v

class RequestType(str, Enum):
    REFUND = "返金"
    EXCHANGE = "交換"
    REPAIR = "修理"
    COMPLAINT = "苦情"
    OTHER = "その他"

class AfterSalesTicket(BaseModel):
    """ユーザーのアフターサービスに関する説明から抽出した構造化チケット。"""

    order_id: str | None = Field(
        default=None, description="注文番号。原文に明示されていない場合はnull。推測・捏造は禁止"
    )
    request_type: RequestType = Field(description="ユーザーの要望種別")
    expected_solution: str = Field(description="ユーザーが期待する対応内容を一文で要約")
