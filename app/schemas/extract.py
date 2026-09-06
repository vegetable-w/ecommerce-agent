from enum import Enum

from pydantic import BaseModel, Field

class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, description="ユーザーのアフターサービスに関する説明原文")

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
