"""返金・返品の申請フォームを画面へ出すためのツール。"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools import registry


class RefundInput(BaseModel):
    order_id: str = Field(description="返金・返品の対象となる注文番号。例: 1001")
    reason: str | None = Field(
        default=None,
        description="返品・返金の理由。分かる場合だけ書く。最終的な理由は画面のフォームでユーザーが選ぶ",
    )


@tool(args_schema=RefundInput)
async def submit_refund(order_id: str, reason: str | None = None) -> dict:
    """返品・返金が可能だと判断できた場合に呼ぶ。**返金の実行ではない。**

    **これがユーザーの画面に申請フォームを出す唯一の手段である。** このツールを
    呼ばずに「申請フォームを提示します」「手続きを進めます」と書いても、画面には
    何も出ず、ユーザーは待たされたまま何もできない。可能だと判断したなら、
    そう書く前にこのツールを呼ぶこと。

    実際に申請が作られるのは、ユーザーがその画面で送信したときだけなので、
    「返金を受け付けました」「返金処理を開始しました」のように完了した言い方は
    してはならない。

    規約の期限を超えている場合、または可否を判断できない場合は呼ばないこと。"""
    # ここは実行されない。app/graph/nodes.py の agent_tools が create_ticket と同じ作法で
    # 横取りし、画面の選択肢(refund_form)へ変換する。**DB には書かない。**
    # それでも本体を空にしないのは、横取りの手前でツールとして成立していないと、
    # 横取りを外したときに静かに「不明なツール」へ倒れて原因が見えなくなるため。
    return {"status": "ユーザーの確認待ち", "order_id": order_id, "reason": reason}


registry.register(registry.spec_from_langchain_tool(submit_refund, source="builtin"))
