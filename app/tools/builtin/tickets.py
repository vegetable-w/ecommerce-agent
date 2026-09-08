"""有人対応チケットの作成ツール。"""
import asyncio
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool

from app.config import settings
from app.core import labels
from app.db import repository
from app.tools import registry


@tool
async def create_ticket(
    description: str,
    ticket_type: Literal["after_sales", "complaint", "inquiry"],
    conversation_id: Annotated[int, InjectedToolArg],
) -> dict:
    """ユーザーがオペレーター対応を明確に希望した場合、苦情の場合、またはセルフサービスで解決できない場合にチケットを作成する。
    description にはユーザーの問題を記載する。ticket_type は次から 1 つだけ選び、英語の識別子をそのまま指定する:
      after_sales = 返品・交換・修理などアフターサービス関連
      complaint   = 苦情・クレーム
      inquiry     = 上記以外の問い合わせ
    ユーザーへ状況を伝えるときは status_label（日本語）を使う。"""
    if settings.demo_ticket_delay_seconds > 0:
        # 受け入れ確認 6: 書き込み系ツールの timeout を実際に見せるための遅延。
        # 既定は 0 なので通常は 1 行も効かない
        await asyncio.sleep(settings.demo_ticket_delay_seconds)
    ticket_no = await repository.create_ticket(conversation_id, description, ticket_type)
    return {
        "ticket_no": ticket_no,
        "status": "escalated",
        "status_label": labels.label(labels.CONVERSATION_STATUS, "escalated"),
    }


registry.register(registry.spec_from_langchain_tool(
    create_ticket, source="builtin", inject_conversation=True))
