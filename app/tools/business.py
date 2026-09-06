"""Mock business tools for querying orders, products, and logistics."""
import random
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field, field_validator

from app.core import labels, retrieval
from app.db import repository


class OrderInput(BaseModel):
    order_id: str = Field(description="注文番号。例: 1001")


class ProductInput(BaseModel):
    product_name: str = Field(description="商品名またはキーワード。例: キャットフード")


class LogisticsInput(BaseModel):
    order_id: str = Field(description="注文番号。対象注文の配送履歴を確認するために使用")


@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> dict:
    """注文のステータス、金額、注文日時、商品名を確認する。ユーザーが特定の注文について質問した場合に使用する。"""
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": rng.choice(["支払い待ち", "支払い済み", "発送済み", "配達完了"]),
        "amount": rng.randint(50, 2000),
        "created_at": f"2026-07-{rng.randint(1, 12):02d} 10:00",
        "product": rng.choice(["自動猫トイレ", "キャットフード 5kg", "キャットタワー", "自動給水器"]),
    }


@tool(args_schema=ProductInput)
async def query_product(product_name: str) -> dict:
    """商品の価格、在庫、仕様を確認する。ユーザーが商品の在庫や価格を尋ねた場合に使用する。"""
    rng = random.Random(f"product:{product_name}")
    return {
        "product_name": product_name,
        "price": rng.randint(20, 999),
        "stock": rng.randint(0, 500),
        "spec": rng.choice(["標準パック", "ファミリーパック", "お試しパック"]),
    }


@tool(args_schema=LogisticsInput)
async def query_logistics(order_id: str) -> dict:
    """注文の配送状況、現在地、配送履歴を確認する。ユーザーが荷物の現在地や配送状況を尋ねた場合に使用する。"""
    rng = random.Random(f"logistics:{order_id}")
    status = rng.choice(["集荷済み", "輸送中", "配達中", "配達完了"])
    city = rng.choice(["東京", "横浜", "名古屋", "大阪", "福岡"])
    return {
        "order_id": order_id,
        "status": status,
        "location": f"{city}配送センター",
        "timeline": [f"{city}配送センターから発送", f"現在の状態: {status}"],
    }


class FaqInput(BaseModel):
    keyword: str = Field(min_length=1, description="FAQ を検索するためのキーワード。例:『返品』『発送までの目安』")

    # min_lengthはOpenAPIスキーマのminLengthとして表出させるために残し、
    # 空白のみの値(min_lengthを通過してしまう)はこのvalidatorで拒否する
    @field_validator("keyword")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("空白のみの値は許可されない")
        return v


@tool(args_schema=FaqInput)
async def query_faq(keyword: str) -> dict:
    """ナレッジベースを検索する。ポリシー、ルール、操作方法などの一般的な質問に加えて、
    商品の仕様・型番・機能・マニュアルの内容についてもこのツールで調べる
    （例:「ロボット掃除機の吸引力」「EC-RV300 の稼働時間」「静かなキーボードはあるか」）。
    在庫数と価格そのものを知りたい場合だけ query_product を使う。"""
    hits = await retrieval.search_knowledge(keyword)
    if not hits:
        return {"hits": [], "message": f"「{keyword}」に関連するFAQが見つかりませんでした"}
    return {"hits": [{"question": h["question"], "answer": h["answer"]} for h in hits]}


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
    ticket_no = await repository.create_ticket(conversation_id, description, ticket_type)
    return {
        "ticket_no": ticket_no,
        "status": "escalated",
        "status_label": labels.label(labels.CONVERSATION_STATUS, "escalated"),
    }
