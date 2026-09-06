"""Mock business tools for querying orders, products, and logistics."""
import random

from langchain_core.tools import tool
from pydantic import BaseModel, Field

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
    keyword: str = Field(description="FAQ を検索するためのキーワード。例:『返品』『発送までの目安』")


@tool(args_schema=FaqInput)
async def query_faq(keyword: str) -> dict:
    """キーワードで FAQ を検索する。ポリシー、ルール、操作方法などの一般的な質問に使用する。"""
    rows = await repository.search_faq(keyword)
    if not rows:
        return {"hits": [], "message": f"「{keyword}」に関連するFAQが見つかりませんでした"}
    return {"hits": [{"question": r.question, "answer": r.answer} for r in rows]}
