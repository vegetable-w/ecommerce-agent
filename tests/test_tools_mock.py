"""Tests for mock business tools (query_order, query_product)。

08 章で query_logistics は built-in から外し、配送状況の照会は MCP 側へ移した。
その分のテストはここには無い(MCP サーバを足す章で改めて書く)。
"""
import pathlib
import re

import pytest

from app.tools.builtin import orders
from app.tools.builtin.orders import query_order, query_product


@pytest.mark.asyncio
async def test_query_order_deterministic_and_shaped():
    """query_order must return reproducible output for the same order_id."""
    r1 = await query_order.ainvoke({"order_id": "1001"})
    r2 = await query_order.ainvoke({"order_id": "1001"})

    # Same seed => same output (determinism within process)
    assert r1 == r2

    # Correct shape and valid values
    assert r1["order_id"] == "1001"
    assert r1["status"] in {"支払い待ち", "支払い済み", "発送済み", "配達完了"}
    assert isinstance(r1["amount"], int)
    assert 50 <= r1["amount"] <= 2000
    assert isinstance(r1["created_at"], str)
    assert isinstance(r1["product"], str)
    assert isinstance(r1["tracking_no"], str)
    assert r1["tracking_no"]


@pytest.mark.asyncio
async def test_query_product_deterministic_and_shaped():
    """query_product must return reproducible output for the same product."""
    p1 = await query_product.ainvoke({"product_name": "メカニカルキーボード"})
    p2 = await query_product.ainvoke({"product_name": "メカニカルキーボード"})

    # Same seed => same output
    assert p1 == p2

    # Correct shape and valid values
    assert p1["found"] is True
    assert p1["product_name"] == "メカニカルキーボード"
    assert p1["model_number"] == "EC-KB20"
    assert isinstance(p1["price"], int)
    assert 20 <= p1["price"] <= 999
    assert isinstance(p1["stock"], int)
    assert 0 <= p1["stock"] <= 500


@pytest.mark.asyncio
async def test_query_product_seeds_on_model_number_not_wording():
    """同じ商品を別の言い方で聞いても、価格と在庫は同じであること。

    以前は問い合わせ文字列そのものを seed にしていたため、「キーボード」と
    「静かなキーボード」で違う価格が返っていた。同じ商品なのに聞き方で値段が変わるのは、
    mock であってもユーザーから見れば矛盾した回答になる。
    """
    a = await query_product.ainvoke({"product_name": "静かなキーボード"})
    b = await query_product.ainvoke({"product_name": "EC-KB20 の在庫"})
    assert a["model_number"] == b["model_number"] == "EC-KB20"
    assert (a["price"], a["stock"]) == (b["price"], b["stock"])


@pytest.mark.asyncio
async def test_query_product_refuses_unstocked_items():
    """取り扱いのない商品に価格と在庫をでっち上げないこと。

    以前は商品名を seed にして必ず price/stock を返していたため、「宇宙船」にも
    値段と在庫が付いた。mock の粗さではなくユーザーへ嘘をつく挙動なので、
    カタログに無いものは在庫も価格も返さない。
    """
    r = await query_product.ainvoke({"product_name": "宇宙船"})
    assert r["found"] is False
    assert "price" not in r and "stock" not in r
    assert "取り扱いを確認できませんでした" in r["message"]


def test_product_catalogue_matches_the_knowledge_document():
    """カタログの写しが原本(商品仕様マニュアル)とずれていないこと。

    カタログを builtin/orders.py に写しているのは query_product を deterministic に保つため。
    写しである以上ずれるので、原本を parse して機械的に突き合わせる
    (02 章で labels.py と DDL を突き合わせたのと同じ方式)。
    """
    doc = pathlib.Path("data/kb/product-spec-manual.md").read_text(encoding="utf-8")
    # 見出しの形式: "## ロボット掃除機 Standard（型番 EC-RV100）"
    found = dict(
        (m.group(2), m.group(1).strip())
        for m in re.finditer(r"^##\s+(.+?)（型番\s*([A-Z0-9-]+)）\s*$", doc, re.M)
    )
    assert found, "原本から型番を 1 件も抽出できていない(見出し書式が変わった可能性)"
    assert orders.PRODUCT_CATALOGUE == found


@pytest.mark.asyncio
async def test_query_order_tracking_no_is_deterministic_per_order():
    """追跡番号は注文ごとに一意で、同じ注文なら何度引いても同じであること。

    追跡番号は配送状況を照会する唯一の入口なので、呼ぶたびに変わると
    「注文を引く → その追跡番号で配送を引く」という手順そのものが成立しなくなる。
    """
    a = await query_order.ainvoke({"order_id": "1001"})
    b = await query_order.ainvoke({"order_id": "1001"})
    c = await query_order.ainvoke({"order_id": "1002"})
    assert a["tracking_no"] == b["tracking_no"]
    assert a["tracking_no"] != c["tracking_no"]


@pytest.mark.asyncio
async def test_query_product_and_order_names():
    """All tools must have correct .name attributes."""
    assert query_product.name == "query_product"
    assert query_order.name == "query_order"


@pytest.mark.asyncio
async def test_different_inputs_produce_different_outputs():
    """Different inputs should (almost certainly) produce different outputs."""
    o1 = await query_order.ainvoke({"order_id": "1001"})
    o2 = await query_order.ainvoke({"order_id": "1002"})
    assert o1 != o2

    p1 = await query_product.ainvoke({"product_name": "キャットフード"})
    p2 = await query_product.ainvoke({"product_name": "キャットタワー"})
    assert p1 != p2


@pytest.mark.asyncio
async def test_query_order_golden_value():
    """
    Golden value test for query_order to protect Task 14's eval reproducibility.

    This literal is pinned for eval reproducibility. Changing it will invalidate
    recorded eval results. If the seeding logic must change, update this value
    after verifying the new output is intentional.
    """
    result = await query_order.ainvoke({"order_id": "1001"})
    expected = {
        "order_id": "1001",
        "status": "支払い済み",
        "amount": 1739,
        "product": "自動猫トイレ",
        "tracking_no": "JP213502378238",
    }
    assert {k: v for k, v in result.items() if k != "created_at"} == expected

    # 注文日だけは「今日から何日前か」で決まる。固定の日付にすると、時間が経つほど
    # 全注文が古くなり、規約の「受取後 7 日以内」を満たす注文が 1 件も作れなくなる
    # (返品可能と判断される経路に到達できなくなる)。決定的なのは日付そのものではなく
    # **経過日数**なので、そちらを固定する。
    from datetime import datetime

    ordered = datetime.strptime(result["created_at"], "%Y-%m-%d %H:%M")
    assert (datetime.now().date() - ordered.date()).days == 13


@pytest.mark.asyncio
async def test_query_product_golden_value():
    """
    Golden value test for query_product to protect Task 14's eval reproducibility.

    This literal is pinned for eval reproducibility. Changing it will invalidate
    recorded eval results. If the seeding logic must change, update this value
    after verifying the new output is intentional.
    """
    result = await query_product.ainvoke({"product_name": "メカニカルキーボード"})
    expected = {
        "product_name": "メカニカルキーボード",
        "model_number": "EC-KB20",
        "found": True,
        "price": 468,
        "stock": 265,
    }
    assert result == expected
