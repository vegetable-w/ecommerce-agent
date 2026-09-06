"""Tests for mock business tools (query_order, query_product, query_logistics)."""
import pathlib
import re

import pytest

from app.tools import business
from app.tools.business import query_logistics, query_order, query_product


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

    カタログを business.py に写しているのは query_product を deterministic に保つため。
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
    assert business.PRODUCT_CATALOGUE == found


@pytest.mark.asyncio
async def test_query_logistics_deterministic_and_shaped():
    """query_logistics must return reproducible output for the same order_id."""
    lg1 = await query_logistics.ainvoke({"order_id": "1001"})
    lg2 = await query_logistics.ainvoke({"order_id": "1001"})

    # Same seed => same output
    assert lg1 == lg2

    # Correct shape and valid values
    assert lg1["order_id"] == "1001"
    assert lg1["status"] in {"集荷済み", "輸送中", "配達中", "配達完了"}
    assert isinstance(lg1["location"], str)
    assert "配送センター" in lg1["location"]
    assert isinstance(lg1["timeline"], list)


@pytest.mark.asyncio
async def test_query_product_and_logistics_names():
    """All tools must have correct .name attributes."""
    assert query_product.name == "query_product"
    assert query_logistics.name == "query_logistics"
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

    lg1 = await query_logistics.ainvoke({"order_id": "1001"})
    lg2 = await query_logistics.ainvoke({"order_id": "1002"})
    assert lg1 != lg2


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
        "created_at": "2026-07-04 10:00",
        "product": "自動猫トイレ",
    }
    assert result == expected


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


@pytest.mark.asyncio
async def test_query_logistics_golden_value():
    """
    Golden value test for query_logistics to protect Task 14's eval reproducibility.

    This literal is pinned for eval reproducibility. Changing it will invalidate
    recorded eval results. If the seeding logic must change, update this value
    after verifying the new output is intentional.
    """
    result = await query_logistics.ainvoke({"order_id": "1001"})
    expected = {
        "order_id": "1001",
        "status": "集荷済み",
        "location": "横浜配送センター",
        "timeline": ["横浜配送センターから発送", "現在の状態: 集荷済み"],
    }
    assert result == expected
