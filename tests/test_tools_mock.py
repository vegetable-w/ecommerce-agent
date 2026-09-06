"""Tests for mock business tools (query_order, query_product, query_logistics)."""
import pytest

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
    """query_product must return reproducible output for the same product name."""
    p1 = await query_product.ainvoke({"product_name": "キャットフード"})
    p2 = await query_product.ainvoke({"product_name": "キャットフード"})

    # Same seed => same output
    assert p1 == p2

    # Correct shape and valid values
    assert p1["product_name"] == "キャットフード"
    assert isinstance(p1["price"], int)
    assert 20 <= p1["price"] <= 999
    assert isinstance(p1["stock"], int)
    assert 0 <= p1["stock"] <= 500
    assert p1["spec"] in {"標準パック", "ファミリーパック", "お試しパック"}


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
    result = await query_product.ainvoke({"product_name": "キャットフード"})
    expected = {
        "product_name": "キャットフード",
        "price": 641,
        "stock": 287,
        "spec": "お試しパック",
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
