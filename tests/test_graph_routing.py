"""graph の分岐関数(routing / gate / stop condition)。

分岐は graph の骨格そのもので、間違えると「返金の質問が knowledge へ入らない」
「ReAct が止まらない」のような、ユーザーから見える壊れ方をする。
node の中身と違って純粋関数なので、上流を呼ばずにここで固めておく。
"""

import pytest
from langchain_core.messages import AIMessage

from app.config import settings
from app.graph.routing import (
    INTENT_TO_ROUTE,
    confidence_gate,
    route_by_intent,
    should_continue,
)


@pytest.mark.parametrize("intent,expect", [
    ("苦情", "escalate"),
    ("雑談", "fallback_script"),
    ("その他", "fallback_script"),
    ("商品相談", "knowledge"),
    ("返金返品", "refund_flow"),
    ("アフターサービス", "refund_flow"),
    ("配送", "business"),
    ("注文", "business"),
])
def test_route_by_intent_five_outlets(intent, expect):
    assert route_by_intent({"intent": intent}) == expect


def test_route_by_intent_unknown_defaults_business():
    """未知の分類は Agent が拾い直せる business へ。固定文の出口へ倒すと会話が終わる。"""
    assert route_by_intent({"intent": "未知"}) == "business"
    assert route_by_intent({}) == "business"


def test_intent_to_route_covers_eight_classes():
    """8 分類ちょうど。増減したら分類器と routing のどちらかが片方だけ変わっている。"""
    assert set(INTENT_TO_ROUTE) == {
        "苦情", "雑談", "その他", "商品相談", "返金返品", "アフターサービス", "配送", "注文"}


def test_the_five_outlets_are_exactly_these():
    """出口の集合を固定する。build.py の conditional edge の mapping key と一対一。"""
    assert set(INTENT_TO_ROUTE.values()) == {
        "escalate", "fallback_script", "knowledge", "refund_flow", "business"}


def test_confidence_gate():
    assert confidence_gate({"evidence_strong": True}) == "strong"
    assert confidence_gate({"evidence_strong": False}) == "weak"
    assert confidence_gate({}) == "weak"


def test_should_continue_stops_when_no_tool_calls():
    state = {"messages": [AIMessage("回答完了")], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "stop"


def test_should_continue_continues_on_tool_calls_under_budget():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "continue"


def test_should_continue_stops_on_step_cap():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": settings.max_agent_steps, "tokens_used": 0}
    assert should_continue(state) == "stop"


def test_token_cost_is_not_a_stop_condition():
    # loop は step 数だけで上限を設ける。token は trace / 09 集計用で判断には使わない
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": 1, "tokens_used": 10_000_000}
    assert should_continue(state) == "continue"


def test_should_continue_stops_when_messages_empty():
    # messages が空のまま呼ばれても graph 全体を落とさない。
    # 直前の AIMessage が無い = tool_calls も無いので、収束と同じ "stop" が正しい出口
    assert should_continue({"messages": [], "steps": 0, "tokens_used": 0}) == "stop"
    assert should_continue({}) == "stop"
