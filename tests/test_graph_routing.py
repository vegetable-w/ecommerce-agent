"""graph の分岐関数(routing / gate / stop condition)。

分岐は graph の骨格そのもので、間違えると「返金の質問が knowledge へ入らない」
「ReAct が止まらない」のような、ユーザーから見える壊れ方をする。
node の中身と違って純粋関数なので、上流を呼ばずにここで固めておく。
"""

from langchain_core.messages import AIMessage

from app.config import settings
from app.graph.routing import confidence_gate, route_by_intent, should_continue


def test_route_maps_seven_to_four():
    cases = {
        "商品相談": "knowledge", "返金返品": "knowledge",
        "配送": "business", "注文": "business", "アフターサービス": "business",
        "苦情": "complaint", "雑談": "chitchat",
    }
    for intent, route in cases.items():
        assert route_by_intent({"intent": intent}) == route


def test_route_unknown_intent_defaults_business():
    assert route_by_intent({"intent": "未知"}) == "business"


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
