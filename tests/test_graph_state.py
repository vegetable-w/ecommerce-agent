"""ConversationState と trace reducer を固定する。

State は graph 全体を貫くので、field を 1 つ落とすと「その値を書いたつもりの node」と
「読むつもりの node」が静かにすれ違う。LangGraph は未知の key を黙って捨てるため、
例外も警告も出ない。だから key の集合をここで固定する。
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.graph.state import ConversationState, merge_dict


def test_merge_dict_accumulates():
    assert merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dict(None, {"b": 2}) == {"b": 2}
    assert merge_dict({"a": 1}, None) == {"a": 1}
    assert merge_dict({"a": 1}, {"a": 9}) == {"a": 9}      # 同じ key は後勝ち


def test_merge_dict_does_not_mutate_its_inputs():
    """reducer は毎 step 呼ばれる。片方を書き換えると、前の step の trace が後から変わる。"""
    a = {"a": 1}
    b = {"b": 2}
    merge_dict(a, b)
    assert a == {"a": 1} and b == {"b": 2}


def test_add_messages_reducer_appends():
    merged = add_messages([HumanMessage("hi")], [AIMessage("yo")])
    assert [m.content for m in merged] == ["hi", "yo"]


def test_state_has_required_keys():
    keys = ConversationState.__annotations__
    for k in ["messages", "user_id", "conversation_id", "intent", "route",
              "evidence", "citations", "evidence_strong", "answer",
              "steps", "tokens_used", "suggested_actions", "trace"]:
        assert k in keys
