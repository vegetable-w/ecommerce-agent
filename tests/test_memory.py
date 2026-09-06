from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.core.memory import SessionStore, trim_history


def test_store_get_unknown_session_returns_empty():
    assert SessionStore().get("nope") == []


def test_store_append_and_get_isolated_by_session():
    store = SessionStore()
    store.append("a", HumanMessage("hi"), AIMessage("hello"))
    store.append("b", HumanMessage("こんにちは"))
    assert len(store.get("a")) == 2
    assert len(store.get("b")) == 1


def test_trim_keeps_recent_and_starts_on_human():
    msgs = []
    for i in range(20):
        msgs.append(HumanMessage(f"質問{i}:" + "にゃー" * 50))
        msgs.append(AIMessage(f"回答{i}:" + "にゃー" * 50))
    trimmed = trim_history(msgs, max_tokens=200)
    assert 0 < len(trimmed) < len(msgs)
    assert trimmed[0].type == "human"
    assert trimmed[-1] == msgs[-1]


def test_trim_noop_when_under_budget():
    msgs = [HumanMessage("hi"), AIMessage("hello")]
    assert trim_history(msgs, max_tokens=2000) == msgs


def test_trim_drops_leading_ai_message():
    msgs = [AIMessage("先頭のAI発話"), HumanMessage("質問"), AIMessage("回答")]
    trimmed = trim_history(msgs, max_tokens=2000)
    assert trimmed[0].type == "human"
    assert trimmed == msgs[1:]


def test_trim_single_human_message_survives():
    msgs = [HumanMessage("hi")]
    assert trim_history(msgs, max_tokens=2000) == msgs


def test_trim_budget_exactly_equal_to_history_tokens_keeps_all():
    msgs = [HumanMessage("hi"), AIMessage("hello")]
    exact_budget = count_tokens_approximately(msgs)
    assert trim_history(msgs, max_tokens=exact_budget) == msgs


def test_store_clear_one_session():
    store = SessionStore()
    store.append("a", HumanMessage("hi"))
    store.append("b", HumanMessage("hello"))
    store.clear("a")
    assert store.get("a") == []
    assert len(store.get("b")) == 1


def test_store_clear_all_sessions():
    store = SessionStore()
    store.append("a", HumanMessage("hi"))
    store.append("b", HumanMessage("hello"))
    store.clear()
    assert store.get("a") == []
    assert store.get("b") == []
