from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.core.memory import (
    SessionStore,
    build_window,
    summary_line,
    summary_system,
    trim_history,
)


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


# ---------------------------------------------------------------------------
# 07 スライディングウィンドウ(要約の境界で切り、token 上限で刈る)
# ---------------------------------------------------------------------------


def _dialog(n_turns: int, start_db_id: int = 1) -> list:
    """n ターンの会話。user の発話にだけ db-id の anchor を付ける
    (1 ターンで id を 2 つ使う: user / assistant)。"""
    msgs = []
    db_id = start_db_id
    for i in range(n_turns):
        msgs.append(HumanMessage(f"質問{i}", id=f"db-{db_id}"))
        msgs.append(AIMessage(f"回答{i}"))
        db_id += 2
    return msgs


def test_build_window_cuts_at_anchor():
    msgs = _dialog(10)                       # user の anchor: db-1,3,...,19
    win = build_window(msgs, summary_upto_msg_id=8, max_tokens=100000)
    assert win[0].id == "db-9"               # 境界より後ろの最初の user 発話
    assert len(win) == 12                    # 5〜10 ターン目の 6 ターン
    assert win[-1] == msgs[-1]


def test_build_window_no_boundary_keeps_all():
    msgs = _dialog(3)
    assert build_window(msgs, summary_upto_msg_id=0, max_tokens=100000) == msgs


def test_build_window_anchor_missing_degrades_to_trim():
    """07 より前からある会話には anchor が無い。切らずに token 上限だけを掛ける。"""
    msgs = [HumanMessage("古い発話には anchor が無い"), AIMessage("回答")] * 3
    win = build_window(list(msgs), summary_upto_msg_id=99, max_tokens=100000)
    assert len(win) == 6


def test_build_window_cuts_from_the_first_anchored_message():
    """途中から anchor が付いた会話。anchor のある最初の発話から窓を始める。

    07 の導入前後をまたぐ会話がこの形になる。前半に anchor が無いからといって
    「anchor 無し」と決めつけて全部を渡すと、要約済みの区間が窓にも重複して載る。
    """
    legacy = [HumanMessage("導入前の質問"), AIMessage("導入前の回答")]
    msgs = legacy + _dialog(3, start_db_id=11)   # user の anchor: db-11,13,15
    win = build_window(msgs, summary_upto_msg_id=10, max_tokens=100000)
    assert win[0].id == "db-11"
    assert len(win) == 6


def test_build_window_all_summarized_still_returns_messages():
    """境界より後ろに user 発話が 1 件も無くても空を返さない。

    要約が全区間を覆った直後がこの状態になる。空を渡すとモデルは
    「いま聞かれたこと」すら見ずに答えることになる。
    """
    msgs = _dialog(3)                        # user の anchor: db-1,3,5
    win = build_window(msgs, summary_upto_msg_id=100, max_tokens=100000)
    assert win == msgs


def test_build_window_token_cap_still_applies():
    msgs = _dialog(20)
    for m in msgs:
        m.content = m.content + "ニャ" * 200
    win = build_window(msgs, summary_upto_msg_id=0, max_tokens=300)
    assert 0 < len(win) < len(msgs)
    assert win[0].type == "human"


def test_build_window_never_returns_empty():
    msgs = [HumanMessage("非常に長い" + "ニャ" * 5000, id="db-1")]
    win = build_window(msgs, summary_upto_msg_id=0, max_tokens=10)
    assert win == msgs                       # 刈った結果が空なら上限を超えていても元の窓を返す


def test_build_window_does_not_mutate_the_input_list():
    """State の履歴そのものを触らないこと。ここで刈ってしまうと、checkpoint の
    全履歴が二度と戻せない。"""
    msgs = _dialog(20)
    before = list(msgs)
    build_window(msgs, summary_upto_msg_id=10, max_tokens=50)
    assert msgs == before and len(msgs) == 40


def test_summary_helpers():
    assert summary_line(None) == "" and summary_line("") == ""
    assert "注文1001" in summary_line("ユーザーは注文1001について問い合わせた")
    assert summary_system(None) is None
    ss = summary_system("ユーザーは注文1001について問い合わせた")
    assert isinstance(ss, SystemMessage) and "注文1001" in ss.content
