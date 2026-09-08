"""07 章の文脈組み立て — _agent_messages と _history_text。

確かめる中心は「**並びが固定であること**」の 1 点に尽きる(spec §3)。

- system は常に AGENT_SYSTEM。可変の内容を連結しない。連結するとターンごとに
  prompt の先頭が変わり、prefix cache が毎回 miss する。
- このターンの材料(evidence / 注文 / 返金の判断指示)は、**最後の HumanMessage の
  直後**に置く。末尾に置くと、ReAct loop の 2 周目で材料が AI の発話より後ろへ回り、
  step 1 の prompt が step 2 の prefix でなくなる。
- 窓は要約の境界で切る。切らずに全履歴を渡すと、要約と原文が二重に載る。

上流にも DB にも触らない(どちらも呼ばない純粋な関数だけを見る)。
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.prompts import AGENT_SYSTEM
from app.graph import nodes


def _state(n_turns: int = 3, summary: str = "", upto: int = 0, **extra) -> dict:
    """anchor 付きの会話。user の発話に db-1, db-3, ... が付く(runtime と同じ形)。"""
    msgs: list = []
    db_id = 1
    for i in range(n_turns):
        msgs.append(HumanMessage(f"質問{i}", id=f"db-{db_id}"))
        msgs.append(AIMessage(f"回答{i}"))
        db_id += 2
    return {"messages": msgs, "summary": summary, "summary_upto_msg_id": upto, **extra}


# --- 並び --------------------------------------------------------------------


def test_agent_messages_order_with_summary():
    """persona の system → 要約の system → 窓の原文、の順で固定。"""
    ms = nodes._agent_messages(_state(summary="ユーザーは注文1001について問い合わせた"))

    assert isinstance(ms[0], SystemMessage)
    assert ms[0].content == AGENT_SYSTEM
    assert isinstance(ms[1], SystemMessage)
    assert "注文1001" in ms[1].content
    assert isinstance(ms[2], HumanMessage) and ms[2].content == "質問0"
    assert ms[-1].content == "回答2"


def test_agent_messages_no_summary_no_extra_system():
    """要約が無ければ 2 番目の system を入れない(見出しだけの空の要約を渡さない)。"""
    ms = nodes._agent_messages(_state(summary=""))
    assert isinstance(ms[0], SystemMessage)
    assert not isinstance(ms[1], SystemMessage)


def test_agent_messages_window_cut_by_boundary():
    """境界より後ろの最初の user 発話から原文を並べる。"""
    ms = nodes._agent_messages(_state(n_turns=6, summary="以前の要約", upto=6))
    win = [m for m in ms if not isinstance(m, SystemMessage)]

    assert win[0].id == "db-7"
    assert all("質問0" != m.content for m in win)


def test_agent_messages_keeps_the_whole_history_when_nothing_is_summarized():
    """要約が無い会話は従来どおり全履歴。境界が 0 のときに切ってはいけない。"""
    state = _state(n_turns=3)
    ms = nodes._agent_messages(state)
    assert [m.content for m in ms[1:]] == [m.content for m in state["messages"]]


# --- system は常に AGENT_SYSTEM ------------------------------------------------
#
# 可変の内容を system prompt 本体へ連結しないこと(spec §3)。コメントは無視できても
# このテストは無視できない。後続章で system に可変情報を入れたらここが赤くなる。


def test_system_message_is_always_the_static_prompt():
    for state in (
        {"route": "business", "messages": [HumanMessage("注文1001は今どこですか")]},
        {"route": "knowledge", "evidence": "[1] 返品: 7日以内",
         "messages": [HumanMessage("返品できますか")]},
        {"route": "refund_flow", "evidence": "[1] 返品: 7日以内",
         "order_data": {"order_id": "1001", "created_at": "2026-09-05 10:00"},
         "messages": [HumanMessage("返品したい")]},
        {"route": "knowledge", "summary": "以前の要約", "evidence": "[1] 返品: 7日以内",
         "messages": [HumanMessage("返品できますか")]},
    ):
        assert nodes._agent_messages(state)[0].content == AGENT_SYSTEM


# --- 材料の位置 ----------------------------------------------------------------


def test_turn_context_follows_the_last_human_message():
    """材料は末尾ではなく、最後の HumanMessage の直後に入る。

    末尾に置くと、ReAct の 2 周目で材料が AI の発話と tool 結果より後ろへ回り、
    step 1 の prompt が step 2 の prefix でなくなる(次のテストが見ている不変条件)。
    """
    q = HumanMessage("返品できますか", id="db-1")
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}])
    tool = ToolMessage(content='{"order_id": "1001"}', tool_call_id="c1", name="query_order")
    ms = nodes._agent_messages({"route": "knowledge", "evidence": "[1] 返品: 7日以内",
                                "messages": [q, ai, tool]})

    assert ms[1].content == "返品できますか"
    assert isinstance(ms[2], SystemMessage) and "[1] 返品: 7日以内" in ms[2].content
    assert ms[-1].content == tool.content          # 材料は末尾ではない
    assert isinstance(ms[-1], ToolMessage)


def test_step1_prompt_is_strict_prefix_of_step2():
    """ReAct loop の 2 周目が 1 周目の厳密な prefix 拡張であること(cache が当たる条件)。"""
    base = {"route": "knowledge", "evidence": "[1] 返品: 7日以内"}
    q = HumanMessage("返品できますか", id="db-1")
    s1 = nodes._agent_messages({**base, "messages": [q]})
    s2 = nodes._agent_messages({**base, "messages": [q, AIMessage("確認しています")]})

    def shape(ms):
        return [(m.type, m.content) for m in ms]

    assert shape(s2)[:len(s1)] == shape(s1)
    assert len(s2) > len(s1)


def test_the_turn_materials_carry_the_evidence_the_order_and_the_elapsed_days():
    """材料の中身は 06 章のまま。届く message が変わるだけで、届かなくなってはいけない。"""
    from datetime import datetime, timedelta

    ordered = datetime.now() - timedelta(days=3)
    ms = nodes._agent_messages({
        "route": "refund_flow",
        "evidence": "[1] 返品: 受取後7日以内は返品可能",
        "order_data": {"order_id": "1001", "created_at": ordered.strftime("%Y-%m-%d %H:%M")},
        "messages": [HumanMessage("返品できますか")]})
    ctx = ms[-1].content

    assert isinstance(ms[-1], SystemMessage)
    assert "[1] 返品: 受取後7日以内は返品可能" in ctx
    assert "1001" in ctx
    assert "submit_refund" in ctx
    assert "経過日数は 3 日" in ctx


def test_no_material_message_when_there_is_nothing_for_this_turn():
    """business route は材料が無い。空の材料 message を足さない。"""
    ms = nodes._agent_messages({"route": "business", "evidence": "",
                                "messages": [HumanMessage("注文1001はどこ")]})
    assert len(ms) == 2
    assert isinstance(ms[1], HumanMessage)


# --- _history_text --------------------------------------------------------------


def test_history_text_prepends_summary_and_windows():
    """要約を先頭に置き、境界より前の原文は入れない。

    max_turns を大きく取るのは、件数の上限で切れたのか窓の境界で切れたのかを
    区別するため。既定の 6 のままだと、窓を通さなくても 質問0 は件数で落ちる。
    """
    text = nodes._history_text(
        _state(n_turns=6, summary="ユーザーは注文1001について問い合わせた", upto=6),
        max_turns=20)

    assert text.startswith("(以前の会話の要約:ユーザーは注文1001")
    assert "質問0" not in text          # 境界より前の原文は載らない
    assert "質問3" in text              # 窓の中の原文は載る


def test_history_text_still_excludes_the_current_utterance():
    """今回の発話は従来どおり履歴に含めない(窓で切っても同じ)。"""
    state = _state(n_turns=3, summary="以前の要約", upto=0)
    state["messages"].append(HumanMessage("いま聞いていること", id="db-7"))
    assert "いま聞いていること" not in nodes._history_text(state)


def test_history_text_no_summary_same_as_before():
    """要約が無ければ 05/06 と同じ出力。"""
    text = nodes._history_text(_state(n_turns=2))
    assert "要約" not in text
    assert text == "ユーザー:質問0\nサポート:回答0\nユーザー:質問1"
