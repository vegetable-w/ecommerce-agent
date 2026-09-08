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

import logging
import re

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


# --- model_ctx のログ ------------------------------------------------------------
#
# 受け入れ検証のための覗き窓(spec §「Observability」)。長い会話で文脈が保たれて
# いることを人が確かめるには、そのターンでモデルへ実際に何を渡したかが見える必要が
# ある。**観測のためだけの関数なので、ここで落ちてターンを壊してはいけない。**


def test_log_model_context_shows_summary_and_window(caplog):
    state = _state(n_turns=6,
                   summary="ユーザーは注文1001について問い合わせ、電話番号13800138000を伝えた",
                   upto=6)
    msgs = nodes._agent_messages(state)
    with caplog.at_level(logging.INFO, logger="app.graph.nodes"):
        nodes._log_model_context(state, msgs)

    assert "model_ctx" in caplog.text and "注文1001" in caplog.text
    assert "[human] '質問3" in caplog.text     # 窓の各 message が 1 行ずつ見える
    assert "質問0" not in caplog.text          # 境界より前の原文は context にもログにも出ない


def test_log_model_context_reports_the_window_size_and_tokens(caplog):
    """件数と概算 token。膨らんでいく様子をログだけで追えるようにする。"""
    state = _state(n_turns=3)
    with caplog.at_level(logging.INFO, logger="app.graph.nodes"):
        nodes._log_model_context(state, nodes._agent_messages(state))

    assert "window=6" in caplog.text          # system は窓の件数に数えない
    assert re.search(r"tokens≈[1-9]\d*", caplog.text)


def test_log_model_context_does_not_dump_the_whole_system_prompt(caplog):
    """各 message は先頭だけ。全文を出すと 1 ターンでログが AGENT_SYSTEM で埋まる。"""
    state = _state(n_turns=1)
    with caplog.at_level(logging.INFO, logger="app.graph.nodes"):
        nodes._log_model_context(state, nodes._agent_messages(state))

    assert len(AGENT_SYSTEM) > 200
    assert AGENT_SYSTEM not in caplog.text
    assert AGENT_SYSTEM[:20] in caplog.text   # 何が入っているかは分かる


def test_log_model_context_keeps_the_summary_in_full(caplog):
    """要約だけは全文。ここを切ると「何が失われたか」を後から追えない。"""
    summary = "ユーザーは注文1001の配送を問い合わせ、" + "電話番号13800138000を伝えた。" * 6
    state = _state(n_turns=2, summary=summary, upto=0)
    with caplog.at_level(logging.INFO, logger="app.graph.nodes"):
        nodes._log_model_context(state, nodes._agent_messages(state))
    assert summary in caplog.text


def test_log_model_context_never_breaks_the_turn(caplog):
    """観測のための関数なので、壊れた入力でも例外を投げない。

    ここで送出すると、ログの都合でユーザーのターンが落ちる。
    """
    class _Broken:
        @property
        def content(self):
            raise RuntimeError("読めない message")

    with caplog.at_level(logging.INFO, logger="app.graph.nodes"):
        nodes._log_model_context({"messages": None}, [_Broken()])
        nodes._log_model_context(None, None)


async def test_agent_llm_logs_the_context_it_sends(monkeypatch):
    """agent_llm が呼ぶこと。呼ばれなければ受け入れ検証の覗き窓が塞がる。

    **モデルへ渡した並びそのものを記録する**(組み立て直さない)。別々に作ると、
    ログには出ていない message がモデルへ届く、という状態を検出できなくなる。
    """
    sent, logged = [], []

    class _Model:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, msgs, config=None):
            sent.append(msgs)
            return AIMessage("はい")

    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: _Model())
    monkeypatch.setattr(nodes, "_log_model_context",
                        lambda state, msgs: logged.append(msgs))

    await nodes.agent_llm(_state(n_turns=2))

    assert len(logged) == 1
    assert logged[0] is sent[0]
