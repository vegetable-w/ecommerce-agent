"""graph の終端 node(log_node)と、回答文を 1 つに決める resolve_answer。

4 つの出口はすべてここを通って END へ行くので、この node が落ちると
「回答は作れたのに履歴に残らない」あるいは「graph ごと止まる」のどちらかになる。
State の形が経路によって違う(決定的な node は answer、Agent は messages 末尾)ため、
resolve_answer は**どの経路から来ても文字列を返しきる**ことが要件になる。

append_message は必ず差し替える(support は本番相当の DB であり、テストから書き込まない)。
既定では「呼ばれたら落ちる」に固定し、保存を確かめるテストだけが記録用の偽物を置く。
"""

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph import nodes


@pytest.fixture(autouse=True)
def _forbid_db_write(monkeypatch):
    """DB への保存を既定で禁止する。差し替え忘れたまま本番相当の DB を叩く事故を仕組みで止める。"""

    async def _boom(*args, **kwargs):
        raise AssertionError("テストが本物の append_message を呼んだ")

    monkeypatch.setattr(nodes.repository, "append_message", _boom)


def _record(monkeypatch) -> list:
    """append_message の呼び出しを記録するだけの偽物に差し替え、記録先の list を返す。"""
    calls = []

    async def _fake(conversation_id, role, content=None, tool_calls=None, tool_call_id=None):
        calls.append({"conversation_id": conversation_id, "role": role, "content": content})
        return len(calls)

    monkeypatch.setattr(nodes.repository, "append_message", _fake)
    return calls


def _ai_with_tool_call(content="") -> AIMessage:
    return AIMessage(content=content,
                     tool_calls=[{"name": "query_order", "args": {"order_id": "1001"},
                                  "id": "c1"}])


# --- resolve_answer -----------------------------------------------------------


def test_resolve_answer_prefers_the_state_answer():
    """決定的な出口(chitchat / complaint / fallback)は answer をそのまま使う。

    messages 末尾に前 turn の AIMessage が残っていても、今回の回答はこちら。
    checkpointer が履歴を持ち回る以上、末尾を先に見ると前 turn の回答を
    今回の回答として保存してしまう。
    """
    state = {"answer": nodes.CHITCHAT_REPLY,
             "messages": [HumanMessage("こんにちは"), AIMessage("前の turn の回答")]}
    assert nodes.resolve_answer(state) == nodes.CHITCHAT_REPLY


def test_resolve_answer_falls_back_to_the_last_ai_message():
    """Agent 経路は answer を書かない(token を stream するため)ので、末尾の AIMessage を使う。"""
    state = {"messages": [HumanMessage("注文1001は?"), AIMessage("配送中です")]}
    assert nodes.resolve_answer(state) == "配送中です"


def test_resolve_answer_is_empty_without_any_answer():
    """answer も AIMessage も無ければ空文字。例外にしない。

    ここで落とすと、回答の保存に失敗しただけの turn が graph 全体の停止に化ける。
    """
    assert nodes.resolve_answer({}) == ""
    assert nodes.resolve_answer({"messages": [HumanMessage("こんにちは")]}) == ""


def test_resolve_answer_flattens_list_content_blocks():
    """content が list(マルチモーダルのブロック)でも文字列を返すこと。

    上流やモデルによって content は str か block の list かが変わる。list のまま
    repository へ渡すと、保存時ではなく DB の型で落ちる。
    """
    ai = AIMessage(content=[{"type": "text", "text": "配送中"},
                            {"type": "text", "text": "です"}])
    assert nodes.resolve_answer({"messages": [ai]}) == "配送中です"


def test_resolve_answer_ignores_non_text_blocks():
    """text を持たないブロック(思考ブロック等)や dict でない要素が混ざっても落ちないこと。"""
    ai = AIMessage(content=[{"type": "thinking", "thinking": "内部の思考"},
                            "生の文字列", {"type": "text", "text": "本文"}])
    assert nodes.resolve_answer({"messages": [ai]}) == "本文"


def test_resolve_answer_skips_the_tool_call_only_ai_message():
    """本文が空で tool_calls だけの AIMessage を飛ばし、その手前の本文を拾うこと。

    ReAct loop では末尾の AIMessage が「本文なし + tool_calls あり」になる瞬間があり、
    should_continue が steps 上限で打ち切ると、その State のまま log へ来る。
    末尾だけを見ると回答が空文字になり、ユーザーに何か返したのに履歴には
    何も残らない turn ができる。
    """
    state = {"messages": [HumanMessage("注文1001は?"),
                          AIMessage("お調べします"),
                          _ai_with_tool_call(),
                          ToolMessage(content='{"ok": true}', tool_call_id="c1",
                                      name="query_order"),
                          _ai_with_tool_call()]}
    assert nodes.resolve_answer(state) == "お調べします"


def test_resolve_answer_keeps_the_text_of_an_ai_message_that_also_calls_tools():
    """本文と tool_calls を両方持つ AIMessage は、本文をそのまま使うこと。"""
    state = {"messages": [_ai_with_tool_call("確認のため注文を照会します")]}
    assert nodes.resolve_answer(state) == "確認のため注文を照会します"


def test_resolve_answer_is_empty_when_every_ai_message_is_tool_calls_only():
    """本文を持つ AIMessage が 1 つも無ければ空文字。探し続けて落ちないこと。"""
    state = {"messages": [HumanMessage("注文1001は?"), _ai_with_tool_call()]}
    assert nodes.resolve_answer(state) == ""


# --- log_node -----------------------------------------------------------------


async def test_log_node_saves_the_assistant_message(monkeypatch):
    calls = _record(monkeypatch)
    out = await nodes.log_node({"conversation_id": 7, "answer": "承知しました",
                                "route": "chitchat", "intent": "雑談"})

    assert calls == [{"conversation_id": 7, "role": "assistant", "content": "承知しました"}]
    # State は書き換えない。ここは観測と保存だけの終端 node
    assert out == {}


async def test_log_node_saves_the_agent_answer_from_the_messages(monkeypatch):
    """Agent 経路(answer 無し)でも、末尾の AIMessage の本文が保存されること。"""
    calls = _record(monkeypatch)
    await nodes.log_node({"conversation_id": 9,
                          "messages": [HumanMessage("注文1001は?"), AIMessage("配送中です")]})
    assert calls[0]["content"] == "配送中です"


async def test_log_node_does_not_save_without_a_conversation_id(monkeypatch):
    """conversation_id が無い State では保存しない。

    append_message は差し替え済みでも autouse fixture が「呼ばれたら落ちる」ままなので、
    呼べばこのテストが落ちる。0 は「未採番」であって会話 1 番ではない。
    """
    await nodes.log_node({"answer": "承知しました"})
    await nodes.log_node({"conversation_id": 0, "answer": "承知しました"})
    await nodes.log_node({"conversation_id": None, "answer": "承知しました"})


async def test_log_node_saves_null_content_for_an_empty_answer(monkeypatch):
    """回答が空のときは空文字ではなく NULL で残すこと。

    空文字で保存すると、履歴を読み直す側(_build_history)に中身の無い assistant 行が
    毎 turn 混ざる。「回答が無かった」ことは NULL で表す。
    """
    calls = _record(monkeypatch)
    await nodes.log_node({"conversation_id": 7, "messages": [HumanMessage("こんにちは")]})
    assert calls[0]["content"] is None


async def test_log_node_logs_the_trace(monkeypatch, caplog):
    """observability の本体。intent / route / trace がログに出ること。"""
    _record(monkeypatch)
    with caplog.at_level(logging.INFO, logger=nodes.logger.name):
        await nodes.log_node({"conversation_id": 7, "intent": "配送", "route": "business",
                              "trace": {"forced_rag": True, "confidence": "strong"},
                              "messages": [AIMessage("配送中です")]})

    text = caplog.text
    assert "business" in text and "配送" in text
    assert "forced_rag" in text and "strong" in text


async def test_log_node_logs_even_for_a_bare_state(monkeypatch):
    """key が揃っていない State でも整形で落ちないこと(ログのために graph を止めない)。"""
    _record(monkeypatch)
    assert await nodes.log_node({"conversation_id": 7}) == {}
