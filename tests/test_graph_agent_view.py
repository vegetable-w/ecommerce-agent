"""app/api/agent.py の _views_from_state — 最終 State から tool trace を組み直す。

/api/agent は評価とテストのための入口なので、「モデルが何を選び、その結果どうだったか」
が読めることが値打ちになる。graph は tool の呼び出しも結果も messages にしか残さない
(State に専用の field は無い)ので、ここが唯一の再構成箇所。

上流も DB も一切触らない。State は手で組んだ messages の list を渡すだけ。
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.api.agent import _views_from_state


def test_views_rebuilt_from_messages():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])
    tm = ToolMessage(content='{"status":"発送済み"}', tool_call_id="c1", name="query_order")
    calls, results = _views_from_state({"messages": [HumanMessage("q"), ai, tm, AIMessage("回答")]})

    assert [c.name for c in calls] == ["query_order"]
    assert calls[0].id == "c1"
    assert calls[0].args == {"order_id": "1001"}
    assert [r.tool_call_id for r in results] == ["c1"]
    assert results[0].name == "query_order"
    assert results[0].ok is True
    assert results[0].content == '{"status":"発送済み"}'


def test_views_keep_every_step_of_a_multi_step_loop():
    """ReAct が 2 周した turn では、両方の呼び出しと両方の結果が順番どおりに並ぶ。

    末尾の AIMessage だけを見る実装(= 最後の 1 手しか残らない)への回帰テスト。
    評価は「注文を引いてから配送を引いたか」を見るので、途中の手が消えると意味を失う。
    """
    state = {
        "messages": [
            HumanMessage("注文1001の配送状況は?"),
            AIMessage("", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]),
            ToolMessage(content="{}", tool_call_id="c1", name="query_order"),
            AIMessage("", tool_calls=[{"name": "query_logistics", "args": {"tracking_no": "JP1"}, "id": "c2"}]),
            ToolMessage(content="{}", tool_call_id="c2", name="query_logistics"),
            AIMessage("輸送中です。"),
        ]
    }
    calls, results = _views_from_state(state)
    assert [c.name for c in calls] == ["query_order", "query_logistics"]
    assert [r.tool_call_id for r in results] == ["c1", "c2"]


def test_views_are_empty_for_a_turn_without_tools():
    """雑談のように tool を1つも使わなかった turn では両方とも空になる。"""
    calls, results = _views_from_state(
        {"messages": [HumanMessage("こんにちは"), AIMessage("こんにちは。ご用件をどうぞ。")]}
    )
    assert calls == []
    assert results == []


def test_views_are_empty_when_state_has_no_messages():
    """決定的な経路(chitchat / complaint)の State には messages が無いこともある。
    KeyError で 500 にせず空で返す。"""
    assert _views_from_state({"answer": "こんにちは。"}) == ([], [])


def test_failed_tool_result_is_not_reported_as_ok():
    """status="error" の ToolMessage は ok=False になる。

    app/tools/engine.py も LangGraph の ToolNode も、失敗した tool は例外ではなく
    status="error" の ToolMessage として返す。ここを常に True にすると、評価は失敗した
    呼び出しを成功として数え、「tool は呼べているのに答えがおかしい」の切り分けができなくなる。
    """
    ok = ToolMessage(content="{}", tool_call_id="c1", name="query_order")
    ng = ToolMessage(
        content="ツール実行失敗: 注文が見つかりません", tool_call_id="c2", name="query_order",
        status="error",
    )
    _, results = _views_from_state({"messages": [ok, ng]})
    assert [r.ok for r in results] == [True, False]


def test_tool_call_without_id_does_not_break_the_view():
    """id を省略する OpenAI 互換ゲートウェイ経由でも、成功した turn を 500 で捨てない。

    ToolCallView.id は必須の str なので tc["id"] の素朴な添字アクセスでは
    ValidationError になる。`.get("id") or ""` の回帰テスト。
    """
    ai = AIMessage.model_construct(
        content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": None}]
    )
    calls, _ = _views_from_state({"messages": [ai]})
    assert calls[0].id == ""
    assert calls[0].name == "query_order"


def test_block_style_tool_content_is_flattened_to_text():
    """ToolMessage.content がブロック形式(list[dict])でも str になり、
    reasoning など type="text" 以外のブロックは落ちる(.content ではなく .text を使う作法)。"""
    tm = ToolMessage(
        content=[
            {"type": "text", "text": "配送中です"},
            {"type": "reasoning", "text": "内部推論は非公開"},
        ],
        tool_call_id="c1",
        name="query_logistics",
    )
    _, results = _views_from_state({"messages": [tm]})
    assert results[0].content == "配送中です"
    assert isinstance(results[0].content, str)
    assert "非公開" not in results[0].content
