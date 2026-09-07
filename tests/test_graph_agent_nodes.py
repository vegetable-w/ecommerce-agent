"""main Agent node(ReAct loop の agent_llm / agent_tools)と、その入力を組み立てる _agent_messages。

graph の中で唯一「次に何をするか」をモデルが決める場所なので、ここが壊れると
knowledge route の根拠が届かない・step が数えられず loop が止まらない・
ユーザーが望んでいないチケットが実際に作られる、といった形で表に出る。

上流のモデルも本物のツールもここでは呼ばない。既定では両方「呼ばれたら落ちる」に
差し替えてあり、必要なテストだけが明示的に偽物を置く(差し替え忘れたまま本物の
上流を叩いて課金する事故を、テストの書き方ではなく仕組みで止める)。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.prompts import AGENT_SYSTEM
from app.graph import nodes
from app.tools.infra import ToolRun


@pytest.fixture(autouse=True)
def _forbid_upstream_and_tools(monkeypatch):
    """モデル取得とツール実行を既定で禁止する。使うテストは自分で偽物を置く。"""

    def _boom_model(*args, **kwargs):
        raise AssertionError("テストが上流のチャットモデルを呼んだ")

    async def _boom_tool(*args, **kwargs):
        raise AssertionError("テストが本物のツールを実行した")

    monkeypatch.setattr(nodes, "get_chat_model", _boom_model)
    monkeypatch.setattr(nodes, "execute_tool_call", _boom_tool)


class _FakeModel:
    """bind_tools して ainvoke されるだけの最小の偽モデル。呼ばれた内容を記録する。"""

    def __init__(self, ai: AIMessage):
        self._ai = ai
        self.bound_tools = None
        self.seen_messages = None
        self.seen_config = "(未設定)"

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages, config=None):
        self.seen_messages = list(messages)
        self.seen_config = config
        return self._ai


def _use_model(monkeypatch, ai: AIMessage) -> _FakeModel:
    fake = _FakeModel(ai)
    seen = {}

    def _get(streaming=False, **kwargs):
        seen["streaming"] = streaming
        return fake

    monkeypatch.setattr(nodes, "get_chat_model", _get)
    fake.get_kwargs = seen
    return fake


def _use_tools(monkeypatch) -> list:
    """ツール実行を記録するだけの偽物に差し替え、記録先の list を返す。"""
    called = []

    async def _fake(tc, conversation_id):
        called.append((tc, conversation_id))
        return ToolRun(
            tool_call_id=tc["id"],
            name=tc["name"],
            ok=True,
            tool_message=ToolMessage(
                content='{"ok": true}', tool_call_id=tc["id"], name=tc["name"]
            ),
        )

    monkeypatch.setattr(nodes, "execute_tool_call", _fake)
    return called


# --- _agent_messages -----------------------------------------------------------


def test_knowledge_route_appends_evidence_to_the_system_message():
    """knowledge route では forced_rag が引いた evidence を system の末尾へ連結する。

    ToolMessage として差し込めないのは、対応する tool_call が存在せず上流に弾かれるため。
    """
    state = {"route": "knowledge", "evidence": "[1] 返品ポリシー: 7日以内は返品可能",
             "messages": [HumanMessage("返品できますか")]}
    msgs = nodes._agent_messages(state)

    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content.startswith(AGENT_SYSTEM)
    assert "[1] 返品ポリシー: 7日以内は返品可能" in msgs[0].content


def test_knowledge_route_tells_the_model_not_to_search_again():
    """引き直しを禁じる指示が system に入ること。

    forced_rag が引き終えた検索をやり直されると、1 step と 1 回分の課金を捨てるだけでなく、
    2 度目の結果の番号が State の citations とずれ、本文の [n] と出典が食い違う。
    """
    state = {"route": "knowledge", "evidence": "[1] q: a", "messages": []}
    sys = nodes._agent_messages(state)[0].content
    assert "query_faq" in sys
    assert "再度呼ばないでください" in sys


def test_business_route_does_not_append_evidence():
    """business route では evidence を連結しない(そもそも検索していない)。"""
    state = {"route": "business", "evidence": "[1] 混ざってはいけない根拠",
             "messages": [HumanMessage("注文1001はどこ")]}
    sys = nodes._agent_messages(state)[0].content
    assert sys == AGENT_SYSTEM
    assert "混ざってはいけない根拠" not in sys


def test_empty_evidence_is_not_appended():
    """evidence が空文字なら連結しない。

    forced_rag は weak のとき evidence="" を書く(前 turn の出典を残さないため)。
    見出しだけ付いた空の evidence は「根拠はあるが中身が無い」という誤った指示になる。
    """
    state = {"route": "knowledge", "evidence": "", "messages": []}
    assert nodes._agent_messages(state)[0].content == AGENT_SYSTEM


def test_history_follows_the_system_message_in_order():
    """system の後ろに、turn をまたいだ履歴がそのままの順で並ぶこと。"""
    history = [HumanMessage("注文1001は?"), AIMessage("確認します"), HumanMessage("今どこ?")]
    msgs = nodes._agent_messages({"route": "business", "messages": history})
    assert msgs[1:] == history


# --- agent_llm -----------------------------------------------------------------


async def test_agent_llm_counts_a_step_and_accumulates_tokens(monkeypatch):
    ai = AIMessage(content="はい", usage_metadata={
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
    _use_model(monkeypatch, ai)

    out = await nodes.agent_llm({"messages": [HumanMessage("hi")], "steps": 2,
                                 "tokens_used": 500})
    assert out["messages"] == [ai]
    assert out["steps"] == 3
    assert out["tokens_used"] == 620


async def test_agent_llm_starts_from_zero_when_the_state_is_fresh(monkeypatch):
    _use_model(monkeypatch, AIMessage(content="はい"))
    out = await nodes.agent_llm({"messages": [HumanMessage("hi")]})
    assert out["steps"] == 1


async def test_agent_llm_survives_missing_usage_metadata(monkeypatch):
    """usage_metadata を付けない上流でも落ちないこと。

    token 量は trace と集計のためだけの数字で、loop の制御には使わない。
    ここで落とすと、token を報告しない上流に繋いだ瞬間に graph 全体が止まる。
    """
    ai = AIMessage(content="はい")
    assert ai.usage_metadata is None
    _use_model(monkeypatch, ai)

    out = await nodes.agent_llm({"messages": [HumanMessage("hi")], "tokens_used": 7})
    assert out["tokens_used"] == 7
    assert out["steps"] == 1


async def test_agent_llm_binds_every_tool_and_passes_the_config(monkeypatch):
    """全ツールを bind し、LangGraph から渡された config をそのまま ainvoke へ渡すこと。

    config を落とすと stream 用の callback が届かず、token が frontend へ流れない。
    """
    from app.tools.registry import get_all_tools

    fake = _use_model(monkeypatch, AIMessage(content="はい"))
    cfg = {"callbacks": ["dummy"]}
    await nodes.agent_llm({"messages": [HumanMessage("hi")], "route": "business"}, cfg)

    assert {t.name for t in fake.bound_tools} == {t.name for t in get_all_tools()}
    assert fake.seen_config is cfg
    assert fake.get_kwargs["streaming"] is True


async def test_agent_llm_sends_the_evidence_bearing_system_message(monkeypatch):
    """agent_llm が組み立てる入力が _agent_messages の結果であること。"""
    fake = _use_model(monkeypatch, AIMessage(content="はい"))
    await nodes.agent_llm({"route": "knowledge", "evidence": "[1] q: a",
                           "messages": [HumanMessage("返品は?")]})
    assert "[1] q: a" in fake.seen_messages[0].content


# --- agent_tools ---------------------------------------------------------------


def _ai_with(*tool_calls) -> AIMessage:
    return AIMessage(content="", tool_calls=list(tool_calls))


async def test_agent_tools_executes_a_normal_tool(monkeypatch):
    called = _use_tools(monkeypatch)
    tc = {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
    out = await nodes.agent_tools({"messages": [_ai_with(tc)], "conversation_id": 42})

    assert [c[0]["name"] for c in called] == ["query_order"]
    assert called[0][1] == 42  # conversation_id が注入経路へ渡ること
    assert [m.tool_call_id for m in out["messages"]] == ["c1"]
    assert isinstance(out["messages"][0], ToolMessage)
    # 通常ツールだけなら選択肢は積まない
    assert "suggested_actions" not in out


async def test_agent_tools_defaults_conversation_id_when_the_state_lacks_one(monkeypatch):
    called = _use_tools(monkeypatch)
    await nodes.agent_tools(
        {"messages": [_ai_with({"name": "query_order", "args": {}, "id": "c1"})]}
    )
    assert called[0][1] == 0


async def test_create_ticket_is_never_executed(monkeypatch):
    """create_ticket は**実行しない**。実行されたらテストが落ちること。

    苦情や要望の言葉が出るたびにチケットを立てると、ユーザーが望んでいないものが
    運用側の待ち行列に積み上がる。作るかどうかを決めるのはユーザーであり、
    ここは選択肢の提示までしかしない(complaint_reply と同じ方針)。
    execute_tool_call は autouse fixture で「呼ばれたら落ちる」に差し替えてある。
    """
    tc = {"name": "create_ticket",
          "args": {"description": "商品が壊れていた", "ticket_type": "after_sales"},
          "id": "c1"}
    out = await nodes.agent_tools({"messages": [_ai_with(tc)], "conversation_id": 42})

    assert out["suggested_actions"] == [
        {"type": "create_ticket",
         "draft": {"description": "商品が壊れていた", "ticket_type": "after_sales"}}
    ]


async def test_intercepted_ticket_answers_the_original_tool_call_id(monkeypatch):
    """合成 ToolMessage の tool_call_id が元の tool_call と一致すること。

    上流は tool_calls と ToolMessage の対応を検査する。欠けても食い違っても
    次の呼び出しが 400 になり、loop がその場で止まる。
    """
    tc = {"name": "create_ticket", "args": {"description": "d", "ticket_type": "complaint"},
          "id": "call_abc123"}
    out = await nodes.agent_tools({"messages": [_ai_with(tc)]})

    assert len(out["messages"]) == 1
    tm = out["messages"][0]
    assert tm.tool_call_id == "call_abc123"
    assert tm.name == "create_ticket"
    assert "選択肢" in tm.content


async def test_ticket_type_defaults_to_the_english_enum_identifier(monkeypatch):
    """ticket_type が無い場合の既定は DB の ENUM と同じ inquiry(日本語ではない)。"""
    tc = {"name": "create_ticket", "args": {}, "id": "c1"}
    out = await nodes.agent_tools({"messages": [_ai_with(tc)]})
    assert out["suggested_actions"][0]["draft"] == {"description": "", "ticket_type": "inquiry"}


async def test_existing_suggested_actions_are_kept(monkeypatch):
    """前の step や complaint_reply が積んだ選択肢を上書きで消さないこと。

    suggested_actions には reducer が無く後勝ちの上書きになるので、
    積み直しはこの node の責任になる。
    """
    tc = {"name": "create_ticket", "args": {"description": "d"}, "id": "c1"}
    out = await nodes.agent_tools({"messages": [_ai_with(tc)],
                                   "suggested_actions": [{"type": "transfer_human"}]})
    assert [a["type"] for a in out["suggested_actions"]] == ["transfer_human", "create_ticket"]


async def test_normal_tool_still_runs_when_mixed_with_create_ticket(monkeypatch):
    """1 step に通常ツールと create_ticket が並んでも、通常ツールは実行される。

    create_ticket を見つけた時点で step ごと打ち切ると、同時に要求された注文照会が
    実行されないまま ToolMessage も返らず、上流が対応の欠けを 400 で弾く。
    """
    called = _use_tools(monkeypatch)
    order = {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
    ticket = {"name": "create_ticket", "args": {"description": "壊れていた"}, "id": "c2"}
    out = await nodes.agent_tools({"messages": [_ai_with(order, ticket)],
                                   "conversation_id": 3})

    assert [c[0]["name"] for c in called] == ["query_order"]  # ticket は実行されない
    # tool_calls と同じ順・同じ id で ToolMessage が揃うこと
    assert [m.tool_call_id for m in out["messages"]] == ["c1", "c2"]
    assert [a["type"] for a in out["suggested_actions"]] == ["create_ticket"]
