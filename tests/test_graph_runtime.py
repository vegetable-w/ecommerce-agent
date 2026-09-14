"""app/graph/runtime.py — checkpointer のライフサイクルと 2 つの実行入口。

ここは graph と FastAPI の間にある唯一の層で、**何を画面へ流すか**を決める。
なので確かめる中心は 2 つある。

1. **中間の model call の token を漏らさないこと**(spec R1)。classify_intent は
   構造化出力で model を呼ぶ。その token をそのまま流すと「配送」のような分類結果が
   ユーザーの画面に突然表示される。stream_mode="messages" の chunk は node 名でしか
   区別できないので、filter が外れても stream 自体は動き続ける。壊れたことに気づける
   場所はここしかない。
2. **checkpointer を全リクエストで共有しても混ざらないこと**(spec R2)。
   AsyncSqliteSaver は起動時に 1 つだけ開いて使い回す。thread_id の違う turn を
   同時に流して履歴が混ざらないことを、実物の sqlite で実測する。

上流(model / retrieval)と本番相当の DB は一切呼ばない。event の写像は台本どおりの
chunk を返す偽 graph で、checkpointer は tmp_path の sqlite + 偽 node の小さな graph で
確かめる(本番の data/05_checkpoints.sqlite には触らない)。
"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt

from app.graph import runtime
from app.graph.state import ConversationState


@pytest.fixture(autouse=True)
def _forbid_upstream_and_db(monkeypatch):
    """本番相当の DB と本物の上流 model を既定で禁止する。

    support は本番相当の DB なのでテストから書き込まない。差し替え忘れを
    「テストがたまたま通る」ではなく「その場で落ちる」に変える。model の方を
    覆っておかないと、偽 graph の組み立てを間違えた瞬間に本物の課金が発生する。
    """

    async def _boom_db(*args, **kwargs):
        raise AssertionError("テストが本物の repository を呼んだ")

    for name in ("create_conversation", "get_conversation", "append_message"):
        monkeypatch.setattr(runtime.repository, name, _boom_db)

    def _boom_model(*args, **kwargs):
        raise AssertionError("テストが本物の上流 model を呼んだ")

    monkeypatch.setattr("app.core.llm.get_chat_model", _boom_model)
    monkeypatch.setattr("app.graph.nodes.get_chat_model", _boom_model)


@pytest.fixture(autouse=True)
def _runtime_globals_clean():
    """module 変数をテスト間で持ち越さない。

    _graph / _cm は module 変数なので、閉じ忘れた 1 本が後続のテストの
    「未初期化なら例外」を静かに壊す。実体の後始末は各テストが close_graph() で行い、
    ここは取りこぼしの保険として値だけ戻す。
    """
    yield
    runtime._graph = None
    runtime._cm = None


@pytest.fixture(autouse=True)
def _summary_calls(monkeypatch) -> list:
    """要約の起動を記録用の偽物へ差し替える。

    本物は repository を叩いて background の task を立てる。turn の後で必ず呼ばれる
    ようになったので、覆っておかないと runtime を通す全テストが本番相当の DB へ
    出ていく。起動されたことを確かめるテストは、この list を受け取って見る。
    """
    calls: list = []

    async def _fake(conversation_id):
        calls.append(conversation_id)

    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", _fake)
    return calls


class _Conv:
    """get_conversation が返す会話。runtime が読むのは要約の 2 つだけ。"""

    def __init__(self, summary: str = "", upto: int = 0) -> None:
        self.summary = summary
        self.summary_upto_msg_id = upto


def _fake_repo(monkeypatch, *, new_id: int = 101, known: tuple[int, ...] = (7,),
               summary: str = "", upto: int = 0) -> list:
    """repository の 3 関数を記録用の偽物へ差し替え、呼び出し記録の list を返す。"""
    calls: list = []

    async def _create(user_id):
        calls.append(("create", user_id))
        return new_id

    async def _get(conversation_id):
        calls.append(("get", conversation_id))
        return _Conv(summary, upto) if conversation_id in known else None

    async def _append(conversation_id, role, content=None, tool_calls=None, tool_call_id=None):
        calls.append(("append", conversation_id, role, content))
        return len(calls)

    monkeypatch.setattr(runtime.repository, "create_conversation", _create)
    monkeypatch.setattr(runtime.repository, "get_conversation", _get)
    monkeypatch.setattr(runtime.repository, "append_message", _append)
    return calls


# --- 台本どおりの chunk を返す偽 graph -------------------------------------------


class _ScriptedGraph:
    """astream / ainvoke が台本を返すだけの偽 graph。

    langgraph 1.2.11 の astream(stream_mode=["messages","updates"])は旧来の
    (mode, chunk) tuple を返す(scripts/smoke_langgraph.py で実測済み)。その形を
    そのまま台本にするので、写像側の parser を実物と同じ入力で試せる。
    """

    def __init__(self, parts: list | None = None, final: dict | None = None):
        self.parts = parts or []
        self.final = final if final is not None else {}
        self.calls: list[dict] = []

    async def astream(self, inp, config, **kwargs):
        self.calls.append({"input": inp, "config": config, "kwargs": kwargs})
        for part in self.parts:
            yield part

    async def ainvoke(self, inp, config):
        self.calls.append({"input": inp, "config": config, "kwargs": {}})
        return self.final


def _use(monkeypatch, graph: _ScriptedGraph) -> _ScriptedGraph:
    monkeypatch.setattr(runtime, "_graph", graph)
    return graph


def _token(node: str, text) -> tuple:
    """messages mode の chunk。metadata の langgraph_node が唯一の出所の手がかり。"""
    return ("messages", (AIMessageChunk(content=text), {"langgraph_node": node}))


def _update(node: str, delta: dict) -> tuple:
    """updates mode の chunk。{node: State の差分}。"""
    return ("updates", {node: delta})


async def _events(**kwargs) -> list[dict]:
    return [ev async for ev in runtime.stream_turn(**kwargs)]


# --- get_graph ------------------------------------------------------------------


async def test_get_graph_は未初期化なら例外を出す():
    """None を返すと呼び出し側が AttributeError まで進み、原因が lifespan だと分からなくなる。"""
    with pytest.raises(RuntimeError, match="init_graph"):
        runtime.get_graph()


# --- 会話 ID の確定 ---------------------------------------------------------------


async def test_run_turn_は会話IDが無ければ採番する(monkeypatch):
    calls = _fake_repo(monkeypatch, new_id=55)
    graph = _use(monkeypatch, _ScriptedGraph(final={"answer": "はい"}))

    result = await runtime.run_turn("u1", "こんにちは", None)

    assert result == {"conversation_id": 55, "state": {"answer": "はい"}, "interrupt": None}
    assert calls == [("create", "u1"), ("append", 55, "user", "こんにちは")]
    # thread_id は会話 ID の文字列。ここがずれると turn をまたいで履歴が続かない。
    # 09: langfuse_session_id も同じ会話 ID。turn の入口 3 つすべてに乗っていることを
    # それぞれの入口で押さえる(1 箇所で足しているが、抜けたときに気づける場所は入口ごと)
    assert graph.calls[0]["config"] == {"configurable": {"thread_id": "55"},
                                        "metadata": {"langfuse_session_id": "55"}}


async def test_run_turn_は既存の会話をそのまま使う(monkeypatch):
    calls = _fake_repo(monkeypatch, known=(7,))
    graph = _use(monkeypatch, _ScriptedGraph(final={"answer": "はい"}))

    result = await runtime.run_turn("u1", "続きです", 7)

    assert result["conversation_id"] == 7
    assert calls == [("get", 7), ("append", 7, "user", "続きです")]
    assert graph.calls[0]["config"] == {"configurable": {"thread_id": "7"},
                                        "metadata": {"langfuse_session_id": "7"}}


async def test_run_turn_は知らない会話IDを拒否して何も保存しない(monkeypatch):
    """存在しない会話へ user message を書くと、誰も読まない行が DB に残る。"""
    calls = _fake_repo(monkeypatch, known=(7,))
    graph = _use(monkeypatch, _ScriptedGraph())

    with pytest.raises(runtime.ConversationNotFound):
        await runtime.run_turn("u1", "続きです", 999)

    assert not any(c[0] == "append" for c in calls)
    assert graph.calls == []


async def test_stream_turn_は知らない会話IDを拒否して何も保存しない(monkeypatch):
    calls = _fake_repo(monkeypatch, known=(7,))
    graph = _use(monkeypatch, _ScriptedGraph())

    with pytest.raises(runtime.ConversationNotFound):
        await _events(user_id="u1", message="続きです", conversation_id=999)

    assert not any(c[0] == "append" for c in calls)
    assert graph.calls == []


async def test_graphへの入力は1turn分の初期値になっている(monkeypatch):
    """steps と tokens_used に reducer は無い。turn ごとに 0 へ戻さないと、
    前 turn の step 数を引き継いだまま should_continue の上限に当たる。"""
    _fake_repo(monkeypatch, new_id=55)
    graph = _use(monkeypatch, _ScriptedGraph(final={}))

    await runtime.run_turn("u1", "こんにちは", None)

    inp = graph.calls[0]["input"]
    assert inp["user_id"] == "u1"
    assert inp["conversation_id"] == 55
    assert inp["steps"] == 0
    assert inp["tokens_used"] == 0
    assert [type(m) for m in inp["messages"]] == [HumanMessage]
    assert inp["messages"][0].content == "こんにちは"


# --- spec R1: 中間の model call の token を漏らさない -------------------------------


async def test_deltaになるのはagent_llmのtokenだけ(monkeypatch):
    """classify_intent / forced_rag / log も上流を呼ぶ node であり、その token が
    delta として出ると、ユーザーの画面に分類結果や検索の内部出力が突然表示される。

    stream 自体は filter が外れても動くので、壊れたことに気づけるのはこの assert だけ。
    """
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        _token("classify_intent", "配送"),
        _token("forced_rag", "検索クエリを書き換え中"),
        _token("agent_llm", "ご注文は"),
        _token("agent_llm", "発送済みです。"),
        _token("log", "trace"),
    ]))

    events = await _events(user_id="u1", message="注文はどこ", conversation_id=None)

    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["ご注文は", "発送済みです。"]
    assert "配送" not in "".join(deltas)


async def test_agent_toolsのtokenもdeltaにしない(monkeypatch):
    """ReAct loop の tool 側にも chunk が流れうる。ANSWER_NODES に無い node は
    名前を問わず一律で落とす、という決めを固定する。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        _token("agent_tools", '{"order_id": "1001"}'),
        _token("confidence_check", "strong"),
    ]))

    events = await _events(user_id="u1", message="注文はどこ", conversation_id=None)

    assert [e["type"] for e in events] == ["done"]


async def test_node名の無いmessages_chunkはdeltaにしない(monkeypatch):
    """metadata に langgraph_node が無い chunk(graph の外側で走る model 呼び出し)を
    「出所不明だから流す」に倒さない。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        ("messages", (AIMessageChunk(content="出所不明"), {})),
    ]))

    events = await _events(user_id="u1", message="注文はどこ", conversation_id=None)

    assert [e["type"] for e in events] == ["done"]


async def test_空文字のtokenはdeltaにしない(monkeypatch):
    """tool_calls だけを持つ AIMessageChunk は本文が空。素通しすると中身の無い
    delta が frontend に並ぶ。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        _token("agent_llm", ""),
        _token("agent_llm", "はい。"),
    ]))

    events = await _events(user_id="u1", message="注文はどこ", conversation_id=None)

    assert [e for e in events if e["type"] == "delta"] == [{"type": "delta", "text": "はい。"}]


async def test_本文ブロックのtokenはtextだけを取り出す(monkeypatch):
    """content は str とブロックの list のどちらもありうる。list をそのまま流すと
    frontend に dict が届き、思考ブロックまで画面に出る。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        _token("agent_llm", [{"type": "text", "text": "はい。"},
                             {"type": "thinking", "thinking": "内心"}]),
    ]))

    events = await _events(user_id="u1", message="注文はどこ", conversation_id=None)

    assert [e for e in events if e["type"] == "delta"] == [{"type": "delta", "text": "はい。"}]


# --- 決定的 node の回答 -----------------------------------------------------------


@pytest.mark.parametrize("node", ["script_reply", "complaint_reply", "fallback_reply"])
async def test_決定的nodeのanswerは1塊のdeltaとして出る(monkeypatch, node):
    """この 3 つは model を呼ばないので token が流れてこない。updates から拾わないと
    画面には何も出ないまま done だけが届く。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([_update(node, {"answer": "申し訳ありません。"})]))

    events = await _events(user_id="u1", message="遅い", conversation_id=None)

    assert events[0] == {"type": "delta", "text": "申し訳ありません。"}


async def test_answerを持たないupdatesはdeltaを出さない(monkeypatch):
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        _update("coref", {"trace": {"coref": "passthrough"}}),
        _update("confidence_check", {"trace": {"confidence": "strong"}}),
    ]))

    events = await _events(user_id="u1", message="こんにちは", conversation_id=None)

    assert [e["type"] for e in events] == ["done"]


async def test_dictでないupdatesを読み飛ばす(monkeypatch):
    """node が None を返す経路や __interrupt__ のような特殊な key が混ざっても、
    写像の途中で例外にして turn ごと落とさない。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        _update("log", None),
        _token("agent_llm", "はい。"),
    ]))

    events = await _events(user_id="u1", message="こんにちは", conversation_id=None)

    assert [e["type"] for e in events] == ["delta", "done"]


# --- citations / tool / actions ---------------------------------------------------


async def test_forced_ragのcitationsが回答本文より先に出る(monkeypatch):
    """frontend は [n] を描き始める時点で出典を持っていないと、クリックできる注釈にできない。"""
    _fake_repo(monkeypatch, new_id=55)
    items = [{"n": 1, "id": "c1", "question": "返品は?", "answer": "8日以内"}]
    _use(monkeypatch, _ScriptedGraph([
        _update("forced_rag", {"citations": items, "evidence_strong": True}),
        _token("agent_llm", "8日以内です [1]。"),
    ]))

    events = await _events(user_id="u1", message="返品は", conversation_id=None)

    assert events[0] == {"type": "citations", "items": items}
    assert events[1] == {"type": "delta", "text": "8日以内です [1]。"}


async def test_citationsが空ならcitationsイベントを出さない(monkeypatch):
    """forced_rag は weak のとき citations=[] を書く(前 turn の出典を残さないため)。
    それを流すと、断りの返答の横に空の出典欄が開く。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        _update("forced_rag", {"citations": [], "evidence_strong": False}),
    ]))

    events = await _events(user_id="u1", message="返品は", conversation_id=None)

    assert [e["type"] for e in events] == ["done"]


async def test_実行したtoolはcreate_ticketも含めてイベントにする(monkeypatch):
    """08 で create_ticket は確認カードを経て**実際に実行される**ようになった。

    05〜07 ではここで create_ticket だけを除いていた。理由は「実行していないものを
    実行しましたと画面に出すのは嘘になる」だったが、その前提が変わっている。
    engine を通った tool だけが ToolMessage を返すので、名前があるものは素直に出す。
    """
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        _update("agent_tools", {
            "messages": [
                ToolMessage(content="{}", tool_call_id="c1", name="query_order"),
                ToolMessage(content='{"ticket_no": "TK-1"}', tool_call_id="c2",
                            name="create_ticket"),
            ],
        }),
    ]))

    events = await _events(user_id="u1", message="注文1001が届かない", conversation_id=None)

    assert [e for e in events if e["type"] == "tool"] == [
        {"type": "tool", "name": "query_order"},
        {"type": "tool", "name": "create_ticket"},
    ]


async def test_名前の無いmessageはtoolイベントにしない(monkeypatch):
    """agent_tools が返す messages に ToolMessage 以外が混ざっても、name の無いものを
    name=None の tool イベントにして画面へ出さない。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        _update("agent_tools", {"messages": [AIMessage(content="補足")]}),
    ]))

    events = await _events(user_id="u1", message="注文1001", conversation_id=None)

    assert [e["type"] for e in events] == ["done"]


async def test_suggested_actionsは本文の後に1回だけ出す(monkeypatch):
    """actions は選択肢のボタンとして描かれる。本文の途中で出すと、まだ回答が
    終わっていないのにボタンが現れる。State は後勝ちなので最後の値を採る。"""
    _fake_repo(monkeypatch, new_id=55)
    first = [{"type": "transfer_human"}]
    second = [{"type": "transfer_human"}, {"type": "create_ticket", "draft": {}}]
    _use(monkeypatch, _ScriptedGraph([
        _update("agent_tools", {"suggested_actions": first}),
        _token("agent_llm", "選択肢をご用意しました。"),
        _update("agent_tools", {"suggested_actions": second}),
    ]))

    events = await _events(user_id="u1", message="人と話したい", conversation_id=None)

    assert [e["type"] for e in events] == ["delta", "actions", "done"]
    assert events[1]["items"] == second


async def test_complaint_replyのactionsも拾う(monkeypatch):
    """actions を出す node は agent_tools だけではない。complaint_reply も
    有人対応 / チケット作成の 2 択を State へ書く。"""
    _fake_repo(monkeypatch, new_id=55)
    actions = [{"type": "transfer_human"},
               {"type": "create_ticket",
                "draft": {"description": "遅い", "ticket_type": "complaint"}}]
    _use(monkeypatch, _ScriptedGraph([
        _update("complaint_reply", {"answer": "申し訳ありません。", "suggested_actions": actions}),
    ]))

    events = await _events(user_id="u1", message="遅い", conversation_id=None)

    assert [e["type"] for e in events] == ["delta", "actions", "done"]
    assert events[1]["items"] == actions


async def test_doneは最後に1回だけ会話IDを添えて出る(monkeypatch):
    """frontend は done の conversation_id を次の turn へ送る。新規会話ではここが
    唯一の受け渡し口なので、欠けると毎 turn 新しい会話が生える。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([_token("agent_llm", "はい。")]))

    events = await _events(user_id="u1", message="こんにちは", conversation_id=None)

    assert events[-1] == {"type": "done", "conversation_id": 55}
    assert [e["type"] for e in events].count("done") == 1


async def test_stream_turnは2つのmodeを要求する(monkeypatch):
    """messages だけだと決定的 node の回答が出ず、updates だけだと Agent の回答が
    token で流れない。両方要ることを呼び出しの形で固定する。"""
    _fake_repo(monkeypatch, new_id=55)
    graph = _use(monkeypatch, _ScriptedGraph())

    await _events(user_id="u1", message="こんにちは", conversation_id=None)

    assert graph.calls[0]["kwargs"]["stream_mode"] == ["messages", "updates"]
    assert graph.calls[0]["config"] == {"configurable": {"thread_id": "55"},
                                        "metadata": {"langfuse_session_id": "55"}}


async def test_stream_turnはuser_messageを先に保存する(monkeypatch):
    calls = _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([_token("agent_llm", "はい。")]))

    await _events(user_id="u1", message="こんにちは", conversation_id=None)

    assert calls == [("create", "u1"), ("append", 55, "user", "こんにちは")]


# --- checkpointer のライフサイクル(実物の sqlite を使う) --------------------------


def _fake_builder(record: list | None = None, gate=None):
    """上流を呼ばない 1 node の graph を組む、build_graph の代役。

    build_graph の中の node は上流を叩くので、checkpointer の挙動だけを見たいここでは
    使えない。State の型と messages の reducer は本物と同じ ConversationState を使う。
    """

    def _build(checkpointer=None):
        b = StateGraph(ConversationState)

        async def reply(state):
            if gate is not None:
                await gate.arrive()
            text = state["messages"][-1].content
            if record is not None:
                record.append((state.get("conversation_id"), text, len(state["messages"])))
            return {"messages": [AIMessage(f"返答:{text}")], "answer": f"返答:{text}"}

        b.add_node("reply", reply)
        b.add_edge(START, "reply")
        b.add_edge("reply", END)
        return b.compile(checkpointer=checkpointer)

    return _build


@asynccontextmanager
async def _running(monkeypatch, tmp_path, builder, name="cp.sqlite"):
    """init_graph → (テスト本体)→ close_graph。checkpointer は必ず tmp_path の下に作る。

    本番の data/05_checkpoints.sqlite をテストが触ると、実際の会話の履歴が汚れる。
    """
    path = tmp_path / name
    monkeypatch.setattr(runtime.settings, "checkpointer_db_path", str(path))
    monkeypatch.setattr(runtime, "build_graph", builder)
    await runtime.init_graph()
    try:
        yield path
    finally:
        await runtime.close_graph()


async def test_init_graphは設定のパスにcheckpointerを開く(monkeypatch, tmp_path):
    async with _running(monkeypatch, tmp_path, _fake_builder()) as path:
        assert runtime.get_graph() is not None
        assert path.exists()  # 本番のパスではなく tmp_path に作られている
    assert runtime._graph is None
    assert runtime._cm is None
    with pytest.raises(RuntimeError):
        runtime.get_graph()


async def test_init_graphの二重呼び出しは既存のcheckpointerを保つ(monkeypatch, tmp_path):
    """lifespan が二重に走っても、前の接続を捨てたり漏らしたりしない。"""
    async with _running(monkeypatch, tmp_path, _fake_builder()):
        first_cm, first_graph = runtime._cm, runtime._graph
        await runtime.init_graph()
        assert runtime._cm is first_cm
        assert runtime._graph is first_graph


async def test_close_graphは例外を伝えつつ状態を戻す(monkeypatch):
    """__aexit__ が落ちても None へ戻さないと、次の init_graph が閉じかけの接続を
    「初期化済み」と見なして使い続ける。例外自体は握り潰さない。"""

    class _BrokenCM:
        async def __aexit__(self, *exc):
            raise RuntimeError("接続を閉じられなかった")

    monkeypatch.setattr(runtime, "_cm", _BrokenCM())
    monkeypatch.setattr(runtime, "_graph", object())

    with pytest.raises(RuntimeError, match="閉じられなかった"):
        await runtime.close_graph()

    assert runtime._graph is None
    assert runtime._cm is None


async def test_close_graphは未初期化でも落ちない():
    await runtime.close_graph()
    assert runtime._graph is None


async def test_2turn目のStateに1turn目の履歴が残る(monkeypatch, tmp_path):
    """checkpointer が turn をまたいで State を持ち回ることを実物の sqlite で実測する。

    thread_id が会話 ID になっているかも同時に効いており、どちらが壊れても
    2 turn 目の messages が 2 件のままになる。
    """
    _fake_repo(monkeypatch, known=(7,))
    async with _running(monkeypatch, tmp_path, _fake_builder()):
        first = await runtime.run_turn("u1", "1つ目の質問", 7)
        second = await runtime.run_turn("u1", "2つ目の質問", 7)

    assert [m.content for m in first["state"]["messages"]] == ["1つ目の質問", "返答:1つ目の質問"]
    assert [m.content for m in second["state"]["messages"]] == [
        "1つ目の質問", "返答:1つ目の質問", "2つ目の質問", "返答:2つ目の質問",
    ]


async def test_別の会話は履歴を共有しない(monkeypatch, tmp_path):
    _fake_repo(monkeypatch, known=(7, 8))
    async with _running(monkeypatch, tmp_path, _fake_builder()):
        await runtime.run_turn("u1", "会話7の発話", 7)
        other = await runtime.run_turn("u2", "会話8の発話", 8)

    assert [m.content for m in other["state"]["messages"]] == ["会話8の発話", "返答:会話8の発話"]


class _Gate:
    """n 本の turn が同時に node の中に居る状態を必ず作る。

    sleep で「たぶん重なるだろう」と待つと、重ならなかった実行でもテストが通ってしまい、
    同時実行を確かめたことにならない。全員が着くまで誰も進めない形にする。
    ラウンドごとに Event を差し替えるので、2 turn 目も同じように待ち合わせできる。
    """

    def __init__(self, n: int):
        self.n = n
        self._count = 0
        self._event = asyncio.Event()

    async def arrive(self) -> None:
        event = self._event
        self._count += 1
        if self._count >= self.n:
            self._count = 0
            self._event = asyncio.Event()
            event.set()
            return
        await asyncio.wait_for(event.wait(), timeout=5)


async def test_別thread_idのturnを同時に流しても履歴が混ざらない(monkeypatch, tmp_path):
    """spec R2: AsyncSqliteSaver は 1 つだけ開いて全リクエストで共有する。

    _Gate が両方の turn を node の中で待ち合わせるので、checkpointer の読み書きは
    必ず重なる。それでも thread_id ごとに State が分かれていることを 2 turn 分見る。
    """
    _fake_repo(monkeypatch, known=(11, 22))
    gate = _Gate(2)
    seen: list = []

    async with _running(monkeypatch, tmp_path, _fake_builder(record=seen, gate=gate)):
        first = await asyncio.gather(
            runtime.run_turn("ua", "Aの1回目", 11),
            runtime.run_turn("ub", "Bの1回目", 22),
        )
        second = await asyncio.gather(
            runtime.run_turn("ua", "Aの2回目", 11),
            runtime.run_turn("ub", "Bの2回目", 22),
        )

    # 1 回目: それぞれ自分の発話しか見ていない
    assert [m.content for m in first[0]["state"]["messages"]] == ["Aの1回目", "返答:Aの1回目"]
    assert [m.content for m in first[1]["state"]["messages"]] == ["Bの1回目", "返答:Bの1回目"]

    # 2 回目: 自分の 1 回目だけを引き継いでいる(相手の発話が 1 件も混ざらない)
    a_texts = [m.content for m in second[0]["state"]["messages"]]
    b_texts = [m.content for m in second[1]["state"]["messages"]]
    assert a_texts == ["Aの1回目", "返答:Aの1回目", "Aの2回目", "返答:Aの2回目"]
    assert b_texts == ["Bの1回目", "返答:Bの1回目", "Bの2回目", "返答:Bの2回目"]
    assert not any("B" in t for t in a_texts)
    assert not any("A" in t for t in b_texts)

    # node へ入った時点の履歴の長さも会話ごとに独立している(4 回とも待ち合わせ済み)
    assert sorted(seen) == [(11, "Aの1回目", 1), (11, "Aの2回目", 3),
                            (22, "Bの1回目", 1), (22, "Bの2回目", 3)]


# ---------------------------------------------------------------------------
# turn をまたいだ出力の持ち越し
#
# checkpointer は State を丸ごと持ち越すので、前の turn が書いた出力を明示的に
# 戻さないと次の turn へ漏れる。実測した実害: 1 turn 目に苦情 → 2 turn 目に配送の
# 質問をすると、resolve_answer が 1 turn 目の共感文を返し、苦情のボタンも出たまま。
# ---------------------------------------------------------------------------

def test_graph_input_resets_every_output_channel():
    inp = runtime._graph_input("u1", "こんにちは", 7, 12, "", 0)
    assert inp["answer"] == "" and inp["suggested_actions"] == []
    assert inp["evidence"] == "" and inp["citations"] == []
    assert inp["evidence_strong"] is False
    assert inp["intent"] == "" and inp["route"] == ""
    assert inp["resolved_query"] == "" and inp["intent_confidence"] == 0.0
    assert inp["order_id"] == "" and inp["order_data"] == {}
    assert inp["steps"] == 0 and inp["tokens_used"] == 0


def test_graph_input_does_not_reset_the_history():
    """messages は履歴。ここで消すと checkpointer の意味が無くなる。"""
    inp = runtime._graph_input("u1", "こんにちは", 7, 12, "", 0)
    assert len(inp["messages"]) == 1
    assert inp["messages"][0].content == "こんにちは"


def test_trace_is_rebuilt_at_the_turn_boundary():
    """trace は reducer 付きなので空 dict では消えない。目印で作り直す。"""
    from app.graph.state import TRACE_RESET, merge_dict

    prev = {"forced_rag": True, "route": "knowledge"}
    fresh = runtime._graph_input("u1", "注文1001は?", 7, 12, "", 0)["trace"]
    assert merge_dict(prev, fresh) == {}          # 前 turn の値が残らない
    # 通常の merge は従来どおり足し合わせる
    assert merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert TRACE_RESET not in merge_dict(prev, fresh)


# --- 06 章: interrupt / resume -----------------------------------------------------
#
# Task 1 の red-line smoke(scripts/smoke_interrupt.py)で実測した形を、そのまま台本にする。
#   - ainvoke の戻り値には "__interrupt__" が **list** で入る
#   - astream の updates には {"__interrupt__": (Interrupt(...),)} が **tuple** で出る
# 2 つの入口で容れ物の型が違うので、片方だけを見る実装は他方で黙って素通しする。
# そのため list / tuple の両方を明示的に試す。


_SELECT_ORDER = {
    "type": "select_order",
    "orders": [
        {"order_id": "1001", "product": "自動猫トイレ", "status": "支払い済み", "amount": 1739},
        {"order_id": "2002", "product": "スマート体重計", "status": "発送済み", "amount": 3280},
    ],
}


def _interrupts(payload, *, box=tuple):
    """__interrupt__ の中身。box で list / tuple を切り替える。"""
    return box([Interrupt(value=payload)])


def _interrupt_update(payload, *, box=tuple) -> tuple:
    """updates mode に出る interrupt の chunk。node 名の位置に "__interrupt__" が来る。"""
    return ("updates", {"__interrupt__": _interrupts(payload, box=box)})


async def _resume_events(**kwargs) -> list[dict]:
    return [ev async for ev in runtime.stream_resume(**kwargs)]


@pytest.mark.parametrize("box", [tuple, list], ids=["tuple", "list"])
async def test_stream_turnはupdatesのinterruptをイベントにする(monkeypatch, box):
    """実測では tuple で来るが、容れ物の型に依存しないことを list でも確かめる。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([_interrupt_update(_SELECT_ORDER, box=box)]))

    events = await _events(user_id="u1", message="返金したい", conversation_id=None)

    assert events[0] == {
        "type": "interrupt",
        "kind": "select_order",
        "orders": _SELECT_ORDER["orders"],
        # 会話 ID を載せるのは、interrupt では done を出さないため。初回ターンで
        # 中断されたとき、画面はここでしか会話 ID を知る手立てがなく、
        # /api/actions/resume を叩けなくなる
        "conversation_id": 55,
    }


async def test_interruptで止まったらdoneを出さない(monkeypatch):
    """done は「その turn が完結した」という印。中断はまだ完結していない。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        _update("retrieve_policy", {"citations": [{"n": 1}]}),
        _interrupt_update(_SELECT_ORDER),
    ]))

    events = await _events(user_id="u1", message="返金したい", conversation_id=None)

    assert not any(ev["type"] == "done" for ev in events)
    assert events[-1]["type"] == "interrupt"


async def test_interruptのkindはpayloadのtypeから取る(monkeypatch):
    """select_order 以外の中断が増えたとき、画面が種類で分岐できる必要がある。
    ここを固定値にすると、新しい中断が全部「注文を選ぶ」画面として描かれる。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([
        _interrupt_update({"type": "confirm_refund", "orders": []}),
    ]))

    events = await _events(user_id="u1", message="返金したい", conversation_id=None)

    assert events[0]["kind"] == "confirm_refund"


_CONFIRM_TICKET = {
    "type": "confirm_ticket",
    "preview": {"ticket_type": "after_sales", "ticket_type_label": "アフターサービス",
                "description": "充電器が発熱します"},
}


async def test_confirm_ticketのinterruptはpreviewを運ぶ(monkeypatch):
    """08 の中断は注文の一覧ではなくチケットの下書きを運ぶ。

    ここで payload を落とすと、画面は「何のチケットを作るのか」を出せないまま
    確認ボタンだけを描くことになる。
    """
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([_interrupt_update(_CONFIRM_TICKET)]))

    events = await _events(user_id="u1", message="チケットを作って", conversation_id=None)

    assert events[0]["kind"] == "confirm_ticket"
    assert events[0]["preview"] == _CONFIRM_TICKET["preview"]
    # 06 の互換。画面は kind で描き分けるが、orders の key 自体は必ずある
    assert events[0]["orders"] == []
    assert events[0]["conversation_id"] == 55
    assert not any(ev["type"] == "done" for ev in events)


async def test_select_orderのinterruptにpreviewを足さない(monkeypatch):
    """payload に無い key を勝手に作らない。空の preview は「中身の無い確認カード」に
    なり、画面が注文の一覧ではなく確認ボタンを描く分岐へ倒れうる。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([_interrupt_update(_SELECT_ORDER)]))

    events = await _events(user_id="u1", message="返金したい", conversation_id=None)
    assert "preview" not in events[0]


_BROKEN_PAYLOADS = [
    pytest.param(("updates", {"__interrupt__": ()}), id="empty-tuple"),
    pytest.param(("updates", {"__interrupt__": None}), id="none"),
    pytest.param(("updates", {"__interrupt__": _interrupts(None)}), id="value-none"),
    pytest.param(("updates", {"__interrupt__": _interrupts("注文を選んでください")}), id="value-str"),
    pytest.param(("updates", {"__interrupt__": _interrupts({})}), id="value-empty-dict"),
    pytest.param(("updates", {"__interrupt__": _interrupts({"orders": []})}), id="no-type"),
    pytest.param(("updates", {"__interrupt__": _interrupts({"type": 7, "orders": {}})}),
                 id="wrong-types"),
    pytest.param(("updates", {"__interrupt__": _interrupts({"type": "confirm_ticket",
                                                            "preview": "作りますか"})}),
                 id="preview-not-a-dict"),
    pytest.param(("updates", {"__interrupt__": ["これは Interrupt ではない"]}), id="not-interrupt"),
]


@pytest.mark.parametrize("chunk", _BROKEN_PAYLOADS)
async def test_壊れたinterrupt_payloadでもturnごと落ちない(monkeypatch, chunk):
    """payload の形はこちら側で保証しきれない。読めなくても turn を落とさず、
    画面が読める分だけを渡す(kind と orders は空でも key は必ずある)。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([chunk]))

    events = await _events(user_id="u1", message="返金したい", conversation_id=None)

    itr = [ev for ev in events if ev["type"] == "interrupt"]
    assert len(itr) == 1
    assert set(itr[0]) == {"type", "kind", "orders", "conversation_id"}
    assert isinstance(itr[0]["kind"], str)
    assert isinstance(itr[0]["orders"], list)
    # 読めなくても graph は確かに止まっている。done を出すと画面は完結したと思う
    assert not any(ev["type"] == "done" for ev in events)


async def test_retrieve_policyのcitationsもイベントになる(monkeypatch):
    """返金フローの規約は retrieve_policy が引く。forced_rag だけを見ていると、
    返金の回答に付いた [n] の出典が画面に出ない。"""
    _fake_repo(monkeypatch, new_id=55)
    citations = [{"n": 1, "id": 9, "section_path": "返品ポリシー", "question": "返品条件",
                  "answer": "7日以内", "content_type": "policy"}]
    _use(monkeypatch, _ScriptedGraph([
        _update("retrieve_policy", {"citations": citations, "evidence": "[1] 返品条件: 7日以内"}),
        _token("agent_llm", "返品できます[1]。"),
        _update("log", {}),
    ]))

    events = await _events(user_id="u1", message="注文1001を返品したい", conversation_id=None)

    assert events[0] == {"type": "citations", "items": citations}
    assert events[1] == {"type": "delta", "text": "返品できます[1]。"}


async def test_retrieve_policyのcitationsが空ならイベントを出さない(monkeypatch):
    """引けなかった turn は前 turn の出典を消すために [] を書く。それは
    「出典なし」であって「空の出典欄を開け」ではない(forced_rag と同じ規律)。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([_update("retrieve_policy", {"citations": []})]))

    events = await _events(user_id="u1", message="返金したい", conversation_id=None)

    assert not any(ev["type"] == "citations" for ev in events)


# --- run_turn の interrupt --------------------------------------------------------


@pytest.mark.parametrize("box", [list, tuple], ids=["list", "tuple"])
async def test_run_turnはinterruptのpayloadを取り出す(monkeypatch, box):
    """実測では ainvoke 側は list。容れ物の型に依存しないことを tuple でも確かめる。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph(
        final={"answer": "", "__interrupt__": _interrupts(_SELECT_ORDER, box=box)}
    ))

    result = await runtime.run_turn("u1", "返金したい", None)

    assert result["interrupt"] == _SELECT_ORDER
    assert result["conversation_id"] == 55


async def test_run_turnはinterruptが無ければNoneを返す(monkeypatch):
    """key の有無で分岐させないため、中断していない turn でも key は必ず置く。"""
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph(final={"answer": "はい"}))

    assert await runtime.run_turn("u1", "こんにちは", None) == {
        "conversation_id": 55, "state": {"answer": "はい"}, "interrupt": None,
    }


@pytest.mark.parametrize("raw", [(), None, "文字列", ["Interrupt ではない"], _interrupts(None)])
async def test_run_turnは読めないinterruptでも落ちない(monkeypatch, raw):
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph(final={"answer": "", "__interrupt__": raw}))

    result = await runtime.run_turn("u1", "返金したい", None)
    assert result["interrupt"] is None


# --- resume_turn / stream_resume --------------------------------------------------


async def test_resume_turnはCommandを渡しuser_messageを保存しない(monkeypatch):
    """resume は新しい発話ではなく、前の発話の続き。ここで append_message すると
    同じ turn の user 行が 2 つ並び、以後すべてのターンの prompt に混入する。"""
    calls = _fake_repo(monkeypatch, known=(7,))
    graph = _use(monkeypatch, _ScriptedGraph(final={"answer": "返品できます"}))

    result = await runtime.resume_turn(7, "1001")

    assert result == {"conversation_id": 7,
                      "state": {"answer": "返品できます"}, "interrupt": None}
    assert not any(c[0] == "append" for c in calls)
    inp = graph.calls[0]["input"]
    # _graph_input を通すと turn の入口のリセットが走り、interrupt 待ちの State が壊れる
    assert isinstance(inp, Command)
    assert inp.resume == "1001"
    assert graph.calls[0]["config"] == {"configurable": {"thread_id": "7"},
                                        "metadata": {"langfuse_session_id": "7"}}


async def test_resume_turnは知らない会話を拒否してgraphを呼ばない(monkeypatch):
    calls = _fake_repo(monkeypatch, known=(7,))
    graph = _use(monkeypatch, _ScriptedGraph())

    with pytest.raises(runtime.ConversationNotFound):
        await runtime.resume_turn(999, "1001")

    assert graph.calls == []
    assert not any(c[0] == "append" for c in calls)


async def test_resume_turnは再びinterruptしたらそれを返す(monkeypatch):
    """読めない値で再開したときなど、resume の先でもう一度止まりうる。
    戻り値の形は run_turn と同じで、中断はそのまま表に出す。"""
    _fake_repo(monkeypatch, known=(7,))
    _use(monkeypatch, _ScriptedGraph(final={"__interrupt__": _interrupts(_SELECT_ORDER)}))

    result = await runtime.resume_turn(7, "よくわからない")
    assert result["interrupt"] == _SELECT_ORDER


async def test_stream_resumeはstream_turnと同じイベントを流す(monkeypatch):
    calls = _fake_repo(monkeypatch, known=(7,))
    citations = [{"n": 1, "id": 9, "question": "返品条件", "answer": "7日以内"}]
    graph = _use(monkeypatch, _ScriptedGraph([
        _update("fetch_order", {"order_id": "1001"}),
        _update("retrieve_policy", {"citations": citations}),
        _token("agent_llm", "注文 1001 は返品できます[1]。"),
        _update("agent_llm", {"suggested_actions": [{"type": "submit_refund"}]}),
    ]))

    events = await _resume_events(conversation_id=7, resume_value={"order_id": "1001"})

    assert events == [
        {"type": "citations", "items": citations},
        {"type": "delta", "text": "注文 1001 は返品できます[1]。"},
        {"type": "actions", "items": [{"type": "submit_refund"}]},
        {"type": "done", "conversation_id": 7},
    ]
    # user message は保存しない(resume は前の発話の続き)
    assert not any(c[0] == "append" for c in calls)
    inp = graph.calls[0]["input"]
    assert isinstance(inp, Command)
    assert inp.resume == {"order_id": "1001"}
    assert graph.calls[0]["kwargs"]["stream_mode"] == ["messages", "updates"]


async def test_stream_resumeも中断したらinterruptを出しdoneを出さない(monkeypatch):
    """選び直しが要る場合(2 段階の中断)でも、画面は同じイベントで扱える。"""
    _fake_repo(monkeypatch, known=(7,))
    _use(monkeypatch, _ScriptedGraph([_interrupt_update(_SELECT_ORDER)]))

    events = await _resume_events(conversation_id=7, resume_value="よくわからない")

    assert events == [{"type": "interrupt", "kind": "select_order",
                       "orders": _SELECT_ORDER["orders"], "conversation_id": 7}]


async def test_stream_resumeは知らない会話を拒否してgraphを呼ばない(monkeypatch):
    calls = _fake_repo(monkeypatch, known=(7,))
    graph = _use(monkeypatch, _ScriptedGraph())

    with pytest.raises(runtime.ConversationNotFound):
        await _resume_events(conversation_id=999, resume_value="1001")

    assert graph.calls == []
    assert not any(c[0] == "append" for c in calls)


def test_決定的nodeの一覧が06のnode名と一致する():
    """回答を state["answer"] に書く 3 つ。名前がずれると、その node の回答が
    1 文字も画面に出ない(05 章の scripted_reply は 06 で script_reply になった)。"""
    assert runtime.DETERMINISTIC_ANSWER_NODES == {
        "script_reply", "complaint_reply", "fallback_reply"
    }


# --- 07 章: anchor / 要約の受け渡し / 要約の起動 ---------------------------------
#
# スライディングウィンドウは「要約がどの message まで覆っているか」でしか窓を切れない。
# その突き合わせに使う anchor(MySQL の message id)を作れるのはここだけで、
# runtime が付け忘れると memory 側は静かに全履歴を渡し続ける。


async def test_今回の発話にはdb_idのanchorが付く(monkeypatch):
    """本文や並び順で境界を推測させないための唯一の手がかり。"""
    calls = _fake_repo(monkeypatch, new_id=55)
    graph = _use(monkeypatch, _ScriptedGraph(final={}))

    await runtime.run_turn("u1", "こんにちは", None)

    msg_id = len(calls)                       # 偽 append_message は呼び出し順を id にする
    assert graph.calls[0]["input"]["messages"][0].id == f"db-{msg_id}"


async def test_要約は毎turn会話から読み直してStateへ入れる(monkeypatch):
    """前 turn の要約が残ると、更新された後も古い要約を読み続ける。"""
    _fake_repo(monkeypatch, known=(7,), summary="ユーザーは注文1001を問い合わせた", upto=12)
    graph = _use(monkeypatch, _ScriptedGraph(final={}))

    await runtime.run_turn("u1", "続きです", 7)

    inp = graph.calls[0]["input"]
    assert inp["summary"] == "ユーザーは注文1001を問い合わせた"
    assert inp["summary_upto_msg_id"] == 12


async def test_採番したての会話には要約が無い(monkeypatch):
    _fake_repo(monkeypatch, new_id=55)
    graph = _use(monkeypatch, _ScriptedGraph(final={}))

    await runtime.run_turn("u1", "はじめまして", None)

    inp = graph.calls[0]["input"]
    assert inp["summary"] == "" and inp["summary_upto_msg_id"] == 0


async def test_会話の読み出しはturnの入口で1回だけ(monkeypatch):
    """存在確認と要約の取得を分けると、ターンごとに DB 往復が 1 つ増える。"""
    calls = _fake_repo(monkeypatch, known=(7,))
    _use(monkeypatch, _ScriptedGraph(final={}))

    await runtime.run_turn("u1", "続きです", 7)

    assert [c for c in calls if c[0] == "get"] == [("get", 7)]


async def test_run_turnはturnの後に要約の起動を試す(monkeypatch, _summary_calls):
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph(final={}))

    await runtime.run_turn("u1", "こんにちは", None)

    assert _summary_calls == [55]


async def test_stream_turnもturnの後に要約の起動を試す(monkeypatch, _summary_calls):
    _fake_repo(monkeypatch, new_id=55)
    _use(monkeypatch, _ScriptedGraph([_token("agent_llm", "はい")]))

    await _events(user_id="u1", message="こんにちは", conversation_id=None)

    assert _summary_calls == [55]


async def test_resumeの2つの入口も要約の起動を試す(monkeypatch, _summary_calls):
    """再開の後にも assistant の発話が 1 件積まれるので、履歴は伸びる。"""
    _fake_repo(monkeypatch, known=(7,))
    _use(monkeypatch, _ScriptedGraph([_token("agent_llm", "はい")], final={}))

    await runtime.resume_turn(7, "1001")
    await _resume_events(conversation_id=7, resume_value="1001")

    assert _summary_calls == [7, 7]


async def test_知らない会話では要約を起動しない(monkeypatch, _summary_calls):
    """存在しない会話 ID で要約を走らせても読む素材が無い。"""
    _fake_repo(monkeypatch, known=(7,))
    _use(monkeypatch, _ScriptedGraph())

    with pytest.raises(runtime.ConversationNotFound):
        await runtime.run_turn("u1", "続きです", 999)

    assert _summary_calls == []


# --- get_turn_snapshot (09 Task 7) ----------------------------------------------


def _snapshot_builder(record: list | None = None):
    """検索の写しを State へ書く 1 node の graph。forced_rag の出力の最小形。

    本物の forced_rag は上流(retrieval / rerank)を叩くのでここでは使えない。
    確かめたいのは「checkpointer に残った写しを読み出せるか」なので、写しを
    書く node があれば足りる。
    """

    def _build(checkpointer=None):
        b = StateGraph(ConversationState)

        async def reply(state):
            text = state["messages"][-1].content
            if record is not None:
                record.append(text)
            return {
                "messages": [AIMessage(f"返答:{text}")],
                "retrieved_snapshot": [{"question": text, "answer": "本文",
                                        "rerank_score": 0.5, "section_path": "節"}],
            }

        b.add_node("reply", reply)
        b.add_edge(START, "reply")
        b.add_edge("reply", END)
        return b.compile(checkpointer=checkpointer)

    return _build


async def test_get_turn_snapshotは直前の発話とそのターンの写しを返す(monkeypatch, tmp_path):
    """👎 が拾うのはこの 2 つ。実物の sqlite に書かれた State から読む。"""
    _fake_repo(monkeypatch, known=(7,))
    async with _running(monkeypatch, tmp_path, _snapshot_builder()):
        await runtime.run_turn("u1", "返品の送料は誰が負担しますか", 7)
        got = await runtime.get_turn_snapshot(7)

    assert got["question"] == "返品の送料は誰が負担しますか"
    assert [c["question"] for c in got["snapshot"]] == ["返品の送料は誰が負担しますか"]


async def test_get_turn_snapshotは最新のターンの値を返す(monkeypatch, tmp_path):
    """State は turn をまたいで残る。押すのが遅れれば別の質問の写しが載っている。

    👎 に写しを添えてよいかを呼び出し側が判断できるよう、質問を一緒に返している。
    ここが最新ターンを返さなくなると、その突き合わせが意味を失う。
    """
    _fake_repo(monkeypatch, known=(7,))
    async with _running(monkeypatch, tmp_path, _snapshot_builder()):
        await runtime.run_turn("u1", "1つ目の質問", 7)
        await runtime.run_turn("u1", "2つ目の質問", 7)
        got = await runtime.get_turn_snapshot(7)

    assert got["question"] == "2つ目の質問"
    assert [c["question"] for c in got["snapshot"]] == ["2つ目の質問"]


async def test_get_turn_snapshotはStateの無い会話でも落ちない(monkeypatch, tmp_path):
    """graph を 1 度も通っていない会話は「読めなかった」ではなく「まだ何も無い」。"""
    async with _running(monkeypatch, tmp_path, _snapshot_builder()):
        got = await runtime.get_turn_snapshot(4242)

    assert got == {"question": "", "snapshot": []}


async def test_get_turn_snapshotはStateを進めない(monkeypatch, tmp_path):
    """読むだけ。ここで turn が 1 つ進むと、👎 を押した操作が会話を動かす。"""
    seen: list = []
    _fake_repo(monkeypatch, known=(7,))
    async with _running(monkeypatch, tmp_path, _snapshot_builder(seen)):
        await runtime.run_turn("u1", "1つ目の質問", 7)
        await runtime.get_turn_snapshot(7)
        await runtime.get_turn_snapshot(7)
        after = await runtime.run_turn("u1", "2つ目の質問", 7)

    assert seen == ["1つ目の質問", "2つ目の質問"]   # node は turn の分しか走らない
    assert [m.content for m in after["state"]["messages"]] == [
        "1つ目の質問", "返答:1つ目の質問", "2つ目の質問", "返答:2つ目の質問",
    ]


async def test_get_turn_snapshotは未初期化なら例外を出す():
    """読めなかったことを空の写しとして隠さない。握るかどうかは呼び出し側が決める。"""
    with pytest.raises(RuntimeError, match="init_graph"):
        await runtime.get_turn_snapshot(7)


async def test_get_turn_snapshotは会話ごとに切り分けて読む(monkeypatch, tmp_path):
    """thread_id を取り違えると、👎 に別の会話の質問と写しが付く。

    プールに残るのは「押されていない質問」で、レビュー担当者にはそれが分からない。
    """
    _fake_repo(monkeypatch, known=(7, 8))
    async with _running(monkeypatch, tmp_path, _snapshot_builder()):
        await runtime.run_turn("u1", "会話7の質問", 7)
        await runtime.run_turn("u2", "会話8の質問", 8)
        # **両方を確かめる。** 片方だけだと、thread_id をその会話 ID に固定した
        # 実装(取り違えの典型)がそのまま通ってしまう
        seven = await runtime.get_turn_snapshot(7)
        eight = await runtime.get_turn_snapshot(8)

    assert seven["question"] == "会話7の質問"
    assert [c["question"] for c in seven["snapshot"]] == ["会話7の質問"]
    assert eight["question"] == "会話8の質問"
    assert [c["question"] for c in eight["snapshot"]] == ["会話8の質問"]
