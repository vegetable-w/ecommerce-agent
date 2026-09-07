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


def _fake_repo(monkeypatch, *, new_id: int = 101, known: tuple[int, ...] = (7,)) -> list:
    """repository の 3 関数を記録用の偽物へ差し替え、呼び出し記録の list を返す。"""
    calls: list = []

    async def _create(user_id):
        calls.append(("create", user_id))
        return new_id

    async def _get(conversation_id):
        calls.append(("get", conversation_id))
        return object() if conversation_id in known else None

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

    assert result == {"conversation_id": 55, "state": {"answer": "はい"}}
    assert calls == [("create", "u1"), ("append", 55, "user", "こんにちは")]
    # thread_id は会話 ID の文字列。ここがずれると turn をまたいで履歴が続かない
    assert graph.calls[0]["config"] == {"configurable": {"thread_id": "55"}}


async def test_run_turn_は既存の会話をそのまま使う(monkeypatch):
    calls = _fake_repo(monkeypatch, known=(7,))
    graph = _use(monkeypatch, _ScriptedGraph(final={"answer": "はい"}))

    result = await runtime.run_turn("u1", "続きです", 7)

    assert result["conversation_id"] == 7
    assert calls == [("get", 7), ("append", 7, "user", "続きです")]
    assert graph.calls[0]["config"] == {"configurable": {"thread_id": "7"}}


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


@pytest.mark.parametrize("node", ["chitchat_reply", "complaint_reply", "fallback_reply"])
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


async def test_toolイベントにcreate_ticketを出さない(monkeypatch):
    """agent_tools は create_ticket を実行せず選択肢へ変換する(app/graph/nodes.py)。
    実行していないものを「実行しました」と画面に出すのは嘘になる。"""
    _fake_repo(monkeypatch, new_id=55)
    draft = {"description": "届かない", "ticket_type": "complaint"}
    _use(monkeypatch, _ScriptedGraph([
        _update("agent_tools", {
            "messages": [
                ToolMessage(content="{}", tool_call_id="c1", name="query_order"),
                ToolMessage(content="提示しました", tool_call_id="c2", name="create_ticket"),
            ],
            "suggested_actions": [{"type": "create_ticket", "draft": draft}],
        }),
    ]))

    events = await _events(user_id="u1", message="注文1001が届かない", conversation_id=None)

    assert [e for e in events if e["type"] == "tool"] == [{"type": "tool", "name": "query_order"}]
    assert {"type": "actions", "items": [{"type": "create_ticket", "draft": draft}]} in events


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
    assert graph.calls[0]["config"] == {"configurable": {"thread_id": "55"}}


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
    inp = runtime._graph_input("u1", "こんにちは", 7)
    assert inp["answer"] == "" and inp["suggested_actions"] == []
    assert inp["evidence"] == "" and inp["citations"] == []
    assert inp["evidence_strong"] is False
    assert inp["intent"] == "" and inp["route"] == ""
    assert inp["resolved_query"] == "" and inp["intent_confidence"] == 0.0
    assert inp["order_id"] == "" and inp["order_data"] == {}
    assert inp["steps"] == 0 and inp["tokens_used"] == 0


def test_graph_input_does_not_reset_the_history():
    """messages は履歴。ここで消すと checkpointer の意味が無くなる。"""
    inp = runtime._graph_input("u1", "こんにちは", 7)
    assert len(inp["messages"]) == 1
    assert inp["messages"][0].content == "こんにちは"


def test_trace_is_rebuilt_at_the_turn_boundary():
    """trace は reducer 付きなので空 dict では消えない。目印で作り直す。"""
    from app.graph.state import TRACE_RESET, merge_dict

    prev = {"forced_rag": True, "route": "knowledge"}
    fresh = runtime._graph_input("u1", "注文1001は?", 7)["trace"]
    assert merge_dict(prev, fresh) == {}          # 前 turn の値が残らない
    # 通常の merge は従来どおり足し合わせる
    assert merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert TRACE_RESET not in merge_dict(prev, fresh)
