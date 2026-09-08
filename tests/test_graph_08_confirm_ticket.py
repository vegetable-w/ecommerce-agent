"""08 章: create_ticket の確認フロー(interrupt → resume)。

create_ticket は唯一の write 操作で、engine の権限ゲートが「confirmed でなければ
実行しない」と決めている(app/tools/engine.py の ③)。その印を渡せるのは agent_tools
だけで、渡す条件は「ユーザーが確認カードで実際に押したこと」。ここで固定するのは、
その 1 本道が本当に閉じているかどうか。

確かめる 3 つ:

1. 引数が揃った create_ticket は **DB へ書く前に**止まり、チケットの下書きを
   画面へ出す。「作成する」で再開して初めて tickets へ書かれ、監査は success。
2. 「取り消す」で再開すると **DB へは書かず**、監査に permission_denied が残る。
   拒否も記録に残す(「作らなかった」ことが後から読めないと、監査は成功の記録にしかならない)。
3. 引数が足りない create_ticket は **確認カードを出さず**、engine の検証ブロックを
   モデルへ返す。監査は validation_blocked で、モデルは次の step でユーザーへ聞き直す。

interrupt() は compiled graph の中でしか動かないので、agent_llm ⇄ agent_tools の
最小 loop を組んで ainvoke する。checkpointer は InMemorySaver で、本番の
data/05_checkpoints.sqlite には触らない。

**本物の上流 model にも DB にも触らない。** モデルは台本どおりの偽物、
repository.create_ticket と insert_tool_audit は記録するだけの偽物、
async_session は到達不能なホストへ向けてある(差し替え忘れがその場で落ちるように)。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import repository
from app.graph import nodes, routing
from app.graph.state import ConversationState
from app.tools import engine, registry

# 到達不能なポート。ここへの接続は即座に拒否される(tests/test_resume_api.py と同じ作法)
_DEAD_DB_URL = "mysql+asyncmy://root:root@127.0.0.1:59999/nonexistent_db"

_TICKET_NO = "TK20260909001"
_DESCRIPTION = "充電器が発熱します"


class _FakeModel:
    """台本どおりの AIMessage を順に返すだけの偽モデル。

    1 回目に create_ticket の tool_call、2 回目以降は通常のテキストを返す。
    **同じ instance が中断をまたいで使われる**(resume でも agent_llm が呼ばれる)ので、
    何回目の呼び出しかを自分で数える。
    """

    def __init__(self, *script: AIMessage):
        self._script = list(script)
        self.calls = 0

    def bind_tools(self, tools):
        self.bound = list(tools)
        return self

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        return self._script[min(self.calls, len(self._script)) - 1]


def _ticket_call(args: dict, tc_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": "create_ticket", "args": args, "id": tc_id}])


@pytest.fixture
def env(monkeypatch):
    """上流と DB を塞ぎ、書き込みと監査を記録するだけの偽物へ差し替える。

    registry は built-in だけを返す関数に差し替える(MCP Server へ HTTP を投げない)。
    get_all_specs は agent_llm と agent_tools で 1 ターンに 2 回、さらに resume の
    再実行でもう一度走るので、実物のままだと単体テストが Server の生死に左右される。
    """
    monkeypatch.setattr(
        "app.db.base.async_session",
        async_sessionmaker(create_async_engine(_DEAD_DB_URL), expire_on_commit=False),
    )

    tickets: list[tuple] = []
    audits: list[dict] = []

    async def _create_ticket(conversation_id, description, ticket_type):
        tickets.append((conversation_id, description, ticket_type))
        return _TICKET_NO

    async def _insert_audit(**kw):
        audits.append(kw)
        return len(audits)

    async def _append_message(*a, **kw):
        return 0

    async def _builtin_only():
        return registry.builtin_specs()

    monkeypatch.setattr(repository, "create_ticket", _create_ticket)
    monkeypatch.setattr(repository, "insert_tool_audit", _insert_audit)
    monkeypatch.setattr(repository, "append_message", _append_message)
    monkeypatch.setattr(nodes.registry, "get_all_specs", _builtin_only)
    return {"tickets": tickets, "audits": audits}


def _graph():
    """agent_llm ⇄ agent_tools の最小 loop。build.py の該当部分と同じ繋ぎ方。"""
    b = StateGraph(ConversationState)
    b.add_node("agent_llm", nodes.agent_llm)
    b.add_node("agent_tools", nodes.agent_tools)
    b.add_edge(START, "agent_llm")
    b.add_conditional_edges("agent_llm", routing.should_continue,
                            {"continue": "agent_tools", "stop": END})
    b.add_edge("agent_tools", "agent_llm")
    return b.compile(checkpointer=InMemorySaver())


def _run(monkeypatch, model, thread: str):
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: model)
    return _graph(), {"configurable": {"thread_id": thread}}


def _input(text="チケットを作ってください"):
    return {"messages": [HumanMessage(text)], "conversation_id": 42,
            "user_id": "u-1", "steps": 0}


def _statuses(audits, name="create_ticket"):
    return [a["status"] for a in audits if a["tool_name"] == name]


def _tool_messages(state):
    return [m for m in state["messages"] if m.type == "tool"]


async def test_confirm_true_creates_ticket(monkeypatch, env):
    """作成する側。**確認カードを出した時点では 1 行も書いていない**ことが要点。

    interrupt より手前に書き込みを置くと、resume で node が先頭から実行し直される
    ため(06 章の実測)、ユーザーが 1 回押しただけでチケットが 2 件できる。
    """
    model = _FakeModel(
        _ticket_call({"description": _DESCRIPTION, "ticket_type": "after_sales"}),
        AIMessage("チケットを作成しました。"),
    )
    g, cfg = _run(monkeypatch, model, "confirm-yes")

    out = await g.ainvoke(_input(), cfg)

    payload = out["__interrupt__"][0].value
    assert payload["type"] == "confirm_ticket"
    assert payload["preview"] == {
        "ticket_type": "after_sales",             # DB の ENUM と同じ英語の識別子
        "ticket_type_label": "アフターサービス",    # 画面に出す日本語は labels から引く
        "description": _DESCRIPTION,
    }
    # ここが本題: まだ 1 行も書いていない
    assert env["tickets"] == []
    assert _statuses(env["audits"]) == []

    done = await g.ainvoke(Command(resume={"confirmed": True}), cfg)

    assert env["tickets"] == [(42, _DESCRIPTION, "after_sales")]    # ちょうど 1 件
    assert _statuses(env["audits"]) == [engine.STATUS_SUCCESS]
    assert _TICKET_NO in _tool_messages(done)[0].content
    assert "__interrupt__" not in done


async def test_confirm_false_denied_and_audited(monkeypatch, env):
    """取り消す側。DB へは書かず、拒否したことを監査に残す。

    合成の ToolMessage で済ませずに engine を通すのは、この 1 行を残すため。
    「作らなかった」ことが読めないと、監査は成功の記録にしかならない。
    """
    model = _FakeModel(
        _ticket_call({"description": _DESCRIPTION, "ticket_type": "complaint"}),
        AIMessage("承知しました。チケットは作成していません。"),
    )
    g, cfg = _run(monkeypatch, model, "confirm-no")

    await g.ainvoke(_input(), cfg)
    done = await g.ainvoke(Command(resume={"confirmed": False}), cfg)

    assert env["tickets"] == []
    assert _statuses(env["audits"]) == [engine.STATUS_PERMISSION_DENIED]
    content = _tool_messages(done)[0].content
    assert "キャンセル" in content
    # 取り消しを「引数が悪かった」と読ませない。読ませると同じ step で呼び直し、
    # ユーザーから見て取り消しが効かない
    assert "再実行しないでください" in content


async def test_missing_description_asks_instead_of_interrupt(monkeypatch, env):
    """引数が足りなければ確認カードを出さず、engine の検証ブロックを返す。

    ここで確認カードを出すと、ユーザーが「作成する」を押した後に engine が引数不足で
    弾き、押したのに何も起きない画面になる。足りないことはモデルへ返し、
    ユーザーへ聞き直させる(推測や仮の文言で埋めさせない)。
    """
    model = _FakeModel(
        _ticket_call({"ticket_type": "inquiry"}),           # description が無い
        AIMessage("どのような内容でお困りでしょうか。"),
    )
    g, cfg = _run(monkeypatch, model, "confirm-missing")

    out = await g.ainvoke(_input(), cfg)

    assert "__interrupt__" not in out                        # 中断していない
    assert env["tickets"] == []
    assert _statuses(env["audits"]) == [engine.STATUS_VALIDATION_BLOCKED]
    assert "パラメータ検証に失敗しました" in _tool_messages(out)[0].content
    # 2 turn 目でモデルはユーザーへの質問に収束する(tool を呼び直さない)
    assert model.calls == 2
    assert out["messages"][-1].content == "どのような内容でお困りでしょうか。"


async def test_the_write_gate_is_outside_the_node(monkeypatch, env):
    """確認を経ない create_ticket は engine の権限ゲートで止まる。

    agent_tools を通さずに engine を叩いた場合(tool を呼ぶ経路が将来増えた場合)でも、
    confirmed を渡さない限り書き込みは起きない。ゲートが node の外にあることの確認。
    """
    specs = {s.name: s for s in registry.builtin_specs()}
    run = await engine.execute_tool_call(
        {"name": "create_ticket", "id": "c1",
         "args": {"description": _DESCRIPTION, "ticket_type": "inquiry"}}, 42, specs)

    assert env["tickets"] == []
    assert run.status == engine.STATUS_PERMISSION_DENIED


async def test_a_second_ticket_call_in_the_same_step_is_not_created(monkeypatch, env):
    """1 step に create_ticket が 2 件並んでも、作られるのは確認を出した 1 件だけ。

    確認カードは 1 件ぶんしか出していない。2 件目も「確認済み」として実行すると、
    ユーザーが見ていない内容のチケットが作られる。
    """
    model = _FakeModel(
        AIMessage(content="", tool_calls=[
            {"name": "create_ticket", "id": "c1",
             "args": {"description": _DESCRIPTION, "ticket_type": "after_sales"}},
            {"name": "create_ticket", "id": "c2",
             "args": {"description": "別件です", "ticket_type": "inquiry"}},
        ]),
        AIMessage("1 件だけ受け付けました。"),
    )
    g, cfg = _run(monkeypatch, model, "confirm-two")

    await g.ainvoke(_input(), cfg)
    done = await g.ainvoke(Command(resume={"confirmed": True}), cfg)

    assert env["tickets"] == [(42, _DESCRIPTION, "after_sales")]
    # tool_calls と ToolMessage の対応は崩さない(崩すと上流が 400 を返す)
    assert [m.tool_call_id for m in _tool_messages(done)] == ["c1", "c2"]
    assert "無視されました" in _tool_messages(done)[1].content
