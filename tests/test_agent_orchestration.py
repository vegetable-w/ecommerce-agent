"""app.core.agent の非ストリーミング出口(run_agent_turn)と共通コア(_prepare_turn)のテスト。

DB を触るため、モジュール先頭に loop_scope="session" の asyncio マーカーが必要
(tests/conftest.py の説明、および pyproject.toml のコメント参照)。
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import select

from app.core import agent
from app.core.prompts import AGENT_SYSTEM
from app.db import repository as repo
from app.db.models import Ticket

pytestmark = pytest.mark.asyncio(loop_scope="session")


class FakeModel:
    """turn1 は scripted[0] を返す(tool_calls を含んでもよい)。非ストリーミング収束は scripted[1]。
    ストリーミング収束は stream_tokens を順番に AIMessageChunk として yield する。
    bind_calls は bind_tools の呼び出し回数を記録する(収束時 bind なし → 常に 1 の想定)。"""

    def __init__(self, scripted, stream_tokens=None):
        self._scripted = list(scripted)
        self._stream_tokens = list(stream_tokens or [])
        self.bind_calls = 0
        self.invoke_messages = []
        self.astream_messages = []

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def ainvoke(self, messages):
        self.invoke_messages.append(list(messages))
        return self._scripted.pop(0)

    async def astream(self, messages):
        self.astream_messages.append(list(messages))
        for t in self._stream_tokens:
            yield AIMessageChunk(content=t)


async def test_no_tool_calls_returns_direct_answer(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="こんにちは。どのようなご用件でしょうか?")])
    res = await agent.run_agent_turn("u1", "こんにちは", None, model=model)
    assert res.answer.startswith("こんにちは") and res.tool_calls == []
    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant"]


async def test_direct_answer_uses_text_property_and_excludes_reasoning_blocks(
    db_session_factory, db_clean
):
    """.text は content が block list のとき type="text" の部分だけを連結し、それ以外の
    ブロック(reasoning など)は含めない。chapter 1 が踏んだ Critical バグと同じ形
    (`content if isinstance(content, str) else ""` のような自前実装)への回帰テスト。
    ツールなし分岐の ai.text 呼び出しを直接検証する。"""
    content = [
        {"type": "text", "text": "ご案内します"},
        {"type": "reasoning", "text": "内部思考は非公開"},
    ]
    model = FakeModel([AIMessage(content=content)])
    res = await agent.run_agent_turn("u1", "こんにちは", None, model=model)
    assert res.answer == "ご案内します"
    assert "非公開" not in res.answer


async def test_convergence_final_text_excludes_reasoning_blocks(db_session_factory, db_clean):
    """収束呼び出し(非ストリーミング)の final.text も同じ回帰テスト対象。"""
    first = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"tracking_no": "JP213502378238"}, "id": "c1"}],
    )
    final_content = [
        {"type": "text", "text": "現在配送中です"},
        {"type": "reasoning", "text": "非公開の内部推論"},
    ]
    model = FakeModel([first, AIMessage(content=final_content)])
    res = await agent.run_agent_turn("u1", "注文1001はどこですか", None, model=model)
    assert res.answer == "現在配送中です"
    assert "非公開" not in res.answer

    # レビュー指摘: append_message(content=final.text) の保存側も同じ回帰対象。
    # ここが breakして final.content(block list)をそのまま保存するようになっても、
    # 今ターンの res.answer は正しいままなので気づけない。しかし _build_history は
    # 保存された assistant.content をそのまま次ターンの履歴として再生するため、
    # 症状は次ターンのプロンプト汚染として遅れて出る(このチャプターで2回目の同型バグ)。
    msgs = await repo.list_messages(res.conversation_id)
    assert msgs[-1].content == "現在配送中です"


async def test_tool_call_flow_executes_and_converges(db_session_factory, db_clean):
    first = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"tracking_no": "JP213502378238"}, "id": "c1"}],
    )
    model = FakeModel([first, AIMessage(content="注文 1001 は現在輸送中です。")])
    res = await agent.run_agent_turn("u1", "注文 1001 は今どこですか", None, model=model)
    assert res.answer == "注文 1001 は現在輸送中です。"
    assert res.tool_calls[0]["name"] == "query_logistics"
    assert res.tool_runs[0].ok is True
    assert model.bind_calls == 1  # 収束時は bind しない
    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[2].tool_call_id == "c1"

    # レビュー指摘: 収束呼び出しにツール結果が実際に渡っていることをFakeModelの記録で確認する。
    # ここを確認しないと、収束が [*messages, ai] だけ(tool_message を落とす)に壊れても
    # FakeModelはscriptedを順にpopするだけなので他の全テストが緑のまま通ってしまう。
    convergence_messages = model.invoke_messages[1]
    tool_msgs = [m for m in convergence_messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"
    assert tool_msgs[0].content == res.tool_runs[0].tool_message.content


async def test_continue_conversation_replays_only_final_answers(db_session_factory, db_clean):
    m1 = FakeModel(
        [
            AIMessage(
                content="確認します。",
                tool_calls=[{"name": "query_logistics", "args": {"tracking_no": "JP213502378238"}, "id": "c1"}],
            ),
            AIMessage(content="注文 1001 は集荷済みです。"),
        ]
    )
    cid = (await agent.run_agent_turn("u1", "注文1001は今どこですか", None, model=m1)).conversation_id
    m2 = FakeModel([AIMessage(content="ほかにお手伝いできることはありますか?")])
    await agent.run_agent_turn("u1", "ありがとう", cid, model=m2)
    ai_contents = [m.content for m in m2.invoke_messages[0] if isinstance(m, AIMessage)]
    assert "注文 1001 は集荷済みです。" in ai_contents  # 最終回答は再利用
    assert "確認します。" not in ai_contents  # tool-calling preamble は次ターンへ渡さない


async def test_unknown_conversation_raises(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    with pytest.raises(agent.ConversationNotFound):
        await agent.run_agent_turn("u1", "hi", 999999, model=model)


# --- 申し送り事項B: list_messages が存在しない/空の conversation_id に対して [] を返す契約。
# _build_history は新規会話でこれに依存するため、明示的にテストで固定する。


async def test_list_messages_returns_empty_for_nonexistent_conversation(db_session_factory, db_clean):
    assert await repo.list_messages(999999) == []


async def test_list_messages_returns_empty_for_new_conversation(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    assert await repo.list_messages(cid) == []


# --- システムプロンプトの生存確認(調査で発覚した trim_history の include_system=False 問題への回帰テスト)。
# _build_history は SystemMessage を trim 対象の history リストに含めず、AGENT_PROMPT で
# trim 後に付与する。この2本のテストは「モデルへ実際に渡る先頭メッセージが system prompt であること」
# を短い会話・trim が発生するほど長い会話の両方で固定する。


async def test_system_prompt_survives_into_model_messages(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="こんにちは")])
    await agent.run_agent_turn("u1", "こんにちは", None, model=model)
    first_call_messages = model.invoke_messages[0]
    assert isinstance(first_call_messages[0], SystemMessage)
    assert first_call_messages[0].content == AGENT_SYSTEM


async def test_multiple_tool_calls_execute_and_both_converge(db_session_factory, db_clean):
    """1ターンで複数ツール呼び出しがある場合、両方実行され、両方の tool 行が正しい
    tool_call_id で保存され、両方の ToolMessage が収束呼び出しへ渡ることを確認する。"""
    first = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"},
            {"name": "query_logistics", "args": {"tracking_no": "JP213502378238"}, "id": "c2"},
        ],
    )
    model = FakeModel([first, AIMessage(content="両方確認しました。")])
    res = await agent.run_agent_turn("u1", "注文1001について教えて", None, model=model)

    assert {tc["name"] for tc in res.tool_calls} == {"query_order", "query_logistics"}
    assert len(res.tool_runs) == 2
    assert all(r.ok for r in res.tool_runs)
    assert {r.tool_call_id for r in res.tool_runs} == {"c1", "c2"}

    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "tool", "assistant"]
    assert {m.tool_call_id for m in msgs if m.role == "tool"} == {"c1", "c2"}

    convergence_tool_msgs = [m for m in model.invoke_messages[1] if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in convergence_tool_msgs} == {"c1", "c2"}


async def test_prepare_turn_survives_tool_call_with_none_id_alongside_good_one(
    db_session_factory, db_clean
):
    """id=None の tool_call が1つ混ざっていても、_prepare_turn は落ちずに完走し、
    正常な兄弟ツール(create_ticket、書き込み系で副作用を持つ)の結果が tool 行なしに
    孤児化しないことを確認する、エンドツーエンドの統合テスト。

    レビュー指摘の訂正: このテストは app/tools/infra.py 側の `tool_call.get("id") or
    "unknown"` を直接は踏まない。execute_tool_call が(infra.py 側の防御が生きていれば)
    正常に ToolRun を返すため、agent.py 側の _exception_to_tool_run(gather の
    return_exceptions=True による多重防御のバックストップ)は呼ばれずに完走する。
    実際に確かめると、infra.py 側の防御だけを外して agent.py 側を残した場合でも
    execute_tool_call が ValidationError を送出 → gather がそれを拾う →
    _exception_to_tool_run が同じ ("unknown", ok=False) の ToolRun を作るため、
    この統合テストの assert は全部通ってしまう(2層が同じ入出力を作るので、
    どちらか一方が壊れても統合レベルでは見分けが付かない)。
    infra.py の `tool_call.get("id") or "unknown"` 自体は
    tests/test_tools_infra.py::test_execute_tool_call_none_id_returns_error_run_instead_of_raising
    が単体レベルで直接カバーしている。ここではあくまで
    「_prepare_turn 全体として id=None を持つ tool_call を落とさず処理し切れるか」
    (どちらの層が実際に守っているかは問わない)を確認する。

    id=None は理論上の話ではなく、AIMessage(tool_calls=[{"name":..., "args":...,
    "id": None}]) は langchain_core の ToolCall.id: str | None の宣言どおり正常に
    構築できる(以前の版で使っていた「構築後に list へ append する」トリックは不要)。
    OpenAI互換ゲートウェイが id を省略した tool_call チャンクを astream でマージすると
    この形になりうるため、実運用でも到達しうる。
    """
    first = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "create_ticket",
                "args": {"description": "配送が遅い", "ticket_type": "complaint"},
                "id": "c1",
            },
            # args を欠落させ、LogisticsInput の必須フィールド tracking_no 不足で
            # ValidationError(=エラー ToolRun)になる経路を確実に踏ませる。
            # (tracking_no を渡すと単に成功して ok=True になり、"id=None でも success/error
            # どちらの分岐でも ToolMessage(tool_call_id=None) が構築される" ことの
            # error側の実例を示せなくなるため)
            {"name": "query_logistics", "args": {}, "id": None},
        ],
    )
    model = FakeModel([first, AIMessage(content="ご案内します。")])
    res = await agent.run_agent_turn("u1", "配送が遅いです", None, model=model)

    assert len(res.tool_runs) == 2
    runs_by_id = {r.tool_call_id: r for r in res.tool_runs}
    assert runs_by_id["c1"].ok is True
    assert runs_by_id["unknown"].ok is False  # id=None は "unknown" に落ちてエラー化される

    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "tool", "assistant"]
    assert {m.tool_call_id for m in msgs if m.role == "tool"} == {"c1", "unknown"}

    # 良い方(create_ticket)の副作用が孤児化せず記録されていることを確認する
    conv = await repo.get_conversation(res.conversation_id)
    assert conv.status == "escalated"
    async with db_session_factory() as s:
        tickets = list(
            (
                await s.execute(
                    select(Ticket).where(Ticket.conversation_id == res.conversation_id)
                )
            ).scalars()
        )
    assert len(tickets) == 1


async def test_prepare_turn_survives_execute_tool_call_raising(
    db_session_factory, db_clean, monkeypatch
):
    """execute_tool_call 自体は「例外を漏らさない」契約を持つが、asyncio.gather に
    return_exceptions=True を渡す防御的多重化がある。ここでは execute_tool_call を
    直接壊して例外を送出させ、それでも _prepare_turn がクラッシュせず、他の兄弟タスクを
    最後まで実行し切り(取りこぼしなく tool 行を保存し)、例外側もエラー ToolRun として
    扱われることを確認する。"""
    real_execute = agent.execute_tool_call

    async def flaky_execute(tool_call, conversation_id, *args, **kwargs):
        if tool_call["id"] == "c2":
            raise RuntimeError("boom")
        return await real_execute(tool_call, conversation_id, *args, **kwargs)

    monkeypatch.setattr(agent, "execute_tool_call", flaky_execute)

    first = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"},
            {"name": "query_logistics", "args": {"tracking_no": "JP213502378238"}, "id": "c2"},
        ],
    )
    model = FakeModel([first, AIMessage(content="ご案内します。")])
    res = await agent.run_agent_turn("u1", "注文1001について教えて", None, model=model)

    assert len(res.tool_runs) == 2
    ok_by_id = {r.tool_call_id: r.ok for r in res.tool_runs}
    assert ok_by_id["c1"] is True
    assert ok_by_id["c2"] is False

    # _exception_to_tool_run が組み立てる ToolMessage の中身も軽く固定しておく
    # (ok と tool_call_id だけでは、本文が空文字や status="success" のままでも
    # 検出できないため)。
    c2_run = next(r for r in res.tool_runs if r.tool_call_id == "c2")
    assert c2_run.tool_message.status == "error"
    assert c2_run.tool_message.content  # 空文字ではない

    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "tool", "assistant"]
    assert {m.tool_call_id for m in msgs if m.role == "tool"} == {"c1", "c2"}


async def test_system_prompt_survives_when_history_is_trimmed(monkeypatch):
    # token_budget を極端に小さくして、trim_history が実際に history を間引くことを強制する。
    monkeypatch.setattr(agent.settings, "token_budget", 5)
    rows = [
        SimpleNamespace(role="user", content=f"注文について質問その{i}です" * 5, tool_calls=None)
        for i in range(20)
    ]
    result = agent._build_history(rows)
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == AGENT_SYSTEM
    # 実際に間引かれたこと(system 1件 + 全20件がそのまま残ったのでは検証にならない)を確認する
    assert len(result) < len(rows) + 1
