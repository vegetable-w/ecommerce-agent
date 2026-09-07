"""04 章 Task 11: 引用イベントと、根拠不足のときの低信頼プール投入。

DB を触るテストが混ざるため、非同期テストには loop_scope="session" のマーカーを
個別に付ける(tests/conftest.py と tests/test_agent_api.py の説明を参照)。
"""

import json

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import select

from app.api import agent as agent_api
from app.core import agent
from app.db import repository as repo
from app.db.models import LowConfidenceQuestion
from app.main import app
from app.tools.infra import ToolRun
from tests.test_agent_orchestration import FakeModel

session_loop = pytest.mark.asyncio(loop_scope="session")

_CITATION = {"n": 1, "id": 5, "section_path": "送料ポリシー", "question": "送料",
             "answer": "3,000円以上で送料無料", "content_type": "faq"}


class _FakeToolRun:
    """infra.ToolRun と同じ形(name / tool_message.content)だけを持つ最小の代役。"""

    def __init__(self, name, payload, cid="tc1"):
        self.name = name
        self.tool_call_id = cid
        self.ok = True
        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.tool_message = type("_M", (), {"content": content})()


def teardown_function():
    app.dependency_overrides.clear()


def _faq_call(tool_call_id="c1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "query_faq", "args": {"keyword": "送料"}, "id": tool_call_id}],
    )


def _stub_faq(monkeypatch, payload, name="query_faq"):
    """query_faq ツールの実行結果を差し替える(検索・上流を一切呼ばない)。

    本物の ToolRun を返すこと。_prepare_turn は isinstance(r, ToolRun) で判定し、
    違うものはツール失敗として握り潰すので、代役の dataclass では素通りしてしまう。
    """
    async def fake_execute(tool_call, conversation_id, **kw):
        tc_id = tool_call.get("id") or "unknown"
        content = json.dumps(payload, ensure_ascii=False)
        return ToolRun(
            tool_call_id=tc_id, name=name, ok=True,
            tool_message=ToolMessage(content=content, tool_call_id=tc_id, name=name),
        )

    monkeypatch.setattr(agent, "execute_tool_call", fake_execute)


# --- helper 単体 --------------------------------------------------------------


def test_extract_faq_result():
    runs = [_FakeToolRun("query_faq", {"sufficient": True, "evidence": "[1] x",
                                       "citations": [_CITATION]})]
    faq = agent._faq_result(runs)
    assert faq["sufficient"] is True and faq["citations"][0]["id"] == 5


def test_extract_faq_none_when_absent():
    assert agent._faq_result([_FakeToolRun("query_order", {"x": 1})]) is None


def test_extract_faq_none_when_tool_errored():
    """ツールが失敗すると content は JSON ではなく日本語のエラー文になる。
    ここで例外を漏らすとターン全体が 500 に化ける。"""
    assert agent._faq_result([_FakeToolRun("query_faq", "ツール実行失敗: 一時的なエラー")]) is None


def test_extract_faq_none_when_content_is_not_an_object():
    """json.loads は数値や文字列も通す。dict でなければ扱わない。"""
    assert agent._faq_result([_FakeToolRun("query_faq", "123")]) is None


def test_extract_faq_picks_query_faq_among_several_tools():
    runs = [_FakeToolRun("query_order", {"x": 1}, cid="a"),
            _FakeToolRun("query_faq", {"sufficient": True, "citations": [_CITATION]}, cid="b")]
    assert agent._faq_result(runs)["citations"][0]["id"] == 5


# --- ストリーミング出口 --------------------------------------------------------


@session_loop
async def test_stream_emits_citations_event_before_the_answer(
    monkeypatch, db_session_factory, db_clean
):
    """引用イベントが実際に流れること。ここが落ちるとフロントエンドは
    引用の中身を得る手段が無くなり、回答内の [1] がただの文字列になる。"""
    _stub_faq(monkeypatch, {"sufficient": True, "evidence": "[1] 送料: 3,000円以上で送料無料",
                            "citations": [_CITATION]})
    model = FakeModel([_faq_call()], stream_tokens=["3,000円以上で", "送料無料です[1]。"])
    events = [ev async for ev in agent.stream_agent_turn("u1", "送料はいくら?", None, model=model)]

    types = [e["type"] for e in events]
    assert types.count("citations") == 1
    cit = events[types.index("citations")]
    assert cit["items"] == [_CITATION]
    # tool フレームの後、最初の delta より前に届くこと(フロントエンドが回答を
    # 描き始める前に引用元を握っておけるように)
    assert types.index("tool") < types.index("citations") < types.index("delta")


@session_loop
async def test_stream_insufficient_inserts_low_confidence_and_sends_no_citations(
    monkeypatch, db_session_factory, db_clean
):
    _stub_faq(monkeypatch, {"sufficient": False, "source": "self_check",
                            "reason": "型番の質問だが根拠は送料のみ", "citations": [],
                            "notice": "..."})
    model = FakeModel([_faq_call()], stream_tokens=["確認できませんでした。"])
    events = [ev async for ev in
              agent.stream_agent_turn("u1", "Pro は自動清掃できますか", None, model=model)]

    assert not any(e["type"] == "citations" for e in events)
    conversation_id = events[-1]["conversation_id"]

    async with db_session_factory() as s:
        rows = (await s.execute(select(LowConfidenceQuestion))).scalars().all()
    assert len(rows) == 1
    assert rows[0].raw_question == "Pro は自動清掃できますか"  # ユーザーの原文
    assert rows[0].source == "self_check"
    assert rows[0].reason == "型番の質問だが根拠は送料のみ"
    assert rows[0].conversation_id == conversation_id


@session_loop
async def test_stream_retrieval_low_conf_keeps_the_real_reason(
    monkeypatch, db_session_factory, db_clean
):
    """機械ゲート側の理由もそのままプールへ落ちること。人がプールを見るときに
    「惜しかったのか全く外れていたのか」を判断する材料はこの 1 行しかない。"""
    _stub_faq(monkeypatch, {"sufficient": False, "source": "retrieval_low_conf",
                            "reason": "リランクの最高スコアが閾値未満(top=0.123)",
                            "citations": [], "notice": "..."})
    model = FakeModel([_faq_call()], stream_tokens=["確認できませんでした。"])
    [ev async for ev in agent.stream_agent_turn("u1", "火星探査車", None, model=model)]

    async with db_session_factory() as s:
        rows = (await s.execute(select(LowConfidenceQuestion))).scalars().all()
    assert rows[0].source == "retrieval_low_conf"
    assert "0.123" in rows[0].reason


@session_loop
async def test_stream_without_faq_tool_touches_nothing(
    monkeypatch, db_session_factory, db_clean
):
    _stub_faq(monkeypatch, {"status": "輸送中"}, name="query_logistics")

    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"tracking_no": "JP213502378238"}, "id": "c1"}])
    events = [ev async for ev in
              agent.stream_agent_turn("u1", "1001 はどこ?", None, model=FakeModel(
                  [first], stream_tokens=["輸送中です。"]))]

    assert not any(e["type"] == "citations" for e in events)
    async with db_session_factory() as s:
        assert (await s.execute(select(LowConfidenceQuestion))).scalars().all() == []


# --- 非ストリーミング出口 ------------------------------------------------------


@session_loop
async def test_run_agent_turn_returns_citations(monkeypatch, db_session_factory, db_clean):
    _stub_faq(monkeypatch, {"sufficient": True, "evidence": "[1] x", "citations": [_CITATION]})
    model = FakeModel([_faq_call(), AIMessage(content="3,000円以上で送料無料です[1]。")])
    res = await agent.run_agent_turn("u1", "送料はいくら?", None, model=model)
    assert res.citations == [_CITATION]


@session_loop
async def test_run_agent_turn_inserts_low_confidence(monkeypatch, db_session_factory, db_clean):
    _stub_faq(monkeypatch, {"sufficient": False, "source": "retrieval_low_conf",
                            "reason": "検索で根拠が 1 件も得られなかった", "citations": [],
                            "notice": "..."})
    model = FakeModel([_faq_call(), AIMessage(content="確認できませんでした。")])
    res = await agent.run_agent_turn("u1", "火星探査車はどう買えますか", None, model=model)
    assert res.citations == []

    async with db_session_factory() as s:
        rows = (await s.execute(select(LowConfidenceQuestion))).scalars().all()
    assert len(rows) == 1 and rows[0].raw_question == "火星探査車はどう買えますか"


@session_loop
async def test_agent_result_defaults_to_no_citations(db_session_factory, db_clean):
    """ツールを使わないターンでも citations は必ず存在する(欠落で落とさない)。"""
    res = await agent.run_agent_turn("u1", "こんにちは", None,
                                     model=FakeModel([AIMessage(content="こんにちは。")]))
    assert res.citations == []


# --- SSE ワイヤ ----------------------------------------------------------------


async def _post_stream(payload: dict) -> str:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/api/agent/stream", json=payload) as resp:
            return (await resp.aread()).decode("utf-8")


def _frames(body: str) -> list[str]:
    """フロントエンド(static/index.html)と同じ切り出し: 空行 2 つでフレームを割る。"""
    return [f for f in body.split("\n\n") if f]


@session_loop
async def test_sse_forwards_citations_frame(monkeypatch, db_session_factory, db_clean):
    _stub_faq(monkeypatch, {"sufficient": True, "evidence": "[1] x", "citations": [_CITATION]})
    app.dependency_overrides[agent_api.get_model] = lambda: FakeModel(
        [_faq_call()], stream_tokens=["3,000円以上で送料無料です[1]。"])

    body = await _post_stream({"user_id": "u1", "message": "送料はいくら?"})
    payloads = [json.loads(f[len("data: "):]) for f in _frames(body)
                if f.startswith("data: ") and f != "data: [DONE]"]
    cits = [p for p in payloads if p.get("event") == "citations"]
    assert len(cits) == 1
    assert cits[0]["items"] == [_CITATION]


@session_loop
async def test_citations_frame_survives_newlines_in_the_cited_text(
    monkeypatch, db_session_factory, db_clean
):
    """引用本文に改行や空行が入ってもフレームが割れないこと。

    SSE のフレーム区切りは生の "\\n\\n" なので、本文をそのまま流すと 1 フレームが
    2 つに割れ、フロントエンドの JSON.parse が両方失敗して引用が丸ごと消える。
    json.dumps が改行を \\n へエスケープすることが唯一の防壁であり、ここはその回帰テスト
    (02 章の delta で同じ形のバグを踏んでいる)。
    """
    nasty = dict(_CITATION, answer="1 行目\n\n2 行目\r\n3 行目", section_path="送料\nポリシー")
    _stub_faq(monkeypatch, {"sufficient": True, "evidence": "[1] x", "citations": [nasty]})
    app.dependency_overrides[agent_api.get_model] = lambda: FakeModel(
        [_faq_call()], stream_tokens=["ご案内します[1]。"])

    body = await _post_stream({"user_id": "u1", "message": "送料はいくら?"})
    frames = _frames(body)
    # 生バイトの中に本文の改行がそのまま出ていないこと
    assert "1 行目\n\n2 行目" not in body
    cit_frames = [f for f in frames if '"citations"' in f]
    assert len(cit_frames) == 1
    assert cit_frames[0].count("\n") == 0  # 1 フレーム = 1 行
    obj = json.loads(cit_frames[0][len("data: "):])
    assert obj["items"][0]["answer"] == "1 行目\n\n2 行目\r\n3 行目"


@session_loop
async def test_sse_has_no_citations_frame_when_refused(
    monkeypatch, db_session_factory, db_clean
):
    _stub_faq(monkeypatch, {"sufficient": False, "source": "self_check", "reason": "無関係",
                            "citations": [], "notice": "..."})
    app.dependency_overrides[agent_api.get_model] = lambda: FakeModel(
        [_faq_call()], stream_tokens=["確認できませんでした。"])

    body = await _post_stream({"user_id": "u1", "message": "火星探査車"})
    assert "citations" not in body
    assert body.endswith("data: [DONE]\n\n")


# --- 回答拒否がターンを壊さないこと --------------------------------------------


@session_loop
async def test_refusal_survives_a_missing_conversation_row(
    monkeypatch, db_session_factory, db_clean
):
    """conversations への FK が外れていてもプール投入は例外にしない
    (repository 側で会話への紐付けだけ捨てる)。穏当な回答拒否が 500 に化けないこと。"""
    _stub_faq(monkeypatch, {"sufficient": False, "source": "self_check", "reason": "無関係",
                            "citations": [], "notice": "..."})
    real_insert = repo.insert_low_confidence

    async def insert_with_bogus_conversation(conversation_id, raw_question, source, reason):
        return await real_insert(9_999_999, raw_question, source, reason)

    monkeypatch.setattr(repo, "insert_low_confidence", insert_with_bogus_conversation)
    model = FakeModel([_faq_call()], stream_tokens=["確認できませんでした。"])
    events = [ev async for ev in agent.stream_agent_turn("u1", "火星探査車", None, model=model)]

    assert events[-1]["type"] == "done"
    async with db_session_factory() as s:
        rows = (await s.execute(select(LowConfidenceQuestion))).scalars().all()
    assert len(rows) == 1 and rows[0].conversation_id is None
