"""/api/agent/stream (SSE) と /api/agent (非ストリーミング JSON) のテスト。

このモジュールは非同期(DB を触る 12A)と同期(monkeypatch で run_agent_turn を差し替える
12B)のテストが同居する。そのためモジュール先頭に
`pytestmark = pytest.mark.asyncio(loop_scope="session")` は置かない: 同期テストにまで
asyncio マーカーが付き "is marked with '@pytest.mark.asyncio' but it is not an async
function" の PytestWarning が出るため(実測で確認済み)。非同期テストにだけ
`@pytest.mark.asyncio(loop_scope="session")` をデコレータで個別に付ける
(_test_engine がセッションスコープのループ上にあるため、loop_scope の指定自体は必須。
tests/conftest.py と pyproject.toml のコメント参照)。
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from app.api import agent as agent_api
from app.core import agent
from app.core.agent import AgentResult
from app.db import repository as repo
from app.main import app
from app.tools.infra import ToolRun
from tests.test_agent_orchestration import FakeModel

session_loop = pytest.mark.asyncio(loop_scope="session")

_NOT_FOUND_MSG = "会話が見つかりません"


def teardown_function():
    app.dependency_overrides.clear()


def _use_model(model) -> None:
    app.dependency_overrides[agent_api.get_model] = lambda: model


async def _post_stream(payload: dict) -> tuple[int, str, str]:
    """/api/agent/stream を叩き、(status_code, content_type, 生のボディ文字列) を返す。

    行単位ではなく生バイトを読むのは、`event: error` フレームがワイヤ上で
    SSE の1フレームとして正しく組み上がっているかを確認するため。
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/api/agent/stream", json=payload) as resp:
            body = (await resp.aread()).decode("utf-8")
            return resp.status_code, resp.headers.get("content-type", ""), body


def _data_payloads(body: str) -> list[dict]:
    """`data: ` 行のうち [DONE] 以外を JSON として取り出す。"""
    out = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload != "[DONE]":
            out.append(json.loads(payload))
    return out


# --- 12A: /api/agent/stream -------------------------------------------------


@session_loop
async def test_stream_endpoint_emits_tool_deltas_done_and_terminator(
    db_session_factory, db_clean
):
    first = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
    )
    _use_model(FakeModel([first], stream_tokens=["注文 1001 は", "現在", "輸送中です。"]))

    status, content_type, body = await _post_stream(
        {"user_id": "u1", "message": "注文 1001 は今どこですか"}
    )
    assert status == 200
    assert content_type.startswith("text/event-stream")

    payloads = _data_payloads(body)
    assert payloads[0] == {"event": "tool", "name": "query_logistics"}

    deltas = [p["delta"] for p in payloads if "delta" in p]
    assert "".join(deltas) == "注文 1001 は現在輸送中です。"
    assert len(deltas) > 1  # 一括ではなく token ごとに届く

    done = payloads[-1]
    assert done["event"] == "done"
    assert isinstance(done["conversation_id"], int)

    assert body.endswith("data: [DONE]\n\n")

    msgs = await repo.list_messages(done["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]


@session_loop
async def test_stream_endpoint_without_tools_has_no_tool_frame(db_session_factory, db_clean):
    _use_model(FakeModel([AIMessage(content="こんにちは。どのようなご用件でしょうか?")]))

    status, _, body = await _post_stream({"user_id": "u1", "message": "こんにちは"})
    assert status == 200

    payloads = _data_payloads(body)
    assert not any(p.get("event") == "tool" for p in payloads)
    deltas = [p["delta"] for p in payloads if "delta" in p]
    assert deltas == ["こんにちは。どのようなご用件でしょうか?"]

    done = payloads[-1]
    assert done["event"] == "done"
    assert body.endswith("data: [DONE]\n\n")

    msgs = await repo.list_messages(done["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant"]


@session_loop
async def test_stream_endpoint_unknown_conversation_emits_error_frame_without_done(
    db_session_factory, db_clean
):
    """ConversationNotFound はジェネレータの内側、つまり HTTP 200 とヘッダを送出した後に
    発生するため 404 にはできない。クライアントには `event: error` フレームが届き、
    正常終了の印である [DONE] は届かないことを固定する。"""
    _use_model(FakeModel([AIMessage(content="hi")]))

    status, content_type, body = await _post_stream(
        {"user_id": "u1", "message": "hi", "conversation_id": 999999}
    )
    assert status == 200  # ストリーム開始後の失敗なのでステータスは変えられない
    assert content_type.startswith("text/event-stream")

    # ワイヤ上で SSE の1フレームとして正しく組み上がっていること
    # (_sse_error は2回に分けて yield するが、連結が1フレームになる必要がある)
    expected_frame = "event: error\ndata: {}\n\n".format(
        json.dumps({"message": _NOT_FOUND_MSG}, ensure_ascii=False)
    )
    assert expected_frame in body

    assert [p["message"] for p in _data_payloads(body)] == [_NOT_FOUND_MSG]
    assert "[DONE]" not in body


@session_loop
async def test_stream_endpoint_rejects_empty_message_before_stream_starts(
    db_session_factory, db_clean
):
    _use_model(FakeModel([AIMessage(content="hi")]))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/agent/stream", json={"user_id": "u1", "message": ""})

    assert resp.status_code == 422
    # ストリームは始まっておらず、FastAPI のバリデーションエラー JSON が返る
    assert resp.headers["content-type"].startswith("application/json")
    assert "detail" in resp.json()
    assert "event: error" not in resp.text


# --- 12B: /api/agent --------------------------------------------------------


def _fake_result() -> AgentResult:
    tm = ToolMessage(
        content='{"status": "輸送中"}', tool_call_id="c1", name="query_logistics"
    )
    return AgentResult(
        conversation_id=12,
        answer="注文 1001 は現在輸送中です。",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
        tool_runs=[ToolRun("c1", "query_logistics", True, tm)],
    )


def test_agent_endpoint_returns_tool_trace(monkeypatch):
    async def fake_run(user_id, message, conversation_id, model=None):
        return _fake_result()

    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "注文 1001 は今どこですか"})
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == 12
    assert body["answer"] == "注文 1001 は現在輸送中です。"
    assert body["tool_calls"][0]["name"] == "query_logistics"
    assert body["tool_calls"][0]["id"] == "c1"
    assert body["tool_calls"][0]["args"] == {"order_id": "1001"}
    assert body["tool_results"][0]["ok"] is True
    assert body["tool_results"][0]["tool_call_id"] == "c1"
    assert body["tool_results"][0]["name"] == "query_logistics"
    assert body["tool_results"][0]["content"] == '{"status": "輸送中"}'


def test_agent_endpoint_survives_tool_call_with_none_id(monkeypatch):
    """id=None の tool_call(langchain_core の ToolCall.id: str | None。id を省略する
    OpenAI 互換ゲートウェイ経由で実際に到達しうる)でも、成功したターンを日本語メッセージの
    ない素の 500 で捨てない。app/api/agent.py の `.get("id") or ""` の回帰テスト。"""

    async def fake_run(user_id, message, conversation_id, model=None):
        res = _fake_result()
        res.tool_calls = [{"name": "query_logistics", "args": {"order_id": "1001"}, "id": None}]
        return res

    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "注文 1001 は今どこですか"})
    assert r.status_code == 200
    assert r.json()["tool_calls"][0]["id"] == ""


def test_agent_endpoint_404_on_unknown_conversation(monkeypatch):
    async def fake_run(*a, **k):
        raise agent.ConversationNotFound(999)

    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post(
        "/api/agent", json={"user_id": "u1", "message": "hi", "conversation_id": 999}
    )
    assert r.status_code == 404
    assert r.json()["detail"] == _NOT_FOUND_MSG


def test_agent_endpoint_502_on_upstream_failure(monkeypatch):
    async def fake_run(*a, **k):
        raise RuntimeError("上流モデル呼び出し失敗(テスト用)")

    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 502


def test_agent_endpoint_422_on_missing_fields(monkeypatch):
    # run_agent_turn を「呼ばれたら失敗する」フェイクに差し替えるのは、この検証が
    # バリデーションだけを対象にしていることを明示するためだけではなく、安全装置でもある。
    # 差し替えないと、スキーマ側のバリデーションが壊れた瞬間にこのテストが本物の
    # オーケストレーションへ流れ落ち、本番相当の support DB へ書き込み、実際に上流 LLM を
    # 呼んでしまう(実測で確認済み: user_id を必須から外すミューテーションで実際に発生した)。
    async def must_not_run(*a, **k):
        raise AssertionError("バリデーションで弾かれるべきリクエストが本体まで到達した")

    monkeypatch.setattr(agent, "run_agent_turn", must_not_run)
    client = TestClient(app)
    assert client.post("/api/agent", json={"message": "hi"}).status_code == 422
