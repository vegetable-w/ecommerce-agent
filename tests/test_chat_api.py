import json

from fastapi.testclient import TestClient
from langchain_core.language_models import FakeListChatModel

from app.api import chat as chat_api
from app.main import app


def make_client(responses: list[str]) -> TestClient:
    fake = FakeListChatModel(responses=responses)
    app.dependency_overrides[chat_api.get_model] = lambda: fake
    return TestClient(app)


def collect_sse(resp) -> tuple[list[str], str]:
    """(deltaのリスト, 終了フレーム)を返す。"""
    deltas, last = [], ""
    for line in resp.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            last = payload
        else:
            deltas.append(json.loads(payload)["delta"])
    return deltas, last


def teardown_function():
    app.dependency_overrides.clear()
    chat_api.store._sessions.clear()


def test_chat_streams_tokens_and_done():
    client = make_client(["こんにちは！"])
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s1", "message": "いますか？"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        deltas, last = collect_sse(resp)
    assert "".join(deltas) == "こんにちは！"
    assert len(deltas) > 1  # tokenごとに逐次送信し、一括で全文を返さない
    assert last == "[DONE]"


def test_chat_persists_history_for_next_turn():
    client = make_client(["1回目の回答", "2回目の回答"])
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s2", "message": "1つ目の質問"}
    ) as r:
        collect_sse(r)
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s2", "message": "2つ目の質問"}
    ) as r:
        collect_sse(r)
    history = chat_api.store.get("s2")
    assert [m.content for m in history] == ["1つ目の質問", "1回目の回答", "2つ目の質問", "2回目の回答"]


def test_chat_validates_empty_message():
    client = make_client(["x"])
    assert (
        client.post("/api/chat", json={"session_id": "s3", "message": ""}).status_code
        == 422
    )


def test_chat_validates_whitespace_only_message():
    # カード A: min_lengthは"   "を通してしまうため、field_validatorで空白のみを拒否する
    client = make_client(["x"])
    assert (
        client.post(
            "/api/chat", json={"session_id": "s4", "message": "   "}
        ).status_code
        == 422
    )


def test_chat_logs_when_trimming_drops_turns(caplog):
    import logging

    client = make_client(["短い回答1", "短い回答2", "短い回答3"])
    session_id = "s5"
    # token_budgetを小さく上書きし、次のターンで確実にトリムが発生するようにする
    from app.config import settings

    original_budget = settings.token_budget
    settings.token_budget = 5
    try:
        with caplog.at_level(logging.INFO, logger="app.api.chat"):
            with client.stream(
                "POST", "/api/chat", json={"session_id": session_id, "message": "1つ目の質問"}
            ) as r:
                collect_sse(r)
            with client.stream(
                "POST", "/api/chat", json={"session_id": session_id, "message": "2つ目の質問"}
            ) as r:
                collect_sse(r)
        assert any(session_id in rec.message for rec in caplog.records)
    finally:
        settings.token_budget = original_budget


def test_chat_no_trim_log_when_under_budget(caplog):
    import logging

    client = make_client(["短い回答"])
    with caplog.at_level(logging.INFO, logger="app.api.chat"):
        with client.stream(
            "POST", "/api/chat", json={"session_id": "s6", "message": "質問"}
        ) as r:
            collect_sse(r)
    assert not any("s6" in rec.message for rec in caplog.records)
