import json

from fastapi.testclient import TestClient
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk

from app.api import chat as chat_api
from app.main import app


class _FakeBlockContentModel:
    """.contentがブロック形式(list[dict])のチャンクを返す簡易フェイク。
    新しめのLLMプロバイダが返すcontent-block形式を模し、
    isinstance(.content, str)判定が無音で空文字列を返す回帰を検出する。"""

    def __init__(self, text: str) -> None:
        self._text = text

    async def astream(self, messages):
        for ch in self._text:
            yield AIMessageChunk(content=[{"type": "text", "text": ch}])


class _FakeEmptyModel:
    """1文字も生成しないフェイク(空応答)。"""

    async def astream(self, messages):
        return
        yield  # noqa: unreachable - astreamを非同期ジェネレータにするためだけの行


class _FakeRaisingModel:
    """astream中に例外を送出するフェイク。"""

    async def astream(self, messages):
        raise RuntimeError("上流モデル呼び出し失敗(テスト用)")
        yield  # noqa: unreachable - astreamを非同期ジェネレータにするためだけの行


class _FakeRaisingAfterYieldModel:
    """いくつかチャンクを送出した後に例外を送出するフェイク。
    部分応答が既にクライアントへ届いた後の中断を模す。"""

    async def astream(self, messages):
        for ch in ["部", "分", "応答"]:
            yield AIMessageChunk(content=ch)
        raise RuntimeError("上流モデル呼び出し失敗(部分応答後・テスト用)")


def make_client(responses: list[str]) -> TestClient:
    fake = FakeListChatModel(responses=responses)
    app.dependency_overrides[chat_api.get_model] = lambda: fake
    return TestClient(app)


def make_client_with_model(model) -> TestClient:
    app.dependency_overrides[chat_api.get_model] = lambda: model
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
    chat_api.store.clear()


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


def test_chat_streams_block_style_content():
    # レビュー指摘1の回帰テスト: .contentがブロック形式(list[dict])でも
    # デルタが届き、履歴にも保存されることを確認する
    client = make_client_with_model(_FakeBlockContentModel("こんにちは！"))
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s7", "message": "質問"}
    ) as resp:
        assert resp.status_code == 200
        deltas, last = collect_sse(resp)
    assert "".join(deltas) == "こんにちは！"
    assert len(deltas) > 0
    assert last == "[DONE]"
    history = chat_api.store.get("s7")
    assert [m.content for m in history] == ["質問", "こんにちは！"]


def test_chat_does_not_store_empty_reply():
    client = make_client_with_model(_FakeEmptyModel())
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s8", "message": "質問"}
    ) as resp:
        assert resp.status_code == 200
        deltas, last = collect_sse(resp)
    assert deltas == []
    assert last == "[DONE]"
    assert chat_api.store.get("s8") == []


def test_chat_error_stream_ends_with_done():
    client = make_client_with_model(_FakeRaisingModel())
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s9", "message": "質問"}
    ) as resp:
        assert resp.status_code == 200
        raw = list(resp.iter_lines())
    assert any(line == "event: error" for line in raw)
    data_lines = [line[len("data: "):] for line in raw if line.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"


def test_chat_mid_stream_failure_does_not_persist_partial_reply():
    # 部分応答を送出した後に例外が起きた場合、クライアントは既に届いたデルタと
    # エラーイベント・[DONE]を受け取るが、その部分応答は履歴に保存されない
    # (中断された応答をアシスタントの発言として記憶してはならないという設計判断)。
    client = make_client_with_model(_FakeRaisingAfterYieldModel())
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s10", "message": "質問"}
    ) as resp:
        assert resp.status_code == 200
        raw = list(resp.iter_lines())
    assert any(line == "event: error" for line in raw)
    data_lines = [line[len("data: "):] for line in raw if line.startswith("data: ")]
    payloads = [json.loads(p) for p in data_lines if p != "[DONE]"]
    deltas = [p["delta"] for p in payloads if "delta" in p]
    assert deltas == ["部", "分", "応答"]
    assert data_lines[-1] == "[DONE]"
    assert chat_api.store.get("s10") == []
