import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from app.api import extract as extract_api
from app.config import Settings, settings
from app.main import app
from app.schemas.extract import AfterSalesTicket

def override(runnable):
    app.dependency_overrides[extract_api.get_extractor] = lambda: runnable
    return TestClient(app)

def teardown_function():
    app.dependency_overrides.clear()

def test_extract_returns_structured_json():
    ticket = AfterSalesTicket(
        order_id="MH20260701123", request_type="返金", expected_solution="到着時に破損していたため返金を希望"
    )
    envelope = {"raw": AIMessage(content="dummy"), "parsed": ticket, "parsing_error": None}
    client = override(RunnableLambda(lambda _: envelope))
    resp = client.post("/api/extract", json={"text": "注文 MH20260701123 が壊れていたので返金してほしい"})
    assert resp.status_code == 200
    assert resp.json() == {
        "order_id": "MH20260701123",
        "request_type": "返金",
        "expected_solution": "到着時に破損していたため返金を希望",
    }

def test_extract_upstream_failure_returns_502():
    def boom(_):
        raise RuntimeError("upstream down")

    client = override(RunnableLambda(boom))
    resp = client.post("/api/extract", json={"text": "適当に何か話す"})
    assert resp.status_code == 502
    assert "detail" in resp.json()

def test_extract_parsing_error_returns_500():
    envelope = {
        "raw": AIMessage(content="スキーマに合わない生の応答"),
        "parsed": None,
        "parsing_error": ValueError("schema mismatch"),
    }
    client = override(RunnableLambda(lambda _: envelope))
    resp = client.post("/api/extract", json={"text": "適当に何か話す"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "抽出結果の解析に失敗しました"
    # 内部エラー文字列や生の応答内容をレスポンスへ漏らさない
    assert "schema mismatch" not in resp.text
    assert "スキーマに合わない生の応答" not in resp.text

def test_extract_validates_empty_text():
    client = override(RunnableLambda(lambda _: None))
    assert client.post("/api/extract", json={"text": ""}).status_code == 422

def test_extract_validates_blank_text():
    client = override(RunnableLambda(lambda _: None))
    assert client.post("/api/extract", json={"text": "   "}).status_code == 422

def test_invalid_extract_method_rejected(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("EXTRACT_METHOD", "not_a_real_method")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_get_extractor_binds_configured_method(monkeypatch):
    # get_extractor()は常にテストで差し替えられるため、実際の構成
    # (EXTRACT_PROMPT | model.with_structured_output(...))はどのテストでも
    # 構築されない。ここではget_chat_model()をスタブに差し替え、
    # with_structured_output()へ渡されるkwargsを直接捕捉することで、
    # settings.extract_methodとinclude_raw=Trueが実際にモデル呼び出しまで
    # 届いていることを検証する。langchain_openaiの内部実装(RunnableParallel/
    # RunnableBindingの内部属性名)には一切依存しない
    monkeypatch.setattr(settings, "extract_method", "function_calling")

    captured: dict = {}

    class StubModel:
        def with_structured_output(self, schema, **kwargs):
            captured["schema"] = schema
            captured.update(kwargs)
            return RunnableLambda(lambda x: x)

    monkeypatch.setattr(extract_api, "get_chat_model", lambda: StubModel())

    extractor = extract_api.get_extractor()

    assert isinstance(extractor, Runnable)
    assert captured["method"] == settings.extract_method
    assert captured["include_raw"] is True
