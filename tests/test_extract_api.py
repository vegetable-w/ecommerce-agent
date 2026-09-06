from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from app.api import extract as extract_api
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
    client = override(RunnableLambda(lambda _: ticket))
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

def test_extract_validates_empty_text():
    client = override(RunnableLambda(lambda _: None))
    assert client.post("/api/extract", json={"text": ""}).status_code == 422

def test_extract_validates_blank_text():
    client = override(RunnableLambda(lambda _: None))
    assert client.post("/api/extract", json={"text": "   "}).status_code == 422
