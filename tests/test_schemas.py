import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest
from app.schemas.extract import AfterSalesTicket, ExtractRequest, RequestType

def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="")

def test_chat_request_rejects_empty_session_id():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="", message="msg")

def test_chat_request_rejects_whitespace_only_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="   ")

def test_chat_request_rejects_whitespace_only_session_id():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="   ", message="msg")

def test_extract_request_rejects_empty_text():
    with pytest.raises(ValidationError):
        ExtractRequest(text="")

def test_extract_request_rejects_whitespace_only_text():
    with pytest.raises(ValidationError):
        ExtractRequest(text="   ")

def test_ticket_order_id_nullable():
    t = AfterSalesTicket(order_id=None, request_type="返金", expected_solution="全額返金")
    assert t.order_id is None
    assert t.request_type is RequestType.REFUND

def test_ticket_rejects_unknown_request_type():
    with pytest.raises(ValidationError):
        AfterSalesTicket(order_id=None, request_type="値下げ交渉", expected_solution="x")

def test_ticket_expected_solution_is_required():
    # expected_solution は必須。レスポンス契約は常に3フィールド全て含む。スキーマの required リストで LLM に生成を強く促す
    with pytest.raises(ValidationError):
        AfterSalesTicket(order_id=None, request_type="返金")
