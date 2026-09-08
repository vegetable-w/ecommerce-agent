"""GET /api/conversations と GET /api/conversations/{id}/messages のテスト。

この 2 本は**読み取り専用**で、07 章の受け入れ検証(会話を切り替えて、前の会話の
文脈が保たれているか人が確かめる)のために足した。書き込む経路は持たない。

repository は各テストで差し替える。差し替え忘れは _no_real_db が到達不能な
ホストで落とす(黙って本番相当の support を読ませない)。real DB を使う検証は
tests/test_repository.py 側にある: TestClient の anyio portal と pytest-asyncio の
event loop は同じ接続を共有できず、ここから DB へ繋ぐと "attached to a different
loop" になる。
"""

from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import conversations
from app.core import labels

# router だけを載せた最小の app で叩く(tests/test_actions_api.py と同じ理由)。
# 本番 app へ登録されているかは tests/test_main_lifespan.py が見る。
_app = FastAPI()
_app.include_router(conversations.router)
client = TestClient(_app)

_LIST_URL = "/api/conversations"
_DB_DOWN_MSG = "会話一覧を一時的に取得できません。しばらくしてからもう一度お試しください"

# 到達不能なポート。ここへの接続は即座に拒否され、SQLAlchemyError になる。
_DEAD_DB_URL = "mysql+asyncmy://root:root@127.0.0.1:59999/nonexistent_db"

_UPDATED = datetime(2026, 9, 8, 12, 34, 56)


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """本番相当の DB を既定で塞ぐ。差し替え忘れは SQLAlchemyError で落ちる。"""
    monkeypatch.setattr(
        "app.db.base.async_session",
        async_sessionmaker(create_async_engine(_DEAD_DB_URL), expire_on_commit=False),
    )


class _Msg:
    """repository.list_dialog_messages が返す ORM 行の代わり。"""

    def __init__(self, role: str, content: str | None):
        self.role = role
        self.content = content
        self.created_at = _UPDATED


def _spy_list(monkeypatch, rows: list[dict] | None = None) -> list[tuple]:
    """repository.list_conversations を記録用の偽物へ差し替える。"""
    calls: list[tuple] = []

    async def _fake(user_id, limit=50):
        calls.append((user_id, limit))
        return rows if rows is not None else []

    monkeypatch.setattr(conversations.repository, "list_conversations", _fake)
    return calls


def _row(**over) -> dict:
    row = {"id": 12, "status": "in_progress", "preview": "返品したいのですが",
           "has_summary": True, "updated_at": _UPDATED}
    row.update(over)
    return row


# --- 一覧 ----------------------------------------------------------------------


def test_list_passes_the_user_id_through(monkeypatch):
    """絞り込みの条件は repository へそのまま渡す。ここで解釈しない。"""
    calls = _spy_list(monkeypatch)
    assert client.get(_LIST_URL, params={"user_id": "web-abc"}).status_code == 200
    assert calls == [("web-abc", 50)]


def test_list_serializes_every_field_the_sidebar_needs(monkeypatch):
    _spy_list(monkeypatch, [_row()])
    items = client.get(_LIST_URL, params={"user_id": "u1"}).json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == 12
    assert items[0]["preview"] == "返品したいのですが"
    assert items[0]["has_summary"] is True
    assert items[0]["updated_at"].startswith("2026-09-08T12:34:56")


def test_status_comes_from_the_label_table(monkeypatch):
    """表示名の出所を 1 つに保つ(app/api/actions.py と同じ規約)。

    画面側に英語 → 日本語の対応表を持たせると、labels を直したときに
    ここと画面で別の日本語が出る。
    """
    _spy_list(monkeypatch, [_row(status="escalated")])
    items = client.get(_LIST_URL, params={"user_id": "u1"}).json()["items"]
    assert items[0]["status"] == labels.label(labels.CONVERSATION_STATUS, "escalated")


def test_list_of_a_user_without_conversations_is_empty_not_an_error(monkeypatch):
    _spy_list(monkeypatch, [])
    res = client.get(_LIST_URL, params={"user_id": "u1"})
    assert res.status_code == 200 and res.json()["items"] == []


def test_list_requires_a_user_id(monkeypatch):
    """user_id が無いと全ユーザーの会話が並ぶ。落とすのではなく弾く。"""
    calls = _spy_list(monkeypatch)
    assert client.get(_LIST_URL).status_code == 422
    assert calls == []


def test_list_limit_is_capped(monkeypatch):
    """上限を外から無制限に広げられないこと(1 リクエストで全履歴を引かせない)。"""
    calls = _spy_list(monkeypatch)
    assert client.get(_LIST_URL, params={"user_id": "u1", "limit": 999}).status_code == 422
    assert calls == []


def test_list_returns_503_when_the_database_is_down(monkeypatch):
    """差し替えずに叩く。_no_real_db の到達不能な DB がそのまま障害になる。"""
    res = client.get(_LIST_URL, params={"user_id": "u1"})
    assert res.status_code == 503
    assert res.json()["detail"] == _DB_DOWN_MSG


def test_list_does_not_leak_the_database_error_to_the_client(monkeypatch):
    """接続文字列や SQL が detail に混ざらないこと。"""
    async def _boom(user_id, limit=50):
        raise OperationalError("SELECT 1", {}, Exception("root:secret@10.0.0.1:3306"))

    monkeypatch.setattr(conversations.repository, "list_conversations", _boom)
    detail = client.get(_LIST_URL, params={"user_id": "u1"}).json()["detail"]
    assert detail == _DB_DOWN_MSG
    assert "secret" not in detail and "3306" not in detail


# --- 履歴 ----------------------------------------------------------------------


def _spy_messages(monkeypatch, rows: list[_Msg], conv=object()) -> list[int]:
    calls: list[int] = []

    async def _get(conversation_id):
        return conv

    async def _list(conversation_id):
        calls.append(conversation_id)
        return rows

    monkeypatch.setattr(conversations.repository, "get_conversation", _get)
    monkeypatch.setattr(conversations.repository, "list_dialog_messages", _list)
    return calls


def test_messages_returns_the_dialog_in_order(monkeypatch):
    _spy_messages(monkeypatch, [_Msg("user", "注文1001はどこ"), _Msg("assistant", "配送中です")])
    items = client.get("/api/conversations/7/messages").json()["items"]
    assert [(i["role"], i["content"]) for i in items] == [
        ("user", "注文1001はどこ"), ("assistant", "配送中です")]
    assert items[0]["created_at"].startswith("2026-09-08T12:34:56")


def test_messages_role_stays_the_english_identifier(monkeypatch):
    """role は日本語ラベルにしない。画面がどちら側の吹き出しかを決める鍵になる。"""
    _spy_messages(monkeypatch, [_Msg("user", "こんにちは")])
    assert client.get("/api/conversations/7/messages").json()["items"][0]["role"] == "user"


def test_messages_skips_rows_without_content(monkeypatch):
    """本文の無い assistant 行は返さない。

    log_node は回答が空のターンを content=NULL で残す。そのまま返すと、
    履歴を読み直した画面に空の吹き出しが毎ターン並ぶ。
    """
    _spy_messages(monkeypatch, [
        _Msg("user", "こんにちは"), _Msg("assistant", None), _Msg("assistant", "  ")])
    items = client.get("/api/conversations/7/messages").json()["items"]
    assert [i["content"] for i in items] == ["こんにちは"]


def test_messages_of_a_missing_conversation_is_404(monkeypatch):
    calls = _spy_messages(monkeypatch, [], conv=None)
    res = client.get("/api/conversations/999999/messages")
    assert res.status_code == 404
    assert res.json()["detail"] == "会話が見つかりません"
    # 存在しないと分かった時点で止める(無い会話の履歴を引きにいかない)
    assert calls == []


def test_messages_returns_503_when_the_database_is_down(monkeypatch):
    """差し替えずに叩く。会話の有無を確かめる問い合わせ自体が落ちる。

    ここで 404 に倒すと、実際には在る会話に「見つかりません」と答えてしまう。
    """
    res = client.get("/api/conversations/7/messages")
    assert res.status_code == 503
    assert res.json()["detail"] == "会話履歴を一時的に取得できません。しばらくしてからもう一度お試しください"


def test_messages_does_not_leak_the_database_error_to_the_client(monkeypatch):
    async def _get(conversation_id):
        return object()

    async def _boom(conversation_id):
        raise OperationalError("SELECT 1", {}, Exception("root:secret@10.0.0.1:3306"))

    monkeypatch.setattr(conversations.repository, "get_conversation", _get)
    monkeypatch.setattr(conversations.repository, "list_dialog_messages", _boom)
    detail = client.get("/api/conversations/7/messages").json()["detail"]
    assert "secret" not in detail and "3306" not in detail


# --- 契約 ----------------------------------------------------------------------


def test_endpoints_are_published_in_the_openapi_schema():
    """画面側がこの契約を読む。パスと項目が変わったら気づけるようにする。"""
    schema = _app.openapi()
    assert _LIST_URL in schema["paths"]
    assert "/api/conversations/{conversation_id}/messages" in schema["paths"]

    ref = (schema["paths"][_LIST_URL]["get"]["responses"]["200"]["content"]
           ["application/json"]["schema"]["$ref"])
    body = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    item_ref = body["properties"]["items"]["items"]["$ref"]
    item = schema["components"]["schemas"][item_ref.rsplit("/", 1)[-1]]
    assert set(item["properties"]) == {"id", "status", "preview", "has_summary", "updated_at"}


def test_the_endpoints_are_read_only():
    """このモジュールに書き込みの入口を足さない。

    会話の一覧と履歴は受け入れ検証のための覗き窓で、ここから会話を消したり
    状態を変えたりできるようにすると、画面の事故がそのまま DB に残る。
    """
    for path in _app.openapi()["paths"].values():
        assert set(path) == {"get"}
