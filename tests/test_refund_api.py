"""POST /api/actions/create-refund(返金申請フォームの送信)のテスト。

返金申請は専用のテーブルを作らず、既存の tickets を ticket_type='refund' で再利用する
(spec §9 D2)。したがってここは **tickets へ書き込む 2 本目の経路**であり、
05 章の create-ticket と同じ作法(Literal での検証 / 404 と 503 の切り分け /
表示ラベルは labels から引く)に揃っていることを固定する。

Agent 側の submit_refund は agent_tools が横取りして画面の選択肢へ変えるだけで
DB には届かない。書き込みが起きるのはユーザーがフォームを送信してここへ POST した
ときだけなので、「押していないのに返金チケットが増えた」という事故は、ここの
バリデーションが緩んだときにしか起きない。

**本番相当の support DB へは絶対に書かない。** repository.create_ticket は各テストで
差し替える。差し替え忘れは _no_real_db が到達不能なホストで落とす。
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import actions
from app.core import labels

# router だけを載せた最小の app で叩く(tests/test_actions_api.py と同じ理由)。
_app = FastAPI()
_app.include_router(actions.router)
client = TestClient(_app)

_URL = "/api/actions/create-refund"
_DB_DOWN_MSG = "返金申請を一時的に受け付けられません。しばらくしてからもう一度お試しください"

# 到達不能なポート。ここへの接続は即座に拒否され、SQLAlchemyError になる。
_DEAD_DB_URL = "mysql+asyncmy://root:root@127.0.0.1:59999/nonexistent_db"

# 画面のドロップダウンに並ぶ固定の理由(spec §7)。増減したら画面と食い違うので、
# ここで一覧そのものを固定する。
_REASONS = ["7日以内の自己都合返品", "品質問題", "誤配送", "不要になった", "その他"]

# 空白のみと見なすべき入力。ASCII 空白だけでは足りない。U+3000(全角スペース)は
# 日本語 IME がそのまま出す「ありがちな空入力」で、これを通すと注文番号の分からない
# 返金チケットが残る(tests/test_actions_api.py と同じ一覧)。
_BLANK_VARIANTS = [
    pytest.param("   ", id="ascii-space"),
    pytest.param("　　", id="ideographic-space-u3000"),
    pytest.param("\t", id="tab"),
    pytest.param("\n", id="newline"),
    pytest.param(" ", id="nbsp"),
]


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """本番相当の DB を既定で塞ぐ(tests/test_actions_api.py と同じ)。

    create_ticket の差し替えを忘れたテストは support へ書き込むのではなく
    SQLAlchemyError で落ちる。DB 障害のテストはこの既定をそのまま使う。
    """
    monkeypatch.setattr(
        "app.db.base.async_session",
        async_sessionmaker(create_async_engine(_DEAD_DB_URL), expire_on_commit=False),
    )


def _spy_create_ticket(monkeypatch, ticket_no: str = "T20260908120000002") -> list[tuple]:
    """repository.create_ticket を記録用の偽物へ差し替える。呼ばれた引数を返す。"""
    calls: list[tuple] = []

    async def _fake(conversation_id, description, ticket_type):
        calls.append((conversation_id, description, ticket_type))
        return ticket_no

    monkeypatch.setattr(actions.repository, "create_ticket", _fake)
    return calls


def _body(**over) -> dict:
    body = {"conversation_id": 7, "order_id": "1001", "reason": "品質問題"}
    body.update(over)
    return body


def test_creates_a_refund_ticket_and_returns_ticket_no(monkeypatch):
    calls = _spy_create_ticket(monkeypatch)
    res = client.post(_URL, json=_body())
    assert res.status_code == 200
    assert res.json()["ticket_no"] == "T20260908120000002"
    assert len(calls) == 1
    conversation_id, description, ticket_type = calls[0]
    assert conversation_id == 7
    assert ticket_type == "refund"
    # 後から人が見て何の申請か分かること。注文番号と理由の両方が要る
    assert "1001" in description and "品質問題" in description


def test_uses_the_english_enum_identifier_not_the_japanese_label(monkeypatch):
    """DB へ書くのは英語の識別子。日本語を書くと ENUM 違反で落ちる(02 章の決定)。"""
    calls = _spy_create_ticket(monkeypatch)
    client.post(_URL, json=_body())
    assert calls[0][2] == "refund"
    assert calls[0][2] != labels.TICKET_TYPE["refund"]


def test_the_refund_type_is_declared_in_the_orm_and_the_label_table():
    """書こうとしている値が ORM の ENUM と表示用の対応表の両方にあること。

    ここが欠けると、テストは全部緑のまま本番の INSERT だけが落ちる。
    """
    from app.db.models import Ticket

    assert "refund" in Ticket.__table__.columns["ticket_type"].type.enums
    assert labels.TICKET_TYPE["refund"] == "返金"


def test_status_comes_from_the_label_table(monkeypatch):
    """表示名の出所を 1 つに保つ(05 章の create-ticket と同じ)。

    repository.create_ticket は会話の status も escalated へ動かすので、画面へ返す
    ラベルもチケット作成時と同じものになる。ここで別の文字列を書き起こすと、
    DB の状態と画面の表示がずれる。
    """
    _spy_create_ticket(monkeypatch)
    res = client.post(_URL, json=_body())
    assert res.json()["status"] == labels.label(labels.CONVERSATION_STATUS, "escalated")


@pytest.mark.parametrize("reason", _REASONS)
def test_accepts_every_fixed_reason(monkeypatch, reason):
    """画面のドロップダウンの選択肢がすべて通ること。1 つでも落ちると選べなくなる。"""
    calls = _spy_create_ticket(monkeypatch)
    assert client.post(_URL, json=_body(reason=reason)).status_code == 200
    assert reason in calls[0][1]


@pytest.mark.parametrize("reason", [
    pytest.param("返品したい", id="free-text"),
    pytest.param("品質問題です", id="near-miss-with-suffix"),
    pytest.param("quality", id="english"),
    pytest.param("", id="empty"),
    pytest.param("その他 ", id="trailing-space"),
])
def test_rejects_a_reason_outside_the_fixed_choices(monkeypatch, reason):
    """自由記述を通さない。理由は集計とオペレーションの分岐に使う固定の分類。"""
    calls = _spy_create_ticket(monkeypatch)
    assert client.post(_URL, json=_body(reason=reason)).status_code == 422
    assert calls == []


def test_rejects_an_empty_order_id(monkeypatch):
    calls = _spy_create_ticket(monkeypatch)
    assert client.post(_URL, json=_body(order_id="")).status_code == 422
    assert calls == []


@pytest.mark.parametrize("order_id", _BLANK_VARIANTS)
def test_rejects_a_blank_only_order_id(monkeypatch, order_id):
    """min_length=1 は空白のみを通す。全角スペースまで落ちることを固定する。

    通すと、どの注文の申請か分からない返金チケットが残る。
    """
    calls = _spy_create_ticket(monkeypatch)
    assert client.post(_URL, json=_body(order_id=order_id)).status_code == 422
    assert calls == []


@pytest.mark.parametrize("missing", ["conversation_id", "order_id", "reason"])
def test_rejects_missing_field(monkeypatch, missing):
    calls = _spy_create_ticket(monkeypatch)
    body = _body()
    del body[missing]
    assert client.post(_URL, json=body).status_code == 422
    assert calls == []


def test_returns_503_when_the_database_is_down(monkeypatch):
    """差し替えずに叩く。_no_real_db の到達不能な DB がそのまま障害になる。"""
    res = client.post(_URL, json=_body())
    assert res.status_code == 503
    assert res.json()["detail"] == _DB_DOWN_MSG


def test_a_missing_conversation_is_404_not_503(monkeypatch):
    """存在しない会話は「今は無理」ではなく「何度やっても無理」(05 章と同じ切り分け)。"""
    async def _boom(conversation_id, description, ticket_type):
        raise IntegrityError("INSERT INTO tickets", {}, Exception("FK 制約違反"))

    async def _missing(conversation_id):
        return None

    monkeypatch.setattr(actions.repository, "create_ticket", _boom)
    monkeypatch.setattr(actions.repository, "get_conversation", _missing)
    res = client.post(_URL, json=_body())
    assert res.status_code == 404
    assert res.json()["detail"] == "会話が見つかりません"


def test_a_real_db_outage_is_still_503(monkeypatch):
    """会話は在るのに書けない場合は、これまでどおり再試行を促す。"""
    async def _boom(conversation_id, description, ticket_type):
        raise OperationalError("INSERT INTO tickets", {}, Exception("接続断"))

    async def _exists(conversation_id):
        return object()

    monkeypatch.setattr(actions.repository, "create_ticket", _boom)
    monkeypatch.setattr(actions.repository, "get_conversation", _exists)
    res = client.post(_URL, json=_body())
    assert res.status_code == 503
    assert res.json()["detail"] == _DB_DOWN_MSG


def test_when_the_check_itself_fails_it_stays_503(monkeypatch):
    """会話の有無を確かめる問い合わせも落ちたら、存在しないとは言い切らない。"""
    async def _boom(*a, **k):
        raise OperationalError("INSERT INTO tickets", {}, Exception("接続断"))

    monkeypatch.setattr(actions.repository, "create_ticket", _boom)
    monkeypatch.setattr(actions.repository, "get_conversation", _boom)
    assert client.post(_URL, json=_body()).status_code == 503


def test_does_not_leak_the_database_error_to_the_client(monkeypatch):
    """接続文字列や SQL が detail に混ざらないこと。"""
    async def _boom(conversation_id, description, ticket_type):
        raise OperationalError("SELECT 1", {}, Exception("root:secret@10.0.0.1:3306"))

    monkeypatch.setattr(actions.repository, "create_ticket", _boom)
    detail = client.post(_URL, json=_body()).json()["detail"]
    assert detail == _DB_DOWN_MSG
    assert "secret" not in detail and "3306" not in detail


def test_endpoint_is_published_in_the_openapi_schema():
    """画面のフォームがこの契約を読む。選択肢が変わったら気づけるようにする。"""
    schema = _app.openapi()
    assert _URL in schema["paths"]
    op = schema["paths"][_URL]["post"]
    ref = op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    body = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    assert set(body["required"]) == {"conversation_id", "order_id", "reason"}
    assert body["properties"]["reason"]["enum"] == _REASONS
