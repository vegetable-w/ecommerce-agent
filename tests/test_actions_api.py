"""POST /api/actions/create-ticket(チケット作成ボタン)のテスト。

このエンドポイントは **tickets テーブルへ実際に書き込む唯一の経路**(spec §6.2)。
Agent は create_ticket tool を呼んでも agent_tools に横取りされて選択肢へ変わるだけで、
DB には触らない。だから「ユーザーが押していないのにチケットが増えた」という事故は、
ここのバリデーションが緩んだときにしか起きない。テストもそこへ寄せる。

**本番相当の support DB へは絶対に書かない。** repository.create_ticket は各テストで
差し替える。差し替え忘れは _no_real_db が到達不能なホストで落とす(黙って本番へ
書かせない)。
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import actions
from app.core import labels

# router だけを載せた最小の app で叩く。app.main を import すると lifespan を持つ
# 本番の app が付いてきて、このモジュールの関心(エンドポイント単体の入出力)に
# 関係のない配線まで一緒に壊れる。本番 app へ登録されているかは
# tests/test_main_lifespan.py が見る。
_app = FastAPI()
_app.include_router(actions.router)
client = TestClient(_app)

_URL = "/api/actions/create-ticket"
_DB_DOWN_MSG = "チケットを一時的に作成できません。しばらくしてからもう一度お試しください"

# 到達不能なポート。ここへの接続は即座に拒否され、SQLAlchemyError になる。
_DEAD_DB_URL = "mysql+asyncmy://root:root@127.0.0.1:59999/nonexistent_db"

# 空白のみと見なすべき入力。ASCII 空白だけでは足りない。U+3000(全角スペース)は
# 日本語 IME がそのまま出す「ありがちな空入力」で、これを通すと空の description で
# チケットが立ち、運用側は何の苦情か分からない行を受け取る。
# tests/test_agent_api.py と同じ一覧(新しい書き方を発明しない)。
_BLANK_VARIANTS = [
    pytest.param("   ", id="ascii-space"),
    pytest.param("　　", id="ideographic-space-u3000"),
    pytest.param("\t", id="tab"),
    pytest.param("\n", id="newline"),
    pytest.param(" ", id="nbsp"),
]


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """本番相当の DB を既定で塞ぐ。

    repository が見る app.db.base.async_session を到達不能なホストへ向ける。
    create_ticket の差し替えを忘れたテストは support へ書き込むのではなく
    SQLAlchemyError で落ちる。DB 障害のテストはこの既定をそのまま使う。
    """
    monkeypatch.setattr(
        "app.db.base.async_session",
        async_sessionmaker(create_async_engine(_DEAD_DB_URL), expire_on_commit=False),
    )


def _spy_create_ticket(monkeypatch, ticket_no: str = "T20260908120000001") -> list[tuple]:
    """repository.create_ticket を記録用の偽物へ差し替える。呼ばれた引数を返す。"""
    calls: list[tuple] = []

    async def _fake(conversation_id, description, ticket_type):
        calls.append((conversation_id, description, ticket_type))
        return ticket_no

    # actions.repository 経由で差し替える。エンドポイントが
    # `from app.db.repository import create_ticket` と書いていたらここは効かない。
    monkeypatch.setattr(actions.repository, "create_ticket", _fake)
    return calls


def _body(**over) -> dict:
    body = {"conversation_id": 5, "description": "届いた商品が壊れていました",
            "ticket_type": "complaint"}
    body.update(over)
    return body


def test_creates_ticket_and_returns_ticket_no(monkeypatch):
    calls = _spy_create_ticket(monkeypatch)
    res = client.post(_URL, json=_body())
    assert res.status_code == 200
    assert res.json()["ticket_no"] == "T20260908120000001"
    # 引数の順番と値をそのまま固定する。description と ticket_type を取り違えると
    # ENUM 違反で落ちるので気づけるが、conversation_id の取り違えは黙って
    # 別の会話へチケットがぶら下がる。
    assert calls == [(5, "届いた商品が壊れていました", "complaint")]


def test_status_comes_from_the_label_table(monkeypatch):
    """表示名の出所を 1 つに保つ。

    02 章の create_ticket tool も labels 経由で同じ文言を出している。ここで文字列を
    べた書きすると、対応表を直したときに画面の 2 か所で別の日本語が出る。
    """
    _spy_create_ticket(monkeypatch)
    res = client.post(_URL, json=_body())
    assert res.json()["status"] == labels.label(labels.CONVERSATION_STATUS, "escalated")


@pytest.mark.parametrize("ticket_type", ["after_sales", "complaint", "inquiry"])
def test_accepts_every_ddl_enum_value(monkeypatch, ticket_type):
    """DDL の ENUM 3 値がすべて通ること。1 つでも落ちると画面の選択肢が死ぬ。"""
    calls = _spy_create_ticket(monkeypatch)
    assert client.post(_URL, json=_body(ticket_type=ticket_type)).status_code == 200
    assert calls[0][2] == ticket_type


@pytest.mark.parametrize("ticket_type", [
    pytest.param("苦情", id="japanese-label"),
    pytest.param("アフターサービス", id="japanese-label-after-sales"),
    pytest.param("相談", id="japanese-label-not-in-ddl"),
    pytest.param("after-sales", id="hyphen-instead-of-underscore"),
    pytest.param("不正値", id="garbage"),
    pytest.param("", id="empty"),
    pytest.param("COMPLAINT", id="uppercase"),
])
def test_rejects_ticket_type_outside_the_ddl_enum(monkeypatch, ticket_type):
    """DB の ENUM は英語の識別子。日本語やハイフン形を通すと write の瞬間に落ちる。

    日本語ラベルを明示的に並べているのは、spec §6.3 の画面表記(アフターサービス /
    苦情 / 相談)をそのまま送ってしまう実装ミスがいちばん起きやすいため。画面表記から
    識別子への変換は frontend の仕事で、逆方向の対応表は labels に無い(日本語を DB へ
    書く経路を作らないという 02 章の決定)。
    """
    # 差し替えるのは「422 で弾かれず repository まで届いた」ことを検出するため。
    calls = _spy_create_ticket(monkeypatch)
    assert client.post(_URL, json=_body(ticket_type=ticket_type)).status_code == 422
    assert calls == []


def test_rejects_empty_description(monkeypatch):
    calls = _spy_create_ticket(monkeypatch)
    assert client.post(_URL, json=_body(description="")).status_code == 422
    assert calls == []


@pytest.mark.parametrize("description", _BLANK_VARIANTS)
def test_rejects_blank_only_description(monkeypatch, description):
    """min_length=1 は空白のみを通す。全角スペースまで落ちることを固定する。"""
    calls = _spy_create_ticket(monkeypatch)
    assert client.post(_URL, json=_body(description=description)).status_code == 422
    assert calls == []


@pytest.mark.parametrize("missing", ["conversation_id", "description", "ticket_type"])
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


def test_returns_503_on_a_foreign_key_violation(monkeypatch):
    """存在しない conversation_id は tickets の FK 制約に当たる。

    IntegrityError は SQLAlchemyError の子なので、現状は DB 障害と同じ 503 になる。
    再試行しても必ず同じ結果になる入力に「しばらくしてからもう一度」と言う点は
    正確ではないが、画面は SSE の done フレームで受け取った会話 ID しか送らないため、
    実際にこの経路へ落ちるのは手書きのリクエストだけ。現状の挙動をここで固定し、
    変えるときにこのテストが気づかせる。
    """
    async def _boom(conversation_id, description, ticket_type):
        raise IntegrityError("INSERT INTO tickets", {}, Exception("FK 制約違反"))

    monkeypatch.setattr(actions.repository, "create_ticket", _boom)
    res = client.post(_URL, json=_body())
    assert res.status_code == 503
    assert res.json()["detail"] == _DB_DOWN_MSG


def test_does_not_leak_the_database_error_to_the_client(monkeypatch):
    """接続文字列や SQL が detail に混ざらないこと。"""
    async def _boom(conversation_id, description, ticket_type):
        raise OperationalError("SELECT 1", {}, Exception("root:secret@10.0.0.1:3306"))

    monkeypatch.setattr(actions.repository, "create_ticket", _boom)
    detail = client.post(_URL, json=_body()).json()["detail"]
    assert detail == _DB_DOWN_MSG
    assert "secret" not in detail and "3306" not in detail


def test_endpoint_is_published_in_the_openapi_schema():
    """画面側がこの契約を読む。パスと必須項目が変わったら気づけるようにする。"""
    schema = _app.openapi()
    assert _URL in schema["paths"]
    op = schema["paths"][_URL]["post"]
    ref = op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    body = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    assert set(body["required"]) == {"conversation_id", "description", "ticket_type"}
    assert body["properties"]["ticket_type"]["enum"] == ["after_sales", "complaint", "inquiry"]
