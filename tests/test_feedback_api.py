"""POST /api/feedback(満足度の 👍 / 👎)のテスト。

09 章のデータフライホイールの 3 つ目の入口(spec §4)。確かめる中心は 3 つ:

1. **👎 だけがプールへ入る。** 👍 を保存しないのは spec の明示的な非目標で、
   「貯めておけば後で使える」と気を利かせた瞬間に、査読する人は選り分けから
   始めることになる。
2. **写しは best effort だが、嘘は付けない。** checkpointer の最新 State が
   押された質問のものでなければ、検索結果は付けない(None)。別の質問の検索結果を
   添えると、査読画面は「この質問でこれを引いていた」という嘘の材料を読む。
3. **写しの失敗でプールへの投入を落とさない。** いちばん拾いたい「答えたが
   外していた」質問が、いちばん壊れているときに限って失われる形になる。

**本番相当の support DB にも checkpointer にも上流にも触らない。**
repository.insert_low_confidence と runtime.get_turn_snapshot は各テストで
差し替える。差し替え忘れは _no_real_db が到達不能なホストで落とす。
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import feedback

# router だけを載せた最小の app で叩く。app.main を import すると lifespan を持つ
# 本番の app が付いてきて、このモジュールの関心(エンドポイント単体の入出力)に
# 関係のない配線まで一緒に壊れる。本番 app へ登録されているかは
# tests/test_main_lifespan.py が見る(app/api/actions.py のテストと同じ作法)。
_app = FastAPI()
_app.include_router(feedback.router)
client = TestClient(_app)

_URL = "/api/feedback"
_Q = "返品の送料は誰が負担しますか"
# 到達不能なポート。ここへの接続は即座に拒否され、SQLAlchemyError になる。
_DEAD_DB_URL = "mysql+asyncmy://root:root@127.0.0.1:59999/nonexistent_db"

# 09 章の確信度ゲートが残す形と同じ(app/core/confidence.py の snapshot_from_hits)。
_SNAPSHOT = [
    {"question": "返品の送料", "answer": "自己都合はお客様負担です",
     "rerank_score": 0.42, "section_path": "返品/送料"},
]

# 空白のみと見なすべき入力。ASCII 空白だけでは足りない。U+3000(全角スペース)は
# 日本語 IME がそのまま出す「ありがちな空入力」。
# tests/test_actions_api.py と同じ一覧(新しい書き方を発明しない)。
_BLANK_VARIANTS = [
    pytest.param("   ", id="ascii-space"),
    pytest.param("　　", id="ideographic-space-u3000"),
    pytest.param("\t", id="tab"),
    pytest.param("\n", id="newline"),
    pytest.param(" ", id="nbsp"),
]


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """本番相当の DB を既定で塞ぐ。

    repository が見る app.db.base.async_session を到達不能なホストへ向ける。
    insert_low_confidence の差し替えを忘れたテストは support へ書き込むのではなく
    SQLAlchemyError で落ちる。
    """
    monkeypatch.setattr(
        "app.db.base.async_session",
        async_sessionmaker(create_async_engine(_DEAD_DB_URL), expire_on_commit=False),
    )


@pytest.fixture(autouse=True)
def _no_real_checkpointer(monkeypatch):
    """本物の checkpointer を既定で塞ぐ。

    runtime.get_turn_snapshot は未初期化なら RuntimeError を出すが、テストの
    実行順によっては別のテストが開いた graph が module 変数に残りうる。差し替えを
    忘れたテストが本番の data/05_checkpoints.sqlite を読みにいかないよう、
    既定を明示的に落ちる偽物にしておく。
    """

    async def _boom(conversation_id):
        raise AssertionError("テストが本物の checkpointer を読もうとした")

    monkeypatch.setattr(feedback.runtime, "get_turn_snapshot", _boom)


def _spy_insert(monkeypatch) -> list[dict]:
    """repository.insert_low_confidence を記録用の偽物へ差し替える。"""
    calls: list[dict] = []

    async def _fake(conversation_id, raw_question, source, reason,
                    retrieved_chunks=None):
        calls.append({"conversation_id": conversation_id, "raw_question": raw_question,
                      "source": source, "reason": reason,
                      "retrieved_chunks": retrieved_chunks})
        return 1

    # feedback.repository 経由で差し替える。エンドポイントが
    # `from app.db.repository import insert_low_confidence` と書いていたらここは効かない。
    monkeypatch.setattr(feedback.repository, "insert_low_confidence", _fake)
    return calls


def _stub_snapshot(monkeypatch, question: str, snapshot: list) -> None:
    async def _fake(conversation_id):
        return {"question": question, "snapshot": snapshot}

    monkeypatch.setattr(feedback.runtime, "get_turn_snapshot", _fake)


def _body(**over) -> dict:
    body = {"conversation_id": 5, "rating": "down", "question": _Q}
    body.update(over)
    return body


def test_down_pools_with_snapshot_when_question_matches(monkeypatch):
    """👎 は user_feedback としてプールへ入り、そのターンの写しを引き継ぐ。"""
    calls = _spy_insert(monkeypatch)
    _stub_snapshot(monkeypatch, _Q, _SNAPSHOT)

    res = client.post(_URL, json=_body())
    assert res.status_code == 200
    assert res.json() == {"ok": True, "pooled": True}
    assert len(calls) == 1
    assert calls[0]["conversation_id"] == 5
    assert calls[0]["raw_question"] == _Q
    # source を取り違えると、査読画面で「agent が自分で断った質問」と
    # 「人が否と言った質問」が同じ山に混ざる。直し方が違うので分けている
    assert calls[0]["source"] == "user_feedback"
    assert calls[0]["retrieved_chunks"] == _SNAPSHOT


def test_down_pools_without_snapshot_when_question_differs(monkeypatch):
    """質問が一致しなければ写しは付けない。**None であって [] ではない。**

    押すのが遅れて次のターンが始まっていると、最新 State の写しは別の質問のもの。
    添えると査読画面が嘘の材料を読むので、付けないより悪い。

    [] を渡すと JSON の [] として保存され、DDL が「検索を通っていない」の意味で
    使う NULL から漏れる(app/db/models.py の none_as_null と同じ話)。
    """
    calls = _spy_insert(monkeypatch)
    _stub_snapshot(monkeypatch, "まったく別のターンの質問", _SNAPSHOT)

    res = client.post(_URL, json=_body())
    assert res.status_code == 200
    assert res.json() == {"ok": True, "pooled": True}
    assert calls[0]["raw_question"] == _Q          # 押された質問の方が残る
    assert calls[0]["retrieved_chunks"] is None


def test_down_passes_none_not_empty_list_when_the_turn_had_no_search(monkeypatch):
    """検索を通らないターン(業務経路や定型応答)の 👎 も None で積む。"""
    calls = _spy_insert(monkeypatch)
    _stub_snapshot(monkeypatch, _Q, [])

    assert client.post(_URL, json=_body()).status_code == 200
    assert calls[0]["retrieved_chunks"] is None


def test_down_survives_snapshot_failure(monkeypatch):
    """写しの取得が落ちてもプールへの投入は続く(写しは本体ではない)。"""
    calls = _spy_insert(monkeypatch)

    async def _boom(conversation_id):
        raise RuntimeError("checkpointer を読めない")

    monkeypatch.setattr(feedback.runtime, "get_turn_snapshot", _boom)

    res = client.post(_URL, json=_body())
    assert res.status_code == 200
    assert res.json() == {"ok": True, "pooled": True}
    assert len(calls) == 1
    assert calls[0]["retrieved_chunks"] is None


def test_up_logs_only(monkeypatch):
    """👍 は log だけ。**DB へ保存しない**(spec の明示的な非目標)。"""
    calls = _spy_insert(monkeypatch)

    res = client.post(_URL, json=_body(rating="up"))
    assert res.status_code == 200
    assert res.json() == {"ok": True, "pooled": False}
    assert calls == []


def test_up_touches_neither_the_database_nor_the_checkpointer():
    """差し替えずに叩く。👍 が DB か checkpointer に触れば、既定の偽物が落とす。

    _spy_insert を使うテストは「repository をどう呼んだか」しか見ておらず、
    別経路で書き込みが起きても気づけない。ここは何も差し替えずに 200 を要求する。
    """
    res = client.post(_URL, json=_body(rating="up"))
    assert res.status_code == 200
    assert res.json() == {"ok": True, "pooled": False}


@pytest.mark.parametrize("rating", [
    pytest.param("UP", id="uppercase"),
    pytest.param("good", id="synonym"),
    pytest.param("👍", id="emoji"),
    pytest.param("いいね", id="japanese"),
    pytest.param("", id="empty"),
    pytest.param("neutral", id="third-value"),
])
def test_rejects_bad_rating(monkeypatch, rating):
    """許容値は up / down の 2 つだけ。3 つ目を黙って通すと source の意味が濁る。"""
    calls = _spy_insert(monkeypatch)
    assert client.post(_URL, json=_body(rating=rating)).status_code == 422
    assert calls == []


@pytest.mark.parametrize("question", _BLANK_VARIANTS)
def test_rejects_blank_only_question(monkeypatch, question):
    """min_length=1 は空白のみを通す。全角スペースまで落ちることを固定する。

    何を訊かれたのか分からない行がプールに残ると、人が見ても直しようがない。
    """
    calls = _spy_insert(monkeypatch)
    assert client.post(_URL, json=_body(question=question)).status_code == 422
    assert calls == []


@pytest.mark.parametrize("missing", ["conversation_id", "rating", "question"])
def test_rejects_missing_field(monkeypatch, missing):
    calls = _spy_insert(monkeypatch)
    body = _body()
    del body[missing]
    assert client.post(_URL, json=body).status_code == 422
    assert calls == []


def test_returns_503_when_the_pool_cannot_be_written(monkeypatch):
    """書けなかったことを 500 の traceback ではなく、中立な文面で返す。"""
    async def _boom(*a, **k):
        raise OperationalError("INSERT INTO low_confidence_questions", {},
                               Exception("root:secret@10.0.0.1:3306"))

    monkeypatch.setattr(feedback.repository, "insert_low_confidence", _boom)
    _stub_snapshot(monkeypatch, _Q, [])

    res = client.post(_URL, json=_body())
    assert res.status_code == 503
    detail = res.json()["detail"]
    # 接続文字列や SQL が画面まで届かないこと
    assert "secret" not in detail and "3306" not in detail


def test_endpoint_is_published_in_the_openapi_schema():
    """画面側がこの契約を読む。パスと許容値が変わったら気づけるようにする。"""
    schema = _app.openapi()
    assert _URL in schema["paths"]
    op = schema["paths"][_URL]["post"]
    ref = op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    body = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    assert set(body["required"]) == {"conversation_id", "rating", "question"}
    assert body["properties"]["rating"]["enum"] == ["up", "down"]
