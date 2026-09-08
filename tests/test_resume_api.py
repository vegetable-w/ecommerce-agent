"""POST /api/actions/resume(中断した turn の再開)のテスト。

06 章で fetch_order は、注文が特定できないと interrupt で止まって画面に一覧を出す
(spec §6)。ユーザーがそこで 1 件選ぶと、画面はこのエンドポイントを叩いて **中断した
turn の続き**を走らせる。08 章でもう 1 種類の中断が増えた(チケットの確認カード)。
入口は同じで、body の field だけが違う: 注文の選択は order_id、確認は confirmed。

/api/chat と同じ SSE で返すのが要点。再開の後には Agent の回答がそのまま続くので、
画面は選択の前後で描画を分けずに済む。したがって確かめるのは 2 つ。

1. **/api/chat と同じフレームが出ること**。フレーム変換を 2 か所に書くと、片方だけを
   直した瞬間に「回答は出るが出典は出ない」のような半端な壊れ方をする。
2. **失敗の伝え方が /api/chat と揃っていること**。ストリームは HTTP 200 とヘッダを
   送出した後に失敗しうるので、ステータスコードでは何も伝えられない。

**上流 model にも本番相当の DB にも一切触らない。** runtime.stream_resume を台本どおりの
偽物へ差し替える。差し替え忘れは _no_real_upstream がその場で落とす。
"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.graph import runtime
from app.main import app

_URL = "/api/actions/resume"
_NOT_FOUND_MSG = "会話が見つかりません"
_DB_DOWN_MSG = "データベースを一時的に利用できません。しばらくしてからもう一度お試しください"
_UPSTREAM_MSG = "上流モデルを一時的に利用できません。しばらくしてからもう一度お試しください"

# 到達不能なポート。ここへの接続は即座に拒否され、SQLAlchemyError(OperationalError)になる。
_DEAD_DB_URL = "mysql+asyncmy://root:root@127.0.0.1:59999/nonexistent_db"

_ORDERS = [
    {"order_id": "1001", "product": "自動猫トイレ", "status": "支払い済み", "amount": 1739},
]


@pytest.fixture(autouse=True)
def _no_real_upstream(monkeypatch):
    """本物の DB・上流 model・本物の graph を既定で塞ぐ(tests/test_chat_api.py と同じ作法)。"""
    monkeypatch.setattr(
        "app.db.base.async_session",
        async_sessionmaker(create_async_engine(_DEAD_DB_URL), expire_on_commit=False),
    )

    def _boom_model(*args, **kwargs):
        raise AssertionError("テストが本物の上流 model を呼んだ")

    monkeypatch.setattr("app.core.llm.get_chat_model", _boom_model)
    monkeypatch.setattr("app.graph.nodes.get_chat_model", _boom_model)

    def _boom_graph():
        raise AssertionError("テストが偽の stream_resume を注入し忘れた")

    monkeypatch.setattr(runtime, "get_graph", _boom_graph)


def _use_events(monkeypatch, events: list[dict]) -> list[tuple]:
    """runtime.stream_resume を台本どおりのイベントを流す偽物へ差し替える。

    戻り値の list に (conversation_id, resume_value) が記録される。
    """
    seen: list[tuple] = []

    async def fake_stream(conversation_id, resume_value):
        seen.append((conversation_id, resume_value))
        for ev in events:
            yield ev

    monkeypatch.setattr(runtime, "stream_resume", fake_stream)
    return seen


def _body(payload: dict) -> str:
    client = TestClient(app)
    with client.stream("POST", _URL, json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        return b"".join(resp.iter_bytes()).decode("utf-8")


def _frames(body: str) -> list[str]:
    return [f for f in body.split("\n\n") if f]


def _payloads(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload != "[DONE]":
            out.append(json.loads(payload))
    return out


def test_resume_streams_the_same_frames_as_chat(monkeypatch):
    """citations / delta / actions / done が /api/chat と同じ形で出て、[DONE] で終わる。

    ここが /api/chat とずれると、選択の後だけ出典やボタンが描かれない画面になる。
    """
    citations = [{"n": 1, "id": 9, "section_path": "返品ポリシー", "question": "返品条件",
                  "answer": "7日以内", "content_type": "policy"}]
    actions = [{"type": "submit_refund", "draft": {"order_id": "1001"}}]
    _use_events(monkeypatch, [
        {"type": "tool", "name": "query_order"},
        {"type": "citations", "items": citations},
        {"type": "delta", "text": "注文 1001 は"},
        {"type": "delta", "text": "返品できます[1]。"},
        {"type": "actions", "items": actions},
        {"type": "done", "conversation_id": 42},
    ])
    body = _body({"conversation_id": 42, "order_id": "1001"})

    assert _payloads(body) == [
        {"event": "tool", "name": "query_order"},
        {"event": "citations", "items": citations},
        {"delta": "注文 1001 は"},
        {"delta": "返品できます[1]。"},
        {"event": "actions", "items": actions},
        {"event": "done", "conversation_id": 42},
    ]
    assert body.endswith("data: [DONE]\n\n")


def test_resume_forwards_the_selected_order_unchanged(monkeypatch):
    """06: 選ばれた注文番号はそのまま graph へ渡る。

    値を解釈するのは fetch_order の _normalize_order_id の仕事で、ここではない。
    """
    seen = _use_events(monkeypatch, [{"type": "done", "conversation_id": 42}])
    _body({"conversation_id": 42, "order_id": "1001"})
    assert seen == [(42, "1001")]


@pytest.mark.parametrize("confirmed", [True, False])
def test_resume_wraps_the_ticket_decision_in_a_dict(monkeypatch, confirmed):
    """08: チケットの確認は {"confirmed": bool} に包んで渡す。

    包むのは、確認を要する操作が今後増えても agent_tools 側が 1 つの形だけを
    読めばよいようにするため。**False も必ず渡す**(取り消しは「何も答えなかった」
    ではなく「作らないと答えた」で、engine の監査に権限拒否として残る)。
    """
    seen = _use_events(monkeypatch, [{"type": "done", "conversation_id": 42}])
    _body({"conversation_id": 42, "confirmed": confirmed})
    assert seen == [(42, {"confirmed": confirmed})]


def test_resume_prefers_the_order_when_both_are_sent(monkeypatch):
    """両方来たら注文の選択として扱う。中断は 1 会話に 1 つしか待っていないので、
    どちらか一方に決め打つ必要がある。"""
    seen = _use_events(monkeypatch, [{"type": "done", "conversation_id": 42}])
    _body({"conversation_id": 42, "order_id": "1001", "confirmed": False})
    assert seen == [(42, "1001")]


def test_resume_rejects_a_request_that_answers_nothing(monkeypatch):
    """order_id も confirmed も無い要求は 400。

    素通しすると None が resume の値として graph へ届き、fetch_order は「読めない選択」
    として、agent_tools は「未確認」として中断を黙って解いてしまう。ユーザーは
    何も押していないのに、選択カードが消えた画面を見ることになる。
    """

    async def must_not_run(*a, **k):
        raise AssertionError("何も答えていない要求が本体まで到達した")
        yield  # noqa: unreachable - 非同期ジェネレータにするためだけの行

    monkeypatch.setattr(runtime, "stream_resume", must_not_run)
    resp = TestClient(app).post(_URL, json={"conversation_id": 42})
    assert resp.status_code == 400
    assert "order_id" in resp.json()["detail"]
    assert "confirmed" in resp.json()["detail"]


def test_resume_can_interrupt_again_with_a_ticket_preview(monkeypatch):
    """08 の確認カードで止まった場合も、/api/chat と同じ interrupt フレーム。"""
    preview = {"ticket_type": "after_sales", "ticket_type_label": "アフターサービス",
               "description": "充電器が発熱します"}
    _use_events(monkeypatch, [
        {"type": "interrupt", "kind": "confirm_ticket", "orders": [], "preview": preview,
         "conversation_id": 42},
    ])
    body = _body({"conversation_id": 42, "order_id": "1001"})

    assert _payloads(body) == [
        {"event": "interrupt", "kind": "confirm_ticket", "orders": [], "preview": preview,
         "conversation_id": 42},
    ]


def test_resume_can_interrupt_again(monkeypatch):
    """選んだ値が読めないなどで再び止まった場合も、/api/chat と同じ interrupt フレーム。"""
    _use_events(monkeypatch, [
        {"type": "interrupt", "kind": "select_order", "orders": _ORDERS,
         "conversation_id": 42},
    ])
    body = _body({"conversation_id": 42, "order_id": "???"})

    assert _payloads(body) == [
        {"event": "interrupt", "kind": "select_order", "orders": _ORDERS,
         "conversation_id": 42},
    ]
    assert not any(p.get("event") == "done" for p in _payloads(body))
    assert body.endswith("data: [DONE]\n\n")


def test_resume_frames_survive_japanese_and_newlines(monkeypatch):
    """SSE のフレーム区切りは生の "\\n\\n"。本文をそのまま流すと 1 フレームが割れ、
    画面の JSON.parse が両方失敗して回答も選択肢も消える(/api/chat と同じ回帰テスト)。"""
    actions = [{"type": "submit_refund",
                "draft": {"order_id": "1001", "reason": "品質不良\n\n動作しません"}}]
    _use_events(monkeypatch, [
        {"type": "delta", "text": "承知しました。\n\n返品を受け付けます。"},
        {"type": "actions", "items": actions},
        {"type": "done", "conversation_id": 42},
    ])
    body = _body({"conversation_id": 42, "order_id": "1001"})

    assert "品質不良\n\n動作しません" not in body
    for f in _frames(body):
        assert f.count("\n") == 0
    payloads = _payloads(body)
    assert payloads[0] == {"delta": "承知しました。\n\n返品を受け付けます。"}
    assert payloads[1]["items"] == actions


def test_resume_ignores_unknown_event_types(monkeypatch):
    """runtime が将来新しい種類のイベントを足しても、知らないものは黙って捨てる。"""
    _use_events(monkeypatch, [
        {"type": "trace", "payload": {"node": "fetch_order"}},
        {"type": "delta", "text": "はい。"},
        {"type": "done", "conversation_id": 42},
    ])
    body = _body({"conversation_id": 42, "order_id": "1001"})
    assert "fetch_order" not in body
    assert [p for p in _payloads(body) if "delta" in p] == [{"delta": "はい。"}]


def test_resume_error_frame_on_unknown_conversation(monkeypatch):
    """ConversationNotFound は HTTP 200 とヘッダを送出した後に発生するので 404 にできない。
    /api/chat と同じ error フレームにする。"""

    async def fake_stream(conversation_id, resume_value):
        raise runtime.ConversationNotFound(999999)
        yield  # noqa: unreachable - 非同期ジェネレータにするためだけの行

    monkeypatch.setattr(runtime, "stream_resume", fake_stream)
    body = _body({"conversation_id": 999999, "order_id": "1001"})

    expected = "event: error\ndata: {}\n\n".format(
        json.dumps({"message": _NOT_FOUND_MSG}, ensure_ascii=False)
    )
    assert expected in body
    assert "[DONE]" not in body


def test_resume_error_frame_when_database_is_down():
    """偽の stream_resume を入れない。_no_real_upstream が向けた到達不能な DB へ
    本物の stream_resume が会話の存在確認で当たり、graph へ着く前に落ちる。"""
    body = _body({"conversation_id": 42, "order_id": "1001"})
    expected = "event: error\ndata: {}\n\n".format(
        json.dumps({"message": _DB_DOWN_MSG}, ensure_ascii=False)
    )
    assert expected in body
    assert "[DONE]" not in body


def test_resume_error_frame_on_upstream_failure(monkeypatch):
    async def fake_stream(conversation_id, resume_value):
        raise RuntimeError("上流モデル呼び出し失敗(テスト用)")
        yield  # noqa: unreachable - 非同期ジェネレータにするためだけの行

    monkeypatch.setattr(runtime, "stream_resume", fake_stream)
    body = _body({"conversation_id": 42, "order_id": "1001"})

    expected = "event: error\ndata: {}\n\n".format(
        json.dumps({"message": _UPSTREAM_MSG}, ensure_ascii=False)
    )
    assert expected in body
    assert "[DONE]" not in body


def test_resume_mid_stream_failure_keeps_the_delivered_deltas(monkeypatch):
    """本文を流し始めた後で落ちても、届いた分はそのままで後ろに error フレームが続く。"""

    async def fake_stream(conversation_id, resume_value):
        yield {"type": "delta", "text": "注文 1001 は"}
        raise RuntimeError("上流モデル呼び出し失敗(部分応答後・テスト用)")

    monkeypatch.setattr(runtime, "stream_resume", fake_stream)
    body = _body({"conversation_id": 42, "order_id": "1001"})

    assert [p["delta"] for p in _payloads(body) if "delta" in p] == ["注文 1001 は"]
    assert "event: error" in body
    assert "[DONE]" not in body


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"order_id": "1001"}, id="missing-conversation-id"),
        pytest.param({"conversation_id": "abc", "order_id": "1001"}, id="non-numeric-id"),
        pytest.param({"conversation_id": 42, "order_id": ""}, id="empty-order-id"),
        pytest.param({"conversation_id": 42, "confirmed": "たぶん"}, id="non-boolean-confirmed"),
    ],
)
def test_resume_rejects_malformed_requests(monkeypatch, payload):
    """バリデーションで弾かれたリクエストが本体まで到達しないこと。

    偽物を「呼ばれたら失敗する」ものにしておくのは安全装置でもある。スキーマ側の
    検証が壊れた瞬間に、本物の graph と本番相当の DB へ流れ落ちるのを防ぐ。
    """

    async def must_not_run(*a, **k):
        raise AssertionError("バリデーションで弾かれるべきリクエストが本体まで到達した")
        yield  # noqa: unreachable - 非同期ジェネレータにするためだけの行

    monkeypatch.setattr(runtime, "stream_resume", must_not_run)
    resp = TestClient(app).post(_URL, json=payload)
    assert resp.status_code == 422
    # ストリームは始まっておらず、FastAPI のバリデーションエラー JSON が返る
    assert resp.headers["content-type"].startswith("application/json")
    assert "event: error" not in resp.text


def test_resume_is_registered_on_the_real_app():
    """画面が叩くのは本番の app。router の登録漏れをここで見る。"""
    assert _URL in app.openapi()["paths"]
