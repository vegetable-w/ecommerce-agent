"""/api/chat(SSE)のテスト。

05 章でこのエンドポイントは graph の astream 入口になった(spec §7 / D2)。1 章の
SessionStore による履歴保持は無くなり、フロントエンド(static/index.html)の唯一の
入口でもある(旧 /api/agent/stream は削除した)。

確かめるのは 2 つ。

1. **runtime のイベントが SSE のフレームへ正しく写ること**。どのイベントを出すかの判断は
   app/graph/runtime.py の担当なので(そちらは tests/test_graph_runtime.py で確認する)、
   ここはワイヤ上の形だけを見る。特に **1 フレームが割れないこと**: 回答も引用もチケットの
   draft も日本語と改行を含むため、json.dumps のエスケープが唯一の防壁になる。
2. **失敗の伝え方**。ストリームは HTTP 200 とヘッダを送出した後に失敗しうるので、
   ステータスコードでは何も伝えられない。error フレームが届き、正常終了の印である
   [DONE] が届かないことを固定する。

**上流 model にも本番相当の DB にも一切触らない。** runtime.stream_turn を台本どおりの
偽物へ差し替える。差し替え忘れは _no_real_upstream がその場で落とす。
"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.graph import runtime
from app.main import app

_NOT_FOUND_MSG = "会話が見つかりません"
_DB_DOWN_MSG = "データベースを一時的に利用できません。しばらくしてからもう一度お試しください"
_UPSTREAM_MSG = "上流モデルを一時的に利用できません。しばらくしてからもう一度お試しください"

# 到達不能なポート。ここへの接続は即座に拒否され、SQLAlchemyError(OperationalError)になる。
_DEAD_DB_URL = "mysql+asyncmy://root:root@127.0.0.1:59999/nonexistent_db"

# 空白のみと見なすべき入力。ASCII 空白だけでは不十分で、U+3000(全角スペース)は
# 日本語 IME が出す「ありがちな空入力」そのもの。
_BLANK_VARIANTS = [
    pytest.param("   ", id="ascii-space"),
    pytest.param("　　", id="ideographic-space-u3000"),
    pytest.param("\t", id="tab"),
    pytest.param("\n", id="newline"),
    pytest.param(" ", id="nbsp"),
]


def _dead_session_factory():
    return async_sessionmaker(create_async_engine(_DEAD_DB_URL), expire_on_commit=False)


@pytest.fixture(autouse=True)
def _no_real_upstream(monkeypatch):
    """本物の DB・上流 model・本物の graph を既定で塞ぐ。

    async_session を到達不能なホストへ向けるので、差し替え忘れたテストは support
    (本番相当)へ書き込むのではなく SQLAlchemyError で落ちる。DB 障害のテストは
    この既定をそのまま使う。
    """
    monkeypatch.setattr("app.db.base.async_session", _dead_session_factory())

    def _boom_model(*args, **kwargs):
        raise AssertionError("テストが本物の上流 model を呼んだ")

    monkeypatch.setattr("app.core.llm.get_chat_model", _boom_model)
    monkeypatch.setattr("app.graph.nodes.get_chat_model", _boom_model)

    def _boom_graph():
        raise AssertionError("テストが偽の stream_turn を注入し忘れた")

    monkeypatch.setattr(runtime, "get_graph", _boom_graph)


def _use_events(monkeypatch, events: list[dict]) -> list[tuple]:
    """runtime.stream_turn を台本どおりのイベントを流す偽物へ差し替える。

    戻り値の list に (user_id, message, conversation_id) が記録される。
    """
    seen: list[tuple] = []

    async def fake_stream(user_id, message, conversation_id):
        seen.append((user_id, message, conversation_id))
        for ev in events:
            yield ev

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    return seen


def _body(payload: dict) -> str:
    """/api/chat を叩き、生のボディ文字列を返す。

    行単位ではなく生バイトを読むのは、フレームがワイヤ上で SSE の 1 フレームとして
    正しく組み上がっているかを確認するため。
    """
    client = TestClient(app)
    with client.stream("POST", "/api/chat", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        return b"".join(resp.iter_bytes()).decode("utf-8")


def _frames(body: str) -> list[str]:
    """フロントエンド(static/index.html)と同じ切り出し: 空行 2 つでフレームを割る。"""
    return [f for f in body.split("\n\n") if f]


def _payloads(body: str) -> list[dict]:
    """`data: ` 行のうち [DONE] 以外を JSON として取り出す。"""
    out = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload != "[DONE]":
            out.append(json.loads(payload))
    return out


def test_chat_maps_every_event_type_to_a_frame(monkeypatch):
    """tool / citations / delta / actions / done が順番どおり SSE に出て、[DONE] で終わる。"""
    citations = [{"n": 1, "id": 5, "section_path": "送料ポリシー", "question": "送料",
                  "answer": "3,000円以上で送料無料", "content_type": "faq"}]
    actions = [{"type": "transfer_human"}]
    events = [
        {"type": "tool", "name": "query_order"},
        {"type": "citations", "items": citations},
        {"type": "delta", "text": "注文 1001 は"},
        {"type": "delta", "text": "輸送中です。"},
        {"type": "actions", "items": actions},
        {"type": "done", "conversation_id": 42},
    ]
    _use_events(monkeypatch, events)
    body = _body({"user_id": "u1", "message": "注文1001は今どこですか"})

    payloads = _payloads(body)
    assert payloads[0] == {"event": "tool", "name": "query_order"}
    assert payloads[1] == {"event": "citations", "items": citations}
    assert [p["delta"] for p in payloads if "delta" in p] == ["注文 1001 は", "輸送中です。"]
    assert payloads[-2] == {"event": "actions", "items": actions}
    assert payloads[-1] == {"event": "done", "conversation_id": 42}
    assert body.endswith("data: [DONE]\n\n")


def test_chat_forwards_request_fields_to_runtime(monkeypatch):
    """リクエストの3フィールドが stream_turn へそのまま渡ることを固定する。

    conversation_id を落とすと毎ターン新しい会話が始まり、checkpointer の thread も
    切り替わる。画面上は動いているように見えたまま、多ターンの記憶だけが静かに壊れる。
    """
    seen = _use_events(monkeypatch, [{"type": "done", "conversation_id": 7}])
    _body({"user_id": "u9", "message": "m9", "conversation_id": 7})
    assert seen == [("u9", "m9", 7)]


def test_chat_omits_conversation_id_for_a_new_conversation(monkeypatch):
    """conversation_id 未指定は None のまま渡す(採番は runtime の担当)。"""
    seen = _use_events(monkeypatch, [{"type": "done", "conversation_id": 1}])
    _body({"user_id": "u1", "message": "こんにちは"})
    assert seen == [("u1", "こんにちは", None)]


def test_chat_streams_deltas_one_frame_each(monkeypatch):
    """token ごとに 1 フレーム。全文を 1 回で返す退行(画面が最後まで固まる)を検出する。"""
    _use_events(monkeypatch, [
        {"type": "delta", "text": "こ"},
        {"type": "delta", "text": "んに"},
        {"type": "delta", "text": "ちは。"},
        {"type": "done", "conversation_id": 3},
    ])
    payloads = _payloads(_body({"user_id": "u1", "message": "こんにちは"}))
    deltas = [p["delta"] for p in payloads if "delta" in p]
    assert deltas == ["こ", "んに", "ちは。"]
    assert "".join(deltas) == "こんにちは。"


def test_chat_actions_frame_survives_japanese_and_newlines(monkeypatch):
    """チケットの draft に日本語と改行が入ってもフレームが割れず、そのまま往復する。

    SSE のフレーム区切りは生の "\\n\\n" なので、本文をそのまま流すと 1 フレームが 2 つに
    割れ、フロントエンドの JSON.parse が両方失敗して選択肢が丸ごと消える。json.dumps が
    改行を \\n へエスケープすることが唯一の防壁であり、ここはその回帰テスト。
    draft は苦情の本文をそのまま抱えるので、改行は例外ではなく通常ケースになる。
    """
    draft = {"description": "届いた商品が破損していました。\n\n交換をお願いします。\r\n以上です。",
             "ticket_type": "return"}
    actions = [{"type": "create_ticket", "draft": draft}, {"type": "transfer_human"}]
    _use_events(monkeypatch, [
        {"type": "delta", "text": "ご不便をおかけしました。"},
        {"type": "actions", "items": actions},
        {"type": "done", "conversation_id": 9},
    ])
    body = _body({"user_id": "u1", "message": "壊れていた"})

    # 生バイトの中に draft の改行がそのまま出ていないこと
    assert "破損していました。\n\n交換" not in body
    act_frames = [f for f in _frames(body) if '"actions"' in f]
    assert len(act_frames) == 1
    assert act_frames[0].count("\n") == 0  # 1 フレーム = 1 行
    obj = json.loads(act_frames[0][len("data: ") :])
    assert obj["items"] == actions
    assert obj["items"][0]["draft"]["description"] == draft["description"]


def test_chat_citations_frame_survives_newlines_in_the_cited_text(monkeypatch):
    """引用本文に改行や空行が入ってもフレームが割れないこと(04 章から引き継いだ回帰テスト)。"""
    citation = {"n": 1, "id": 5, "section_path": "送料\nポリシー", "question": "送料",
                "answer": "1 行目\n\n2 行目\r\n3 行目", "content_type": "faq"}
    _use_events(monkeypatch, [
        {"type": "citations", "items": [citation]},
        {"type": "delta", "text": "ご案内します[1]。"},
        {"type": "done", "conversation_id": 4},
    ])
    body = _body({"user_id": "u1", "message": "送料はいくら?"})

    assert "1 行目\n\n2 行目" not in body
    cit_frames = [f for f in _frames(body) if '"citations"' in f]
    assert len(cit_frames) == 1
    assert cit_frames[0].count("\n") == 0
    obj = json.loads(cit_frames[0][len("data: ") :])
    assert obj["items"][0]["answer"] == "1 行目\n\n2 行目\r\n3 行目"


def test_chat_delta_frames_survive_newlines_and_spaces(monkeypatch):
    """回答本文は見出し・箇条書き・表を含む Markdown なので、改行を含む delta は通常ケース。
    前後の空白も含めて完全に往復する。"""
    tokens = ["行1\n行2", "\n\n", "  前後に空白  "]
    _use_events(monkeypatch,
                [{"type": "delta", "text": t} for t in tokens]
                + [{"type": "done", "conversation_id": 2}])
    body = _body({"user_id": "u1", "message": "返品の手順は?"})

    frames = _frames(body)
    # delta*3 + done + [DONE] = 5 フレーム、1 フレームちょうど 1 行
    assert len(frames) == 5
    for f in frames:
        assert "\n" not in f
        assert f.startswith("data: ")
    deltas = [p["delta"] for p in _payloads(body) if "delta" in p]
    assert deltas == tokens


def test_chat_ignores_unknown_event_types(monkeypatch):
    """runtime が将来新しい種類のイベントを足しても、知らないものは黙って捨てる。
    未知のイベントで turn ごと落ちるより、フロントエンドが解釈できる分だけ届く方がよい。"""
    _use_events(monkeypatch, [
        {"type": "trace", "payload": {"node": "classify_intent"}},
        {"type": "delta", "text": "はい。"},
        {"type": "done", "conversation_id": 5},
    ])
    body = _body({"user_id": "u1", "message": "hi"})
    assert "classify_intent" not in body
    assert [p for p in _payloads(body) if "delta" in p] == [{"delta": "はい。"}]


def test_chat_error_frame_on_unknown_conversation(monkeypatch):
    """ConversationNotFound はジェネレータの内側、つまり HTTP 200 とヘッダを送出した後に
    発生するため 404 にはできない。error フレームが届き、正常終了の印である [DONE] は
    届かないことを固定する。"""

    async def fake_stream(user_id, message, conversation_id):
        raise runtime.ConversationNotFound(999999)
        yield  # noqa: unreachable - 非同期ジェネレータにするためだけの行

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    body = _body({"user_id": "u1", "message": "hi", "conversation_id": 999999})

    # ワイヤ上で SSE の 1 フレームとして正しく組み上がっていること
    # (_sse_error は 2 回に分けて yield するが、連結が 1 フレームになる必要がある)
    expected = "event: error\ndata: {}\n\n".format(
        json.dumps({"message": _NOT_FOUND_MSG}, ensure_ascii=False)
    )
    assert expected in body
    assert [p["message"] for p in _payloads(body)] == [_NOT_FOUND_MSG]
    assert "[DONE]" not in body


def test_chat_error_frame_when_database_is_down():
    """DB 障害を「上流モデルが…」と誤って報告する退行(SQLAlchemyError 分岐の削除)を検出する。

    偽の stream_turn を入れない。_no_real_upstream が向けた到達不能な DB へ本物の
    stream_turn が最初の create_conversation で当たり、graph へ着く前に落ちる。
    """
    body = _body({"user_id": "u1", "message": "hi"})
    expected = "event: error\ndata: {}\n\n".format(
        json.dumps({"message": _DB_DOWN_MSG}, ensure_ascii=False)
    )
    assert expected in body
    assert "[DONE]" not in body


def test_chat_error_frame_on_upstream_failure(monkeypatch):
    async def fake_stream(user_id, message, conversation_id):
        raise RuntimeError("上流モデル呼び出し失敗(テスト用)")
        yield  # noqa: unreachable - 非同期ジェネレータにするためだけの行

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    body = _body({"user_id": "u1", "message": "hi"})

    expected = "event: error\ndata: {}\n\n".format(
        json.dumps({"message": _UPSTREAM_MSG}, ensure_ascii=False)
    )
    assert expected in body
    assert "[DONE]" not in body


def test_chat_mid_stream_failure_keeps_the_delivered_deltas(monkeypatch):
    """本文を流し始めた後で落ちても、既に届いた delta はそのままで、後ろに error フレームが
    続く。[DONE] は来ないので、フロントエンドは中断だったと判断できる。"""

    async def fake_stream(user_id, message, conversation_id):
        yield {"type": "delta", "text": "部"}
        yield {"type": "delta", "text": "分応答"}
        raise RuntimeError("上流モデル呼び出し失敗(部分応答後・テスト用)")

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    body = _body({"user_id": "u1", "message": "hi"})

    assert [p["delta"] for p in _payloads(body) if "delta" in p] == ["部", "分応答"]
    assert "event: error" in body
    assert "[DONE]" not in body


def test_chat_rejects_empty_message(monkeypatch):
    # stream_turn を「呼ばれたら失敗する」偽物に差し替えるのは、この検証がバリデーションだけを
    # 対象にしていることを明示するためだけではなく、安全装置でもある。差し替えないと、
    # スキーマ側のバリデーションが壊れた瞬間にこのテストが本物の graph へ流れ落ちる。
    _forbid_stream(monkeypatch)
    client = TestClient(app)
    resp = client.post("/api/chat", json={"user_id": "u1", "message": ""})
    assert resp.status_code == 422
    # ストリームは始まっておらず、FastAPI のバリデーションエラー JSON が返る
    assert resp.headers["content-type"].startswith("application/json")
    assert "event: error" not in resp.text


@pytest.mark.parametrize("blank", _BLANK_VARIANTS)
def test_chat_rejects_whitespace_only_input(blank, monkeypatch):
    """min_length=1 は空白のみの値を通してしまう。空白のみの user 行を一度 DB に作ると、
    その会話の履歴に残り続ける(1 章の SessionStore と違い MySQL に残るので高くつく)。"""
    _forbid_stream(monkeypatch)
    client = TestClient(app)
    assert client.post("/api/chat", json={"user_id": "u1", "message": blank}).status_code == 422
    assert client.post("/api/chat", json={"user_id": blank, "message": "hi"}).status_code == 422


def test_chat_rejects_missing_user_id(monkeypatch):
    _forbid_stream(monkeypatch)
    client = TestClient(app)
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 422


def _forbid_stream(monkeypatch) -> None:
    async def must_not_run(*a, **k):
        raise AssertionError("バリデーションで弾かれるべきリクエストが本体まで到達した")
        yield  # noqa: unreachable - 非同期ジェネレータにするためだけの行

    monkeypatch.setattr(runtime, "stream_turn", must_not_run)
