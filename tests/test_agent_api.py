"""/api/agent/stream (SSE) と /api/agent (非ストリーミング JSON) のテスト。

このモジュールは非同期(DB を触る 12A)と同期(monkeypatch で run_agent_turn を差し替える
12B)のテストが同居する。そのためモジュール先頭に
`pytestmark = pytest.mark.asyncio(loop_scope="session")` は置かない: 同期テストにまで
asyncio マーカーが付き "is marked with '@pytest.mark.asyncio' but it is not an async
function" の PytestWarning が出るため(実測で確認済み)。非同期テストにだけ
`@pytest.mark.asyncio(loop_scope="session")` をデコレータで個別に付ける
(_test_engine がセッションスコープのループ上にあるため、loop_scope の指定自体は必須。
tests/conftest.py と pyproject.toml のコメント参照)。
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import agent as agent_api
from app.core import agent
from app.core.agent import AgentResult
from app.db import repository as repo
from app.main import app
from app.schemas.agent import ToolResultView
from app.tools.infra import ToolRun
from tests.test_agent_orchestration import FakeModel

session_loop = pytest.mark.asyncio(loop_scope="session")

_NOT_FOUND_MSG = "会話が見つかりません"
_DB_DOWN_MSG = "データベースを一時的に利用できません。しばらくしてからもう一度お試しください"
_UPSTREAM_MSG = "上流モデルを一時的に利用できません。しばらくしてからもう一度お試しください"

# 到達不能なポート。ここへの接続は即座に拒否され、SQLAlchemyError(OperationalError)になる。
_DEAD_DB_URL = "mysql+asyncmy://root:root@127.0.0.1:59999/nonexistent_db"

# 空白のみと見なすべき入力。ASCII 空白だけでは不十分で、U+3000(全角スペース)は
# 日本語 IME が出す「ありがちな空入力」そのもの。bare な .strip() はこれらを全て
# 落とすが、.strip(" ") のような「明示化」に書き換えると ASCII 空白しか落ちなくなる。
_BLANK_VARIANTS = [
    pytest.param("   ", id="ascii-space"),
    pytest.param("　　", id="ideographic-space-u3000"),
    pytest.param("\t", id="tab"),
    pytest.param("\n", id="newline"),
    pytest.param(" ", id="nbsp"),
]


class _MustNotBeUsedModel:
    """組み立てはできるが、使われたら失敗するモデル。

    テスト対象ではなく安全網。run_agent_turn は `model or get_chat_model()` を
    _prepare_turn より前に実行するので、構築時点で例外を投げるフェイクにすると
    DB 障害より先に落ちてしまい 503 を検証できない。使用時にだけ落とす。
    """

    def bind_tools(self, tools):
        raise AssertionError("上流モデルが呼ばれた(ガードが壊れている)")

    async def ainvoke(self, messages):
        raise AssertionError("上流モデルが呼ばれた(ガードが壊れている)")

    async def astream(self, messages):
        raise AssertionError("上流モデルが呼ばれた(ガードが壊れている)")
        yield  # noqa: unreachable - astream を非同期ジェネレータにするためだけの行


def _dead_session_factory():
    """本番相当の DB が落ちている状況を作る session factory。

    repository は app.db.base.async_session を参照するので、そこをこれに差し替えると
    _prepare_turn の最初の create_conversation が実際に SQLAlchemyError を送出する。
    DB より手前で失敗するため、上流 LLM は呼ばれない。
    """
    return async_sessionmaker(create_async_engine(_DEAD_DB_URL), expire_on_commit=False)


def teardown_function():
    app.dependency_overrides.clear()


def _use_model(model) -> None:
    app.dependency_overrides[agent_api.get_model] = lambda: model


async def _post_stream(payload: dict) -> tuple[int, str, str]:
    """/api/agent/stream を叩き、(status_code, content_type, 生のボディ文字列) を返す。

    行単位ではなく生バイトを読むのは、`event: error` フレームがワイヤ上で
    SSE の1フレームとして正しく組み上がっているかを確認するため。
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/api/agent/stream", json=payload) as resp:
            body = (await resp.aread()).decode("utf-8")
            return resp.status_code, resp.headers.get("content-type", ""), body


def _data_payloads(body: str) -> list[dict]:
    """`data: ` 行のうち [DONE] 以外を JSON として取り出す。"""
    out = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload != "[DONE]":
            out.append(json.loads(payload))
    return out


# --- 12A: /api/agent/stream -------------------------------------------------


@session_loop
async def test_stream_endpoint_emits_tool_deltas_done_and_terminator(
    db_session_factory, db_clean
):
    first = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
    )
    _use_model(FakeModel([first], stream_tokens=["注文 1001 は", "現在", "輸送中です。"]))

    status, content_type, body = await _post_stream(
        {"user_id": "u1", "message": "注文 1001 は今どこですか"}
    )
    assert status == 200
    assert content_type.startswith("text/event-stream")

    payloads = _data_payloads(body)
    assert payloads[0] == {"event": "tool", "name": "query_logistics"}

    deltas = [p["delta"] for p in payloads if "delta" in p]
    assert "".join(deltas) == "注文 1001 は現在輸送中です。"
    assert len(deltas) > 1  # 一括ではなく token ごとに届く

    done = payloads[-1]
    assert done["event"] == "done"
    assert isinstance(done["conversation_id"], int)

    assert body.endswith("data: [DONE]\n\n")

    msgs = await repo.list_messages(done["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]


@session_loop
async def test_stream_endpoint_without_tools_has_no_tool_frame(db_session_factory, db_clean):
    _use_model(FakeModel([AIMessage(content="こんにちは。どのようなご用件でしょうか?")]))

    status, _, body = await _post_stream({"user_id": "u1", "message": "こんにちは"})
    assert status == 200

    payloads = _data_payloads(body)
    assert not any(p.get("event") == "tool" for p in payloads)
    deltas = [p["delta"] for p in payloads if "delta" in p]
    assert deltas == ["こんにちは。どのようなご用件でしょうか?"]

    done = payloads[-1]
    assert done["event"] == "done"
    assert body.endswith("data: [DONE]\n\n")

    msgs = await repo.list_messages(done["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant"]


@session_loop
async def test_stream_endpoint_unknown_conversation_emits_error_frame_without_done(
    db_session_factory, db_clean
):
    """ConversationNotFound はジェネレータの内側、つまり HTTP 200 とヘッダを送出した後に
    発生するため 404 にはできない。クライアントには `event: error` フレームが届き、
    正常終了の印である [DONE] は届かないことを固定する。"""
    _use_model(FakeModel([AIMessage(content="hi")]))

    status, content_type, body = await _post_stream(
        {"user_id": "u1", "message": "hi", "conversation_id": 999999}
    )
    assert status == 200  # ストリーム開始後の失敗なのでステータスは変えられない
    assert content_type.startswith("text/event-stream")

    # ワイヤ上で SSE の1フレームとして正しく組み上がっていること
    # (_sse_error は2回に分けて yield するが、連結が1フレームになる必要がある)
    expected_frame = "event: error\ndata: {}\n\n".format(
        json.dumps({"message": _NOT_FOUND_MSG}, ensure_ascii=False)
    )
    assert expected_frame in body

    assert [p["message"] for p in _data_payloads(body)] == [_NOT_FOUND_MSG]
    assert "[DONE]" not in body


@session_loop
async def test_stream_endpoint_rejects_empty_message_before_stream_starts(
    db_session_factory, db_clean
):
    _use_model(FakeModel([AIMessage(content="hi")]))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/agent/stream", json={"user_id": "u1", "message": ""})

    assert resp.status_code == 422
    # ストリームは始まっておらず、FastAPI のバリデーションエラー JSON が返る
    assert resp.headers["content-type"].startswith("application/json")
    assert "detail" in resp.json()
    assert "event: error" not in resp.text


@session_loop
@pytest.mark.parametrize("blank", _BLANK_VARIANTS)
async def test_stream_endpoint_rejects_whitespace_only_input(
    blank, db_session_factory, db_clean
):
    """min_length=1 は空白のみの値を通してしまう。空白のみの user 行を一度 DB に作ると、
    _build_history が user 行を無条件に拾うため以後のターンで永久に再生される
    (chapter 1 の SessionStore と違い MySQL に残るので、より高くつく)。"""
    _use_model(FakeModel([AIMessage(content="hi")]))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        blank_msg = await client.post(
            "/api/agent/stream", json={"user_id": "u1", "message": blank}
        )
        blank_user = await client.post(
            "/api/agent/stream", json={"user_id": blank, "message": "hi"}
        )

    assert blank_msg.status_code == 422
    assert blank_user.status_code == 422


@session_loop
async def test_stream_frames_survive_newlines_and_spaces_in_deltas(
    db_session_factory, db_clean
):
    """SSE のフレーム境界を生バイトで固定する。

    他のストリームテストは _data_watch 相当の splitlines() 経由で読むため、フレーム境界の
    崩れ(_sse が \\n\\n ではなく \\n を出す / JSON のエスケープが崩れて delta 内の改行が
    そのまま本文に出る)を原理的に検出できない。Task 13 の回答は見出し・箇条書き・表を含む
    Markdown なので、改行を含む delta は例外ではなく通常ケースになる。
    """
    tokens = ["行1\n行2", "\n\n", "  前後に空白  "]
    first = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
    )
    _use_model(FakeModel([first], stream_tokens=tokens))

    _, _, body = await _post_stream({"user_id": "u1", "message": "注文1001はどこですか"})

    # 本文は必ず \n\n で終わる。末尾の空要素を落とした残りが「フレームの列」。
    raw_frames = body.split("\n\n")
    assert raw_frames[-1] == ""
    frames = raw_frames[:-1]

    # tool + delta*3 + done + [DONE] = 6 フレーム、1フレームちょうど1行。
    # JSON が正しくエスケープされていれば delta 内の改行は "\\n" の2文字になり、
    # 生の改行はフレーム区切り以外に現れない。
    assert len(frames) == 6
    for f in frames:
        assert "\n" not in f
        assert f.startswith("data: ")

    payloads = _data_payloads(body)
    deltas = [p["delta"] for p in payloads if "delta" in p]
    assert deltas == tokens  # 前後の空白も含めて完全に round-trip する
    assert "".join(deltas) == "行1\n行2\n\n  前後に空白  "


def test_stream_endpoint_returns_error_frame_when_database_is_down(monkeypatch):
    """SQLAlchemyError 分岐を実際の DB 障害で踏む。到達不能なホストへ向けた engine を
    差し込むだけなので、fixture も上流 LLM も要らない。DB 障害を「上流モデルが…」と
    誤って報告する退行(SQLAlchemyError 分岐の削除)を検出する。"""
    monkeypatch.setattr("app.db.base.async_session", _dead_session_factory())
    _use_model(FakeModel([AIMessage(content="hi")]))

    client = TestClient(app)
    with client.stream(
        "POST", "/api/agent/stream", json={"user_id": "u1", "message": "hi"}
    ) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode("utf-8")

    expected_frame = "event: error\ndata: {}\n\n".format(
        json.dumps({"message": _DB_DOWN_MSG}, ensure_ascii=False)
    )
    assert expected_frame in body
    assert "[DONE]" not in body


# --- 12B: /api/agent --------------------------------------------------------


def _fake_result() -> AgentResult:
    tm = ToolMessage(
        content='{"status": "輸送中"}', tool_call_id="c1", name="query_logistics"
    )
    return AgentResult(
        conversation_id=12,
        answer="注文 1001 は現在輸送中です。",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
        tool_runs=[ToolRun("c1", "query_logistics", True, tm)],
    )


def test_agent_endpoint_returns_tool_trace(monkeypatch):
    async def fake_run(user_id, message, conversation_id, model=None):
        return _fake_result()

    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "注文 1001 は今どこですか"})
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == 12
    assert body["answer"] == "注文 1001 は現在輸送中です。"
    assert body["tool_calls"][0]["name"] == "query_logistics"
    assert body["tool_calls"][0]["id"] == "c1"
    assert body["tool_calls"][0]["args"] == {"order_id": "1001"}
    assert body["tool_results"][0]["ok"] is True
    assert body["tool_results"][0]["tool_call_id"] == "c1"
    assert body["tool_results"][0]["name"] == "query_logistics"
    assert body["tool_results"][0]["content"] == '{"status": "輸送中"}'


def test_agent_endpoint_survives_tool_call_with_none_id(monkeypatch):
    """id=None の tool_call(langchain_core の ToolCall.id: str | None。id を省略する
    OpenAI 互換ゲートウェイ経由で実際に到達しうる)でも、成功したターンを日本語メッセージの
    ない素の 500 で捨てない。app/api/agent.py の `.get("id") or ""` の回帰テスト。"""

    async def fake_run(user_id, message, conversation_id, model=None):
        res = _fake_result()
        res.tool_calls = [{"name": "query_logistics", "args": {"order_id": "1001"}, "id": None}]
        return res

    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "注文 1001 は今どこですか"})
    assert r.status_code == 200
    assert r.json()["tool_calls"][0]["id"] == ""


def test_agent_endpoint_404_on_unknown_conversation(monkeypatch):
    async def fake_run(*a, **k):
        raise agent.ConversationNotFound(999)

    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post(
        "/api/agent", json={"user_id": "u1", "message": "hi", "conversation_id": 999}
    )
    assert r.status_code == 404
    assert r.json()["detail"] == _NOT_FOUND_MSG


def test_agent_endpoint_502_on_upstream_failure(monkeypatch):
    async def fake_run(*a, **k):
        raise RuntimeError("上流モデル呼び出し失敗(テスト用)")

    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 502
    # 文言そのものを固定する。ステータスコードだけだと、detail に例外クラス名や
    # ファイル名・行番号といった内部詳細が漏れる退行を検出できない
    # (この章で一度実際に起きた形)。
    assert r.json()["detail"] == _UPSTREAM_MSG


def test_agent_endpoint_422_on_missing_fields(monkeypatch):
    # run_agent_turn を「呼ばれたら失敗する」フェイクに差し替えるのは、この検証が
    # バリデーションだけを対象にしていることを明示するためだけではなく、安全装置でもある。
    # 差し替えないと、スキーマ側のバリデーションが壊れた瞬間にこのテストが本物の
    # オーケストレーションへ流れ落ち、本番相当の support DB へ書き込み、実際に上流 LLM を
    # 呼んでしまう(実測で確認済み: user_id を必須から外すミューテーションで実際に発生した)。
    async def must_not_run(*a, **k):
        raise AssertionError("バリデーションで弾かれるべきリクエストが本体まで到達した")

    monkeypatch.setattr(agent, "run_agent_turn", must_not_run)
    client = TestClient(app)
    assert client.post("/api/agent", json={"message": "hi"}).status_code == 422


def test_agent_endpoint_forwards_request_fields_to_core(monkeypatch):
    """リクエストの3フィールドが run_agent_turn へそのまま渡ることを固定する。

    他の /api/agent テストは引数を無視するフェイクを使うため、ハンドラが
    conversation_id を落として常に新規会話を開始するようになっても全部緑のままになる
    (マルチターンが静かに壊れ、Task 14 の eval はこのエンドポイントを叩く)。
    ストリーム側は unknown-conversation テストが conversation_id を実際に送るので
    既に固定されているが、こちらは無防備だった。
    """
    seen: dict = {}

    async def fake_run(user_id, message, conversation_id, model=None):
        seen.update(user_id=user_id, message=message, conversation_id=conversation_id)
        return AgentResult(conversation_id or 1, "a", [], [])

    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post(
        "/api/agent", json={"user_id": "u9", "message": "m9", "conversation_id": 77}
    )
    assert r.status_code == 200
    assert seen == {"user_id": "u9", "message": "m9", "conversation_id": 77}
    assert r.json()["conversation_id"] == 77


def test_agent_endpoint_handles_block_style_tool_content(monkeypatch):
    """ToolMessage.content がブロック形式(list[dict])でも 200 で正しく返す。

    ToolResultView.content が .content(生の list)を受けていると、ここは
    ValidationError → detail のない素の英語 500 になる。しかも DB 行・チケット・回答は
    既に永続化された「成功したターン」なので、成功したのに捨てられる。
    .text は type="text" のブロックだけを連結し、reasoning などの非公開ブロックを落とす。
    """
    blocks = [
        {"type": "text", "text": "配送中です"},
        {"type": "reasoning", "text": "内部推論は非公開"},
    ]
    tm = ToolMessage(content=blocks, tool_call_id="c1", name="query_logistics")

    async def fake_run(user_id, message, conversation_id, model=None):
        return AgentResult(
            conversation_id=12,
            answer="配送中です。",
            tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
            tool_runs=[ToolRun("c1", "query_logistics", True, tm)],
        )

    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "注文1001はどこですか"})
    assert r.status_code == 200
    content = r.json()["tool_results"][0]["content"]
    assert content == "配送中です"
    assert isinstance(content, str)  # ブロックの list がそのまま漏れ出さない
    assert "非公開" not in content


def test_tool_result_view_content_must_be_a_string():
    """ToolResultView.content: str は API の公開契約であり、二重防御の外側の層でもある。

    ハンドラが .text を使っている限り実行時には常に str が入るので、この契約は
    エンドツーエンドのテストからは見えない(型を object に緩めても全部緑のまま通る)。
    schema 層で直接固定して、緩められたことに気づけるようにする。
    """
    with pytest.raises(PydanticValidationError):
        ToolResultView(
            tool_call_id="c1",
            name="query_logistics",
            ok=True,
            content=[{"type": "text", "text": "配送中です"}],
        )


def test_agent_endpoint_500_with_japanese_detail_on_response_build_failure(monkeypatch):
    """応答組み立ての ValidationError は、素の英語 500 ではなく日本語 detail 付きの
    500 になる(そして DB/上流の障害を意味する 502・503 とは区別される)。

    tool_call_id=None は app/tools/infra.py の `or "unknown"` 防御と .text 化により
    実運用では到達しないはずの形。最後の砦が実際に砦として働くことを固定する。
    """
    tm = ToolMessage(content="x", tool_call_id="c1", name="query_logistics")

    async def fake_run(user_id, message, conversation_id, model=None):
        return AgentResult(
            conversation_id=12,
            answer="ご案内します。",
            tool_calls=[],
            tool_runs=[ToolRun(None, "query_logistics", True, tm)],
        )

    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 500
    assert r.json()["detail"] == "応答の生成に失敗しました"


def test_agent_endpoint_503_when_database_is_down(monkeypatch):
    """DB 障害は 503 + DB 用の文言。502「上流モデルが…」に落ちる退行
    (SQLAlchemyError 分岐の削除)を検出する。リトライ判断が変わるため、
    この 503 と 502 の区別は利用者にも自動リトライにも意味がある。"""
    monkeypatch.setattr("app.db.base.async_session", _dead_session_factory())
    # 安全網: この新規テスト群で唯一、本物の run_agent_turn を呼ぶ。DB ガードと
    # SQLAlchemyError 分岐が同時に退行した場合、ここは本物の上流を叩いて実際に
    # 1ターン消費してしまう(support への書き込みはできないが、課金は発生する)。
    # 兄弟テストが must_not_run を仕込んでいるのと同じ理由。/api/agent は model を
    # Depends で取らないので dependency_overrides では塞げず、core 側を差し替える。
    monkeypatch.setattr(
        "app.core.agent.get_chat_model", lambda *a, **k: _MustNotBeUsedModel()
    )

    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 503
    assert r.json()["detail"] == _DB_DOWN_MSG


@pytest.mark.parametrize("blank", _BLANK_VARIANTS)
def test_agent_endpoint_rejects_whitespace_only_input(blank, monkeypatch):
    async def must_not_run(*a, **k):
        raise AssertionError("バリデーションで弾かれるべきリクエストが本体まで到達した")

    monkeypatch.setattr(agent, "run_agent_turn", must_not_run)
    client = TestClient(app)
    assert (
        client.post("/api/agent", json={"user_id": "u1", "message": blank}).status_code == 422
    )
    assert (
        client.post("/api/agent", json={"user_id": blank, "message": "hi"}).status_code == 422
    )
