"""/api/agent(非ストリーミング JSON)のテスト。

05 章でこのエンドポイントは graph の ainvoke 入口になった(spec §7 / D2)。旧 2-call
orchestration(app/core/agent.py)は通らない。ストリーミングの入口は /api/chat 1 つに
なったので、旧 /api/agent/stream のテストはここには無い(tests/test_chat_api.py を参照)。

**上流 model にも本番相当の DB にも一切触らない。** graph は台本どおりの最終 State を
返す偽物に、repository は記録用の偽物に差し替える。差し替え忘れは _no_real_upstream が
その場で落とす(黙って本物を叩いて課金と本番書き込みを起こさせない)。
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Interrupt
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.graph import runtime
from app.main import app
from app.schemas.agent import ToolResultView

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
    pytest.param(" ", id="nbsp"),
]


@pytest.fixture(autouse=True)
def _no_real_upstream(monkeypatch):
    """本物の DB・上流 model・本物の graph を既定で塞ぐ。

    - async_session を到達不能なホストへ向ける。差し替え忘れたテストが support(本番相当)へ
      書き込むのではなく SQLAlchemyError で落ちる。DB 障害のテストはこの既定をそのまま使う。
    - get_chat_model を落とす。偽 graph の組み立てを間違えた瞬間に本物の課金が発生するため。
    - get_graph を落とす。lifespan が走らないテストでは本来 RuntimeError になるが、
      「偽 graph を注入し忘れた」ことが 502 の陰に隠れないよう明示的に失敗させる。
    """
    monkeypatch.setattr("app.db.base.async_session", _dead_session_factory())

    def _boom_model(*args, **kwargs):
        raise AssertionError("テストが本物の上流 model を呼んだ")

    monkeypatch.setattr("app.core.llm.get_chat_model", _boom_model)
    monkeypatch.setattr("app.graph.nodes.get_chat_model", _boom_model)

    def _boom_graph():
        raise AssertionError("テストが偽 graph を注入し忘れた")

    monkeypatch.setattr(runtime, "get_graph", _boom_graph)


def _dead_session_factory():
    """本番相当の DB が落ちている状況を作る session factory。

    repository は app.db.base.async_session を参照するので、そこをこれに差し替えると
    run_turn の最初の create_conversation が実際に SQLAlchemyError を送出する。
    """
    return async_sessionmaker(create_async_engine(_DEAD_DB_URL), expire_on_commit=False)


class _FakeGraph:
    """台本どおりの最終 State を返す偽 graph。呼ばれた入力と config を記録する。"""

    def __init__(self, state: dict) -> None:
        self._state = state
        self.invocations: list[tuple[dict, dict]] = []

    async def ainvoke(self, inp, config):
        self.invocations.append((inp, config))
        return self._state


class _Conv:
    """get_conversation が返す会話。runtime が読むのは要約の 2 つだけ(07 章)。"""

    summary = ""
    summary_upto_msg_id = 0


def _use_graph(monkeypatch, state: dict, *, new_id: int = 101) -> _FakeGraph:
    """偽 graph と偽 repository を差し込む。戻り値で graph への入力を検査できる。"""
    graph = _FakeGraph(state)
    monkeypatch.setattr(runtime, "get_graph", lambda: graph)

    async def _create(user_id):
        return new_id

    async def _get(conversation_id):
        return _Conv()

    async def _append(conversation_id, role, content=None, tool_calls=None, tool_call_id=None):
        return 1

    async def _no_summary(conversation_id):
        """要約の起動は 07 章で turn の後に必ず走る。本物は DB を叩くので塞ぐ。"""

    monkeypatch.setattr(runtime.repository, "create_conversation", _create)
    monkeypatch.setattr(runtime.repository, "get_conversation", _get)
    monkeypatch.setattr(runtime.repository, "append_message", _append)
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", _no_summary)
    return graph


def _tool_turn_state() -> dict:
    """tool を 1 回使って収束した turn の最終 State。"""
    return {
        "messages": [
            HumanMessage("注文 1001 は今どこですか"),
            AIMessage(
                "",
                tool_calls=[
                    {"name": "query_logistics", "args": {"tracking_no": "JP213502378238"}, "id": "c1"}
                ],
            ),
            ToolMessage(
                content='{"status": "輸送中"}', tool_call_id="c1", name="query_logistics"
            ),
            AIMessage("注文 1001 は現在輸送中です。"),
        ],
        "suggested_actions": [],
    }


def test_agent_endpoint_returns_tool_trace(monkeypatch):
    """graph を通した turn の answer と tool trace がそのまま JSON になる。"""
    _use_graph(monkeypatch, _tool_turn_state())
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "注文 1001 は今どこですか"})

    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == 101
    assert body["answer"] == "注文 1001 は現在輸送中です。"
    assert body["tool_calls"][0]["name"] == "query_logistics"
    assert body["tool_calls"][0]["id"] == "c1"
    assert body["tool_calls"][0]["args"] == {"tracking_no": "JP213502378238"}
    assert body["tool_results"][0]["ok"] is True
    assert body["tool_results"][0]["tool_call_id"] == "c1"
    assert body["tool_results"][0]["name"] == "query_logistics"
    assert body["tool_results"][0]["content"] == '{"status": "輸送中"}'
    assert body["suggested_actions"] == []


def test_agent_endpoint_returns_no_tool_calls_for_chitchat(monkeypatch):
    """雑談は決定的な node が答えるので tool は 1 つも呼ばれず、回答は state["answer"] から来る。

    resolve_answer が messages の末尾しか見ない実装に戻ると、ここは空文字になる。
    """
    _use_graph(monkeypatch, {"answer": "こんにちは。ご用件をどうぞ。", "messages": [HumanMessage("こんにちは")]})
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "こんにちは"})

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "こんにちは。ご用件をどうぞ。"
    assert body["tool_calls"] == []
    assert body["tool_results"] == []
    assert body["suggested_actions"] == []


def test_agent_endpoint_returns_suggested_actions(monkeypatch):
    """選択肢は State から素通しで返る。/api/agent は評価の入口なので、
    「チケットを提案したか」をここで確認できる必要がある。"""
    actions = [
        {"type": "transfer_human"},
        {"type": "create_ticket", "draft": {"description": "商品が破損していた", "ticket_type": "return"}},
    ]
    _use_graph(monkeypatch, {"answer": "ご不便をおかけしました。", "suggested_actions": actions})
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "壊れていた"})

    assert r.status_code == 200
    assert r.json()["suggested_actions"] == actions


def test_agent_endpoint_forwards_request_fields_to_runtime(monkeypatch):
    """リクエストの3フィールドが run_turn へそのまま渡ることを固定する。

    他のテストは引数を無視する偽物を使うため、ハンドラが conversation_id を落として
    常に新規会話を開始するようになっても全部緑のままになる(マルチターンが静かに壊れる)。
    """
    seen: dict = {}

    async def fake_run(user_id, message, conversation_id):
        seen.update(user_id=user_id, message=message, conversation_id=conversation_id)
        # run_turn の戻り値の形をそのまま真似る。interrupt を省くと、
        # ハンドラが中断を読み落としても気づけない偽物になる
        return {"conversation_id": conversation_id or 1, "state": {"answer": "a"},
                "interrupt": None}

    monkeypatch.setattr(runtime, "run_turn", fake_run)
    client = TestClient(app)
    r = client.post(
        "/api/agent", json={"user_id": "u9", "message": "m9", "conversation_id": 77}
    )
    assert r.status_code == 200
    assert seen == {"user_id": "u9", "message": "m9", "conversation_id": 77}
    assert r.json()["conversation_id"] == 77


def test_agent_endpoint_runs_the_turn_on_the_requested_thread(monkeypatch):
    """graph へ渡る thread_id が会話 ID であること(checkpointer の切り分けの要)。

    ここが固定されていないと、継続のつもりの turn が毎回まっさらな State で走る退行に
    気づけない。/api/agent は評価の入口なので、マルチターンの評価が静かに壊れる。
    """
    graph = _use_graph(monkeypatch, {"answer": "はい。"})
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "続きです", "conversation_id": 7})

    assert r.status_code == 200
    inp, config = graph.invocations[0]
    assert config["configurable"]["thread_id"] == "7"
    assert inp["conversation_id"] == 7
    assert inp["user_id"] == "u1"
    assert [m.content for m in inp["messages"]] == ["続きです"]


def test_agent_endpoint_survives_tool_call_with_none_id(monkeypatch):
    """id=None の tool_call(langchain_core の ToolCall.id: str | None。id を省略する
    OpenAI 互換ゲートウェイ経由で実際に到達しうる)でも、成功したターンを日本語メッセージの
    ない素の 500 で捨てない。app/api/agent.py の `.get("id") or ""` の回帰テスト。"""
    ai = AIMessage.model_construct(
        content="", tool_calls=[{"name": "query_logistics", "args": {}, "id": None}]
    )
    _use_graph(monkeypatch, {"messages": [ai, AIMessage("輸送中です。")]})
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "注文 1001 は今どこですか"})

    assert r.status_code == 200
    assert r.json()["tool_calls"][0]["id"] == ""


def test_agent_endpoint_handles_block_style_tool_content(monkeypatch):
    """ToolMessage.content がブロック形式(list[dict])でも 200 で正しく返す。

    ToolResultView.content が .content(生の list)を受けていると、ここは
    ValidationError → 500 になる。しかも DB 行も回答も既に永続化された
    「成功したターン」なので、成功したのに捨てられる。
    """
    tm = ToolMessage(
        content=[
            {"type": "text", "text": "配送中です"},
            {"type": "reasoning", "text": "内部推論は非公開"},
        ],
        tool_call_id="c1",
        name="query_logistics",
    )
    _use_graph(monkeypatch, {"messages": [tm, AIMessage("配送中です。")]})
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "注文1001はどこですか"})

    assert r.status_code == 200
    content = r.json()["tool_results"][0]["content"]
    assert content == "配送中です"
    assert isinstance(content, str)  # ブロックの list がそのまま漏れ出さない
    assert "非公開" not in content


def test_agent_endpoint_404_on_unknown_conversation(monkeypatch):
    async def fake_run(*a, **k):
        raise runtime.ConversationNotFound(999)

    monkeypatch.setattr(runtime, "run_turn", fake_run)
    client = TestClient(app)
    r = client.post(
        "/api/agent", json={"user_id": "u1", "message": "hi", "conversation_id": 999}
    )
    assert r.status_code == 404
    assert r.json()["detail"] == _NOT_FOUND_MSG


def test_agent_endpoint_502_on_upstream_failure(monkeypatch):
    async def fake_run(*a, **k):
        raise RuntimeError("上流モデル呼び出し失敗(テスト用)")

    monkeypatch.setattr(runtime, "run_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 502
    # 文言そのものを固定する。ステータスコードだけだと、detail に例外クラス名や
    # ファイル名・行番号といった内部詳細が漏れる退行を検出できない
    # (02 章で一度実際に起きた形)。
    assert r.json()["detail"] == _UPSTREAM_MSG


def test_agent_endpoint_503_when_database_is_down():
    """DB 障害は 503 + DB 用の文言。502「上流モデルが…」に落ちる退行
    (SQLAlchemyError 分岐の削除)を検出する。リトライ判断が変わるため、
    この 503 と 502 の区別は利用者にも自動リトライにも意味がある。

    偽 graph も偽 repository も入れない。_no_real_upstream が向けた到達不能な DB へ
    本物の run_turn が最初の create_conversation で当たり、graph へ着く前に落ちる。
    """
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 503
    assert r.json()["detail"] == _DB_DOWN_MSG


def test_agent_endpoint_500_with_japanese_detail_on_response_build_failure(monkeypatch):
    """応答組み立ての ValidationError は、素の英語 500 ではなく日本語 detail 付きの
    500 になる(そして DB/上流の障害を意味する 502・503 とは区別される)。

    args が dict でない tool_call は、langchain の検証を通る経路では作れない形。
    最後の砦が実際に砦として働くことを固定する。
    """
    ai = AIMessage.model_construct(
        content="", tool_calls=[{"name": "query_order", "args": "order_id=1001", "id": "c1"}]
    )
    _use_graph(monkeypatch, {"messages": [ai]})
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})

    assert r.status_code == 500
    assert r.json()["detail"] == "応答の生成に失敗しました"


def test_agent_endpoint_422_on_missing_fields(monkeypatch):
    # run_turn を「呼ばれたら失敗する」偽物に差し替えるのは、この検証がバリデーションだけを
    # 対象にしていることを明示するためだけではなく、安全装置でもある。差し替えないと、
    # スキーマ側のバリデーションが壊れた瞬間にこのテストが本物のオーケストレーションへ
    # 流れ落ちる(02 章で実測: user_id を必須から外すミューテーションで実際に発生した)。
    async def must_not_run(*a, **k):
        raise AssertionError("バリデーションで弾かれるべきリクエストが本体まで到達した")

    monkeypatch.setattr(runtime, "run_turn", must_not_run)
    client = TestClient(app)
    assert client.post("/api/agent", json={"message": "hi"}).status_code == 422


@pytest.mark.parametrize("blank", _BLANK_VARIANTS)
def test_agent_endpoint_rejects_whitespace_only_input(blank, monkeypatch):
    """min_length=1 は空白のみの値を通してしまう。空白のみの user 行を一度 DB に作ると、
    その会話の履歴に残り続ける(1 章の SessionStore と違い MySQL に残るので高くつく)。"""

    async def must_not_run(*a, **k):
        raise AssertionError("バリデーションで弾かれるべきリクエストが本体まで到達した")

    monkeypatch.setattr(runtime, "run_turn", must_not_run)
    client = TestClient(app)
    assert (
        client.post("/api/agent", json={"user_id": "u1", "message": blank}).status_code == 422
    )
    assert (
        client.post("/api/agent", json={"user_id": blank, "message": "hi"}).status_code == 422
    )


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


# --- 06 章: 注文の選択待ち(interrupt) ---------------------------------------------


_SELECT_ORDER = {
    "type": "select_order",
    "orders": [
        {"order_id": "1001", "product": "自動猫トイレ", "status": "支払い済み", "amount": 1739},
        {"order_id": "2002", "product": "スマート体重計", "status": "発送済み", "amount": 3280},
    ],
}


def test_agent_endpoint_surfaces_the_interrupt(monkeypatch):
    """注文が特定できないと fetch_order は interrupt で止まる。回答が無いまま
    200 を返すと、呼び出し側は「答えられなかった turn」と区別できない。
    中断の payload をそのまま載せて、何を選ばせればよいかを伝える。"""
    _use_graph(monkeypatch, {
        "messages": [HumanMessage("返金したいです")],
        "answer": "",
        "__interrupt__": [Interrupt(value=_SELECT_ORDER)],
    })
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "返金したいです"})

    assert r.status_code == 200
    assert r.json()["interrupt"] == _SELECT_ORDER


def test_agent_endpoint_reports_no_interrupt_as_null(monkeypatch):
    """中断していない turn でも key は必ずある。呼び出し側に key の有無で
    分岐させない(既定を空にする suggested_actions と同じ規約)。"""
    _use_graph(monkeypatch, _tool_turn_state())
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "注文 1001 は今どこですか"})

    body = r.json()
    assert "interrupt" in body
    assert body["interrupt"] is None
