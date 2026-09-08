"""08 章の統一実行エンジン app.tools.engine のテスト。

すべてフェイクのツールで完結し DB は触らない(insert_tool_audit を monkeypatch する)。
そのため tests/conftest.py の DB 用 fixture も、モジュールレベルの
`pytest.mark.asyncio(loop_scope="session")` も必要ない。
pyproject.toml の asyncio_mode="auto" が素の async 関数をそのまま拾うので、
tests/test_tools_infra.py の非 DB テストと同じくマーカーを付けない
(session ループへ載せると、無関係なテストの後片付けが互いに漏れる)。

audit の status は DB の ENUM に合わせた英語識別子。日本語の表示名は
app/core/labels.py の TOOL_AUDIT_STATUS だけが持つ。
"""

import asyncio
import json

import httpx
import pytest

from app.core import labels
from app.tools import engine
from app.tools.registry import ToolSpec

SCHEMA = {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}
TICKET_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
}
LOGISTICS_SCHEMA = {
    "type": "object",
    "properties": {"tracking_no": {"type": "string"}},
    "required": ["tracking_no"],
}


def _spec(name="query_order", *, permission="read", source="builtin", tool=None,
          schema=SCHEMA, timeout=None, inject=False, fmt=None):
    return ToolSpec(name=name, description="test tool", json_schema=schema, tool=tool,
                    permission=permission, source=source, timeout=timeout,
                    inject_conversation=inject, format_result=fmt)


def _tool(fn, name="query_order"):
    return type("T", (), {"name": name, "ainvoke": staticmethod(fn)})()


@pytest.fixture()
def audits(monkeypatch):
    """insert_tool_audit の呼び出しを記録するフェイク。

    engine は必ずキーワード引数で呼ぶので、位置引数への対応は要らない。

    後始末で「実際に書きに行った status が DDL の ENUM から外れていないか」を
    まとめて確認する。engine 側の _audit は例外を握り潰す(監査の失敗で実行を
    止めない)ので、フェイクの中で assert しても飲み込まれてしまう。
    テスト本体が終わった後のこの位置なら、確実に失敗として表に出る。
    """
    rows = []

    async def fake_audit(**kw):
        rows.append(kw)

    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)
    yield rows
    unknown = {r["status"] for r in rows} - set(labels.TOOL_AUDIT_STATUS)
    assert not unknown, f"DB の ENUM に無い status を書こうとした: {unknown}"


def test_statuses_are_english_db_identifiers():
    """engine が出す status は必ず labels.TOOL_AUDIT_STATUS のキーに収まる。

    この status はそのまま tool_audit_logs.status(ENUM)へ書き込まれるので、
    engine 側で綴りを間違えると ENUM に無い値を書きに行って落ちる。
    定数を 1 か所へ集めたうえで、その集合を対応表と突き合わせて塞ぐ。
    """
    assert engine.ALL_STATUSES == {
        engine.STATUS_SUCCESS,
        engine.STATUS_FAILED,
        engine.STATUS_TIMEOUT,
        engine.STATUS_VALIDATION_BLOCKED,
        engine.STATUS_PERMISSION_DENIED,
    }
    assert engine.ALL_STATUSES <= set(labels.TOOL_AUDIT_STATUS)
    # 日本語をそのまま status にしていないこと(DB へ書くのは英語識別子だけ)
    assert all(s.isascii() for s in engine.ALL_STATUSES)


async def test_unknown_tool(audits):
    run = await engine.execute_tool_call({"name": "nope", "args": {}, "id": "c1"}, 1, {})
    assert run.ok is False and "未知のツール" in run.tool_message.content
    assert run.status == engine.STATUS_FAILED
    assert audits[-1]["status"] == "failed"


async def test_validation_blocks_and_feeds_back(audits):
    async def boom(_):
        raise AssertionError("検証に失敗したら実行してはいけない")

    spec = _spec(tool=_tool(boom))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {}, "id": "c1"},
        1,
        {"query_order": spec},
    )
    assert run.ok is False and run.status == "validation_blocked"
    assert "パラメータ検証に失敗" in run.tool_message.content and run.tool_message.status == "error"
    assert audits[-1]["status"] == "validation_blocked" and audits[-1]["retry_count"] == 0


async def test_write_without_confirmation_denied(audits):
    async def boom(_):
        raise AssertionError("未確認の書き込み操作を実行してはいけない")

    spec = _spec("create_ticket", permission="write", tool=_tool(boom, "create_ticket"),
                 schema=TICKET_SCHEMA)
    run = await engine.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
        1,
        {"create_ticket": spec},
    )
    assert run.ok is False and run.status == "permission_denied"
    assert audits[-1]["status"] == "permission_denied"


async def test_write_confirmed_executes_and_not_retried(audits):
    calls = {"n": 0}

    async def flaky(_):
        calls["n"] += 1
        raise ConnectionError("一時的なネットワーク障害")   # 一過性でも書き込みは再試行しない

    spec = _spec("create_ticket", permission="write", tool=_tool(flaky, "create_ticket"),
                 schema=TICKET_SCHEMA)
    run = await engine.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
        1,
        {"create_ticket": spec},
        confirmed=True,
    )
    assert run.ok is False and calls["n"] == 1 and run.retry_count == 0


async def test_transient_timeout_retries_then_gives_up(audits):
    async def slow(_):
        await asyncio.sleep(1)

    spec = _spec(tool=_tool(slow), timeout=0.05)
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        1,
        {"query_order": spec},
    )
    assert run.ok is False and run.status == "timeout"
    assert run.retry_count == 2                      # settings.tool_max_retries の既定値
    assert audits[-1]["status"] == "timeout" and audits[-1]["retry_count"] == 2
    assert audits[-1]["duration_ms"] is not None


async def test_http_timeout_is_recorded_as_timeout(audits):
    """MCP ツールは HTTP 越しなので、上流の timeout は httpx 側の例外として届く。

    httpx.TimeoutException は組み込みの TimeoutError を継承しないため、
    isinstance(e, TimeoutError) だけで振り分けると「失敗」に化けて、
    監査ログから timeout の傾向が読めなくなる。
    """
    calls = {"n": 0}

    async def http_timeout(_):
        calls["n"] += 1
        raise httpx.ConnectTimeout("upstream timeout")

    spec = _spec("query_logistics", source="mcp", tool=_tool(http_timeout, "query_logistics"),
                 schema=LOGISTICS_SCHEMA)
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": "SF1"}, "id": "c1"},
        1,
        {"query_logistics": spec},
    )
    assert run.status == "timeout" and calls["n"] == 3   # 一過性なので再試行の対象
    assert audits[-1]["status"] == "timeout"


async def test_business_error_not_retried(audits):
    calls = {"n": 0}

    async def fail(_):
        calls["n"] += 1
        raise ValueError("business error")           # 一過性ではない → 再試行しない

    spec = _spec(tool=_tool(fail))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        1,
        {"query_order": spec},
    )
    assert run.ok is False and calls["n"] == 1
    assert "ツールは一時的に利用できません" in run.tool_message.content
    assert audits[-1]["status"] == "failed"


async def test_retry_succeeds_second_attempt(audits):
    calls = {"n": 0}

    async def flaky(_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("一時的なネットワーク障害")
        return {"order_id": "1", "status": "発送済み"}

    spec = _spec(tool=_tool(flaky))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        1,
        {"query_order": spec},
    )
    assert run.ok is True and calls["n"] == 2 and run.retry_count == 1
    assert "発送済み" in run.tool_message.content       # ensure_ascii=False。日本語を escape しない
    assert audits[-1]["status"] == "success" and audits[-1]["retry_count"] == 1


async def test_format_result_hook_translates_enum(audits):
    async def ok(_):
        return {"tracking_no": "SF1", "status_code": "IN_TRANSIT", "internal_ref": "x9"}

    def fmt(d):
        return {
            "tracking_no": d["tracking_no"],
            "status": {"IN_TRANSIT": "輸送中"}.get(d.get("status_code"), d.get("status_code")),
        }

    spec = _spec("query_logistics", source="mcp", tool=_tool(ok, "query_logistics"),
                 schema=LOGISTICS_SCHEMA, fmt=fmt)
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": "SF1"}, "id": "c1"},
        1,
        {"query_logistics": spec},
    )
    assert "輸送中" in run.tool_message.content and "internal_ref" not in run.tool_message.content
    assert audits[-1]["tool_source"] == "mcp"


async def test_mcp_content_blocks_are_unwrapped_before_formatting(audits):
    """MCP の戻り(content block の list)を剥がしてから formatter へ渡すこと。

    adapters は `[{"type": "text", "text": "...", "id": "lc_<uuid>"}]` を返す。
    dict でも JSON 文字列でもないので、剥がさずに渡すと format_result は
    「dict ではない」という理由で一度も呼ばれず、内部 enum も内部コードも
    そのままモデルへ流れる(実 Server と繋ぐまで表に出なかった穴)。
    id は呼び出しごとに変わるので、残すと監査の突き合わせもできなくなる。
    """
    async def ok(_):
        return [{"type": "text",
                 "text": '{"tracking_no": "SF1", "status_code": "IN_TRANSIT", "carrier_code": "X"}',
                 "id": "lc_1111"}]

    def fmt(d):
        return {"tracking_no": d["tracking_no"],
                "status": {"IN_TRANSIT": "輸送中"}.get(d.get("status_code"))}

    spec = _spec("query_logistics", source="mcp", tool=_tool(ok, "query_logistics"),
                 schema=LOGISTICS_SCHEMA, fmt=fmt)
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": "SF1"}, "id": "c1"},
        1,
        {"query_logistics": spec},
    )
    assert run.ok is True
    assert "輸送中" in run.tool_message.content
    assert "carrier_code" not in run.tool_message.content
    assert "lc_1111" not in run.tool_message.content       # block の id は答えに要らない


async def test_mcp_plain_text_block_is_passed_through(audits):
    """JSON でない text block は、そのままの文字列としてモデルへ渡すこと。"""
    async def ok(_):
        return [{"type": "text", "text": "該当する配送情報が見つかりません", "id": "lc_2"}]

    spec = _spec("query_logistics", source="mcp", tool=_tool(ok, "query_logistics"),
                 schema=LOGISTICS_SCHEMA, fmt=lambda d: {"never": "called"})
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": "SF1"}, "id": "c1"},
        1,
        {"query_logistics": spec},
    )
    assert run.tool_message.content == "該当する配送情報が見つかりません"


async def test_a_plain_list_result_is_not_mistaken_for_content_blocks(audits):
    """ただの list を返すツールを content block と取り違えないこと。"""
    async def ok(_):
        return [{"order_id": "1"}, {"order_id": "2"}]

    spec = _spec(tool=_tool(ok))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        1,
        {"query_order": spec},
    )
    assert run.ok is True
    assert json.loads(run.tool_message.content) == [{"order_id": "1"}, {"order_id": "2"}]


async def test_audit_failure_never_blocks_execution(monkeypatch):
    async def audit_boom(*a, **kw):
        raise RuntimeError("audit DB unavailable")

    monkeypatch.setattr(engine.repository, "insert_tool_audit", audit_boom)

    async def ok(_):
        return {"order_id": "1"}

    spec = _spec(tool=_tool(ok))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        1,
        {"query_order": spec},
    )
    assert run.ok is True                              # 監査の失敗は実行を止めない


async def test_inject_conversation_after_validation(audits):
    seen = {}

    async def ok(args):
        seen.update(args)
        return {"ticket_no": "T1"}

    schema = dict(TICKET_SCHEMA, additionalProperties=False)
    spec = _spec("create_ticket", permission="write", tool=_tool(ok, "create_ticket"),
                 schema=schema, inject=True)
    run = await engine.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
        7,
        {"create_ticket": spec},
        confirmed=True,
    )
    # 検証の後に注入するので、スキーマに無い引数でも衝突しない
    assert run.ok is True and seen.get("conversation_id") == 7
