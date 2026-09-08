"""08 章の MCP Server 結合テスト。

real subprocess として 2 台を起動し、adapters の client で tool list を取り、
実際に tool を呼ぶ。接続できない場合は environment issue として fail させる
(2 台の Server 自体が本章の deliverable なので skip しない)。

port は 18101 / 18102。本番の 8101 / 8102 とはぶつけない。

Task 5 からは、この上に **配線そのもの**の結合テストも載せている:
registry.get_all_specs() が built-in と MCP を実際に合成できること、
engine が MCP のツールを実行して formatter の効いた日本語を返すこと、
そして「注文 → 追跡番号 → 配送状況」の連鎖が 2 つの実装をまたいで成立すること。
上流のモデルは一切呼ばない(連鎖はツールの戻り値を手で渡して確かめる)。
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools import engine, mcp_client, registry

# tests/conftest.py の _no_live_mcp が既定で塞ぐ本物の実装。import は
# monkeypatch より先に走るので、ここで掴んでおけば結合テストだけ元へ戻せる。
_REAL_GET_TOOLS_OF = mcp_client._get_tools_of

_ROOT = Path(__file__).resolve().parent.parent
_PORTS = {"logistics": 18101, "aftersales": 18102}
# 起動待ちの上限。Windows では subprocess の起動が遅く、固定 sleep は flaky になるので
# 「接続できるまで試す」形にしてある(plan の time.sleep(2.0) は使わない)。
_READY_TIMEOUT = 60.0
_POLL_INTERVAL = 0.5


def _texts(result) -> list[str]:
    """tool の戻りの content block から text だけを取り出す。

    戻りは `[{'type': 'text', 'text': '...', 'id': 'lc_<uuid>'}]` の形で、`id` は
    **呼び出しごとに変わる**(adapters 0.3.2 で実測)。plan の `r1 == r2` はそのままでは
    成立しないので、比較は id を含まない text の側だけで行う。
    """
    return [b["text"] for b in result if b.get("type") == "text"]


def _client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            # transport の key は "streamable_http"(ハイフンではない)。scripts/smoke_mcp.py で実測済み。
            name: {"transport": "streamable_http", "url": f"http://127.0.0.1:{port}/mcp"}
            for name, port in _PORTS.items()
        },
        handle_tool_errors=False,
    )


@pytest.fixture(scope="module")
def mcp_procs():
    procs: list[tuple[str, subprocess.Popen, Path]] = []
    sinks = []
    tmp = Path(tempfile.mkdtemp(prefix="mcp-test-"))
    try:
        for name, port in _PORTS.items():
            log = tmp / f"{name}.log"
            sink = log.open("wb")
            sinks.append(sink)
            procs.append((
                name,
                subprocess.Popen(
                    [sys.executable, str(_ROOT / "mcp_servers" / f"{name}_server.py")],
                    # PORT だけ差し替える。MOCK_DELAY_SECONDS は継承させない
                    # (遅延ありの環境でテストを回すと理由の分からない timeout になる)。
                    env={**os.environ, "PORT": str(port), "MOCK_DELAY_SECONDS": "0",
                         "PYTHONUTF8": "1"},
                    stdout=sink, stderr=subprocess.STDOUT,
                    cwd=str(_ROOT),
                ),
                log,
            ))
        _wait_ready(procs)
        yield
    finally:
        for _, p, _log in procs:
            p.terminate()
        for _, p, _log in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        for sink in sinks:
            sink.close()


def _wait_ready(procs) -> None:
    """2 台とも tool list を返せるようになるまで待つ。落ちていれば log を添えて fail する。"""
    deadline = time.monotonic() + _READY_TIMEOUT
    last: Exception | None = None
    while time.monotonic() < deadline:
        for name, p, log in procs:
            if p.poll() is not None:
                pytest.fail(
                    f"{name}_server.py が起動直後に終了した(exit={p.returncode})。log:\n"
                    + log.read_text(encoding="utf-8", errors="replace")
                )
        try:
            asyncio.run(_client().get_tools())
            return
        except Exception as e:      # まだ listen していないだけかもしれないので握って再試行
            last = e
        time.sleep(_POLL_INTERVAL)
    logs = "\n".join(
        f"--- {name} ---\n" + log.read_text(encoding="utf-8", errors="replace")
        for name, _p, log in procs
    )
    pytest.fail(f"MCP Server へ {_READY_TIMEOUT} 秒以内に接続できなかった: {last!r}\n{logs}")


async def test_list_tools_three_essentials(mcp_procs):
    """2 台合わせて 3 つの tool が、説明と JSON Schema 付きで見えること。"""
    tools = await _client().get_tools()
    by = {t.name: t for t in tools}
    assert set(by) == {"query_logistics", "query_warranty", "query_return_status"}
    for t in by.values():
        assert t.description                                    # 用途の説明
        schema = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
        assert schema.get("properties")                         # JSON Schema の引数定義


async def test_logistics_docstring_keeps_the_order_first_rule(mcp_procs):
    """query_logistics の説明が「注文番号をそのまま渡すな」の護柵を保っていること。

    05 章の受け入れ条件(注文 → 追跡番号 → 配送 の連鎖)が通る理由そのものなので、
    MCP へ移しても文言を落とさない。
    """
    tools = {t.name: t for t in await _client().get_tools(server_name="logistics")}
    desc = tools["query_logistics"].description
    assert "query_order" in desc
    assert "注文番号をそのまま渡してはならない" in desc


async def test_invoke_logistics_stable_mock(mcp_procs):
    """同じ伝票番号は同じ結果。internal enum が翻訳されずに残っていること。"""
    tools = {t.name: t for t in await _client().get_tools(server_name="logistics")}
    r1 = _texts(await tools["query_logistics"].ainvoke({"tracking_no": "JP123456789012"}))
    r2 = _texts(await tools["query_logistics"].ainvoke({"tracking_no": "JP123456789012"}))
    assert r1 == r2                                             # 固定 seed
    body = "".join(r1)
    assert "status_code" in body                                # internal enum の key が残る
    assert any(code in body for code in
               ("PICKED_UP", "IN_TRANSIT", "DELIVERING", "DELIVERED"))
    # 地名は日本のもの。日本語は escape されず、そのまま text に乗る。
    assert "配送センター" in body
    assert any(city in body for city in ("東京", "横浜", "名古屋", "大阪", "福岡"))


async def test_invoke_aftersales_tools(mcp_procs):
    """aftersales の 2 つの tool が internal enum を返すこと。"""
    tools = {t.name: t for t in await _client().get_tools(server_name="aftersales")}
    w = "".join(_texts(await tools["query_warranty"].ainvoke({"order_id": "1001"})))
    assert "warranty_code" in w
    assert "IN_WARRANTY" in w or "EXPIRED" in w
    assert "AS-POLICY-07" in w                                  # internal policy ref も落とさない
    r = "".join(_texts(await tools["query_return_status"].ainvoke({"order_id": "1001"})))
    assert "return_code" in r
    assert any(code in r for code in ("AUDITING", "RETURNING", "REFUNDED", "NONE"))


# ---------------------------------------------------------------------------
# Task 5 の配線(registry の合成 → engine の実行 → formatter)を実 Server で通す
# ---------------------------------------------------------------------------


@pytest.fixture()
def wired_to_test_servers(mcp_procs, monkeypatch):
    """settings の URL をテスト用の port へ向け、本物の MCP Client を使わせる。

    _client はモジュール変数として使い回されるので、URL を差し替えたら None へ
    戻さないと本番 port(8101 / 8102)へ向いた古い client がそのまま使われる。
    """
    monkeypatch.setattr(mcp_client, "_get_tools_of", _REAL_GET_TOOLS_OF)
    monkeypatch.setattr(mcp_client.settings, "mcp_logistics_url",
                        f"http://127.0.0.1:{_PORTS['logistics']}/mcp")
    monkeypatch.setattr(mcp_client.settings, "mcp_aftersales_url",
                        f"http://127.0.0.1:{_PORTS['aftersales']}/mcp")
    monkeypatch.setattr(mcp_client, "_client", None)
    yield
    mcp_client._client = None      # 次のテストへ test port 向きの client を残さない


@pytest.fixture()
def audits(monkeypatch):
    """監査の書き込みをフェイクへ。結合テストから本番相当の DB へ書かないための安全装置。"""
    rows = []

    async def fake_audit(**kw):
        rows.append(kw)

    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)
    return rows


async def test_get_all_specs_merges_builtin_and_live_mcp(wired_to_test_servers):
    """built-in と、実際に起動した 2 台の MCP の合成が 1 つの一覧になること。"""
    specs = {s.name: s for s in await registry.get_all_specs()}

    assert {"query_order", "query_product", "query_faq", "create_ticket",
            "submit_refund", "query_logistics", "query_warranty", "query_return_status"} == set(specs)
    assert specs["query_order"].source == "builtin"
    assert specs["query_logistics"].source == "mcp"
    assert specs["query_logistics"].mcp_server == "logistics"
    assert specs["query_warranty"].mcp_server == "aftersales"
    # 権限はこちら側の規則だけで決まる。MCP のツールは全部 read。
    assert all(s.permission == "read" for s in specs.values() if s.source == "mcp")
    assert specs["create_ticket"].permission == "write"


async def test_engine_runs_an_mcp_tool_and_formats_it_in_japanese(wired_to_test_servers, audits):
    """engine 経由で MCP のツールを実行し、内部 code が日本語へ変換されて返ること。

    Server 側は internal enum のまま返す。翻訳と不要フィールドの除去は
    こちら側の formatter の担当で、そこが繋がっているかはここでしか分からない。
    """
    specs = {s.name: s for s in await registry.get_all_specs()}
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": "JP123456789012"}, "id": "c1"},
        0, specs,
    )

    assert run.ok is True and run.status == engine.STATUS_SUCCESS
    payload = json.loads(run.tool_message.content)
    assert payload["tracking_no"] == "JP123456789012"
    assert payload["status"] in {"集荷済み", "輸送中", "配達中", "配達完了"}   # 訳されている
    assert "status_code" not in payload and "carrier_code" not in payload   # 内部 code は落ちる
    assert payload["current_city"] in {"東京", "横浜", "名古屋", "大阪", "福岡"}
    assert audits[-1]["tool_source"] == "mcp" and audits[-1]["mcp_server"] == "logistics"
    assert audits[-1]["status"] == "success"


async def test_the_tracking_number_actually_chains_the_two_tools(wired_to_test_servers, audits):
    """注文 → 追跡番号 → 配送状況 の連鎖が、built-in と MCP をまたいで成立すること。

    05 章の受け入れ条件 #5 そのもの。08 章で query_logistics が MCP 側へ移り、
    追跡番号を渡す相手が別プロセスになったので、両側の実装が食い違うと静かに壊れる。
    上流のモデルは呼ばず、query_order の戻り値をそのまま次のツールへ渡して確かめる。
    """
    specs = {s.name: s for s in await registry.get_all_specs()}

    order = json.loads((await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1001"}, "id": "o1"}, 0, specs
    )).tool_message.content)
    tracking_no = order["tracking_no"]
    assert tracking_no

    logi = json.loads((await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": tracking_no}, "id": "l1"}, 0, specs
    )).tool_message.content)
    assert logi["tracking_no"] == tracking_no

    # 別の注文からは別の追跡番号が出る(定数を返していないこと)
    other = json.loads((await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "2002"}, "id": "o2"}, 0, specs
    )).tool_message.content)
    assert other["tracking_no"] != tracking_no


async def test_an_order_id_cannot_be_passed_to_the_logistics_tool(wired_to_test_servers, audits):
    """注文番号では配送を引けないこと(連鎖を強制している仕組みそのもの)。

    両ツールが同じ order_id を取れてしまうと、モデルが 1 step で 2 つを並べて呼べて
    順序依存が消える。追跡番号を query_order の戻り値からしか得られなくすることで
    その順序を強制している。engine では JSON Schema の検証がこれを弾く。
    """
    specs = {s.name: s for s in await registry.get_all_specs()}
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}, 0, specs
    )
    assert run.ok is False and run.status == engine.STATUS_VALIDATION_BLOCKED
