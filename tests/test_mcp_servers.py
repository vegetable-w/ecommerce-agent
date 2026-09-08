"""08 章の MCP Server 結合テスト。

real subprocess として 2 台を起動し、adapters の client で tool list を取り、
実際に tool を呼ぶ。接続できない場合は environment issue として fail させる
(2 台の Server 自体が本章の deliverable なので skip しない)。

port は 18101 / 18102。本番の 8101 / 8102 とはぶつけない。
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

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
