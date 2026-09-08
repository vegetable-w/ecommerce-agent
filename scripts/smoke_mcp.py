"""08 章の red line。MCP の実際の API と往復を実測する。上流の LLM は呼ばない。

plan は Context7 の記述をもとに「high-level class は MCPServer(FastMCP は rename 済み)、
transport option は run() へ渡す」を前提にしている。**実装前にそれを確かめる。**
ここが違うと Task 4 の Server 2 台が丸ごと書き直しになる。

使い方: PYTHONUTF8=1 uv run --env-file .env python scripts/smoke_mcp.py
"""

import asyncio
import importlib
import inspect
import subprocess
import sys
import textwrap
from importlib.metadata import version
from pathlib import Path

PORT = 8199   # 本番の 8101 / 8102 とぶつけない

SERVER_SRC = textwrap.dedent('''
    from mcp.server.fastmcp import FastMCP

    # host/port は constructor 側。run() は transport と mount_path しか取らない
    mcp = FastMCP("smoke", host="127.0.0.1", port={port})


    @mcp.tool()
    def echo_order(order_id: str) -> dict:
        """注文番号をそのまま返すだけの確認用 tool。"""
        return {{"order_id": order_id, "status": "配送中"}}


    if __name__ == "__main__":
        mcp.run(transport="streamable-http")
''')


def report_api() -> None:
    """どの class が実在し、transport option をどこで受けるかを出す。"""
    print(f"mcp version = {version('mcp')}")
    print(f"langchain-mcp-adapters version = {version('langchain-mcp-adapters')}")
    for path, name in (("mcp.server.mcpserver", "MCPServer"),
                       ("mcp.server.fastmcp", "FastMCP")):
        try:
            cls = getattr(importlib.import_module(path), name)
        except Exception as e:
            print(f"  {path}.{name}: なし ({e.__class__.__name__})")
            continue
        init = list(inspect.signature(cls.__init__).parameters)
        print(f"  {path}.{name}: あり")
        print(f"     host/port を constructor で受けるか: "
              f"{'host' in init and 'port' in init}")
        print(f"     run() = {inspect.signature(cls.run)}")


async def round_trip(proc_port: int) -> None:
    """adapters 経由で tool list を取り、実際に 1 回呼ぶ。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({
        # transport の key は "streamable_http"(ハイフンではない)
        "smoke": {"url": f"http://127.0.0.1:{proc_port}/mcp",
                  "transport": "streamable_http"},
    })
    tools = await client.get_tools()
    print(f"  取得した tool = {[t.name for t in tools]}")
    t = next(t for t in tools if t.name == "echo_order")
    print(f"  args_schema の型 = {type(t.args_schema).__name__}")
    print(f"  args_schema      = {t.args_schema}")
    out = await t.ainvoke({"order_id": "1001"})
    print(f"  実行結果         = {out!r}")


async def main() -> int:
    print("--- A) 実装されている API ---")
    report_api()

    src = Path(__file__).with_name("_smoke_mcp_server.py")
    src.write_text(SERVER_SRC.format(port=PORT), encoding="utf-8")
    print(f"\n--- B) Server を起動して往復する(port {PORT}) ---")
    proc = subprocess.Popen([sys.executable, str(src)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(60):          # 起動待ち。落ちていれば早く抜ける
            await asyncio.sleep(0.5)
            if proc.poll() is not None:
                print("  Server が起動直後に終了した:")
                print((proc.stdout.read() or b"").decode("utf-8", "replace"))
                return 1
            try:
                await round_trip(PORT)
                return 0
            except Exception as e:
                last = e
        print(f"  接続できなかった: {last!r}")
        return 1
    finally:
        proc.terminate()
        src.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
