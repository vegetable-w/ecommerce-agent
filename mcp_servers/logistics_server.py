"""Logistics MCP Server(08 章で自作。mock data のみ。実システム / DB には接続しない)。

independent process として起動する:
    uv run python mcp_servers/logistics_server.py

acceptance 6:
    MOCK_DELAY_SECONDS=8 で起動すると各 tool call を 8 秒遅延させる。
    customer-support 側で timeout / retry / audit を確認するための注入口。

**API の形(実測)**: この環境の mcp 1.30.0 に `MCPServer` は無く(`mcp.server.mcpserver`
module ごと存在しない)、high-level class は `FastMCP` のまま。しかも `run()` が受けるのは
transport / mount_path だけなので、host / port は constructor 側で渡す
(`run(transport=..., host=..., port=...)` は TypeError になる)。scripts/smoke_mcp.py で実測済み。
"""

import asyncio
import os
import random
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# tool call 1 回あたりの遅延秒。既定は 0(遅延なし)。
_DELAY = float(os.environ.get("MOCK_DELAY_SECONDS", "0"))

# internal enum。**Server 側では日本語へ翻訳しない。** 人間が読む文への変換は
# client(customer-support)側の formatter の担当で、そこが 08 章の見せどころ。
_STATUS_CODES = ["PICKED_UP", "IN_TRANSIT", "DELIVERING", "DELIVERED"]
# 配送センターの所在地。02 章から built-in 版の mock が使っていた日本の地名に揃える。
_CITIES = ["東京", "横浜", "名古屋", "大阪", "福岡"]

mcp = FastMCP("logistics", host="127.0.0.1", port=int(os.environ.get("PORT", "8101")))


@mcp.tool()
async def query_logistics(
    tracking_no: Annotated[str, Field(
        description="配送伝票番号。query_order の戻り値の tracking_no をそのまま渡す。"
                    "注文番号ではない(例: JP123456789012)")],
) -> dict:
    """配送伝票番号から配送状況、現在地、配送履歴を確認する。ユーザーが荷物の現在地や配送状況を尋ねた場合に使用する。

    **tracking_no は query_order の結果からのみ取得できる。** ユーザーが伝票番号を直接
    伝えてきた場合を除き、まず query_order で注文を確認して戻り値の tracking_no を得てから、
    その番号でこのツールを呼ぶこと。注文番号をそのまま渡してはならない。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    # 固定 seed。同じ伝票番号なら何度呼んでも同じ結果を返す(再試行や監査の突き合わせのため)。
    rng = random.Random(f"logistics:{tracking_no}")
    code = rng.choice(_STATUS_CODES)
    city = rng.choice(_CITIES)
    return {
        "tracking_no": tracking_no,
        "status_code": code,                  # internal enum。Server 側では翻訳しない
        "current_city": city,
        "trace": [f"{city}配送センターに到着", f"{city}配送センターから発送"],
        "carrier_code": "JP-EXP-01",          # internal carrier code。回答には不要なので client 側で除外
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
