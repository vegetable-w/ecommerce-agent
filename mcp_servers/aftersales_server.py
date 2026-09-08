"""Aftersales MCP Server(08 章で自作。mock data のみ。実システム / DB には接続しない)。

independent process として起動する:
    uv run python mcp_servers/aftersales_server.py

acceptance 6:
    MOCK_DELAY_SECONDS=8 で起動すると各 tool call を 8 秒遅延させる。
    customer-support 側で timeout / retry / audit を確認するための注入口。

**API の形(実測)**: この環境の mcp 1.30.0 に `MCPServer` は無く、high-level class は
`FastMCP` のまま。`run()` は transport / mount_path しか受けないので、host / port は
constructor 側で渡す(logistics_server.py と同じ理由)。scripts/smoke_mcp.py で実測済み。
"""

import asyncio
import os
import random
from datetime import date, datetime, timedelta
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# tool call 1 回あたりの遅延秒。既定は 0(遅延なし)。
_DELAY = float(os.environ.get("MOCK_DELAY_SECONDS", "0"))

# internal enum。**Server 側では日本語へ翻訳しない**(client 側の formatter の担当)。
_WARRANTY_CODES = ["IN_WARRANTY", "EXPIRED"]
_RETURN_CODES = ["AUDITING", "RETURNING", "REFUNDED", "NONE"]
# 保証期限を今日から何日ずらすかの幅。固定の年月日を書くと時間が経つほど全部が過去日になり、
# IN_WARRANTY なのに期限切れ、という矛盾した mock になる(02 章で注文日に同じ問題があった)。
_WARRANTY_OFFSET_DAYS = (30, 300)
# 返品ステータスの最終更新は直近の日付にする。理由は上と同じ。
_RETURN_AGE_DAYS = (0, 14)

mcp = FastMCP("aftersales", host="127.0.0.1", port=int(os.environ.get("PORT", "8102")))


@mcp.tool()
async def query_warranty(
    order_id: Annotated[str, Field(description="注文番号。例: 1001")],
) -> dict:
    """注文商品の保証期間内/期間終了と保証期限を確認する。
    ユーザーが warranty / 保証期間について質問した場合に使用する。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    # 固定 seed。同じ注文番号なら何度呼んでも同じ結果を返す。
    rng = random.Random(f"warranty:{order_id}")
    code = rng.choice(_WARRANTY_CODES)
    days = rng.randint(*_WARRANTY_OFFSET_DAYS)
    until = date.today() + timedelta(days=days if code == "IN_WARRANTY" else -days)
    return {
        "order_id": order_id,
        "warranty_code": code,                    # internal enum。Server 側では翻訳しない
        "warranty_until": until.strftime("%Y-%m-%d"),
        "policy_ref": "AS-POLICY-07",             # internal policy ref。回答には不要
    }


@mcp.tool()
async def query_return_status(
    order_id: Annotated[str, Field(description="注文番号。例: 1001")],
) -> dict:
    """注文の返品進捗(審査中 / 返品中 / 返金済み / 返品記録なし)を確認する。
    ユーザーが返品処理の進捗を問い合わせた場合に使用する。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"return:{order_id}")
    code = rng.choice(_RETURN_CODES)
    updated = datetime.now() - timedelta(days=rng.randint(*_RETURN_AGE_DAYS))
    return {
        "order_id": order_id,
        "return_code": code,                      # internal enum。Server 側では翻訳しない
        "updated_at": updated.strftime("%Y-%m-%d 10:00"),
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
