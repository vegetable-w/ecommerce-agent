"""08 章の MCP Client。MultiServerMCPClient で複数の Server へ繋ぎ、
tool 一覧は毎回その場で取り直す(cache しない)。

adapters は tool 一覧の取得と呼び出しのたびに session を張り直すので、
Server 側へ tool を足しても本体を再起動せずに次の turn から見えるようになる。

権限と結果の整形は、こちら側だけが決める。Server が名乗る使い方の説明は
モデルの tool 選択の参考にはするが、実行してよいかどうかは
registry.WRITE_TOOLS のローカルなルールで決める(サーバの自己申告は信用しない)。
"""

import logging
from collections.abc import Callable

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings
from app.tools import registry
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)


def _translate(mapping: dict[str, str], code):
    """対応表に無い code はそのまま返す。

    Server 側が新しい enum を足したときに、翻訳できないからといって
    値そのものを落とすと、モデルには「配送状況が空」に見えてしまう。
    訳せないなら生の code を渡す方が、嘘をつかない分だけ害が小さい。
    """
    return mapping.get(code, code)


def _fmt_logistics(d: dict) -> dict:
    return {"tracking_no": d.get("tracking_no"),
            "status": _translate({"PICKED_UP": "集荷済み", "IN_TRANSIT": "輸送中",
                                  "DELIVERING": "配達中", "DELIVERED": "配達完了"}, d.get("status_code")),
            "current_city": d.get("current_city"),
            "trace": d.get("trace")}


def _fmt_warranty(d: dict) -> dict:
    return {"order_id": d.get("order_id"),
            "warranty": _translate({"IN_WARRANTY": "保証期間内", "EXPIRED": "保証期間終了"},
                                   d.get("warranty_code")),
            "warranty_until": d.get("warranty_until")}


def _fmt_return(d: dict) -> dict:
    return {"order_id": d.get("order_id"),
            "return_status": _translate({"AUDITING": "審査中", "RETURNING": "返品中",
                                         "REFUNDED": "返金済み", "NONE": "返品記録なし"},
                                        d.get("return_code")),
            "updated_at": d.get("updated_at")}


# 結果の整形はこちら側で登録する。回答に必要なフィールドだけを残し、
# internal enum の code を人が読める日本語へ変換する。
# ここに載っていない MCP tool は素通し(passthrough)。
FORMATTERS: dict[str, Callable[[dict], dict]] = {
    "query_logistics": _fmt_logistics,
    "query_warranty": _fmt_warranty,
    "query_return_status": _fmt_return,
}

_client: MultiServerMCPClient | None = None


def _connections() -> dict:
    return {"logistics": {"transport": "streamable_http", "url": settings.mcp_logistics_url},
            "aftersales": {"transport": "streamable_http", "url": settings.mcp_aftersales_url}}


def get_client() -> MultiServerMCPClient:
    global _client
    if _client is None:
        # handle_tool_errors=False:tool のエラーを ToolException として投げさせ、
        # 切り分けは engine 側で一括して行う(adapters に文字列へ潰させない)。
        _client = MultiServerMCPClient(_connections(), handle_tool_errors=False)
    return _client


async def _get_tools_of(*, server_name: str):
    """薄い wrapper。単体テストが monkeypatch で差し込む場所。"""
    return await get_client().get_tools(server_name=server_name)


async def fetch_mcp_specs() -> list[ToolSpec]:
    """接続できた Server の tool だけを ToolSpec にして返す。

    1 台落ちていても warning を出してその Server 分だけ諦める。会話全体を
    落とさないため(残りの Server と built-in だけで答えられることは多い)。
    """
    specs: list[ToolSpec] = []
    for server in _connections():
        try:
            tools = await _get_tools_of(server_name=server)
        except Exception as e:  # noqa: BLE001 - 1 台の不通で会話全体を落とさない
            logger.warning("MCP Server「%s」に到達できません。この turn では当該 tool を skip します:%s",
                           server, type(e).__name__)
            continue
        for t in tools:
            specs.append(registry.spec_from_langchain_tool(
                t, source="mcp", mcp_server=server, format_result=FORMATTERS.get(t.name)))
    return specs
