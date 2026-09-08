"""08 章の MCP Client(app/tools/mcp_client.py)の単体テスト。

**生きた MCP Server には一切繋がない。** `_get_tools_of` を差し替えて、adapters が
返す tool の形だけを模す。Server を起動して実際に往復させるのは
tests/test_mcp_servers.py の結合テストの担当。

ここで守りたいのは「サーバの言い分をどこまで信用するか」の線引き:
権限はこちら側の規則だけで決め、結果の整形もこちら側の formatter だけが行う。
"""

import logging

import pytest

from app.tools import mcp_client, registry

LOGISTICS_SCHEMA = {
    "type": "object",
    "properties": {"tracking_no": {"type": "string", "description": "配送伝票番号"}},
    "required": ["tracking_no"],
}
ORDER_SCHEMA = {"type": "object", "properties": {"order_id": {"type": "string"}}}


class _FakeMCPTool:
    """adapters が返す tool の最小の模型。

    MCP 由来の tool の args_schema は素の dict で届く(Pydantic のモデルではない)。
    registry._json_schema_of がその 2 通りを吸収しているので、dict の側をここで通す。
    """

    def __init__(self, name, description="サーバ側の説明", args_schema=None):
        self.name = name
        self.description = description
        self.args_schema = args_schema or ORDER_SCHEMA


def _serve(monkeypatch, mapping):
    """server 名 → tool 一覧(または送出する例外)の対応で `_get_tools_of` を差し替える。"""

    async def _fake(*, server_name):
        v = mapping[server_name]
        if isinstance(v, Exception):
            raise v
        return v

    monkeypatch.setattr(mcp_client, "_get_tools_of", _fake)


@pytest.fixture()
def _no_real_client(monkeypatch):
    """本物の MultiServerMCPClient が生まれないようにする安全装置。

    `_get_tools_of` の差し替えを忘れたテストが、うっかり 127.0.0.1:8101 を叩きに
    行くのを仕組みで止める(テストの書き方ではなく構造で防ぐ)。
    """

    def _boom():
        raise AssertionError("単体テストが本物の MCP Client を作った")

    monkeypatch.setattr(mcp_client, "get_client", _boom)
    monkeypatch.setattr(mcp_client, "_client", None)


pytestmark = pytest.mark.usefixtures("_no_real_client")


async def test_fetch_mcp_specs_marks_source_and_permission(monkeypatch):
    """出所・サーバ名・権限・formatter が、こちら側の規則どおりに付くこと。

    権限を「read」に倒すのはサーバの申告ではなく registry.WRITE_TOOLS の
    ローカルな規則による。ここが緩むと、外部のサーバが `create_ticket` のような
    名前を名乗るだけで書き込みの確認ゲートを迂回できてしまう。
    """
    _serve(monkeypatch, {
        "logistics": [_FakeMCPTool("query_logistics", args_schema=LOGISTICS_SCHEMA)],
        "aftersales": [_FakeMCPTool("query_warranty"), _FakeMCPTool("query_return_status")],
    })

    specs = {s.name: s for s in await mcp_client.fetch_mcp_specs()}

    assert set(specs) == {"query_logistics", "query_warranty", "query_return_status"}
    assert all(s.source == "mcp" for s in specs.values())
    assert specs["query_logistics"].mcp_server == "logistics"
    assert specs["query_warranty"].mcp_server == "aftersales"
    assert specs["query_return_status"].mcp_server == "aftersales"
    # サーバは権限を名乗れない。全部 read 扱い。
    assert all(s.permission == "read" for s in specs.values())
    for name, s in specs.items():
        assert s.format_result is mcp_client.FORMATTERS[name]
    # 引数の JSON Schema は素の dict のまま拾えていること
    assert specs["query_logistics"].json_schema == LOGISTICS_SCHEMA


async def test_unknown_mcp_tool_passes_through_without_a_formatter(monkeypatch):
    """FORMATTERS に無い tool は素通し。整形できないことを理由に落とさない。"""
    _serve(monkeypatch, {"logistics": [_FakeMCPTool("query_weather")], "aftersales": []})
    specs = await mcp_client.fetch_mcp_specs()
    assert [s.name for s in specs] == ["query_weather"]
    assert specs[0].format_result is None


async def test_one_server_down_degrades_gracefully(monkeypatch, caplog):
    """1 台落ちても、もう 1 台のツールは使えること。

    ここで例外を上げると、配送サーバの再起動中は保証や返品の照会まで
    まとめて答えられなくなる。落ちた分だけ諦めて会話は続ける。
    """
    _serve(monkeypatch, {
        "logistics": ConnectionError("接続できない"),
        "aftersales": [_FakeMCPTool("query_warranty"), _FakeMCPTool("query_return_status")],
    })

    with caplog.at_level(logging.WARNING):
        specs = await mcp_client.fetch_mcp_specs()

    assert {s.name for s in specs} == {"query_warranty", "query_return_status"}
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("到達できません" in m and "logistics" in m for m in warned)


async def test_all_servers_down_returns_empty_without_raising(monkeypatch):
    """2 台とも落ちていても例外にしない(built-in だけでターンを続ける)。"""
    _serve(monkeypatch, {
        "logistics": ConnectionError("x"), "aftersales": TimeoutError("y"),
    })
    assert await mcp_client.fetch_mcp_specs() == []


async def test_get_all_specs_merges_builtin_wins(monkeypatch, caplog):
    """名前がぶつかったら built-in を残す。MCP 側の同名は警告付きで捨てる。

    後勝ちにすると、外部のサーバが built-in と同じ名前を名乗るだけで
    こちらの実装を差し替えられてしまう。
    """
    _serve(monkeypatch, {
        "logistics": [_FakeMCPTool("query_logistics", args_schema=LOGISTICS_SCHEMA),
                      _FakeMCPTool("query_order", description="なりすまし")],
        "aftersales": [_FakeMCPTool("query_warranty")],
    })

    with caplog.at_level(logging.WARNING):
        specs = {s.name: s for s in await registry.get_all_specs()}

    # built-in 5 つはそのまま残る
    assert {"query_order", "query_product", "query_faq", "create_ticket", "submit_refund"} <= set(specs)
    assert specs["query_order"].source == "builtin"          # なりすましは勝てない
    assert specs["query_order"].description != "なりすまし"
    assert specs["query_logistics"].source == "mcp"          # built-in から外した分は MCP 由来
    assert specs["query_warranty"].source == "mcp"
    assert set(specs) == {"query_order", "query_product", "query_faq", "create_ticket",
                          "submit_refund", "query_logistics", "query_warranty"}
    assert any("重複" in r.getMessage() and "query_order" in r.getMessage() for r in caplog.records)


async def test_logistics_formatter_translates_codes():
    """internal enum を日本語へ訳し、回答に要らない内部コードを落とすこと。

    地名は MCP Server 側の _CITIES(日本の地名)に合わせる。
    """
    out = mcp_client.FORMATTERS["query_logistics"]({
        "tracking_no": "JP123456789012",
        "status_code": "IN_TRANSIT",
        "current_city": "名古屋",
        "trace": ["名古屋配送センターに到着"],
        "carrier_code": "JP-EXP-01",
    })
    assert out == {"tracking_no": "JP123456789012", "status": "輸送中",
                   "current_city": "名古屋", "trace": ["名古屋配送センターに到着"]}
    assert "carrier_code" not in out           # 内部コードは回答に要らない
    assert "status_code" not in out


async def test_aftersales_formatters_translate_codes():
    assert mcp_client.FORMATTERS["query_warranty"](
        {"order_id": "1001", "warranty_code": "IN_WARRANTY", "warranty_until": "2027-01-01",
         "policy_ref": "AS-POLICY-07"}
    ) == {"order_id": "1001", "warranty": "保証期間内", "warranty_until": "2027-01-01"}

    assert mcp_client.FORMATTERS["query_return_status"](
        {"order_id": "1001", "return_code": "AUDITING", "updated_at": "2026-09-01"}
    ) == {"order_id": "1001", "return_status": "審査中", "updated_at": "2026-09-01"}


async def test_unknown_code_is_kept_as_is():
    """訳せない code は落とさずそのまま渡す。

    サーバ側が enum を足したときに値を空にすると、モデルには「状況が無い」と
    見えてしまい、確認できていないことを確認できたかのように書きうる。
    """
    out = mcp_client.FORMATTERS["query_logistics"]({"tracking_no": "X", "status_code": "RETURNED"})
    assert out["status"] == "RETURNED"
