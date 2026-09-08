"""08 章の ToolSpec レジストリのテスト。

built-in ツールは app/tools/builtin/ 配下の module を import した時点で自己登録する。
registry 側は scan して束ねるだけで、ツールの名前を 1 つも持たない
(「ファイルを足すだけで増える」という受け入れ条件がこの形に乗っている)。
"""

import logging

from app.tools import registry


def test_builtin_scan_registers_five_tools_without_query_logistics():
    names = {s.name for s in registry.builtin_specs()}
    assert names == {"query_order", "query_product", "query_faq", "create_ticket", "submit_refund"}
    # 08: 配送状況の照会は MCP 側が担う。built-in の query_logistics は廃止し、
    # 同じ名前が 2 か所に存在する状態を残さない


def test_every_spec_has_three_essentials():
    for s in registry.builtin_specs():
        assert s.name and s.description                      # 名前と使い方の説明
        assert isinstance(s.json_schema, dict) and s.json_schema.get("properties") is not None


def test_permissions_only_trust_our_side():
    by = {s.name: s for s in registry.builtin_specs()}
    assert by["create_ticket"].permission == "write"
    assert all(s.permission == "read" for n, s in by.items() if n != "create_ticket")
    assert registry.permission_for("mcp_random_tool") == "read"   # 未登録の MCP ツールは read 扱い


def test_create_ticket_schema_excludes_injected_conversation_id():
    spec = registry.get_builtin_spec("create_ticket")
    assert spec.inject_conversation is True
    assert "conversation_id" not in spec.json_schema.get("properties", {})  # 注入する引数はモデルへ見せない


def test_duplicate_register_is_dropped_with_warning(caplog):
    spec = registry.builtin_specs()[0]
    dup = registry.ToolSpec(name=spec.name, description="dup", json_schema={"type": "object", "properties": {}},
                            tool=spec.tool, permission="read", source="builtin")
    with caplog.at_level(logging.WARNING):
        registry.register(dup)
    assert registry.get_builtin_spec(spec.name).description != "dup"   # 先に登録した方を残す
    assert any("重複" in r.message for r in caplog.records)
