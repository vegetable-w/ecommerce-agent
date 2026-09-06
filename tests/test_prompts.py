import re

from langchain_core.messages import HumanMessage

from app.core.prompts import CUSTOMER_SERVICE_PROMPT, EXTRACT_PROMPT, AGENT_SYSTEM, AGENT_PROMPT

def test_customer_service_prompt_renders_with_history():
    msgs = CUSTOMER_SERVICE_PROMPT.format_messages(
        history=[HumanMessage("キャットフードは売っていますか？")]
    )
    assert msgs[0].type == "system"
    assert msgs[-1].content == "キャットフードは売っていますか？"
    # 行動制約は必ずsystem promptに含める
    for keyword in ("推測・捏造しない", "照会システムへのアクセス権限がない", "このセッション内でユーザー自身が伝えた情報", "アフターサービス規約に準じます", "まず気持ちに配慮"):
        assert keyword in msgs[0].content

def test_extract_prompt_renders_text():
    msgs = EXTRACT_PROMPT.format_messages(text="注文 MH1 が壊れたので返金してほしい")
    assert msgs[0].type == "system"
    assert "注文 MH1 が壊れたので返金してほしい" in msgs[-1].content

def test_agent_system_covers_tool_principles():
    """Strengthened keyword test with unique anchors for each of the six rules."""
    # Rule 1: Three-tool rule (query_order / query_product / query_logistics)
    assert "query_order / query_product / query_logistics" in AGENT_SYSTEM

    # Rule 2: query_faq usage rule
    assert "query_faq でキーワード検索してFAQを確認する" in AGENT_SYSTEM

    # Rule 3: create_ticket and conversation_id rule
    assert "チケットに紐づく conversation_id はシステムが設定するため、推測しない" in AGENT_SYSTEM

    # Rule 4: Out-of-scope chat rule (don't call tools for non-support chat)
    assert "不要なツールは呼び出さない" in AGENT_SYSTEM

    # Rule 5: Honesty rule (don't fabricate data)
    assert "情報を作らない" in AGENT_SYSTEM

    # Rule 6: Refund policy rule
    assert "プラットフォームのアフターサービス規定に従います" in AGENT_SYSTEM

    # Rule 7: Tool failure handling rule
    assert "エラーの内部的なメッセージをそのまま伝えたり" in AGENT_SYSTEM

def test_agent_prompt_has_history_placeholder():
    """Test that AGENT_PROMPT has the history placeholder."""
    assert any(getattr(m, "variable_name", None) == "history"
               for m in AGENT_PROMPT.messages)

def test_agent_system_names_match_registry():
    """Verify that tool names hardcoded in AGENT_SYSTEM match registry.

    AGENT_SYSTEM mentions tool names by string (e.g. "query_order").
    If a tool is renamed in the registry, the prompt name must stay in sync
    or the model will be told to call a tool that no longer exists under
    that name. This test catches that desync.
    """
    from app.tools.registry import get_all_tools

    # Get actual tool names from registry
    registry_names = {t.name for t in get_all_tools()}

    # Extract tool names mentioned in AGENT_SYSTEM (query_* or create_* pattern)
    mentioned_names = set(re.findall(r'\b(?:query|create)_[a-z_]+\b', AGENT_SYSTEM))

    # Forward: all registry tools should be mentioned in the prompt
    for name in registry_names:
        assert name in mentioned_names, f"Tool '{name}' in registry but not mentioned in AGENT_SYSTEM"

    # Reverse: all mentioned names should correspond to real tools in registry
    for name in mentioned_names:
        assert name in registry_names, f"Tool '{name}' mentioned in AGENT_SYSTEM but not in registry"
