from langchain_core.messages import HumanMessage

from app.core.prompts import CUSTOMER_SERVICE_PROMPT, EXTRACT_PROMPT

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
