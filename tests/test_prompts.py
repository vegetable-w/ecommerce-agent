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
    """Verify that tool names hardcoded in the prompts match registry.

    AGENT_SYSTEM mentions tool names by string (e.g. "query_order").
    If a tool is renamed in the registry, the prompt name must stay in sync
    or the model will be told to call a tool that no longer exists under
    that name. This test catches that desync.

    06 章から、常時渡す AGENT_SYSTEM だけでは足りない。submit_refund は返金フローでしか
    呼ばせたくないツールで、AGENT_SYSTEM に書くと全経路で「返金申請を出す」選択肢が
    見えてしまう(注文を特定していない経路でも呼ばれる)。そのため使い方は
    REFUND_JUDGE_HINT の側にあり、対応の検査もその 2 つを合わせて行う。
    """
    from app.core.prompts import REFUND_JUDGE_HINT
    from app.tools.registry import get_all_tools

    # Get actual tool names from registry
    registry_names = {t.name for t in get_all_tools()}

    # Extract tool names mentioned in the prompts (query_* / create_* / submit_* pattern)
    instructions = AGENT_SYSTEM + REFUND_JUDGE_HINT
    mentioned_names = set(re.findall(r'\b(?:query|create|submit)_[a-z_]+\b', instructions))

    # Forward: all registry tools should be mentioned in the prompts
    for name in registry_names:
        assert name in mentioned_names, f"Tool '{name}' in registry but not mentioned in the prompts"

    # 08 章の移行期間だけの例外。query_logistics は built-in から外し、配送状況の照会は
    # MCP 側へ移した。AGENT_SYSTEM の書き換えは MCP 側のツール名が決まってから行うので、
    # それまでの間だけ prompt が registry に無い名前をモデルへ案内している状態が続く。
    # 例外そのものを assert しておくことで、prompt から名前が消えた時点でここも落ち、
    # この抜け道が黙って残り続けないようにする。
    in_transition = {"query_logistics"}
    assert in_transition <= mentioned_names, \
        "prompt から query_logistics が消えたなら、この移行用の例外も外すこと"

    # Reverse: all mentioned names should correspond to real tools in registry
    for name in mentioned_names - in_transition:
        assert name in registry_names, f"Tool '{name}' mentioned in the prompts but not in registry"


# ---- 04 章: RAG 生成 / セルフチェック / 忠実性 ---------------------------------

# 各ルールを 1 つだけ特定できるアンカー。短い語(「番号」「拒否」「約束」など)は
# 使わない: 下の DECOY のようにルールを 1 つも表現していない文でも通ってしまい、
# assertion が同語反復になる(01 章の "ない" in PROMPT と同じ失敗)。
_RAG_ANSWER_ANCHORS = (
    "回答内の重要な結論には、その直後に根拠番号を付ける",   # 引用ルール
    "根拠番号を付けられない結論は回答に含めない",           # 根拠外を書かない
    "「現在、関連する情報を確認できませんでした」と明示したうえで",  # 回答拒否
    "時間をおいての再試行は案内しない",                     # ツール障害との切り分け
    "「プラットフォームの実際の処理状況に準じます」と案内する",      # 時効を約束しない
    "補償の金額や期限を約束しない",                         # 補償を約束しない
    "取り扱いのない商品について、仕様や価格をでっち上げない",        # 捏造しない
)

# ルールを 1 つも表現していないのに、素朴なアンカー(「番号」「evidence」「拒否」
# 「約束」「案内」)は全部含む文字列。アンカーの識別力を測るための対照。
_DECOY = (
    "あなたは案内係です。evidence の番号を読み上げ、拒否されたら別の窓口へ回し、"
    "約束の時間に集合してください。"
)


def test_rag_answer_anchors_are_not_tautological():
    """アンカーがルールの存在を本当に指しているか(デコイでは落ちるか)を確かめる。"""
    for anchor in _RAG_ANSWER_ANCHORS:
        assert anchor not in _DECOY, f"アンカー '{anchor}' はルール抜きの文でも通る"


def test_rag_answer_system_covers_citation_refusal_and_negative_knowledge():
    from app.core.prompts import RAG_ANSWER_SYSTEM

    for anchor in _RAG_ANSWER_ANCHORS:
        assert anchor in RAG_ANSWER_SYSTEM


def test_rag_answer_system_uses_japanese_shipping_facts():
    """引用例の金額はこのナレッジベースの実際の送料ポリシー(3,000円)であること。"""
    from app.core.prompts import RAG_ANSWER_SYSTEM

    assert "3,000円以上のご注文は送料無料です[1]" in RAG_ANSWER_SYSTEM
    assert "元" not in RAG_ANSWER_SYSTEM


def test_rag_refusal_rule_does_not_collide_with_agent_tool_error_rule():
    """AGENT_SYSTEM のツール障害ルールと、根拠不足の回答拒否ルールを取り違えさせない。

    ツール障害(上流が落ちた)は再試行の案内が正しいが、根拠不足(検索は成功したが
    ナレッジに該当が無い)で再試行を案内すると、直らないものを待たせることになる。
    """
    from app.core.prompts import AGENT_SYSTEM, RAG_ANSWER_SYSTEM, RAG_INSUFFICIENT_NOTICE

    assert "しばらく時間をおいて再度お試しいただくよう案内する" in AGENT_SYSTEM
    assert "しばらく時間をおいて再度お試し" not in RAG_ANSWER_SYSTEM
    assert "時間をおいての再試行は案内しない" in RAG_ANSWER_SYSTEM
    # 生成 prompt 側では行き先を create_ticket に揃える(モデルへの指示)
    assert "create_ticket" in RAG_ANSWER_SYSTEM
    # ただし拒否の notice は「収束生成」が読む。そのターンではもうツールを呼べないので、
    # ここでツール名を出すと、モデルが実行できない行動を宣言してしまう
    # (実測: 「チケットを作成いたしますね」と言うが、実際には作られない)。
    assert "create_ticket" not in RAG_INSUFFICIENT_NOTICE
    assert "約束してはいけません" in RAG_INSUFFICIENT_NOTICE
    # ただしツール名はモデルへの指示であって、ユーザーへ見せる文面ではない
    # (実測: この一文が無いと「オペレーター対応(create_ticket)をお願いいたします」と
    #  そのまま出力された)
    assert "ツール名や内部の仕組みをユーザーへの文面に書かない" in RAG_ANSWER_SYSTEM
    assert "ツール名や内部の仕組みはユーザーへの文面に書かないでください" in RAG_INSUFFICIENT_NOTICE


def test_rag_insufficient_notice_carries_the_instruction_in_its_text():
    """根拠不足はフラグではなく本文で伝える。

    02 章の実測: ToolMessage(status="error") の status は上流へ渡る際に落ち、
    モデルには本文しか届かない。sufficient=False という真偽値だけでは
    モデルは何をすべきか分からないため、指示そのものを本文に書く。
    """
    from app.core.prompts import RAG_INSUFFICIENT_NOTICE

    assert "推測で回答を作らず" in RAG_INSUFFICIENT_NOTICE
    assert "確認できなかったことをユーザーへ正直に伝えてください" in RAG_INSUFFICIENT_NOTICE
    assert "時間をおいての再試行は案内しないでください" in RAG_INSUFFICIENT_NOTICE


def test_rag_answer_prompt_renders_query_and_evidence():
    from app.core.prompts import RAG_ANSWER_PROMPT

    msgs = RAG_ANSWER_PROMPT.format_messages(
        query="送料はいくらですか", evidence="[1] 3,000円以上のご注文は送料無料です"
    )
    assert msgs[0].type == "system"
    assert "送料はいくらですか" in msgs[-1].content
    assert "[1] 3,000円以上のご注文は送料無料です" in msgs[-1].content


def test_faithfulness_prompt_renders_evidence_and_answer():
    from app.core.prompts import FAITHFULNESS_PROMPT, FAITHFULNESS_SYSTEM

    # 文言そのものではなく、judge の判定を左右する取り決めが載っていることを見る。
    # 実測: これらが抜けると judge は「答えていない/足りない」を理由に回答拒否を
    # 幻覚と呼び、人手の判断との一致率が 10/11 から 0/11 へ落ちる。
    assert "回答拒否" in FAITHFULNESS_SYSTEM          # 拒否は事実の主張ではない
    assert "網羅" in FAITHFULNESS_SYSTEM              # 網羅性は忠実性ではない
    assert "条件のすり替え" in FAITHFULNESS_SYSTEM      # 幻覚の類型 B
    assert "でっち上げた数値" in FAITHFULNESS_SYSTEM     # 幻覚の類型 A
    msgs = FAITHFULNESS_PROMPT.format_messages(
        evidence="[1] 3,000円以上のご注文は送料無料です",
        answer="3,000円以上で送料無料です[1]",
    )
    assert msgs[0].type == "system"
    assert "[1] 3,000円以上のご注文は送料無料です" in msgs[-1].content
    assert "3,000円以上で送料無料です[1]" in msgs[-1].content


def test_self_check_system_defines_both_verdicts():
    from app.core.prompts import SELF_CHECK_SYSTEM

    assert "useful=true: 回答に必要な情報が evidence に含まれている" in SELF_CHECK_SYSTEM
    assert "質問の一部にしか答えられない" in SELF_CHECK_SYSTEM
    assert "evidence の外にある知識で補って判定しない" in SELF_CHECK_SYSTEM


def test_rag_insufficient_notice_bans_general_knowledge_answers():
    """「推測するな」だけでは足りない。一般知識の列挙を止める文言が要る。

    実測: この一文が無いと、モデルは公開知識(例: 火星探査車の型番)を並べることを
    「推測ではない」と解釈して回答してしまい、3/3 で拒否しなかった。
    追加後は、ナレッジに無い商品仕様の質問で 3/3 拒否になった。
    """
    from app.core.prompts import RAG_INSUFFICIENT_NOTICE

    assert "一般知識として知っている内容も、ここでは回答してはいけません" in RAG_INSUFFICIENT_NOTICE
    assert "確認できませんでした" in RAG_INSUFFICIENT_NOTICE


def test_rag_insufficient_notice_does_not_promise_unreachable_actions():
    """収束生成のターンではもうツールを呼べない。

    実測: 旧文面では「チケットを作成いたしますね」と宣言したが、単一ターン制約により
    実際には作成されなかった。ユーザーへの空約束になるため、希望を尋ねる案内に留める。
    """
    from app.core.prompts import RAG_INSUFFICIENT_NOTICE

    assert "実行できない行動を約束してはいけません" in RAG_INSUFFICIENT_NOTICE
    assert "create_ticket" not in RAG_INSUFFICIENT_NOTICE


def test_both_answer_paths_carry_the_condition_rule():
    """生産経路と評価経路で「条件を落とすな」の規則を揃える。

    生産の収束生成が読むのは 02 章で凍結した AGENT_SYSTEM なので、この規則は
    tool の本文(RAG_CITATION_NOTICE)でしか届かない。評価側は RAG_ANSWER_SYSTEM を
    使う。片方だけに入れると、評価が本番より良い生成器を測ることになる。
    """
    from app.core.prompts import RAG_ANSWER_SYSTEM, RAG_CITATION_NOTICE

    for text in (RAG_ANSWER_SYSTEM, RAG_CITATION_NOTICE):
        assert "条件" in text
        assert "表の 1 行" in text
        assert "通常" in text
