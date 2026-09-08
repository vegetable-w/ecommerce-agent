"""graph の組み立て。node と分岐関数をつないで 1 つの実行可能な graph にする。

この章で graph を使う理由は、**hard constraint を経路として固定する**ことにある。
「苦情は Agent に入らない」「ポリシーや仕様の質問は必ず retrieval を通ってから答える」は
prompt でのお願いではなく、そこへ通じる辺が存在しないという構造で担保する。
プロンプトは破られうるが、無い辺は通れない。

    START → coref → classify_intent → [route_by_intent]
       ├ escalate        → complaint_reply → log → END
       ├ fallback_script → script_reply    → log → END
       ├ knowledge       → forced_rag → confidence_check → [confidence_gate]
       │                      weak   → fallback_reply → log → END
       │                      strong → agent_llm ↘
       ├ refund_flow     → fetch_order → retrieve_policy → agent_llm
       └ business  ──────────────────────────────────────→ agent_llm

    agent_llm → [should_continue] ─ continue → agent_tools → agent_llm
                                   └ stop     → log → END

分岐は 3 箇所だけで、いずれも conditional edge の path_map に行き先を書き切ってある。
モデルが決めるのは should_continue が読む「tool を呼ぶかどうか」だけで、
経路そのものは決めさせない。

返金の 3 node が 1 本道なのも同じ理由。「注文を特定する → 規約を引く → 判断する」の
順序を Agent に任せると、注文を確かめないまま規約だけで、あるいは規約を引かずに
一般論で答えてしまう。順序を辺として固定すれば、飛ばす手段が無くなる。

4 つの出口をすべて log へ集めるのは、保存とログを出口ごとに書くと、出口が増えたときに
書き忘れた経路だけ履歴に残らないため。END へ入る辺は log からの 1 本だけにしてある。
"""

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.routing import confidence_gate, route_by_intent, should_continue
from app.graph.state import ConversationState


# routing の出口名 → 最初に入る node。
#
# **route_by_intent が返しうる値と、この dict のキーは必ず一致させること。**
# 食い違うと、その出口へ行くはずの会話だけが実行時に落ちる。topology のテストは
# 辺しか見ないので気づけない(実際、出口名を 5 つに増やしたとき build のテストは
# 全部緑のままだった)。tests/test_graph_build.py が 2 つの集合の一致を固定している。
ROUTE_TO_NODE: dict[str, str] = {
    "escalate": "complaint_reply",
    "fallback_script": "script_reply",
    "knowledge": "forced_rag",
    "refund_flow": "fetch_order",
    "business": "agent_llm",
}


def _builder() -> StateGraph:
    """compile 前の StateGraph を返す。compile 済みの graph からは形を変えられないため、
    checkpointer 違いで組み直したいときにここから作り直す。"""
    b = StateGraph(ConversationState)

    b.add_node("coref", nodes.coref)
    b.add_node("classify_intent", nodes.classify_intent)
    b.add_node("forced_rag", nodes.forced_rag)
    b.add_node("confidence_check", nodes.confidence_check)
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_node("retrieve_policy", nodes.retrieve_policy)
    b.add_node("agent_llm", nodes.agent_llm)
    b.add_node("agent_tools", nodes.agent_tools)
    b.add_node("complaint_reply", nodes.complaint_reply)
    b.add_node("script_reply", nodes.script_reply)
    b.add_node("fallback_reply", nodes.fallback_reply)
    b.add_node("log", nodes.log_node)

    b.add_edge(START, "coref")
    b.add_edge("coref", "classify_intent")
    # 9 intent → 5 出口。表は routing.INTENT_TO_ROUTE にあり、ここはその 5 つの
    # 行き先を辺として置くだけ
    b.add_conditional_edges("classify_intent", route_by_intent, ROUTE_TO_NODE)

    # knowledge は forced_rag を経由してからしか agent_llm へ行けない。
    # business と違い「検索するかどうか」を Agent に選ばせない
    b.add_edge("forced_rag", "confidence_check")
    b.add_conditional_edges("confidence_check", confidence_gate, {
        "strong": "agent_llm",
        "weak": "fallback_reply",
    })

    # 返金の決定的な subflow。注文を特定してから規約を引き、そのうえで Agent が
    # 返品可否を判断する。retrieve_policy を通らずに agent_llm へ入る辺は作らない
    # (規約を引かずに一般論で答えることを、経路の側で不可能にする)
    b.add_edge("fetch_order", "retrieve_policy")
    b.add_edge("retrieve_policy", "agent_llm")

    # ReAct loop。戻りの辺が無いと tool の結果をモデルへ返せない
    b.add_conditional_edges("agent_llm", should_continue, {
        "continue": "agent_tools",
        "stop": "log",
    })
    b.add_edge("agent_tools", "agent_llm")

    b.add_edge("complaint_reply", "log")
    b.add_edge("script_reply", "log")
    b.add_edge("fallback_reply", "log")
    b.add_edge("log", END)
    return b


def build_graph(checkpointer=None):
    """実行可能な graph を返す。checkpointer を渡すと turn をまたいで State を持ち回る。

    checkpointer を引数にするのは、本番(永続)とテスト(メモリ)で保存先が変わる一方、
    graph の形は同じであるため。既定の None は 1 turn 完結(履歴を残さない)。
    """
    return _builder().compile(checkpointer=checkpointer)
