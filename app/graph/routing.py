"""graph の分岐関数。State を読んで「次にどの辺を通るか」を表す文字列を返す。

node と違って State を書き換えず、副作用も上流呼び出しも持たない。分岐の判断材料は
すべて手前の node が State へ書き終えている(intent は classify_intent、
evidence_strong は forced_rag、messages と steps は agent_llm)ので、
ここでは追加の判断をせず、書かれている値をそのまま読むだけにする。
"""

from langchain_core.messages import AIMessage

from app.config import settings

# 9 intent → 5 出口。spec §3.1 の固定 routing で、model には決めさせない。
#
# 05 からの変更点は 2 つ。
#  - 「返金返品」と「アフターサービス」を refund_flow へ寄せた。どちらも
#    「まず対象の注文を特定し、規約を引いてから可否を判断する」という同じ順序で、
#    その順序を Agent の判断に任せると注文を確かめずに規約だけで答えてしまう。
#    決定的な subflow に固定する。
#  - 「その他」を追加し、雑談と同じ fallback_script へ送る。分類に迷ったものを
#    無理にどこかの業務経路へ入れるより、聞き直す方が害が小さい。
#
# 08 で「人工対応」を足し、business へ送る。escalate(苦情の 2 択)へ送らないのは、
# あちらが model を呼ばない固定文の出口で、チケットの中身をユーザー自身に
# 書かせるフォームしか出せないため。business なら main Agent が会話から内容を
# まとめ、agent_tools の確認カードで中身を見せてから作れる。
INTENT_TO_ROUTE: dict[str, str] = {
    "苦情": "escalate",
    "人工対応": "business",
    "雑談": "fallback_script",
    "その他": "fallback_script",
    "商品相談": "knowledge",
    "返金返品": "refund_flow",
    "アフターサービス": "refund_flow",
    "配送": "business",
    "注文": "business",
}


def route_by_intent(state) -> str:
    """intent で routing する。escalate | fallback_script | knowledge | refund_flow | business のいずれかを返す。

    未知の intent(prompt の変更、上流の schema 違反、9 分類にない文字列)は
    business へ倒す。business は Agent が tool を使って自分で判断する出口なので、
    分類を外しても Agent 側で拾い直せる。逆に fallback_script へ倒すと固定文を返して
    そこで会話が終わってしまい、取り返しがきかない。**迷ったら手数の多い方**。
    """
    return INTENT_TO_ROUTE.get(state.get("intent", ""), "business")


def confidence_gate(state) -> str:
    """knowledge route の生成前 evidence gate。strong は通し、weak は fallback へ倒す。

    evidence_strong を書くのは forced_rag(rerank スコアの機械ゲート + セルフチェック)で、
    ここはその結果を読むだけ。key が無い場合も weak 扱いにする。gate を書き忘れた node が
    上流にいたときに、根拠なしのまま生成へ進むより、答えられない旨を返す方が安全。
    """
    return "strong" if state.get("evidence_strong") else "weak"


def should_continue(state) -> str:
    """ReAct loop の停止条件。"continue"(agent_tools へ)か "stop"(loop を抜ける)を返す。

    判断は 3 つ。
    1. messages が空 → stop。直前の AIMessage が存在しない以上 tool_calls も無く、
       回すべき tool が無い。ここを見ないと state["messages"][-1] が IndexError になり、
       loop どころか graph 全体が落ちる。
    2. 直前の AIMessage に tool_calls が無い → stop。model が tool を要求せずに
       本文を返した = 自分で答えを出した、が ReAct の収束条件。
    3. tool_calls はあるが steps が settings.max_agent_steps に達した → stop。
       同じ tool を無限に呼び続ける暴走を、ここで強制的に打ち切る。

    **step 数で止めて token では止めない。** 1 step = model 呼び出し 1 回なので、
    上限は「最大 6 回まで考えさせる」とそのまま読め、ログを見た人が何が起きたかを
    数えて確かめられる。token 量は同じ 1 step でも履歴の長さで何倍にも振れるため、
    「何回考えたか」の代わりにはならない。token の上限はコストの話であって
    loop の制御の話ではないので、State の tokens_used は trace と 09 章の集計に回し、
    ここでは一切見ない(tests/test_graph_routing.py で固定してある)。
    """
    messages = state.get("messages") or []
    if not messages:
        return "stop"
    last = messages[-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if not has_tool_calls:
        return "stop"
    if state.get("steps", 0) >= settings.max_agent_steps:
        return "stop"
    return "continue"
