"""graph の分岐関数。State を読んで「次にどの辺を通るか」を表す文字列を返す。

node と違って State を書き換えず、副作用も上流呼び出しも持たない。分岐の判断材料は
すべて手前の node が State へ書き終えている(intent は classify_intent、
evidence_strong は forced_rag、messages と steps は agent_llm)ので、
ここでは追加の判断をせず、書かれている値をそのまま読むだけにする。
"""

from langchain_core.messages import AIMessage

from app.config import settings

# 7 intent → 4 出口。spec §3.1 の固定 routing で、model には決めさせない。
# 「返金返品」を knowledge に入れているのは、返品・交換の可否や期限が
# ほぼ規約と FAQ の読み上げで、注文照会よりナレッジ検索の方が当たるため。
# 「アフターサービス」は逆に修理や交換の進捗という個別注文の話が主なので business。
INTENT_TO_ROUTE: dict[str, str] = {
    "商品相談": "knowledge",
    "返金返品": "knowledge",
    "配送": "business",
    "注文": "business",
    "アフターサービス": "business",
    "苦情": "complaint",
    "雑談": "chitchat",
}


def route_by_intent(state) -> str:
    """intent で routing する。knowledge | business | complaint | chitchat のいずれかを返す。

    未知の intent(prompt の変更、上流の schema 違反、7 分類にない文字列)は
    business へ倒す。business は Agent が tool を使って自分で判断する出口なので、
    分類を外しても Agent 側で拾い直せる。逆に chitchat へ倒すと固定文を返して
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
