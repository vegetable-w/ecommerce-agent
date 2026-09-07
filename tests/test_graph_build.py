"""graph の組み立て(app/graph/build.py)。

この章で graph を使う理由は、**hard constraint を経路として固定する**ことにある。
「苦情は Agent に入らない」「ポリシーの質問は必ず retrieval を通る」は prompt の
お願いではなく通れる辺が無いという構造で担保しており、辺のつなぎ方が崩れた瞬間に
その保証は消える。

なので、ここでは node が存在するかではなく**経路**を確かめる。node の存在だけを
見るテストは、苦情が Agent に繋がっていても knowledge が retrieval を飛ばしていても
通ってしまい、守るべきものを 1 つも守れない。

上流のモデルは呼ばない。compile と topology の確認だけで、ainvoke / astream は
実行しない(それは 05 章の後続タスクの担当)。
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START

from app.graph.build import build_graph

# langgraph 1.2.11 の compiled.get_graph() は langchain_core.runnables.graph.Graph を返す。
# nodes は {id: Node} の dict、edges は Edge(source, target, data, conditional) の list で、
# conditional edge の data には add_conditional_edges の path_map の**キー**(分岐関数の
# 戻り値)が入る。属性名は version で変わりうるので、ここを唯一の前提として明示しておく。
#
# **落とし穴**: path_map の複数のキーが同じ node を指すと、辺は行き先ごとに 1 本へ
# まとめられ、data には片方のキーしか残らない(もう一方のラベルは消える)。つまり
# ラベルで引いた結果が空集合なのは「その分岐が存在しない」だけでなく
# 「別の分岐と行き先が同じになった」場合もありうる。ラベルで辺を引くときは
# _branch_target のように「ちょうど 1 本あること」を要求しないと、経路が壊れた瞬間に
# 空集合が返ってテストが静かに通ってしまう。
NODE_NAMES = {
    "coref", "classify_intent", "forced_rag", "confidence_check",
    "agent_llm", "agent_tools",
    "complaint_reply", "chitchat_reply", "fallback_reply", "log",
}

# 4 つの出口。ここから END までの間に必ず log が挟まる
EXITS = {"agent_llm", "complaint_reply", "chitchat_reply", "fallback_reply"}


def _graph():
    return build_graph().get_graph()


def _targets(g, source, *, label=None) -> set[str]:
    """source から出る辺の行き先。label を渡すとその分岐(path_map のキー)だけに絞る。"""
    return {e.target for e in g.edges
            if e.source == source and (label is None or e.data == label)}


def _branch_target(g, source: str, label: str) -> str:
    """source の label 分岐の行き先。ちょうど 1 本無ければその場で落とす。

    空集合を黙って返すと、行き先が別の分岐と同じになってラベルが消えた場合(上の
    落とし穴)に、後続の到達可能性の assert が「行き先が無いので何処にも行けない」で
    通ってしまう。壊れ方が「テストが通る」に化けるのを防ぐ。

    intent の 5 出口は行き先が重なりうるので、そちらは _route_target を使うこと。
    """
    targets = _targets(g, source, label=label)
    assert len(targets) == 1, f"{source} の {label} 分岐が 1 本ではない: {targets}"
    return next(iter(targets))


def _route_target(g, route: str) -> str:
    """intent の出口名 route の行き先 node。

    **辺のラベルからは引かない。** LangGraph は複数のラベルが同じ node を指すと辺を
    1 本にまとめ、`data` には片方のラベルしか残さない(実測: knowledge と refund_flow を
    同じ node に向けたら knowledge のラベルが消えた)。ラベルで引くと空集合が返り、
    到達可能性の assert が素通りする。

    そこで宣言側(build.ROUTE_TO_NODE)を正とし、graph 側では「その node が実在し、
    classify_intent から実際に辿れること」だけを確かめる。
    """
    from app.graph.build import ROUTE_TO_NODE

    assert route in ROUTE_TO_NODE, f"{route} は routing の出口ではない"
    target = ROUTE_TO_NODE[route]
    assert target in set(g.nodes), f"{route} の行き先 {target} が graph に無い"
    assert target in _targets(g, "classify_intent"), f"classify_intent から {target} への辺が無い"
    return target


def _reachable(g, start: str, *, cut: frozenset[str] = frozenset()) -> set[str]:
    """start から辿り着ける node の集合。cut に挙げた node は通れないものとして扱う。

    「必ず X を通る」は「X を塞ぐと辿り着けない」と言い換えられる。辺の有無を 1 本ずつ
    数えるより、迂回路が 1 本でも生えたら落ちる形になる。
    """
    seen: set[str] = set()
    stack = [start]
    while stack:
        n = stack.pop()
        if n in seen or n in cut:
            continue
        seen.add(n)
        stack.extend(e.target for e in g.edges if e.source == n)
    return seen


# --- node と入口 ----------------------------------------------------------------


def test_every_node_is_registered():
    g = _graph()
    assert set(g.nodes) == NODE_NAMES | {START, END}


def test_the_graph_starts_at_coref_and_classifies_before_routing():
    """入口は coref → classify_intent。分岐はその後にしか無い。

    分類の前に routing すると、route_by_intent が読む intent を誰も書いていない。
    """
    g = _graph()
    assert _targets(g, START) == {"coref"}
    assert _targets(g, "coref") == {"classify_intent"}


def test_every_node_is_reachable_from_start():
    """辿り着けない node が無いこと(繋ぎ忘れた node をここで見つける)。"""
    assert _reachable(_graph(), START) >= NODE_NAMES


# --- hard constraint: 苦情は Agent に入らない(spec D5)---------------------------


def test_complaint_is_routed_to_the_fixed_reply():
    assert _route_target(_graph(), "escalate") == "complaint_reply"


def test_complaint_never_reaches_the_agent():
    """苦情の分岐から先に、Agent へ辿り着ける経路が 1 本も無いこと。

    苦情に対してモデルに tool を選ばせると、謝るべき場面で注文照会やチケット作成が走る。
    ここは固定文と選択肢の提示で必ず終わる。行き先を名前で決め打ちせず分岐から辿るのは、
    苦情の行き先そのものが差し替えられたときに落とすため。
    """
    g = _graph()
    reach = _reachable(g, _route_target(g, "escalate"))
    assert "agent_llm" not in reach
    assert "agent_tools" not in reach
    # 苦情の出口から先は log を通って終わるだけ
    assert reach == {"complaint_reply", "log", END}


def test_chitchat_never_reaches_the_agent():
    """雑談も同じ。固定文を返すことそのものが目的なので、上流へ落ちる経路を作らない。"""
    g = _graph()
    assert _route_target(g, "fallback_script") == "chitchat_reply"
    assert _reachable(g, "chitchat_reply") == {"chitchat_reply", "log", END}


# --- hard constraint: knowledge は必ず retrieval を通る ---------------------------


def test_knowledge_enters_the_forced_retrieval():
    assert _route_target(_graph(), "knowledge") == "forced_rag"


def test_knowledge_cannot_reach_the_agent_without_the_forced_retrieval():
    """forced_rag を塞ぐと、knowledge の経路から Agent へ辿り着けないこと。

    ポリシーや仕様の質問はナレッジベースにしか正が無く、引かずに答えれば必ず作り話になる。
    「検索するかどうか」を Agent に選ばせないことがこの route の存在理由なので、
    retrieval を迂回する辺が 1 本でも生えたら落ちる形にしてある。
    """
    g = _graph()
    entry = _route_target(g, "knowledge")
    assert "agent_llm" not in _reachable(g, entry, cut=frozenset({"forced_rag"}))


def test_knowledge_passes_the_evidence_gate_before_generating():
    """forced_rag → confidence_check → gate の順であること。

    gate を通さず生成へ行くと、根拠が弱いまま答えてしまう。
    """
    g = _graph()
    assert _targets(g, "forced_rag") == {"confidence_check"}
    assert _branch_target(g, "confidence_check", "strong") == "agent_llm"
    assert _branch_target(g, "confidence_check", "weak") == "fallback_reply"
    # gate を塞ぐと生成へ行けない = 迂回路が無い
    assert "agent_llm" not in _reachable(g, "forced_rag",
                                         cut=frozenset({"confidence_check"}))


# --- business は retrieval を通らない --------------------------------------------


def test_business_goes_straight_to_the_agent():
    """business は Agent が tool を使って自分で判断する出口。強制 retrieval は挟まない。

    注文や配送の答えは backend にあってナレッジベースには無いので、
    ここで検索を強制しても当たらない検索に 1 往復ぶんの時間と課金を使うだけになる。
    """
    g = _graph()
    entry = _route_target(g, "business")
    assert entry == "agent_llm"
    reach = _reachable(g, entry)
    assert "forced_rag" not in reach
    assert "confidence_check" not in reach


# --- ReAct loop -----------------------------------------------------------------


def test_the_react_loop_is_closed():
    """agent_llm ⇄ agent_tools が閉じていること。

    戻りの辺が無いと、tool の結果をモデルへ返せないまま 1 回で終わる。
    """
    g = _graph()
    assert _branch_target(g, "agent_llm", "continue") == "agent_tools"
    assert _targets(g, "agent_tools") == {"agent_llm"}


def test_the_agent_stops_into_the_log_node():
    assert _branch_target(_graph(), "agent_llm", "stop") == "log"


# --- 出口は必ず log を通って END へ ------------------------------------------------


def test_every_exit_reaches_the_end():
    g = _graph()
    for exit_node in EXITS:
        assert END in _reachable(g, exit_node), exit_node


def test_no_exit_can_skip_the_log_node():
    """log を塞ぐと、どの出口からも END へ辿り着けないこと。

    保存とログを飛ばす経路ができると、その経路の turn だけ履歴に残らない。
    ユーザーには回答が返っているので、次の turn の履歴から静かに消える形で表に出る。
    """
    g = _graph()
    for exit_node in EXITS:
        assert END not in _reachable(g, exit_node, cut=frozenset({"log"})), exit_node


def test_the_log_node_is_the_only_way_into_the_end():
    g = _graph()
    assert {e.source for e in g.edges if e.target == END} == {"log"}
    assert _targets(g, "log") == {END}


# --- compile --------------------------------------------------------------------


def test_it_compiles_with_a_checkpointer():
    """checkpointer 付きで compile できること(turn をまたいだ State の持ち回り)。"""
    compiled = build_graph(checkpointer=InMemorySaver())
    assert compiled.checkpointer is not None
    assert set(compiled.get_graph().nodes) == NODE_NAMES | {START, END}


def test_it_compiles_without_a_checkpointer():
    assert build_graph().checkpointer in (None, False)


def test_every_route_the_router_can_return_has_a_node():
    """分岐関数の戻り値と、conditional edge のキーが一致していること。

    ここが食い違うと、その出口へ行く会話だけが実行時に落ちる。**辺を見るテストでは
    捕まらない**(出口名を 4 つから 5 つへ増やしたとき、build のテストは全部緑のまま
    だった)。routing と build は別ファイルなので、片方だけ直す事故が起きやすい。
    """
    from app.graph.build import ROUTE_TO_NODE
    from app.graph.routing import INTENT_TO_ROUTE

    assert set(INTENT_TO_ROUTE.values()) == set(ROUTE_TO_NODE)


def test_every_mapped_node_exists_in_the_graph():
    """行き先の node が実在すること。名前を打ち間違えても compile は通ってしまう。"""
    from app.graph.build import ROUTE_TO_NODE

    names = set(_graph().nodes)
    for route, node in ROUTE_TO_NODE.items():
        assert node in names, f"{route} の行き先 {node} が graph に無い"
