"""main Agent node(ReAct loop の agent_llm / agent_tools)と、その入力を組み立てる _agent_messages。

graph の中で唯一「次に何をするか」をモデルが決める場所なので、ここが壊れると
knowledge route の根拠が届かない・step が数えられず loop が止まらない・
ユーザーが望んでいないチケットが実際に作られる、といった形で表に出る。

上流のモデルも本物のツールもここでは呼ばない。既定では両方「呼ばれたら落ちる」に
差し替えてあり、必要なテストだけが明示的に偽物を置く(差し替え忘れたまま本物の
上流を叩いて課金する事故を、テストの書き方ではなく仕組みで止める)。
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.prompts import AGENT_SYSTEM
from app.graph import nodes
from app.tools.builtin import faq as faq_tool
from app.tools import registry
from app.tools.engine import ToolRun


def _ok_run(tc, content='{"ok": true}') -> ToolRun:
    return ToolRun(
        tool_call_id=tc["id"], name=tc["name"], ok=True, status="success",
        tool_message=ToolMessage(content=content, tool_call_id=tc["id"], name=tc["name"]),
    )


@pytest.fixture(autouse=True)
def _forbid_upstream_and_tools(monkeypatch):
    """モデル取得・ツール実行・MCP への接続を既定で禁止する。使うテストは自分で偽物を置く。

    08 章で node が毎ターン registry.get_all_specs() を呼ぶようになった。素のままだと
    単体テストが 127.0.0.1:8101/8102 を叩きに行く(繋がらなければ warning を出して
    skip するので黙って遅くなるだけ、という一番たちの悪い形になる)。ここで built-in
    だけを返す偽物へ差し替え、MCP のツールが要るテストは自分で足す。
    """

    def _boom_model(*args, **kwargs):
        raise AssertionError("テストが上流のチャットモデルを呼んだ")

    async def _boom_tool(*args, **kwargs):
        raise AssertionError("テストが本物のツールを実行した")

    async def _builtin_only():
        return registry.builtin_specs()

    monkeypatch.setattr(nodes, "get_chat_model", _boom_model)
    monkeypatch.setattr(nodes.engine, "execute_tool_call", _boom_tool)
    monkeypatch.setattr(nodes.registry, "get_all_specs", _builtin_only)


class _FakeModel:
    """bind_tools して ainvoke されるだけの最小の偽モデル。呼ばれた内容を記録する。"""

    def __init__(self, ai: AIMessage):
        self._ai = ai
        self.bound_tools = None
        self.seen_messages = None
        self.seen_config = "(未設定)"

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages, config=None):
        self.seen_messages = list(messages)
        self.seen_config = config
        return self._ai


def _use_model(monkeypatch, ai: AIMessage) -> _FakeModel:
    fake = _FakeModel(ai)
    seen = {}

    def _get(streaming=False, **kwargs):
        seen["streaming"] = streaming
        return fake

    monkeypatch.setattr(nodes, "get_chat_model", _get)
    fake.get_kwargs = seen
    return fake


def _use_tools(monkeypatch) -> list:
    """ツール実行を記録するだけの偽物に差し替え、記録先の list を返す。"""
    called = []

    async def _fake(tc, conversation_id, specs):
        called.append((tc, conversation_id, specs))
        return _ok_run(tc)

    monkeypatch.setattr(nodes.engine, "execute_tool_call", _fake)
    return called


# --- _agent_messages -----------------------------------------------------------


def test_knowledge_route_puts_the_evidence_in_the_turn_material():
    """knowledge route では forced_rag が引いた evidence が材料として届く。

    07 章で置き場所が変わった(system への連結 → 最後の HumanMessage の直後の
    SystemMessage)。**届くこと自体は変わらない。** system 本体へ連結すると
    ターンごとに prompt の先頭が変わり、prefix cache が毎回外れる。

    ToolMessage として差し込めないのは、対応する tool_call が存在せず上流に弾かれるため。
    """
    state = {"route": "knowledge", "evidence": "[1] 返品ポリシー: 7日以内は返品可能",
             "messages": [HumanMessage("返品できますか")]}
    msgs = nodes._agent_messages(state)

    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == AGENT_SYSTEM
    assert isinstance(msgs[-1], SystemMessage)
    assert "[1] 返品ポリシー: 7日以内は返品可能" in msgs[-1].content


def test_knowledge_route_tells_the_model_not_to_search_again():
    """引き直しを禁じる指示が材料に入ること。

    forced_rag が引き終えた検索をやり直されると、1 step と 1 回分の課金を捨てるだけでなく、
    2 度目の結果の番号が State の citations とずれ、本文の [n] と出典が食い違う。
    """
    state = {"route": "knowledge", "evidence": "[1] q: a", "messages": []}
    ctx = nodes._agent_messages(state)[-1].content
    assert "query_faq" in ctx
    assert "再度呼ばないでください" in ctx


def test_business_route_has_no_evidence_to_append():
    """business route では evidence が空なので材料そのものが付かない(検索していない)。

    06 章で連結の条件を「route が knowledge」から「evidence があれば」へ緩めた。
    強制検索は forced_rag(knowledge)と retrieve_policy(refund_flow)の 2 つに増え、
    route を条件にすると経路が増えるたびに書き足すことになるため。前 turn の evidence が
    business route へ漏れないことは、runtime が turn の入口で evidence / citations を
    空へ戻していること(app/graph/runtime.py の _graph_input)が担保する。
    """
    state = {"route": "business", "evidence": "",
             "messages": [HumanMessage("注文1001はどこ")]}
    msgs = nodes._agent_messages(state)
    assert msgs[0].content == AGENT_SYSTEM
    assert len(msgs) == 2 and isinstance(msgs[1], HumanMessage)


def test_empty_evidence_is_not_appended():
    """evidence が空文字なら連結しない。

    forced_rag は weak のとき evidence="" を書く(前 turn の出典を残さないため)。
    見出しだけ付いた空の evidence は「根拠はあるが中身が無い」という誤った指示になる。
    """
    state = {"route": "knowledge", "evidence": "", "messages": []}
    assert nodes._agent_messages(state) == [SystemMessage(AGENT_SYSTEM)]


def test_history_follows_the_system_message_in_order():
    """system の後ろに、turn をまたいだ履歴がそのままの順で並ぶこと。"""
    history = [HumanMessage("注文1001は?"), AIMessage("確認します"), HumanMessage("今どこ?")]
    msgs = nodes._agent_messages({"route": "business", "messages": history})
    assert msgs[1:] == history


# --- agent_llm -----------------------------------------------------------------


async def test_agent_llm_counts_a_step_and_accumulates_tokens(monkeypatch):
    ai = AIMessage(content="はい", usage_metadata={
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
    _use_model(monkeypatch, ai)

    out = await nodes.agent_llm({"messages": [HumanMessage("hi")], "steps": 2,
                                 "tokens_used": 500})
    assert out["messages"] == [ai]
    assert out["steps"] == 3
    assert out["tokens_used"] == 620


async def test_agent_llm_starts_from_zero_when_the_state_is_fresh(monkeypatch):
    _use_model(monkeypatch, AIMessage(content="はい"))
    out = await nodes.agent_llm({"messages": [HumanMessage("hi")]})
    assert out["steps"] == 1


async def test_agent_llm_survives_missing_usage_metadata(monkeypatch):
    """usage_metadata を付けない上流でも落ちないこと。

    token 量は trace と集計のためだけの数字で、loop の制御には使わない。
    ここで落とすと、token を報告しない上流に繋いだ瞬間に graph 全体が止まる。
    """
    ai = AIMessage(content="はい")
    assert ai.usage_metadata is None
    _use_model(monkeypatch, ai)

    out = await nodes.agent_llm({"messages": [HumanMessage("hi")], "tokens_used": 7})
    assert out["tokens_used"] == 7
    assert out["steps"] == 1


async def test_agent_llm_binds_every_tool_and_passes_the_config(monkeypatch):
    """built-in も MCP も区別なく bind し、config をそのまま ainvoke へ渡すこと。

    config を落とすと stream 用の callback が届かず、token が frontend へ流れない。
    MCP 側のツールも混ぜるのは、08 章で配送状況の照会が MCP へ移ったため。
    出所で選り分けていると、bind されずにモデルからは存在しないツールになる。
    """
    mcp_spec = registry.ToolSpec(
        name="query_logistics", description="配送状況",
        json_schema={"type": "object", "properties": {"tracking_no": {"type": "string"}}},
        tool=type("T", (), {"name": "query_logistics"})(),
        permission="read", source="mcp", mcp_server="logistics")

    async def _both():
        return [*registry.builtin_specs(), mcp_spec]

    monkeypatch.setattr(nodes.registry, "get_all_specs", _both)

    fake = _use_model(monkeypatch, AIMessage(content="はい"))
    cfg = {"callbacks": ["dummy"]}
    await nodes.agent_llm({"messages": [HumanMessage("hi")], "route": "business"}, cfg)

    builtin = {s.name for s in registry.builtin_specs()}
    assert {t.name for t in fake.bound_tools} == builtin | {"query_logistics"}
    assert fake.seen_config is cfg
    assert fake.get_kwargs["streaming"] is True


async def test_agent_llm_sends_the_evidence_bearing_messages(monkeypatch):
    """agent_llm が組み立てる入力が _agent_messages の結果であること。"""
    fake = _use_model(monkeypatch, AIMessage(content="はい"))
    await nodes.agent_llm({"route": "knowledge", "evidence": "[1] q: a",
                           "messages": [HumanMessage("返品は?")]})
    assert fake.seen_messages[0].content == AGENT_SYSTEM
    assert "[1] q: a" in fake.seen_messages[-1].content


# --- agent_tools ---------------------------------------------------------------


def _ai_with(*tool_calls) -> AIMessage:
    return AIMessage(content="", tool_calls=list(tool_calls))


async def test_agent_tools_executes_a_normal_tool(monkeypatch):
    called = _use_tools(monkeypatch)
    tc = {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
    out = await nodes.agent_tools({"messages": [_ai_with(tc)], "conversation_id": 42})

    assert [c[0]["name"] for c in called] == ["query_order"]
    assert called[0][1] == 42  # conversation_id が注入経路へ渡ること
    # 一覧を engine へ渡すのは node の仕事。空のまま渡すと、engine からは全部が
    # 「未知のツール」に見えて、原因の分からない失敗としてモデルへ返る。
    assert "query_order" in called[0][2]
    assert [m.tool_call_id for m in out["messages"]] == ["c1"]
    assert isinstance(out["messages"][0], ToolMessage)
    # 通常ツールだけなら選択肢は積まない
    assert "suggested_actions" not in out


async def test_agent_tools_defaults_conversation_id_when_the_state_lacks_one(monkeypatch):
    called = _use_tools(monkeypatch)
    await nodes.agent_tools(
        {"messages": [_ai_with({"name": "query_order", "args": {}, "id": "c1"})]}
    )
    assert called[0][1] == 0


# 08: create_ticket は「実行しない」から「確認を経てから実行する」へ変わった。
# 確認フロー(interrupt → resume)そのものは compiled graph が要るので
# tests/test_graph_08_confirm_ticket.py が見る。ここで見るのは、node を直に
# 呼べる範囲、つまり **確認を出さない側の分岐**と preview の組み立て。


async def test_an_incomplete_create_ticket_is_not_confirmed_but_sent_to_the_engine(monkeypatch):
    """引数が足りない create_ticket は確認カードを出さず、engine の検証ブロックへ回す。

    ここで確認カードを出してしまうと、ユーザーが「作成する」を押した後に engine が
    引数不足で弾き、押したのに何も起きない画面になる。足りないことはモデルに返して
    ユーザーへ聞き直させるのが正しい(推測で埋めさせない)。

    node を直に呼んでいるので、もし interrupt を発火すれば例外になる
    (interrupt() は compiled graph の中でしか動かない)。「確認を出していない」ことが
    テストの書き方ではなく仕組みで確かめられる。
    """
    called = _use_tools(monkeypatch)
    tc = {"name": "create_ticket", "args": {"ticket_type": "after_sales"}, "id": "c1"}
    out = await nodes.agent_tools({"messages": [_ai_with(tc)], "conversation_id": 42})

    assert [c[0]["name"] for c in called] == ["create_ticket"]
    assert [m.tool_call_id for m in out["messages"]] == ["c1"]
    # 08 でチケットは選択肢ではなくなった。確認カードは interrupt で出す
    assert "suggested_actions" not in out


async def test_a_normal_tool_still_runs_alongside_an_incomplete_create_ticket(monkeypatch):
    """1 step に通常ツールと create_ticket が並んでも、両方に ToolMessage が返ること。

    片方だけを処理して step を打ち切ると、tool_calls と ToolMessage の対応が揃わず、
    上流が次の呼び出しを 400 で弾いて loop がその場で止まる。
    """
    called = _use_tools(monkeypatch)
    order = {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
    ticket = {"name": "create_ticket", "args": {"description": "壊れていた"}, "id": "c2"}
    out = await nodes.agent_tools({"messages": [_ai_with(order, ticket)],
                                   "conversation_id": 3})

    assert [c[0]["name"] for c in called] == ["query_order", "create_ticket"]
    assert [m.tool_call_id for m in out["messages"]] == ["c1", "c2"]


def test_the_ticket_preview_carries_both_the_identifier_and_the_japanese_label():
    """preview は英語の識別子と日本語のラベルの両方を持つ。

    ticket_type は DB の ENUM と同じ英語の識別子(spec §6.1)で、画面に出すのは日本語。
    対応表を画面側へ置くと labels.py と 2 か所で同じ表を保つことになり、ENUM に値が
    増えたときに片方だけ古くなる。create_ticket tool が status / status_label を
    両方返すのと同じ形にする。
    """
    from app.core import labels

    preview = nodes._ticket_preview({"ticket_type": "after_sales", "description": "壊れていた"})
    assert preview == {"ticket_type": "after_sales",
                       "ticket_type_label": "アフターサービス",
                       "description": "壊れていた"}
    assert preview["ticket_type_label"] == labels.label(labels.TICKET_TYPE, "after_sales")


def test_the_ticket_preview_defaults_to_the_english_enum_identifier():
    """種別が無い場合の既定は DB の ENUM と同じ inquiry(日本語ではない)。

    日本語を入れると、engine の JSON Schema 検証(enum は英語の 3 語)で弾かれ、
    確認カードまで到達しない。
    """
    preview = nodes._ticket_preview({})
    assert preview["ticket_type"] == "inquiry"
    assert preview["ticket_type_label"] == "問い合わせ"
    assert preview["description"] == ""


@pytest.mark.parametrize(("decision", "expect"), [
    ({"confirmed": True}, True),
    ({"confirmed": False}, False),
    ({}, False),
    (None, False),
    ("はい", False),
    ({"confirmed": "true"}, False),
    (True, True),
])
def test_only_an_explicit_yes_counts_as_confirmation(decision, expect):
    """読めない戻り値は「作らない」側へ倒す。

    _normalize_order_id と同じく画面の戻り値は形を保証できないが、こちらは書き込みなので
    倒し先が逆になる。読めない値を確認済みと解釈すると、ユーザーが押していない
    チケットが実際に作られる。
    """
    assert nodes._is_confirmed(decision) is expect


async def test_existing_suggested_actions_are_kept(monkeypatch):
    """前の step や complaint_reply が積んだ選択肢を上書きで消さないこと。

    suggested_actions には reducer が無く後勝ちの上書きになるので、
    積み直しはこの node の責任になる。
    """
    tc = {"name": "submit_refund", "args": {"order_id": "1001"}, "id": "r1"}
    out = await nodes.agent_tools({"messages": [_ai_with(tc)],
                                   "suggested_actions": [{"type": "transfer_human"}]})
    assert [a["type"] for a in out["suggested_actions"]] == ["transfer_human", "refund_form"]


# ---------------------------------------------------------------------------
# business route で query_faq が断られたときの低信頼プール投入
#
# knowledge route は forced_rag → fallback_reply が担当するが、business route で
# モデルが自分から query_faq を呼んで断られた場合はそこを通らない。04 章では
# orchestration 側がこの投入を持っていたので、graph へ移す際に落とすと、
# プール(09 章のデータフライホイールの入口)が静かに取りこぼす。
# ---------------------------------------------------------------------------

def _faq_run(payload, name="query_faq", raw=None):
    """engine が返す形。**本文と生の戻り値を別々に持てるようにする。**

    検索の写しはモデルへ見せない(app/tools/builtin/faq.py の _strip_internal が
    本文から落とす)ので、本物の engine でも content には載らない。node が読むのは
    raw_result の方で、そこを取り違えると写しが黙って NULL のまま積まれる。
    """
    return ToolRun(tool_call_id="t1", name=name, ok=True, status="success",
                   tool_message=ToolMessage(content=json.dumps(payload, ensure_ascii=False),
                                            tool_call_id="t1", name=name),
                   raw_result=raw)


async def test_a_refused_faq_on_the_business_route_reaches_the_pool(monkeypatch):
    saved = {}

    async def _fake(cid, raw, source, reason, retrieved_chunks=None):
        saved.update(cid=cid, raw=raw, source=source, reason=reason,
                     chunks=retrieved_chunks)
        return 1

    async def _exec(tc, cid, specs):
        return _faq_run({"sufficient": False, "source": "retrieval_low_conf",
                         "reason": "リランクの最高スコアが閾値未満(top=0.012)"})

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", _fake)
    monkeypatch.setattr(nodes.engine, "execute_tool_call", _exec)
    ai = AIMessage("", tool_calls=[{"name": "query_faq", "args": {"keyword": "置き配"}, "id": "t1"}])
    await nodes.agent_tools({"messages": [HumanMessage("置き配できますか"), ai],
                             "conversation_id": 9})
    assert saved["cid"] == 9
    assert saved["raw"] == "置き配できますか"          # ユーザーの原文
    assert saved["source"] == "retrieval_low_conf"
    assert "0.012" in saved["reason"]                 # 本当の top スコアが残る


async def test_a_refused_faq_carries_the_retrieval_snapshot_into_the_pool(monkeypatch):
    """**この経路も検索とリランクを通っている。** 写しを付けずに積んではいけない。

    DDL はこの列の NULL を「検索を通っていない」の意味で使う(app/db/models.py)ので、
    付け忘れると `retrieved_chunks IS NULL` で数える側が誤分類し、査読画面は
    この行にだけ「検索を通っていないため写しはありません」と嘘の説明を出す。
    """
    saved = {}
    snapshot = [{"question": "置き配について", "answer": "対応していません",
                 "rerank_score": 0.12, "section_path": "配送/置き配"}]

    async def _fake(cid, raw, source, reason, retrieved_chunks=None):
        saved.update(chunks=retrieved_chunks)
        return 1

    async def _exec(tc, cid, specs):
        # 本文(モデルが読む側)には写しが無い。engine が format_result で落とすため
        return _faq_run(
            {"sufficient": False, "source": "retrieval_low_conf", "reason": "閾値未満"},
            raw={"sufficient": False, "source": "retrieval_low_conf",
                 "reason": "閾値未満", faq_tool.SNAPSHOT_KEY: snapshot},
        )

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", _fake)
    monkeypatch.setattr(nodes.engine, "execute_tool_call", _exec)
    ai = AIMessage("", tool_calls=[{"name": "query_faq", "args": {"keyword": "置き配"}, "id": "t1"}])
    await nodes.agent_tools({"messages": [HumanMessage("置き配できますか"), ai],
                             "conversation_id": 9})
    assert saved["chunks"] == snapshot


async def test_an_empty_snapshot_is_kept_as_an_empty_list(monkeypatch):
    """「検索は通ったが 0 件」を None へ丸めない。査読画面は 2 つを別の文言で出す。"""
    saved = {}

    async def _fake(cid, raw, source, reason, retrieved_chunks=None):
        saved.update(chunks=retrieved_chunks)
        return 1

    async def _exec(tc, cid, specs):
        return _faq_run(
            {"sufficient": False, "source": "retrieval_low_conf", "reason": "0 件"},
            raw={"sufficient": False, faq_tool.SNAPSHOT_KEY: []},
        )

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", _fake)
    monkeypatch.setattr(nodes.engine, "execute_tool_call", _exec)
    ai = AIMessage("", tool_calls=[{"name": "query_faq", "args": {"keyword": "x"}, "id": "t1"}])
    await nodes.agent_tools({"messages": [HumanMessage("x"), ai], "conversation_id": 9})
    assert saved["chunks"] == [], "0 件の写しが「検索を通っていない」と同じ扱いになっている"


async def test_a_missing_snapshot_stays_none(monkeypatch):
    """生の戻り値が読めなければ None のまま積む(写しは投入の条件ではない)。"""
    saved = {}

    async def _fake(cid, raw, source, reason, retrieved_chunks=None):
        saved.update(chunks=retrieved_chunks)
        return 1

    async def _exec(tc, cid, specs):
        return _faq_run({"sufficient": False, "source": "self_check", "reason": "不足"})

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", _fake)
    monkeypatch.setattr(nodes.engine, "execute_tool_call", _exec)
    ai = AIMessage("", tool_calls=[{"name": "query_faq", "args": {"keyword": "x"}, "id": "t1"}])
    await nodes.agent_tools({"messages": [HumanMessage("x"), ai], "conversation_id": 9})
    assert saved["chunks"] is None


async def test_a_sufficient_faq_does_not_reach_the_pool(monkeypatch):
    async def _boom(*a, **k):
        raise AssertionError("答えられたターンをプールへ積んではいけない")

    async def _exec(tc, cid, specs):
        return _faq_run({"sufficient": True, "evidence": "[1] ...", "citations": []})

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", _boom)
    monkeypatch.setattr(nodes.engine, "execute_tool_call", _exec)
    ai = AIMessage("", tool_calls=[{"name": "query_faq", "args": {"keyword": "送料"}, "id": "t1"}])
    await nodes.agent_tools({"messages": [HumanMessage("送料は"), ai], "conversation_id": 9})


async def test_a_broken_faq_payload_does_not_break_the_turn(monkeypatch):
    """ツールが失敗すると content は日本語のエラー文になる。読めなければ静かに諦める。"""
    async def _boom(*a, **k):
        raise AssertionError("読めない payload でプールへ積んではいけない")

    async def _exec(tc, cid, specs):
        return ToolRun(tool_call_id="t1", name="query_faq", ok=False, status="failed",
                       tool_message=ToolMessage(content="ツールの実行に失敗しました",
                                                tool_call_id="t1", name="query_faq"))

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", _boom)
    monkeypatch.setattr(nodes.engine, "execute_tool_call", _exec)
    ai = AIMessage("", tool_calls=[{"name": "query_faq", "args": {"keyword": "x"}, "id": "t1"}])
    out = await nodes.agent_tools({"messages": [HumanMessage("x"), ai], "conversation_id": 9})
    assert out["messages"]        # ターン自体は続く


async def test_other_tools_are_not_pooled(monkeypatch):
    async def _boom(*a, **k):
        raise AssertionError("query_faq 以外を積んではいけない")

    async def _exec(tc, cid, specs):
        return _faq_run({"sufficient": False}, name="query_order")

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", _boom)
    monkeypatch.setattr(nodes.engine, "execute_tool_call", _exec)
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "t1"}])
    await nodes.agent_tools({"messages": [HumanMessage("x"), ai], "conversation_id": 9})


# --- _faq_payload の解析（04 章の _faq_result から移植）------------------------


def test_faq_payload_reads_the_tool_result():
    assert nodes._faq_payload(_faq_run({"sufficient": True, "evidence": "[1] x"}))["sufficient"] is True


def test_faq_payload_is_none_for_other_tools():
    assert nodes._faq_payload(_faq_run({"x": 1}, name="query_order")) is None


def test_faq_payload_is_none_when_the_tool_errored():
    """ツールが失敗すると content は JSON ではなく日本語のエラー文になる。
    ここで例外を漏らすとターン全体が落ちる。"""
    run = ToolRun(tool_call_id="t1", name="query_faq", ok=False, status="failed",
                  tool_message=ToolMessage(content="ツール実行失敗: 一時的なエラー",
                                           tool_call_id="t1", name="query_faq"))
    assert nodes._faq_payload(run) is None


def test_faq_payload_is_none_when_the_content_is_not_an_object():
    """json.loads は数値や文字列も通す。dict でなければ扱わない。"""
    run = ToolRun(tool_call_id="t1", name="query_faq", ok=True, status="success",
                  tool_message=ToolMessage(content="123", tool_call_id="t1", name="query_faq"))
    assert nodes._faq_payload(run) is None


async def test_only_the_refused_faq_is_pooled_when_several_tools_run(monkeypatch):
    """1 step に複数の tool が並んでも、断られた query_faq だけを積む。"""
    saved = []

    async def _fake(cid, raw, source, reason, retrieved_chunks=None):
        saved.append(source)
        return 1

    async def _exec(tc, cid, specs):
        if tc["name"] == "query_faq":
            return _faq_run({"sufficient": False, "source": "self_check", "reason": "根拠不足"})
        return _faq_run({"order_id": "1001"}, name="query_order")

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", _fake)
    monkeypatch.setattr(nodes.engine, "execute_tool_call", _exec)
    ai = AIMessage("", tool_calls=[
        {"name": "query_order", "args": {}, "id": "t1"},
        {"name": "query_faq", "args": {"keyword": "x"}, "id": "t2"}])
    await nodes.agent_tools({"messages": [HumanMessage("x"), ai], "conversation_id": 9})
    assert saved == ["self_check"]
