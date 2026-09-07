"""graph の node。State の一部を dict で返し、reducer が既存の State へ畳み込む。

node は 2 種類に分かれる。

- **決定的な出口**(chitchat_reply / complaint_reply / fallback_reply): model を呼ばず
  固定文を返す。呼ばないことそのものが価値で、雑談のたびに上流を叩くなら固定文にする
  意味が無い。この 3 つは必ず終わるので、graph に「絶対に返答が返る経路」を作る。
- **上流を使う node**(classify_intent / forced_rag): 分類と検索を行う。どちらも
  上流が落ちても例外を投げない下位実装(app/core/intent.py、app/core/query_understanding.py、
  app/core/selfcheck.py)の上に乗せ、障害を graph 全体の停止に化けさせない。
- **ReAct loop の 2 node**(agent_llm / agent_tools): 両者を routing.should_continue が
  つないで loop になる。graph の中で唯一「次に何をするか」をモデルが決める場所。

State に無い key を返しても LangGraph が黙って捨てるため、返す key は
app/graph/state.py の ConversationState に宣言済みのものだけにすること。
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import settings
from app.core import intent as intent_mod
from app.core import query_understanding, retrieval, selfcheck
from app.core.llm import get_chat_model
from app.core.prompts import (
    AGENT_SYSTEM,
    CHITCHAT_REPLY_TEXT,
    COMPLAINT_REPLY_TEXT,
    FALLBACK_REPLY_TEXT,
)
from app.db import repository
from app.graph import routing
from app.tools.infra import execute_tool_call
from app.tools.registry import get_all_tools

logger = logging.getLogger(__name__)

CHITCHAT_REPLY = CHITCHAT_REPLY_TEXT
COMPLAINT_REPLY = COMPLAINT_REPLY_TEXT
FALLBACK_REPLY = FALLBACK_REPLY_TEXT

# search_knowledge へ「足切りなし」を伝える番兵(app/tools/business.py と同じ)。
# rerank スコアは 0〜1 なので 0.0 でも実質素通しだが、それは上流の値域に対する暗黙の仮定になる。
_UNGATED = float("-inf")


def _user_text(state) -> str:
    """最後の HumanMessage の本文。無ければ空文字。

    末尾から探すのは、checkpointer が turn をまたいで履歴を持ち回るため。
    「無ければ空文字」にするのは、messages[-1] を直に見ると HumanMessage が
    1 通も無い State(復元直後や tool 応答だけの State)で IndexError になり、
    node ではなく graph 全体が落ちるため。
    """
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


# ---------------------------------------------------------------------------
# 決定的な出口
# ---------------------------------------------------------------------------


async def chitchat_reply(state) -> dict:
    """雑談: 固定文を返す。model は呼ばない。"""
    return {"answer": CHITCHAT_REPLY, "trace": {"route": "chitchat"}}


async def complaint_reply(state) -> dict:
    """苦情: 共感・案内文 + 有人対応 / チケット作成の 2 択を返す。

    **backend は実行しない。** チケットを勝手に立てると、苦情の言葉が出るたびに
    運用側の対応待ち行列が伸びる。作るかどうかはユーザーが選ぶ話なので、ここでは
    そのまま create_ticket へ渡せる draft を用意して選択肢として提示するに留める。
    ticket_type は DB の ENUM と同じ英語の識別子(spec §6.1)。
    """
    actions = [
        {"type": "transfer_human"},
        {"type": "create_ticket",
         "draft": {"description": _user_text(state), "ticket_type": "complaint"}},
    ]
    return {"answer": COMPLAINT_REPLY, "suggested_actions": actions,
            "trace": {"route": "complaint"}}


async def fallback_reply(state) -> dict:
    """evidence が弱いときの出口。固定文を返し、質問を低信頼プールへ積む。

    答えられなかった質問こそナレッジベースの穴なので、断って終わりにせず
    low_confidence_questions へ残す(04 章の query_faq と同じ経路)。理由に載せる
    top スコアは、プールを人が見たときに「惜しかったのか、全く外れていたのか」を
    区別する唯一の数字になる。
    """
    # trace ごと無い経路(gate を通らずここへ倒れてきた場合)でも整形で落とさない
    top = state.get("trace", {}).get("evidence_top") or 0.0
    reason = f"retrieval evidence が不十分(top={top:.3f})"
    await repository.insert_low_confidence(
        state.get("conversation_id"), _user_text(state), "retrieval_low_conf", reason
    )
    return {"answer": FALLBACK_REPLY, "trace": {"route": "fallback"}}


# ---------------------------------------------------------------------------
# knowledge route(強制 retrieval)
# ---------------------------------------------------------------------------


# evidence が弱いと判定したときに State へ書く値。
# **空にすることが必要**で、単に「書かない」では足りない。checkpointer が State を
# turn をまたいで保持するため、前の turn で strong だった citations がそのまま残り、
# 拒否の返答の横に前回の出典が並ぶ(実際に起こりうる: 1 turn 目 knowledge/strong →
# 2 turn 目 knowledge/weak)。読む側それぞれに evidence_strong を見させるより、
# 書く側で 1 度だけ揃える方が取りこぼさない。
_NO_EVIDENCE = {"evidence_strong": False, "evidence": "", "citations": []}


async def coref(state) -> dict:
    """指示対象解決: 本章では最小実装としてそのまま透過する。正式版は 06 章。

    node の枠だけ先に置くのは、後から差し込むと graph の形が変わってしまうため。
    messages には手を触れない(書き換えた時点で素通しではなくなる)。
    """
    return {"trace": {"coref": "passthrough"}}


async def classify_intent(state) -> dict:
    """発話を 7 分類のいずれか 1 語に落とす。routing 表を引くのは後続の route_by_intent。

    intent.classify は上流が落ちても例外を投げず「雑談」へ倒すので、ここでは握らない。
    """
    intent = await intent_mod.classify(_user_text(state))
    # route も一緒に確定させて State へ書く。conditional edge(route_by_intent)は
    # 分岐先を返すだけで State を書けないので、ここで持たないと ConversationState の
    # route field を誰も埋めないまま残る(log node と trace が読めなくなる)。
    route = routing.route_by_intent({"intent": intent})
    return {"intent": intent, "route": route,
            "trace": {"intent": intent, "route": route}}


async def forced_rag(state) -> dict:
    """knowledge route の強制 retrieval。番号付き evidence と根拠の強さを State へ書く。

    business route と違い、Agent に「検索するかどうか」を選ばせない。ポリシーや仕様の
    質問はナレッジベースにしか正が無く、引かずに答えれば必ず作り話になるため。
    ゲートは query_faq(04 章)と同じ 2 段階で、リランクスコアの機械ゲートと
    セルフチェックの意味ゲートを通ったものだけを strong とする。
    """
    u = await query_understanding.understand(_user_text(state))
    query = u["standard"]
    # 同義語は検索テキストにだけ足す。標準質問の意味は変えない(セルフチェックと
    # 生成には query の方を使う)
    search_query = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    # 足切りは search_knowledge に任せず、ここで掛ける。理由は 2 つ:
    # ① 拒否理由に載せる「本当の top スコア」は足切り前にしか存在しない。向こう側で
    #    切ってもらうと hits が空で届き、fallback_reply が低信頼プールへ積む理由が
    #    必ず top=0.000 になって、「惜しかったのか全く外れていたのか」を人が区別できなくなる。
    # ② 閾値は「ユーザーへ答えるか断るか」という方針であって検索の性質ではない。
    #    断る主体であるこちら側に置く方が筋が通る。
    hits = await retrieval.search_knowledge(
        search_query, strategy="hybrid_rerank", min_score=_UNGATED
    )
    if not hits:
        return {**_NO_EVIDENCE,
                "trace": {"forced_rag": True, "evidence_top": 0.0}}

    # 機械ゲート。リランク上流が落ちた場合、search_knowledge は rerank_score なしの
    # ハイブリッド順で返す(app/core/retrieval.py)。掛ける数字が無いのでゲートは飛ばし、
    # 意味ゲートへ委ねる。ここで拒否に倒すと、上流の一時障害がそのまま回答拒否 +
    # 低信頼プールへの投入に化けてしまう。
    top = hits[0].get("rerank_score")
    trace = {"forced_rag": True, "evidence_top": top if top is not None else 0.0}
    if top is None:
        trace["rerank"] = "unavailable"  # evidence_top の 0.0 を実測値と読み違えないため
    else:
        hits = [h for h in hits if h.get("rerank_score", 0.0) >= settings.rerank_min_score]
        if not hits:
            return {**_NO_EVIDENCE, "trace": trace}

    # 意味ゲート: この根拠だけで答えきれるかをモデル自身に判定させる
    chk = await selfcheck.check_sufficient(
        query, [f"{h.get('question', '')} {h.get('answer', '')}" for h in hits]
    )
    if not chk["useful"]:
        return {**_NO_EVIDENCE, "trace": {**trace, "self_check": chk["reason"]}}

    # head/tail 配置の「後」に番号を振る。evidence 本文の [n] と citations の n が
    # 同じ chunk を指すのは、この順番が唯一の正であるため
    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h.get("id"), "section_path": h.get("section_path"),
         "question": h.get("question"), "answer": h.get("answer"),
         "content_type": h.get("content_type")}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"evidence_strong": True, "evidence": evidence, "citations": citations,
            "trace": trace}


async def confidence_check(state) -> dict:
    """生成前の evidence gate の実 node。判定を trace に残すだけ。

    実際の分岐は後続の conditional edge(app/graph/routing.py の confidence_gate)が
    evidence_strong を読んで行う。分岐関数は State を書けないため、
    「どちらに倒れたか」をログへ残す場所として node を 1 つ挟んでいる。
    """
    decision = "strong" if state.get("evidence_strong") else "weak"
    return {"trace": {"confidence": decision}}


# ---------------------------------------------------------------------------
# main Agent(ReAct loop)
# ---------------------------------------------------------------------------


# knowledge route で system の末尾に足す指示。evidence 本文はこの後ろへ連結する。
# 「query_faq を再度呼ぶな」と書くのは、forced_rag が既に引き終えた検索をモデルが
# もう一度やり直すため。同じ根拠を取り直すだけで 1 step と 1 回分の課金を捨てる上、
# 2 度目の検索結果は State の citations と番号がずれるので、本文の [n] が
# frontend の出典と食い違う。
_KNOWLEDGE_EVIDENCE_HINT = (
    "\n\n## retrieval 済みの knowledge evidence"
    "(この evidence に基づいて回答し、重要な結論の後に [1] のような source number を付けてください。"
    "evidence はすでに取得済みなので query_faq を再度呼ばないでください。"
    "必要であれば注文 / 配送など他の tool は呼び出せます。)\n"
)


def _agent_messages(state) -> list:
    """system(knowledge route では evidence を連結)+ turn をまたいだ履歴。

    evidence を system 側へ入れるのは、ToolMessage として差し込むと対応する tool_call が
    存在せず上流に弾かれるため。空文字を連結しないのは、forced_rag が weak のときに
    evidence="" を書くからで、見出しだけ付いた空の evidence は「根拠はあるが中身が無い」
    という誤った指示になる。
    """
    sys = AGENT_SYSTEM
    if state.get("route") == "knowledge" and state.get("evidence"):
        sys = AGENT_SYSTEM + _KNOWLEDGE_EVIDENCE_HINT + state["evidence"]
    return [SystemMessage(sys), *state.get("messages", [])]


async def agent_llm(state, config=None) -> dict:
    """ReAct の reasoning step。tool を bind した model を呼び、steps と token を加算する。

    config をそのまま ainvoke へ渡すのは、LangGraph が stream 用の callback を config に
    載せて node へ渡すため。落とすと token が frontend へ流れない。

    steps は「model を何回呼んだか」で、routing.should_continue の停止条件になる。
    tokens_used は trace と集計のためだけで、loop の制御には使わない。usage_metadata は
    上流によっては付かない(None)ので、無ければ 0 を足す。ここで落とすと、
    token を報告しない上流に繋いだ瞬間に graph 全体が止まる。
    """
    model = get_chat_model(streaming=True).bind_tools(get_all_tools())
    ai: AIMessage = await model.ainvoke(_agent_messages(state), config)
    used = (ai.usage_metadata or {}).get("total_tokens", 0)
    return {
        "messages": [ai],
        "steps": state.get("steps", 0) + 1,
        "tokens_used": state.get("tokens_used", 0) + used,
    }


async def agent_tools(state) -> dict:
    """ReAct の action step。**create_ticket だけは実行せず選択肢へ変換する。**

    complaint_reply と同じ理由で、チケットは勝手に立てない(モデルが呼ぶと決めた時点で
    作ってしまうと、ユーザーが望まないチケットで運用側の待ち行列が伸びる)。代わりに
    そのまま create_ticket へ渡せる draft を suggested_actions へ積み、モデルには
    「選択肢を提示した」と伝える合成 ToolMessage を返す。

    合成 ToolMessage の tool_call_id は元の tool_call と必ず一致させること。上流は
    tool_calls と ToolMessage の対応を検査しており、欠けても食い違っても 400 になる。
    """
    last = state["messages"][-1]
    tool_msgs = []
    actions = list(state.get("suggested_actions", []))
    for tc in last.tool_calls:
        if tc["name"] == "create_ticket":
            # ticket_type は DB の ENUM と同じ英語の識別子(spec §6.1)。
            # 既定を inquiry にするのは、種別を外すなら一般問い合わせが最も無害なため。
            draft = {"description": tc["args"].get("description", ""),
                     "ticket_type": tc["args"].get("ticket_type", "inquiry")}
            actions.append({"type": "create_ticket", "draft": draft})
            tool_msgs.append(ToolMessage(
                content="「チケット作成」の選択肢をユーザーへ提示しました。"
                        "1 文で簡潔に説明して終了し、これ以上 tool を呼ばないでください。",
                tool_call_id=tc["id"], name="create_ticket"))
        else:
            run = await execute_tool_call(tc, state.get("conversation_id", 0))
            tool_msgs.append(run.tool_message)
    out = {"messages": tool_msgs}
    # 1 件も無いときに書かないのは、suggested_actions に reducer が無く後勝ちの
    # 上書きになるため。空リストを返すと、前の step で積んだ選択肢が消える。
    if actions:
        out["suggested_actions"] = actions
    return out
