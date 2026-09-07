"""graph の node。State の一部を dict で返し、reducer が既存の State へ畳み込む。

node は 2 種類に分かれる。

- **決定的な出口**(script_reply / complaint_reply / fallback_reply): model を呼ばず
  固定文を返す。呼ばないことそのものが価値で、雑談のたびに上流を叩くなら固定文にする
  意味が無い。この 3 つは必ず終わるので、graph に「絶対に返答が返る経路」を作る。
- **上流を使う node**(classify_intent / forced_rag): 分類と検索を行う。どちらも
  上流が落ちても例外を投げない下位実装(app/core/intent.py、app/core/query_understanding.py、
  app/core/selfcheck.py)の上に乗せ、障害を graph 全体の停止に化けさせない。
- **ReAct loop の 2 node**(agent_llm / agent_tools): 両者を routing.should_continue が
  つないで loop になる。graph の中で唯一「次に何をするか」をモデルが決める場所。
- **終端の node**(log_node): 4 つの出口がすべてここへ合流し、trace を残して
  assistant message を保存する。出口ごとに書くと、出口が増えたときに
  書き忘れた経路だけ履歴に残らない。

State に無い key を返しても LangGraph が黙って捨てるため、返す key は
app/graph/state.py の ConversationState に宣言済みのものだけにすること。
"""

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from app.config import settings
from app.core import coref as coref_mod
from app.core import intent as intent_mod
from app.core import query_understanding, retrieval, selfcheck
from app.core.llm import get_chat_model
from app.core.prompts import (
    AGENT_SYSTEM,
    COMPLAINT_REPLY_TEXT,
    FALLBACK_REPLY_TEXT,
    REFUND_JUDGE_HINT,
    SCRIPT_REPLY_CHITCHAT,
    SCRIPT_REPLY_OTHER,
)
from app.db import repository
from app.graph import routing
from app.tools.business import list_user_orders, order_snapshot
from app.tools.infra import execute_tool_call
from app.tools.registry import get_all_tools

logger = logging.getLogger(__name__)

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


def _history_text(state, max_turns: int = 6) -> str:
    """直近の会話を短いテキストにする。**今回の発話は含めない**。

    含めると、分類器が「いまの発話」と「履歴」を区別できず、同じ文が 2 回出る。

    本文が str でない message(block の list)は読み飛ばす。文脈の補助でしかない
    ここで整形に失敗して node ごと落とすのは割に合わない。
    """
    msgs = state.get("messages", [])
    prior = msgs[:-1] if msgs else []
    lines = []
    for m in prior[-max_turns:]:
        role = "ユーザー" if isinstance(m, HumanMessage) else "サポート"
        text = m.content if isinstance(m.content, str) else ""
        if text:
            lines.append(f"{role}:{text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 決定的な出口
# ---------------------------------------------------------------------------


async def script_reply(state) -> dict:
    """雑談 / その他の出口。model を呼ばず、intent で文面を出し分ける。

    雑談は買い物の話題へ戻す案内、その他は何を知りたいのかを具体的に書いてもらう依頼。

    **雑談だと分かっているときだけ**挨拶側の文面を出す。買い物の話題へ案内し直す文は
    「雑談である」ことが前提なので、分類が壊れて別の intent が届いた場合に出すと
    的外れになる。ここへ来るのは routing.INTENT_TO_ROUTE の 雑談 / その他 の 2 つだけで、
    それ以外は分類側の異常なので、用件を聞き直す方へ倒す(どちらに転んでも会話は進む)。

    evidence が弱いときの fallback_reply とは別の出口。あちらは検索して答えられなかった
    結果で、こちらは routing 直後に確定する。低信頼プールへも積まない(答えるべき質問が
    そもそも無いので、ナレッジベースの穴として数えると穴の数が水増しされる)。
    """
    text = SCRIPT_REPLY_CHITCHAT if state.get("intent") == "雑談" else SCRIPT_REPLY_OTHER
    return {"answer": text, "trace": {"route": "fallback_script"}}


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
    """指示対象を解決し、文脈なしで通じる質問を resolved_query に書く。

    **messages には手を触れない。** 書き換えた発話を履歴へ混ぜると、ユーザーが実際に
    打った文が残らなくなり、次のターンの書き下しがこちらの推測を土台にしてしまう。
    書き下しは resolved_query に置き、原文は messages に残す。

    trace の coref は 05 章から同じ key。素通しか書き下したかを残しておかないと、
    分類を外したときに「書き下しが余計だったのか、そもそも効かなかったのか」を
    後から切り分けられない。
    """
    original = _user_text(state)
    resolved = await coref_mod.resolve(original, _history_text(state))
    # 末尾の句読点だけの差は「書き下した」と数えない。モデルは素通しのつもりでも
    # 「。」を足してくることがあり(実測: 素通しすべき 9 件のうち 4 件)、そのままだと
    # trace が rewrite だらけになって、本当に書き下した回を見分けられなくなる。
    changed = resolved.rstrip("。.?？!！ ") != original.rstrip("。.?？!！ ")
    return {"resolved_query": resolved,
            "trace": {"coref": "rewrite" if changed else "passthrough"}}


async def classify_intent(state) -> dict:
    """発話を 8 分類のいずれか 1 語 + confidence に落とす。routing 表を引くのは後続の route_by_intent。

    分類するのは resolved_query(coref が指示対象を解決して書き下した完全な質問)。
    無ければ元の発話へ倒す。「それいくらだった?」のような発話は、書き下す前に分類すると
    どの分類にも寄らず、確信度だけが下がる。

    intent.classify は上流が落ちても例外を投げず「その他」へ倒すので、ここでは握らない。

    **trace の key は intent_confidence にすること。** confidence は confidence_check が
    strong / weak を入れる key で、trace の reducer は同じ key を後勝ちで上書きするため、
    ぶつけると knowledge route の trace からどちらかが黙って消える。
    """
    query = state.get("resolved_query") or _user_text(state)
    r = await intent_mod.classify(query, _history_text(state))
    intent, conf = r["intent"], r["confidence"]
    # route も一緒に確定させて State へ書く。conditional edge(route_by_intent)は
    # 分岐先を返すだけで State を書けないので、ここで持たないと ConversationState の
    # route field を誰も埋めないまま残る(log node と trace が読めなくなる)。
    route = routing.route_by_intent({"intent": intent})
    return {"intent": intent, "intent_confidence": conf, "route": route,
            "trace": {"intent": intent, "intent_confidence": conf, "route": route}}


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
# 返金返品 / アフターサービス(決定的な subflow)
# ---------------------------------------------------------------------------


# 注文番号として拾う数字の桁数の下限。app/tools/business.py の _ORDER_ID_RANGE
# (4 桁で採番)に合わせてある。**下げてはいけない。**「3日以内に返品したい」の 3 や
# 「2回目です」の 2 を注文番号と読むと、存在しない注文の中身を引いて、ユーザーが
# 一度も言っていない商品について返品可否を答えることになる。
_ORDER_ID_MIN_DIGITS = 4

# 数字の直後がこの文字なら数量・日付であって注文番号ではない。桁数だけでは
# 「2026年に買った」「1500円の商品」を弾けないため。「番」は逆に注文番号の
# 目印なので、ここには入れない。
_COUNTER_SUFFIX = "年月日円個回件点名分秒時週台冊枚本人%％"

# 日本語入力からは全角数字がそのまま出てくる。半角へ寄せてから数字を探す
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

# 注文番号の目印が付いた書き方(「注文1001」「注文番号: 1001」「1001番」)。
# 目印のない数字より先に見るのは、1 文に数字が複数あるとき
# (「3日以内に注文1001を返品したい」)に目印の付いた方を選ぶため。
_ORDER_MARKED = re.compile(
    rf"(?:注文番号|注文|オーダー|order)\s*(?:番号|no\.?|#|は|[:：]|の)?\s*"
    rf"(\d{{{_ORDER_ID_MIN_DIGITS},}})"
    rf"|(\d{{{_ORDER_ID_MIN_DIGITS},}})\s*番",
    re.IGNORECASE,
)
_ORDER_BARE = re.compile(rf"\d{{{_ORDER_ID_MIN_DIGITS},}}")


def _extract_order_id(text: str) -> str | None:
    """質問文から注文番号を取り出す。見つからなければ None。

    **見つからないことを None で正直に返すのが仕事。** ここで無理に数字を拾うと、
    fetch_order がユーザーに選ばせる経路へ行かなくなり、結局「番号を推測して進む」
    という 06 章が避けたい形になる。
    """
    if not text:
        return None
    s = text.translate(_FULLWIDTH_DIGITS)
    for pattern in (_ORDER_MARKED, _ORDER_BARE):
        for m in pattern.finditer(s):
            group = next((i for i, g in enumerate(m.groups(), start=1) if g), None)
            digits = m.group(0) if group is None else m.group(group)
            end = m.end() if group is None else m.end(group)
            tail = s[end:end + 1]
            # tail が空文字のときに in で判定すると常に True になるので先に弾く
            if tail and tail in _COUNTER_SUFFIX:
                continue
            return digits
    return None


def _normalize_order_id(picked) -> str | None:
    """interrupt() の戻り値を注文番号の文字列に揃える。揃えられなければ None。

    画面が何を返してくるかはこちら側では決めきれない。一覧の要素をそのまま返して
    {"order_id": "1001"} で来ることも、JSON の数値として 1001 で来ることもある。
    どれで来ても同じ注文に落ちるようにする。

    **桁数の下限はここでは掛けない。** 一覧に出した番号はこちらが採番したものなので、
    ユーザーが選んだ値を桁数で弾く理由が無い。下限が要るのは、番号かどうか分からない
    自由入力を相手にする _extract_order_id の方だけ。
    """
    if isinstance(picked, dict):
        picked = picked.get("order_id")
    # bool は int の subclass。True を "1" にすると存在しない注文を引いてしまう
    if isinstance(picked, bool):
        return None
    if isinstance(picked, int):
        return str(picked)
    if not isinstance(picked, str):
        return None
    s = picked.translate(_FULLWIDTH_DIGITS).strip()
    if s.isdigit():
        return s
    # 「注文1001」のように文で返してくる画面もありうるので、文からの抽出も試す
    return _extract_order_id(s)


async def fetch_order(state) -> dict:
    """返金フローの最初の段。対象の注文を特定する。

    見つからないときは model に推測させず、interrupt で画面に一覧を出して選ばせる。

    **interrupt() より前に副作用を置かないこと。** Command(resume=v) で再開すると、
    この node は先頭から実行し直され、interrupt() の呼び出しがそこで v を返す
    (Task 1 の実測: node の足跡が
     ['ask_order:enter', 'ask_order:enter', 'ask_order:resumed(1001)', 'confirm'] と
     なり、node に 2 回入っている)。したがって interrupt より手前に DB 書き込みや
    上流呼び出しを置くと、ユーザーが 1 回選んだだけで 2 回起きる。手前に置いてよいのは
    2 回走っても同じ結果になる純粋な処理だけ。

    trace の source は、注文をどうやって特定したかを後から切り分けるためのもの。
    state = 既に State にあった / query = 発話から拾った / selected = ユーザーが選んだ /
    unresolved = 画面の戻り値が読めなかった。
    """
    order_id = state.get("order_id")
    source = "state"
    if not order_id:
        order_id = _extract_order_id(state.get("resolved_query") or _user_text(state))
        source = "query"
    if not order_id:
        # ここから先は model に選ばせない。一覧を画面へ出してユーザーの選択を待つ。
        # list_user_orders は純粋な関数なので、再実行で 2 回呼ばれても害が無い。
        order_id = _normalize_order_id(interrupt({
            "type": "select_order",
            "orders": list_user_orders(state.get("user_id") or ""),
        }))
        source = "selected"
    if not order_id:
        # 画面の戻り値を解釈できなかった場合。もう一度 interrupt すると、壊れた値が
        # 返り続ける限り画面と往復し続けて会話が進まない。注文なしで先へ進め、
        # 後段の会話で聞き直す(「注文をでっち上げない」という要件は満たしたまま)。
        logger.warning("fetch_order: 選択された注文番号を解釈できなかった")
        return {"trace": {"fetch_order": {"order_id": None, "source": "unresolved"}}}
    return {
        "order_id": order_id,
        "order_data": order_snapshot(order_id),
        "trace": {"fetch_order": {"order_id": order_id, "source": source}},
    }


def _rerank_score(hit: dict) -> float:
    """並べ替えと重複除去に使うスコア。無ければ 0.0。

    search_knowledge は rerank 上流が落ちると、rerank_score を付けずに
    ハイブリッド検索の並びのまま返す(app/core/retrieval.py)。
    h["rerank_score"] の直接添字は KeyError になり、node ではなく graph 全体を止める。

    **欠けている hit は 0.0 扱いにして末尾へ寄せる。** リランクの効いた hit
    (0〜1)の方が関連度を実測できているので、そちらを先に見せる。上流が
    完全に落ちて全件が欠けた場合は全部が同じ値になり、sorted が安定なので
    検索が返した順(ハイブリッドの並び)がそのまま残る。
    """
    score = hit.get("rerank_score")
    return 0.0 if score is None else score


# 返金フローで Agent へ渡す規約の上限。3 クエリぶんを重複除去しても
# 最大 30 chunk になりうるので、ここで絞る。
_POLICY_TOP_K = 8


async def retrieve_policy(state) -> dict:
    """返金フローの規約検索。1 つの質問を複数の観点へ展開してから引く。

    fetch_order と同じく、Agent に「検索するかどうか」を選ばせない。返品可否の根拠は
    規約にしか無く、引かずに答えれば一般論で断定することになるため。

    forced_rag と違ってゲートは掛けない。knowledge route には回答拒否の出口
    (fallback_reply)があるが、返金フローは fetch_order で注文が特定できていれば
    会話を続けられるので、引けなかったことを理由に止める方が失うものが大きい。
    判断材料が無いことは空の evidence として Agent へ伝わる。
    """
    query = state.get("resolved_query") or _user_text(state)
    queries = await query_understanding.expand_queries(query)

    # chunk ごとに「最もスコアの高かった hit」だけを持つ。3 つのクエリは観点違いなので
    # 同じ規約を引きやすく、まとめずに並べると同じ条項が 3 回出て Agent の入力を埋める。
    merged: dict = {}
    for q in queries:
        # **min_score は必ず明示する。** 省略すると search_knowledge が内部で
        # rerank_min_score の足切りをしてから返し、並べ替えと重複除去に使う
        # 「本当のスコア」が失われる(forced_rag / query_faq と同じ理由)。
        hits = await retrieval.search_knowledge(
            q, strategy="hybrid_rerank", min_score=_UNGATED
        )
        for h in hits:
            # id が欠けた hit は question/answer で区別する。id で揃えて None に潰すと、
            # 別の条項同士が同じ chunk と見なされて 1 件に消える。
            key = h.get("id")
            if key is None:
                key = (h.get("question"), h.get("answer"))
            cur = merged.get(key)
            if cur is None or _rerank_score(h) > _rerank_score(cur):
                merged[key] = h

    # 上位だけを残す。3 クエリ × rerank_top_k(既定 10)で、重複を除いても最大 30 chunk が
    # そのまま system prompt に入る。規約は 1 件が長いので、入れるほど良くなるどころか
    # 判断に効く条項が埋もれ、token も無駄に増える。順位の高いものだけを渡す。
    ranked = sorted(merged.values(), key=_rerank_score, reverse=True)[:_POLICY_TOP_K]
    # head/tail 配置の「後」に番号を振る。evidence 本文の [n] と citations の n が
    # 同じ chunk を指すのは、この順番が唯一の正であるため(forced_rag と同じ規律)
    arranged = retrieval.arrange_head_tail(ranked)
    citations = [
        {"n": i + 1, "id": h.get("id"), "section_path": h.get("section_path"),
         "question": h.get("question"), "answer": h.get("answer"),
         "content_type": h.get("content_type")}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    # 1 件も引けなくても、**空の evidence / citations を明示的に書いて** 先へ進める。
    # 書かないと checkpointer が前の turn の citations を持ち越し、根拠なしの判断の横に
    # 前回の出典が並ぶ(_NO_EVIDENCE と同じ理由)。
    return {"evidence": evidence, "citations": citations,
            "trace": {"retrieve_policy": {"queries": queries, "hits": len(ranked)}}}


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
    """system(evidence があれば連結)+ turn をまたいだ履歴。

    evidence を system 側へ入れるのは、ToolMessage として差し込むと対応する tool_call が
    存在せず上流に弾かれるため。空文字を連結しないのは、forced_rag / retrieve_policy が
    引けなかったときに evidence="" を書くからで、見出しだけ付いた空の evidence は
    「根拠はあるが中身が無い」という誤った指示になる。

    **route で絞らず「evidence があれば連結」にしている。** 強制検索は knowledge route の
    forced_rag と refund_flow の retrieve_policy の 2 つがあり、どちらも同じ形で
    evidence / citations を書く。route を条件にすると、経路が増えるたびにここを
    書き足すことになり、書き忘れた経路だけ根拠が黙って届かなくなる。

    refund_flow ではさらに、対象の注文と「可否だけを判断する」指示を足す。注文の中身を
    渡さないと、規約だけを読んで一般論で答えてしまう。
    """
    sys = AGENT_SYSTEM
    if state.get("evidence"):
        sys = sys + _KNOWLEDGE_EVIDENCE_HINT + state["evidence"]
    if state.get("route") == "refund_flow":
        # ensure_ascii=False にするのは、日本語を unicode escape にすると読ませる
        # 文字数が数倍になり、注文の中身が人にもモデルにも読めなくなるため
        sys = sys + REFUND_JUDGE_HINT + json.dumps(state.get("order_data") or {},
                                                   ensure_ascii=False)
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


def _faq_payload(run) -> dict | None:
    """query_faq の戻り値を dict として取り出す。query_faq 以外や壊れていれば None。

    ツールが失敗したときの content は JSON ではなく日本語のエラー文になる。ここは
    「低信頼プールへ積むか」を決めるだけの補助なので、読めないものは静かに None に
    して、ターン本体を巻き込まない。
    """
    if getattr(run, "name", None) != "query_faq":
        return None
    try:
        parsed = json.loads(run.tool_message.content)
    except (ValueError, TypeError):
        return None
    # json.loads は数値や文字列も通す。dict でなければ後段の .get で落ちる
    return parsed if isinstance(parsed, dict) else None


async def _record_faq_refusal(state, run) -> None:
    """query_faq が根拠不足で断ったターンを低信頼プールへ積む。

    knowledge route は forced_rag → fallback_reply が担当するが、**business route で
    モデルが自分から query_faq を呼んで断られた場合はそちらを通らない**。04 章では
    orchestration 側がこの投入を持っていたので、graph へ移した際に落とすと、
    プール(09 章のデータフライホイールの入口)が静かに取りこぼす。

    source は DDL の ENUM に合わせる。query_faq が付けなかった場合に self_check へ
    倒すのは、ENUM に無い値を書いて DB エラーにする方が害が大きいため。
    """
    faq = _faq_payload(run)
    if not faq or faq.get("sufficient") is not False:
        return
    cid = state.get("conversation_id")
    if not cid:
        return
    await repository.insert_low_confidence(
        cid, _user_text(state), faq.get("source") or "self_check", faq.get("reason")
    )


async def agent_tools(state) -> dict:
    """ReAct の action step。**create_ticket と submit_refund は実行せず選択肢へ変換する。**

    complaint_reply と同じ理由で、チケットも返金申請も勝手には作らない(モデルが呼ぶと
    決めた時点で作ってしまうと、ユーザーが望まないものが運用側の待ち行列へ積み上がる)。
    代わりにそのまま渡せる draft を suggested_actions へ積み、モデルには
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
        elif tc["name"] == "submit_refund":
            # 注文番号は State の値で補う。モデルは文脈から番号を落とすことがあり、
            # 空の draft を画面へ出すと、ユーザーには何の申請フォームか分からない。
            order_id = tc["args"].get("order_id") or state.get("order_id") or ""
            if order_id:
                actions.append({"type": "refund_form",
                                "draft": {"order_id": order_id,
                                          "reason": tc["args"].get("reason")}})
            else:
                # 注文が特定できていないのに申請フォームを出さない。submit_refund は
                # 全 route で bind されているので、注文を確かめていない business route
                # からも呼ばれうる。空の draft を出すと、ユーザーには何の申請か分からない
                # フォームが出て、押しても弾かれる。選択肢は出さず、モデルには
                # 「まず注文を確かめろ」と返して聞き直させる。
                logger.info("注文が特定できていないため返金フォームを提示しない conv=%s",
                            state.get("conversation_id"))
            # 提示していないのに「提示しました」と返すとモデルが嘘の案内を書く。
            # 実際に起きたことをそのまま返す。
            content = (
                "「返金・返品を申請する」の選択肢をユーザーへ提示しました。"
                "申請はまだ送信されていません。この注文が対象になる理由を"
                "規約の番号を引いて 1 文で説明し、画面から申請できることを"
                "案内して終了してください。これ以上 tool を呼ばないでください。"
                if order_id else
                "対象の注文が特定できていないため、申請の選択肢は提示していません。"
                "注文番号を尋ねてから、もう一度判断してください。"
            )
            tool_msgs.append(ToolMessage(content=content,
                                         tool_call_id=tc["id"], name="submit_refund"))
        else:
            run = await execute_tool_call(tc, state.get("conversation_id", 0))
            tool_msgs.append(run.tool_message)
            await _record_faq_refusal(state, run)
    out = {"messages": tool_msgs}
    # 1 件も無いときに書かないのは、suggested_actions に reducer が無く後勝ちの
    # 上書きになるため。空リストを返すと、前の step で積んだ選択肢が消える。
    if actions:
        out["suggested_actions"] = actions
    return out


# ---------------------------------------------------------------------------
# 終端(observability と履歴の保存)
# ---------------------------------------------------------------------------


def _message_text(m: AIMessage) -> str:
    """AIMessage の本文を文字列にする。content は str か block の list のどちらでもありうる。

    list のまま repository へ渡すと、保存の瞬間ではなく DB の型で落ちる。
    text を持たないブロック(思考ブロックなど)や dict でない要素は読み飛ばす。
    """
    content = m.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def resolve_answer(state) -> str:
    """最終的な回答文を 1 つに決める。経路によって回答の置き場所が違うため。

    決定的な node(script / complaint / fallback)は state["answer"] に固定文を書く。
    Agent は書かない(token を stream して frontend へ直接流すため、State へ溜めると
    「stream した本文」と「State の本文」の 2 つの正が生まれる)ので、末尾から
    AIMessage を辿る。answer を先に見るのは、checkpointer が履歴を turn をまたいで
    保持しており、末尾を先に見ると前 turn の回答を今回の回答として保存してしまうため。

    本文が空の AIMessage は飛ばして探し続ける。ReAct loop では末尾の AIMessage が
    「本文なし + tool_calls あり」になる瞬間があり、should_continue が steps 上限で
    打ち切るとその State のまま log へ来る。末尾だけを見ると空文字になり、
    ユーザーには何か返したのに履歴には何も残らない turn ができる。

    最後まで見つからなければ空文字。例外にしないのは、回答の保存に失敗しただけの turn を
    graph 全体の停止に化けさせないため。
    """
    if state.get("answer"):
        return state["answer"]
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            text = _message_text(m)
            if text:
                return text
    return ""


async def log_node(state) -> dict:
    """4 つの出口が合流する終端。trace を残し、assistant message を 1 件保存する。

    全経路をここへ集めるのは、保存とログを各出口に散らすと、出口が増えたときに
    書き忘れた経路だけ履歴に残らないから。State は書き換えない。

    conversation_id が無ければ保存しない。0 は「まだ採番されていない」であって
    会話 1 番ではないので、書けば無関係な会話の履歴が汚れる。
    回答が空のときは空文字ではなく NULL で残す。空文字で保存すると、履歴を
    読み直す側に中身の無い assistant 行が毎 turn 混ざる。
    """
    logger.info(
        "05 turn conv=%s intent=%s route=%s trace=%s",
        state.get("conversation_id"), state.get("intent"), state.get("route"),
        state.get("trace", {}),
    )
    answer = resolve_answer(state)
    if state.get("conversation_id"):
        await repository.append_message(state["conversation_id"], "assistant",
                                        content=answer or None)
    return {}
