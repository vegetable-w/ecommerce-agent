"""graph を FastAPI から使えるようにする層。checkpointer の寿命と実行の入口。

graph そのものは app/graph/build.py で組み上がっている。ここが受け持つのは、

- **checkpointer を 1 つだけ開いて使い回すこと**。会話の State は turn をまたいで
  sqlite に残る(spec D1: business DB は MySQL、checkpointer は sqlite と役割を分ける)。
  接続をリクエストごとに開くと、同時に走る turn の数だけ sqlite のハンドルが増え、
  閉じ忘れがそのままファイルロックとして残る。起動時に 1 つ開き、終了時に閉じる。
- **graph の出力を frontend が読める event に写像すること**。/api/agent(非ストリーミング)は
  最終 State を、/api/chat(SSE)は event の列を使う。
- **中断した turn を再開できるようにすること**(06 章)。fetch_order は注文が特定
  できないと interrupt で止まり、08 章の agent_tools はチケットを作る前に確認で止まる。
  止まったことを表に出す(run_turn / stream_turn)のと、ユーザーが選んだ / 押した値で
  続きから走らせる(resume_turn / stream_resume)のがここの仕事。中断の種類は
  payload の "type" だけが決めるので、この層は種類ごとの分岐を持たない。

写像の要点は「**何を流さないか**」にある。stream_mode="messages" は graph の中で
起きた model 呼び出しの token を**すべて**運んでくるので、素通しすると
classify_intent の分類結果(「配送」の 2 文字)や forced_rag のクエリ書き換えが、
回答本文と同じ delta としてユーザーの画面に出る。node 名で絞るのはそのため。
"""

import logging
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.config import settings
from app.core import summarizer
from app.core.observability import attach_observability
from app.db import repository
from app.graph import state as state_mod
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

# 回答の token を frontend へ流す node。ここに無い node の model token は流さない。
# 「回答を書いている node はどれか」を白リストで持つのは、除外リストにすると
# model を呼ぶ node が増えるたびに書き足さねばならず、書き忘れが即漏洩になるため。
ANSWER_NODES = {"agent_llm"}
# 決定的 node は model を呼ばないので token が流れてこない。回答は state["answer"] に
# 入るので、updates から 1 塊の delta として拾う
DETERMINISTIC_ANSWER_NODES = {"script_reply", "complaint_reply", "fallback_reply"}
# 出典を書く node。**強制検索は 2 つある**(knowledge route の forced_rag と
# refund_flow の retrieve_policy)。片方だけを見ると、その経路の回答に付いた [n] が
# 画面でクリックできない注釈のまま残る。node が増えたらここへ足す。
CITATION_NODES = {"forced_rag", "retrieve_policy"}
# updates の chunk では node 名の位置にこの key が現れる(Task 1 の実測)。
# 値は node の差分 dict ではなく Interrupt の並びなので、node と同じ扱いをしない。
_INTERRUPT_KEY = "__interrupt__"

_graph = None
# AsyncSqliteSaver.from_conn_string() が返す async context manager。
# これを捨てると GC が接続ごと回収しにいくので、閉じるまで参照を持ち続ける
_cm = None


class ConversationNotFound(Exception):
    """指定された conversation_id が DB に無い。API 層が 404 へ変換する。"""


async def init_graph() -> None:
    """checkpointer を開き、graph を組んで module 変数へ置く。FastAPI の lifespan から呼ぶ。

    **2 回目以降は何もしない。** lifespan が二重に走る状況(ASGI の起動を 2 度踏む構成や
    テストの入れ子)で開き直すと、閉じずに _cm を上書きすれば sqlite のハンドルが漏れ、
    閉じてから開き直せば実行中のリクエストが握っている graph の下から接続を引き抜く。
    どちらも黙って壊れる方の失敗なので、「初期化済みなら成功として返す」を選ぶ。
    開き直したいときは close_graph() を先に呼ぶ(テストはそうしている)。
    """
    global _graph, _cm
    if _graph is not None:
        logger.warning("init_graph が二重に呼ばれた。既存の checkpointer を使い続ける")
        return
    _cm = AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path)
    checkpointer = await _cm.__aenter__()
    # setup() はテーブルを作る。冪等だが、呼ばないと初回の書き込みで落ちる
    await checkpointer.setup()
    # 09: compile 済みの graph に observability の callback を 1 回だけ被せる。
    # Langfuse が未設定なら attach_observability は graph をそのまま返すので、
    # ここ以外に分岐は要らない。
    _graph = attach_observability(build_graph(checkpointer=checkpointer))
    logger.info("05 graph compiled, checkpointer=%s", settings.checkpointer_db_path)


async def close_graph() -> None:
    """checkpointer を閉じ、module 変数を戻す。lifespan の後始末から呼ぶ。

    **先に None へ戻してから閉じる。** __aexit__ が落ちたときに参照を残すと、次の
    init_graph が「初期化済み」と見なして閉じかけの接続を使い続ける。例外そのものは
    握り潰さない(閉じられなかったことは呼び出し側に知らせる)。
    """
    global _graph, _cm
    cm, _cm, _graph = _cm, None, None
    if cm is not None:
        await cm.__aexit__(None, None, None)


def get_graph():
    """初期化済みの graph。未初期化なら例外。

    None を返すと呼び出し側の AttributeError まで進んでしまい、本当の原因
    (lifespan で init_graph を呼んでいない)が読み取れなくなる。
    """
    if _graph is None:
        raise RuntimeError(
            "graph が初期化されていません。FastAPI lifespan で init_graph() を呼ぶ必要があります"
        )
    return _graph


async def _require_conversation(conversation_id: int) -> int:
    """指定された会話が存在することを確かめる。無ければ ConversationNotFound。

    存在しない ID を素通しすると、誰も読まない会話へ user message を書き込み、
    graph の thread_id もその番号で切られる。ここで止める。
    """
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    return conversation_id


async def _ensure_conversation(user_id: str,
                               conversation_id: int | None) -> tuple[int, str, int]:
    """(会話 ID, 要約, 要約が覆う範囲)を返す。None なら採番、指定があれば存在を確かめる。

    **要約の 2 つは会話を読んだこの 1 回で一緒に取る。** 存在確認と要約の取得で
    get_conversation を 2 回叩くと、ターンの入口の DB 往復が毎回 1 つ増える。
    採番したての会話には要約が無いので ("", 0) を返す。
    """
    if conversation_id is None:
        return await repository.create_conversation(user_id), "", 0
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        raise ConversationNotFound(conversation_id)
    return conversation_id, conv.summary or "", conv.summary_upto_msg_id or 0


def _graph_input(user_id: str, message: str, cid: int, msg_id: int,
                 summary: str, summary_upto: int) -> dict:
    """graph へ渡す 1 turn 分の入力。

    messages は今回の発話 1 通だけでよい。過去の履歴は checkpointer が thread_id ごとに
    持っており、add_messages が追記する。steps と tokens_used を 0 で入れ直すのは、
    この 2 つに reducer が無く(後勝ちの上書き)、turn ごとに戻さないと前 turn の
    step 数を引き継いだまま should_continue の上限に当たるため。

    07: 発話に MySQL の message id を `db-{id}` の形で付ける。スライディングウィンドウは
    この anchor だけを見て要約の境界と突き合わせる(本文や並び順で境界を推測すると、
    同じ文面が何度も出てくる会話で静かに 1 ターンずれる)。
    """
    return {
        "messages": [HumanMessage(message, id=f"db-{msg_id}")],
        "user_id": user_id,
        "conversation_id": cid,
        # --- turn ごとに戻す出力チャネル ---
        # checkpointer は State を丸ごと持ち越すので、前の turn が書いた値は
        # 明示的に戻さない限り残る。実測した実害:
        #   1 turn 目に苦情 → answer に共感文、suggested_actions に 2 つの選択肢
        #   2 turn 目に配送の質問 → どの node も answer を書かないので、
        #   resolve_answer が 1 turn 目の共感文を返し、苦情のボタンも出たまま
        "steps": 0,
        "tokens_used": 0,
        "answer": "",
        "suggested_actions": [],
        "evidence": "",
        "citations": [],
        "evidence_strong": False,
        "intent": "",
        "route": "",
        "resolved_query": "",
        "intent_confidence": 0.0,
        "order_id": "",
        "order_data": {},
        # 要約もここで入れ直す。会話が進めば要約は更新されるので、前 turn の値が
        # 残ると、更新された後も古い要約を読み続ける
        "summary": summary,
        "summary_upto_msg_id": summary_upto,
        # trace は reducer 付きなので空 dict では消えない。目印を付けて作り直す
        "trace": {state_mod.TRACE_RESET: True},
    }


def _config(cid: int) -> dict:
    """thread_id は会話 ID。checkpointer はこの値で State を切り分ける。

    09: metadata の langfuse_session_id も同じ会話 ID。Langfuse の CallbackHandler は
    LangChain の metadata から `langfuse_` 接頭辞の key を拾って trace 根へ引き上げるので、
    これだけで 1 会話の複数 turn が Langfuse の 1 session にまとまる。**Langfuse を
    使っていなければただの未使用 metadata**で、graph の挙動には何も影響しない。

    turn の入口 3 つ(run_turn / resume_turn / _stream_events)はどれもこの関数から
    config を得ているので、session の紐付けもここ 1 箇所で済む。
    """
    return {
        "configurable": {"thread_id": str(cid)},
        "metadata": {"langfuse_session_id": str(cid)},
    }


def _interrupt_payload(raw) -> dict | None:
    """__interrupt__ の中身から、画面へ渡す payload の dict を取り出す。読めなければ None。

    容れ物の型は入口によって違う(Task 1 の実測): ainvoke の戻り値では **list**、
    astream の updates では **tuple**。片方だけを見る実装は他方で黙って素通しし、
    「中断しているのに画面が待ち続ける」形で壊れるので、並びとしてだけ扱う。

    要素は langgraph.types.Interrupt で、payload は .value にある。ただしその中身は
    node が渡した任意の値であり、こちら側では形を保証できない。dict でないものは
    「読めなかった」として None を返す(例外にすると turn ごと止まる)。
    """
    if isinstance(raw, (list, tuple)):
        for item in raw:
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                return value
    return None


def _interrupt_event(raw, cid: int) -> dict:
    """updates に出た __interrupt__ を frontend 向けの event dict にする。

    **payload が読めなくても event は出す。** graph は確かに止まっており、この後
    done は来ない。何も出さないと画面は無言のまま待ち続けるので、空でも key の
    揃った event を渡して「中断した」ことだけは伝える。

    kind は payload の "type" から取る。ここを "select_order" 固定にすると、
    今後増える別種の中断がすべて「注文を選ぶ」画面として描かれる。

    conversation_id を載せるのは done を出さないため。初回ターンで中断されたとき、
    画面はここでしか会話 ID を知る手立てがなく、/api/actions/resume を叩けない。
    """
    payload = _interrupt_payload(raw)
    if payload is None:
        logger.warning("interrupt の payload を解釈できなかった conv=%s", cid)
        payload = {}
    kind = payload.get("type")
    orders = payload.get("orders")
    ev = {
        "type": "interrupt",
        "kind": kind if isinstance(kind, str) else "",
        # 06 章の互換。select_order の画面はこの key を必ず読むので、payload に
        # 無くても空の list を置く(画面側に「key があるか」の分岐を持たせない)。
        "orders": orders if isinstance(orders, list) else [],
        "conversation_id": cid,
    }
    # 08 章: 中断の種類ごとの追加 payload。confirm_ticket はチケットの下書きを
    # preview に載せる。**種類が増えても event の組み立てを枝分かれさせない**ため、
    # 読める形のものだけをそのまま通す(kind で分岐すると、新しい中断を足すたびに
    # ここへ if を書き足すことになり、書き忘れた種類だけ中身が届かない)。
    preview = payload.get("preview")
    if isinstance(preview, dict):
        ev["preview"] = preview
    return ev


async def run_turn(user_id: str, message: str, conversation_id: int | None) -> dict:
    """非ストリーミングの入口。user message を保存し、graph を走らせて最終 State を返す。

    assistant message の保存は log node が行う(4 つの出口がすべてそこへ合流する)ので、
    ここでは user 側だけを書く。

    戻り値の interrupt は、graph が中断したときの payload(fetch_order なら
    {"type": "select_order", "orders": [...]})。中断していなければ None。
    中断した turn は答えが無いので、呼び出し側はこれを見て画面の分岐を決める。

    07: turn の**後**に要約の起動を試す。前でやると、いま答えるために要る履歴を
    圧縮しながら返答を組み立てることになる。maybe_schedule_summary は件数を見て
    background の task を立てるだけですぐ返るので、応答は待たされない。
    """
    cid, summary, upto = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    final = await get_graph().ainvoke(
        _graph_input(user_id, message, cid, msg_id, summary, upto), _config(cid))
    await summarizer.maybe_schedule_summary(cid)
    return {
        "conversation_id": cid,
        "state": final,
        "interrupt": _interrupt_payload(final.get(_INTERRUPT_KEY)),
    }


async def resume_turn(conversation_id: int, resume_value) -> dict:
    """中断した turn を、ユーザーが選んだ値で再開する。戻り値は run_turn と同じ形。

    run_turn と違うのは 2 点で、どちらも「resume は新しい発話ではない」ことから来る。

    - **user message を保存しない。** 保存すると同じ turn の user 行が 2 つ並び、
      log node が書く assistant 行との対応が崩れる。履歴は毎ターン prompt へ
      積み直されるので、以後すべてのターンにその重複が乗り続ける。
    - **_graph_input を通さない。** あれは turn の入口で出力チャネルを 0 に戻す
      入力で、interrupt 待ちの State に被せると再開前の途中経過が消える。
      渡すのは Command(resume=...) だけで、State は checkpointer が持っている。

    user 発話を保存しないので anchor も作らない。要約の起動だけは run_turn と同じく
    行う(再開の後にも assistant の発話が 1 件積まれ、履歴は伸びる)。
    """
    cid = await _require_conversation(conversation_id)
    final = await get_graph().ainvoke(Command(resume=resume_value), _config(cid))
    await summarizer.maybe_schedule_summary(cid)
    return {
        "conversation_id": cid,
        "state": final,
        "interrupt": _interrupt_payload(final.get(_INTERRUPT_KEY)),
    }


async def _stream_events(graph_input, cid: int) -> AsyncIterator[dict]:
    """graph の出力を frontend 向けの event dict へ写像する。turn の開始と再開で共通。

    出す event は 6 種類:
        {"type": "tool", "name": str}            agent_tools が実行した tool
        {"type": "citations", "items": list}     強制検索が引いた出典
        {"type": "delta", "text": str}           回答本文(agent_llm の token / 決定的 node の固定文)
        {"type": "actions", "items": list}       選択肢(有人対応 / チケット作成)
        {"type": "interrupt", ...}               ユーザーの選択待ちで停止した
        {"type": "done", "conversation_id": int} 終端

    2 つの stream_mode を同時に要求する。messages だけだと model を呼ばない決定的 node の
    回答が 1 文字も流れず、updates だけだと Agent の回答が完成するまで画面が止まる。
    langgraph 1.2.11 はこの指定に対して (mode, chunk) の tuple を返す
    (scripts/smoke_langgraph.py で実測済み)。

    actions を最後まで溜めるのは、選択肢がボタンとして描かれるため。本文の途中で送ると
    回答が終わる前にボタンが現れる。suggested_actions に reducer は無く後勝ちなので、
    最後に見た値がその turn の全量になる。

    **中断したときは done を出さない。** done は「その turn が完結した」という印で、
    選択待ちはまだ完結していない。ただし画面を待ちっぱなしにはできないので、終端は
    HTTP ストリームの終了(/api/chat の "data: [DONE]")が受け持つ。役割を分けている:
    [DONE] は「このレスポンスは終わり」、done event は「turn が終わった」。
    画面は interrupt を見たら注文の一覧を描いて /api/actions/resume を待てばよい。
    """
    actions: list = []
    interrupted = False
    async for mode, chunk in get_graph().astream(
        graph_input, _config(cid), stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, meta = chunk
            # metadata の langgraph_node だけが token の出所を示す。無ければ流さない
            # (「出所不明だから流す」に倒すと、node が増えた瞬間に中間出力が漏れる)
            if isinstance(meta, dict) and meta.get("langgraph_node") in ANSWER_NODES:
                # .text を使うのは content が str とブロックの list のどちらもありうるため。
                # list を素通しすると frontend に dict が届き、思考ブロックまで画面に出る
                text = msg.text
                if text:
                    yield {"type": "delta", "text": str(text)}
        elif mode == "updates":
            for node, upd in chunk.items():
                if node == _INTERRUPT_KEY:
                    # node の差分ではなく Interrupt の並び。下の isinstance で読み飛ばす前に拾う
                    interrupted = True
                    yield _interrupt_event(upd, cid)
                    continue
                # node が None を返す経路のような特殊な値を読み飛ばす。
                # ここで落ちると turn ごと止まる
                if not isinstance(upd, dict):
                    continue
                if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                    yield {"type": "delta", "text": upd["answer"]}
                # 本文より先に出典を送る。frontend は [n] を描き始める時点で出典を
                # 持っていないと、クリックできる注釈にできない。
                # 空の citations を送らないのは、強制検索が引けなかったとき前 turn の
                # 出典を消すために [] を書くからで、それは「出典なし」であって
                # 「空の出典欄を開け」ではない
                if node in CITATION_NODES and upd.get("citations"):
                    yield {"type": "citations", "items": upd["citations"]}
                if node == "agent_tools":
                    for m in upd.get("messages", []):
                        name = getattr(m, "name", None)
                        if name:
                            yield {"type": "tool", "name": name}
                if upd.get("suggested_actions"):
                    actions = upd["suggested_actions"]
    if actions:
        yield {"type": "actions", "items": actions}
    if not interrupted:
        yield {"type": "done", "conversation_id": cid}


async def stream_turn(
    user_id: str, message: str, conversation_id: int | None
) -> AsyncIterator[dict]:
    """ストリーミングの入口。user message を保存し、graph の出力を event として流す。"""
    cid, summary, upto = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    async for ev in _stream_events(
        _graph_input(user_id, message, cid, msg_id, summary, upto), cid
    ):
        yield ev
    await summarizer.maybe_schedule_summary(cid)


async def stream_resume(conversation_id: int, resume_value) -> AsyncIterator[dict]:
    """中断した turn を再開し、stream_turn と同じ event を流す。

    user message を保存せず _graph_input も通さない理由は resume_turn と同じ。
    再開の後には Agent の回答が続くので、画面が受け取る event の並びは
    stream_turn と同じでなければならない(選択の前後で描画を分けずに済む)。
    """
    cid = await _require_conversation(conversation_id)
    async for ev in _stream_events(Command(resume=resume_value), cid):
        yield ev
    await summarizer.maybe_schedule_summary(cid)
