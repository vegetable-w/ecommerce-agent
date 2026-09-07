"""graph を FastAPI から使えるようにする層。checkpointer の寿命と 2 つの実行入口。

graph そのものは app/graph/build.py で組み上がっている。ここが受け持つのは、

- **checkpointer を 1 つだけ開いて使い回すこと**。会話の State は turn をまたいで
  sqlite に残る(spec D1: business DB は MySQL、checkpointer は sqlite と役割を分ける)。
  接続をリクエストごとに開くと、同時に走る turn の数だけ sqlite のハンドルが増え、
  閉じ忘れがそのままファイルロックとして残る。起動時に 1 つ開き、終了時に閉じる。
- **graph の出力を frontend が読める event に写像すること**。/api/agent(非ストリーミング)は
  最終 State を、/api/chat(SSE)は event の列を使う。

写像の要点は「**何を流さないか**」にある。stream_mode="messages" は graph の中で
起きた model 呼び出しの token を**すべて**運んでくるので、素通しすると
classify_intent の分類結果(「配送」の 2 文字)や forced_rag のクエリ書き換えが、
回答本文と同じ delta としてユーザーの画面に出る。node 名で絞るのはそのため。
"""

import logging
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.db import repository
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

# 回答の token を frontend へ流す node。ここに無い node の model token は流さない。
# 「回答を書いている node はどれか」を白リストで持つのは、除外リストにすると
# model を呼ぶ node が増えるたびに書き足さねばならず、書き忘れが即漏洩になるため。
ANSWER_NODES = {"agent_llm"}
# 決定的 node は model を呼ばないので token が流れてこない。回答は state["answer"] に
# 入るので、updates から 1 塊の delta として拾う
DETERMINISTIC_ANSWER_NODES = {"chitchat_reply", "complaint_reply", "fallback_reply"}

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
    _graph = build_graph(checkpointer=checkpointer)
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


async def _ensure_conversation(user_id: str, conversation_id: int | None) -> int:
    """会話 ID を確定する。None なら採番、指定があれば存在を確かめる。

    存在しない ID を素通しすると、誰も読まない会話へ user message を書き込み、
    graph の thread_id もその番号で切られる。ここで止める。
    """
    if conversation_id is None:
        return await repository.create_conversation(user_id)
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    return conversation_id


def _graph_input(user_id: str, message: str, cid: int) -> dict:
    """graph へ渡す 1 turn 分の入力。

    messages は今回の発話 1 通だけでよい。過去の履歴は checkpointer が thread_id ごとに
    持っており、add_messages が追記する。steps と tokens_used を 0 で入れ直すのは、
    この 2 つに reducer が無く(後勝ちの上書き)、turn ごとに戻さないと前 turn の
    step 数を引き継いだまま should_continue の上限に当たるため。
    """
    return {"messages": [HumanMessage(message)], "user_id": user_id,
            "conversation_id": cid, "steps": 0, "tokens_used": 0}


def _config(cid: int) -> dict:
    """thread_id は会話 ID。checkpointer はこの値で State を切り分ける。"""
    return {"configurable": {"thread_id": str(cid)}}


async def run_turn(user_id: str, message: str, conversation_id: int | None) -> dict:
    """非ストリーミングの入口。user message を保存し、graph を走らせて最終 State を返す。

    assistant message の保存は log node が行う(4 つの出口がすべてそこへ合流する)ので、
    ここでは user 側だけを書く。
    """
    cid = await _ensure_conversation(user_id, conversation_id)
    await repository.append_message(cid, "user", content=message)
    final = await get_graph().ainvoke(_graph_input(user_id, message, cid), _config(cid))
    return {"conversation_id": cid, "state": final}


async def stream_turn(
    user_id: str, message: str, conversation_id: int | None
) -> AsyncIterator[dict]:
    """ストリーミングの入口。graph の出力を frontend 向けの event dict へ写像する。

    出す event は 5 種類:
        {"type": "tool", "name": str}            agent_tools が実行した tool
        {"type": "citations", "items": list}     forced_rag が引いた出典
        {"type": "delta", "text": str}           回答本文(agent_llm の token / 決定的 node の固定文)
        {"type": "actions", "items": list}       選択肢(有人対応 / チケット作成)
        {"type": "done", "conversation_id": int} 終端

    2 つの stream_mode を同時に要求する。messages だけだと model を呼ばない決定的 node の
    回答が 1 文字も流れず、updates だけだと Agent の回答が完成するまで画面が止まる。
    langgraph 1.2.11 はこの指定に対して (mode, chunk) の tuple を返す
    (scripts/smoke_langgraph.py で実測済み)。

    actions を最後まで溜めるのは、選択肢がボタンとして描かれるため。本文の途中で送ると
    回答が終わる前にボタンが現れる。suggested_actions に reducer は無く後勝ちなので、
    最後に見た値がその turn の全量になる。
    """
    cid = await _ensure_conversation(user_id, conversation_id)
    await repository.append_message(cid, "user", content=message)

    actions: list = []
    async for mode, chunk in get_graph().astream(
        _graph_input(user_id, message, cid), _config(cid),
        stream_mode=["messages", "updates"],
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
                # node が None を返す経路や __interrupt__ のような特殊な key を読み飛ばす。
                # ここで落ちると turn ごと止まる
                if not isinstance(upd, dict):
                    continue
                if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                    yield {"type": "delta", "text": upd["answer"]}
                # 本文より先に出典を送る。frontend は [n] を描き始める時点で出典を
                # 持っていないと、クリックできる注釈にできない。
                # 空の citations を送らないのは、forced_rag が weak のとき前 turn の
                # 出典を消すために [] を書くからで、それは「出典なし」であって
                # 「空の出典欄を開け」ではない
                if node == "forced_rag" and upd.get("citations"):
                    yield {"type": "citations", "items": upd["citations"]}
                if node == "agent_tools":
                    for m in upd.get("messages", []):
                        name = getattr(m, "name", None)
                        # create_ticket は実行していない(選択肢へ変換しただけ)。
                        # tool として出すと「チケットを作成しました」と画面に嘘が出る
                        if name and name != "create_ticket":
                            yield {"type": "tool", "name": name}
                if upd.get("suggested_actions"):
                    actions = upd["suggested_actions"]
    if actions:
        yield {"type": "actions", "items": actions}
    yield {"type": "done", "conversation_id": cid}
