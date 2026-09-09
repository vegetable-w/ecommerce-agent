"""graph 全体を貫く会話 State。

LangGraph の node は「State の一部を返す」と、reducer がそれを既存の State に畳み込む。
未宣言の key を返しても**黙って捨てられる**（例外も警告も出ない）ので、node 間で
受け渡す値はすべてここに宣言しておく必要がある。

reducer を付けていない field は「後勝ちの上書き」になる。累積したいものだけ
Annotated で reducer を指定する（messages と trace）。
"""

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# trace を「この turn の分だけ」に畳み直すための目印。
# reducer 付きの field は上書きができない(必ず merge される)ので、turn の入口で
# 空 dict を渡しても前 turn の値が残る。checkpointer が State を持ち越すため、
# それをやらないと 2 turn 目のログに 1 turn 目の forced_rag が混ざる。
TRACE_RESET = "__reset__"


def merge_dict(a: dict | None, b: dict | None) -> dict:
    """trace 用の reducer。同じ key は後勝ち、それ以外は足し合わせる。

    b に TRACE_RESET が入っていたら、そこから作り直す(turn の切り替え)。

    新しい dict を作って返す(引数を書き換えない)。reducer は step ごとに呼ばれるので、
    片方を in-place で更新すると、前の step の trace が後から書き換わる。
    """
    if isinstance(b, dict) and b.get(TRACE_RESET):
        return {k: v for k, v in b.items() if k != TRACE_RESET}
    return {**(a or {}), **(b or {})}


class ConversationState(TypedDict, total=False):
    """total=False にするのは、node が「自分が触る key だけ」を返すため。

    全 key を必須にすると、どの node も State 全体を組み立て直すことになり、
    触っていない値を取りこぼす事故が起きる。
    """

    # 会話履歴。checkpointer が turn をまたいで保持し、add_messages が追記する
    messages: Annotated[list[AnyMessage], add_messages]

    # 古いターンの要約と、それがどの message まで覆っているか。どちらもターンの入口で
    # conversations から読み直す(runtime の _graph_input)。checkpointer は State を
    # 持ち越すので、入れ直さないと要約が更新されても前のターンの値を使い続ける
    summary: str
    summary_upto_msg_id: int  # スライディングウィンドウはこの次の message から原文

    user_id: str
    conversation_id: int

    # 指示対象を解決して書き下した完全な質問。以降の分類と検索はこちらを使う
    resolved_query: str
    intent: str              # app/core/intent.py の INTENTS のいずれか
    intent_confidence: float # 分類の確信度(0-1)
    route: str               # 5 つの出口のいずれか

    # 返金フロー
    order_id: str            # 抽出、または画面で選ばれた注文番号
    order_data: dict         # query_order で取得した注文の中身

    # knowledge route の retrieval 結果。business route では空のまま
    evidence: str            # 番号付きの evidence 本文
    citations: list          # frontend から出典を開くための chunk 情報
    evidence_strong: bool    # 生成前の evidence gate の判定結果

    # 09: 確信度ゲートの出力。3 つとも reducer は付けない(turn ごとの後勝ちの上書き)。
    # 累積させると、前の turn の写しや別を引きずったまま次の turn の判断に混ざる。
    evidence_confidence: float   # このターンの根拠の総合点(app/core/confidence.py)
    # プールへ入れるときの source。"retrieval_low_conf"(根拠そのものが弱い)か
    # "self_check"(根拠は取れたが答えきれない)。DDL の ENUM に合わせた英語の識別子
    fallback_source: str
    # 検索を通ったなら Top3 の写し。通っていなければ空 list。
    # **strong で通ったターンでも残す**。Task 7 の 👎 が後からこの写しを拾うため
    retrieved_snapshot: list

    # 決定的な node（script/complaint/fallback）の回答。
    # Agent の回答はここに入れない。token を stream して frontend へ直接流すため、
    # ここへ溜め込むと「stream した本文」と「State の本文」の 2 つの正が生まれる
    answer: str

    steps: int               # ReAct の step 数。max_agent_steps と比べて打ち切る
    # token 使用量。trace と 09 章の集計用で、**停止条件には使わない**
    tokens_used: int

    # [{"type": "transfer_human"} | {"type": "create_ticket", "draft": {...}}]
    suggested_actions: list

    trace: Annotated[dict, merge_dict]   # observability
