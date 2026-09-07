"""graph の node。State の一部を dict で返し、reducer が既存の State へ畳み込む。

node は 2 種類に分かれる。

- **決定的な出口**(chitchat_reply / complaint_reply / fallback_reply): model を呼ばず
  固定文を返す。呼ばないことそのものが価値で、雑談のたびに上流を叩くなら固定文にする
  意味が無い。この 3 つは必ず終わるので、graph に「絶対に返答が返る経路」を作る。
- **上流を使う node**(classify_intent / forced_rag): 分類と検索を行う。どちらも
  上流が落ちても例外を投げない下位実装(app/core/intent.py、app/core/query_understanding.py、
  app/core/selfcheck.py)の上に乗せ、障害を graph 全体の停止に化けさせない。

State に無い key を返しても LangGraph が黙って捨てるため、返す key は
app/graph/state.py の ConversationState に宣言済みのものだけにすること。
"""

import logging

from langchain_core.messages import HumanMessage

from app.config import settings
from app.core import intent as intent_mod
from app.core import query_understanding, retrieval, selfcheck
from app.core.prompts import (
    CHITCHAT_REPLY_TEXT,
    COMPLAINT_REPLY_TEXT,
    FALLBACK_REPLY_TEXT,
)
from app.db import repository

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
