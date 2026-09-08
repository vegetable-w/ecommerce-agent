from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


class SessionStore:
    """メモリ内の会話ストレージ。session_id -> メッセージ一覧。この章では永続化しない。"""

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, *messages: BaseMessage) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)

    def clear(self, session_id: str | None = None) -> None:
        """session_idを指定した場合はそのセッションのみ、指定しない場合は全セッションを消去する。"""
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """max_tokensは正の値であること。trim_messagesは0以下だと例外を投げず履歴を無音で全消去するため、
    ここでは保証しない(settings.token_budgetのgt=0が唯一の防波堤)。"""
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        allow_partial=False,
    )


# ---------------------------------------------------------------------------
# 07 スライディングウィンドウと要約の差し込み
# ---------------------------------------------------------------------------

# runtime が graph へ入れる user の発話には、MySQL の message id を
# `id="db-123"` の形で付ける。窓の境界合わせはこの anchor だけを見る。
# 発話の本文や並び順で境界を推測しない: 同じ文面が何度も出てくる会話では
# どの発話のことか決められず、静かに 1 ターンずれる。
_DB_ID_PREFIX = "db-"


def _db_msg_id(m: BaseMessage) -> int | None:
    """message の id から MySQL の message id を取り出す。anchor でなければ None。"""
    mid = getattr(m, "id", None)
    if isinstance(mid, str) and mid.startswith(_DB_ID_PREFIX):
        try:
            return int(mid[len(_DB_ID_PREFIX):])
        except ValueError:
            return None
    return None


def build_window(messages: list[BaseMessage], summary_upto_msg_id: int,
                 max_tokens: int) -> list[BaseMessage]:
    """要約が覆う範囲より後ろだけを残し、さらに token 上限で刈る。

    切る位置は「anchor を持ち、その id が境界より大きい最初の user 発話」。
    07 より前からある会話には anchor が無く、要約が全区間を覆った直後は
    境界より後ろに user 発話が 1 件も無い。どちらの場合も切らずに全体を渡し、
    token 上限だけを掛ける。刈った結果が空になったら、上限を超えていても
    元のウィンドウを返す(空を渡すと、モデルはいま聞かれたことすら見ずに答える)。

    **引数の list も State も書き換えない。** モデルを呼ぶ直前に組み立てるだけで、
    checkpoint の全履歴はそのまま残す。ここで履歴そのものを刈ると、一度刈られた
    発話は二度と戻せない。
    """
    start = 0
    if summary_upto_msg_id:
        for i, m in enumerate(messages):
            if isinstance(m, HumanMessage):
                did = _db_msg_id(m)
                if did is not None and did > summary_upto_msg_id:
                    start = i
                    break
    window = messages[start:]
    trimmed = trim_history(window, max_tokens=max_tokens)
    return trimmed or window


def summary_line(summary: str | None) -> str:
    """指示対象の解決と意図分類へ渡す 1 行。要約が無ければ空文字。

    あちらは履歴を 1 つの文字列として受け取るので、SystemMessage ではなく
    行として混ぜる。
    """
    return f"(以前の会話の要約:{summary})" if summary else ""


def summary_system(summary: str | None) -> SystemMessage | None:
    """main_agent へ渡す要約の SystemMessage。要約が無ければ None。

    persona の SystemMessage とは**別の message** にする。可変の要約を persona 本体へ
    連結すると、ターンごとに prompt の先頭が変わって prefix cache が毎回外れる。
    """
    if not summary:
        return None
    return SystemMessage(
        "## 以前の会話の要約(古いターンは圧縮済み。ここに書かれた事実は信用してよい)\n"
        + summary
    )
