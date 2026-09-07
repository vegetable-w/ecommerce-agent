# app/core/labels.py
"""英語の DB 識別子 → 日本語の表示ラベル。

DB / ORM / ツール引数は英語識別子で統一し、日本語はここでのみ対応付ける。
逆方向（日本語 → 識別子）は用意しない。日本語を DB へ書く経路を作らないため。
"""

CONVERSATION_STATUS: dict[str, str] = {
    "in_progress": "対応中",
    "escalated": "オペレーター対応",
    "closed": "終了",
}

TICKET_TYPE: dict[str, str] = {
    "after_sales": "アフターサービス",
    "complaint": "苦情",
    "inquiry": "問い合わせ",
    "refund": "返金",
}

TICKET_STATUS: dict[str, str] = {
    "pending": "未対応",
    "resolved": "対応済み",
}


def label(table: dict[str, str], value: str) -> str:
    """未知の値が来ても落とさず、識別子をそのまま返す。

    表示のためだけの関数がデータ起因で例外を投げると、会話全体が落ちてしまう。
    未知値は DDL と対応表がずれているサインなので、そのまま見せて気づけるようにする。
    """
    return table.get(value, value)
