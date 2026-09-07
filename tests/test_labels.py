"""app.core.labels のテスト。

DB を触らないため、DBフィクスチャや `pytest.mark.asyncio(loop_scope="session")` は不要
(素の同期関数として書ける)。

各対応表のキー集合は sql/*.sql の ENUM 定義を実際にパースして突き合わせる
(期待値をここに二重管理しない)。CREATE TABLE の定義だけでなく、後の章が
ALTER TABLE ... MODIFY COLUMN で足した値も反映する。ALTER を読まないと、
DDL 上は許されている値を対応表に足した瞬間に「DDL に無い値」として落ちる。
"""

import pathlib
import re

from app.core import labels

_SQL_DIR = pathlib.Path(__file__).resolve().parent.parent / "sql"
_DDL_PATH = _SQL_DIR / "02-ddl.sql"
# ENUM 列を後から書き換える章のファイル。適用順に並べる(後のものが勝つ)。
_ALTER_PATHS = [_SQL_DIR / "06-ticket-type.sql"]

# (テーブル名, カラム名) -> labels.py 内の対応表。
# messages.role はユーザーへ表示することがない内部プロトコル値なので、意図的に
# 対応表を持たない(_NO_LABEL_NEEDED で明示)。今後 DDL に新しい ENUM 列が
# 追加されたとき、ここにも _NO_LABEL_NEEDED にも書き足されなければ
# test_all_ddl_enums_are_accounted_for が失敗し、対応漏れに気づける。
_ENUM_TO_LABEL_TABLE: dict[tuple[str, str], dict[str, str]] = {
    ("conversations", "status"): labels.CONVERSATION_STATUS,
    ("tickets", "ticket_type"): labels.TICKET_TYPE,
    ("tickets", "status"): labels.TICKET_STATUS,
}
_NO_LABEL_NEEDED: set[tuple[str, str]] = {
    ("messages", "role"),  # user/assistant/tool は内部プロトコル値であり画面表示しない
}

_TABLE_RE = re.compile(r"^CREATE TABLE (\w+)")
_ALTER_RE = re.compile(
    r"ALTER TABLE\s+(\w+)\s+MODIFY COLUMN\s+(\w+)\s+ENUM\(([^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)
_ENUM_RE = re.compile(r"^\s*(\w+)\s+ENUM\(([^)]*)\)")
_TABLE_END_RE = re.compile(r"^\)\s*ENGINE=")


def _parse_ddl_enums(ddl_text: str) -> dict[tuple[str, str], set[str]]:
    """DDL の `CREATE TABLE ... ) ENGINE=...` ブロックを走査し、
    (テーブル名, カラム名) -> ENUM 値集合 を返す。
    """
    enums: dict[tuple[str, str], set[str]] = {}
    current_table: str | None = None
    for line in ddl_text.splitlines():
        m = _TABLE_RE.match(line)
        if m:
            current_table = m.group(1)
            continue
        if _TABLE_END_RE.match(line):
            current_table = None
            continue
        if current_table is None:
            continue
        m = _ENUM_RE.match(line)
        if m:
            column = m.group(1)
            values = {v.strip().strip("'") for v in m.group(2).split(",")}
            enums[(current_table, column)] = values
    return enums


def _apply_alter_enums(
    enums: dict[tuple[str, str], set[str]], sql_text: str
) -> dict[tuple[str, str], set[str]]:
    """ALTER TABLE ... MODIFY COLUMN ... ENUM(...) を反映する。

    MODIFY COLUMN は列の定義を丸ごと置き換えるので、ここも上書きにする(和集合に
    しない)。既存の値を書き落とした ALTER は DB 側では列から値が消えることを意味し、
    テストでも同じように消えて対応表との不一致として表に出る必要がある。
    """
    for table, column, values in _ALTER_RE.findall(sql_text):
        enums[(table, column)] = {v.strip().strip("'") for v in values.split(",")}
    return enums


def _ddl_enums() -> dict[tuple[str, str], set[str]]:
    enums = _parse_ddl_enums(_DDL_PATH.read_text(encoding="utf-8"))
    for path in _ALTER_PATHS:
        enums = _apply_alter_enums(enums, path.read_text(encoding="utf-8"))
    return enums


def test_ddl_has_expected_enum_columns():
    """パーサ自体が壊れていないことの前提チェック。"""
    enums = _ddl_enums()
    assert enums == {
        ("conversations", "status"): {"in_progress", "escalated", "closed"},
        ("messages", "role"): {"user", "assistant", "tool"},
        # refund は 06 章の sql/06-ticket-type.sql が ALTER で足した値
        ("tickets", "ticket_type"): {"after_sales", "complaint", "inquiry", "refund"},
        ("tickets", "status"): {"pending", "resolved"},
    }


def test_label_table_keys_exactly_match_ddl_enum_values():
    enums = _ddl_enums()
    for key, label_table in _ENUM_TO_LABEL_TABLE.items():
        assert key in enums, f"DDL に ENUM 列 {key} が見つからない"
        assert set(label_table.keys()) == enums[key], (
            f"{key} の対応表キーが DDL の ENUM 値と一致しない: "
            f"label={set(label_table.keys())} ddl={enums[key]}"
        )


def test_all_ddl_enum_columns_are_accounted_for():
    """DDL の ENUM 列はすべて、対応表を持つか『表示しないので不要』と明示されているかの
    どちらかでなければならない。新しい ENUM 列が追加されて、まだどちらの判断もされて
    いない場合にここで検出する。
    """
    enums = _ddl_enums()
    decided = set(_ENUM_TO_LABEL_TABLE) | _NO_LABEL_NEEDED
    unaccounted = set(enums) - decided
    assert not unaccounted, f"labels.py での対応が未決定の ENUM 列: {unaccounted}"


def test_label_returns_unknown_value_unchanged():
    assert labels.label(labels.CONVERSATION_STATUS, "not_a_real_status") == "not_a_real_status"
    assert labels.label(labels.TICKET_TYPE, "unknown_type") == "unknown_type"
    assert labels.label(labels.TICKET_STATUS, "") == ""
