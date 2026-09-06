"""app.core.labels のテスト。

DB を触らないため、DBフィクスチャや `pytest.mark.asyncio(loop_scope="session")` は不要
(素の同期関数として書ける)。

各対応表のキー集合は sql/02-ddl.sql の ENUM 定義を実際にパースして突き合わせる
(期待値をここに二重管理しない)。
"""

import pathlib
import re

from app.core import labels

_DDL_PATH = pathlib.Path(__file__).resolve().parent.parent / "sql" / "02-ddl.sql"

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


def _ddl_enums() -> dict[tuple[str, str], set[str]]:
    return _parse_ddl_enums(_DDL_PATH.read_text(encoding="utf-8"))


def test_ddl_has_expected_enum_columns():
    """パーサ自体が壊れていないことの前提チェック。"""
    enums = _ddl_enums()
    assert enums == {
        ("conversations", "status"): {"in_progress", "escalated", "closed"},
        ("messages", "role"): {"user", "assistant", "tool"},
        ("tickets", "ticket_type"): {"after_sales", "complaint", "inquiry"},
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
