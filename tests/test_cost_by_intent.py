"""scripts/cost_by_intent.py の純関数(09 章の intent 別 token 集計)。

Langfuse にも上流にも DB にも触らずに測れるのは 2 つ:

- build_rows: 割合と平均を**確定させる**場所。端末の表と /observability の画面が
  同じ list を読むので、ここが変われば両方が同時に変わる。画面側で計算し直すと
  丸め方の違いだけで数字が食い違う道が開く。
- _intent_from_state: trace 根の output(graph の最終 State)から intent を取る
  唯一の経路。09 章の実測で、node から付ける tag は 1 つも届かないと分かっている
  (app/core/observability.py)ので、実際に効いているのはこちら。

成果物 JSON の**形**が画面と結ばれていることは tests/test_observability_api.py が見る。
"""

import pytest

from scripts import cost_by_intent as cost


@pytest.fixture(autouse=True)
def _quiet():
    """端末出力の控え(module 変数)をテストごとに空にする。"""
    cost._LINES.clear()
    yield
    cost._LINES.clear()


# --- build_rows --------------------------------------------------------------


def test_rows_are_ordered_by_tokens_and_carry_the_finished_numbers():
    """token の降順。平均も割合も表示用の文字列もここで確定していること。"""
    rows = cost.build_rows({
        "雑談": {"count": 1, "tokens": 1429},
        "配送": {"count": 3, "tokens": 15548},
        "商品相談": {"count": 1, "tokens": 5322},
    })

    assert [r["intent"] for r in rows] == ["配送", "商品相談", "雑談"]
    assert rows[0]["avg_tokens"] == 5182          # 15548 // 3(切り捨て)
    assert round(rows[0]["share"], 4) == 0.6973
    # 画面に丸め直させない。70% と 0.6973 が別々に丸められると端末と食い違う
    assert rows[0]["share_label"] == "70%"
    assert set(rows[0]) == {"intent", "count", "tokens", "avg_tokens",
                            "share", "share_label"}


def test_shares_add_up_over_the_window():
    """割合の分母は窓の合計。1 つの intent しか無ければ 100%。"""
    rows = cost.build_rows({"配送": {"count": 2, "tokens": 100}})
    assert rows[0]["share"] == 1.0 and rows[0]["share_label"] == "100%"


def test_a_zero_token_window_does_not_divide_by_zero():
    """token が 1 つも記録されていない窓でも落ちないこと。

    Langfuse の observation に usage が乗らない構成でこの形になる。ここで
    ZeroDivisionError にすると、レポートが 1 行も出ないまま script が死ぬ。
    """
    rows = cost.build_rows({"配送": {"count": 2, "tokens": 0}})
    assert rows[0]["share"] == 0.0 and rows[0]["share_label"] == "0%"
    assert rows[0]["avg_tokens"] == 0


def test_an_empty_table_is_an_empty_list_not_an_error():
    assert cost.build_rows({}) == []


def test_a_count_of_zero_does_not_divide_by_zero():
    """平均の分母は最低 1。trace 数が 0 の行は集計側では作られないが、
    ここが落ちると窓の端で表そのものが出なくなる。"""
    assert cost.build_rows({"配送": {"count": 0, "tokens": 10}})[0]["avg_tokens"] == 10


# --- _intent_from_state ------------------------------------------------------


def test_intent_is_read_from_the_trace_root_output():
    """dict でも JSON 文字列でも読めること(API の版で形が変わる)。"""
    assert cost._intent_from_state({"intent": "配送", "answer": "..."}) == "配送"
    assert cost._intent_from_state('{"intent": "配送"}') == "配送"


def test_an_unknown_intent_is_not_believed():
    """9 分類の外の値は採らない。**表に知らない行を生やさない**。

    ここを素通しにすると、モデルの生の出力(丸め前)や別経路の trace が
    intent の行として表に混ざり、どれが本物か分からなくなる。
    """
    assert cost._intent_from_state({"intent": "配送に関するお問い合わせ"}) == ""
    assert cost._intent_from_state({"intent": ""}) == ""


@pytest.mark.parametrize("output", [
    pytest.param(None, id="none"),
    pytest.param("壊れた JSON {", id="not-json"),
    pytest.param("[1, 2, 3]", id="json-but-not-an-object"),
    pytest.param(123, id="number"),
    pytest.param({"intent": 7}, id="intent-is-not-a-string"),
])
def test_unreadable_output_is_counted_as_unresolved(output):
    """読めない形は空文字。呼び出し側が「解決できなかった」として数える。

    例外にすると、trace が 1 本壊れているだけで窓全体の集計が出なくなる。
    """
    assert cost._intent_from_state(output) == ""
