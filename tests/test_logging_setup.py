"""app.* のログが実際に出ることのテスト(app/core/logging_setup.py)。

**caplog を使わない。** caplog は自分で水準を下げて handler を挿すので、
「設定が無くてもログが出る」ように見えてしまう。07 章の model_ctx は
まさにそれで、単体テストは緑なのに動いているサービスでは 1 行も出ていなかった。
ここで確かめたいのは書式ではなく、**設定を通したときに出力先へ届くこと**。
"""

import logging

import pytest

from app.core import logging_setup


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    """logger と「設定済み」フラグを毎回まっさらに戻す。

    logging.getLogger("app") はプロセスで 1 つしかないので、後始末をしないと
    このモジュールの handler が他のテストのログを掴んだまま残る。
    """
    logger = logging.getLogger("app")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    logger.handlers = []
    monkeypatch.setattr(logging_setup, "_configured", False)
    monkeypatch.setattr(logging_setup.settings, "log_file",
                        str(tmp_path / "app.log"))
    yield tmp_path / "app.log"
    logger.handlers, logger.level, logger.propagate = saved


def test_info_from_app_loggers_reaches_the_file(fresh):
    """設定した後は app.* の INFO がファイルに残る(受け入れ検証はこれを読む)。"""
    logging_setup.configure()
    logging.getLogger("app.graph.nodes").info("model_ctx conv=1 window=6")

    for h in logging.getLogger("app").handlers:
        h.flush()
    assert "model_ctx conv=1 window=6" in fresh.read_text(encoding="utf-8")


def test_info_is_dropped_without_configure(fresh):
    """設定を呼ばなければ出ない。この差がテストの意味そのもの。"""
    logging.getLogger("app.graph.nodes").info("model_ctx conv=1")
    assert not fresh.exists()


def test_japanese_survives_the_file_handler(fresh):
    """日本語が化けない。要約は日本語なので、既定 encoding のままだと落ちる。"""
    logging_setup.configure()
    logging.getLogger("app.core.summarizer").info("要約を保存しました conv=1")

    for h in logging.getLogger("app").handlers:
        h.flush()
    assert "要約を保存しました" in fresh.read_text(encoding="utf-8")


def test_configure_twice_does_not_duplicate_handlers(fresh):
    """二度呼んでも handler は増えない(増えると 1 行が 2 回ずつ出る)。"""
    logging_setup.configure()
    n = len(logging.getLogger("app").handlers)
    logging_setup._configured = False   # 「初回のつもりで」もう一度呼ぶ
    logging_setup.configure()
    assert len(logging.getLogger("app").handlers) == n * 2   # 素朴に呼べば倍になる

    # 実際の入口(configure が既に済んでいる)では増えないこと
    before = len(logging.getLogger("app").handlers)
    logging_setup.configure()
    assert len(logging.getLogger("app").handlers) == before


def test_unopenable_log_file_does_not_stop_the_app(fresh, monkeypatch):
    """ログの置き場所が無くても起動は続ける(コンソールには出る)。"""
    # ファイルとして開けないパス(既存ファイルの下)を渡す
    bad = fresh.parent / "notadir"
    bad.write_text("x", encoding="utf-8")
    monkeypatch.setattr(logging_setup.settings, "log_file", str(bad / "app.log"))

    logging_setup.configure()   # 例外を出さない
    assert logging.getLogger("app").handlers                 # console は付いている


def test_records_still_reach_the_root_logger(fresh):
    """propagate を切らない。切ると caplog を使う既存テストがこの後すべて空になる。"""
    logging_setup.configure()
    seen = []

    class Catch(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    root = logging.getLogger()
    h = Catch()
    root.addHandler(h)
    try:
        logging.getLogger("app.graph.nodes").info("model_ctx conv=1")
    finally:
        root.removeHandler(h)
    assert "model_ctx conv=1" in seen
