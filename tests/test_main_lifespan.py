"""app/main.py の配線: lifespan による graph の起動・停止と action router の登録。

**このモジュールは本物の init_graph を絶対に呼ばない。** 呼ぶと本番の checkpointer
(data/05_checkpoints.sqlite)をテストが作りにいく。すべての lifespan テストで
runtime.init_graph / close_graph を差し替える。
"""

import pytest
from fastapi.testclient import TestClient

from app.graph import runtime
from app.main import app, lifespan


@pytest.fixture
def graph_lifecycle(monkeypatch) -> list[str]:
    """init_graph / close_graph を記録用の偽物へ差し替え、呼ばれた順を返す。"""
    events: list[str] = []

    async def _init():
        events.append("init")

    async def _close():
        events.append("close")

    monkeypatch.setattr(runtime, "init_graph", _init)
    monkeypatch.setattr(runtime, "close_graph", _close)
    return events


def test_lifespan_opens_the_graph_on_startup_and_closes_it_on_shutdown(graph_lifecycle):
    """checkpointer は起動で 1 つ開き、終了で閉じる(runtime の前提そのもの)。

    閉じ忘れると sqlite のハンドルがそのままファイルロックとして残る。
    """
    with TestClient(app) as client:
        assert graph_lifecycle == ["init"]
        assert client.get("/").status_code == 200
    assert graph_lifecycle == ["init", "close"]


async def test_lifespan_closes_the_graph_when_an_error_is_thrown_into_it(graph_lifecycle):
    """異常終了で例外を投げ込まれても checkpointer は閉じる。

    ASGI サーバは異常終了時に lifespan の中断を例外として通知しうる。close を
    finally に置かないと、そのプロセスは sqlite を握ったまま消え、ロックが残る。

    TestClient の `with` の中で例外を出すやり方ではこの経路を通らない
    (portal は shutdown を普通に走らせるだけで、generator へは投げ込まない)ため、
    lifespan を直接開いて __aexit__ に例外を渡す。
    """
    cm = lifespan(app)
    await cm.__aenter__()
    assert graph_lifecycle == ["init"]

    err = RuntimeError("異常終了")
    # 同じ例外がそのまま抜けるので __aexit__ は False を返す(握り潰さない)。
    assert await cm.__aexit__(RuntimeError, err, err.__traceback__) is False
    assert graph_lifecycle == ["init", "close"]


def test_startup_fails_loudly_when_the_graph_cannot_be_opened(monkeypatch):
    """init_graph が落ちたらアプリを起動させない(degrade しない)。

    graph は /api/chat と /api/agent の唯一の実装なので、開けないまま起動すると
    「正常に立ち上がっているのに、チャットは毎回 500」というサーバになる。
    管理画面だけ生きている状態は監視から見て健康に見えてしまうため、
    起動時点で落として気づかせる。
    """
    async def _boom():
        raise RuntimeError("checkpointer を開けない")

    monkeypatch.setattr(runtime, "init_graph", _boom)
    with pytest.raises(RuntimeError, match="checkpointer を開けない"):
        with TestClient(app):
            pass


def test_lifespan_does_not_run_without_a_with_block(graph_lifecycle):
    """`with` を付けない TestClient は lifespan を走らせない。

    既存のテストの大半は module 直下で TestClient(app) を作って画面や API を叩く。
    ここが変わると、それらのテストが一斉に本物の checkpointer を作りにいく。
    starlette 側の挙動だが、このリポジトリの安全性がそれに乗っているので固定する。
    """
    bare = TestClient(app)
    assert bare.get("/").status_code == 200
    assert graph_lifecycle == []


def test_action_router_is_registered_on_the_real_app():
    """画面が叩くのは本番の app。router の登録漏れをここで見る。"""
    assert "/api/actions/create-ticket" in app.openapi()["paths"]


@pytest.mark.parametrize("path", ["/", "/kb", "/admin", "/rag-eval"])
def test_pages_still_carry_the_no_cache_header_after_the_lifespan_was_added(path):
    """lifespan を足すときに middleware か _PAGES を壊していないこと。"""
    res = TestClient(app).get(path)
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-cache"
