"""管理画面が開けて、共通シェルを読み込んでいることだけを見る到達性テスト。

画面の中身は Vibe Coding でブラウザ実測するので、ここでは描画内容を固定しない。
「新しい管理画面を足したのにナビへ登録し忘れた」「静的ファイルの mount が壊れた」を
気づけるだけの最小限にとどめる。
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.parametrize("path", ["/", "/kb", "/admin", "/rag-eval"])
def test_page_is_reachable(path):
    res = client.get(path)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")


@pytest.mark.parametrize("path", ["/kb", "/admin", "/rag-eval"])
def test_admin_pages_load_the_shared_shell(path):
    """共通ナビを読み込まない管理画面ができると、そのページだけ導線から外れる。"""
    assert "/static/admin.js" in client.get(path).text


def test_shared_shell_is_served():
    res = client.get("/static/admin.js")
    assert res.status_code == 200
    assert "mountAdminNav" in res.text


def test_chat_page_links_to_admin():
    assert 'href="/admin"' in client.get("/").text


@pytest.mark.parametrize("path", ["/", "/kb", "/admin", "/rag-eval", "/static/admin.js"])
def test_frontend_is_never_served_stale(path):
    """画面と静的ファイルは必ず確認してから使わせる。

    Cache-Control を付けないと browser は発見的キャッシュに落ち、サーバへ聞かずに
    手元の写しを再利用する。実測: kb.html にボタンを足しても画面に出てこなかった
    (サーバ側は新しい内容を返せる状態だった)。「直したのに反映されない」は、
    この画面群でいちばん時間を溶かす類の不具合なので、ここで固定する。
    """
    res = client.get(path)
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-cache"


def test_api_responses_are_not_touched_by_the_cache_header():
    """no-cache を足すのは画面と静的ファイルだけ。API の応答には付けない。"""
    res = client.get("/api/jobs")
    assert res.status_code == 200
    assert "cache-control" not in {k.lower() for k in res.headers}
