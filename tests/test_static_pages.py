"""管理画面が開けて、共通シェルを読み込んでいることだけを見る到達性テスト。

画面の中身は Vibe Coding でブラウザ実測するので、ここでは描画内容を固定しない。
「新しい管理画面を足したのにナビへ登録し忘れた」「静的ファイルの mount が壊れた」を
気づけるだけの最小限にとどめる。
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.parametrize("path", ["/", "/kb", "/admin"])
def test_page_is_reachable(path):
    res = client.get(path)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")


@pytest.mark.parametrize("path", ["/kb", "/admin"])
def test_admin_pages_load_the_shared_shell(path):
    """共通ナビを読み込まない管理画面ができると、そのページだけ導線から外れる。"""
    assert "/static/admin.js" in client.get(path).text


def test_shared_shell_is_served():
    res = client.get("/static/admin.js")
    assert res.status_code == 200
    assert "mountAdminNav" in res.text


def test_chat_page_links_to_admin():
    assert 'href="/admin"' in client.get("/").text
