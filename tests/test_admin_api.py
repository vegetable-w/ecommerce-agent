"""/api/admin/overview のテスト。

この画面の値打ちは「何かが落ちている最中に見られること」なので、テストの主題も
そこに置く: 依存先を全部止めても 200 で、カードごとに自分の失敗を名乗ること。

TestClient ではなく httpx の ASGITransport を使う理由は tests/test_kb_api.py の
モジュール docstring と同じ(DB を触る fixture と同じイベントループで回すため)。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import admin as admin_api
from app.core import jobs
from app.db import repository
from app.kb import milvus_client
from app.main import app

session_loop = pytest.mark.asyncio(loop_scope="session")

CARD_KEYS = ["conversations", "knowledge", "vectors", "staging", "sources", "jobs", "config"]


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://admintest")


def _kill_everything(monkeypatch) -> None:
    """外部依存を全部落とす。設定カードだけは外部に何も依存していないので生き残る。"""

    async def async_boom(*args, **kwargs):
        raise RuntimeError("依存先が停止している")

    def sync_boom(*args, **kwargs):
        raise ConnectionError("依存先が停止している")

    for name in ("conversation_stats", "knowledge_stats", "list_recent_chunks", "staging_stats"):
        monkeypatch.setattr(repository, name, async_boom)
    monkeypatch.setattr(milvus_client, "get_client", sync_boom)
    monkeypatch.setattr(admin_api.documents, "build_chunks", sync_boom)
    monkeypatch.setattr(jobs, "status_all", sync_boom)


@session_loop
async def test_every_card_renders_when_everything_is_down(monkeypatch):
    _kill_everything(monkeypatch)
    async with _client() as c:
        resp = await c.get("/api/admin/overview")

    assert resp.status_code == 200, "依存先が落ちると管理画面ごと 500 になっている"
    cards = resp.json()["cards"]
    assert [card["key"] for card in cards] == CARD_KEYS, "落ちたカードが一覧から消えた"

    broken = [card for card in cards if card["key"] != "config"]
    for card in broken:
        assert card["ok"] is False, card["key"]
        assert card["stats"] is None
        assert card["error"], card["key"]
        assert "取得できません" in card["error"]
        assert card["title"]
    assert resp.json()["degraded"] == [card["key"] for card in broken]


@session_loop
async def test_config_card_survives_and_leaks_no_secret(monkeypatch):
    _kill_everything(monkeypatch)
    async with _client() as c:
        cards = {c_["key"]: c_ for c_ in (await c.get("/api/admin/overview")).json()["cards"]}

    config = cards["config"]
    assert config["ok"] is True, "外部依存の無いカードまで巻き添えで落ちている"
    # ホワイトリストで固定する。フィールドを足すときに、資格情報を含む項目
    # (DATABASE_URL や *_api_key)が紛れ込めばここが赤くなる
    assert set(config["stats"]) == {
        "chat_model", "embed_model", "extract_method", "token_budget",
        "request_timeout", "retrieval_top_k", "retrieval_min_score", "milvus_uri",
    }


@session_loop
async def test_one_dead_dependency_does_not_affect_other_cards(db_session_factory, monkeypatch):
    """Milvus だけ止める。ベクトルのカードだけが失敗し、他はそのまま数字を出す。"""

    def boom(uri=None):
        raise ConnectionError("milvus unreachable")

    monkeypatch.setattr(milvus_client, "get_client", boom)
    async with _client() as c:
        body = (await c.get("/api/admin/overview")).json()

    cards = {card["key"]: card for card in body["cards"]}
    assert body["degraded"] == ["vectors"]
    assert cards["vectors"]["ok"] is False
    for key in ("conversations", "knowledge", "staging", "sources", "jobs", "config"):
        assert cards[key]["ok"] is True, key
        assert cards[key]["error"] is None


@session_loop
async def test_cards_carry_real_numbers(db_session_factory, monkeypatch):
    await repository.insert_knowledge_chunk("c", "送料は", "9900円以上で無料", is_key_clause=1)
    await repository.insert_staging("b1", "conv:1", "返品は", "7日以内")

    class FakeClient:
        def has_collection(self, name):
            return True

    monkeypatch.setattr(milvus_client, "get_client", lambda uri=None: FakeClient())
    monkeypatch.setattr(milvus_client, "count", lambda client: 3)

    async with _client() as c:
        cards = {x["key"]: x for x in (await c.get("/api/admin/overview")).json()["cards"]}

    assert cards["knowledge"]["stats"]["total"] == 1
    assert cards["knowledge"]["stats"]["key_clauses"] == 1
    assert cards["knowledge"]["stats"]["recent"][0]["questions"] == "送料は"
    assert cards["staging"]["stats"]["extracted"] == 1
    assert cards["staging"]["stats"]["batches"] == 1
    assert cards["vectors"]["stats"]["count"] == 3
    # 資料が増減するたびに壊れる固定値ではなく、内訳との整合を見る。
    # 「合計が内訳の和になっている」ほうが、数字を 1 つ書き写すより実際にバグを捕まえる。
    src = cards["sources"]["stats"]
    assert src["total_chunks"] == sum(f["chunks"] for f in src["files"])
    assert src["total_chunks"] > 0
    assert cards["conversations"]["stats"] == {
        "conversations": 0, "messages": 0, "tickets": 0, "faq": 0,
    }
    assert set(cards["jobs"]["stats"]["jobs"][0]) >= {"name", "target", "heavy", "running"}
    assert cards["jobs"]["stats"]["make_error"] is None


@session_loop
async def test_missing_make_is_reported_on_the_jobs_card_only(db_session_factory, monkeypatch):
    """make が見つからなくてもカードは出る。理由が読める形で載ること。"""
    from app.config import settings

    monkeypatch.setattr(settings, "make_bin", "make-that-does-not-exist-xyz")
    async with _client() as c:
        cards = {x["key"]: x for x in (await c.get("/api/admin/overview")).json()["cards"]}

    card = cards["jobs"]
    assert card["ok"] is True
    assert card["stats"]["make"] is None
    assert "MAKE_BIN" in card["stats"]["make_error"]
