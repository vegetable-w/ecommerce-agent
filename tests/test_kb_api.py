"""/api/kb の HTTP テスト。

TestClient は使わない。あれはアプリを**別のイベントループ**(内部の portal スレッド)で
回すため、conftest の _test_engine(セッションスコープのループに紐付く)を使う経路が
asyncmy の "attached to a different loop" で落ちる。httpx の ASGITransport なら
呼び出し側のループでそのままアプリを実行するので、DB を触るテストと同居できる。

同期テストと非同期テストが混ざるモジュールでモジュールレベルの pytestmark を置くと、
同期テスト側にも asyncio マーカーが付いて PytestWarning が出る。マーカーは
関数ごとに付ける(session_loop)。
"""

import pathlib
from collections import Counter

import pytest
from httpx import ASGITransport, AsyncClient

from app.core import jobs
from app.db import repository
from app.kb import documents, milvus_client, sources
from app.main import app

session_loop = pytest.mark.asyncio(loop_scope="session")

MANUAL = "after-sales-manual.md"


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://kbtest")


def _manual_chunks() -> list:
    md = sources.source_path(MANUAL).read_text(encoding="utf-8")
    return documents.build_chunks(md, content_type=sources.SOURCE_TYPES[MANUAL])


async def _inventory() -> tuple[dict, set[str], list[int]]:
    """「何も書いていない」ことを言うための在庫。件数・指紋・id の 3 点で見る。"""
    stats = await repository.knowledge_stats()
    fps = await repository.list_chunk_fingerprints()
    ids = [c.id for c in await repository.list_recent_chunks(500)]
    return stats, fps, ids


# ---------------------------------------------------------------------------
# preview は読むだけ
# ---------------------------------------------------------------------------


@session_loop
async def test_preview_writes_nothing(db_session_factory):
    # 先に 2 件入れておく。空の DB 相手だと「0 件のままだった」しか言えず、
    # 「書き込みが起きなかった」の証明として弱い
    await repository.insert_knowledge_chunk("既存", "既存の質問", "既存の答え", content_type="faq")
    await repository.insert_knowledge_chunk("既存", "もう1件", "もう1件の答え", content_type="faq")
    before = await _inventory()

    async with _client() as c:
        resp = await c.post("/api/kb/preview", json={"source": MANUAL})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == len(_manual_chunks()) > 0
    assert body["content_type"] == "manual"
    assert body["tables"] >= 1

    assert await _inventory() == before, "preview が knowledge_chunks を書き換えた"


@session_loop
async def test_preview_reports_per_chunk_detail(db_session_factory):
    async with _client() as c:
        body = (await c.post("/api/kb/preview", json={"source": MANUAL})).json()
    expected = _manual_chunks()
    assert [x["answer"] for x in body["chunks"]] == [ch.answer for ch in expected]
    for item, ch in zip(body["chunks"], expected):
        assert item["section_path"] == ch.section_path
        assert item["questions"] == ch.questions
        assert item["chars"] == len(ch.answer)
        assert item["is_key_clause"] == bool(ch.is_key_clause)
        assert item["is_duplicate"] is False  # DB は空なので既存重複は無い
    assert any(x["is_table"] for x in body["chunks"])
    assert body["key_clauses"] == sum(ch.is_key_clause for ch in expected)


@session_loop
async def test_preview_and_ingest_agree_on_chunk_count(db_session_factory):
    """プレビューの件数と実際に入る件数が食い違わないこと(分割器が 1 つである証拠)。"""
    async with _client() as c:
        preview = (await c.post("/api/kb/preview", json={"source": MANUAL})).json()
        ingest = (await c.post("/api/kb/ingest", json={"source": MANUAL})).json()
    assert ingest["inserted"] == preview["total"]
    assert (await repository.knowledge_stats())["total"] == preview["total"]


# ---------------------------------------------------------------------------
# 重複判定
# ---------------------------------------------------------------------------


@session_loop
async def test_reingesting_same_text_inserts_nothing(db_session_factory):
    payload = {"source": MANUAL}
    async with _client() as c:
        first = (await c.post("/api/kb/ingest", json=payload)).json()
        total_after_first = (await repository.knowledge_stats())["total"]
        second = (await c.post("/api/kb/ingest", json=payload)).json()

    n = len(_manual_chunks())
    assert (first["inserted"], first["skipped"]) == (n, 0)
    assert (second["inserted"], second["skipped"]) == (0, n)
    assert second["ids"] == []
    assert (await repository.knowledge_stats())["total"] == total_after_first == n


@session_loop
async def test_table_split_chunks_sharing_a_title_are_all_kept(db_session_factory):
    """大きな表が割れてできた chunk は見出しを共有する。それを重複として落とさないこと。

    指紋が questions だけだと、2 枚目以降が丸ごと消える。answer を含めているから
    残る、という差がここに出る。
    """
    chunks = _manual_chunks()
    shared = [q for q, k in Counter(c.questions for c in chunks).items() if k > 1]
    assert shared, "見出しを共有する chunk が無く、この不変条件を検証できない"

    async with _client() as c:
        body = (await c.post("/api/kb/ingest", json={"source": MANUAL})).json()
    assert body["inserted"] == len(chunks)

    stored = await repository.list_recent_chunks(500)
    for q in shared:
        want = [c.answer for c in chunks if c.questions == q]
        got = [r.answer for r in stored if r.questions == q]
        assert sorted(got) == sorted(want), f"見出し {q!r} の chunk が欠けた"
        assert len(set(got)) == len(got) > 1


@session_loop
async def test_preview_marks_already_stored_chunks_as_duplicates(db_session_factory):
    async with _client() as c:
        await c.post("/api/kb/ingest", json={"source": MANUAL})
        body = (await c.post("/api/kb/preview", json={"source": MANUAL})).json()
    assert body["duplicate_check"] == "ok"
    assert body["duplicates"] == body["total"]
    assert all(x["is_duplicate"] for x in body["chunks"])


@session_loop
async def test_duplicate_flag_is_none_when_mysql_unreadable(db_session_factory, monkeypatch):
    """既存を読めないときは「重複ではない」ではなく「判定していない」と言うこと。"""

    async def boom() -> set[str]:
        raise RuntimeError("MySQL 停止中")

    monkeypatch.setattr(repository, "list_chunk_fingerprints", boom)
    async with _client() as c:
        resp = await c.post("/api/kb/preview", json={"source": MANUAL})
    assert resp.status_code == 200
    body = resp.json()
    assert body["duplicate_check"] == "unavailable"
    assert body["duplicates"] is None
    assert all(x["is_duplicate"] is None for x in body["chunks"])


# ---------------------------------------------------------------------------
# 入力検査(すべて 400)
# ---------------------------------------------------------------------------


@session_loop
@pytest.mark.parametrize("payload", [
    {},
    {"text": ""},
    {"text": "   ", "content_type": "faq"},
    {"text": "# 見出し\n本文", "content_type": None},
    {"text": "# 見出し\n本文", "content_type": "unknown"},
    {"text": "# 見出し\n本文", "content_type": "FAQ"},
    {"text": "# 見出し\n本文", "content_type": ""},
    {"source": "not-registered.md"},
    {"source": ""},
])
async def test_invalid_input_is_400(db_session_factory, payload):
    async with _client() as c:
        for path in ("/api/kb/preview", "/api/kb/ingest"):
            resp = await c.post(path, json=payload)
            assert resp.status_code == 400, (path, payload, resp.text)


@session_loop
async def test_source_outside_registry_is_rejected_without_reading(db_session_factory, monkeypatch):
    """`../` を含む名前などは、ファイルを 1 バイトも読まずに 400。

    「拒否された」だけでなく「読まれていない」ことまで見る。read_text をスパイに
    差し替え、呼ばれた記録が空であることを主張する。読んだ中身は当然どこにも出さない。
    """
    reads: list[str] = []
    real_read = pathlib.Path.read_text

    def spy(self, *args, **kwargs):
        reads.append(str(self))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", spy)

    hostile = [
        "../../.env",
        "..\\..\\.env",
        "../data/kb/product-faq.md",
        "data/kb/../../.env",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        ".env",
        "product-faq.md ",
    ]
    async with _client() as c:
        for name in hostile:
            resp = await c.post("/api/kb/preview", json={"source": name})
            assert resp.status_code == 400, name
            resp = await c.post("/api/kb/ingest", json={"source": name})
            assert resp.status_code == 400, name
    assert reads == [], "拒否されるべき資料名でファイルを読んだ"
    assert (await repository.knowledge_stats())["total"] == 0


@session_loop
async def test_registered_source_is_accepted(db_session_factory):
    async with _client() as c:
        for name in sources.SOURCE_TYPES:
            resp = await c.post("/api/kb/preview", json={"source": name})
            assert resp.status_code == 200, name
            assert resp.json()["total"] > 0


@session_loop
async def test_search_requires_query(db_session_factory):
    async with _client() as c:
        assert (await c.post("/api/kb/search", json={})).status_code == 400
        assert (await c.post("/api/kb/search", json={"query": "  "})).status_code == 400


# ---------------------------------------------------------------------------
# 依存先が落ちているとき
# ---------------------------------------------------------------------------


@session_loop
async def test_overview_survives_milvus_being_unreachable(db_session_factory, monkeypatch):
    """Milvus が居なくても 200。consistent は False ではなく null。

    False は「両方読めて数が食い違っている」という別の事実で、対処も違う
    (再ベクトル化するか、まず依存先を起こすか)。ここを取り違えると画面が嘘をつく。
    """

    def boom(uri=None):
        raise ConnectionError("milvus unreachable")

    monkeypatch.setattr(milvus_client, "get_client", boom)
    async with _client() as c:
        resp = await c.get("/api/kb/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["consistent"] is None
    assert body["milvus_error"]
    # MySQL 側は生きているので、そちらの数字は出続ける
    assert body["knowledge"] == {"total": 0, "pending": 0, "done": 0, "key_clauses": 0}
    assert body["staging"]["total"] == 0
    assert [s["name"] for s in body["sources"]] == list(sources.SOURCE_TYPES)
    assert all(s["chunks"] > 0 for s in body["sources"])
    # 数を直書きするとジョブを 1 つ増やすたびに無関係なテストが赤くなる。
    # ここで見たいのは「登録済みのジョブが全部載っていること」。
    assert {j["name"] for j in body["jobs"]} == set(jobs.JOBS)


@session_loop
async def test_overview_consistent_is_none_when_mysql_unreadable(db_session_factory, monkeypatch):
    async def boom() -> dict:
        raise RuntimeError("MySQL 停止中")

    monkeypatch.setattr(repository, "knowledge_stats", boom)
    monkeypatch.setattr(milvus_client, "count", lambda client, collection=None: 0)
    async with _client() as c:
        body = (await c.get("/api/kb/overview")).json()
    assert body["knowledge"] is None
    assert body["knowledge_error"]
    assert body["consistent"] is None


@session_loop
async def test_overview_reports_consistency_when_both_sides_readable(
    db_session_factory, monkeypatch
):
    """両方読めるときは bool を返す。null しか返さない実装では通らない。"""
    await repository.insert_knowledge_chunk("c", "q", "a")
    counts = {"n": 0}

    class FakeClient:
        def has_collection(self, name):
            return True

    monkeypatch.setattr(milvus_client, "get_client", lambda uri=None: FakeClient())
    monkeypatch.setattr(milvus_client, "count", lambda client, collection=None: counts["n"])
    async with _client() as c:
        assert (await c.get("/api/kb/overview")).json()["consistent"] is True  # done=0, milvus=0
        counts["n"] = 5
        assert (await c.get("/api/kb/overview")).json()["consistent"] is False


@session_loop
async def test_ingest_vectorize_failure_returns_502_and_keeps_rows(db_session_factory, monkeypatch):
    """ベクトル化が落ちても MySQL は巻き戻さない。502 と「再実行で補完できる」を返す。"""

    async def boom(client, batch_size: int = 64, collection: str = "knowledge") -> int:
        raise RuntimeError("埋め込み上流が停止")

    monkeypatch.setattr(milvus_client, "get_client", lambda uri=None: object())
    monkeypatch.setattr(milvus_client, "ensure_collection", lambda client, collection=None: None)
    monkeypatch.setattr("app.api.kb.dualwrite.vectorize_pending", boom)

    n = len(_manual_chunks())
    async with _client() as c:
        resp = await c.post("/api/kb/ingest", json={"source": MANUAL, "vectorize": True})
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert str(n) in detail and "pending" in detail

    stats = await repository.knowledge_stats()
    assert stats["total"] == stats["pending"] == n, "失敗を理由に保存済みの chunk を消した"


@session_loop
async def test_ingest_then_vectorize_marks_rows_done(db_session_factory, monkeypatch):
    """成功経路。ベクトル化は dualwrite.vectorize_pending を通ること(自前実装でないこと)。"""
    seen = {"called": 0}
    real = repository.list_pending_chunks

    async def spy(chunk_ids=None):
        seen["called"] += 1
        seen["chunk_ids"] = chunk_ids
        return await real(chunk_ids)

    monkeypatch.setattr(repository, "list_pending_chunks", spy)
    monkeypatch.setattr(milvus_client, "get_client", lambda uri=None: object())
    monkeypatch.setattr(milvus_client, "ensure_collection", lambda client, collection=None: None)
    monkeypatch.setattr(milvus_client, "upsert_vectors",
                        lambda client, rows, collection=None: None)
    monkeypatch.setattr(milvus_client, "flush", lambda client, collection=None: None)

    async def fake_embed(texts):
        return [[0.1] * milvus_client.DIM for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", fake_embed)

    n = len(_manual_chunks())
    async with _client() as c:
        body = (await c.post(
            "/api/kb/ingest", json={"source": MANUAL, "vectorize": True}
        )).json()
    assert body["inserted"] == n
    assert body["vectorized"] == n
    assert seen["called"] >= 1, "vectorize_pending を経由していない"
    # 取り込みの経路は今までどおり全 pending を拾う(id で絞るのは査読の承認だけ)
    assert seen["chunk_ids"] is None
    assert (await repository.knowledge_stats())["done"] == n


@session_loop
async def test_staging_endpoint_lists_rows(db_session_factory):
    await repository.insert_staging("b1", "conv:1", "送料は", "9900円以上で無料")
    await repository.insert_staging("b2", "conv:2", "返品は", "7日以内")
    async with _client() as c:
        body = (await c.get("/api/kb/staging")).json()
    assert len(body["rows"]) == 2
    assert body["stats"]["batches"] == 2
    assert body["stats"]["extracted"] == 2
    async with _client() as c:
        assert (await c.get("/api/kb/staging?status=kept")).json()["rows"] == []


@session_loop
async def test_search_returns_hits_with_scores(db_session_factory, monkeypatch):
    async def fake_search(query, top_k=None, min_score=None, client=None):
        return [{"id": 1, "score": 0.87, "question": "送料は", "answer": "9900円以上で無料"}]

    monkeypatch.setattr("app.api.kb.retrieval.search_knowledge", fake_search)
    async with _client() as c:
        body = (await c.post("/api/kb/search", json={"query": "送料"})).json()
    assert body["hits"][0]["score"] == 0.87
    assert body["query"] == "送料"


@session_loop
async def test_search_upstream_failure_is_502(db_session_factory, monkeypatch):
    async def boom(query, top_k=None, min_score=None, client=None):
        raise RuntimeError("埋め込み上流が停止")

    monkeypatch.setattr("app.api.kb.retrieval.search_knowledge", boom)
    async with _client() as c:
        resp = await c.post("/api/kb/search", json={"query": "送料"})
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# 画面ルート(HTML は Task 18 で配置する)
# ---------------------------------------------------------------------------


def test_page_routes_are_registered():
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert {"/kb", "/admin"} <= paths
