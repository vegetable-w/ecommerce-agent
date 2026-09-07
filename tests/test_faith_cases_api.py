"""幻覚ケース台帳(faith_cases)の repository と /api/rag-eval/faith-cases のテスト。

この台帳の存在理由は「実行をまたいで積み上がること」なので、テストの主題も
1 回の呼び出しの戻り値ではなく **2 回目以降に何が起きるか** に置く:

- 同じ問いが再び幻覚になっても行は増えず、最新の実行の内容で上書きされること
- 対処済みだった行が再発したら未対処へ戻ること(ここが黙って消えると、台帳は
  「解決済み」の顔をしたまま再発を飲み込む。画面を見ても何も起きていないように見える)
- 画面のタブに出す件数が、いま開いているタブに引きずられないこと

TestClient は使わない。あれはアプリを別のイベントループ(内部の portal スレッド)で
回すため、conftest の _test_engine を使う経路が asyncmy の "attached to a different loop"
で落ちる(tests/test_kb_api.py の冒頭を参照)。
"""

import re

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.config import settings
from app.db import repository
from app.db.models import FaithCase
from app.main import app

pytestmark = pytest.mark.asyncio(loop_scope="session")

# 本番相当の support を絶対に触らないこと。テストが向いている先をここで一度固定する
assert "support_test" in settings.test_database_url, (
    "テスト用 DB が support_test を指していない。本番相当の support へ書き込む恐れがある"
)


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://fctest")


# 根拠スナップショット。日本語・改行・全角記号を含めておく(JSON の往復で壊れる箇所)
CITATIONS = [
    {
        "n": 1, "chunk_id": 41, "section_path": "返品ポリシー > 未開封の場合",
        "question": "未開封なら返品できますか？",
        "answer": "未開封であれば、到着後 7 日以内に限り返品を承ります。\n返送料はお客様のご負担です。",
    },
    {
        "n": 2, "chunk_id": 42, "section_path": "返品ポリシー > 開封済みの場合",
        "question": "開封してしまった商品は返品できますか？",
        "answer": "開封済みの商品は、初期不良の場合に限り交換で対応いたします。",
    },
    {
        "n": 3, "chunk_id": 7, "section_path": "配送 > お届け日数",
        "question": "何日で届きますか？",
        "answer": "本州は 2〜3 日、北海道・沖縄は 4〜5 日が目安です。",
    },
    {
        "n": 4, "chunk_id": 88, "section_path": "送料 > 無料条件",
        "question": "送料はいくらですか？",
        "answer": "5,000 円以上のご購入で送料無料です。\nそれ未満は一律 550 円を頂戴します。",
    },
]


async def _add(eval_id: str, **kw) -> dict:
    """台帳へ 1 件積む。指定しなかった項目はもっともらしい既定値で埋める。"""
    payload = {
        "bucket": "A_policy",
        "query": f"{eval_id} の問い: 開封した商品を返品したいのですが",
        "answer": f"{eval_id} の回答です。",
        "reason": "evidence に無い日数を答えている",
        "citations": CITATIONS,
        "judge_model": "gpt-4o-mini",
    }
    payload.update(kw)
    return await repository.upsert_faith_case(eval_id, **payload)


async def _rows() -> list[dict]:
    async with _client() as c:
        resp = await c.get("/api/rag-eval/faith-cases", params={"size": 100})
    assert resp.status_code == 200
    return resp.json()["rows"]


async def _row(eval_id: str) -> dict:
    rows = [r for r in await _rows() if r["eval_id"] == eval_id]
    assert len(rows) == 1, f"{eval_id} の行が {len(rows)} 件ある(1 問 1 行のはず)"
    return rows[0]


async def _set_status(case_id: int, status: str, resolution: str | None = None):
    body = {"status": status}
    if resolution is not None:
        body["resolution"] = resolution
    async with _client() as c:
        return await c.post(f"/api/rag-eval/faith-cases/{case_id}/status", json=body)


# ---------------------------------------------------------------------------
# 1 問 1 行 · 積み上がり
# ---------------------------------------------------------------------------


async def test_the_same_question_updates_the_row_instead_of_adding_one(db_session_factory):
    """同じ eval_id を 2 度積んでも行は増えず、最新の実行の内容になること。

    ここが行を増やす作りに戻ると、「同じ問いが 3 回幻覚になった」と
    「別々の問いが 3 件幻覚だった」が台帳の上で区別できなくなる。
    """
    first = await _add("A43", answer="1 回目の回答", reason="1 回目の理由",
                       strategy="hybrid", judge_model="judge-v1")
    assert first["created"] is True
    assert first["recurred"] is False
    assert first["seen_count"] == 1
    assert first["previous_status"] is None

    second = await _add("A43", bucket="B_model", query="2 回目に投げた問い",
                        answer="2 回目の回答", reason="2 回目の理由",
                        strategy="hybrid_rerank", judge_model="judge-v2")
    assert second["created"] is False
    assert second["recurred"] is False, "未対処のまま続いているだけなので再発ではない"
    assert second["seen_count"] == 2
    assert second["id"] == first["id"]

    async with db_session_factory() as s:
        n = (await s.execute(text("SELECT COUNT(*) FROM faith_cases"))).scalar_one()
    assert n == 1, "同じ問いで行が増えている"

    row = await _row("A43")
    assert row["answer"] == "2 回目の回答"
    assert row["reason"] == "2 回目の理由"
    assert row["query"] == "2 回目に投げた問い"
    assert row["bucket"] == "B_model"
    assert row["strategy"] == "hybrid_rerank"
    assert row["judge_model"] == "judge-v2"
    assert row["seen_count"] == 2


async def test_first_seen_is_kept_while_last_seen_moves(db_session_factory):
    """初回の時刻は動かさない。いつからの問題かが分からなくなる。"""
    await _add("A44")
    async with db_session_factory() as s:
        await s.execute(text(
            "UPDATE faith_cases SET first_seen_at='2026-01-02 03:04:05', "
            "last_seen_at='2026-01-02 03:04:05' WHERE eval_id='A44'"
        ))
        await s.commit()

    await _add("A44", answer="次の実行の回答")

    row = await _row("A44")
    assert row["first_seen_at"].startswith("2026-01-02T03:04:05")
    assert row["last_seen_at"] > row["first_seen_at"]


# ---------------------------------------------------------------------------
# 再発
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("closed", ["resolved", "no_action_needed"])
async def test_a_closed_case_returns_to_unresolved_when_judged_again(db_session_factory, closed):
    """対処済みの行が再び幻覚と判定されたら未対処へ戻り、再発として呼び出し元に伝わること。

    これを落とすと、直したつもりの問題が再発しても台帳は「解決済み」のまま静かに
    seen_count だけ増やす。作業一覧に浮上しないので、誰も気づけない。
    """
    created = await _add("B7", answer="配送は必ず翌日に届きます", reason="根拠に無い断定")
    note = "配送日数の表をナレッジに追加した" if closed == "resolved" else "judge の誤判定と判断した"
    resp = await _set_status(created["id"], closed, note)
    assert resp.status_code == 200
    before = resp.json()["case"]
    assert before["status"] == closed
    assert before["resolved_at"] is not None

    again = await _add("B7", answer="再発した回答", reason="また根拠に無い断定")

    assert again["recurred"] is True, "再発が呼び出し元に伝わっていない"
    assert again["previous_status"] == closed
    assert again["status"] == "unresolved"
    assert again["seen_count"] == 2

    row = await _row("B7")
    assert row["status"] == "unresolved", "対処済みのまま再発が埋もれている"
    assert row["answer"] == "再発した回答"
    # いつ一度片が付いたことになっていたかは残す(DDL のコメントの通り)
    assert row["resolved_at"] == before["resolved_at"]
    # 前回どう直したつもりだったかは、再発を調べる人が最初に読みたい情報なので残す
    assert row["resolution"] == note


async def test_recurrence_puts_the_case_back_on_the_unresolved_tab(db_session_factory):
    """再発した行が unresolved タブの一覧と件数に戻ってくること。"""
    created = await _add("B8")
    await _set_status(created["id"], "resolved", "ナレッジを直した")
    await _add("B8", answer="再発")

    async with _client() as c:
        body = (await c.get("/api/rag-eval/faith-cases", params={"status": "unresolved"})).json()
    assert [r["eval_id"] for r in body["rows"]] == ["B8"]
    assert body["counts"]["unresolved"] == 1
    assert body["counts"]["resolved"] == 0


# ---------------------------------------------------------------------------
# 一覧 · フィルタ · 件数
# ---------------------------------------------------------------------------


async def test_status_counts_ignore_the_filter_and_the_page(db_session_factory):
    """タブの件数は「いま表示している範囲」ではなく台帳全体の件数であること。

    フィルタ後の件数を出してしまうと、resolved タブを開いた瞬間に
    「未対処 0 件」と表示され、残っている作業が無いように見える。
    """
    ids = {e: (await _add(e))["id"] for e in ["C1", "C2", "C3", "C4", "C5"]}
    await _set_status(ids["C4"], "resolved", "ナレッジに条項を追加した")
    await _set_status(ids["C5"], "no_action_needed", "judge の誤判定")
    expected_counts = {"unresolved": 3, "resolved": 1, "no_action_needed": 1, "total": 5}

    async with _client() as c:
        whole = (await c.get("/api/rag-eval/faith-cases", params={"size": 100})).json()
        filtered = (await c.get(
            "/api/rag-eval/faith-cases", params={"status": "resolved", "size": 100}
        )).json()
        paged = (await c.get("/api/rag-eval/faith-cases", params={"page": 2, "size": 2})).json()

    assert whole["counts"] == expected_counts
    assert whole["total"] == 5

    assert filtered["counts"] == expected_counts, "タブの件数がフィルタに引きずられている"
    assert filtered["total"] == 1, "ページ送りの母数はフィルタ後の件数であること"
    assert [r["eval_id"] for r in filtered["rows"]] == ["C4"]
    assert filtered["status"] == "resolved"

    assert paged["counts"] == expected_counts, "タブの件数がページに引きずられている"
    assert paged["total"] == 5
    assert paged["page"] == 2 and paged["size"] == 2 and paged["pages"] == 3
    assert len(paged["rows"]) == 2


async def test_pagination_walks_every_row_exactly_once(db_session_factory):
    for e in ["D1", "D2", "D3", "D4", "D5"]:
        await _add(e)

    seen = []
    async with _client() as c:
        for page in (1, 2, 3, 4):
            body = (await c.get("/api/rag-eval/faith-cases", params={"page": page, "size": 2})).json()
            seen += [r["eval_id"] for r in body["rows"]]
    assert sorted(seen) == ["D1", "D2", "D3", "D4", "D5"]
    assert len(seen) == len(set(seen)), "ページをまたいで同じ行が二度出ている"


async def test_unresolved_comes_first_then_the_most_recent(db_session_factory):
    """並び順: 未対処が先 → last_seen_at の新しい順。作業一覧としての並びを固定する。"""
    ids = {e: (await _add(e))["id"] for e in ["E1", "E2", "E3"]}
    await _set_status(ids["E1"], "resolved", "対処済みにした")
    # last_seen_at は DATETIME(秒精度)なので、同じ秒に積むと引き分けになる。
    # 並び順そのものを見たいので明示的にずらす
    async with db_session_factory() as s:
        for eval_id, stamp in (("E1", "2026-03-03 10:00:00"), ("E2", "2026-03-01 10:00:00"),
                               ("E3", "2026-03-02 10:00:00")):
            await s.execute(
                text("UPDATE faith_cases SET last_seen_at=:t WHERE eval_id=:e"),
                {"t": stamp, "e": eval_id},
            )
        await s.commit()

    assert [r["eval_id"] for r in await _rows()] == ["E3", "E2", "E1"]


async def test_an_empty_ledger_is_not_an_error(db_session_factory):
    async with _client() as c:
        body = (await c.get("/api/rag-eval/faith-cases")).json()
    assert body["rows"] == []
    assert body["total"] == 0
    assert body["counts"] == {"unresolved": 0, "resolved": 0, "no_action_needed": 0, "total": 0}
    assert body["error"] is None


# ---------------------------------------------------------------------------
# 評価スクリプトが引く状態表
# ---------------------------------------------------------------------------


async def test_the_status_map_covers_every_row_of_the_ledger(db_session_factory):
    """eval_id -> status を 1 回の問い合わせで返すこと。

    評価スクリプトはこれを台帳の累計(hallucination.ledger)に使う。ページ送りの
    list_faith_cases と違い、**全行**が入っていなければ累計が実態より少なく出る。
    """
    a = await _add("A48")
    await _add("B48")
    await _set_status(a["id"], "resolved", "ナレッジに配送日数の表を追加した")

    got = await repository.faith_case_status_map()

    assert got == {"A48": "resolved", "B48": "unresolved"}


async def test_the_status_map_of_an_empty_ledger_is_empty(db_session_factory):
    assert await repository.faith_case_status_map() == {}


# ---------------------------------------------------------------------------
# 根拠スナップショット
# ---------------------------------------------------------------------------


async def test_the_evidence_snapshot_survives_the_round_trip(db_session_factory):
    """citations が順序ごとそのまま取り出せること。

    回答中の [n] はこのリストの n を指すので、順序が入れ替わると引用の指す先が
    静かにずれる。日本語と改行を含む本文で確かめる(JSON の往復で壊れる箇所)。
    """
    await _add("A45", citations=CITATIONS)

    row = await _row("A45")
    assert row["citations"] == CITATIONS
    assert [c["n"] for c in row["citations"]] == [1, 2, 3, 4]
    assert "\n" in row["citations"][0]["answer"]
    assert row["citations"][0]["section_path"] == "返品ポリシー > 未開封の場合"


async def test_a_new_run_replaces_the_snapshot(db_session_factory):
    """スナップショットも最新の実行のものになること(古い根拠で判定を見直さない)。"""
    await _add("A46", citations=CITATIONS)
    fresh = [{"n": 1, "chunk_id": 99, "section_path": "保証 > 期間",
              "question": "保証期間は？", "answer": "1 年間です。"}]
    await _add("A46", citations=fresh)
    assert (await _row("A46"))["citations"] == fresh


async def test_a_case_without_a_snapshot_is_allowed(db_session_factory):
    """citations は NULL 可。古い行やスナップショットを取れなかった実行のため。"""
    await _add("A47", citations=None)
    assert (await _row("A47"))["citations"] is None


# ---------------------------------------------------------------------------
# 状態の手動変更
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["resolved", "no_action_needed"])
@pytest.mark.parametrize("resolution", [None, "", "   ", "　　", " \t\n "])
async def test_closing_a_case_requires_a_note(db_session_factory, status, resolution):
    """対処メモ無しで片付けられないこと。空白のみ(全角スペース含む)も不正。

    メモの無い「解決済み」は、次に見た人には「本当に直したのか / 見なかったことに
    したのか」が区別できず、台帳としての価値が無い。
    """
    created = await _add("F1")
    resp = await _set_status(created["id"], status, resolution)

    assert resp.status_code == 400, f"{resolution!r} が対処メモとして通ってしまった"
    assert resp.json()["detail"]
    assert (await _row("F1"))["status"] == "unresolved", "400 なのに状態が変わっている"


@pytest.mark.parametrize("status", ["resolved", "no_action_needed"])
async def test_closing_a_case_records_the_note_and_the_time(db_session_factory, status):
    created = await _add("F2")
    resp = await _set_status(created["id"], status, "　ナレッジに返品条項を追加した　")

    assert resp.status_code == 200
    case = resp.json()["case"]
    assert case["status"] == status
    # 前後の空白は落として保存する(全角スペースも)
    assert case["resolution"] == "ナレッジに返品条項を追加した"
    assert case["resolved_at"] is not None
    assert case["id"] == created["id"]
    assert await _row("F2") == case, "返した行と一覧の行が食い違っている"


async def test_returning_to_unresolved_clears_the_note(db_session_factory):
    """人手で未対処へ戻したら対処メモと解決時刻を消すこと。

    人が「やはり直っていない」と言った以上、そのメモはもう有効ではない。
    再発(upsert_faith_case)で戻る場合とは扱いが違う。
    """
    created = await _add("F3")
    await _set_status(created["id"], "resolved", "直したつもりだった")

    resp = await _set_status(created["id"], "unresolved")

    assert resp.status_code == 200
    case = resp.json()["case"]
    assert case["status"] == "unresolved"
    assert case["resolution"] is None
    assert case["resolved_at"] is None
    assert await _row("F3") == case


async def test_returning_to_unresolved_needs_no_note(db_session_factory):
    """未対処へ戻すときはメモを求めない(送られてきても捨てる)。"""
    created = await _add("F4")
    await _set_status(created["id"], "no_action_needed", "誤判定と判断した")

    resp = await _set_status(created["id"], "unresolved", "気が変わった")

    assert resp.status_code == 200
    assert resp.json()["case"]["resolution"] is None


async def test_a_note_longer_than_the_column_is_rejected(db_session_factory):
    """VARCHAR(300) を超えるメモは 400。素通しすると MySQL の DataError が 500 になる。"""
    created = await _add("F5")
    resp = await _set_status(created["id"], "resolved", "あ" * 301)
    assert resp.status_code == 400
    assert (await _set_status(created["id"], "resolved", "あ" * 300)).status_code == 200


@pytest.mark.parametrize("body", [
    {"status": "done", "resolution": "直した"},          # ENUM に無い値
    {"status": "Resolved", "resolution": "直した"},      # 大文字違い
    {"status": "", "resolution": "直した"},              # 空
    {"status": None, "resolution": "直した"},            # null
    {"resolution": "直した"},                            # status が無い
])
async def test_an_unknown_status_is_rejected_by_the_schema(db_session_factory, body):
    """status の検証は pydantic の Literal に任せる(=422)。自前の if で 400 にしない。"""
    created = await _add("F6")
    async with _client() as c:
        resp = await c.post(f"/api/rag-eval/faith-cases/{created['id']}/status", json=body)
    assert resp.status_code == 422, f"{body} が 422 で弾かれていない"
    assert (await _row("F6"))["status"] == "unresolved"


async def test_an_unknown_case_id_is_a_404(db_session_factory):
    resp = await _set_status(999_999_999, "resolved", "存在しない行への操作")
    assert resp.status_code == 404
    assert resp.json()["detail"]


async def test_the_repository_reports_a_missing_case_as_none(db_session_factory):
    """API が 404 を返せるよう、repository 側は例外ではなく None で区別できること。"""
    assert await repository.set_faith_case_status(999_999_999, "unresolved") is None


# ---------------------------------------------------------------------------
# DB が落ちているとき
# ---------------------------------------------------------------------------


async def test_a_dead_database_does_not_take_down_the_eval_page(monkeypatch):
    """一覧は 500 にせず、空の一覧 + 理由で返す。counts は None(0 件とは違う)。"""

    async def boom(*args, **kwargs):
        raise OSError("MySQL に接続できません")

    monkeypatch.setattr(repository, "list_faith_cases", boom)
    async with _client() as c:
        resp = await c.get("/api/rag-eval/faith-cases")

    assert resp.status_code == 200
    body = resp.json()
    assert body["rows"] == []
    assert body["counts"] is None, "読めなかったのに 0 件として集計を出している"
    assert body["error"]


async def test_a_dead_database_makes_the_status_button_a_503(monkeypatch):
    """状態変更は黙って成功にしない。押した人に伝わる形(503)で失敗させる。"""
    from sqlalchemy.exc import OperationalError

    async def boom(*args, **kwargs):
        raise OperationalError("UPDATE faith_cases", {}, Exception("gone"))

    monkeypatch.setattr(repository, "set_faith_case_status", boom)
    resp = await _set_status(1, "resolved", "直した")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# ORM と実スキーマの一致
# ---------------------------------------------------------------------------


async def test_orm_columns_match_the_live_faith_cases_schema(_test_engine, db_clean):
    """ORM と DDL のずれをここで捕まえる(tests/test_models.py と同じ考え方)。"""
    async with _test_engine.connect() as conn:
        db_columns = set((await conn.execute(
            text("SELECT COLUMN_NAME FROM information_schema.columns "
                 "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'faith_cases'")
        )).scalars().all())
        column_type = (await conn.execute(
            text("SELECT COLUMN_TYPE FROM information_schema.columns "
                 "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'faith_cases' "
                 "AND COLUMN_NAME = 'status'")
        )).scalar_one()

    db_enum = tuple(re.findall(r"'([^']*)'", column_type))
    assert {c.name for c in FaithCase.__table__.columns} == db_columns
    assert tuple(FaithCase.__table__.columns["status"].type.enums) == db_enum
    assert repository.FAITH_STATUSES == db_enum
