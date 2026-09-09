"""/api/review(査読画面の API)のテスト。09 章のデータフライホイールの最後の一段。

**本物の DB(support_test)を使い、ナレッジベースへの書き込みだけ差し替える。**
DB を本物にするのは、ここで確かめたいことのほとんどが「行がどう変わったか」
だからで、代役の repository では「pending のまま残る」も「二度目は 409」も演技に
なってしまう。逆にナレッジベース側(MySQL の knowledge_chunks + 埋め込み上流 +
Milvus)は差し替える。本番の knowledge collection に触れないことと、テストが
上流へ課金を発生させないことの両方が要る。差し替え忘れは
_no_real_knowledge_base が落とす。

TestClient を使わないのは tests/test_kb_api.py と同じ理由: あれはアプリを別の
イベントループで回すため、conftest の _test_engine(セッションスコープのループに
紐付く)を使う経路が asyncmy の "attached to a different loop" で落ちる。

このモジュールが押さえる中心は 3 つ:

1. **書き戻しが済んでから status を動かす。** 逆順にすると、書き戻しが落ちた行が
   「承認済みなのにナレッジベースには無い」まま固定される。update_review_status は
   pending の行しか動かせないので、その行は二度と承認できない。
2. **ナレッジベースへ入るのは査読者が確定させた答え。** モデルの参考回答をそのまま
   書くなら、人の査読を挟む意味が無い。
3. **判定は一度きり。** pending 以外への操作は 409。承認と却下が後勝ちで
   上書きされると、消えた方の判断には誰も気づけない。
"""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import OperationalError

from app.api import review as review_api
from app.db import repository

pytestmark = pytest.mark.asyncio(loop_scope="session")

# router だけを載せた最小の app で叩く。app.main を import すると lifespan を持つ
# 本番の app が付いてきて、このモジュールの関心(エンドポイント単体の入出力)に
# 関係のない配線まで一緒に壊れる。本番 app へ登録されているかは
# tests/test_main_lifespan.py が見る(tests/test_actions_api.py と同じ作法)。
_app = FastAPI()
_app.include_router(review_api.router)

_Q_NORM = "猫用ベッドは水洗いに対応していますか"
_Q_RAW = "猫用ベッドって洗えますか"
_AI_ANSWER = "プラットフォームのアフターサービス規定をご確認ください"
_HUMAN_ANSWER = "カバーは取り外して手洗いできます。乾燥機は使わず陰干ししてください"
_REASON = "ユーザーが回答に 👎 を付けた"

# 09 章の確信度ゲートが残す形と同じ(app/core/confidence.py の snapshot_from_hits)。
_SNAPSHOT = [
    {"question": "ペット用品のお手入れ", "answer": "商品ごとの表示に従ってください",
     "rerank_score": 0.31, "section_path": "アフターサービス/お手入れ"},
]

# 空白のみと見なすべき入力。ASCII 空白だけでは足りない。U+3000(全角スペース)は
# 日本語 IME がそのまま出す「ありがちな空入力」。
# tests/test_actions_api.py と同じ一覧(新しい書き方を発明しない)。
_BLANK_VARIANTS = [
    pytest.param("   ", id="ascii-space"),
    pytest.param("　　", id="ideographic-space-u3000"),
    pytest.param("\t", id="tab"),
    pytest.param("\n", id="newline"),
    pytest.param(" ", id="nbsp"),
]


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=_app), base_url="http://reviewtest")


@pytest.fixture(autouse=True)
def _no_real_knowledge_base(monkeypatch):
    """既定でナレッジベースへの経路を全部塞ぐ。

    差し替えを忘れたテストが本番の Milvus collection へ書いたり、埋め込み上流を
    実際に叩いたりしないようにする。塞ぐのは 03 章の取り込み経路の入口 2 つと、
    Milvus の client 取得。
    """

    async def _boom_write(chunks):
        raise AssertionError("テストが本物のナレッジベースへ書き込もうとした")

    async def _boom_vectorize(*a, **k):
        raise AssertionError("テストが本物のベクトル化を走らせようとした")

    def _boom_client(*a, **k):
        raise AssertionError("テストが本物の Milvus へ接続しようとした")

    monkeypatch.setattr(review_api.dualwrite, "write_pending", _boom_write)
    monkeypatch.setattr(review_api.dualwrite, "vectorize_pending", _boom_vectorize)
    monkeypatch.setattr(review_api.milvus_client, "get_client", _boom_client)


@pytest.fixture
def stub_kb(monkeypatch) -> dict:
    """ナレッジベースへの書き戻しを記録用の偽物に差し替える。

    review_api の属性経由で差し替える。エンドポイントが dualwrite を直に呼ぶ形へ
    書き換わったら、_no_real_knowledge_base の方が落ちて気づける。
    """
    seen: dict = {"vectorized": False}

    async def _fake_write(chunks):
        seen["chunks"] = chunks
        return [101]

    async def _fake_vectorize():
        seen["vectorized"] = True
        return 1

    monkeypatch.setattr(review_api, "_write_chunks", _fake_write)
    monkeypatch.setattr(review_api, "_vectorize", _fake_vectorize)
    return seen


async def _seed(normalized: str = _Q_NORM, raw: str = _Q_RAW,
                snapshot: list | None = None) -> int:
    """穴を 1 件作り、そこへ生の質問を 1 件ぶら下げる(pipeline が作る形と同じ)。"""
    rid = await repository.insert_review_item(normalized, _AI_ANSWER)
    lcq = await repository.insert_low_confidence(
        None, raw, "user_feedback", _REASON,
        retrieved_chunks=_SNAPSHOT if snapshot is None else snapshot,
    )
    await repository.set_matched_review(lcq, rid)
    return rid


async def _status_of(rid: int) -> str:
    item, _ = await repository.get_review_detail(rid)
    return item.review_status


# ---------------------------------------------------------------------------
# 一覧
# ---------------------------------------------------------------------------


async def test_queue_lists_pending_by_occurrence_count_descending(db_session_factory):
    """よく来る穴が上。この並びがそのまま査読の優先順位になる。

    `/api/review/queue` が `/api/review/{id}` に食われていない
    (経路の宣言順を入れ替えると "queue" が id として解釈され 422 になる)ことも
    同時に見る。
    """
    rare = await _seed("配送日は指定できますか")
    common = await _seed(_Q_NORM)
    await repository.increment_occurrence(common)
    await repository.increment_occurrence(common)

    async with _client() as c:
        res = await c.get("/api/review/queue", params={"status": "pending"})

    assert res.status_code == 200
    items = res.json()["items"]
    assert [i["id"] for i in items] == [common, rare]
    assert items[0]["occurrence_count"] == 3
    assert items[0]["normalized_question"] == _Q_NORM
    assert items[0]["ai_suggested_answer"] == _AI_ANSWER
    assert items[0]["review_status"] == "pending"
    # 表示名は API が付ける。画面が識別子を訳し直すと表示名の出所が 2 つになる
    assert items[0]["status_label"] == "未対応"
    assert items[0]["created_at"]


async def test_queue_filters_by_status_and_returns_everything_without_it(db_session_factory):
    """status で絞れること、省略したら判定済みも含めて全件返ること。"""
    kept = await _seed("返品の送料は誰が負担しますか")
    dropped = await _seed("今日の天気は")
    assert await repository.update_review_status(dropped, "rejected")

    async with _client() as c:
        pending = (await c.get("/api/review/queue", params={"status": "pending"})).json()
        rejected = (await c.get("/api/review/queue", params={"status": "rejected"})).json()
        every = (await c.get("/api/review/queue")).json()

    assert [i["id"] for i in pending["items"]] == [kept]
    assert [i["id"] for i in rejected["items"]] == [dropped]
    assert rejected["items"][0]["status_label"] == "却下"
    assert {i["id"] for i in every["items"]} == {kept, dropped}


async def test_queue_reports_a_database_outage_without_leaking_the_cause(
        db_session_factory, monkeypatch):
    """読めなかったことを 500 の traceback ではなく、中立な文面で返す。"""

    async def _boom(status):
        raise OperationalError("SELECT * FROM review_queue", {},
                               Exception("root:secret@10.0.0.1:3306"))

    monkeypatch.setattr(review_api.repository, "list_review_queue", _boom)

    async with _client() as c:
        res = await c.get("/api/review/queue")

    assert res.status_code == 503
    detail = res.json()["detail"]
    # 接続文字列や SQL が画面まで届かないこと(app/api/actions.py と同じ規約)
    assert "secret" not in detail and "3306" not in detail


# ---------------------------------------------------------------------------
# 詳細
# ---------------------------------------------------------------------------


async def test_detail_carries_the_raw_questions_and_the_retrieval_snapshot(db_session_factory):
    """正規化後の 1 行だけでは承認の判断ができない。

    生の言い回しが要るのは、正規化が意味を削っていた場合(注文番号や条件が
    落ちた場合)に気づくため。検索の写しが要るのは「ナレッジベースに無いのか、
    有るのに引けていないのか」を見分けるため。
    """
    rid = await _seed()

    async with _client() as c:
        res = await c.get(f"/api/review/{rid}")

    assert res.status_code == 200
    body = res.json()
    assert body["normalized_question"] == _Q_NORM
    assert body["approved_answer"] is None
    assert len(body["raws"]) == 1
    raw = body["raws"][0]
    assert raw["raw_question"] == _Q_RAW
    assert raw["source"] == "user_feedback"
    assert raw["reason"] == _REASON
    assert raw["created_at"]
    assert raw["retrieved_chunks"][0]["rerank_score"] == 0.31
    assert raw["retrieved_chunks"][0]["section_path"] == "アフターサービス/お手入れ"


async def test_detail_does_not_invent_a_snapshot_for_entries_that_never_searched(
        db_session_factory):
    """検索を通らなかった経路の写しは null のまま渡す。

    [] へ丸めると画面は「引いたが 0 件だった」と読む。DDL はこの列の NULL を
    「検索を通っていない」の意味で使っている(app/db/models.py)。
    """
    rid = await repository.insert_review_item("有人対応をお願いできますか", None)
    lcq = await repository.insert_low_confidence(None, "人と話したい", "self_check", None)
    await repository.set_matched_review(lcq, rid)

    async with _client() as c:
        body = (await c.get(f"/api/review/{rid}")).json()

    assert body["ai_suggested_answer"] is None
    assert body["raws"][0]["retrieved_chunks"] is None
    assert body["raws"][0]["reason"] is None


async def test_detail_of_an_unknown_id_is_404(db_session_factory):
    async with _client() as c:
        assert (await c.get("/api/review/99999")).status_code == 404


# ---------------------------------------------------------------------------
# 承認
# ---------------------------------------------------------------------------


async def test_approve_writes_the_knowledge_base_then_flips_the_status(
        db_session_factory, stub_kb):
    """承認 = 03 章の取り込み経路へ流し、ベクトル化まで済ませてから status を動かす。"""
    rid = await _seed()

    async with _client() as c:
        res = await c.post(f"/api/review/{rid}/approve",
                           json={"approved_answer": _HUMAN_ANSWER})

    assert res.status_code == 200
    assert res.json() == {"ok": True, "chunk_ids": [101]}

    chunk = stub_kb["chunks"][0]
    assert chunk.questions == _Q_NORM
    # **査読者が確定させた答え**。モデルの参考回答をそのまま書くなら査読の意味が無い
    assert chunk.answer == _HUMAN_ANSWER
    assert chunk.answer != _AI_ANSWER
    assert chunk.content_type == "faq"
    # 出所が分かる固定値(app/kb/mining.py の会話由来 chunk と同じ扱い)。
    # section_path は文書内の位置であって質問の置き場ではない
    assert chunk.category == "フライホイール"
    assert chunk.section_path == "flywheel"
    # ベクトル化まで走ること。ここを飛ばすと knowledge_chunks には入るのに検索へ
    # 出てこない(受け入れ検証 3「承認したら次の質問で答えられる」が崩れる)
    assert stub_kb["vectorized"] is True

    async with _client() as c:
        body = (await c.get(f"/api/review/{rid}")).json()
    assert body["review_status"] == "approved"
    assert body["status_label"] == "承認済み"
    assert body["approved_answer"] == _HUMAN_ANSWER


async def test_approve_marks_a_key_clause_question_the_way_documents_does(
        db_session_factory, stub_kb):
    """is_key_clause は勝手な既定値ではなく documents.py と同じ判定器で決める。"""
    key = await _seed("返品の送料は誰が負担しますか")
    plain = await _seed("配送日は指定できますか")

    async with _client() as c:
        await c.post(f"/api/review/{key}/approve",
                     json={"approved_answer": "お客様都合の返品はお客様のご負担です"})
        assert stub_kb["chunks"][0].is_key_clause == 1
        await c.post(f"/api/review/{plain}/approve",
                     json={"approved_answer": "ご注文時にご指定いただけます"})
        assert stub_kb["chunks"][0].is_key_clause == 0


async def test_approve_of_an_unknown_id_is_404(db_session_factory, stub_kb):
    async with _client() as c:
        res = await c.post("/api/review/99999/approve",
                           json={"approved_answer": _HUMAN_ANSWER})
    assert res.status_code == 404
    assert "chunks" not in stub_kb


async def test_a_second_decision_is_rejected_with_409(db_session_factory, stub_kb):
    """判定は一度きり。承認済みの行への再操作は承認も却下も 409。

    後勝ちを許すと、消えた方の判断には誰も気づけない。承認はナレッジベースへ
    書き戻る操作なので、なおさら黙って上書きされてはいけない。
    """
    rid = await _seed()

    async with _client() as c:
        first = await c.post(f"/api/review/{rid}/approve",
                             json={"approved_answer": _HUMAN_ANSWER})
        again = await c.post(f"/api/review/{rid}/approve",
                             json={"approved_answer": "別の答え"})
        rejected = await c.post(f"/api/review/{rid}/reject")

    assert first.status_code == 200
    assert again.status_code == 409
    assert "承認済み" in again.json()["detail"]
    assert rejected.status_code == 409
    # 2 度目は書き戻しへ進まないこと(進むと同じ知識が二重に入る)
    assert stub_kb["chunks"][0].answer == _HUMAN_ANSWER
    assert await _status_of(rid) == "approved"


async def test_approve_keeps_the_item_pending_when_the_write_fails(
        db_session_factory, monkeypatch):
    """**書き戻しが落ちたら status は pending のまま。** そして押し直せる。

    先に approved にしてから書き戻す実装だと、ここで status が approved のまま
    固定される。update_review_status は pending の行しか動かせないので、その行は
    二度と承認できない = ナレッジベースに無いまま「承認済み」で埋もれる。
    """
    rid = await _seed()

    async def _boom(chunks):
        raise RuntimeError("embedding upstream unavailable")

    monkeypatch.setattr(review_api, "_write_chunks", _boom)

    written: dict = {}

    async def _ok(chunks):
        written["chunks"] = chunks
        return [7]

    async def _vec():
        written["vectorized"] = True
        return 1

    async with _client() as c:
        res = await c.post(f"/api/review/{rid}/approve",
                           json={"approved_answer": _HUMAN_ANSWER})
        assert res.status_code == 502
        assert await _status_of(rid) == "pending"

        # やり直せること。ここまで確かめないと「pending のまま」が
        # 「再挑戦できる」を意味していることの証明にならない
        monkeypatch.setattr(review_api, "_write_chunks", _ok)
        monkeypatch.setattr(review_api, "_vectorize", _vec)
        retry = await c.post(f"/api/review/{rid}/approve",
                             json={"approved_answer": _HUMAN_ANSWER})

    assert retry.status_code == 200 and retry.json()["chunk_ids"] == [7]
    assert written["vectorized"] is True
    assert await _status_of(rid) == "approved"


async def test_approve_keeps_the_item_pending_when_only_the_vectorization_fails(
        db_session_factory, monkeypatch):
    """MySQL へは入ったがベクトル化が落ちた場合も pending のまま。

    ここで approved にすると、検索に出てこない知識を「承認済み」として片付けた
    ことになる。knowledge_chunks 側の行は status=pending で残るので、
    再実行(押し直し / make kb-vectorize)が拾う。
    """
    rid = await _seed()

    async def _ok(chunks):
        return [11]

    async def _boom():
        raise RuntimeError("Milvus offline: root:secret@10.0.0.1:19530")

    monkeypatch.setattr(review_api, "_write_chunks", _ok)
    monkeypatch.setattr(review_api, "_vectorize", _boom)

    async with _client() as c:
        res = await c.post(f"/api/review/{rid}/approve",
                           json={"approved_answer": _HUMAN_ANSWER})

    assert res.status_code == 502
    assert "secret" not in res.json()["detail"]
    assert await _status_of(rid) == "pending"


async def test_approve_reports_409_when_someone_else_decided_first(
        db_session_factory, stub_kb, monkeypatch):
    """詳細を読んでから書き込むまでの間に、別の査読者が判定を確定させた場合。

    update_review_status は pending の行しか動かせないので rowcount が 0 になる。
    黙って 200 を返すと、画面には自分の答えが入ったように見えるのに、保存されて
    いるのは相手の判定になる。
    """
    rid = await _seed()

    async def _lost(review_id, status, approved_answer=None):
        return False

    monkeypatch.setattr(review_api.repository, "update_review_status", _lost)

    async with _client() as c:
        res = await c.post(f"/api/review/{rid}/approve",
                           json={"approved_answer": _HUMAN_ANSWER})

    assert res.status_code == 409
    # 書き戻しは済んでいる。その事実を隠さない文面になっていること
    assert stub_kb["chunks"]
    assert "書き戻し" in res.json()["detail"]


@pytest.mark.parametrize("answer", _BLANK_VARIANTS)
async def test_approve_rejects_a_blank_answer(db_session_factory, stub_kb, answer):
    """空の答えを書き戻すと「根拠は引けるのに中身が無い」chunk が当たる。

    確信度ゲートは根拠の有無しか見ないのでそのまま通過し、空の答えが自信を持って
    返る。min_length=1 は空白のみを通すので validator で弾く。
    """
    rid = await _seed()

    async with _client() as c:
        res = await c.post(f"/api/review/{rid}/approve", json={"approved_answer": answer})

    assert res.status_code == 422
    assert "chunks" not in stub_kb
    assert await _status_of(rid) == "pending"


async def test_approve_requires_the_answer_field(db_session_factory, stub_kb):
    rid = await _seed()
    async with _client() as c:
        assert (await c.post(f"/api/review/{rid}/approve", json={})).status_code == 422
    assert "chunks" not in stub_kb
    assert await _status_of(rid) == "pending"


# ---------------------------------------------------------------------------
# 却下
# ---------------------------------------------------------------------------


async def test_reject_flips_the_status_and_writes_nothing(db_session_factory):
    """却下はナレッジベースに触れない(触れば _no_real_knowledge_base が落とす)。"""
    rid = await _seed()

    async with _client() as c:
        res = await c.post(f"/api/review/{rid}/reject")
        assert res.status_code == 200 and res.json() == {"ok": True}
        body = (await c.get(f"/api/review/{rid}")).json()
        assert body["review_status"] == "rejected"
        assert body["status_label"] == "却下"
        # 却下は最終判断。復活の経路は用意しない(spec §4)
        again = await c.post(f"/api/review/{rid}/approve",
                             json={"approved_answer": _HUMAN_ANSWER})

    assert again.status_code == 409
    assert await _status_of(rid) == "rejected"


async def test_reject_of_an_unknown_id_is_404_not_409(db_session_factory):
    """行が無いのか、既に判定済みなのかを取り違えないこと。

    UPDATE の rowcount だけでは区別できないので、動かせなかったときにだけ読みに行く。
    """
    async with _client() as c:
        assert (await c.post("/api/review/99999/reject")).status_code == 404


# ---------------------------------------------------------------------------
# 契約
# ---------------------------------------------------------------------------


async def test_the_endpoints_are_published_in_the_openapi_schema():
    """画面側がこの契約を読む。パスと必須項目が変わったら気づけるようにする。

    DB を触らないが async にしてあるのは、モジュール先頭の pytestmark が
    同期テストにも付いて PytestWarning になるため(tests/test_kb_api.py の
    docstring と同じ話。マーカーの付け方をこのモジュールで割らない)。
    """
    schema = _app.openapi()
    paths = schema["paths"]
    assert "/api/review/queue" in paths
    assert "/api/review/{review_id}" in paths
    assert "/api/review/{review_id}/approve" in paths
    assert "/api/review/{review_id}/reject" in paths

    ref = (paths["/api/review/{review_id}/approve"]["post"]["requestBody"]
           ["content"]["application/json"]["schema"]["$ref"])
    body = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    assert body["required"] == ["approved_answer"]
    assert body["properties"]["approved_answer"]["minLength"] == 1
