"""データフライホイールのデータ層(09 Task 1)。

低信頼プール → 重複排除 → レビュー待ち、という閉ループの各段が実際に DB へ落ちて
いることを見る。以降の Task(pipeline / レビュー API / 評価パイプライン)は全部
ここの signature に乗るので、戻り値の形まで固定する。

_test_engine はセッションスコープのイベントループ上に作られるため、このモジュールも
同じループで動かす(tests/conftest.py の docstring 参照)。
"""

import pytest
from sqlalchemy import text

from app.db import repository

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_low_confidence_keeps_the_retrieval_snapshot(db_session_factory):
    """投入時の検索結果の写しが JSON として残り、未処理として引けること。

    写しが落ちると、レビューする人は「何を引いた結果として答えられなかったのか」を
    再現できない(その後ナレッジベースが更新されれば、同じ質問を引き直しても
    当時の検索結果は二度と出ない)。
    """
    snapshot = [
        {"chunk_id": 11, "score": 0.31, "text": "返品は到着後7日以内に承ります。"},
        {"chunk_id": 12, "score": 0.22, "text": "送料はお客様のご負担となります。"},
    ]
    lid = await repository.insert_low_confidence(
        None, "開封済みの化粧品は返品できますか？", "retrieval_low_conf", "top=0.310", snapshot
    )

    rows = await repository.fetch_unmatched_low_conf(10)
    got = [r for r in rows if r.id == lid]
    assert len(got) == 1
    assert got[0].matched_review_id is None
    assert got[0].retrieved_chunks == snapshot
    assert got[0].retrieved_chunks[0]["score"] == 0.31  # JSON として往復している


async def test_low_confidence_snapshot_defaults_to_null(db_session_factory):
    """写しを渡さない既存の呼び方(4 引数)がそのまま通り、列は NULL のままであること。

    04 章の回答拒否パスは検索を通らない経路からも呼ぶ。ここが必須引数になると、
    その経路が丸ごと落ちる。
    """
    lid = await repository.insert_low_confidence(
        None, "領収書はもらえますか？", "self_check", "根拠不足"
    )

    async with db_session_factory() as s:
        got = (
            await s.execute(
                text("SELECT retrieved_chunks FROM low_confidence_questions WHERE id=:i"),
                {"i": lid},
            )
        ).scalar()
    assert got is None


async def test_review_item_counts_occurrences(db_session_factory):
    """穴を 1 件積み、候補一覧に出て、同じ穴が来るたびに行が増えずに数だけ進むこと。

    重複排除の要は「行を増やさない」ことなので、出現回数が進むだけでなく
    穴そのものは 1 件のままであることまで見る。
    """
    rid = await repository.insert_review_item(
        "開封済みの化粧品の返品可否", "衛生商品のため開封後の返品は承っておりません。"
    )

    candidates = await repository.list_review_candidates()
    assert {"id": rid, "normalized_question": "開封済みの化粧品の返品可否"} in candidates
    # 突き合わせに要るのは id と質問文だけ。余計な列を積むと prompt が無駄に太る。
    assert set(candidates[0]) == {"id", "normalized_question"}

    await repository.increment_occurrence(rid)
    await repository.increment_occurrence(rid)

    detail = await repository.get_review_detail(rid)
    assert detail is not None
    item, raws = detail
    assert item.occurrence_count == 3  # 初期値 1 + 2 回
    assert item.review_status == "pending"
    assert raws == []  # まだ生の質問を紐づけていない
    assert len(await repository.list_review_queue(None)) == 1


async def test_matching_moves_the_question_out_of_the_unprocessed_pool(db_session_factory):
    """紐づけると詳細に出てきて、未処理の一覧からは消えること。

    この 2 つは同じ 1 列(matched_review_id)の裏表で、片方だけ動くと
    pipeline が同じ質問を毎回拾い直すか、逆に拾ったまま迷子になる。
    """
    lid = await repository.insert_low_confidence(
        None, "化粧品って開けちゃったら返せない？", "retrieval_low_conf", "top=0.290"
    )
    rid = await repository.insert_review_item("開封済みの化粧品の返品可否", None)

    assert lid in [r.id for r in await repository.fetch_unmatched_low_conf(10)]

    await repository.set_matched_review(lid, rid)

    item, raws = await repository.get_review_detail(rid)
    assert item.id == rid
    assert [r.raw_question for r in raws] == ["化粧品って開けちゃったら返せない？"]
    assert lid not in [r.id for r in await repository.fetch_unmatched_low_conf(10)]


async def test_get_review_detail_returns_none_for_unknown_id(db_session_factory):
    assert await repository.get_review_detail(999_999_999) is None


async def test_review_status_moves_only_out_of_pending(db_session_factory):
    """確定できるのは pending の行だけ。承認済み / 却下済みの行は動かせないこと。

    ここが緩むと、承認して 03 章の取り込み経路でナレッジベースへ書き戻したあとに、
    別の人の却下が上書きして「却下したのに知識には載っている」状態を作れてしまう。
    """
    approved = await repository.insert_review_item("承認される穴", None)
    rejected = await repository.insert_review_item("却下される穴", None)

    assert await repository.update_review_status(approved, "approved", "開封後は不可です。")
    assert await repository.update_review_status(rejected, "rejected")

    # 二度目は通らない。承認済みを却下へ倒すのも、却下済みを承認へ倒すのも同じ。
    assert not await repository.update_review_status(approved, "rejected")
    assert not await repository.update_review_status(rejected, "approved", "やっぱり可です。")
    # 存在しない id も False(例外にしない)。
    assert not await repository.update_review_status(999_999_999, "approved")

    item, _ = await repository.get_review_detail(approved)
    assert item.review_status == "approved"
    assert item.approved_answer == "開封後は不可です。"
    item, _ = await repository.get_review_detail(rejected)
    assert item.review_status == "rejected"


async def test_list_review_queue_filters_by_status_and_sorts_by_count(db_session_factory):
    """出現回数の多い穴が上に来ること、status で絞れること。

    レビューは全部は捌けない前提の作業なので、並び順が壊れると
    「よく来る穴」が下に沈んで永久に処理されない。
    """
    rare = await repository.insert_review_item("たまに来る穴", None)
    common = await repository.insert_review_item("よく来る穴", None)
    for _ in range(3):
        await repository.increment_occurrence(common)

    assert [r.id for r in await repository.list_review_queue(None)] == [common, rare]

    await repository.update_review_status(rare, "approved", "こう答えます。")
    assert [r.id for r in await repository.list_review_queue("pending")] == [common]
    assert [r.id for r in await repository.list_review_queue("approved")] == [rare]


async def test_review_queue_ties_are_broken_by_id(db_session_factory):
    """同点の並びを固定する。**穴のほとんどは 1 件のまま**なので同点が普通。

    第 1 キーだけだと並びが実行ごとに変わり、査読キューを開き直すたびに順番が
    入れ替わる(どこまで見たかが分からなくなる)。list_eval_runs と同じ作法。
    """
    a = await repository.insert_review_item("同点の穴 A", None)
    b = await repository.insert_review_item("同点の穴 B", None)
    c = await repository.insert_review_item("同点の穴 C", None)

    first = [r.id for r in await repository.list_review_queue("pending")]
    again = [r.id for r in await repository.list_review_queue("pending")]
    assert first == again
    assert [i for i in first if i in (a, b, c)] == [c, b, a]


async def test_low_confidence_exists_is_scoped_to_the_conversation(db_session_factory):
    """同じ会話の同じ質問だけを「既にある」と答えること。

    会話をまたいで潰すと、別の会話で同じことを訊かれた事実(= その穴が何度も
    来ていること)まで消える。source は見ない: プールの 1 行は「直すべき質問」1 件で
    あって、断り方の記録ではない。
    """
    conv = await repository.create_conversation("u-dup")
    other = await repository.create_conversation("u-dup2")
    q = "領収書の再発行はできますか"
    await repository.insert_low_confidence(conv, q, "retrieval_low_conf", "top=0.100")

    assert await repository.low_confidence_exists(conv, q) is True
    assert await repository.low_confidence_exists(other, q) is False
    assert await repository.low_confidence_exists(conv, "別の質問") is False


async def test_eval_runs_are_listed_newest_first(db_session_factory):
    """評価の実行結果が JSON ごと残り、新しい順に並ぶこと。"""
    first = await repository.insert_eval_run("scheduled", 40, {"recall_at_k": 0.71})
    second = await repository.insert_eval_run(
        "manual", 42, {"recall_at_k": 0.82, "mrr": 0.66, "faithfulness": 0.90}
    )

    runs = await repository.list_eval_runs()
    assert [r.id for r in runs] == [second, first]
    assert runs[0].triggered_by == "manual"
    assert runs[0].dataset_size == 42
    assert runs[0].metrics == {"recall_at_k": 0.82, "mrr": 0.66, "faithfulness": 0.90}
    assert runs[1].triggered_by == "scheduled"

    assert [r.id for r in await repository.list_eval_runs(limit=1)] == [second]
