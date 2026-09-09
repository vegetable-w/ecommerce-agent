"""データフライホイールの pipeline (09 Task 8)。**本物の DB を使い、モデルだけ差し替える。**

DB を本物にするのは、ここで確かめたいことがほぼ全部「行がどう変わったか」だからで、
代役の repository では「rejected のまま」も「カーソルが進んでいない」も演技になる。
逆にモデルは差し替える。正規化と重複判定そのものの品質は
tests/data/flywheel_samples.json のラベル付きサンプルで別途測る(実際の上流を叩くので
ここには置かない)。

_test_engine はセッションスコープのイベントループ上に作られるため、このモジュールも
同じループで動かす(tests/conftest.py の docstring 参照)。
"""

import logging

import pytest

from app.core import flywheel
from app.core.flywheel import NormalizeResult
from app.db import repository

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _stub_llm(monkeypatch, results):
    """用意した結果を順に返す代役。Exception を混ぜるとその行で上流が落ちた状況になる。

    normalize_and_match ごと差し替えるので、prompt も get_chat_model も通らない
    (このモジュールが実際の上流を叩くことは無い)。
    """
    queue = list(results)
    seen: list[list[dict]] = []

    async def fake(raw_question, candidates):
        seen.append(list(candidates))
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(flywheel, "normalize_and_match", fake)
    return seen


async def test_new_question_creates_pending_item(db_session_factory, monkeypatch):
    """新しい質問は pending の穴を 1 件作り、カーソルが進んで次回は拾われないこと。"""
    await repository.insert_low_confidence(
        None, "自動ペット給水器って水洗いできますか", "retrieval_low_conf", None
    )
    _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="自動ペット給水器は水洗いに対応していますか",
        matched_question_id=None,
        ai_suggested_answer="プラットフォームのアフターサービス規定をご確認ください。")])

    stats = await flywheel.process_pending()

    assert stats == {"processed": 1, "merged": 0, "created": 1, "skipped": 0}
    rows = await repository.list_review_queue("pending")
    assert len(rows) == 1
    assert rows[0].normalized_question == "自動ペット給水器は水洗いに対応していますか"
    assert rows[0].occurrence_count == 1
    assert rows[0].ai_suggested_answer == "プラットフォームのアフターサービス規定をご確認ください。"
    # カーソルが進んでいる。ここを見ないと、同じ質問が毎回処理されて穴が増え続ける
    # 状態(課金され続け、レビュー待ちが重複で埋まる)に気づけない。
    assert await repository.fetch_unmatched_low_conf(10) == []


async def test_empty_suggested_answer_is_stored_as_null(db_session_factory, monkeypatch):
    """参考回答が空なら列は NULL。空文字を入れると「回答例あり」と区別が付かなくなる。"""
    await repository.insert_low_confidence(None, "領収書はもらえますか", "self_check", None)
    _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="領収書を発行してもらえますか",
        matched_question_id=None, ai_suggested_answer="")])

    await flywheel.process_pending()

    assert (await repository.list_review_queue(None))[0].ai_suggested_answer is None


async def test_match_merges_and_never_revives(db_session_factory, monkeypatch):
    """match は出現回数を +1 するだけで、rejected を pending へ戻さないこと。

    却下は人が下した最終判断。同じ質問がまた来たことを理由に復活させると、
    一度「これはナレッジにしない」と決めた穴が延々とレビュー待ちへ戻ってくる。
    それでも回数は数える(却下の判断を見直す材料になる)。
    """
    rid = await repository.insert_review_item("自動ペット給水器は水洗いに対応していますか", None)
    await repository.update_review_status(rid, "rejected")
    await repository.insert_low_confidence(
        None, "自動ペット給水器って洗えますか", "user_feedback", None
    )
    _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="自動ペット給水器は水洗いに対応していますか",
        matched_question_id=rid, ai_suggested_answer="")])

    stats = await flywheel.process_pending()

    assert stats == {"processed": 1, "merged": 1, "created": 0, "skipped": 0}
    item, raws = await repository.get_review_detail(rid)
    assert item.occurrence_count == 2
    assert item.review_status == "rejected"
    # 行そのものが増えていないこと(重複排除の要はここ)
    assert len(await repository.list_review_queue(None)) == 1
    # まとめ先が生の質問側へ書き戻っている。レビュー画面は元の言い回しをここから引く
    assert [r.raw_question for r in raws] == ["自動ペット給水器って洗えますか"]


async def test_match_into_approved_item_also_only_counts(db_session_factory, monkeypatch):
    """approved でも同じ。status は動かさず回数だけ進むこと。

    rejected だけを特別扱いする実装(「rejected 以外なら status を戻す」)になっていない
    ことを見る。承認済みの穴を pending へ戻すと、ナレッジベースへ書き戻し済みの答えが
    未処理として二重に取り込まれる。
    """
    rid = await repository.insert_review_item("送料はいくらかかりますか", "3,000円以上で無料です。")
    await repository.update_review_status(rid, "approved", "3,000円以上のご注文は送料無料です。")
    await repository.insert_low_confidence(None, "送料って結局いくら", "self_check", None)
    _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="送料はいくらかかりますか",
        matched_question_id=rid, ai_suggested_answer="")])

    await flywheel.process_pending()

    item, _ = await repository.get_review_detail(rid)
    assert item.review_status == "approved"
    assert item.approved_answer == "3,000円以上のご注文は送料無料です。"
    assert item.occurrence_count == 2


async def test_hallucinated_id_skips_and_keeps_cursor(db_session_factory, monkeypatch):
    """候補に無い id が返ったら、その行を飛ばしてカーソルを残すこと。

    存在しない id をそのまま信じると、生の質問が無関係な穴(または存在しない穴)へ
    紐づき、カーソルが進むので二度と拾い直せない。
    """
    await repository.insert_low_confidence(
        None, "自動ペット給水器って洗えますか", "self_check", None
    )
    _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="自動ペット給水器は水洗いに対応していますか",
        matched_question_id=99999, ai_suggested_answer="")])

    stats = await flywheel.process_pending()

    assert stats == {"processed": 0, "merged": 0, "created": 0, "skipped": 1}
    assert len(await repository.fetch_unmatched_low_conf(10)) == 1  # 次回やり直す
    assert await repository.list_review_queue(None) == []           # 穴も作らない


async def test_id_outside_the_shown_candidates_is_also_skipped(db_session_factory, monkeypatch):
    """実在する id でも、その回に見せた候補の外なら信じないこと。

    存在確認を「DB にその行があるか」で済ませると通ってしまう。モデルは見せていない
    穴と比べようがないので、上限で切り捨てられた古い穴の id が返ってきたとしても、
    それは突き合わせの結果ではなく当てずっぽうでしかない。
    """
    monkeypatch.setattr(flywheel, "CANDIDATE_LIMIT", 1)
    shown = await repository.insert_review_item("最近の穴", None)
    hidden = await repository.insert_review_item("古い穴", None)
    # updated_at 降順なので、あとから触った shown 側だけが候補に残る
    await repository.increment_occurrence(shown)
    await repository.insert_low_confidence(None, "質問", "self_check", None)
    seen = _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="標準化", matched_question_id=hidden, ai_suggested_answer="")])

    stats = await flywheel.process_pending()

    assert [c["id"] for c in seen[0]] == [shown]  # hidden は見せていない
    assert stats["skipped"] == 1 and stats["processed"] == 0
    item, raws = await repository.get_review_detail(hidden)
    assert item.occurrence_count == 1 and raws == []
    assert len(await repository.fetch_unmatched_low_conf(10)) == 1


async def test_llm_failure_skips_row_and_continues(db_session_factory, monkeypatch):
    """1 行の失敗でバッチを止めないこと。後続の行は普通に処理される。

    途中で例外が抜けると、その先の行は今回まったく処理されない。定期実行なので
    次回また同じ行で落ちれば、後ろの質問は永久にレビューへ上がってこない。
    """
    await repository.insert_low_confidence(None, "質問1", "self_check", None)
    await repository.insert_low_confidence(None, "質問2", "self_check", None)
    _stub_llm(monkeypatch, [
        RuntimeError("JSON として読めない応答"),
        NormalizeResult(normalized_question="質問2を標準化したもの",
                        matched_question_id=None, ai_suggested_answer=""),
    ])

    stats = await flywheel.process_pending()

    assert stats == {"processed": 1, "merged": 0, "created": 1, "skipped": 1}
    rows = await repository.list_review_queue(None)
    assert [r.normalized_question for r in rows] == ["質問2を標準化したもの"]
    # 失敗した方だけがカーソルに残る
    assert [r.raw_question for r in await repository.fetch_unmatched_low_conf(10)] == ["質問1"]


async def test_same_batch_duplicates_merge_serially(db_session_factory, monkeypatch):
    """同じバッチの同義 2 件が 1 件にまとまること。

    2 件目の候補一覧に、1 件目でたった今作った穴が入っていることまで見る。候補を
    バッチの先頭で 1 度だけ取る実装や、行を並列に処理する実装だと 2 件目の候補は
    空のままになり、同じ穴が 2 行できる。
    """
    await repository.insert_low_confidence(
        None, "自動ペット給水器は水洗いできますか", "retrieval_low_conf", None
    )
    await repository.insert_low_confidence(
        None, "自動ペット給水器って洗えますか", "user_feedback", None
    )
    seen: list[list[dict]] = []

    async def fake(raw_question, candidates):
        seen.append(list(candidates))
        # 候補があればそれと同じ intent とみなす、という単純な代役。
        # 候補が渡っていなければ新規になるので、直列処理が壊れれば 2 行できる。
        return NormalizeResult(
            normalized_question="自動ペット給水器は水洗いに対応していますか",
            matched_question_id=candidates[0]["id"] if candidates else None,
            ai_suggested_answer="")

    monkeypatch.setattr(flywheel, "normalize_and_match", fake)

    stats = await flywheel.process_pending()

    assert stats == {"processed": 2, "merged": 1, "created": 1, "skipped": 0}
    assert seen[0] == []
    assert len(seen[1]) == 1  # 1 件目で作った穴が 2 件目の候補に入っている
    rows = await repository.list_review_queue(None)
    assert len(rows) == 1
    assert rows[0].occurrence_count == 2
    # 生の質問 2 件が同じ穴に紐づいている
    _, raws = await repository.get_review_detail(rows[0].id)
    assert len(raws) == 2


async def test_limit_caps_the_batch(db_session_factory, monkeypatch):
    """limit を超える行は今回処理せず、カーソルに残すこと(古い順に消化する)。"""
    for i in range(3):
        await repository.insert_low_confidence(None, f"質問{i}", "self_check", None)
    _stub_llm(monkeypatch, [
        NormalizeResult(normalized_question=f"標準化{i}", matched_question_id=None,
                        ai_suggested_answer="") for i in range(2)
    ])

    stats = await flywheel.process_pending(limit=2)

    assert stats["processed"] == 2
    assert [r.raw_question for r in await repository.fetch_unmatched_low_conf(10)] == ["質問2"]


async def test_warns_once_when_candidates_hit_the_cap(db_session_factory, monkeypatch, caplog):
    """候補が上限で頭打ちになったら警告を残すこと(バッチにつき 1 回)。

    上限で切られている間は、切り捨てられた古い穴と同じ質問が来ても match できず、
    重複した穴が黙って積まれる。ログに出ないと、レビュー待ちが重複で膨らんでいる
    ことに誰も気づけない。
    """
    monkeypatch.setattr(flywheel, "CANDIDATE_LIMIT", 2)
    for q in ["穴A", "穴B"]:
        await repository.insert_review_item(q, None)
    for i in range(2):
        await repository.insert_low_confidence(None, f"質問{i}", "self_check", None)
    _stub_llm(monkeypatch, [
        NormalizeResult(normalized_question=f"標準化{i}", matched_question_id=None,
                        ai_suggested_answer="") for i in range(2)
    ])

    with caplog.at_level(logging.WARNING, logger="app.core.flywheel"):
        await flywheel.process_pending()

    hits = [r for r in caplog.records if "頭打ち" in r.getMessage()]
    assert len(hits) == 1
    assert "上限 2 件" in hits[0].getMessage()


async def test_candidates_are_limited_to_the_cap(db_session_factory, monkeypatch):
    """モデルへ渡す候補は CANDIDATE_LIMIT 件まで。既定任せにして食い違わせないこと。"""
    monkeypatch.setattr(flywheel, "CANDIDATE_LIMIT", 2)
    for q in ["穴A", "穴B", "穴C"]:
        await repository.insert_review_item(q, None)
    await repository.insert_low_confidence(None, "質問", "self_check", None)
    seen = _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="標準化", matched_question_id=None, ai_suggested_answer="")])

    await flywheel.process_pending()

    assert len(seen[0]) == 2


async def test_empty_pool_is_a_no_op(db_session_factory):
    """未処理が 0 件なら上流を 1 度も呼ばず、素直に 0 件を返すこと(定期実行の既定の姿)。"""
    assert await flywheel.process_pending() == {
        "processed": 0, "merged": 0, "created": 0, "skipped": 0
    }


async def test_prompt_renders_candidates_and_raw_question():
    """候補一覧と生の質問が prompt に載ること。

    id が載らなければモデルは id を返しようがなく、重複判定が常に新規に倒れる。
    (このモジュールは pytestmark で asyncio が付くので、同期関数にすると警告になる)
    """
    from app.core.prompts import FLYWHEEL_NORMALIZE_PROMPT

    msgs = FLYWHEEL_NORMALIZE_PROMPT.format_messages(
        candidates="- id=7: 送料はいくらかかりますか", raw_question="送料って結局いくら"
    )
    assert msgs[0].type == "system"
    assert "- id=7: 送料はいくらかかりますか" in msgs[-1].content
    assert "送料って結局いくら" in msgs[-1].content


async def test_candidate_list_is_rendered_with_ids(monkeypatch):
    """候補が id 付きの行で組まれ、空なら「候補なし」と明示して渡されること。

    空のときに空文字を渡すと、prompt の候補欄が丸ごと消えて指示の対象が無いように読め、
    モデルが id を捏造しやすくなる(その捏造は process_pending 側で skip されるので、
    バッチが黙って何も進めない形に化ける)。
    """
    seen = {}

    class _FakeChain:
        async def ainvoke(self, params):
            seen.update(params)
            return NormalizeResult(normalized_question="標準化")

    monkeypatch.setattr(flywheel, "_chain", lambda: _FakeChain())

    await flywheel.normalize_and_match("送料は", [
        {"id": 7, "normalized_question": "送料はいくらかかりますか"},
        {"id": 9, "normalized_question": "返金はいつ反映されますか"},
    ])
    assert seen["candidates"] == (
        "- id=7: 送料はいくらかかりますか\n- id=9: 返金はいつ反映されますか"
    )

    await flywheel.normalize_and_match("送料は", [])
    assert seen["candidates"] == "(候補なし)"
