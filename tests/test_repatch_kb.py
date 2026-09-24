"""md の差分をどう当てるかの判定を固定する。

ここが critical path である理由: 判定を間違えると「変わっていない chunk を
全部埋め込み直す(課金が無駄に増える)」か「変わった chunk を据え置く(直したはずの
本文が検索に出てこない)」のどちらかになる。しかもどちらも例外を出さず、
それらしい件数を表示して完走する。

DB にも Milvus にも上流にも触れない。_plan は純粋な突き合わせだけを行う。
"""

import pytest

from app.kb.documents import Chunk
from scripts import repatch_kb as rp


class _Row:
    """knowledge_chunks の 1 行のうち、突き合わせに使う項目だけを持つ替え玉。"""

    def __init__(self, id, section_path, questions, answer, content_type="faq"):
        self.id = id
        self.section_path = section_path
        self.questions = questions
        self.answer = answer
        self.content_type = content_type


def _chunk(section_path, questions, answer, content_type="faq"):
    return ("カテゴリ", Chunk(category="カテゴリ", questions=questions, answer=answer,
                             section_path=section_path, content_type=content_type))


def _names(plan):
    return {
        "same": [r.section_path for r in plan["same"]],
        "changed": [r.section_path for r, _, _ in plan["changed"]],
        "added": [c.section_path for _, c in plan["added"]],
        "removed": [r.section_path for r in plan["removed"]],
    }


def test_untouched_chunks_are_left_alone():
    """本文が同じなら据え置く。ここが壊れると毎回全件を埋め込み直す。"""
    rows = [_Row(1, "FAQ / 送料", "送料は", "3,000円以上で無料")]
    fresh = [_chunk("FAQ / 送料", "送料は", "3,000円以上で無料")]
    assert _names(rp._plan(rows, fresh)) == {
        "same": ["FAQ / 送料"], "changed": [], "added": [], "removed": []}


def test_a_changed_body_is_updated_in_place():
    """本文が変われば同じ id を書き換える。id を保つのは Milvus の PK だから。"""
    rows = [_Row(7, "FAQ / 送料", "送料は", "3,000円以上で無料")]
    fresh = [_chunk("FAQ / 送料", "送料は", "5,000円以上で無料")]
    plan = rp._plan(rows, fresh)
    assert _names(plan)["changed"] == ["FAQ / 送料"]
    row, _, chunk = plan["changed"][0]
    assert row.id == 7                       # 新しい id を振らない
    assert chunk.answer == "5,000円以上で無料"


def test_a_new_section_is_added_and_a_deleted_one_is_removed():
    rows = [_Row(1, "FAQ / 送料", "q", "a"), _Row(2, "FAQ / 廃止", "q", "a")]
    fresh = [_chunk("FAQ / 送料", "q", "a"), _chunk("FAQ / 新設", "q2", "a2")]
    assert _names(rp._plan(rows, fresh)) == {
        "same": ["FAQ / 送料"], "changed": [], "added": ["FAQ / 新設"],
        "removed": ["FAQ / 廃止"]}


def test_a_renamed_heading_is_a_delete_plus_an_add():
    """見出しを変えた節は別の節。埋め込み直すのが正しい。"""
    rows = [_Row(1, "FAQ / 旧見出し", "q", "a")]
    fresh = [_chunk("FAQ / 新見出し", "q", "a")]
    got = _names(rp._plan(rows, fresh))
    assert got["added"] == ["FAQ / 新見出し"] and got["removed"] == ["FAQ / 旧見出し"]


def test_the_same_section_path_twice_is_matched_in_order():
    """大きな表は同じ section_path のまま複数 chunk に割れる(実測: 対応時間の表)。

    出現順に 1 対 1 で対応させないと、2 枚目以降が毎回「追加 + 削除」に化けて
    id が振り直され、変わっていない断片まで埋め込み直すことになる。
    """
    rows = [_Row(1, "AS / 対応時間", "対応時間", "表の 1 枚目"),
            _Row(2, "AS / 対応時間", "対応時間", "表の 2 枚目")]
    fresh = [_chunk("AS / 対応時間", "対応時間", "表の 1 枚目"),
             _chunk("AS / 対応時間", "対応時間", "表の 2 枚目を直した")]
    plan = rp._plan(rows, fresh)
    assert _names(plan) == {"same": ["AS / 対応時間"], "changed": ["AS / 対応時間"],
                            "added": [], "removed": []}
    assert plan["changed"][0][0].id == 2      # 直したのは 2 枚目だけ


def test_the_same_section_path_in_different_documents_does_not_collide():
    """content_type が違えば別物として扱う(見出しが同名でも文書が違う)。"""
    rows = [_Row(1, "共通 / 保証", "q", "a", content_type="faq")]
    fresh = [_chunk("共通 / 保証", "q", "a", content_type="faq"),
             _chunk("共通 / 保証", "q", "b", content_type="policy")]
    got = _names(rp._plan(rows, fresh))
    assert got["same"] == ["共通 / 保証"] and got["added"] == ["共通 / 保証"]
    assert got["removed"] == []


def test_an_empty_source_would_remove_everything():
    """全消しになる形を作れることを示す。実際に走らせないための番人は別途 main にある。"""
    rows = [_Row(i, f"FAQ / {i}", "q", "a") for i in range(1, 5)]
    plan = rp._plan(rows, [])
    assert len(plan["removed"]) == 4
    assert len(plan["removed"]) / len(rows) > rp._MAX_DELETE_RATIO


def test_chunks_that_did_not_come_from_a_document_are_never_removed():
    """フライホイールと会話マイニングで入った chunk は、資料に無くても消さない。

    ここが壊れると被害が大きい: data/kb を 1 文字直して repatch しただけで、
    レビューを通って書き戻されたナレッジが MySQL からも Milvus からも消える。
    件数が少ないので _MAX_DELETE_RATIO の歯止めにも掛からず、黙って通る。
    """
    rows = [_Row(1, "FAQ / 送料", "送料は", "3,000円以上で無料"),
            _Row(48, "flywheel", "ドローンの耐荷重は", "5kg までです"),
            _Row(49, "mined", "返品の期限は", "7 日以内です")]
    fresh = [_chunk("FAQ / 送料", "送料は", "3,000円以上で無料")]
    assert _names(rp._plan(rows, fresh)) == {
        "same": ["FAQ / 送料"], "changed": [], "added": [], "removed": []}


def test_the_delete_ratio_is_measured_against_document_chunks_only():
    """歯止めの分母に書き戻し由来を混ぜない。

    混ぜると分母が膨らみ、「資料由来が全滅しているのに割合は小さい」という
    見落としが起きる。
    """
    rows = [_Row(1, "FAQ / 送料", "送料は", "無料"),
            _Row(2, "FAQ / 返品", "返品は", "7 日以内")]
    rows += [_Row(10 + i, "flywheel", f"q{i}", f"a{i}") for i in range(20)]
    plan = rp._plan(rows, [])
    assert rp._exceeds_delete_guard(plan, rows) is True
