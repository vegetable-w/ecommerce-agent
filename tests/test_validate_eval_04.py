"""評価セットの自己検証(scripts/validate_eval_04.py)が、壊れた出題で本当に落ちること。

検証スクリプトは「データが正しい」と言うのが仕事なので、黙って空のエラーリストを返す
実装になっても誰も気付けない。壊れたサンプルを 1 つずつ与えて、対応する検査が
きちんと鳴ることを固定する。

DB にも Milvus にも触れない。ナレッジ(section_path → 本文)は引数で渡す。
"""

from scripts import validate_eval_04 as vd

SECTIONS = {
    "商品・ショッピング FAQ / 送料はいくらですか":
        "1回の注文金額が3,000円以上の場合は送料無料。3,000円未満の場合は500円の送料がかかる。",
    "返品・返金ポリシー / 交換ポリシー / 交換時の送料":
        "商品不良による交換は当社が送料を負担する。",
    "返品・返金ポリシー / 返品対象外の商品": "生鮮食品は返品できない。",
}


def _sample(**over) -> dict:
    s = {"id": "A1", "bucket": "A_policy", "query": "送料について教えてください",
         "expect_section": ["送料はいくらですか"], "expect_points": ["500円の送料"],
         "should_refuse": False}
    s.update(over)
    return s


def _ok_samples() -> list[dict]:
    """5 bucket が 1 問ずつ揃った、どの検査にも引っかからないセット。"""
    return [
        _sample(),
        _sample(id="B1", bucket="B_model", query="交換のときの送料は誰が払いますか",
                expect_section=["交換ポリシー", "交換時の送料"],
                expect_points=["当社が送料を負担する"]),
        _sample(id="C1", bucket="C_colloquial", query="生ものって返せる？",
                expect_section=["返品対象外の商品"], expect_points=["生鮮食品"]),
        _sample(id="E1", bucket="E_multi", query="送料と、返せない商品をまとめて教えて",
                expect_section=None, expect_sections_all=[["送料はいくらですか"],
                                                          ["返品対象外の商品"]],
                expect_points=["3,000円以上の場合は送料無料", "生鮮食品"]),
        {"id": "D1", "bucket": "D_absent", "query": "店頭で受け取れますか",
         "expect_section": [], "expect_points": [], "should_refuse": True},
    ]


def test_a_clean_set_produces_no_errors():
    samples = _ok_samples()
    samples[3].pop("expect_section")      # 複数根拠は expect_sections_all だけを持つ
    assert vd.validate(samples, SECTIONS) == []


def test_build_sections_joins_chunks_that_share_a_section_path():
    """表が分割されて同じ節が複数 chunk になっても、本文は 1 つにまとめて照合する。"""
    sections = vd.build_sections([("節A", "前半の本文"), ("節A", "後半の本文"),
                                  ("節B", "別の節"), ("", "section_path が無い chunk")])
    assert sections["節A"] == "前半の本文\n後半の本文"
    assert set(sections) == {"節A", "節B"}


def test_duplicate_ids_and_queries_are_reported():
    samples = _ok_samples()
    samples.append(_sample(id="A1", query="まったく別の質問です"))
    samples.append(_sample(id="A2", query="送料に　ついて教えてください"))   # 空白違いも重複
    errors = vd.check_no_duplicates(samples)
    assert any("A1" in e and "id が重複" in e for e in errors)
    assert any("A2" in e and "質問が A1 と重複" in e for e in errors)


def test_a_missing_bucket_is_reported():
    samples = [s for s in _ok_samples() if s["bucket"] != "E_multi"]
    errors = vd.check_bucket_counts(samples)
    assert any("E_multi" in e and "1 問も無い" in e for e in errors)


def test_uneven_bucket_sizes_are_reported():
    samples = _ok_samples()
    samples.append(_sample(id="A2", query="別の送料の質問です"))
    errors = vd.check_bucket_counts(samples)
    assert any("揃っていない" in e and "A_policy=2" in e for e in errors)


def test_an_unknown_bucket_is_reported():
    errors = vd.check_bucket_counts(_ok_samples() + [_sample(id="Z1", bucket="Z_unknown",
                                                             query="謎の bucket")])
    assert any("Z_unknown" in e for e in errors)


def test_a_section_that_does_not_exist_in_the_knowledge_base_is_reported():
    """どこにも当たらない期待値は、検索が何を返しても Recall 0 になる出題ミス。"""
    errors = vd.check_sections_exist([_sample(expect_section=["存在しない見出し"])], SECTIONS)
    assert len(errors) == 1 and "A1" in errors[0] and "存在しない見出し" in errors[0]


def test_each_group_of_a_multi_evidence_sample_is_checked_separately():
    """2 グループ目だけが壊れていても見逃さない(グループ番号も出す)。"""
    s = _sample(id="E9", bucket="E_multi", expect_section=None,
                expect_sections_all=[["送料はいくらですか"], ["存在しない見出し"]])
    s.pop("expect_section")
    errors = vd.check_sections_exist([s], SECTIONS)
    assert len(errors) == 1 and "グループ 2" in errors[0]


def test_an_and_condition_group_that_matches_one_section_passes():
    """グループ内は AND。両方を含む節が 1 つあれば通る。"""
    assert vd.check_sections_exist(
        [_sample(expect_section=["交換ポリシー", "交換時の送料"],
                 expect_points=["当社が送料を負担する"])], SECTIONS) == []


def test_a_point_that_is_not_in_the_target_section_is_reported():
    """原文に無い key point は evidence coverage で永久に 0 になる。"""
    errors = vd.check_points_exist([_sample(expect_points=["1,000円の送料"])], SECTIONS)
    assert len(errors) == 1 and "1,000円の送料" in errors[0]


def test_points_are_matched_ignoring_whitespace():
    assert vd.check_points_exist([_sample(expect_points=["500 円 の 送料"])], SECTIONS) == []


def test_a_point_may_come_from_any_group_of_a_multi_evidence_sample():
    s = _sample(id="E1", bucket="E_multi", expect_sections_all=[["送料はいくらですか"],
                                                                ["返品対象外の商品"]],
                expect_points=["生鮮食品"])
    s.pop("expect_section")
    assert vd.check_points_exist([s], SECTIONS) == []


def test_the_absent_bucket_must_not_carry_an_answer():
    bad = {"id": "D9", "bucket": "D_absent", "query": "q",
           "expect_section": ["送料はいくらですか"], "expect_points": ["500円の送料"],
           "should_refuse": False}
    errors = vd.check_expectations_match_the_bucket([bad])
    assert any("expect_section" in e for e in errors)
    assert any("expect_points" in e for e in errors)
    assert any("should_refuse" in e for e in errors)


def test_a_graded_bucket_without_an_answer_is_reported():
    errors = vd.check_expectations_match_the_bucket(
        [_sample(expect_section=[], expect_points=[], should_refuse=True)])
    assert len(errors) == 3


def test_validate_runs_every_check_and_keeps_going_after_the_first_error():
    """1 件目で止めない。1 回の実行で全部直せることが、このスクリプトの価値なので。"""
    samples = [_sample(id="A1", expect_section=["存在しない見出し"],
                       expect_points=["1,000円の送料"])]
    errors = vd.validate(samples, SECTIONS)
    assert any("bucket" in e for e in errors)          # 2 検査目
    assert any("section がナレッジに無い" in e for e in errors)   # 4 検査目
    assert any("key point" in e for e in errors)       # 5 検査目
