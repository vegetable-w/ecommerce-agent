"""評価スクリプトの指標計算を固定する。

ここが critical path である理由: Recall / MRR / coverage が黙って定数を返すようになると、
4 戦略の比較も、その後の章の「改善したか」も、すべて意味を失う。しかも壊れても
スクリプトは最後まで走り、それらしい表を出す。だから**壊れたときに落ちる**ことを
テストで示しておく必要がある(実際に定数へ差し替えて落ちることを確認済み。
下の docstring に変異と結果を残す)。

上流(埋め込み / リランク / チャット)と Milvus には一切触れない。
"""

import json
import pathlib

import pytest

from scripts import eval_04 as ev


@pytest.fixture(autouse=True)
def _no_upstream(monkeypatch):
    """本物の上流を呼ばないことの保証(カバレッジではなく運用の安全網)。

    eval_04 は生成・リランク・埋め込みを呼ぶスクリプトなので、差し替え漏れがあれば
    テストが緑のまま実 API と課金を叩き続ける。ここで爆弾を仕掛けて気付けるようにする。
    """
    def _boom(*a, **kw):
        raise AssertionError("本物の上流が呼ばれた")

    monkeypatch.setattr("app.core.llm.get_chat_model", _boom)
    monkeypatch.setattr("app.core.embeddings.embed_query", _boom)
    monkeypatch.setattr("app.core.rerank.rerank", _boom)


def _hit(path: str, answer: str = "") -> dict:
    return {"section_path": path, "answer": answer, "question": ""}


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("3 週間", "3週間"),
    ("| 修理 | 4時間 |", "|修理|4時間|"),
    ("PM2.5 / におい", "PM2.5/におい"),
    ("　全角空白", "全角空白"),          # U+3000。日本語入力でそのまま打てる
    ("改行\nあり", "改行あり"),
    (None, ""),
])
def test_norm_removes_every_kind_of_whitespace(raw, expected):
    assert ev.norm(raw) == expected


# ---------------------------------------------------------------------------
# 正解判定 / Recall / MRR
# ---------------------------------------------------------------------------


def test_is_relevant_requires_all_keys():
    """AND であること。片方しか含まない節を正解にしてはいけない。"""
    hit = _hit("返品・返金ポリシー / 交換ポリシー / 交換時の送料")
    assert ev.is_relevant(hit, ["交換ポリシー", "交換時の送料"])
    assert not ev.is_relevant(_hit("返品・返金ポリシー / 返品・交換時の送料負担"),
                              ["交換ポリシー", "交換時の送料"])


def test_is_relevant_is_false_without_expectation():
    """D bucket(正解なし)は何を引いても正解にしない。"""
    assert not ev.is_relevant(_hit("商品・ショッピング FAQ / 送料はいくらですか"), [])


def test_first_relevant_rank_returns_position_of_first_match():
    hits = [_hit("a"), _hit("商品・ショッピング FAQ / 送料はいくらですか"),
            _hit("商品・ショッピング FAQ / 送料はいくらですか")]
    assert ev.first_relevant_rank(hits, ["送料はいくらですか"]) == 2
    assert ev.first_relevant_rank(hits, ["返品対象外の商品"]) is None


def test_recall_at_k_respects_the_cutoff():
    """k 番目は入り、k+1 番目は入らない。

    ここを >= / > で書き違えると Recall が常に 1 に寄る。
    """
    hits = [_hit("x")] * 9 + [_hit("y / 送料はいくらですか")]
    assert ev.recall_at_k(hits, ["送料はいくらですか"], k=10) == 1.0
    assert ev.recall_at_k(hits, ["送料はいくらですか"], k=9) == 0.0


def test_recall_is_zero_when_nothing_matches():
    assert ev.recall_at_k([_hit("a"), _hit("b")], ["送料はいくらですか"]) == 0.0
    assert ev.recall_at_k([], ["送料はいくらですか"]) == 0.0


@pytest.mark.parametrize("position, expected", [(1, 1.0), (2, 0.5), (4, 0.25)])
def test_reciprocal_rank_is_the_inverse_of_the_position(position, expected):
    hits = [_hit("x") for _ in range(position - 1)] + [_hit("y / 送料はいくらですか")]
    assert ev.reciprocal_rank(hits, ["送料はいくらですか"]) == expected


def test_reciprocal_rank_is_zero_when_out_of_range():
    hits = [_hit("x")] * 10 + [_hit("y / 送料はいくらですか")]
    assert ev.reciprocal_rank(hits, ["送料はいくらですか"], k=10) == 0.0


# ---------------------------------------------------------------------------
# evidence coverage
# ---------------------------------------------------------------------------


def test_coverage_counts_only_points_present_in_the_evidence():
    hits = [_hit("s", "1回の注文金額が3,000円以上の場合は送料無料。"),
            _hit("s", "一部の遠隔地域については追加送料が発生する場合がある。")]
    assert ev.coverage_mech(["3,000円以上の場合は送料無料", "500円の送料"], hits) == 0.5
    assert ev.coverage_mech(["3,000円以上の場合は送料無料"], hits) == 1.0
    assert ev.coverage_mech(["存在しない事実"], hits) == 0.0


def test_coverage_ignores_whitespace_differences():
    """「3 週間」と「3週間」、表の「| 修理 | 4時間 |」を同じとみなす。"""
    assert ev.coverage_mech(["3週間"], [_hit("s", "交換目安は 3 週間 です")]) == 1.0
    assert ev.coverage_mech(["4時間"], [_hit("s", "| 修理 | 4時間 | 7営業日 |")]) == 1.0


def test_coverage_is_none_when_there_is_nothing_to_measure():
    """D bucket は key point を持たない。0.0 にすると平均を不当に下げる。"""
    assert ev.coverage_mech([], [_hit("s", "何か")]) is None


def test_coverage_is_zero_when_retrieval_found_nothing():
    assert ev.coverage_mech(["3,000円以上の場合は送料無料"], []) == 0.0


# ---------------------------------------------------------------------------
# 回答拒否の判定(D bucket)
# ---------------------------------------------------------------------------


def test_refusal_is_detected_from_the_fixed_phrase():
    assert ev.looks_refused("現在、関連する情報を確認できませんでした。オペレーターへおつなぎしますか。")
    assert ev.looks_refused("恐れ入りますが、ナレッジに記載がございません。")


def test_a_normal_answer_is_not_counted_as_a_refusal():
    assert not ev.looks_refused("3,000円以上のご注文は送料無料です[1]。")
    assert not ev.looks_refused(None)


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------


def test_aggregate_averages_per_bucket_and_overall():
    records = [
        {"bucket": "A_policy", "v": 1.0},
        {"bucket": "A_policy", "v": 0.0},
        {"bucket": "B_model", "v": 1.0},
    ]
    out = ev.aggregate(records, "v", ["A_policy", "B_model"])
    assert out["A_policy"] == {"value": 0.5, "n": 2}
    assert out["B_model"] == {"value": 1.0, "n": 1}
    assert out["overall"] == {"value": pytest.approx(2 / 3), "n": 3}


def test_aggregate_drops_none_instead_of_scoring_it_zero():
    """None は「測る対象が無かった」。0 として混ぜると overall が実態より低く出る。"""
    records = [{"bucket": "A_policy", "v": 1.0}, {"bucket": "D_absent", "v": None}]
    out = ev.aggregate(records, "v", ["A_policy"])
    assert out["overall"] == {"value": 1.0, "n": 1}


def test_aggregate_reports_none_when_a_bucket_has_no_measurable_sample():
    out = ev.aggregate([{"bucket": "A_policy", "v": None}], "v", ["A_policy"])
    assert out["A_policy"] == {"value": None, "n": 0}


# ---------------------------------------------------------------------------
# 実データに対する健全性(定数を返す実装では通らない)
# ---------------------------------------------------------------------------


def test_metrics_separate_a_good_ranking_from_a_bad_one():
    """指標が入力に反応すること。

    変異で確認済み(いずれも DATABASE_URL を存在しないホストへ向けた状態で実行):
      * `coverage_mech` の本体を `return 1.0` に固定 → 3 件 fail(このテストを含む)。
      * `first_relevant_rank` の本体を `return 1` に固定 → 7 件 fail(このテストを含む)。
      * `aggregate` の集計値を `1.0` に固定 → 2 件 fail。
      * `looks_refused` を `return True` に固定 → 1 件 fail。
    """
    expect = ["送料はいくらですか"]
    points = ["3,000円以上の場合は送料無料", "500円の送料"]
    good = [_hit("商品・ショッピング FAQ / 送料はいくらですか",
                 "1回の注文金額が3,000円以上の場合は送料無料。3,000円未満の場合は500円の送料がかかる。"),
            _hit("商品・ショッピング FAQ / 返品ポリシーについて", "受取後7日以内")]
    bad = [_hit("商品・ショッピング FAQ / 返品ポリシーについて", "受取後7日以内"),
           _hit("返品・返金ポリシー / 返品対象外の商品", "生鮮食品")]

    assert ev.recall_at_k(good, expect) == 1.0
    assert ev.recall_at_k(bad, expect) == 0.0
    assert ev.reciprocal_rank(good, expect) == 1.0
    assert ev.reciprocal_rank(bad, expect) == 0.0
    assert ev.coverage_mech(points, good) == 1.0
    assert ev.coverage_mech(points, bad) == 0.0


# ---------------------------------------------------------------------------
# 評価セットの読み込み
# ---------------------------------------------------------------------------


def test_load_samples_reads_the_real_evaluation_set():
    samples = ev.load_samples()
    assert len(samples) == 80
    assert {s["bucket"] for s in samples} == set(ev.BUCKETS)


def test_load_samples_reports_the_line_number_of_broken_json(tmp_path):
    p = tmp_path / "broken.jsonl"
    p.write_text('{"id":"A1","bucket":"A_policy","query":"q","expect_section":[],'
                 '"expect_points":[],"should_refuse":false}\nこれは JSON ではない\n',
                 encoding="utf-8")
    with pytest.raises(SystemExit, match="2 行目"):
        ev.load_samples(p)


def test_load_samples_reports_missing_fields(tmp_path):
    p = tmp_path / "partial.jsonl"
    p.write_text('{"id":"A1","bucket":"A_policy","query":"q"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="足りません"):
        ev.load_samples(p)


def test_limit_takes_the_head_of_each_bucket():
    samples = [{"bucket": "A_policy", "id": f"A{i}"} for i in range(5)]
    samples += [{"bucket": "D_absent", "id": f"D{i}"} for i in range(5)]
    got = ev._limited(samples, 2)
    assert [s["id"] for s in got] == ["A0", "A1", "D0", "D1"]
    assert ev._limited(samples, 0) == samples


# ---------------------------------------------------------------------------
# collection の事前チェック(実行前に分かる失敗を、検索の奥の例外にしない)
# ---------------------------------------------------------------------------


class _FakeClient:
    """Milvus を立てずに check_collection を試すための最小の偽 client。"""

    def __init__(self, exists=True, fields=None, rows=80):
        self._exists = exists
        self._fields = fields if fields is not None else sorted(milvus_required_fields())
        self._rows = rows

    def has_collection(self, name):
        return self._exists

    def describe_collection(self, name):
        return {"fields": [{"name": f} for f in self._fields]}

    def load_collection(self, name):
        return None

    def query(self, name, filter, output_fields):
        return [{"count(*)": self._rows}]


def milvus_required_fields():
    from app.kb import milvus_client
    return milvus_client._REQUIRED_FIELDS


def test_check_collection_returns_the_row_count_when_ready():
    assert ev.check_collection(_FakeClient(), "knowledge") == 80


def test_check_collection_explains_a_missing_collection():
    with pytest.raises(SystemExit, match="ありません"):
        ev.check_collection(_FakeClient(exists=False), "knowledge")


def test_check_collection_explains_an_old_schema_instead_of_raising_deep_inside_search():
    """03 章の schema の collection に当てても、検索の奥で落ちる前にここで止まること。"""
    old = ["id", "vector", "question", "answer"]
    with pytest.raises(SystemExit, match="再構築"):
        ev.check_collection(_FakeClient(fields=old), "knowledge")


def test_check_collection_explains_an_empty_collection():
    with pytest.raises(SystemExit, match="空です"):
        ev.check_collection(_FakeClient(rows=0), "knowledge")


# ---------------------------------------------------------------------------
# report の書き出し(Windows の既定 cp932 で日本語が壊れないこと)
# ---------------------------------------------------------------------------


def test_report_files_round_trip_japanese_as_utf8(tmp_path, monkeypatch):
    """encoding="utf-8" の指定が落ちると、㎡ や 〜 は cp932 で書けず黙って壊れる。"""
    monkeypatch.setattr(ev, "_LINES", ["Stage 1 Retrieval  Recall@10", "  約8〜12か月 / 約40㎡"])
    report = {"meta": {"generated_at": "2026-09-07T00:00:00", "collection": "knowledge",
                       "collection_rows": 47, "samples": 80,
                       # .json にも非 ASCII を混ぜる。ここが cp932 で書かれると
                       # 評価ページ(JSON を読む側)が壊れる
                       "errors": ["約8〜12か月 / 約40㎡ の検索に失敗"]},
              "retrieval": {"dense": {"recall_at_k": {"overall": {"value": 0.5, "n": 60}}}},
              "evidence_coverage": {}, "generation": None}
    paths = ev.write_report(report, tmp_path)

    for ext in ("txt", "json", "html"):
        assert paths[ext].exists()
        # encoding を明示して読み直す。既定に任せると環境の cp932 で読んでしまう
        text = paths[ext].read_text(encoding="utf-8")
        assert "約8〜12か月" in text and "約40㎡" in text, ext
        # 生バイトが本当に UTF-8 か(cp932 で書かれていれば decode で落ちる)
        paths[ext].read_bytes().decode("utf-8")

    assert json.loads(paths["json"].read_text(encoding="utf-8"))["meta"]["samples"] == 80
    assert "<html" in paths["html"].read_text(encoding="utf-8")


def test_report_is_written_even_when_the_generation_stage_is_missing(tmp_path):
    paths = ev.write_report({"meta": {}, "retrieval": {}, "evidence_coverage": {},
                             "generation": None}, tmp_path)
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["generation"] is None


def test_report_directory_is_created_on_demand(tmp_path):
    target = tmp_path / "data" / "04" / "reports"
    ev.write_report({"meta": {}}, target)
    assert (target / "rag_eval.json").exists()


# ---------------------------------------------------------------------------
# evidence の組み立て
# ---------------------------------------------------------------------------


def test_evidence_is_numbered_from_one_like_query_faq():
    hits = [{"question": "送料はいくらですか", "answer": "3,000円以上は無料"},
            {"question": "返品ポリシーについて", "answer": "7日以内"}]
    assert ev.build_evidence(hits) == (
        "[1] 送料はいくらですか: 3,000円以上は無料\n[2] 返品ポリシーについて: 7日以内")


def test_evidence_is_empty_when_nothing_was_retrieved():
    assert ev.build_evidence([]) == ""


# ---------------------------------------------------------------------------
# 引数
# ---------------------------------------------------------------------------


def test_default_strategies_are_the_four_being_compared():
    args = ev._parse_args([])
    assert args.strategies.split(",") == ["dense", "bm25", "hybrid", "hybrid_rerank"]
    assert args.collection == "knowledge"


def test_scripts_package_is_importable_without_a_live_collection():
    """本番 collection が壊れていても import と単体テストは通ること。"""
    assert pathlib.Path(ev.__file__).name == "eval_04.py"
    assert ev.K == 10
