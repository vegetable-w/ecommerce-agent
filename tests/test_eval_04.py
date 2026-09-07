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
from langchain_core.runnables import RunnableLambda

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
# 複数根拠 (expect_sections_all) の採点
# ---------------------------------------------------------------------------


def test_as_groups_accepts_both_the_old_and_the_new_shape():
    assert ev.as_groups(["送料はいくらですか"]) == [["送料はいくらですか"]]
    assert ev.as_groups([["交換ポリシー", "交換時の送料"], ["送料はいくらですか"]]) == [
        ["交換ポリシー", "交換時の送料"], ["送料はいくらですか"]]
    assert ev.as_groups([]) == []


def test_expect_groups_prefers_the_multi_evidence_field():
    """両方書かれていたら expect_sections_all を採る(load_samples のコメントと同じ順位)。"""
    both = {"expect_section": ["旧い書き方"],
            "expect_sections_all": [["新しい書き方 1"], ["新しい書き方 2"]]}
    assert ev.expect_groups(both) == [["新しい書き方 1"], ["新しい書き方 2"]]
    assert ev.expect_groups({"expect_section": ["送料はいくらですか"]}) == [["送料はいくらですか"]]
    assert ev.expect_groups({"expect_section": []}) == []


def test_recall_gives_partial_credit_per_group():
    """3 グループ中 2 つ当たれば 0.667。全部揃って 1.0、1 つも当たらなければ 0.0。

    変異で確認済み: 「全グループ揃ったときだけ 1.0」の all-or-nothing 実装に差し替え
    → このテストが fail。部分点が落ちると E_multi の Recall だけが構造的に低く出る。
    """
    hits = [_hit("FAQ / 送料はいくらですか"), _hit("無関係な節"),
            _hit("返品・返金ポリシー / 交換ポリシー / 交換時の送料")]
    groups = [["送料はいくらですか"], ["交換ポリシー", "交換時の送料"], ["返品対象外の商品"]]
    assert ev.recall_at_k(hits, groups) == pytest.approx(2 / 3)
    assert ev.recall_at_k(hits, groups[:2]) == 1.0
    assert ev.recall_at_k(hits, [groups[2]]) == 0.0


def test_mrr_of_multiple_groups_is_the_mean_of_each_rank_not_the_rank_they_were_completed():
    """MRR は各根拠の逆順位の**平均**。

    「最後の 1 グループが揃った順位」の逆数を返す実装にすると、2 グループの問いは上限が
    0.5 になり、単一根拠の bucket と並べた瞬間に検索が劣化したように見える。
    1 位と 2 位に出たなら (1/1 + 1/2)/2 = 0.75 であって 0.5 ではない。

    変異で確認済み(DATABASE_URL を存在しないホストへ向けた状態で実行):
      * reciprocal_rank を「全グループが揃った順位の逆数」に差し替え → 3 件 fail。
      * miss したグループを平均から外す(0 を入れない)実装に差し替え → 1 件 fail。
    """
    hits = [_hit("FAQ / 送料はいくらですか"), _hit("返品・返金ポリシー / 返品対象外の商品")]
    groups = [["送料はいくらですか"], ["返品対象外の商品"]]
    assert ev.reciprocal_rank(hits, groups) == 0.75


def test_mrr_counts_a_group_that_was_never_retrieved_as_zero():
    """圏外のグループを平均から外すと、半分しか引けていない問いが満点になる。"""
    hits = [_hit("FAQ / 送料はいくらですか")]
    assert ev.reciprocal_rank(hits, [["送料はいくらですか"], ["どこにも無い節"]]) == 0.5
    assert ev.reciprocal_rank(hits, [["どこにも無い節"], ["これも無い"]]) == 0.0


def test_recall_depth_is_independent_from_the_search_depth():
    """K(検索深度)と RECALL_K は別物。7 位のヒットは MRR には効くが Recall@5 には入らない。"""
    assert (ev.K, ev.RECALL_K) == (10, 5)
    hits = [_hit("無関係な節")] * 6 + [_hit("FAQ / 送料はいくらですか")]
    assert ev.recall_at_k(hits, ["送料はいくらですか"]) == 0.0
    assert ev.reciprocal_rank(hits, ["送料はいくらですか"]) == pytest.approx(1 / 7)


def test_the_old_single_section_format_is_scored_exactly_as_before():
    """後方互換: expect_section の 1 節はグループ 1 個と同じ点になる。"""
    hits = [_hit("無関係な節"), _hit("FAQ / 送料はいくらですか")]
    assert ev.recall_at_k(hits, ["送料はいくらですか"]) == 1.0
    assert ev.reciprocal_rank(hits, ["送料はいくらですか"]) == 0.5
    assert ev.recall_at_k(hits, [["送料はいくらですか"]]) == 1.0
    assert ev.reciprocal_rank(hits, [["送料はいくらですか"]]) == 0.5


async def test_run_deterministic_publishes_recall_under_both_the_new_and_the_old_key(monkeypatch):
    """report に出る指標キーを固定する。

    新: retrieval[戦略]["recall_at_5"]、旧: 同 ["recall_at_k"](同じ値)。
    評価ページと過去 run のトレンドが旧キーを読んでいるので、両方から読めること。
    """
    async def fake_search(query, strategy=None, top_k=10, min_score=None, collection=None):
        return [_hit("商品・ショッピング FAQ / 送料はいくらですか",
                     "1回の注文金額が3,000円以上の場合は送料無料。"),
                _hit("返品・返金ポリシー / 返品対象外の商品", "生鮮食品は返品できない。")]

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    samples = [{"id": "E1", "bucket": "E_multi", "query": "送料と返品対象外を教えて",
                "expect_sections_all": [["送料はいくらですか"], ["返品対象外の商品"]],
                "expect_points": ["3,000円以上の場合は送料無料"], "should_refuse": False}]

    retrieval_out, coverage_out, hits_by = await ev.run_deterministic(
        samples, "knowledge", ["dense"])

    node = retrieval_out["dense"]
    assert node[ev.RECALL_KEY]["E_multi"] == {"value": 1.0, "n": 1}
    assert ev.RECALL_KEY == "recall_at_5"
    assert node["recall_at_k"] == node[ev.RECALL_KEY]
    assert node["mrr"]["E_multi"]["value"] == 0.75
    assert coverage_out["dense"]["coverage"]["E_multi"]["value"] == 1.0


# ---------------------------------------------------------------------------
# 評価セットの読み込み
# ---------------------------------------------------------------------------


def test_load_samples_reads_the_real_evaluation_set():
    """実データが BUCKETS の内側に収まっていること。

    == ではなく部分集合で見る。E_multi は bucket として先に定義してあり、問題そのものは
    後から入る。ここを == にすると「データがまだ無い」だけで指標側のテストが赤くなる。
    """
    samples = ev.load_samples()
    assert samples
    assert {s["bucket"] for s in samples} <= set(ev.BUCKETS)


def test_load_samples_reports_the_line_number_of_broken_json(tmp_path):
    p = tmp_path / "broken.jsonl"
    p.write_text('{"id":"A1","bucket":"A_policy","query":"q","expect_section":[],'
                 '"expect_points":[],"should_refuse":false}\nこれは JSON ではない\n',
                 encoding="utf-8")
    with pytest.raises(SystemExit, match="2 行目"):
        ev.load_samples(p)


def test_load_samples_accepts_the_multi_evidence_format(tmp_path):
    p = tmp_path / "multi.jsonl"
    p.write_text('{"id":"E1","bucket":"E_multi","query":"q",'
                 '"expect_sections_all":[["交換ポリシー","交換時の送料"],["送料はいくらですか"]],'
                 '"expect_points":["500円の送料"],"should_refuse":false}\n', encoding="utf-8")
    assert ev.expect_groups(ev.load_samples(p)[0]) == [
        ["交換ポリシー", "交換時の送料"], ["送料はいくらですか"]]


def test_load_samples_requires_one_of_the_two_answer_formats(tmp_path):
    """expect_section も expect_sections_all も無い行は、正解の無い採点対象になってしまう。"""
    p = tmp_path / "noanswer.jsonl"
    p.write_text('{"id":"A1","bucket":"A_policy","query":"q","expect_points":["x"],'
                 '"should_refuse":false}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="expect_sections_all"):
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


# ---------------------------------------------------------------------------
# 幻覚ケース台帳(faith_cases への保存と、今回の実行だけで測る幻覚率)
#
# DB には触れない。repository の 2 つの関数を差し替えて、
# 「何を渡したか」と「返ってきたものをどう読むか」だけをここで固定する。
# ---------------------------------------------------------------------------


# 根拠スナップショットの元になる hit。citations の chunk_id は hit の id をそのまま使う
_KB_HITS = [
    {"id": 41, "section_path": "返品・返金ポリシー / 未開封の場合",
     "question": "未開封なら返品できますか", "answer": "到着後 7 日以内に限り承ります。"},
    {"id": 42, "section_path": "返品・返金ポリシー / 開封済みの場合",
     "question": "開封後も返品できますか", "answer": "初期不良の場合のみ交換で対応します。"},
    {"id": 7, "section_path": "配送 / お届け日数",
     "question": "何日で届きますか", "answer": "本州は 2〜3 日が目安です。"},
]


def test_citations_are_numbered_exactly_like_the_evidence_given_to_the_model():
    """回答本文の [n] が指す根拠と、台帳へ残す citations の n は同じでなければならない。

    ずれても両方それらしく見えるので、後から判定を見直す人は間違った根拠を読まされる。

    変異で確認済み(DATABASE_URL を存在しないホストへ向けた状態で実行):
      * build_citations の番号を 1 つずらす(enumerate(hits, 2))→ 2 件 fail(このテストを含む)。
    """
    citations = ev.build_citations(_KB_HITS)
    lines = ev.build_evidence(_KB_HITS).splitlines()
    assert [c["n"] for c in citations] == [1, 2, 3]
    assert [c["chunk_id"] for c in citations] == [41, 42, 7]
    for c, h in zip(citations, _KB_HITS, strict=True):
        assert lines[c["n"] - 1] == f"[{c['n']}] {h['question']}: {h['answer']}"


def test_citations_keep_every_retrieved_hit_not_only_the_cited_ones():
    """回答が引用するのは 2〜3 件でも、判定に渡した入力を再現するには全件が要る。"""
    citations = ev.build_citations(_KB_HITS)
    assert len(citations) == len(_KB_HITS)
    assert set(citations[0]) == {"n", "chunk_id", "section_path", "question", "answer"}
    assert citations[2] == {"n": 3, "chunk_id": 7, "section_path": "配送 / お届け日数",
                            "question": "何日で届きますか", "answer": "本州は 2〜3 日が目安です。"}
    assert ev.build_citations([]) == []


class _Msg:
    """chat model の戻り値。_generate_one が読むのは .content だけ。"""

    def __init__(self, content: str):
        self.content = content


class _FakeChat(RunnableLambda):
    """本物の上流の代わり。`prompt | model` と `model.with_structured_output(...)` の
    両方に応える最小の Runnable。"""

    def __init__(self, answer: str, faithful: bool):
        super().__init__(self._reply)
        self._answer = answer
        self._faithful = faithful

    def _reply(self, value):
        return _Msg(self._answer)

    def with_structured_output(self, schema):
        def _judge(value):
            if schema is ev._Faithful:
                return ev._Faithful(faithful=self._faithful,
                                    reason="evidence に無い日数を答えている")
            return ev._Coverage(covered_numbers=[1])
        return RunnableLambda(_judge)


async def _generation(monkeypatch, faithful: bool, seen: list | None = None) -> dict:
    def _factory(**kw):
        if seen is not None:
            seen.append(kw)
        return _FakeChat("到着後 30 日以内なら返品できます[1]。", faithful)

    monkeypatch.setattr(ev, "get_chat_model", _factory)
    samples = [{"id": "A1", "bucket": "A_policy", "query": "返品はいつまでできますか",
                "expect_section": ["未開封の場合"], "expect_points": ["7 日以内"],
                "should_refuse": False}]
    return await ev.run_generation(samples, {("hybrid_rerank", "A1"): _KB_HITS},
                                   ["hybrid_rerank"])


async def test_the_judges_run_at_temperature_zero(monkeypatch):
    """judge は 0、回答の生成は本番と同じ既定値。

    judge を既定の 0.3 のまま回すと、同じ回答に対する判定が実行ごとに揺れる
    (同一入力を 4 回流して 5/6、5/6、6/6、6/6。しかも外したケースが毎回違った)。
    逆に生成側まで 0 にすると「本番より安定した回答」を測ることになるので、
    2 つを取り違えていないことをここで固定する。
    """
    seen: list[dict] = []
    await _generation(monkeypatch, faithful=True, seen=seen)
    assert {"temperature": 0} in seen, seen
    assert {} in seen, seen


async def test_generation_keeps_the_judged_evidence_of_a_hallucination_case(monkeypatch):
    """faithful=false の問いは、判定の理由と「そのとき渡した根拠の全件」ごと残す。"""
    out = await _generation(monkeypatch, faithful=False)
    cases = out["faithfulness_cases"]
    assert [c["id"] for c in cases] == ["A1"]
    assert cases[0]["bucket"] == "A_policy"
    assert cases[0]["query"] == "返品はいつまでできますか"
    assert cases[0]["reason"] == "evidence に無い日数を答えている"
    # judge に渡した evidence と同じ番号付けの全件スナップショット
    assert cases[0]["citations"] == ev.build_citations(_KB_HITS)


async def test_generation_leaves_no_case_when_the_answer_was_faithful(monkeypatch):
    out = await _generation(monkeypatch, faithful=True)
    assert out["faithfulness_cases"] == []
    assert out["faithfulness"]["value"] == 1.0
    # 指標側の per_sample は 0/1 だけ。回答本文や根拠まで並べると 300 問のレポートが読めなくなる
    assert set(out["faithfulness"]["per_sample"][0]) == {"id", "bucket", "faithful"}


def _case(eval_id: str = "A1") -> dict:
    return {"id": eval_id, "bucket": "A_policy", "query": "返品はいつまでできますか",
            "answer": "到着後 30 日以内なら返品できます[1]。",
            "reason": "evidence に無い日数を答えている",
            "citations": ev.build_citations(_KB_HITS)}


def _fake_repository(monkeypatch, status_map: dict, *, previous_status=None,
                     upsert_error: Exception | None = None,
                     map_error: Exception | None = None) -> list[dict]:
    """repository の 2 関数を差し替え、upsert に渡された内容を記録して返す。"""
    saved: list[dict] = []

    async def _upsert(eval_id, **kw):
        if upsert_error is not None:
            raise upsert_error
        saved.append({"eval_id": eval_id, **kw})
        return {"id": len(saved), "eval_id": eval_id, "status": "unresolved",
                "seen_count": 2, "created": False, "recurred": previous_status == "resolved",
                "previous_status": previous_status}

    async def _status_map():
        if map_error is not None:
            raise map_error
        return dict(status_map)

    monkeypatch.setattr("app.db.repository.upsert_faith_case", _upsert)
    monkeypatch.setattr("app.db.repository.faith_case_status_map", _status_map)
    return saved


def _generation_dict(cases: list[dict], n: int = 300) -> dict:
    return {"faithfulness": {"strategy": "hybrid_rerank", "value": 1 - len(cases) / n,
                             "n": n, "per_sample": []},
            "faithfulness_cases": cases}


async def test_hallucination_rate_counts_only_this_run_not_the_whole_ledger(monkeypatch):
    """台帳に 3 行あっても、今回の実行で幻覚と判定されたのが 1 問なら 1/300。

    台帳は実行をまたいで積み上がるので、そこには前回までに直したケースも残っている。
    総行数を分子にすると、修正が進むほど率が上がるという逆の数字になる
    (実測 2.33%、今回の実態は 0.33%)。累計は hallucination.ledger に別途出す。

    変異で確認済み(DATABASE_URL を存在しないホストへ向けた状態で実行):
      * build_hallucination の分子を len(cases) から ledger["total"] へ差し替え
        → 2 件 fail(このテストを含む)。
    """
    _fake_repository(monkeypatch, {"A1": "unresolved", "B2": "resolved",
                                   "C3": "no_action_needed"})
    out = await ev.run_hallucination(_generation_dict([_case("A1")]))

    assert out["rate"] == {"value": pytest.approx(1 / 300), "cases": 1, "n": 300}
    assert out["ledger"] == {"total": 3, "counts": {"unresolved": 1, "resolved": 1,
                                                    "no_action_needed": 1}}


async def test_hallucination_case_is_written_to_the_ledger_with_its_evidence(monkeypatch):
    """台帳へ積むのは今回の実行のケースだけ。根拠は全件スナップショットのまま渡す。"""
    saved = _fake_repository(monkeypatch, {"A1": "unresolved"})
    cases = [_case("A1")]
    out = await ev.run_hallucination(_generation_dict(cases))

    assert [s["eval_id"] for s in saved] == ["A1"]
    assert saved[0]["citations"] == ev.build_citations(_KB_HITS)
    assert saved[0]["strategy"] == "hybrid_rerank"
    assert saved[0]["reason"] == "evidence に無い日数を答えている"
    assert saved[0]["judge_model"]
    assert cases[0]["saved"] is True
    assert cases[0]["seen_count"] == 2
    assert cases[0]["ledger_status"] == "unresolved"
    assert out["strategy"] == "hybrid_rerank"


async def test_confirmed_rate_counts_only_the_cases_a_person_called_a_hallucination(monkeypatch):
    """人手の判断は今回の書き込みで unresolved へ戻る前の状態で見る。

    no_action_needed(judge の行き過ぎ)は確認済みに数えない。
    """
    _fake_repository(monkeypatch, {"A1": "unresolved"}, previous_status="resolved")
    out = await ev.run_hallucination(_generation_dict([_case("A1")]))
    assert out["confirmed_rate"] == {"value": pytest.approx(1 / 300), "cases": 1, "n": 300}

    _fake_repository(monkeypatch, {"A1": "unresolved"}, previous_status="no_action_needed")
    out = await ev.run_hallucination(_generation_dict([_case("A1")]))
    assert out["confirmed_rate"] == {"value": 0.0, "cases": 0, "n": 300}
    assert out["rate"]["cases"] == 1


async def test_a_dead_database_does_not_change_the_numbers_of_this_run(monkeypatch):
    """台帳は追加の管理ビューであって評価そのものではない。

    DB が落ちていても make eval-rag は完走してレポートを出し切ること。
    """
    _fake_repository(monkeypatch, {}, upsert_error=RuntimeError("台帳の DB へ接続できない"),
                     map_error=RuntimeError("台帳の DB へ接続できない"))
    cases = [_case("A1")]
    out = await ev.run_hallucination(_generation_dict(cases))

    assert out["rate"] == {"value": pytest.approx(1 / 300), "cases": 1, "n": 300}
    assert out["ledger"] is None            # 累計だけが欠ける
    assert cases[0]["saved"] is False       # 画面が「積めなかった」と分かる形で残す
    assert any("台帳へ書けませんでした" in ln for ln in ev._LINES)


async def test_a_duplicate_eval_id_is_swallowed_like_any_other_write_failure(monkeypatch):
    """upsert_faith_case は同時書き込みを想定していない。衝突もここで握り潰す。"""
    from sqlalchemy.exc import IntegrityError

    _fake_repository(monkeypatch, {"A1": "resolved"},
                     upsert_error=IntegrityError("INSERT", {}, Exception("uk_eval_id")))
    cases = [_case("A1")]
    out = await ev.run_hallucination(_generation_dict(cases))

    assert out["rate"]["cases"] == 1
    assert cases[0]["saved"] is False
    # 積めなかったケースの人手の判断は、台帳に残っている前回までの状態で見る
    assert cases[0]["human_status"] == "resolved"
    assert out["confirmed_rate"]["cases"] == 1


def test_ledger_summary_reports_every_status_even_when_it_is_empty():
    """0 件の status も 0 として出す。キーが欠けると画面で「0 件」と「集計失敗」が
    区別できなくなる(list_faith_cases の counts と同じ方針)。"""
    assert ev.ledger_summary({}) == {
        "total": 0,
        "counts": {"unresolved": 0, "resolved": 0, "no_action_needed": 0},
    }
    assert ev.ledger_summary({"A1": "resolved", "B2": "resolved"})["counts"]["resolved"] == 2


def test_hallucination_rate_is_none_when_nothing_was_judged():
    """判定が 1 問も成立しなかった実行は 0% ではない(_mean と同じ扱い)。"""
    out = ev.build_hallucination([], 0, "hybrid_rerank", None)
    assert out["rate"] == {"value": None, "cases": 0, "n": 0}
    assert out["confirmed_rate"]["value"] is None


# ---------------------------------------------------------------------------
# 上流のレート制限
#
# 実測: 忠実性 judge のプロンプトを長くした回に TPM 上限(200k)を踏み、300 問中
# 57 問が判定されないまま捨てられた。表に出るのは分母が 243 へ減ったことだけで、
# 幻覚率はもっともらしい数字を出し続ける。だから「待てば通る」失敗だけは
# やり直さなければならない。
# ---------------------------------------------------------------------------

class _RateLimited(Exception):
    status_code = 429


async def test_rate_limited_calls_are_retried(monkeypatch):
    """レート制限は待ってやり直す。1 回で諦めると分母が黙って減る。"""
    monkeypatch.setattr(ev, "RATE_LIMIT_BACKOFF", 0)      # テストで実際に待たない
    calls = []

    async def _flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _RateLimited("Error code: 429 - rate_limit_exceeded")
        return "ok"

    ev._ERRS.clear()
    assert await ev._try(_flaky, "faithfulness[A1]") == "ok"
    assert len(calls) == 3
    assert ev._ERRS == []                                  # 成功したので記録もしない


async def test_other_failures_are_not_retried(monkeypatch):
    """レート制限以外は 1 回で諦める。壊れた入力を何度も投げても課金が増えるだけ。"""
    monkeypatch.setattr(ev, "RATE_LIMIT_BACKOFF", 0)
    calls = []

    async def _broken():
        calls.append(1)
        raise ValueError("そもそも入力が不正")

    ev._ERRS.clear()
    assert await ev._try(_broken, "generate[x/A1]") is None
    assert len(calls) == 1
    assert len(ev._ERRS) == 1


async def test_rate_limit_is_recognised_by_message_alone(monkeypatch):
    """プロバイダごとに例外の型が違うので、本文の 429 でも拾えること。"""
    monkeypatch.setattr(ev, "RATE_LIMIT_BACKOFF", 0)
    calls = []

    async def _flaky():
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("Error code: 429 - {'error': {'code': 'rate_limit_exceeded'}}")
        return "ok"

    ev._ERRS.clear()
    assert await ev._try(_flaky, "coverage[x/A1]") == "ok"
    assert len(calls) == 2


async def test_giving_up_after_the_last_attempt_is_recorded(monkeypatch):
    """やり直しても駄目なら記録して None。無限には待たない。"""
    monkeypatch.setattr(ev, "RATE_LIMIT_BACKOFF", 0)
    calls = []

    async def _always():
        calls.append(1)
        raise _RateLimited("429")

    ev._ERRS.clear()
    assert await ev._try(_always, "faithfulness[A1]") is None
    assert len(calls) == ev.RATE_LIMIT_ATTEMPTS
    assert len(ev._ERRS) == 1
