"""評価セット tests/data/eval_04.jsonl を、実際に build される chunk 群に対して検証する。

評価セットはコードではなくデータなので、単体テストではなく**照合**で守る。守りたい失敗は
03 章で実際に起きた 3 つ:

1. 期待値が「意図した chunk 以外」でも満たされてしまう。03 章の検索 eval は 8/8 だったが、
   「買ったものを返したい」は正解ではなく「返品対象外の商品」(返せない物の一覧)に当たって
   いた。判定文字列「返品」がその見出しにも含まれていたからで、2/8 は偶然の部分一致で
   通っていた。→ expect_section が section_path 1 つだけに当たることを機械的に確かめる。
2. 期待した key point が原文に存在しない。存在しない point は永久に cover されないので、
   気付かないまま coverage の上限が下がる。→ 対象 chunk の本文に実在することを確かめる。
3. C bucket が「見出しそのもの」になっている。見出しを丸写しした質問は口語耐性を測れない。
   → 質問と見出しの最長共通部分文字列が短いことを確かめる。

複数根拠 (expect_sections_all) の問いにも同じ検査が掛かる。節の一意性はグループごとに見て、
さらに 4 つ目として「根拠ごとに key point を 1 つ以上持つこと」を確かめる。片方の文書だけで
満点が取れる出題では cross-document を測れないため。

Milvus にも上流にも触れない。data/kb/*.md を読んで chunk に組み立てるだけ。
"""

import json
import pathlib

import pytest

from app.kb import documents, sources

EVAL_SET = pathlib.Path(__file__).resolve().parent / "data" / "eval_04.jsonl"
BUCKETS = ("A_policy", "B_model", "C_colloquial", "D_absent", "E_multi")
# 04 章の評価セットの最終形。5 bucket × 60 問 = 300 問。
_PER_BUCKET = 60
# C bucket の質問と対象見出しが共有してよい最長の連続文字数。
# 「送料」(2 文字)のような話題語の重なりは避けられないが、「送料はいくらですか」のような
# 見出しの丸写しは口語耐性を測れないので落とす。
_MAX_TITLE_OVERLAP = 4


def _norm(s: str) -> str:
    return "".join((s or "").split())


def _longest_common_substring(a: str, b: str) -> str:
    best = ""
    for i in range(len(a)):
        for j in range(i + len(best) + 1, len(a) + 1):
            if a[i:j] in b:
                best = a[i:j]
            else:
                break
    return best


@pytest.fixture(scope="module")
def chunks():
    out = []
    for name, content_type in sources.SOURCE_TYPES.items():
        md = sources.source_path(name).read_text(encoding="utf-8")
        out += documents.build_chunks(md, content_type)
    return out


@pytest.fixture(scope="module")
def samples():
    return [json.loads(ln) for ln in EVAL_SET.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def _targets(chunks, group):
    """グループの全語を section_path に含む chunk。検索側の判定と同じ AND。"""
    return [c for c in chunks if all(k in c.section_path for k in group)]


def _groups(sample):
    """1 問の期待値を「グループのリスト」へ揃える。グループ 1 つが根拠 chunk 1 つ。

    単一根拠は expect_section(グループ 1 個)、複数根拠は expect_sections_all
    (グループ複数)。scripts/eval_04.py の as_groups と同じ揃え方にして、採点側と
    この検査とで「どこが正解か」の解釈がずれないようにする。ずれると
    「検査は通るのに Recall が 0」という最悪の食い違いが起きる。
    """
    expect = sample.get("expect_sections_all") or sample.get("expect_section") or []
    if all(isinstance(e, str) for e in expect):
        return [list(expect)] if expect else []
    return [list(g) for g in expect if g]


def test_bucket_counts(samples):
    counts = {b: sum(1 for s in samples if s["bucket"] == b) for b in BUCKETS}
    assert counts == {b: _PER_BUCKET for b in BUCKETS}
    total = _PER_BUCKET * len(BUCKETS)
    assert len(samples) == total
    assert len({s["id"] for s in samples}) == total


def test_every_sample_has_required_fields(samples):
    for s in samples:
        assert s["query"].strip(), s["id"]
        assert isinstance(s["should_refuse"], bool), s["id"]
        # 正解の書き方は 2 つ。単一根拠は expect_section、複数根拠は expect_sections_all。
        # 両方書くと片方の直し忘れが静かに効いてしまうので、どちらか一方だけにする。
        keys = {"expect_section", "expect_sections_all"} & set(s)
        assert len(keys) == 1, f"{s['id']}: 正解の書き方は片方だけにする({sorted(keys)})"
        assert isinstance(s[keys.pop()], list) and isinstance(s["expect_points"], list), s["id"]


def test_graded_buckets_have_expectations_and_d_bucket_has_none(samples):
    for s in samples:
        if s["bucket"] == "D_absent":
            assert _groups(s) == [] and s["expect_points"] == [], s["id"]
            assert s["should_refuse"] is True, s["id"]
        else:
            assert _groups(s), s["id"]
            assert s["expect_points"], s["id"]
            assert s["should_refuse"] is False, s["id"]


def test_no_duplicate_queries(samples):
    queries = [_norm(s["query"]) for s in samples]
    assert len(set(queries)) == len(queries), "同じ質問を水増しで並べていないこと"


def test_expect_section_matches_exactly_one_section_path(chunks, samples):
    """期待した節が 1 つに定まること。複数根拠の問いはグループごとに定まること。

    複数の section_path に当たる語(例:「送料」は 4 節に当たる)を期待値にすると、
    Recall が「正解を引けた」ではなく「似た節を引けた」を測ってしまう。
    表が分割された節のように**同じ section_path の chunk が複数ある**のは正常なので、
    数えるのは chunk 数ではなく distinct な section_path。
    """
    bad = []
    for s in samples:
        if s["bucket"] == "D_absent":
            continue
        for i, group in enumerate(_groups(s), 1):
            paths = {c.section_path for c in _targets(chunks, group)}
            if len(paths) != 1:
                bad.append(f"{s['id']}(グループ {i}): {group} → {len(paths)} 節 "
                           f"{sorted(paths)[:3]}")
    assert not bad, "\n".join(bad)


def test_expect_points_exist_verbatim_in_target_chunk(chunks, samples):
    """key point が対象 chunk の本文に実在すること(空白を無視した部分一致)。

    存在しない point は evidence coverage で永久に 0 になり、点数を静かに押し下げる。
    """
    bad = []
    for s in samples:
        if s["bucket"] == "D_absent":
            continue
        body = _norm("\n".join(c.answer for g in _groups(s) for c in _targets(chunks, g)))
        for p in s["expect_points"]:
            if _norm(p) not in body:
                bad.append(f"{s['id']}: {p!r} が対象 chunk 本文に無い")
    assert not bad, "\n".join(bad)


def test_multi_evidence_points_cover_every_group(chunks, samples):
    """複数根拠の問いは、根拠ごとに key point を 1 つ以上持つこと。

    point がすべて片方の文書から取られていると、もう片方を 1 度も引かなくても
    evidence coverage が満点になり、「複数の文書をまたげたか」を測れなくなる。
    根拠の数も 2〜3 に収める(1 つなら単一根拠の書き方で足りる)。
    """
    bad = []
    for s in samples:
        groups = _groups(s)
        if "expect_sections_all" not in s:
            continue
        if not 2 <= len(groups) <= 3:
            bad.append(f"{s['id']}: 根拠が {len(groups)} 個(2〜3 個であること)")
        if len(s["expect_points"]) < len(groups):
            bad.append(f"{s['id']}: 根拠 {len(groups)} 個に対し "
                       f"key point が {len(s['expect_points'])} 個しかない")
        for i, group in enumerate(groups, 1):
            body = _norm("\n".join(c.answer for c in _targets(chunks, group)))
            if not any(_norm(p) in body for p in s["expect_points"]):
                bad.append(f"{s['id']}(グループ {i}): {group} の本文から取った "
                           "key point が 1 つも無い")
    assert not bad, "\n".join(bad)


def test_colloquial_queries_are_not_the_heading(chunks, samples):
    """C bucket の質問が見出しの丸写しになっていないこと。"""
    bad = []
    for s in samples:
        if s["bucket"] != "C_colloquial":
            continue
        title = sorted({c.section_path for c in _targets(chunks, _groups(s)[0])})[0]
        title = title.split(" / ")[-1]
        common = _longest_common_substring(s["query"], title)
        if len(common) > _MAX_TITLE_OVERLAP:
            bad.append(f"{s['id']}: 見出し {title!r} と {len(common)} 文字共有 {common!r}")
    assert not bad, "\n".join(bad)


def test_absent_bucket_terms_are_really_absent_from_the_documents():
    """D bucket の主題語がナレッジ資料に 1 度も出てこないこと。

    03 章では「モデルが正しく断ったのにラベルが『答えられる』だった」ことがあった
    (間違っていたのはモデルではなくラベル)。ここでは逆向きに、断るべきと決めた話題が
    本当に資料へ書かれていないことを資料の側から確かめる。

    D13(ポイントの有効期限)/ D15(ランク昇格の金額)/ D16(EC-RV300 の価格)は
    隣接する記述がある(「累計購入金額に応じて自動的にランクアップ」「価格保証」)ため
    語の不在では示せない。その 3 問は下の資料側の性質で担保する。
    """
    text = "\n".join(sources.source_path(n).read_text(encoding="utf-8")
                     for n in sources.SOURCE_TYPES)
    for term in ["代金引換", "日時指定", "置き配", "店頭", "実店舗", "ギフト", "包装",
                 "メッセージカード", "パスワード", "訪問設置", "下取り", "定期購入",
                 "学生", "分割払い", "リボ", "掛け払い", "有効期限", "譲渡",
                 "EC-RV500", "交換用フィルター", "電話", "代替機",
                 "レビュー", "メールマガジン", "配信", "退会", "二段階認証", "個人情報",
                 "買い物かご", "未成年", "追跡番号", "運送会社", "配送業者", "再配達",
                 "段ボール", "ギフトカード", "商品券", "電子マネー", "誕生日", "抽選",
                 "中古", "アウトレット", "在庫数", "ダウンロード", "iOS", "保証書",
                 "クーリングオフ", "収入印紙", "但し書き", "納品書", "インボイス",
                 "見積", "年末年始", "休業", "会社概要", "アフィリエイト"]:
        assert term not in text, f"D bucket の主題語 {term!r} が資料に存在する"

    spec = sources.source_path("product-spec-manual.md").read_text(encoding="utf-8")
    assert "円" not in spec, "仕様マニュアルに価格が載ると D16 が答えられる質問になる"
    faq = sources.source_path("product-faq.md").read_text(encoding="utf-8")
    assert "累計購入金額に応じて自動的にランクアップする" in faq
    assert "シルバー会員になる" not in faq, "昇格の条件が書かれると D15 が答えられる質問になる"
