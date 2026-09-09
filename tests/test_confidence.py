"""09 確信度ゲート：4 つの信号の重み付き合成。純関数で再現する。

上流にも DB にも触らない。**この関数がターンの「答えるか断るか」を決める**ので、
入力が同じなら必ず同じ値が出ることと、値の向きが直感どおりであることを固定する。
"""

import pytest

from app.core.confidence import compute_evidence_confidence, snapshot_from_hits


def _hit(score, q="返品送料は誰が負担しますか",
         a="3,000 円以上で送料無料、返品送料は購入者負担"):
    return {"question": q, "answer": a, "rerank_score": score,
            "section_path": "アフターサービス / 返品", "id": 1, "content_type": "faq"}


def test_empty_hits_zero_confidence():
    """根拠が 0 件なら 0.0。signals も同じ形で返す(理由を組み立てる側が key を無条件に読む)。"""
    r = compute_evidence_confidence([])
    assert r.score == 0.0
    assert r.signals == {"top1_score": 0.0, "valid_count": 0,
                         "margin": 0.0, "key_clause_hit": False}


def test_strong_evidence_scores_high():
    r = compute_evidence_confidence([_hit(0.95), _hit(0.40), _hit(0.35)])
    assert r.score > 0.7
    assert r.signals["top1_score"] == 0.95
    assert r.signals["margin"] == pytest.approx(0.55)
    assert r.signals["key_clause_hit"] is True   # 「返品」「送料」が重要語に当たる


def test_weak_flat_evidence_scores_low():
    """似た候補が低いスコアで横並び。いちばん危ない形で、断るべきターン。"""
    hits = [_hit(0.22, q="キャットフードの味", a="サーモン味とチキン味"),
            _hit(0.21, q="キャットフードの味", a="サーモン味とチキン味")]
    r = compute_evidence_confidence(hits)
    assert r.score < 0.4
    assert r.signals["valid_count"] == 0          # すべて下限未満
    assert r.signals["key_clause_hit"] is False


def test_single_hit_margin_falls_back_to_top1():
    """1 件しか無いときの margin は Top1 自身。0 にすると孤立した強い根拠が不当に下がる。"""
    r = compute_evidence_confidence([_hit(0.8)])
    assert r.signals["margin"] == pytest.approx(0.8)


def test_score_monotonic_in_top1():
    """Top1 が強いほど点は高い。向きが逆転していないことの番人。"""
    low = compute_evidence_confidence([_hit(0.3), _hit(0.2)])
    high = compute_evidence_confidence([_hit(0.9), _hit(0.2)])
    assert high.score > low.score


def test_score_bounded_zero_one():
    r = compute_evidence_confidence([_hit(1.0), _hit(0.0)])
    assert 0.0 <= r.score <= 1.0


def test_snapshot_from_hits_top3_shape():
    hits = [_hit(0.9), _hit(0.8), _hit(0.7), _hit(0.6)]
    snap = snapshot_from_hits(hits)
    assert len(snap) == 3
    assert snap[0] == {"question": "返品送料は誰が負担しますか",
                       "answer": "3,000 円以上で送料無料、返品送料は購入者負担",
                       "rerank_score": 0.9,
                       "section_path": "アフターサービス / 返品"}


def test_snapshot_empty_hits():
    assert snapshot_from_hits([]) == []
