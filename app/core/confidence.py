"""09 根拠の確信度：この根拠の集合が質問にどれだけ噛み合っているかを数値にする。

見る信号は 4 つ。**どれも検索とリランクの結果から取れるもので、モデルを追加で呼ばない**。
確信度の判定にモデル呼び出しを足すと、そのターンの遅さと不安定さがそのまま
「答えるか断るか」の判断に乗ってしまう。

  top1_score     リランク Top1 のスコア。いちばん強い根拠がどれだけ強いか
  valid_count    スコアが下限以上の件数。強い根拠が 1 本だけなのか複数あるのか
  margin         Top1 と Top2 の差。狙い澄ました 1 件なのか、似た候補が横並びなのか
                 (1 件しか無いときは Top1 自身を使う)
  key_clause_hit Top3 の本文が重要語に当たるか。03 章の取り込みで使っている
                 `_KEY_TERMS` をそのまま再利用する。取り込み側と判定の物差しを
                 揃えるためで、ここで別の語彙を作ると「重要と見なして取り込んだのに
                 重要と見なされない」というずれが生まれる

重み付けは固定で、**設定には出さない**。評価セットで調整するのは
`settings.evidence_confidence_threshold`(どこで切るか)であって重みではない。
調整の層を 2 つ作ると、どちらを動かしたのか後から誰も説明できなくなる。
"""

from dataclasses import dataclass

from app.kb.documents import _KEY_TERMS

VALID_SCORE_FLOOR = 0.3   # 有効な根拠と見なす下限(04 章の rerank_min_score の経験値を引き継ぐ)
VALID_COUNT_CAP = 3       # 件数の頭打ち。3 件以上は満点として扱う
W_TOP1, W_VALID, W_MARGIN, W_KEY = 0.5, 0.2, 0.2, 0.1


@dataclass
class EvidenceConfidence:
    score: float    # 0-1 の総合点
    signals: dict   # 生の信号。プールへ入れる理由と trace の両方で読む


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_evidence_confidence(hits: list[dict]) -> EvidenceConfidence:
    """検索結果から確信度を出す。純関数で、同じ入力からは必ず同じ値が出る。

    根拠が 0 件のときに 0.0 を返すのは当然だが、**signals も同じ形で返す**こと。
    呼び出し側は理由を組み立てるときに key を無条件で読むので、ここだけ形が
    変わると根拠が無かったターンのプール行だけ理由が欠ける。
    """
    if not hits:
        return EvidenceConfidence(0.0, {"top1_score": 0.0, "valid_count": 0,
                                        "margin": 0.0, "key_clause_hit": False})
    scores = [float(h.get("rerank_score", 0.0)) for h in hits]
    top1 = scores[0]
    margin = top1 - scores[1] if len(scores) > 1 else top1
    valid_count = sum(1 for s in scores if s >= VALID_SCORE_FLOOR)
    key_hit = any(
        any(t in f"{h.get('question', '')}{h.get('answer', '')}" for t in _KEY_TERMS)
        for h in hits[:3]
    )
    score = (W_TOP1 * _clip01(top1)
             + W_VALID * min(valid_count, VALID_COUNT_CAP) / VALID_COUNT_CAP
             + W_MARGIN * _clip01(margin)
             + W_KEY * (1.0 if key_hit else 0.0))
    return EvidenceConfidence(round(_clip01(score), 4),
                              {"top1_score": top1, "valid_count": valid_count,
                               "margin": round(margin, 4), "key_clause_hit": key_hit})


def snapshot_from_hits(hits: list[dict], top_n: int = 3) -> list[dict]:
    """プールへ残す検索の写し。Top N の本文とスコアだけを取る。

    後で人がレビュー画面で読み、「ナレッジに本当に無いのか、有るのに引けていないのか」を
    見分けるための材料。全文ではなく件数を絞るのは、1 件の質問に対して検索結果を
    丸ごと持つと、プールが検索ログの置き場所になってしまうため。
    """
    return [{"question": h.get("question", ""), "answer": h.get("answer", ""),
             "rerank_score": float(h.get("rerank_score", 0.0)),
             "section_path": h.get("section_path", "")}
            for h in hits[:top_n]]
