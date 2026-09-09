"""データフライホイールの pipeline: 生の質問 → 正規化と重複判定 → レビュー待ち (09 Task 8)。

低信頼プールに溜まるのは、口語も感情も無関係な detail も付いたままの**生の質問**。
そのままではレビューする人が同じ穴を何度も読むことになり、承認してもナレッジベースへ
書き戻せる形にならない。ここが FAQ 形式の 1 文へ均し、既に積んである穴と重なるものを
まとめて、レビューの単位を作る。

バッチの意味づけ(定期実行を前提にする。リアルタイムには寄せない):

- **カーソルは `low_confidence_questions.matched_review_id IS NULL`。** 処理し終えた行に
  まとめ先を書き戻す。処理済みの印とまとめ先が同じ 1 列なので、途中で落ちても
  次の実行が続きから再開でき、再実行しても二重に積まれない。
- **1 行ずつ直列に処理する。** 同じバッチの 2 件目は、1 件目が作ったばかりの行を候補として
  見られる。並列にしたり候補をバッチの先頭で 1 度だけ取ったりすると、同義の質問 2 件が
  互いを見ないまま別々の穴を作る。
- **候補の status に関わらず、match したら occurrence_count を +1 するだけ。**
  status は動かさない。`rejected` は人が下した最終判断であり、同じ質問がまた来たからと
  いって pending へ戻してよいものではない(それでも回数は数える。何度も来る却下済みの穴は
  「却下の判断が妥当か」を見直す材料になる)。
- **パース失敗と存在しない id は、その行を skip して警告する。** カーソルは進めない。
  次の実行でやり直せば済むので、ここで穴を作ったり中途半端に紐づけたりしない方が安全。
"""

import logging

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import FLYWHEEL_NORMALIZE_PROMPT
from app.db import repository

logger = logging.getLogger(__name__)

# 重複判定でモデルへ見せる候補の上限。repository.list_review_candidates の既定と同じ値を
# ここで明示して渡す。既定任せにすると、あとで repository 側だけ変わったときに
# 「頭打ちを検知する閾値」だけが古い値のまま残り、警告が出なくなる。
CANDIDATE_LIMIT = 200


class NormalizeResult(BaseModel):
    normalized_question: str = Field(description="FAQ 形式に均した標準質問")
    matched_question_id: int | None = Field(
        default=None, description="重なった候補の id。同じ intent の候補が無ければ null"
    )
    ai_suggested_answer: str = Field(default="", description="レビューの参考にする回答例")


def _chain():
    return FLYWHEEL_NORMALIZE_PROMPT | get_chat_model().with_structured_output(NormalizeResult)


async def normalize_and_match(raw_question: str, candidates: list[dict]) -> NormalizeResult:
    """1 回の呼び出しで {正規化した質問, まとめ先の候補 id, 参考回答} を返す。

    候補が空でも「候補なし」と明示して渡す。空文字のまま渡すと、prompt の
    候補欄が丸ごと消えて指示 2 の対象が無いように読め、モデルが id を捏造しやすくなる。
    """
    cand_text = (
        "\n".join(f"- id={c['id']}: {c['normalized_question']}" for c in candidates)
        or "(候補なし)"
    )
    return await _chain().ainvoke({"raw_question": raw_question, "candidates": cand_text})


async def process_pending(limit: int = 50) -> dict:
    """未処理の生の質問を古い順に処理し、今回の内訳を返す。

    戻り値は {"processed", "merged", "created", "skipped"}。processed は
    merged + created と一致する(skipped はカーソルを進めないので processed に入らない)。
    """
    rows = await repository.fetch_unmatched_low_conf(limit)
    stats = {"processed": 0, "merged": 0, "created": 0, "skipped": 0}
    truncation_warned = False

    for row in rows:
        # 行ごとに取り直す。同じバッチで作ったばかりの穴も候補に入れるため。
        candidates = await repository.list_review_candidates(CANDIDATE_LIMIT)
        if len(candidates) >= CANDIDATE_LIMIT and not truncation_warned:
            # 黙って落とさない。候補が上限で切られている間は、切り捨てられた古い穴と
            # 同じ質問が来ても match できず、重複した穴が新しく積まれる。
            # レビューが追いつかず穴が溜まっていることの合図でもある。
            logger.warning(
                "重複判定の候補が上限 %s 件で頭打ちになっている。"
                "これを超える古い穴は突き合わせの対象から外れ、重複した穴が積まれる",
                CANDIDATE_LIMIT,
            )
            truncation_warned = True

        try:
            result = await normalize_and_match(row.raw_question, candidates)
        except Exception:
            # 上流エラーもスキーマ違反も同じ扱い。カーソルを進めないので次回やり直す。
            logger.warning(
                "正規化に失敗したため lcq=%s を今回は飛ばす(次回の実行でやり直す)",
                row.id, exc_info=True,
            )
            stats["skipped"] += 1
            continue

        mid = result.matched_question_id
        if mid is not None:
            if not any(c["id"] == mid for c in candidates):
                # 候補に無い id。ここで信じると、存在しない穴や無関係な穴へ
                # 生の質問を紐づけてしまい、しかもカーソルが進むので二度と直せない。
                logger.warning(
                    "候補一覧に無い id=%s が返ったため lcq=%s を今回は飛ばす"
                    "(次回の実行でやり直す)",
                    mid, row.id,
                )
                stats["skipped"] += 1
                continue
            await repository.increment_occurrence(mid)
            review_id = mid
            stats["merged"] += 1
        else:
            review_id = await repository.insert_review_item(
                result.normalized_question, result.ai_suggested_answer or None
            )
            stats["created"] += 1

        # 出現回数を進めた/穴を作ったあとに紐づける。順序を逆にすると、間で落ちたときに
        # 「処理済みなのに数えられていない」行が残る。
        await repository.set_matched_review(row.id, review_id)
        stats["processed"] += 1
        logger.info(
            "lcq=%s を review=%s へ%s: %s",
            row.id, review_id, "まとめた" if mid is not None else "新しく積んだ",
            result.normalized_question,
        )

    return stats
