"""満足度フィードバックの受け口。👎 をデータフライホイールへ繋ぐ(09 章)。

プールへ入る経路は 3 つあり、すべて low_confidence_questions に集まって source で
区別される(spec §4):

    retrieval_low_conf  根拠そのものが弱く、答えるのをやめた(確信度ゲート)
    self_check          根拠は取れたが、それでは答えきれないとモデルが判断した
    user_feedback       **答えたが外していた**。ここだけが人の操作から始まる

前の 2 つは agent が自分で気づけた失敗で、こちらは気づけなかった失敗。ナレッジの
穴としては後者の方が重いので、同じプールに置いて source で見分けられるようにする。

**👍 は DB へ保存しない(spec の明示的な非目標)。** 「貯めておけば後で使える」は
このプールの目的ではない。プールは「直すべき質問の一覧」であり、満足した質問が
混ざると、レビュー担当者はまず選り分けから始めることになる。数だけが要るなら log から
数えられる。
"""

import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.db import repository
from app.graph import runtime
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# プールへ残す理由。retrieval_low_conf / self_check の理由がスコアの内訳を持つのに対し、
# こちらに書けるのは「人が否と言った」という事実だけ(押した理由は訊いていない)。
_REASON = "ユーザーが回答に 👎 を付けた"


async def _turn_snapshot_for(conversation_id: int, question: str) -> list | None:
    """その質問のターンの検索の写し。取れなければ None。

    **best effort**。写しは後で人がレビュー画面で「ナレッジに無いのか、有るのに
    引けていないのか」を見分けるための材料であって、👎 を記録する条件ではない。

    checkpointer から取った質問と押された質問を突き合わせるのは、State が
    turn をまたいで残るため。押すのが遅れて次のターンが始まっていた場合、
    最新の State に載っているのは**別の質問**の検索結果になる。それを付けると、
    レビュー担当者は「この質問でこれを引いていた」という嘘の材料を読むことになる。
    付けないより悪いので、一致しないときは何も付けない。

    例外をここで握るのは、写しの取得が本筋(プールへの投入)より軽い仕事だから。
    checkpointer が開けていない、State が壊れている、といった事情でプールへの
    投入まで落とすと、いちばん拾いたい「答えたが外していた」質問が
    いちばん壊れているときに限って失われる。**runtime 側では握らない**:
    あちらは読めなかったことを呼び出し元へ伝えるだけにして、それを無視してよいか
    どうかは用途を知っているこちらが決める。
    """
    try:
        turn = await runtime.get_turn_snapshot(conversation_id)
    except Exception:
        # 例外の型を絞らない。checkpointer(sqlite)も State の中身も上流の都合で
        # 変わりうるので、ここで型を並べると、並べ忘れた 1 つが投入を落とす。
        logger.warning("検索の写しを取得できなかった conv=%s", conversation_id,
                       exc_info=True)
        return None
    if turn.get("question") != question:
        logger.info("👎 の質問が最新ターンと一致しないため写しを付けない conv=%s",
                    conversation_id)
        return None
    # 空の写しは [] ではなく None で渡す。DDL はこの列の NULL を「検索を通っていない」
    # の意味で使っており(app/db/models.py)、[] を入れると JSON の [] として保存され、
    # `retrieved_chunks IS NULL` で数える側から漏れる(fallback_reply と同じ扱い)。
    return turn.get("snapshot") or None


@router.post("/api/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    """👍 / 👎 を受け取る。👎 のときだけ質問を低信頼プールへ積む。

    **同じ会話の同じ質問が既にプールに居たら積まない。** agent が自分で断ったターン
    (retrieval_low_conf / self_check)は fallback_reply が既に 1 行積んでいて、その
    断りの吹き出しにも満足度バーが出る。そこへ 👎 が付くと同じ質問が 2 行になり、
    flywheel は 2 行を同じ穴へまとめるので、**1 ターンで occurrence_count が 2 になる。**
    この列はレビューキューの並び順=優先度そのものなので、断ったターンだけが二重に
    重み付けされることになる(正規化のためのモデル呼び出しも 1 ターンで 2 回になる)。
    そもそも user_feedback が意味するのは「答えたが外していた」失敗であって、
    断りに付いた 👎 はこの source の定義に当たらない。

    **pooled は実際に積んだかどうかを返す。** 押した事実(200)と、プールが 1 行
    増えたことは別。

    runtime と repository は属性経由で呼ぶこと(from ... import しない)。テストの
    monkeypatch.setattr が効かなくなり、本番相当の DB と本物の checkpointer へ
    流れ落ちる。
    """
    if req.rating == "up":
        # 保存しないので、後から数えられる場所はここだけ。会話 ID を残す。
        logger.info("👍 のフィードバック conv=%s", req.conversation_id)
        return FeedbackResponse(ok=True, pooled=False)

    chunks = await _turn_snapshot_for(req.conversation_id, req.question)
    try:
        # 重複の判定も同じ try の中で行う。DB が落ちていれば投入も同じく落ちるので、
        # 押した人へ返す答え(503)は同じでなければならない。
        if await repository.low_confidence_exists(req.conversation_id, req.question):
            logger.info("同じ質問が既にプールにあるため積まない conv=%s",
                        req.conversation_id)
            return FeedbackResponse(ok=True, pooled=False)
        await repository.insert_low_confidence(
            req.conversation_id, req.question, "user_feedback", _REASON,
            retrieved_chunks=chunks,
        )
    except SQLAlchemyError:
        # 例外そのものはログにだけ残す。detail へ載せると接続文字列や SQL が画面まで
        # 届く(app/api/actions.py と同じ規約)。存在しない conversation_id は
        # insert_low_confidence が会話への紐付けを外して積み直すので、ここまで来る
        # のは本当に書けなかったときだけ。会話の有無を確かめ直す必要は無い。
        logger.exception("フィードバックをプールへ積めなかった conv=%s",
                         req.conversation_id)
        raise HTTPException(
            status_code=503,
            detail="フィードバックを一時的に受け付けられません。しばらくしてからもう一度お試しください",
        )
    return FeedbackResponse(ok=True, pooled=True)
