"""画面のボタンから叩く action エンドポイント。

**tickets テーブルへ書き込む経路をここ 1 か所に閉じ込める**のがこのモジュールの役目。
Agent は create_ticket tool を「チケットを作るべきだ」という意思表示に使うが、
agent_tools がそれを横取りして選択肢へ変換するので DB には届かない(spec §6)。
書き込みが起きるのは、ユーザーが画面のボタンを押してここへ POST したときだけ。

有人対応は本章では画面上の見た目のみで、backend の処理は無い(spec §6.2)。
"""

import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.core import labels
from app.db import repository
from app.schemas.actions import CreateTicketRequest, CreateTicketResponse

logger = logging.getLogger(__name__)
router = APIRouter()


async def _conversation_missing(conversation_id: int) -> bool:
    """会話が存在しないかどうか。判定できなければ「存在しない」とは言わない。

    この確認自体が DB 障害で落ちることがある。そのときに「会話が無い」と答えると、
    実際には存在する会話に対して 404 を返してしまうので、判断を保留して False を返し、
    呼び出し元の 503 に委ねる。
    """
    try:
        return await repository.get_conversation(conversation_id) is None
    except SQLAlchemyError:
        return False


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket_action(req: CreateTicketRequest) -> CreateTicketResponse:
    """チケット作成ボタン。ユーザーが押したときだけ tickets へ書く。

    種別と内容の検証は CreateTicketRequest に任せる(ここで if 文を足さない)。
    許容値が OpenAPI の enum として表に出ることと、弾かれた理由が 422 の本文に
    載ることが、画面側にとっての契約になる。
    """
    try:
        ticket_no = await repository.create_ticket(
            req.conversation_id, req.description, req.ticket_type
        )
    except SQLAlchemyError:
        # 例外そのものはログにだけ残す。detail へ載せると接続文字列や SQL が画面まで届く。
        logger.exception("チケット作成に失敗 conv=%s", req.conversation_id)
        # 存在しない conversation_id は FK 制約違反として同じ except に落ちるが、
        # それは「今は無理」ではなく「何度やっても無理」なので 503 では嘘になる。
        # 再試行を促す文面を返すと利用者を無駄に待たせ、監視側にも DB 不調として
        # 積み上がる。確認の 1 往復は**失敗したときだけ**払う(正常系は素通り)。
        if await _conversation_missing(req.conversation_id):
            raise HTTPException(status_code=404, detail="会話が見つかりません")
        raise HTTPException(
            status_code=503,
            detail="チケットを一時的に作成できません。しばらくしてからもう一度お試しください",
        )
    # repository.create_ticket は会話の status も escalated へ動かす。画面へ返すのは
    # その日本語ラベルで、02 章の create_ticket tool と同じ対応表から引く。
    # ここで文字列を書き起こすと、表示名の出所が 2 つになる。
    return CreateTicketResponse(
        ticket_no=ticket_no,
        status=labels.label(labels.CONVERSATION_STATUS, "escalated"),
    )
