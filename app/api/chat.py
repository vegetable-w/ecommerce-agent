import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.core.llm import get_chat_model
from app.core.memory import SessionStore, trim_history
from app.core.prompts import CUSTOMER_SERVICE_PROMPT
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()
store = SessionStore()

def get_model() -> BaseChatModel:
    return get_chat_model(streaming=True)

# 既知の制約(意図的に対処しない): この関数は開始時に履歴を読み込み、astreamループが
# 完了した後にのみ履歴を追記する。ループ中には多数のawaitがあるため、同一session_idへの
# 同時リクエストは追記順が入り乱れる可能性がある(データ破損や消失はしないが、
# 保存順序が先に完了したリクエスト次第になり、互いのターンを見ないまま応答する)。
# この章ではセッションあたり単一クライアントを前提とし、ロックは範囲外とする。
@router.post("/api/chat")
async def chat(req: ChatRequest, model: BaseChatModel = Depends(get_model)):
    history = [*store.get(req.session_id), HumanMessage(req.message)]
    trimmed = trim_history(history, max_tokens=settings.token_budget)
    if len(trimmed) < len(history):
        # メッセージ本文はユーザーデータのためログに含めない
        logger.info(
            "履歴をトリムしたためターンを破棄 session_id=%s dropped=%d",
            req.session_id,
            len(history) - len(trimmed),
        )
    messages = CUSTOMER_SERVICE_PROMPT.format_messages(history=trimmed)

    async def event_stream() -> AsyncIterator[str]:
        chunks: list[str] = []
        try:
            async for chunk in model.astream(messages):
                text = chunk.content if isinstance(chunk.content, str) else ""
                if not text:
                    continue
                chunks.append(text)
                yield f"data: {json.dumps({'delta': text}, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("上流LLMのストリーミング呼び出しに失敗 session_id=%s", req.session_id)
            yield "event: error\n"
            yield f"data: {json.dumps({'message': '上流モデルは一時的に利用できません。しばらくしてから再試行してください'}, ensure_ascii=False)}\n\n"
            return
        store.append(req.session_id, HumanMessage(req.message), AIMessage("".join(chunks)))
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
