from datetime import datetime

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import MINING_PROMPT
from app.db import repository
from app.kb import dedup, dualwrite
from app.kb.documents import Chunk


class QaPair(BaseModel):
    question: str = Field(description="注文番号や氏名を除いた一般的な質問文")
    answer: str = Field(description="オペレーターの回答に忠実な回答。捏造しない")


class QaExtraction(BaseModel):
    pairs: list[QaPair] = Field(default_factory=list, description="抽出した問答ペア。空でもよい")


async def extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]:
    model = model or get_chat_model()
    chain = MINING_PROMPT | model.with_structured_output(QaExtraction)
    result: QaExtraction = await chain.ainvoke({"conversations": "\n---\n".join(conversation_texts)})
    return result.pairs


async def _load_conversation_texts() -> list[tuple[str, str]]:
    """[(source_ref, 会話テキスト)] を返す。会話テキストはその会話の user/assistant 発話を連結したもの。"""
    convs = await repository.list_conversations_with_messages()
    out = []
    for conv_id, msgs in convs:
        lines = [f"{m.role}: {m.content}" for m in msgs if m.content]
        if lines:
            out.append((f"conv:{conv_id}", "\n".join(lines)))
    return out


async def mine(batch_size: int = 20, model=None) -> dict:
    sources = await _load_conversation_texts()
    batch_no = datetime.now().strftime("mine-%Y%m%d-%H%M%S")
    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        pairs = await extract_qa([t for _, t in batch], model=model)
        for p in pairs:
            await repository.insert_staging(batch_no, batch[0][0], p.question, p.answer)
    staged = await repository.list_staging_by_status("extracted")
    existing = await repository.list_all_questions()
    kept, discarded = dedup.dedupe(staged, existing)
    await repository.set_staging_status([s.id for s in kept], "kept")
    await repository.set_staging_status([s.id for s in discarded], "discarded")
    chunks = [Chunk(category="過去の会話", questions=s.question, answer=s.answer,
                    section_path="mined", content_type="mined") for s in kept]
    if chunks:
        await dualwrite.write_pending(chunks)
    return {"sources": len(sources), "extracted": len(staged),
            "kept": len(kept), "discarded": len(discarded)}
