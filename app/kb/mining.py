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


_MINED_ROLES = ("user", "assistant")
_SOURCE_REF_MAX = 255  # qa_extraction_staging.source_ref の桁数


async def _load_conversation_texts() -> list[tuple[str, str]]:
    """[(source_ref, 会話テキスト)] を返す。

    role を user / assistant に絞る。02 章では tool の実行結果も messages に入っており、
    中身は `{"hits": [...]}` のような生 JSON なので、抽出プロンプトに混ぜても
    ノイズにしかならない(content の有無だけで絞ると混入する)。
    content が空の行(assistant がツール呼び出しだけ行ったターン)も落とす。
    """
    convs = await repository.list_conversations_with_messages()
    out = []
    for conv_id, msgs in convs:
        lines = [f"{m.role}: {m.content}" for m in msgs if m.content and m.role in _MINED_ROLES]
        if lines:
            out.append((f"conv:{conv_id}", "\n".join(lines)))
    return out


def _batch_source_ref(refs: list[str]) -> str:
    """バッチ内の全会話 ID を連結して source_ref にする。

    先頭 1 件だけを入れると、既定の batch_size=20 では最大 19 件が
    「別の会話に由来する」と誤って記録される。source_ref は「この問答はどの会話から来たか」を
    後で人が辿るための欄なので、自信を持って間違えるより、候補を全部挙げるほうがよい。
    桁数を超える場合は入る分だけ並べて件数を添える。
    """
    joined = ",".join(refs)
    if len(joined) <= _SOURCE_REF_MAX:
        return joined
    suffix = f"...(全{len(refs)}件)"
    room = _SOURCE_REF_MAX - len(suffix)
    return joined[:room].rsplit(",", 1)[0] + suffix


async def mine(batch_size: int = 20, model=None) -> dict:
    sources = await _load_conversation_texts()
    batch_no = datetime.now().strftime("mine-%Y%m%d-%H%M%S")
    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        pairs = await extract_qa([t for _, t in batch], model=model)
        source_ref = _batch_source_ref([ref for ref, _ in batch])
        for p in pairs:
            await repository.insert_staging(batch_no, source_ref, p.question, p.answer)
    # extracted は batch_no で絞らず全件を対象にする。前回の実行が途中で落ちて
    # extracted のまま残った行も、ここで拾って重複排除・昇格まで進めるため(復旧を兼ねる)。
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
