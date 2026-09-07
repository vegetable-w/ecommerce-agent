"""生成前の evidence 十分性セルフチェック(意味ゲート)。

rerank スコアによる機械ゲートを通過した根拠に対して、「この根拠だけで
質問に答えきれるか」をモデル自身に判定させる。useful=False なら
query_faq は回答拒否パスへ倒し、その質問を low_confidence_questions へ積む。
"""

import logging

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import SELF_CHECK_PROMPT

logger = logging.getLogger(__name__)


class _Check(BaseModel):
    useful: bool = Field(description="根拠が回答に十分かどうか")
    reason: str | None = Field(default="", description="判定の根拠を 1 文で")


async def check_sufficient(query: str, evidence_texts: list[str], model=None) -> dict:
    """{"useful": bool, "reason": str} を返す。

    flat field にしているのは扱いやすさのため(このプロジェクトの上流は
    nested object array も問題なく扱える。app/kb/mining.py の QaExtraction を参照)。

    **上流が落ちても例外を投げない。** ここは検索後の品質ゲートであり、
    通過した根拠は既に機械ゲート(settings.rerank_min_score)を越えている。
    チェックの失敗を理由に回答を拒否すると、上流の一時障害がそのまま
    「答えられません」+ 低信頼プールへの投入に化けてしまう。失敗時は
    useful=True で通し、最終的な回答拒否の判断は RAG_ANSWER_SYSTEM の
    回答拒否ルールに委ねる。
    """
    if not evidence_texts:
        return {"useful": False, "reason": "検索で根拠が 1 件も得られなかった"}

    evidence = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(evidence_texts))
    try:
        model = model or get_chat_model()
        chain = SELF_CHECK_PROMPT | model.with_structured_output(_Check)
        r: _Check = await chain.ainvoke({"query": query, "evidence": evidence})
    except Exception as exc:  # 上流エラー、タイムアウト、schema 違反のいずれも同じ扱い
        logger.warning(
            "evidence のセルフチェックに失敗したため十分とみなして続行する: %s: %s",
            type(exc).__name__, exc,
        )
        return {"useful": True, "reason": "セルフチェックの上流が応答しなかった"}
    return {"useful": bool(r.useful), "reason": (r.reason or "").strip()}
