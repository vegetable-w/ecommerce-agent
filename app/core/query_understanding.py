"""検索前のクエリ理解。口語の質問を標準質問へ正規化し、同義語を展開する。

結果は検索側にだけ効く。ナレッジベースに保存された内容は一切変更しない。
"""

import logging

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class _Rewrite(BaseModel):
    standard: str = Field(description="口語と感情を取り除いた 1 文の標準質問")
    expanded: list[str] = Field(default_factory=list,
                                description="キーワード検索用の同義語・言い換え・別称")


async def understand(query: str, model=None) -> dict:
    """{"standard": str, "expanded": list[str]} を返す。

    flat field にしているのは扱いやすさのため(このプロジェクトの上流は
    nested object array も問題なく扱える。app/kb/mining.py の QaExtraction を参照)。

    **上流が落ちても例外を投げない。** クエリ理解は検索精度を上げる前処理であり、
    ここで落ちると問い合わせ全体が落ちる。失敗時は生のクエリをそのまま standard として返し、
    呼び出し側は 03 章と同じ「ユーザーの原文で検索する」挙動へ縮退する。
    """
    try:
        model = model or get_chat_model()
        chain = QUERY_REWRITE_PROMPT | model.with_structured_output(_Rewrite)
        r: _Rewrite = await chain.ainvoke({"query": query})
        standard = (r.standard or "").strip() or query
        expanded = [w.strip() for w in (r.expanded or []) if isinstance(w, str) and w.strip()]
    except Exception as exc:  # 上流エラー、タイムアウト、schema 違反のいずれも同じ扱い
        logger.warning("クエリ理解に失敗したため原文で検索する: %s: %s", type(exc).__name__, exc)
        return {"standard": query, "expanded": []}
    return {"standard": standard, "expanded": expanded}
