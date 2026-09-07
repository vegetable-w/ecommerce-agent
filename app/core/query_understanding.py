"""検索前のクエリ理解。口語の質問を標準質問へ正規化し、同義語を展開する。

結果は検索側にだけ効く。ナレッジベースに保存された内容は一切変更しない。

2 つの関数は目的が違う。understand(05 章)は 1 本の検索テキストを同義語で太らせる。
expand_queries(06 章)は検索そのものを観点の違う 3 本に分ける。後者は返金・
アフターサービスの検索でだけ使い、単純な FAQ には使わない(1 回で当たる質問に
3 回検索しても、上流を 3 回叩くだけで結果は変わらない)。
"""

import logging

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import EXPAND_QUERIES_PROMPT, QUERY_REWRITE_PROMPT

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


class _Expanded(BaseModel):
    # nested にせず field 1 つの flat な list[str] にする。入れ子の output schema は
    # 上流が 502 を返すことがある(app/core/intent.py の flat 2 field と同じ理由)
    queries: list[str] = Field(default_factory=list,
                               description="観点の異なる検索クエリちょうど 3 件")


async def expand_queries(query: str, model=None) -> list[str]:
    """検索に向いた 3 件のクエリへ展開する。失敗しても検索を止めない。

    **重複は除く。** モデルは語順だけ違う似た文を返しがちで、同じクエリを 3 回
    投げても検索結果は 1 通りしか増えない。除いた結果 3 件に満たないことはあるが、
    水増しして件数を揃えるより、実際に観点の違うものだけを返す方がよい。

    **上流が落ちても例外を投げない。** 展開は検索の当たりを増やす前処理であり、
    ここで落とすと返金フローの検索そのものが止まる。失敗時は原文 1 件へ縮退し、
    呼び出し側は「展開なしで 1 回検索する」挙動になる。握るときは warning を 1 行残す
    (黙って縮退すると、上流障害が検索精度の劣化にしか見えなくなる)。

    model は単体テストからの注入口(understand / classify と同じ形)。
    """
    try:
        model = model or get_chat_model(temperature=0)
        chain = EXPAND_QUERIES_PROMPT | model.with_structured_output(_Expanded)
        r: _Expanded = await chain.ainvoke({"query": query})
        raw = [q.strip() for q in (r.queries or []) if isinstance(q, str) and q.strip()]
    except Exception as exc:  # 上流エラー、タイムアウト、schema 違反のいずれも同じ扱い
        logger.warning("クエリ展開に失敗したため原文 1 件で検索する: %s: %s",
                       type(exc).__name__, exc)
        return [query]
    # dict.fromkeys で順序を保ったまま重複を落とす。切り詰めは重複を除いた**後**に行う
    # (先に 3 件へ切ると、同じクエリ 2 件で 1 枠を潰したまま 2 件しか残らない)
    unique = list(dict.fromkeys(raw))
    return unique[:3] or [query]
