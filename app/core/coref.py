"""指示対象の解決と質問の書き下し。graph の coref node が使う。

「それ返品できる?」のような発話は、そのままでは分類も検索もできない。ここで
直近の会話を使って「注文1001のスマート家電は返品できますか」まで書き下し、
以降の分類(app/core/intent.py)と検索は文脈なしで意味の通る 1 文だけを見る。

指示対象の解決と口語の正規化を 1 回の呼び出しにまとめているのは、分けると同じ発話に
2 回課金する上、正規化が先に走ると「それ」を補う手掛かりごと削られるため。

**書き下しは常に得ではない。** すでに完結している質問、特に規約全般の質問へ直前の
注文番号を足すと、その注文に限った検索に化けて一般的な規約が引けなくなる。
素通しは失敗ではなく正規の結果で、その判断はプロンプト(COREF_REWRITE_SYSTEM)側にある。
"""

import logging

from app.core.llm import get_chat_model
from app.core.prompts import COREF_REWRITE_PROMPT

logger = logging.getLogger(__name__)


async def resolve(query: str, history: str = "", model=None) -> str:
    """文脈なしでも意味が通る 1 文にする。完結していればそのまま返す。

    構造化出力を使わないのは、返すものが 1 文の自由文だから。field 1 つの JSON を
    要求しても、書き下しの質は変わらずに parse の失敗経路だけが増える。

    temperature=0 を明示するのは、これが接客の返答ではなく**前処理**だから。
    同じ発話が実行のたびに違う書き下しになると、その先の分類も検索も揺れる
    (app/core/intent.py の分類を 0 にしたのと同じ理由)。

    **上流が落ちても例外を投げず、原文をそのまま返す。** 書き換えに失敗しただけで
    問い合わせ全体を落とすのは割に合わない。原文で進めれば、指示語を含む発話の
    精度が落ちるだけで済む。ただし握ったまま黙っていると、上流障害が
    「書き下しが効かなくなった」という精度の劣化にしか見えないので warning を 1 行残す。

    model は単体テストからの注入口(understand / classify と同じ形)。
    """
    # 履歴が無ければ補える文脈も無い。呼んでも原文が返るだけで、課金だけが増える
    if not (history or "").strip():
        return query
    try:
        # model の組み立てごと try の内側に置く(app/core/intent.py と同じ形)。
        # 設定不備がここで飛ぶと、縮退を通らずに node ごと落ちる
        model = model or get_chat_model(temperature=0)
        chain = COREF_REWRITE_PROMPT | model
        resp = await chain.ainvoke({"query": query, "history": history})
        # content は str とは限らない(block の list)。読めない形は原文へ倒す
        text = (resp.content if isinstance(resp.content, str) else "").strip()
    except Exception as exc:  # 上流エラー、タイムアウトのいずれも同じ扱い
        logger.warning("指示対象の解決に失敗したため原文のまま進める: %s: %s",
                       type(exc).__name__, exc)
        return query
    # 空を返してくることがある。そのまま通すと分類器へ空文字が渡り、
    # 意味の取れない発話として「その他」へ倒れる
    return text or query
