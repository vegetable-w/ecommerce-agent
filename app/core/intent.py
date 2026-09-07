"""発話の intent 分類(single-label 7 class)。graph の classify_intent node が使う。

ここで決めた 1 語が app/graph/routing.py の routing 表を引き、そのまま 4 つの出口の
どれを通るかを決める。分類を外すと会話全体の経路が変わるので、出力は自由文ではなく
Literal に固定して、7 分類の外の文字列が State へ入らないようにする。
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import INTENT_CLASSIFY_PROMPT

logger = logging.getLogger(__name__)

INTENTS = ("配送", "注文", "商品相談", "返金返品", "アフターサービス", "苦情", "雑談")


class _Intent(BaseModel):
    intent: Literal["配送", "注文", "商品相談", "返金返品", "アフターサービス", "苦情", "雑談"] = Field(
        description="7 種類の intent のいずれか"
    )


async def classify(query: str) -> str:
    """7 分類のいずれか 1 語を返す。

    nested ではなく flat な 1 field にしているのは、上流が nested な output schema で
    502 を返すことがあるため。分類結果は 1 語なので、包む理由もない。

    temperature=0 を明示するのは、これが接客の返答ではなく**判定**だから。
    同じ発話は毎回同じ出口へ入ってほしく、実行ごとに routing が揺れるのは
    再現も説明もできない(04 章で judge を 0 にしたのと同じ理由。app/core/llm.py 参照)。

    **上流が落ちても例外を投げず「雑談」へ倒す。** 雑談は model を一切呼ばずに
    固定文を返す唯一の出口なので、上流が落ちている状況で「この先さらに model を呼ぶ経路」
    (business の Agent や knowledge の RAG)へ倒れずに済む。落ちている上流を叩き直して
    タイムアウトを重ねるより、その場で定型文を返して終える方が安全側。

    ただしこの縮退には副作用がある。**握り潰したまま黙っていると、上流障害が
    「雑談が増えた」という分類精度の劣化にしか見えなくなる。** 呼び出し側と運用が
    後から区別できるように、握るときは必ず warning を 1 行残す。
    """
    try:
        # model の組み立てごと try の内側に置く(app/core/selfcheck.py と同じ形)。
        # 実際の 5xx は ainvoke で出るが、設定不備や with_structured_output の
        # 非対応が構築時に飛ぶことがあり、そこだけ素通しになると縮退の意味がない
        model = get_chat_model(temperature=0).with_structured_output(_Intent)
        r: _Intent = await (INTENT_CLASSIFY_PROMPT | model).ainvoke({"query": query})
    except Exception as exc:  # 上流エラー、タイムアウト、schema 違反のいずれも同じ扱い
        logger.warning("intent 分類に失敗したため雑談へ倒す: %s: %s", type(exc).__name__, exc)
        return "雑談"
    # Literal で弾けているはずだが、structured output の実装は上流と版に依存するので念のため。
    # 7 分類の外の文字列は routing 表を素通りして business へ流れてしまう
    return r.intent if r.intent in INTENTS else "雑談"
