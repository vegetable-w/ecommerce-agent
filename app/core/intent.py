"""発話の intent 分類(single-label 8 class + confidence)。graph の classify_intent node が使う。

ここで決めた 1 語が app/graph/routing.py の routing 表を引き、そのまま 5 つの出口の
どれを通るかを決める。分類を外すと会話全体の経路が変わるので、出力は自由文ではなく
Literal に固定して、8 分類の外の文字列が State へ入らないようにする。

05 章からの変更は 2 点。
 - 「その他」を足して 8 分類にし、**迷ったときの退避先をここに一本化した**。
 - confidence を返すようにした。閾値との比較(小さい model へ落として再分類する等)は
   09 章の話で、本章では State と trace に値を残すところまで。数値が残っていれば、
   どの発話で分類が揺れているかを後から実測で見られる。
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import INTENT_CLASSIFY_PROMPT

logger = logging.getLogger(__name__)

INTENTS = ("配送", "注文", "商品相談", "返金返品", "アフターサービス", "苦情", "雑談", "その他")

# 分類できなかったときの退避先。fallback_script(聞き直す出口)へ繋がる。
# 05 章では「雑談」へ倒していたが、雑談は挨拶に固定文を返す出口であって、
# 「分からなかった」の受け皿ではない。両者を同じ語で表すと、上流障害と
# 本物の挨拶がログ上で区別できなくなる。
FALLBACK_INTENT = "その他"


class _Intent(BaseModel):
    intent: Literal["配送", "注文", "商品相談", "返金返品", "アフターサービス", "苦情", "雑談", "その他"] = Field(
        description="8 種類の intent のいずれか"
    )
    # 値域は Field の制約にせず、受け取ってから丸める。ge/le を付けると上流が 1.5 を
    # 返した瞬間に ValidationError になり、**分類自体は当たっていたのに**「その他」へ
    # 倒れてしまう。確信度は分類の付随情報であって、分類を捨てる理由にはしない。
    confidence: float = Field(description="その分類でよいと言い切れる度合い(0〜1)")


def _clamp(value) -> float:
    """confidence を 0〜1 に収める。読めない値は 0.0。

    上流は 1.5 や -0.2 を返すことがある。そのまま State へ入れると、閾値との比較
    (09 章)も trace を読む人の目視も壊れる。値域の保証はこちら側で持つ。
    """
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


async def classify(query: str, history: str = "", model=None) -> dict:
    """{"intent": str, "confidence": float} を返す。intent は 8 分類のいずれか 1 語。

    nested ではなく flat な 2 field にしているのは、上流が nested な output schema で
    502 を返すことがあるため。

    temperature=0 を明示するのは、これが接客の返答ではなく**判定**だから。
    同じ発話は毎回同じ出口へ入ってほしく、実行ごとに routing が揺れるのは
    再現も説明もできない(04 章で judge を 0 にしたのと同じ理由。app/core/llm.py 参照)。

    **上流が落ちても例外を投げず「その他」へ倒す。** その他は fallback_script、つまり
    model を呼ばずに聞き直す出口なので、上流が落ちている状況で「この先さらに model を
    呼ぶ経路」(business の Agent や knowledge の RAG)へ倒れずに済む。落ちている上流を
    叩き直してタイムアウトを重ねるより、その場で聞き直して終える方が安全側。

    ただしこの縮退には副作用がある。**握り潰したまま黙っていると、上流障害が
    「その他が増えた」という分類精度の劣化にしか見えなくなる。** 呼び出し側と運用が
    後から区別できるように、握るときは必ず warning を 1 行残す。

    model は単体テストからの注入口(understand / check_sufficient と同じ形)。
    差し替えられないと、分類の契約を確かめるだけで実際の上流を叩くことになる。
    """
    try:
        # model の組み立てごと try の内側に置く(app/core/selfcheck.py と同じ形)。
        # 実際の 5xx は ainvoke で出るが、設定不備や with_structured_output の
        # 非対応が構築時に飛ぶことがあり、そこだけ素通しになると縮退の意味がない
        model = model or get_chat_model(temperature=0, model=settings.intent_model or None)
        chain = INTENT_CLASSIFY_PROMPT | model.with_structured_output(_Intent)
        r: _Intent = await chain.ainvoke({"query": query, "history": history})
    except Exception as exc:  # 上流エラー、タイムアウト、schema 違反のいずれも同じ扱い
        logger.warning("intent 分類に失敗したため%sへ倒す: %s: %s",
                       FALLBACK_INTENT, type(exc).__name__, exc)
        return {"intent": FALLBACK_INTENT, "confidence": 0.0}
    # Literal で弾けているはずだが、structured output の実装は上流と版に依存するので念のため。
    # 8 分類の外の文字列は routing 表を素通りして business へ流れてしまう
    intent = r.intent if r.intent in INTENTS else FALLBACK_INTENT
    return {"intent": intent, "confidence": _clamp(r.confidence)}
