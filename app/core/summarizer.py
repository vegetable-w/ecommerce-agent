"""会話の非同期要約。ターンの終わりに件数を見て、必要なら background で圧縮する。

**このターンの応答は待たせない。** 要約は「次のターン以降の文脈を軽くする」ための
仕事であって、いま答えている返答に必要なものではない。await して待つと、要約が
走るターンだけ体感で数秒遅くなる。

失敗しても会話は壊さない。上流が落ちたら warning を 1 行だけ残して境界は進めず、
次の機会に同じ区間からやり直す。ここで境界だけ進めてしまうと、その区間の発話は
要約にも窓にも載らず、**どこにも残らない**。
"""

import asyncio
import logging
import time

from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import SUMMARY_PROMPT
from app.db import repository

logger = logging.getLogger(__name__)

# 走行中の会話 -> その task。二重起動を弾くための表であると同時に、task への
# 強参照でもある。asyncio の event loop は task を弱参照でしか持たないので、
# ここで持たずに投げっぱなしにすると、実行中の task が GC に回収されて
# 途中で消えることがある(CPython の asyncio.create_task の既知の落とし穴)。
_running: dict[int, asyncio.Task] = {}


class _Summary(BaseModel):
    summary: str = Field(description="会話の要約。事実と要望だけを含める")


def _render_dialog(messages) -> str:
    """要約へ渡す会話本文。role は日本語の話者名に直す。

    role の生の値(user / assistant)のまま渡すと、要約が「assistant は〜」のように
    英語の役割名を含んだ日本語で返ってくることがある。要約はそのまま次のターンの
    prompt に載るので、話者の呼び方は接客の語彙に揃えておく。
    """
    return "\n".join(
        f"{'ユーザー' if m.role == 'user' else '担当'}:{m.content}"
        for m in messages if m.content
    )


async def summarize_dialog(old_summary: str, dialog: str, model=None) -> str:
    """前回までの要約と今回の区間を 1 つの要約へ繋ぐ。空白だけなら空文字を返す。

    model は単体テストからの注入口(app/core/selfcheck.py と同じ形)。

    temperature=0 を明示するのは、これが接客の返答ではなく**抽出**だから。
    同じ会話からは毎回同じ事実が出てほしく、実行ごとに注文番号が載ったり
    落ちたりするのは説明も再現もできない(04 章の judge、06 章の intent と同じ理由)。
    """
    model = model or get_chat_model(temperature=0)
    chain = SUMMARY_PROMPT | model.with_structured_output(_Summary)
    r: _Summary = await chain.ainvoke({
        "old_summary": old_summary or "(なし)",
        "dialog": dialog,
        "max_chars": settings.summary_max_chars,
    })
    return (r.summary or "").strip()


async def summarize_conversation(conversation_id: int) -> None:
    """要約の本体。素材を読み、要約し、断片を追記して投影を書き戻す。

    素材は **前回の境界より後ろだけ**。すでに要約した発話をもう一度圧縮すると、
    圧縮のたびに事実が削れていく。前回の要約は素材ではなく「落としてはいけない事実」
    としてモデルへ渡す。

    書き込みは断片 -> 投影の順。逆にすると、投影だけ進んで断片が残らなかったときに
    「どの区間の圧縮でその事実が消えたか」を追う手掛かりが無くなる。断片が残って
    投影が失敗した場合は境界が進まないので、次の機会に同じ区間をやり直すだけで済む。
    """
    t0 = time.monotonic()
    try:
        conv = await repository.get_conversation(conversation_id)
        if conv is None:
            return
        msgs = await repository.list_dialog_messages(conversation_id)
        old_upto = conv.summary_upto_msg_id or 0
        seg = [m for m in msgs if m.id > old_upto and m.content]
        if not seg:
            # 素材が無い(会話が空、または前回の要約がすでに全部を覆っている)。
            # 何も書かずに戻る。空の要約で上書きすると事実が丸ごと消える
            return

        # ここでも strip する。空白だけの要約を書き戻すと、それまでに積んだ事実が
        # 丸ごと消えたうえに境界だけが進む(最も取り返しがつかない壊れ方)
        summary = (await summarize_dialog(conv.summary or "", _render_dialog(seg)) or "").strip()
        if not summary:
            raise ValueError("要約が空だったため書き戻しを中止した")

        await repository.append_summary_fragment(
            conversation_id, seg[0].id, seg[-1].id, summary
        )
        await repository.update_conversation_summary(conversation_id, summary, seg[-1].id)
        logger.info("要約を更新 conv=%s 区間=%s..%s 件数=%s 長さ=%s %.0fms",
                    conversation_id, seg[0].id, seg[-1].id, len(seg), len(summary),
                    (time.monotonic() - t0) * 1000)
    except Exception as exc:
        # 例外を外へ出さない。ここは background の task であり、投げたところで
        # 受け取る人がいない。境界を進めていないので、次のターンで同じ区間から
        # やり直せる。握るときは必ず 1 行残す(黙って握ると、要約が止まったまま
        # 履歴だけが伸びていることに誰も気づけない)
        logger.warning("会話 %s の要約に失敗したため次の機会へ回す: %s: %s",
                       conversation_id, type(exc).__name__, exc)


def _forget(conversation_id: int):
    """task の後始末。走行中の表から外し、握り潰されなかった例外だけ記録する。"""

    def _done(task: asyncio.Task) -> None:
        _running.pop(conversation_id, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error("会話 %s の要約 task が異常終了した", conversation_id,
                         exc_info=task.exception())

    return _done


async def maybe_schedule_summary(conversation_id: int) -> None:
    """要約が必要なら background で走らせる。**このターンの応答を待たせない。**

    起動の判定は「前回の境界以降に増えた件数」だけを見る。件数が足りなければ
    ここで終わり、モデルも呼ばない。
    """
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        return
    n = await repository.count_messages_after(conversation_id, conv.summary_upto_msg_id)
    if n < settings.summary_trigger_messages:
        return

    # 走行中なら立てない。同じ会話の要約が 2 本走ると、それぞれが別の境界で
    # 書き戻し、**片方の区間がどこにも残らない**。
    #
    # この検査は create_task の直前、つまり **await を 1 つも挟まない位置** に置く。
    # 入口(関数の先頭)だけで見ると、上の 2 つの DB 待ちの間に同じ会話の呼び出しが
    # 割り込んだとき、両方が検査を通り抜けて 2 本立ってしまう。1 ターンの中で
    # 複数の経路からここへ来る作りなので、その並びは実際に起きる
    if conversation_id in _running:
        return
    logger.info("会話 %s の要約を起動する(前回以降 %s 件)", conversation_id, n)
    task = asyncio.create_task(summarize_conversation(conversation_id))
    _running[conversation_id] = task
    task.add_done_callback(_forget(conversation_id))
