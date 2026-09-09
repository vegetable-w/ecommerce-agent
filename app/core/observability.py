"""09 Observability：Langfuse の取り付けと trace への tagging。すべて optional degradation。

3 つの環境変数 + graph の compile 時に callback を 1 回付けるだけ。node の業務コードには手を入れない。
このプロジェクトの config は pydantic-settings(.env)経由なので、os.environ に依存せず
明示的な引数で singleton を初期化する。

**Langfuse 未設定時、すべての公開関数は安全な no-op でなければならない。**
observability は付加価値であって依存ではない。だから

- 未設定なら langfuse を import もしない（未インストールの環境でも壊れない）
- 例外は握り潰す。trace が 1 本欠けることと、ユーザーの回答が返らないことは釣り合わない

trace の中身の対応：

    trace 根        = graph 1 回の実行（attach_observability が付けた CallbackHandler）
    session_id      = conversation_id（runtime が invoke config の metadata で渡す。
                      CallbackHandler が `langfuse_` 接頭辞の key を trace 根へ引き上げる）
    intent は trace 根の output（graph の最終 State）から取る。理由は下の長い
    コメントを読むこと。node の中から Langfuse へ書く道は塞がっている。
"""

import logging

from app.config import settings

logger = logging.getLogger(__name__)

_client = None

# tag の接頭辞。Langfuse 側で intent の tag だけを絞り込むための目印
_INTENT_TAG_PREFIX = "intent:"


def langfuse_enabled() -> bool:
    """3 つの config が揃っているか。1 つでも欠けたら observability は丸ごと off。

    部分設定を「とりあえず動かす」に倒さないのは、key だけあって base_url が空だと
    SDK の既定が cloud.langfuse.com になり、self-hosted のつもりで会話の中身を
    外へ送ってしまうため。揃っていなければ何もしない、が唯一安全な既定。
    """
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key
                and settings.langfuse_base_url)


def _init_client():
    """Langfuse client の singleton。import はここまで遅らせる。"""
    global _client
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )
    return _client


def _make_handler():
    """LangChain 用の callback。引数は取らない（4.15.1 では singleton の設定を見に行く）。"""
    from langfuse.langchain import CallbackHandler
    return CallbackHandler()


def get_langfuse():
    """enabled なら client singleton、disabled なら None。"""
    if not langfuse_enabled():
        return None
    return _init_client()


def attach_observability(graph):
    """compile 後に callback を 1 回付け、graph 全体を自動で trace する。

    未設定なら graph をそのまま返し、callback も付けず langfuse を import もしない。
    ここで付けておけば node 側は無改造のまま、1 リクエストの全 node / 全 model call が
    1 本の trace に入る。
    """
    if not langfuse_enabled():
        return graph
    _init_client()
    return graph.with_config({"callbacks": [_make_handler()]})


# intent を Langfuse 側へ書く関数はここに置かない。**実測の結果、置けないため。**
#
# 09 章の狙いは「intent 別に token cost を集計する」ことで、plan は
# `get_client().update_current_trace(tags=...)` を classify_intent の中から呼ぶ設計だった。
# 実装版で成り立たないことが 3 段階で分かった:
#
#   1. langfuse 4.15.1 に `update_current_trace` が無い。
#   2. 代替として示唆された ingestion の TraceCreate upsert は、v4 の server が
#      events_only モードのため 400 で弾く(実測: `Event type "trace-create" is not
#      accepted ... when LANGFUSE_MIGRATION_V4_WRITE_MODE is events_only`)。
#   3. v4 に残る `propagate_attributes` は「今の span」に載せるものだが、
#      **node の中に「今の span」が無い**。LangChain の CallbackHandler が作る
#      observation は OTEL の current span にならないため、node から
#      `get_current_trace_id()` を呼ぶと None が返り、属性は黙って捨てられる。
#      (実測: `No active span in current context` + `current_span=NonRecordingSpan`)
#
# したがって intent は **trace 根の output** から取る。root span("LangGraph")の output は
# graph の最終 State で、そこに `intent` が入っている。scripts/cost_by_intent.py が
# 同じ trace_id の GENERATION の usage と突き合わせて集計する。
#
# この方が質的にも正しい: 根の output の intent は `INTENTS` への丸めと上流障害時の
# フォールバックを通った後の値で、**routing が実際に使った intent と必ず一致する**。
