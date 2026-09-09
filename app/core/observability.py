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
    tags / metadata = intent と confidence（tag_intent。Task 3 が classify_intent から呼ぶ。
                      **載るのは呼んだ時点の span** で、trace 根ではない。tag_intent の
                      docstring に理由と Task 11 への影響を書いてある）
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


def tag_intent(intent: str, confidence: float) -> None:
    """intent を今の span の tag と metadata に書く（Cost Control の intent grouping の入口）。

    **plan の `get_client().update_current_trace(...)` は使えない。** 実装版の langfuse
    4.15.1 にそのメソッドは無く、代わりに示唆された ingestion の TraceCreate upsert も
    v4 の server に 400 で弾かれる（実測: `Event type "trace-create" is not accepted by
    /api/public/ingestion when LANGFUSE_MIGRATION_V4_WRITE_MODE is events_only`）。
    v4 で trace 属性を書く公式の口はこの propagate_attributes だけ。

    **どこに載るか（Task 11 への申し送り）**：v4 の trace 属性は span の属性で、
    trace 全体の tag は **app root span**（LangChain の CallbackHandler が開く "LangGraph"）
    のものが使われる。tag_intent はその子（classify_intent）の中から呼ばれるので、
    intent が載るのは**その 1 observation** であって trace 根ではない。
    root は callback が所有していて後から書き換える術が無いため、これが v4 で届く上限。
    intent 別に token cost を集計するときは、同じ trace_id の中で
    「intent tag を持つ observation」と「GENERATION の cost」を突き合わせること。

    例外は握って握り潰す。observability の失敗が業務の流れに影響してはいけない。
    """
    if not langfuse_enabled():
        return
    try:
        _init_client()
        from langfuse import propagate_attributes

        # context manager だが、属性は **enter した時点で現在の span に載る**。
        # tag_intent は値を 1 つ書くだけで context を持ち回らないので即座に閉じる
        # (開いたままにすると、以降の node がすべてこの context の中に入ってしまう)。
        with propagate_attributes(
            tags=[f"{_INTENT_TAG_PREFIX}{intent}"],
            metadata={"intent": intent, "intent_confidence": confidence},
        ):
            pass
    except Exception:  # noqa: BLE001 — observability の失敗で turn を落とさない
        logger.debug("Langfuse への intent tagging に失敗した", exc_info=True)
