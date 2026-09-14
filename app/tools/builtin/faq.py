"""ナレッジベース検索ツール(ハイブリッド検索 + リランク)。"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.core import confidence, query_understanding, retrieval, selfcheck
from app.core.prompts import RAG_CITATION_NOTICE, RAG_INSUFFICIENT_NOTICE
from app.tools import registry


class FaqInput(BaseModel):
    keyword: str = Field(min_length=1, description="FAQ を検索するためのキーワード。例:『返品』『発送までの目安』")
    category: str | None = Field(
        default=None,
        description="任意。カテゴリで絞り込みたいときだけ指定する。例:『送料』『返品』『商品マニュアル』。"
                    "確信が無ければ指定しない(存在しないカテゴリを指定すると何も見つからなくなる)",
    )

    # min_lengthはOpenAPIスキーマのminLengthとして表出させるために残し、
    # 空白のみの値(min_lengthを通過してしまう)はこのvalidatorで拒否する
    @field_validator("keyword")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("空白のみの値は許可されない")
        return v


# search_knowledge へ「足切りなし」を伝える番兵。rerank スコアは 0〜1 なので
# 0.0 でも実質的には素通しだが、それは上流の値域に対する暗黙の仮定になる。
_UNGATED = float("-inf")


# 断ったときの検索の写しを載せる key。**モデルへは渡さない**(下の _strip_internal が
# 落とす)。09 章の低信頼プールが「ナレッジに無いのか、有るのに引けていないのか」を
# 後から見分けるための材料で、モデルに見せると、根拠不足だと伝えた直後に
# その根拠らしきものを読ませることになる(04 章の回答拒否の契約が濁る)。
SNAPSHOT_KEY = "_retrieved_snapshot"


def _strip_internal(result: dict) -> dict:
    """モデルへ渡す本文から内部用の key を落とす(engine の format_result)。

    落とした値は ToolRun.raw_result に残るので、node はそちらから読める。
    """
    return {k: v for k, v in result.items() if k != SNAPSHOT_KEY}


def _insufficient(source: str, reason: str, snapshot: list | None = None) -> dict:
    """根拠不足の戻り値。回答拒否の指示を「本文」として必ず載せる。

    02 章の実測: ToolMessage(status="error") の status は上流へ送る際に落ち、モデルには
    本文しか届かない。したがって sufficient=False という機械的なフラグだけでは指示にならず、
    モデルは残りの JSON から適当に回答を作ってしまう。RAG_INSUFFICIENT_NOTICE をここに
    載せるのは、モデルが実際に読む唯一の場所だからである(このキーを落とさないこと)。

    snapshot は低信頼プールへ持ち回る検索の写し(09 章)。**None と [] は別の意味**で、
    None は「検索を通っていない」、[] は「検索は通ったが 1 件も返らなかった」。
    レビュー画面はこの 2 つを別の文言で出し分ける(static/review.html の snapshotHtml)。
    """
    return {
        "sufficient": False,
        "source": source,
        "reason": reason,
        "citations": [],
        "notice": RAG_INSUFFICIENT_NOTICE,
        SNAPSHOT_KEY: snapshot,
    }


@tool(args_schema=FaqInput)
async def query_faq(keyword: str, category: str | None = None) -> dict:
    """ナレッジベースを検索する(ハイブリッド検索 + リランク)。ポリシー、ルール、操作方法などの
    一般的な質問に加えて、商品の仕様・型番・機能・マニュアルの内容についてもこのツールで調べる
    （例:「ロボット掃除機の吸引力」「EC-RV300 の稼働時間」「静かなキーボードはあるか」）。
    「〜は売っていますか」「〜は取り扱っていますか」のように取り扱いの有無を尋ねられた場合も、
    まずこのツールで調べる（当店の商品マニュアルはナレッジベースに入っている）。知らない商品名
    だからといって、ツールを呼ばずに自分の一般知識で答えてはならない。
    在庫数と価格そのものを知りたい場合だけ query_product を使う。

    回答で引用できる番号付きの evidence を返す。根拠が足りない場合は sufficient=false と
    notice を返すので、その指示に従って推測で回答を作らずに正直に伝えること。"""
    u = await query_understanding.understand(keyword)
    query = u["standard"]
    # 同義語は検索テキストにだけ足す。標準質問の意味は変えない(セルフチェックと
    # 生成には query の方を使う)
    search_query = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    # 足切りは search_knowledge に任せず、ここで掛ける。理由は 2 つ:
    # ① 拒否理由に載せる「本当の top スコア」は足切り前にしか存在しない。向こう側で
    #    切ってもらうと hits が空で届き、理由が必ず top=0.000 になって、低信頼プールを
    #    人が見たときに「惜しかったのか全く外れていたのか」を区別できなくなる。
    # ② 閾値は「ユーザーへ答えるか断るか」という方針であって検索の性質ではない。
    #    断る主体であるこちら側に置く方が筋が通る(search_knowledge は 4 戦略比較の
    #    共通入口なので、戻り値の形を変えて全呼び出し元へ波及させたくない事情もある)。
    hits = await retrieval.search_knowledge(
        search_query, strategy="hybrid_rerank", category=category, min_score=_UNGATED
    )

    if not hits:
        # 写しは [] を渡す。検索は通っているので、None(検索を通っていない)にすると
        # レビュー画面が「この質問は検索を通っていない」と嘘の説明を出す。
        return _insufficient("retrieval_low_conf", "検索で根拠が 1 件も得られなかった",
                             snapshot=[])

    # リランク上流が落ちた場合、search_knowledge は rerank_score なしのハイブリッド順で
    # 返す(app/core/retrieval.py)。掛ける数字が無いので機械ゲートは飛ばし、意味ゲートへ
    # 委ねる。ここで拒否に倒すと、上流の一時障害がそのまま回答拒否 + プール投入に化ける。
    if "rerank_score" in hits[0]:
        top = hits[0]["rerank_score"]
        kept = [h for h in hits if h["rerank_score"] >= settings.rerank_min_score]
        if not kept:
            # 写しは**足切り前**の hits から取る。足切り後は空なので、そちらから取ると
            # 「何を引いていたのか」が残らない(レビューで最も知りたいのがそこ)。
            return _insufficient(
                "retrieval_low_conf", f"リランクの最高スコアが閾値未満(top={top:.3f})",
                snapshot=confidence.snapshot_from_hits(hits),
            )
        hits = kept

    # 意味ゲート: この根拠だけで答えきれるかをモデル自身に判定させる
    chk = await selfcheck.check_sufficient(query, [f"{h['question']} {h['answer']}" for h in hits])
    if not chk["useful"]:
        return _insufficient("self_check", chk["reason"],
                             snapshot=confidence.snapshot_from_hits(hits))

    # head/tail 配置の「後」に番号を振る。evidence 本文の [n] と citations の n が
    # 同じ chunk を指すのは、この順番が唯一の正であるため
    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h.get("id"), "section_path": h.get("section_path"),
         "question": h.get("question"), "answer": h.get("answer"),
         "content_type": h.get("content_type")}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    # notice を足りている側にも載せる。収束生成が読むのは AGENT_SYSTEM(02 章、凍結)で、
    # そこに引用ルールは無い。tool の本文がモデルへ指示を届けられる唯一の場所。
    return {"sufficient": True, "notice": RAG_CITATION_NOTICE,
            "evidence": evidence, "citations": citations}


registry.register(registry.spec_from_langchain_tool(
    query_faq, source="builtin", timeout=30.0, format_result=_strip_internal))
# query_faq は書き換え → ハイブリッド検索 → リランク → セルフチェックと上流を何度も
# 往復するため、他のツールの既定(5 秒)では終わらない
