"""生成前の evidence 十分性セルフチェック。上流は呼ばず、差し替えたモデルで契約を確かめる。

プロンプトの品質そのものは tests/data/selfcheck_samples.jsonl の
ラベル付きサンプルで別途検証する(実際の上流を叩くのでここには置かない)。
"""

from langchain_core.runnables import RunnableLambda

from app.core import selfcheck
from app.core.prompts import SELF_CHECK_PROMPT


class _FakeModel:
    """with_structured_output が result を返す(または例外を送出する)最小のモデル。"""

    def __init__(self, result):
        self._result = result
        self.seen = None

    def with_structured_output(self, schema, **kw):
        async def _run(prompt_value):
            self.seen = prompt_value
            if isinstance(self._result, Exception):
                raise self._result
            return self._result

        return RunnableLambda(_run)


def test_prompt_renders_query_and_evidence():
    msgs = SELF_CHECK_PROMPT.format_messages(
        query="送料はいくらですか", evidence="[1] 3,000円以上のご注文は送料無料です"
    )
    assert msgs[0].type == "system"
    assert "送料はいくらですか" in msgs[-1].content
    assert "[1] 3,000円以上のご注文は送料無料です" in msgs[-1].content


async def test_check_sufficient_parses_flat_fields():
    model = _FakeModel(
        selfcheck._Check(useful=False, reason="根拠は送料の話だけで、型番の仕様に触れていない")
    )
    out = await selfcheck.check_sufficient(
        "EC-RV300 の稼働時間は", ["3,000円以上のご注文は送料無料です"], model=model
    )
    assert out == {"useful": False, "reason": "根拠は送料の話だけで、型番の仕様に触れていない"}


async def test_check_sufficient_numbers_evidence_in_prompt():
    """モデルへ渡す根拠は [1][2] の連番付きで、生成側の引用番号と一致すること。"""
    model = _FakeModel(selfcheck._Check(useful=True, reason="送料の条件が書かれている"))
    await selfcheck.check_sufficient(
        "送料はいくらですか",
        ["3,000円以上は送料無料です", "北海道・沖縄・離島は別途中継料がかかります"],
        model=model,
    )
    human = model.seen.to_messages()[-1].content
    assert "[1] 3,000円以上は送料無料です" in human
    assert "[2] 北海道・沖縄・離島は別途中継料がかかります" in human


async def test_check_sufficient_marks_empty_evidence_insufficient_without_calling_upstream():
    """根拠が 1 件も無いなら上流を呼ぶまでもなく不十分。無駄な課金と待ち時間を避ける。"""
    model = _FakeModel(RuntimeError("呼ばれてはいけない"))
    out = await selfcheck.check_sufficient("代金引換はできますか", [], model=model)
    assert out["useful"] is False
    assert out["reason"]
    assert model.seen is None


async def test_check_sufficient_degrades_to_useful_when_upstream_fails():
    """上流が落ちても例外を投げない。

    ここは検索後の品質ゲートであり、通過した根拠は既に機械ゲート
    (rerank_min_score)を越えている。チェックが落ちたことを理由に回答を
    拒否すると、上流の一時障害がそのまま「答えられません」に化ける。
    生成側の RAG_ANSWER_SYSTEM にも回答拒否ルールが残っているため、
    ここは useful=True で通し、理由に失敗した旨を残す。
    """
    model = _FakeModel(RuntimeError("upstream 502"))
    out = await selfcheck.check_sufficient("送料はいくらですか", ["3,000円以上は送料無料"], model=model)
    assert out["useful"] is True
    assert "セルフチェック" in out["reason"]


async def test_check_sufficient_normalizes_missing_reason():
    model = _FakeModel(selfcheck._Check(useful=True, reason=None))
    out = await selfcheck.check_sufficient("送料は", ["3,000円以上は送料無料"], model=model)
    assert out == {"useful": True, "reason": ""}
