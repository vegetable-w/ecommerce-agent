"""judge の回帰チェック(scripts/judge_check.py)を固定する。

このツールの主張は「judge の変更だけを測っている」の 1 点に尽きる。それが崩れる壊れ方は
どれも静かなので、テストで押さえるのはカバレッジではなく次の 3 つ:

  1. 再生した evidence が、そのとき judge へ渡したものと **byte 単位で同じ**であること
     (ずれると judge ではなく入力の違いを測ることになる)
  2. 対象が「人が印を付けたケース」に限られ、落とした行は理由ごとに数えられること
     (黙って母数が減ると「一致率 100%」が「1 件も測れなかった」の別名になる)
  3. 呼び出し失敗が一致率の母数に入らないこと
     (入ると judge を 1 文字も変えていない実行同士で数字が食い違う)

DB には触れない。台帳の読み出しは repository ごと差し替える(台帳の repository 自体は
tests/test_faith_cases_api.py が support_test で押さえている)。上流も必ずモックする。
"""

import pytest
from langchain_core.runnables import RunnableLambda

from app.db.models import FaithCase
from scripts import eval_04 as ev
from scripts import judge_check as jc


@pytest.fixture(autouse=True)
def _no_upstream(monkeypatch):
    """本物の judge を呼ばないことの保証(カバレッジではなく運用の安全網)。

    差し替え漏れがあればテストは緑のまま実 API を叩き続ける。judge_check が持つ名前
    (from ... import した側)にも爆弾を仕掛けて、漏れたその場で落ちるようにする。
    """
    def _boom(*a, **kw):
        raise AssertionError("本物の上流が呼ばれた")

    monkeypatch.setattr("app.core.llm.get_chat_model", _boom)
    monkeypatch.setattr(jc, "get_chat_model", _boom)


# 日本語・改行・全角記号を含む根拠。JSON の往復と再生で壊れるのはこの手の文字
_HITS = [
    {"id": 41, "section_path": "返品・返金ポリシー / 未開封の場合",
     "question": "未開封なら返品できますか？",
     "answer": "未開封であれば、到着後 7 日以内に限り承ります。\n返送料はお客様のご負担です。"},
    {"id": 42, "section_path": "返品・返金ポリシー / 開封済みの場合",
     "question": "開封してしまった商品は返品できますか？",
     "answer": "開封済みの商品は、初期不良の場合に限り交換で対応いたします。"},
    {"id": 7, "section_path": "配送 / お届け日数",
     "question": "何日で届きますか？",
     "answer": "本州は 2〜3 日、北海道・沖縄は 4〜5 日が目安です。"},
]


# 既定の根拠を使うことを表す番兵。None は「スナップショットが無い古い行」という
# 意味のある値なので、既定値の代わりには使えない
_DEFAULT = object()


def _row(eval_id: str, status: str, *, citations=_DEFAULT,
         bucket: str = "A_policy") -> FaithCase:
    """台帳の 1 行。DB へは入れない(このモジュールは DB に触らない)。"""
    return FaithCase(
        eval_id=eval_id, bucket=bucket, status=status,
        query=f"{eval_id} の質問: 開封した商品を返品したいのですが",
        answer=f"{eval_id} の回答です。到着後 7 日以内なら返品できます[1]。",
        reason="evidence に無い日数を答えている",
        citations=ev.build_citations(_HITS) if citations is _DEFAULT else citations,
        judge_model="judge-v1", strategy="hybrid_rerank", seen_count=1,
    )


# ---------------------------------------------------------------------------
# evidence の再生
# ---------------------------------------------------------------------------


def test_rebuilt_evidence_is_byte_identical_to_the_one_the_judge_saw():
    """台帳のスナップショットから、build_evidence が作るのと同じ文字列に戻ること。

    ここがずれると「judge の変更を測っている」つもりで「入力の違い」を測ることになり、
    しかもどちらの evidence もそれらしく見えるので気づけない。番号付けを judge_check 側へ
    書き写さず build_evidence へ委ねているのはこのため。

    変異で確認済み: rebuild_evidence を「自前で enumerate(citations, 0) して
    [n] を組み立てる」実装(番号が 1 つずれる)へ差し替え → このテストが fail。
    """
    citations = ev.build_citations(_HITS)
    assert jc.rebuild_evidence(citations) == ev.build_evidence(_HITS)
    # 期待する体裁そのものも固定しておく(build_evidence 側と一緒に壊れることを防ぐ)
    assert jc.rebuild_evidence(citations).splitlines()[0] == (
        "[1] 未開封なら返品できますか？: "
        "未開封であれば、到着後 7 日以内に限り承ります。")
    assert jc.rebuild_evidence(citations).startswith("[1] ")
    assert "[3] 何日で届きますか？: 本州は 2〜3 日、北海道・沖縄は 4〜5 日が目安です。" \
        in jc.rebuild_evidence(citations)


def test_an_old_snapshot_without_n_falls_back_to_list_order():
    """n を書き始める前に積まれた行も測れること。当時の並び順を番号とみなす。"""
    old = [{k: v for k, v in c.items() if k != "n"} for c in ev.build_citations(_HITS)]
    assert all("n" not in c for c in old)
    assert jc.rebuild_evidence(old) == ev.build_evidence(_HITS)


def test_the_snapshot_is_replayed_in_the_order_of_n_not_of_the_list():
    """並びが崩れて保存されていても、[n] が指す根拠は n のとおりであること。"""
    shuffled = list(reversed(ev.build_citations(_HITS)))
    assert [c["n"] for c in shuffled] == [3, 2, 1]
    assert jc.rebuild_evidence(shuffled) == ev.build_evidence(_HITS)


def test_restore_hits_maps_the_snapshot_back_to_the_shape_build_evidence_expects():
    hits = jc.restore_hits(ev.build_citations(_HITS))
    assert hits == [{"id": h["id"], "section_path": h["section_path"],
                     "question": h["question"], "answer": h["answer"]} for h in _HITS]


# ---------------------------------------------------------------------------
# 正解ラベルと対象の絞り込み
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status, expected", [
    # 人が「本当に幻覚だった」と認めた -> judge は忠実でないと言うべき
    ("resolved", False),
    # 人が「judge の行き過ぎ」と判断した -> judge は忠実だと言うべき
    ("no_action_needed", True),
    # まだ誰も見ていない。正解が無いので測れない
    ("unresolved", None),
    (None, None),
])
def test_the_human_mark_becomes_the_expected_label(status, expected):
    assert jc.expected_faithful(status) is expected


def test_only_the_cases_a_person_marked_are_measured():
    """unresolved と、根拠スナップショットが無い行は測れないので落とす。"""
    rows = [_row("A1", "resolved"), _row("A2", "no_action_needed"),
            _row("A3", "unresolved"), _row("A4", "resolved", citations=None),
            _row("A5", "no_action_needed", citations=[])]
    targets, skipped = jc.select_cases(rows)

    assert [t["eval_id"] for t in targets] == ["A1", "A2"]
    assert [t["expected"] for t in targets] == [False, True]
    assert skipped == {"unresolved": 1, "no_citations": 2}


def test_the_target_carries_the_replayed_evidence_and_the_stored_answer():
    """judge へ渡すのは台帳の回答と、復元した evidence。検索も生成もやり直さない。"""
    row = _row("A1", "resolved")
    target = jc.select_cases([row])[0][0]
    assert target["evidence"] == ev.build_evidence(_HITS)
    assert target["answer"] == row.answer
    assert target["status"] == "resolved"


async def test_the_ledger_is_read_page_by_page(monkeypatch):
    """1 ページに収まらない台帳でも全行を読むこと。0 件でも無限ループしないこと。"""
    pages = {1: [_row("A1", "resolved"), _row("A2", "unresolved")],
             2: [_row("A3", "no_action_needed")]}

    async def _list(status=None, page=1, size=20):
        return {"rows": pages.get(page, []), "total": 3, "page": page, "size": size,
                "pages": len(pages), "status": status, "counts": {}}

    monkeypatch.setattr("app.db.repository.list_faith_cases", _list)
    assert [r.eval_id for r in await jc.fetch_rows()] == ["A1", "A2", "A3"]

    async def _empty(status=None, page=1, size=20):
        return {"rows": [], "total": 0, "page": page, "size": size, "pages": 0,
                "status": status, "counts": {}}

    monkeypatch.setattr("app.db.repository.list_faith_cases", _empty)
    assert await jc.fetch_rows() == []


# ---------------------------------------------------------------------------
# judge の再実行(上流はすべてモック)
# ---------------------------------------------------------------------------


class _FakeChat(RunnableLambda):
    """本物の judge の代わり。`prompt | model.with_structured_output(...)` に応える。

    script は eval_id -> 応答の並び。1 回の呼び出しごとに先頭から 1 つ取り出す:
      True / False … その faithful で _Faithful を返す
      None          … 構造化出力の parse 失敗(上流がときどきこうなる)
      Exception     … 呼び出しそのものの失敗
    """

    def __init__(self, script: dict[str, list]):
        super().__init__(lambda value: value)
        self._script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []

    def with_structured_output(self, schema):
        assert schema is ev._Faithful, schema

        def _judge(value):
            text = value.to_string() if hasattr(value, "to_string") else str(value)
            eval_id = next(k for k in self._script if k in text)
            self.calls.append(eval_id)
            reply = self._script[eval_id].pop(0)
            if isinstance(reply, BaseException):
                raise reply
            if reply is None:
                return None
            return ev._Faithful(faithful=reply, reason="evidence に無い日数を答えている")

        return RunnableLambda(_judge)


def _install(monkeypatch, script: dict[str, list], rows: list) -> _FakeChat:
    """上流と台帳の読み出しを差し替える。戻り値の .calls で呼び出し回数を見る。"""
    fake = _FakeChat(script)
    seen: list[dict] = []

    def _factory(**kw):
        seen.append(kw)
        return fake

    async def _list(status=None, page=1, size=20):
        return {"rows": rows, "total": len(rows), "page": 1, "size": size, "pages": 1,
                "status": status, "counts": {}}

    monkeypatch.setattr(jc, "get_chat_model", _factory)
    monkeypatch.setattr("app.db.repository.list_faith_cases", _list)
    fake.factory_kwargs = seen
    return fake


async def test_the_judge_runs_at_temperature_zero(monkeypatch):
    """同じ回答に同じ判定を返してほしいので、揺れは害にしかならない(eval_04 と同じ理由)。"""
    fake = _install(monkeypatch, {"A1": [False]}, [_row("A1", "resolved")])
    assert await jc.main(["--no-probe"]) == 0
    assert fake.factory_kwargs == [{"temperature": 0}]


async def test_exit_code_is_zero_when_the_judge_agrees_with_every_human_mark(monkeypatch):
    """resolved は faithful=false、no_action_needed は faithful=true が一致。"""
    fake = _install(monkeypatch, {"A1": [False], "A2": [True]},
                    [_row("A1", "resolved"), _row("A2", "no_action_needed")])
    assert await jc.main(["--no-probe"]) == 0
    assert sorted(fake.calls) == ["A1", "A2"]


async def test_exit_code_is_one_when_the_judge_disagrees(monkeypatch, capsys):
    """1 件でも食い違えば失敗。judge の基準が動いたことに気づけないと意味が無い。"""
    _install(monkeypatch, {"A1": [True], "A2": [True]},
             [_row("A1", "resolved"), _row("A2", "no_action_needed")])
    assert await jc.main(["--no-probe"]) == 1
    out = capsys.readouterr().out
    assert "一致率 50.0% (1/2)" in out
    assert "不一致 1 件" in out


async def test_a_failed_call_is_kept_out_of_the_denominator(monkeypatch, capsys):
    """呼び出し失敗は不一致ではない。母数から外し、件数だけを別に出す。

    失敗を不一致に数えると、judge を 1 文字も変えていない実行同士で数字が食い違い、
    judge の変更を測るというこのツールの目的そのものが崩れる。

    変異で確認済み: summarize の judged を results 全件に変える(失敗を母数に含める)
      -> このテストが fail(一致率が 1/1 ではなく 1/2 になる)。
    """
    _install(monkeypatch, {"A1": [False], "A2": [None, None]},
             [_row("A1", "resolved"), _row("A2", "resolved")])
    # 失敗した 1 件は不一致に数えないので exit code は 0
    assert await jc.main(["--no-probe"]) == 0
    out = capsys.readouterr().out
    assert "一致率 100.0% (1/1)" in out
    assert "呼び出し失敗 1 件" in out


async def test_the_judge_is_retried_once_when_it_returns_nothing(monkeypatch):
    """構造化出力の None は 1 回だけやり直す。2 回目が返ればそのケースは測れる。"""
    fake = _install(monkeypatch, {"A1": [None, False]}, [_row("A1", "resolved")])
    assert await jc.main(["--no-probe"]) == 0
    assert fake.calls == ["A1", "A1"]


async def test_a_failing_call_is_retried_once_and_no_more(monkeypatch):
    """例外の場合も同じ。何度も叩き直して上流の課金と待ち時間を増やさない。"""
    fake = _install(monkeypatch, {"A1": [RuntimeError("upstream 502"), RuntimeError("502")]},
                    [_row("A1", "resolved")])
    assert await jc.main(["--no-probe"]) == 0        # 測れなかっただけで不一致は 0 件
    assert fake.calls == ["A1", "A1"]


def test_summarize_excludes_failures_from_the_rate():
    results = [
        {"actual": False, "agree": True}, {"actual": True, "agree": False},
        {"actual": None, "agree": None}, {"actual": None, "agree": None},
    ]
    assert jc.summarize(results) == {"total": 4, "judged": 2, "agreed": 1,
                                     "mismatched": 1, "failed": 2, "rate": 0.5}


def test_summarize_reports_no_rate_when_nothing_could_be_judged():
    """1 件も判定できなかった実行は 0% ではない(eval_04 の _mean と同じ扱い)。"""
    out = jc.summarize([{"actual": None, "agree": None}])
    assert out["rate"] is None and out["failed"] == 1


# ---------------------------------------------------------------------------
# CLI の振る舞い
# ---------------------------------------------------------------------------


async def test_skipped_cases_are_reported_with_their_reason(monkeypatch, capsys):
    """黙って母数を減らさない。何件を何の理由で測らなかったかを必ず出す。"""
    _install(monkeypatch, {"A1": [False]},
             [_row("A1", "resolved"), _row("A2", "unresolved"),
              _row("A3", "unresolved"), _row("A4", "resolved", citations=None)])
    assert await jc.main(["--no-probe"]) == 0
    out = capsys.readouterr().out
    assert "幻覚ケース台帳 4 件 / 測れるケース 1 件" in out
    assert "skip 2 件: 未対処" in out
    assert "skip 1 件: 根拠スナップショットが無い" in out


async def test_an_empty_ledger_says_so_instead_of_claiming_agreement(monkeypatch, capsys):
    """人が印を付けるまでは何も測れない。「全件一致」と紛らわしくしない。"""
    _install(monkeypatch, {}, [_row("A1", "unresolved")])
    assert await jc.main(["--no-probe"]) == 0
    out = capsys.readouterr().out
    assert "測れるケースが 1 件もありません" in out
    assert "一致率" not in out


async def test_dry_run_calls_no_upstream(monkeypatch, capsys):
    """--dry-run は対象の内訳だけ。judge を 1 回も呼ばないので課金されない。"""
    fake = _install(monkeypatch, {"A1": [False]}, [_row("A1", "resolved")])
    assert await jc.main(["--dry-run"]) == 0
    assert fake.calls == []
    assert "judge は呼びません" in capsys.readouterr().out


async def test_limit_takes_only_the_first_cases(monkeypatch):
    fake = _install(monkeypatch, {"A1": [False], "A2": [False]},
                    [_row("A1", "resolved"), _row("A2", "resolved")])
    assert await jc.main(["--limit", "1"]) == 0
    assert fake.calls == ["A1"]


async def test_the_tool_never_writes_to_the_ledger(monkeypatch):
    """台帳は読み取り専用。本番相当の support を指したまま実行される道具なので、
    書き込み経路に触れないことをここで固定する。"""
    def _boom(*a, **kw):
        raise AssertionError("台帳へ書き込もうとした")

    monkeypatch.setattr("app.db.repository.upsert_faith_case", _boom)
    monkeypatch.setattr("app.db.repository.set_faith_case_status", _boom)
    _install(monkeypatch, {"A1": [False]}, [_row("A1", "resolved")])
    assert await jc.main(["--no-probe"]) == 0


# ---------------------------------------------------------------------------
# 縮退の検出
#
# 台帳の一致率だけでは足りない理由: 台帳は 11 件中 10 件が no_action_needed
# (人が「幻覚ではない」と判断した)なので、何を見せても faithful=true と答える
# だけの judge も 10/11 を取ってしまう。一致率が高いことは「judge が働いている」
# ことの証拠にならない。
# ---------------------------------------------------------------------------

class _AlwaysFaithful(RunnableLambda):
    """何を見せても「忠実」としか答えない judge。判定を放棄した状態。"""

    def __init__(self):
        super().__init__(lambda value: value)

    def with_structured_output(self, schema):
        return RunnableLambda(
            lambda value: ev._Faithful(faithful=True, reason="根拠に基づいている"))


def test_probe_set_has_both_sides():
    """幻覚側と正しい側の両方がなければ、縮退も過検出も見分けられない。"""
    rows = jc.load_probe()
    traps = [r for r in rows if not r["expected"]]
    clean = [r for r in rows if r["expected"]]
    assert traps and clean, (len(traps), len(clean))
    assert len({r["eval_id"] for r in rows}) == len(rows)   # id の重複なし
    for r in rows:
        assert r["evidence"].strip() and r["answer"].strip()


async def test_degenerate_judge_is_reported_even_when_the_ledger_agrees(monkeypatch, capsys):
    """**このテストがこの機能の理由。**

    「何でも忠実」judge は、台帳が no_action_needed だらけなので一致率では満点を取る。
    それでも幻覚を 1 件も捕まえられていないのだから、失敗として扱わなければならない。
    """
    async def _list(status=None, page=1, size=20):
        rows = [_row("A2", "no_action_needed")]
        return {"rows": rows, "total": 1, "page": 1, "size": size, "pages": 1,
                "status": status, "counts": {}}

    monkeypatch.setattr(jc, "get_chat_model", lambda **kw: _AlwaysFaithful())
    monkeypatch.setattr("app.db.repository.list_faith_cases", _list)

    code = await jc.main([])
    out = capsys.readouterr().out
    assert "一致率 100.0% (1/1)" in out      # 台帳の側は満点
    assert "幻覚側の捕捉 0/" in out
    assert "縮退" in out
    assert code == 1                          # それでも失敗
