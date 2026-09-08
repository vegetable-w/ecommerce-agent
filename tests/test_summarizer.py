"""会話の非同期要約。上流も本物の DB も呼ばず、差し替えた repository とモデルで契約を確かめる。

要約が扱うのは**失われたら取り返せない**情報(注文番号、電話番号、未解決の要望)なので、
ここでの関心は「うまく要約できるか」ではなく「取りこぼさないか」に寄せてある:
二重に走らせないこと、失敗したときに境界を進めないこと、前回の要約を必ず繋ぐこと。
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda

from app.core import summarizer


def _msg(mid: int, role: str, content: str = "x"):
    return SimpleNamespace(id=mid, role=role, content=content)


def _dialog_rows(n_turns: int, start_id: int = 1) -> list:
    rows, i = [], start_id
    for t in range(n_turns):
        rows.append(_msg(i, "user", f"質問{t}"))
        rows.append(_msg(i + 1, "assistant", f"回答{t}"))
        i += 2
    return rows


class _FakeRepo:
    """summarizer が使う repository の入り口だけを持つ、メモリ上の代役。

    どのメソッドも 1 度は制御を手放す(await asyncio.sleep(0))。本物は DB へ行くので
    必ず中断点になり、その隙間で同じ会話の要約がもう 1 本立ち上がりうる。中断しない
    代役にすると、その並びを一切踏まずにテストが通ってしまう。
    """

    def __init__(self, conv=None, msgs=None):
        self.conv = conv
        self.msgs = msgs or []
        self.fragments = []          # 追記された断片
        self.updates = []            # (summary, upto_msg_id) の書き戻し

    async def get_conversation(self, conversation_id):
        await asyncio.sleep(0)
        return self.conv

    async def list_dialog_messages(self, conversation_id):
        await asyncio.sleep(0)
        return list(self.msgs)

    async def count_messages_after(self, conversation_id, after_id):
        await asyncio.sleep(0)
        return len([m for m in self.msgs if m.id > (after_id or 0)])

    async def append_summary_fragment(self, conversation_id, from_msg_id, upto_msg_id, content):
        await asyncio.sleep(0)
        self.fragments.append(
            SimpleNamespace(conversation_id=conversation_id, from_msg_id=from_msg_id,
                            upto_msg_id=upto_msg_id, content=content)
        )
        return len(self.fragments)

    async def update_conversation_summary(self, conversation_id, summary, upto_msg_id):
        await asyncio.sleep(0)
        self.updates.append((summary, upto_msg_id))
        self.conv.summary = summary
        self.conv.summary_upto_msg_id = upto_msg_id


def _conv(cid=1, summary=None, upto=None):
    return SimpleNamespace(id=cid, summary=summary, summary_upto_msg_id=upto)


@pytest.fixture(autouse=True)
def _clear_running():
    """走行中テーブルはモジュール変数なので、テストをまたいで残さない。"""
    summarizer._running.clear()
    yield
    summarizer._running.clear()


class _FakeModel:
    """with_structured_output が固定の結果を返す最小のモデル(tests/test_selfcheck.py と同じ形)。"""

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


# ---------------------------------------------------------------------------
# 起動の判定
# ---------------------------------------------------------------------------


async def test_below_threshold_does_not_summarize(monkeypatch):
    repo = _FakeRepo(conv=_conv(), msgs=_dialog_rows(1))
    monkeypatch.setattr(summarizer, "repository", repo)
    monkeypatch.setattr(summarizer.settings, "summary_trigger_messages", 30)
    called = []
    monkeypatch.setattr(summarizer, "summarize_conversation",
                        lambda cid: called.append(cid) or asyncio.sleep(0))

    await summarizer.maybe_schedule_summary(1)
    await asyncio.sleep(0)
    assert called == [] and summarizer._running == {}


async def test_threshold_reached_runs_in_background(monkeypatch):
    repo = _FakeRepo(conv=_conv(), msgs=_dialog_rows(2))
    monkeypatch.setattr(summarizer, "repository", repo)
    monkeypatch.setattr(summarizer.settings, "summary_trigger_messages", 4)
    ran = []

    async def fake_body(cid):
        ran.append(cid)

    monkeypatch.setattr(summarizer, "summarize_conversation", fake_body)

    await summarizer.maybe_schedule_summary(1)
    task = summarizer._running[1]
    assert ran == []                      # 本体はまだ走っていない = このターンを待たせていない
    await task
    assert ran == [1]
    await asyncio.sleep(0)
    assert 1 not in summarizer._running    # 終わったら走行中テーブルから外れる


async def test_second_call_while_running_does_not_start_a_second_task(monkeypatch):
    """走行中にもう 1 本立てない。2 本が別々の境界で書き戻すと、片方の区間が
    どこにも残らない。"""
    repo = _FakeRepo(conv=_conv(), msgs=_dialog_rows(2))
    monkeypatch.setattr(summarizer, "repository", repo)
    monkeypatch.setattr(summarizer.settings, "summary_trigger_messages", 4)
    gate = asyncio.Event()
    ran = []

    async def slow_body(cid):
        ran.append(cid)
        await gate.wait()

    monkeypatch.setattr(summarizer, "summarize_conversation", slow_body)

    await summarizer.maybe_schedule_summary(1)
    task = summarizer._running[1]
    await asyncio.sleep(0)                 # 1 本目を実際に走り出させる
    await summarizer.maybe_schedule_summary(1)
    assert ran == [1] and summarizer._running[1] is task
    gate.set()
    await task


async def test_two_simultaneous_calls_start_only_one_task(monkeypatch):
    """同じターンの 2 経路から同時に呼ばれる場合。判定の途中に DB 待ちがあるので、
    「入口で見て、起動する直前にもう一度見る」ようになっていないとすり抜ける。"""
    repo = _FakeRepo(conv=_conv(), msgs=_dialog_rows(2))
    monkeypatch.setattr(summarizer, "repository", repo)
    monkeypatch.setattr(summarizer.settings, "summary_trigger_messages", 4)
    ran = []

    async def fake_body(cid):
        ran.append(cid)

    monkeypatch.setattr(summarizer, "summarize_conversation", fake_body)

    await asyncio.gather(summarizer.maybe_schedule_summary(1),
                         summarizer.maybe_schedule_summary(1))
    tasks = [t for t in [summarizer._running.get(1)] if t is not None]
    await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    assert ran == [1]


async def test_missing_conversation_is_a_no_op(monkeypatch):
    repo = _FakeRepo(conv=None, msgs=_dialog_rows(4))
    monkeypatch.setattr(summarizer, "repository", repo)
    monkeypatch.setattr(summarizer.settings, "summary_trigger_messages", 1)
    ran = []
    monkeypatch.setattr(summarizer, "summarize_conversation",
                        lambda cid: ran.append(cid) or asyncio.sleep(0))

    await summarizer.maybe_schedule_summary(999)
    await asyncio.sleep(0)
    assert ran == [] and summarizer._running == {}


# ---------------------------------------------------------------------------
# 要約の本体
# ---------------------------------------------------------------------------


async def test_body_appends_a_fragment_and_writes_back_the_projection(monkeypatch):
    repo = _FakeRepo(conv=_conv(), msgs=_dialog_rows(3))    # id 1..6
    monkeypatch.setattr(summarizer, "repository", repo)
    # 直近 1 ターンだけ原文で残す。ここで明示しないと、既定の 8 ターンに全部
    # 吸われて「今回は要約しない」になり、書き戻しの機構を確かめられない
    monkeypatch.setattr(summarizer.settings, "context_window_turns", 1)
    seen = []

    async def fake_dialog(old_summary, dialog):
        seen.append((old_summary, dialog))
        return "ユーザーは注文1001の配送を問い合わせた"

    monkeypatch.setattr(summarizer, "summarize_dialog", fake_dialog)

    await summarizer.summarize_conversation(1)

    assert len(repo.fragments) == 1
    frag = repo.fragments[0]
    # 直近 1 ターン(id 5,6)は原文で残すので、要約が覆うのは id 1..4
    assert (frag.from_msg_id, frag.upto_msg_id) == (1, 4)
    assert frag.content == "ユーザーは注文1001の配送を問い合わせた"
    assert repo.updates == [("ユーザーは注文1001の配送を問い合わせた", 4)]
    assert "質問0" in seen[0][1] and "質問2" not in seen[0][1]


async def test_body_only_summarizes_messages_after_the_previous_boundary(monkeypatch):
    """同じ発話を何度も圧縮し直さない。圧縮のたびに事実が削れていく。"""
    repo = _FakeRepo(conv=_conv(summary="前回の要約", upto=4), msgs=_dialog_rows(4))
    monkeypatch.setattr(summarizer, "repository", repo)
    monkeypatch.setattr(summarizer.settings, "context_window_turns", 1)
    seen = []

    async def fake_dialog(old_summary, dialog):
        seen.append((old_summary, dialog))
        return "新しい要約"

    monkeypatch.setattr(summarizer, "summarize_dialog", fake_dialog)

    await summarizer.summarize_conversation(1)

    dialog = seen[0][1]
    assert "質問0" not in dialog and "質問1" not in dialog   # id 1..4 は前回の要約が覆っている
    assert "質問2" in dialog and "質問3" not in dialog   # 直近 1 ターンは原文で残す
    assert (repo.fragments[0].from_msg_id, repo.fragments[0].upto_msg_id) == (5, 6)


async def test_body_carries_the_previous_summary_into_the_model(monkeypatch):
    """前回の要約をモデルへ渡すこと。渡さないと、古い要約にしかない注文番号が
    今回の書き戻しで消える。"""
    repo = _FakeRepo(conv=_conv(summary="ユーザーは注文1001について問い合わせた", upto=4),
                     msgs=_dialog_rows(4))
    monkeypatch.setattr(summarizer, "repository", repo)
    monkeypatch.setattr(summarizer.settings, "context_window_turns", 1)
    seen = []

    async def fake_dialog(old_summary, dialog):
        seen.append(old_summary)
        return "注文1001と注文1002について問い合わせた"

    monkeypatch.setattr(summarizer, "summarize_dialog", fake_dialog)

    await summarizer.summarize_conversation(1)
    assert seen == ["ユーザーは注文1001について問い合わせた"]


async def test_body_does_not_advance_the_boundary_when_the_model_fails(monkeypatch):
    """上流が落ちても例外を外へ出さず、境界も進めない。進めてしまうと、その区間の
    発話は要約にも窓にも残らない。"""
    repo = _FakeRepo(conv=_conv(summary="前回の要約", upto=4), msgs=_dialog_rows(4))
    monkeypatch.setattr(summarizer, "repository", repo)

    async def boom(old_summary, dialog):
        raise RuntimeError("上流が 500 を返した")

    monkeypatch.setattr(summarizer, "summarize_dialog", boom)

    await summarizer.summarize_conversation(1)          # 例外が外へ出ないこと

    assert repo.updates == [] and repo.fragments == []
    assert repo.conv.summary == "前回の要約" and repo.conv.summary_upto_msg_id == 4


async def test_body_does_not_write_an_empty_summary(monkeypatch):
    """空の要約で書き戻すと、それまでの事実が丸ごと消えたうえに境界だけ進む。"""
    repo = _FakeRepo(conv=_conv(summary="前回の要約", upto=4), msgs=_dialog_rows(4))
    monkeypatch.setattr(summarizer, "repository", repo)

    async def blank(old_summary, dialog):
        return "   "

    monkeypatch.setattr(summarizer, "summarize_dialog", blank)

    await summarizer.summarize_conversation(1)
    assert repo.updates == [] and repo.fragments == []
    assert repo.conv.summary_upto_msg_id == 4


async def test_body_does_nothing_without_material(monkeypatch):
    repo = _FakeRepo(conv=_conv(), msgs=[])
    monkeypatch.setattr(summarizer, "repository", repo)
    called = []
    monkeypatch.setattr(summarizer, "summarize_dialog",
                        lambda old, dialog: called.append(1) or asyncio.sleep(0))

    await summarizer.summarize_conversation(1)
    assert called == [] and repo.updates == [] and repo.fragments == []


async def test_body_does_nothing_when_the_boundary_already_covers_everything(monkeypatch):
    repo = _FakeRepo(conv=_conv(summary="全部要約済み", upto=8), msgs=_dialog_rows(4))
    monkeypatch.setattr(summarizer, "repository", repo)
    called = []
    monkeypatch.setattr(summarizer, "summarize_dialog",
                        lambda old, dialog: called.append(1) or asyncio.sleep(0))

    await summarizer.summarize_conversation(1)
    assert called == [] and repo.updates == []


async def test_body_skips_a_missing_conversation(monkeypatch):
    repo = _FakeRepo(conv=None, msgs=_dialog_rows(4))
    monkeypatch.setattr(summarizer, "repository", repo)
    called = []
    monkeypatch.setattr(summarizer, "summarize_dialog",
                        lambda old, dialog: called.append(1) or asyncio.sleep(0))

    await summarizer.summarize_conversation(1)
    assert called == [] and repo.updates == []


# ---------------------------------------------------------------------------
# モデルへ渡すもの
# ---------------------------------------------------------------------------


async def test_summarize_dialog_renders_old_summary_and_dialog(monkeypatch):
    model = _FakeModel(summarizer._Summary(summary="注文1001の配送を問い合わせた"))
    out = await summarizer.summarize_dialog(
        "ユーザーは電話番号090-0000-0000を伝えた",
        "ユーザー:注文1001はまだ届きません\n担当:確認いたします",
        model=model,
    )
    assert out == "注文1001の配送を問い合わせた"
    human = model.seen.to_messages()[-1].content
    assert "ユーザーは電話番号090-0000-0000を伝えた" in human
    assert "注文1001はまだ届きません" in human


async def test_summarize_dialog_without_a_previous_summary(monkeypatch):
    model = _FakeModel(summarizer._Summary(summary=" 注文1001の配送 "))
    out = await summarizer.summarize_dialog("", "ユーザー:注文1001はまだ届きません", model=model)
    assert out == "注文1001の配送"          # 前後の空白は落とす


# ---------------------------------------------------------------------------
# 直近のターンは原文のまま残す
#
# **これが無いと 2 層構成が崩れる。** 境界を最新の発話まで進めてしまうと、要約が
# 走った直後のターンでは窓に原文がそのターン分しか残らない。トリガが 30 件なので
# 15 ターンごとに「直近は原文」という前提が壊れる。
# ---------------------------------------------------------------------------

def test_keep_boundary_leaves_the_recent_turns():
    from app.core import summarizer as sm

    class M:
        def __init__(self, i, role):
            self.id, self.role = i, role

    # 10 ターン分(user/assistant 交互)
    seg = [M(i, "user" if i % 2 else "assistant") for i in range(1, 21)]
    users = [i for i, m in enumerate(seg) if m.role == "user"]
    assert len(users) == 10

    cut = sm._keep_boundary(seg, 8)
    assert cut == users[-8]              # 後ろから 8 番目の user 発話の手前で切る
    kept = seg[cut:]
    assert sum(1 for m in kept if m.role == "user") == 8


def test_keep_boundary_summarizes_nothing_when_the_conversation_is_short():
    """残すべきターンしか無いなら、今回は要約しない。"""
    from app.core import summarizer as sm

    class M:
        def __init__(self, i, role):
            self.id, self.role = i, role

    seg = [M(i, "user" if i % 2 else "assistant") for i in range(1, 11)]   # 5 ターン
    assert sm._keep_boundary(seg, 8) == 0
