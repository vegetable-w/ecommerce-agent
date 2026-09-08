"""決定的な出口 node(script / complaint / fallback)。

この 3 つは graph の「必ず終わる経路」で、model を呼ばずに固定文を返すことそのものが
存在理由になっている。雑談のたびに上流を叩くなら固定文にする意味が無いので、
上流を取りに行った時点で落ちるように差し替えて確かめる。

雑談 / その他の出口は 06 で script_reply へ置き換わった(intent で文面を出し分ける)。
文面の出し分けそのものは tests/test_graph_06_nodes.py が見ており、ここでは
「model を呼ばずに固定文を返す出口である」ことだけを確かめる。

低信頼プールへの投入は monkeypatch で差し替える(support は本番相当の DB であり、
テストから書き込まない)。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.core import intent, llm, query_understanding, selfcheck
from app.graph import nodes


@pytest.fixture(autouse=True)
def _forbid_upstream(monkeypatch):
    """チャットモデルを取りに行った時点で落とす。

    get_chat_model は各モジュールが from import で束縛済みなので、llm 側だけ差し替えても
    素通りしてしまう。束縛先を 1 つずつ潰す。
    """
    def _boom(*args, **kwargs):
        raise AssertionError("決定的な出口 node が上流のモデルを呼んだ")

    for mod in (llm, intent, selfcheck, query_understanding, nodes):
        monkeypatch.setattr(mod, "get_chat_model", _boom)


def _state(text="こんにちは", **extra):
    return {"messages": [HumanMessage(text)], **extra}


# --- 雑談 / その他 --------------------------------------------------------------


async def test_script_reply_returns_fixed_text_without_actions():
    from app.core.prompts import SCRIPT_REPLY_CHITCHAT

    out = await nodes.script_reply(_state(intent="雑談"))
    assert out["answer"] == SCRIPT_REPLY_CHITCHAT
    assert out["trace"]["route"] == "fallback_script"
    # 選択肢を出す出口ではない。雑談に有人対応の導線を付けない
    assert "suggested_actions" not in out


async def test_script_reply_names_this_store():
    """名乗る店名は既存プロンプトと同じ「STORE」であること。"""
    from app.core.prompts import SCRIPT_REPLY_CHITCHAT

    assert "STORE" in SCRIPT_REPLY_CHITCHAT


# --- 苦情 ---------------------------------------------------------------------


async def test_complaint_reply_offers_two_actions_in_order():
    out = await nodes.complaint_reply(_state("届いた商品が壊れていた", conversation_id=7))
    assert out["answer"] == nodes.COMPLAINT_REPLY
    assert out["trace"]["route"] == "complaint"
    assert [a["type"] for a in out["suggested_actions"]] == ["transfer_human", "create_ticket"]


async def test_complaint_draft_uses_the_english_ticket_type_identifier():
    """ticket_type は DB の ENUM と同じ英語の識別子。画面に出す日本語を入れない。"""
    out = await nodes.complaint_reply(_state("届いた商品が壊れていた"))
    draft = next(a for a in out["suggested_actions"] if a["type"] == "create_ticket")["draft"]
    assert draft["ticket_type"] == "complaint"
    assert draft["description"] == "届いた商品が壊れていた"


async def test_complaint_draft_is_accepted_by_create_ticket():
    """draft はそのまま create_ticket へ渡る前提なので、tool の受け口と突き合わせる。"""
    from app.tools.builtin.tickets import create_ticket

    schema = create_ticket.args_schema.model_json_schema()
    allowed = schema["properties"]["ticket_type"]["enum"]
    out = await nodes.complaint_reply(_state("苦情を言いたい"))
    draft = next(a for a in out["suggested_actions"] if a["type"] == "create_ticket")["draft"]
    assert draft["ticket_type"] in allowed


async def test_complaint_reply_does_not_create_the_ticket(monkeypatch):
    """作成するのは選択肢だけ。この node は backend を実行しない。"""
    async def _boom(*args, **kwargs):
        raise AssertionError("complaint_reply がチケットを作成した")

    monkeypatch.setattr(nodes.repository, "create_ticket", _boom)
    await nodes.complaint_reply(_state("苦情を言いたい"))


# --- fallback -----------------------------------------------------------------


def _capture(monkeypatch):
    calls = {}

    async def fake_insert(conversation_id, raw_question, source, reason):
        calls.update(conversation_id=conversation_id, raw=raw_question,
                     source=source, reason=reason)
        return 1

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    return calls


async def test_fallback_reply_records_low_confidence(monkeypatch):
    calls = _capture(monkeypatch)
    out = await nodes.fallback_reply(
        {"messages": [HumanMessage("アカウントを削除するには?")], "conversation_id": 3,
         "trace": {"evidence_top": 0.123}}
    )
    assert out["answer"] == nodes.FALLBACK_REPLY
    assert out["trace"]["route"] == "fallback"
    assert calls["source"] == "retrieval_low_conf"
    assert calls["conversation_id"] == 3
    assert calls["raw"] == "アカウントを削除するには?"
    # 理由には実際の top スコアを載せる。プールを人が見るとき、惜しかったのか
    # 全く外れていたのかを区別できる唯一の数字になる
    assert "0.123" in calls["reason"]


async def test_fallback_reply_without_trace_does_not_crash(monkeypatch):
    """trace が無い経路(gate を通らず倒れてきた場合)でも整形で落ちないこと。"""
    calls = _capture(monkeypatch)
    out = await nodes.fallback_reply({"messages": [HumanMessage("質問")]})
    assert out["answer"] == nodes.FALLBACK_REPLY
    assert calls["conversation_id"] is None
    assert "0.000" in calls["reason"]


# --- _user_text ---------------------------------------------------------------


async def test_user_text_returns_last_human_message():
    state = {"messages": [HumanMessage("1 通目"), AIMessage("応答"), HumanMessage("2 通目")]}
    assert nodes._user_text(state) == "2 通目"


async def test_user_text_is_empty_without_human_message():
    """HumanMessage が無くても IndexError にしない。落ちると graph 全体が止まる。"""
    assert nodes._user_text({"messages": [AIMessage("応答")]}) == ""
    assert nodes._user_text({}) == ""
