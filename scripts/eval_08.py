"""08 章のラベル付きサンプルを実サービス上で通す。

MySQL / MCP Server / 上流モデルをすべて実際に使う。判定は log を目で読むのではなく
**State と interrupt の payload、そして DB の行を読んで機械的に**行う
(eval_06.py と同じ考え方)。

trace も interrupt の payload も HTTP の応答には載っていないので graph を直接呼ぶ。
上流も DB も本物なので、実サービスであることは変わらない。チケットの確認カードは
runtime.resume_turn で押す(POST /api/actions/resume が渡すのと同じ {"confirmed": bool})。

**先に `make mcp-up` で MCP Server 2 台を起動しておくこと。** サンプル 5 の
query_logistics は logistics Server(:8101)側の tool で、落ちているとツール一覧から
黙って抜け、モデルが選びようのないまま NG になる。

サンプル 1〜4 が測るのは「モデルが確認カードを出すか」という**モデルの振る舞い**で、
モデルは非決定的。揺れたら再実行して記録する。1 回の実行結果だけで合否を断定しない。

このスクリプトは tickets と tool_audit_logs に行を足す(それが測っているものそのもの)。
既存の行は読むだけで、更新も削除もしない。

使い方: PYTHONUTF8=1 uv run --env-file .env python scripts/eval_08.py
"""

import asyncio
import json

from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import select

import app.db.base as db
from app.core import labels
from app.db.models import Ticket, ToolAuditLog
from app.graph import runtime
from app.graph.nodes import resolve_answer
from app.tools.business import list_user_orders, order_snapshot

# チケット作成を頼む発話。1 で用件を伏せ、2 で補う(description を渡せるのは
# ユーザーだけ、という規律をそのまま台本にしている)。
_ASK_TICKET = "チケットを作ってください"
_DESCRIPTION = (
    "配送予定日を1週間過ぎても商品が届かず、問い合わせても回答がありません。"
    "この件で担当者に対応してほしいです"
)
# ticket_type は DB の ENUM と同じ英語の識別子。日本語は preview の
# ticket_type_label 側にしか出ない(app/core/labels.py が唯一の対応表)。
_TICKET_TYPES = set(labels.TICKET_TYPE) - {"refund"}   # refund は 06 章の返金フロー専用
# query_logistics の formatter が返す配送状況。日本語化が効いていることの判定に使う。
_LOGISTICS_STATUSES = {"集荷済み", "輸送中", "配達中", "配達完了"}


async def _ticket_nos(conversation_id: int) -> list[str]:
    """その会話に紐づくチケット番号。読み取りのみ。"""
    async with db.async_session() as s:
        rows = await s.execute(
            select(Ticket.ticket_no).where(Ticket.conversation_id == conversation_id))
        return list(rows.scalars())


async def _audit_statuses(conversation_id: int, tool_name: str) -> list[str]:
    """その会話でその tool を呼んだ監査ログの status(英語識別子)。読み取りのみ。"""
    async with db.async_session() as s:
        rows = await s.execute(
            select(ToolAuditLog.status)
            .where(ToolAuditLog.conversation_id == conversation_id,
                   ToolAuditLog.tool_name == tool_name)
            .order_by(ToolAuditLog.id))
        return list(rows.scalars())


def _tool_args(state, name: str) -> dict | None:
    """その turn でモデルが tool を呼んだときの引数。呼んでいなければ None。"""
    for m in state.get("messages", []):
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                if tc["name"] == name:
                    return dict(tc.get("args") or {})
    return None


def _tool_result(state, name: str):
    """engine が整形してモデルへ返した tool の結果。JSON なら dict に戻す。"""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, ToolMessage) and m.name == name:
            try:
                return json.loads(m.content)
            except (json.JSONDecodeError, TypeError, ValueError):
                return m.content
    return None


async def sample_1(ctx: dict) -> tuple[bool, object]:
    """用件を伏せてチケットだけ頼む。カードは出さず、聞き返して止まること。

    description を推測で埋めると、ユーザーが書いていない内容のチケットが
    オペレーターへ渡る。引数が欠けたまま呼んで engine の JSON Schema 検証に
    弾かれる経路を通ってもよいが、確認カードが出てはいけない。
    """
    out = await runtime.run_turn("eval-08-ticket", _ASK_TICKET, None)
    cid = ctx["cid"] = out["conversation_id"]
    kind = (out.get("interrupt") or {}).get("type")
    answer = resolve_answer(out["state"])
    tickets = await _ticket_nos(cid)
    ok = kind != "confirm_ticket" and not tickets and ("?" in answer or "？" in answer)
    return ok, {"中断": kind, "チケット": tickets, "回答": answer[:60]}


async def sample_2(ctx: dict) -> tuple[bool, object]:
    """用件を補うと確認カードが出て、preview に種別と内容がそろうこと。"""
    if "cid" not in ctx:
        return False, "サンプル 1 が会話を作れていない"
    out = await runtime.run_turn("eval-08-ticket", _DESCRIPTION, ctx["cid"])
    itr = out.get("interrupt") or {}
    preview = itr.get("preview") or {}
    ttype = preview.get("ticket_type")
    ok = (itr.get("type") == "confirm_ticket"
          and ttype in _TICKET_TYPES
          and preview.get("ticket_type_label") == labels.label(labels.TICKET_TYPE, ttype or "")
          and bool((preview.get("description") or "").strip()))
    return ok, {"中断": itr.get("type"), "preview": preview}


async def sample_3(ctx: dict) -> tuple[bool, object]:
    """確認カードで「送信」を押すと、チケットが 1 件だけ増えて番号が返ること。"""
    cid = ctx.get("cid")
    if cid is None:
        return False, "サンプル 1 が会話を作れていない"
    before = await _ticket_nos(cid)
    out = await runtime.resume_turn(cid, {"confirmed": True})
    added = [t for t in await _ticket_nos(cid) if t not in before]
    answer = resolve_answer(out["state"])
    ok = (len(added) == 1
          and added[0] in answer
          and "success" in await _audit_statuses(cid, "create_ticket"))
    return ok, {"増えたチケット": added, "回答": answer[:60]}


async def sample_4(_ctx: dict) -> tuple[bool, object]:
    """確認カードで「取り消し」を押すと、チケットは増えず監査に権限拒否が残ること。

    ここだけ別の会話を使う。サンプル 3 で既にチケットが 1 件できている会話で
    取り消すと、「増えていない」ことをチケットの件数で言えなくなる。
    """
    out = await runtime.run_turn("eval-08-cancel", f"{_ASK_TICKET}。{_DESCRIPTION}", None)
    cid = out["conversation_id"]
    kind = (out.get("interrupt") or {}).get("type")
    if kind != "confirm_ticket":
        return False, {"中断": kind, "備考": "確認カードが出ないので取り消しを試せない"}
    done = await runtime.resume_turn(cid, {"confirmed": False})
    tickets = await _ticket_nos(cid)
    statuses = await _audit_statuses(cid, "create_ticket")
    ok = not tickets and "permission_denied" in statuses and "success" not in statuses
    return ok, {"チケット": tickets,
                "監査": [labels.label(labels.TOOL_AUDIT_STATUS, s) for s in statuses],
                "回答": resolve_answer(done["state"])[:60]}


async def sample_5(_ctx: dict) -> tuple[bool, object]:
    """伝票番号を渡すと MCP の query_logistics が使われ、日本語化されて返ること。

    伝票番号は order_snapshot が実際に採番した本物を使う(JP + 12 桁)。整形は
    client 側の formatter の担当なので、内部 enum(status_code)と carrier_code が
    モデルへ渡っていないことまで見る。
    """
    order_id = list_user_orders("eval-08-logi")[0]["order_id"]
    tracking = order_snapshot(order_id)["tracking_no"]
    out = await runtime.run_turn(
        "eval-08-logi", f"配送伝票番号 {tracking} の荷物はいつ届きますか?", None)
    state = out["state"]
    args = _tool_args(state, "query_logistics") or {}
    result = _tool_result(state, "query_logistics")
    answer = resolve_answer(state)
    status = result.get("status") if isinstance(result, dict) else None
    ok = (args.get("tracking_no") == tracking
          and isinstance(result, dict)
          and status in _LOGISTICS_STATUSES
          and "status_code" not in result and "carrier_code" not in result
          and status in answer)
    return ok, {"伝票": tracking, "引数": args, "整形後": result, "回答": answer[:60]}


async def main() -> int:
    await runtime.init_graph()
    ctx: dict = {}
    results = []
    try:
        for name, fn in [
            ("1 用件が無ければ確認カードを出さず聞き返す", sample_1),
            ("2 用件を補うと確認カードが出る", sample_2),
            ("3 送信でチケットが 1 件増え番号を返す", sample_3),
            ("4 取り消しでチケットは増えず権限拒否が残る", sample_4),
            ("5 MCP の配送照会が日本語化されて回答に出る", sample_5),
        ]:
            ok, detail = await fn(ctx)
            results.append((name, ok, detail))
    finally:
        await runtime.close_graph()

    print()
    for name, ok, detail in results:
        print(f"{'OK' if ok else 'NG'}  {name}\n      {detail}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} 件が期待どおり（モデルは非決定的。揺れたら再実行して記録する）")
    print("\nファイルを 1 つ置くだけでツールが増えることは scripts/demo_08_promotions.py.txt を")
    print("app/tools/builtin/promotions.py へ cp して確かめる(受け入れ 1)。")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
