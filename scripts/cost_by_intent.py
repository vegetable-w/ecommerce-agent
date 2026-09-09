"""09 Cost Control：intent 別の token 集計。どの種類の質問に token を使っているかを見る。

データ源は Langfuse（self-hosted）。使い方は `make cost-report`（DAYS=N、既定 7）で、
Langfuse の起動（make langfuse-up）と 3 つの環境変数が要る。
成果物は data/09/reports/cost_by_intent.{txt,json}。txt は端末で読むための控えで、
json は /observability の画面が読む。**割合も平均 token もここで計算して両方へ書く。**
画面側が計算し直すと、端末の数字と画面の数字が食い違う道が開く。

custom の model 名は Langfuse に組み込みの価格が無いため、この script は token 数を
基準にする。金額で見たい場合は Langfuse の UI で model の価格を設定する。
この script の範囲外。

---------------------------------------------------------------------------
**どの API を使うか（実測した結果と、そう決めた理由）**

plan は「Metrics API の traces view を tags で group して totalTokens を合計する」
設計だったが、実装版の server（Langfuse 4.32.0、既定の events_only mode）では
**その口が開いていない**。同じ credential で実際に叩いた結果：

    GET /api/public/metrics          → 404 "not available ... in Langfuse v4 events_only mode"
    GET /api/public/traces           → 404（client.api.trace.list() も同じ）
    GET /api/public/v2/observations  → 200

trace 単位の一覧も集計 API も無く、**observation の一覧だけが返る**。そこで
「observation を窓で取り、trace_id ごとに client 側で結合する」形にした。
1 回の graph 実行 = 1 trace なので、trace がそのまま「リクエスト 1 件」になる。

  token : type=GENERATION の usage_details["total"] を trace ごとに合計する。
          total は input + output + cache 読み出しを含む（実測: input 145 /
          output 12 で total 1437。cache 読み出しが 1280 ある）。
  intent: 下の 2 段で解決する。

**intent の取り方を 2 段にしてある理由（Task 3 の申し送りとの差分）**

Task 2 の申し送りは「intent の tag は classify_intent の observation に載る」だった。
それは SDK の span を手で開いた smoke での実測で、**実サービスの graph では tag が
1 つも載らない**ことを本 Task の実測で確認した：

    実グラフの classify_intent の中で
      OTEL の current span = NonRecordingSpan（trace_id / span_id とも 0）
      langfuse.get_client().get_current_trace_id() = None
      → propagate_attributes は "No active span in current context" で黙って捨てる

LangChain の CallbackHandler が作る observation は「今の span」にならない
（async の callback は OTEL の context を持ち回らない）ため、node の中から
attribute を載せる先が無い。実グラフの trace を API で引くと全 observation が
tags=[] だった。

代わりに使うのが **trace 根（"LangGraph"）の output** で、これは graph の最終 State
そのものなので intent が必ず入っている。しかも classify の生の model 出力ではなく、
9 分類への丸めとフォールバック（app/core/intent.py）を通った後の値なので、
**routing が実際に使った intent と一致する**。

tag の道も残してあるのは、それが plan の設計であり、Task 3 側の tagging が届くように
直された時点で（field group を 1 つ足すだけの費用で）自動的にそちらが使われるため。
今はどの trace にも tag が無いので、実際に効くのは根の output の側。
"""

import argparse
import json
import pathlib
import sys
import unicodedata
from datetime import datetime, timedelta, timezone

from app.core.intent import INTENTS
from app.core.observability import get_langfuse

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / "data" / "09" / "reports"
_OUT = _OUT_DIR / "cost_by_intent.txt"
_OUT_JSON = _OUT_DIR / "cost_by_intent.json"

# tag_intent が付ける接頭辞（app/core/observability.py と同じ形）
_TAG_PREFIX = "intent:"
# 1 度に取る observation の数。API の上限は 1000
_PAGE = 1000
# 暴走止め。1 窓で 100 ページ（= 10 万 observation）を超えたら打ち切る
_MAX_PAGES = 100

_LINES: list[str] = []


def _log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(msg)


def _width(s: str) -> int:
    """端末上の見た目の幅。日本語の intent 名を縦に揃えるので東アジア幅で数える。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, n: int) -> str:
    return s + " " * max(0, n - _width(s))


def _rpad(s: str, n: int) -> str:
    """右寄せ。書式指定子の :>N は**文字数**で数えるので、日本語の見出しが
    その下の数字の列と 1 文字ぶんずつずれる。見出しだけこちらで揃える。
    """
    return " " * max(0, n - _width(s)) + s


def _fetch(client, **kwargs) -> list:
    """observation を cursor で全ページ取る。窓の指定は呼び出し側の kwargs。

    cursor が返り続けても data が空になったら止める（空ページで無限に回さない）。
    """
    rows, cursor = [], None
    for _ in range(_MAX_PAGES):
        resp = client.api.observations.get_many(limit=_PAGE, cursor=cursor, **kwargs)
        data = list(getattr(resp, "data", None) or [])
        rows.extend(data)
        cursor = getattr(getattr(resp, "meta", None), "cursor", None)
        if not cursor or not data:
            break
    return rows


def _intent_from_tags(obs) -> str:
    """observation の tag から intent を取る。plan の設計どおりの道（今は空振りする）。"""
    for tag in getattr(obs, "tags", None) or []:
        if isinstance(tag, str) and tag.startswith(_TAG_PREFIX):
            value = tag[len(_TAG_PREFIX):]
            if value in INTENTS:
                return value
    return ""


def _intent_from_state(output) -> str:
    """trace 根の output（graph の最終 State）から intent を取る。

    output は str でも dict でも返りうる（API の版に依存する）。読めない形は
    空文字にして、呼び出し側に「解決できなかった」と数えさせる。
    """
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (ValueError, TypeError):
            return ""
    if not isinstance(output, dict):
        return ""
    value = output.get("intent")
    return value if isinstance(value, str) and value in INTENTS else ""


def _tokens(obs) -> int:
    """observation 1 件の token 数。usage が無ければ 0。

    total を使うのは input + output だけだと cache 読み出し分が落ちるため。
    total を持たない版に備えて input + output へ落とす。
    """
    usage = getattr(obs, "usage_details", None)
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total")
    if total is None:
        total = (usage.get("input") or 0) + (usage.get("output") or 0)
    try:
        return int(total)
    except (TypeError, ValueError):
        return 0


def collect(client, frm: datetime, to: datetime) -> tuple[dict, int, int]:
    """(intent -> {"count", "tokens"}, 窓内の trace 数, intent 不明の trace 数)。"""
    # 1 パス目：窓内の全 observation。io は取らない（回答本文まで運ぶと重い）
    rows = _fetch(client, fields="core,basic,usage,trace_context",
                  from_start_time=frm, to_start_time=to)

    tokens: dict[str, int] = {}
    intents: dict[str, str] = {}
    for obs in rows:
        tid = getattr(obs, "trace_id", None)
        if not tid:
            continue
        tokens[tid] = tokens.get(tid, 0) + _tokens(obs)
        if not intents.get(tid):
            found = _intent_from_tags(obs)
            if found:
                intents[tid] = found

    # 2 パス目：tag で解決できなかった trace を根の State から埋める。根は 1 trace に
    # 1 件なので、io を取ってもページ数は trace 数で収まる
    if any(not intents.get(tid) for tid in tokens):
        for obs in _fetch(client, fields="core,io", is_root_observation=True,
                          from_start_time=frm, to_start_time=to):
            tid = getattr(obs, "trace_id", None)
            if tid in tokens and not intents.get(tid):
                found = _intent_from_state(getattr(obs, "output", None))
                if found:
                    intents[tid] = found

    table: dict[str, dict] = {}
    unknown = 0
    for tid in tokens:
        intent = intents.get(tid)
        if not intent:
            unknown += 1
            continue
        row = table.setdefault(intent, {"count": 0, "tokens": 0})
        row["count"] += 1
        row["tokens"] += tokens[tid]
    return table, len(tokens), unknown


def build_rows(table: dict) -> list[dict]:
    """token の降順で 1 行 1 intent。**平均も割合もここで確定させる。**

    端末の表と JSON の両方がこの list だけを見るので、同じ数を 2 か所で計算する余地が
    無くなる。share_label まで作るのは、0.6174 を渡された画面が自分で丸めると、
    丸め方の違いだけで端末の 62% と食い違いうるため。
    """
    total = sum(r["tokens"] for r in table.values())
    ordered = sorted(table.items(), key=lambda kv: kv[1]["tokens"], reverse=True)
    rows = []
    for name, r in ordered:
        share = r["tokens"] / total if total else 0.0
        rows.append({
            "intent": name,
            "count": r["count"],
            "tokens": r["tokens"],
            "avg_tokens": r["tokens"] // max(r["count"], 1),
            "share": share,
            "share_label": f"{share:.0%}",
        })
    return rows


def report(table: dict, traces: int, unknown: int, days: int) -> dict:
    """端末へ表を出し、**同じ値で組んだ**成果物を返す。"""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "days": days,
        "traces": traces,
        "resolved": traces - unknown,
        "unknown": unknown,
        "total_tokens": sum(r["tokens"] for r in table.values()),
        "rows": build_rows(table),
    }
    rows = payload["rows"]

    _log(f"=== intent 別の token 集計(直近 {days} 日 / 出典: Langfuse)===")
    _log(f"窓内の trace: {traces} 本   intent を解決できたもの: {traces - unknown} 本")
    if not rows:
        _log("")
        _log("この窓には intent の分かる trace がありません。"
             "先に会話をいくつか流してから実行してください。")
        return payload

    label_w = max(_width(row["intent"]) for row in rows) + 2

    _log("")
    _log(f"{_pad('intent', label_w)}{_rpad('リクエスト', 10)} {_rpad('合計token', 13)} "
         f"{_rpad('平均token', 13)} {_rpad('割合', 8)}")
    for i, row in enumerate(rows):
        mark = "  ← 最もコストが高い" if i == 0 else ""
        _log(f"{_pad(row['intent'], label_w)}{row['count']:>10d} {row['tokens']:>13,d} "
             f"{row['avg_tokens']:>13,d} {row['share_label']:>8s}{mark}")
    _log(f"{_pad('合計', label_w)}{sum(r['count'] for r in rows):>10d} "
         f"{payload['total_tokens']:>13,d}")

    if unknown:
        _log("")
        _log(f"※ intent を解決できなかった trace が {unknown} 本あります"
             "(graph 以外の経路で作られた trace はここに入ります)。")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="intent 別の token 集計(Langfuse)")
    ap.add_argument("--days", type=int, default=7, help="集計する窓の日数(既定 7)")
    args = ap.parse_args()
    if args.days < 1:
        print("--days は 1 以上を指定してください", file=sys.stderr)
        return 2

    client = get_langfuse()
    if client is None:
        # **空の表を出さない。** 0 行の表は「コストがかからなかった」と読めてしまう
        print("Langfuse が設定されていません(LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / "
              "LANGFUSE_BASE_URL の 3 つが要ります)。"
              "集計できるデータが無いため、レポートは出しません。", file=sys.stderr)
        return 1

    to = datetime.now(timezone.utc)
    frm = to - timedelta(days=args.days)
    try:
        table, traces, unknown = collect(client, frm, to)
    except Exception as exc:  # noqa: BLE001 — 落ちた事実を、空の表の代わりに出す
        print(f"Langfuse へ問い合わせできませんでした: {type(exc).__name__}: {exc}\n"
              "make langfuse-up で起動しているか確認してください。", file=sys.stderr)
        return 1

    payload = report(table, traces, unknown, args.days)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _log(f"\nレポート: {_OUT.relative_to(_ROOT)} / {_OUT_JSON.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
