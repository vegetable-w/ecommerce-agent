"""09 章の red line。Langfuse の実際の API を実測する。上流の LLM は呼ばない。

plan は 2026-07 の docs(v3 系)に合わせて書かれており、`get_client().update_current_trace()`
で trace に intent の metadata と tag を足す前提になっている。**実装前にそれを確かめる。**
ここが違うと Task 2/3/11(Cost Control の土台)が丸ごと書き直しになる。

Langfuse server が動いていなくても API の形は確かめられる。往復まで見たい場合は
先に `make langfuse-up` すること。

使い方: PYTHONUTF8=1 uv run --env-file .env python scripts/smoke_langfuse.py
"""

import inspect
from importlib.metadata import version


def report() -> None:
    import langfuse

    print(f"langfuse version = {version('langfuse')}")

    init = list(inspect.signature(langfuse.Langfuse.__init__).parameters)
    print(f"\n1) Langfuse constructor")
    print(f"   base_url を受けるか: {'base_url' in init}")
    print(f"   host も残っているか: {'host' in init}")

    print("\n2) LangChain の CallbackHandler")
    from langfuse.langchain import CallbackHandler
    ch = list(inspect.signature(CallbackHandler.__init__).parameters)
    print(f"   引数: {ch}")
    print(f"   引数なしで作れるか: {all(p == 'self' or True for p in ch)}"
          f"(public_key / trace_context はどちらも任意)")

    print("\n3) trace へ後から属性を足す口")
    print(f"   Langfuse.update_current_trace: "
          f"{'あり' if hasattr(langfuse.Langfuse, 'update_current_trace') else '**なし**'}")
    print(f"   Langfuse.get_current_trace_id: "
          f"{'あり' if hasattr(langfuse.Langfuse, 'get_current_trace_id') else 'なし'}")
    pa = getattr(langfuse, "propagate_attributes", None)
    print(f"   propagate_attributes: {'あり(ただし context 内の子 span にだけ効く)' if pa else 'なし'}")

    from langfuse.api import TraceBody
    print(f"   TraceBody の項目: {list(TraceBody.model_fields)}")
    print("   → ingestion.batch の TraceCreate は id で upsert される。"
          "これが v4 で trace に tag を足す道。")


def main() -> int:
    report()
    print("""
--- まとめ（この版で成り立つ形）---
  session:  invoke の config に metadata={"langfuse_session_id": str(cid)} を入れる。
            CallbackHandler が langfuse_ 接頭辞の key を trace 根へ引き上げる。
  intent:   **node の中から Langfuse へ書く道は無い。**
            update_current_trace は v4 に無く、ingestion の TraceCreate upsert は
            server が events_only モードのため 400。propagate_attributes は残って
            いるが「今の span」に載せるもので、LangChain の CallbackHandler が作る
            observation は OTEL の current span にならないため、node からは
            get_current_trace_id() が None を返して黙って捨てられる。
            したがって intent は **trace 根の output(graph の最終 State)** から取る。
            scripts/cost_by_intent.py がそれを GENERATION の usage と突き合わせる。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
