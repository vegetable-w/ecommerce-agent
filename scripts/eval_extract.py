"""/api/extractにラベル付きサンプルを入力し、order_idとrequest_typeを照合する(expected_solutionは目視確認)。"""
import json
import pathlib
import sys

import httpx

SAMPLES = pathlib.Path(__file__).parent.parent / "tests/data/extract_samples.json"
BASE = "http://localhost:8000"

def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    failures = 0
    for i, s in enumerate(samples, 1):
        exp = s["expected"]
        try:
            r = httpx.post(f"{BASE}/api/extract", json={"text": s["text"]}, timeout=60)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            failures += 1
            body = exc.response.text[:120]
            print(f"[FAIL] #{i} {s['text'][:24]}...")
            print(f"       期待 order_id={exp['order_id']} type={exp['request_type']}")
            print(f"       実際 HTTPエラー status={exc.response.status_code} body={body!r}")
            continue
        except httpx.HTTPError as exc:
            failures += 1
            print(f"[FAIL] #{i} {s['text'][:24]}...")
            print(f"       期待 order_id={exp['order_id']} type={exp['request_type']}")
            print(f"       実際 通信エラー {exc!r}")
            continue

        got = r.json()
        ok = got["order_id"] == exp["order_id"] and got["request_type"] == exp["request_type"]
        status = "PASS" if ok else "FAIL"
        failures += not ok
        print(f"[{status}] #{i} {s['text'][:24]}...")
        print(f"       期待 order_id={exp['order_id']} type={exp['request_type']}")
        print(f"       実際 order_id={got['order_id']} type={got['request_type']} 対応={got['expected_solution']}")
    print(f"\n{len(samples) - failures}/{len(samples)} 合格")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
