"""/api/rag-eval/overview のテスト。

この API の役目は 1 つだけ:「すでに出来上がっているレポートをそのまま配る」。
`make eval-rag` は 80 問 × 4 戦略で上流 LLM を実際に呼ぶ数分がかりの処理なので、
画面を開いただけで再計算が走るのは事故になる(時間と課金の両方を焼く)。
したがってこのモジュールの主題は次の 2 点に置く:

- 画面を開いても上流を 1 回も呼ばないこと(test_opening_the_page_never_touches_upstream)
- 返す数値が artifact の値と 1 つも違わないこと(web 層で metric を作り直さない)

レポートが無い / 壊れている場合に 500 や白画面にしないことも、ここで固定する。
DB にも Milvus にも触れないため、すべて同期テストで書く。
"""

import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from app.api import rageval
from app.core import embeddings, jobs, llm, rerank, retrieval
from app.main import app

REAL_REPORT = (
    pathlib.Path(__file__).resolve().parent.parent / "data" / "04" / "reports" / "rag_eval.json"
)


@pytest.fixture()
def client():
    return TestClient(app)


def _metric(value: float, n: int = 20) -> dict:
    return {"value": value, "n": n}


def _buckets(a: float, b: float, c: float, overall: float) -> dict:
    return {
        "A_policy": _metric(a), "B_model": _metric(b),
        "C_colloquial": _metric(c), "overall": _metric(overall, 60),
    }


def _report(*, mrr: dict[str, float], generation: dict | None = ..., note: str = "日本語の覚書") -> dict:
    """テスト用のレポート。戦略ごとの overall MRR だけを指定して作る。"""
    strategies = list(mrr)
    report = {
        "meta": {
            "generated_at": "2026-09-07T11:19:05",
            "collection": "knowledge",
            "collection_rows": 47,
            "samples": 80,
            "bucket_counts": {"A_policy": 20, "B_model": 20, "C_colloquial": 20, "D_absent": 20},
            "k": 10,
            "strategies": strategies,
            "embed_model": "BAAI/bge-m3",
            "chat_model": "gpt-4o-mini",
            "generation_complete": generation is not None,
            "note": note,
        },
        "retrieval": {
            s: {
                "recall_at_k": _buckets(1.0, 1.0, 0.9, 0.95),
                "mrr": _buckets(1.0, 1.0, v, v),
                "per_sample": [{"id": "A1", "bucket": "A_policy", "recall": 1.0, "rr": 1.0}],
            }
            for s, v in mrr.items()
        },
        "evidence_coverage": {
            s: {
                "coverage": _buckets(1.0, 1.0, 0.85, 0.95),
                "per_sample": [{"id": "A1", "bucket": "A_policy", "coverage": 1.0}],
            }
            for s in strategies
        },
    }
    if generation is ...:
        generation = {
            s: {
                "answer_coverage": _buckets(0.9, 1.0, 0.8, 0.9),
                "refusal_rate": _metric(0.95),
                "per_sample": [],
            }
            for s in strategies
        }
        generation["faithfulness"] = {
            "strategy": strategies[0], "value": 0.9625, "n": 80, "per_sample": [],
        }
    if generation is not None:
        report["generation"] = generation
    return report


@pytest.fixture()
def artifact(monkeypatch, tmp_path):
    """レポートの置き場所を tmp へ逃がす。中身を書く関数を返す。"""
    path = tmp_path / "rag_eval.json"
    monkeypatch.setattr(rageval, "REPORT_PATH", path)

    def write(obj, *, raw: str | None = None) -> pathlib.Path:
        if raw is not None:
            path.write_text(raw, encoding="utf-8")
        else:
            path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    return write


# ---------------------------------------------------------------------------
# 再計算しないこと(この API の存在理由)
# ---------------------------------------------------------------------------


def test_opening_the_page_never_touches_upstream(artifact, client, monkeypatch):
    """埋め込み / リランク / チャット / 検索のどれか 1 つでも呼ぶと落ちる状態で 200 を返すこと。

    ここが赤くなる変更は「画面を開くと評価が走り出す」変更である。80 問 × 4 戦略の
    実呼び出しは数分と実費を焼くので、この 1 本は落としてはいけない。
    """
    called: list[str] = []

    def boom(name):
        async def _async(*args, **kwargs):
            called.append(name)
            raise AssertionError(f"上流 {name} が呼ばれた")

        return _async

    monkeypatch.setattr(embeddings, "embed_texts", boom("embed_texts"))
    monkeypatch.setattr(embeddings, "embed_query", boom("embed_query"))
    monkeypatch.setattr(rerank, "rerank", boom("rerank"))
    monkeypatch.setattr(retrieval, "search_knowledge", boom("search_knowledge"))

    def chat_boom(*args, **kwargs):
        called.append("get_chat_model")
        raise AssertionError("上流 get_chat_model が呼ばれた")

    monkeypatch.setattr(llm, "get_chat_model", chat_boom)

    artifact(_report(mrr={"dense": 0.96, "bm25": 0.77, "hybrid": 0.87, "hybrid_rerank": 0.97}))
    resp = client.get("/api/rag-eval/overview")

    assert resp.status_code == 200
    assert called == [], f"artifact を配るだけのはずが上流を呼んだ: {called}"
    assert resp.json()["present"] is True


def test_metrics_are_served_verbatim_from_the_artifact(artifact, client):
    """retrieval / evidence_coverage は file の中身と完全一致。web 層で作り直さない。"""
    report = _report(mrr={"dense": 0.96, "bm25": 0.77, "hybrid": 0.87, "hybrid_rerank": 0.97})
    path = artifact(report)
    on_disk = json.loads(path.read_text(encoding="utf-8"))

    body = client.get("/api/rag-eval/overview").json()

    assert body["retrieval"] == on_disk["retrieval"]
    assert body["evidence_coverage"] == on_disk["evidence_coverage"]
    assert body["generation"] == on_disk["generation"]
    assert body["meta"] == on_disk["meta"]


def test_japanese_survives_the_round_trip(artifact, client):
    """レポートは UTF-8。cp932 で読むと化けるか落ちるので、日本語を 1 往復させて固定する。"""
    report = _report(mrr={"dense": 0.9}, note="口語 bucket で 差が出た（全角）")
    artifact(report)
    body = client.get("/api/rag-eval/overview").json()
    assert body["meta"]["note"] == "口語 bucket で 差が出た（全角）"


# ---------------------------------------------------------------------------
# 結論(best)は artifact の数値に従う
# ---------------------------------------------------------------------------


def test_best_strategy_is_the_highest_overall_mrr(artifact, client):
    artifact(_report(mrr={"dense": 0.961, "bm25": 0.772, "hybrid": 0.878, "hybrid_rerank": 0.976}))
    best = client.get("/api/rag-eval/overview").json()["best"]
    assert best["strategy"] == "hybrid_rerank"
    assert best["mrr"] == 0.976


def test_recall_is_read_from_the_new_key(artifact, client):
    """レポートが recall_at_5 を出していれば、それが Recall として読まれる。"""
    report = _report(mrr={"dense": 0.9})
    sec = report["retrieval"]["dense"]
    sec["recall_at_5"] = sec.pop("recall_at_k")
    sec["recall_at_5"]["overall"] = {"value": 0.83, "n": 240}
    artifact(report)
    assert client.get("/api/rag-eval/overview").json()["best"]["recall_at_k"] == 0.83


def test_recall_falls_back_to_the_old_key_for_past_reports(artifact, client):
    """指標キーを改名する前に作られたレポートでも Recall が空にならない。

    ここが抜けると、古い artifact を開いたときに例外も警告も出ないまま
    Recall の行だけが「取得不可」になり、画面を見ても原因が分からない。
    """
    report = _report(mrr={"dense": 0.9})
    report["retrieval"]["dense"]["recall_at_k"]["overall"] = {"value": 0.77, "n": 60}
    assert "recall_at_5" not in report["retrieval"]["dense"]
    artifact(report)
    assert client.get("/api/rag-eval/overview").json()["best"]["recall_at_k"] == 0.77


def test_best_strategy_follows_the_data_not_the_name(artifact, client):
    """数値を入れ替えれば結論も変わる(戦略名を決め打ちしていないこと)。"""
    artifact(_report(mrr={"dense": 0.40, "bm25": 0.99, "hybrid": 0.50, "hybrid_rerank": 0.60}))
    best = client.get("/api/rag-eval/overview").json()["best"]
    assert best["strategy"] == "bm25"
    assert best["mrr"] == 0.99


def test_best_carries_the_generation_numbers_of_that_strategy(artifact, client):
    report = _report(mrr={"dense": 0.4, "hybrid_rerank": 0.9})
    report["generation"]["hybrid_rerank"]["answer_coverage"]["overall"] = {"value": 0.9083, "n": 60}
    report["generation"]["hybrid_rerank"]["refusal_rate"] = {"value": 0.95, "n": 20}
    artifact(report)
    best = client.get("/api/rag-eval/overview").json()["best"]
    assert best["strategy"] == "hybrid_rerank"
    assert best["answer_coverage"] == 0.9083
    assert best["refusal_rate"] == 0.95


# ---------------------------------------------------------------------------
# artifact が無い / 壊れている / 途中まで
# ---------------------------------------------------------------------------


def test_missing_artifact_is_a_state_not_an_error(monkeypatch, tmp_path, client):
    monkeypatch.setattr(rageval, "REPORT_PATH", tmp_path / "does-not-exist.json")
    resp = client.get("/api/rag-eval/overview")

    assert resp.status_code == 200, "レポート未生成が 500 になっている(画面が白くなる)"
    body = resp.json()
    assert body["present"] is False
    assert body["retrieval"] is None and body["generation"] is None
    assert body["generation_done"] is False
    assert body["best"] is None
    assert body["message"], "未実行の理由が日本語で入っていない"
    assert body["job"]["name"] == "eval-rag", "実行すべき job 名を返していない"


@pytest.mark.parametrize("raw", [
    '{"meta": {"samples": 80}, "retrieval": {"dense": {"mrr"',   # 途中で切れた JSON
    "",                                                          # 空ファイル
    "[]",                                                        # dict ではない
    '{"meta": {"samples": 80}}',                                 # retrieval が無い
])
def test_broken_artifact_is_treated_as_not_run(artifact, client, raw):
    artifact(None, raw=raw)
    resp = client.get("/api/rag-eval/overview")
    assert resp.status_code == 200, f"壊れた artifact で 500 になった: {raw!r}"
    body = resp.json()
    assert body["present"] is False
    assert body["message"]


def test_generation_may_be_missing_while_retrieval_is_still_served(artifact, client):
    """judge 上流が止まった run。検索側はそのまま出し、生成側だけ未完了とする。"""
    report = _report(mrr={"dense": 0.96, "hybrid_rerank": 0.97}, generation=None)
    path = artifact(report)
    on_disk = json.loads(path.read_text(encoding="utf-8"))

    body = client.get("/api/rag-eval/overview").json()

    assert body["present"] is True
    assert body["retrieval"] == on_disk["retrieval"]
    assert body["evidence_coverage"] == on_disk["evidence_coverage"]
    assert body["generation"] is None
    assert body["generation_done"] is False
    assert body["best"]["strategy"] == "hybrid_rerank"
    assert body["best"]["answer_coverage"] is None
    assert body["best"]["refusal_rate"] is None


def test_generation_done_is_true_only_for_a_complete_run(artifact, client):
    artifact(_report(mrr={"dense": 0.9}))
    assert client.get("/api/rag-eval/overview").json()["generation_done"] is True

    report = _report(mrr={"dense": 0.9})
    report["meta"]["generation_complete"] = False
    artifact(report)
    assert client.get("/api/rag-eval/overview").json()["generation_done"] is False


# ---------------------------------------------------------------------------
# job 登録と画面の導線
# ---------------------------------------------------------------------------


def test_eval_rag_is_registered_as_a_heavy_job():
    spec = jobs.JOBS["eval-rag"]
    assert spec.target == "eval-rag"
    assert spec.heavy is True, "数分かかり実費も出るジョブが確認なしで押せる"


def test_job_status_is_included_so_the_page_can_offer_a_rerun(artifact, client):
    artifact(_report(mrr={"dense": 0.9}))
    job = client.get("/api/rag-eval/overview").json()["job"]
    assert job["name"] == "eval-rag"
    assert job["heavy"] is True
    assert set(job) >= {"running", "exit_code", "label"}


def test_page_is_reachable_and_uses_the_shared_shell(client):
    res = client.get("/rag-eval")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "/static/admin.js" in res.text


# ---------------------------------------------------------------------------
# 実物の artifact(あれば)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_REPORT.is_file(), reason="レポート未生成(clone 直後など)")
def test_the_real_report_on_disk_is_servable(client):
    """実物の rag_eval.json がこの API の期待する形のままであること。

    eval スクリプト側の出力形を変えたのに画面を直し忘れると、ここで気づける。
    """
    body = client.get("/api/rag-eval/overview").json()
    assert body["present"] is True
    on_disk = json.loads(REAL_REPORT.read_text(encoding="utf-8"))
    assert body["retrieval"] == on_disk["retrieval"]
    assert body["best"]["strategy"] in on_disk["meta"]["strategies"]
