"""/api/observability(可観測性の画面の API)のテスト。09 章の最後の一段。

この層は **成果物を読むだけ** で、平均も割合も推奨しきい値も 1 つも計算し直さない
(app/api/observability.py の docstring)。したがってこのモジュールが押さえるのも
「計算が正しいか」ではなく、**読めなかったときにどう振る舞うか**に寄る:

1. 成果物が 1 つも無いのは異常ではなく「まだ実行していない」状態である。clone 直後は
   必ずそうなる。500 ではなく、**どの job を押せばよいか**を返す。
2. 「script は動いたが窓に何も無かった」と「script をまだ動かしていない」は別物で、
   運用者の次の行動が違う。前者に「まず実行してください」と出すと、実行済みの人を
   同じところで無限に足踏みさせる。
3. **section ごとに閉じる。** トレンドは MySQL、他の 2 つはローカルのファイルという
   別々の依存を持つ。MySQL が落ちている間にコストの内訳まで読めなくなるのは割に
   合わない(app/api/admin.py がカードごとに try を閉じるのと同じ考え方)。

router だけを載せた最小の app で叩く理由は tests/test_review_api.py と同じ。
"""

import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import observability as obs
from app.config import settings
from app.db import repository

pytestmark = pytest.mark.asyncio(loop_scope="session")

_app = FastAPI()
_app.include_router(obs.router)


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=_app), base_url="http://obstest")


# scripts/cost_by_intent.py が書く形。**手で作る**(実際に make cost-report を回すと
# Langfuse の起動が要り、テストが環境に依存する)。
_COST = {
    "generated_at": "2026-09-10T02:00:00+00:00",
    "days": 7,
    "traces": 11,
    "resolved": 7,
    "unknown": 4,
    "total_tokens": 25182,
    "rows": [
        {"intent": "配送", "count": 3, "tokens": 15548, "avg_tokens": 5182,
         "share": 0.6174, "share_label": "62%"},
        {"intent": "商品相談", "count": 1, "tokens": 5322, "avg_tokens": 5322,
         "share": 0.2113, "share_label": "21%"},
        {"intent": "雑談", "count": 1, "tokens": 1429, "avg_tokens": 1429,
         "share": 0.0567, "share_label": "6%"},
    ],
}

# scripts/calibrate_confidence.py が書く形。
_CALIBRATION = {
    "generated_at": "2026-09-10T02:10:00+00:00",
    "strategy": "hybrid_rerank",
    "distributions": [
        {"name": "答えられる(ABC)", "n": 180, "min": 0.0, "p25": 0.0,
         "p50": 0.732, "p75": 0.765, "max": 0.907},
        {"name": "断るべき(D)", "n": 60, "min": 0.0, "p25": 0.0,
         "p50": 0.0, "p75": 0.0, "max": 0.67},
    ],
    "scan": [
        {"t": 0.4, "tpr": 0.694, "fpr": 0.033, "j": 0.661},
        {"t": 0.45, "tpr": 0.683, "fpr": 0.017, "j": 0.667},
        {"t": 0.7, "tpr": 0.583, "fpr": 0.0, "j": 0.583},
    ],
    "recommended_threshold": 0.45,
    "best_j": 0.667,
    "weak_separation": False,
}


@pytest.fixture
def artifacts(monkeypatch, tmp_path):
    """成果物の置き場を tmp へ寄せる。既定では 2 つとも「まだ無い」状態。

    戻り値の write() で中身を置く。文字列をそのまま渡せば壊れた JSON も作れる。
    """
    monkeypatch.setattr(obs, "COST_JSON", tmp_path / "cost_by_intent.json")
    monkeypatch.setattr(obs, "CALIBRATION_JSON", tmp_path / "confidence_calibration.json")

    def write(which: str, body) -> None:
        path = obs.COST_JSON if which == "cost" else obs.CALIBRATION_JSON
        text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        path.write_text(text, encoding="utf-8")

    return write


@pytest.fixture
def no_runs(monkeypatch):
    """eval_runs を空にする(MySQL を使わない section のテスト用)。"""

    async def _empty(limit: int = 10):
        return []

    monkeypatch.setattr(repository, "list_eval_runs", _empty)


async def _overview() -> dict:
    async with _client() as c:
        resp = await c.get("/api/observability/overview")
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# 1. 何も実行していない状態
# ---------------------------------------------------------------------------


async def test_nothing_has_been_run_yet_is_a_state_not_an_error(artifacts, no_runs):
    """clone 直後の状態。3 つとも present=false で、**押すべき job の名前**が付く。

    ここを 500 にすると、初めて画面を開いた人には「壊れている」としか見えない。
    """
    body = await _overview()

    assert set(body) == {"cost", "trend", "calibration"}
    for key, job in (("cost", "cost-report"), ("trend", "eval-flywheel"),
                     ("calibration", "calibrate-confidence")):
        sec = body[key]
        assert sec["present"] is False, key
        assert sec["status"] == "not_run", key
        assert sec["message"], key
        # 手がかり = どの job を押せばよいか。job の状態と本文の両方から辿れること
        assert sec["job"]["name"] == job, key
        assert job in sec["hint"], key


# ---------------------------------------------------------------------------
# 2. コストの成果物はそのまま通す
# ---------------------------------------------------------------------------


async def test_cost_rows_are_passed_through_untouched(artifacts, no_runs):
    """行は成果物のまま。平均も割合もここでは計算し直さない。

    端末の `make cost-report` と画面の数字が食い違うのは、どちらかがもう一度
    計算しているときだけなので、**通しているか**を項目ごとに突き合わせる。
    """
    artifacts("cost", _COST)
    cost = (await _overview())["cost"]

    assert cost["present"] is True
    assert cost["status"] == "ok"
    assert cost["rows"] == _COST["rows"], "成果物の行を作り替えている"
    assert cost["total_tokens"] == 25182
    assert cost["traces"] == 11
    assert cost["unknown"] == 4
    assert cost["days"] == 7
    assert cost["generated_at"] == _COST["generated_at"]

    # 結論(最もコストが高い intent)を選ぶのは 1 か所だけ。画面側で最大値を探させない
    assert cost["top"] == _COST["rows"][0]
    assert cost["rows"][0]["tokens"] == max(r["tokens"] for r in cost["rows"])


# ---------------------------------------------------------------------------
# 3. 実行済みだが窓に何も無い
# ---------------------------------------------------------------------------


async def test_report_without_traces_in_the_window_asks_for_conversations(artifacts, no_runs):
    """未実行と「実行したが空だった」を混ぜない。運用者の次の行動が違う。

    ここで「まず make cost-report を実行してください」と出すと、実行済みの人は
    何度押しても同じ表示に戻ってくる。要るのは「先に何会話か流す」の方。
    """
    artifacts("cost", {**_COST, "rows": [], "traces": 0, "resolved": 0,
                       "unknown": 0, "total_tokens": 0})
    cost = (await _overview())["cost"]

    assert cost["status"] == "missing", "空の成果物が未実行と同じ扱いになっている"
    assert cost["present"] is True, "script は実際に走っている"
    assert cost["rows"] == []
    assert cost["top"] is None
    assert "会話" in cost["hint"], "次にすることが読めない"
    assert "cost-report" not in cost["hint"], "実行済みの人にもう一度実行させている"


async def test_calibration_without_a_scan_is_missing_too(artifacts, no_runs):
    artifacts("calibration", {**_CALIBRATION, "scan": [], "distributions": [],
                              "recommended_threshold": None})
    cal = (await _overview())["calibration"]

    assert cal["status"] == "missing"
    assert cal["recommended_threshold"] is None
    assert cal["in_sync"] is None, "比べる相手が無いのに同期の判定を出している"


# ---------------------------------------------------------------------------
# 4. トレンドは eval_runs を読む
# ---------------------------------------------------------------------------


async def test_trend_reads_eval_runs_newest_first(db_session_factory, artifacts):
    """トレンドの正本は成果物ではなく eval_runs。新しい順に並ぶこと。"""
    old = await repository.insert_eval_run(
        "scheduled", 40, {"recall_at_10": 0.700, "mrr": 0.600,
                          "faithfulness": 0.900, "refusal_rate": 0.950})
    new = await repository.insert_eval_run(
        "manual", 42, {"recall_at_10": 0.820, "mrr": 0.600,
                       "faithfulness": 0.800, "refusal_rate": 0.950})

    trend = (await _overview())["trend"]

    assert trend["present"] is True
    assert trend["status"] == "ok"
    assert [r["id"] for r in trend["runs"]] == [new, old]
    assert trend["runs"][0]["dataset_size"] == 42
    assert trend["runs"][0]["triggered_by_label"] == "手動実行"
    assert trend["runs"][1]["triggered_by_label"] == "定期実行"
    # 時刻は DB の値をそのまま返し、UTC であることを明示する(app/api/observability.py)
    assert trend["runs"][0]["created_at"]
    assert trend["timezone"] == "UTC"

    cells = trend["runs"][0]["cells"]
    assert cells["recall_at_10"]["arrow"] == "↑"
    assert cells["recall_at_10"]["warn"] is False
    assert cells["faithfulness"]["arrow"] == "↓"
    assert cells["faithfulness"]["warn"] is True, "悪い方へ動いた指標に印が付いていない"
    assert cells["mrr"]["arrow"] == "→", "誤差の範囲の揺れを増減として拾っている"
    # 最も古い行には比較の相手がいない
    assert trend["runs"][1]["cells"]["recall_at_10"]["arrow"] is None
    assert trend["dropped"] == ["faithfulness"]


async def test_trend_metric_metadata_matches_the_script():
    """表に出す指標と ⚠ の付け方を script と一致させる。

    ここがずれると、端末のトレンドでは ⚠ が付く動きに画面では何も出ない
    (しかも両方それらしく見えるので誰も気づけない)。
    """
    from scripts import eval_flywheel as fw

    assert [m["key"] for m in obs.TREND_METRICS] == fw.METRIC_NAMES
    assert {m["key"]: m["direction"] for m in obs.TREND_METRICS} == fw.DIRECTIONS
    assert obs.TREND_EPS == fw.EPS
    assert obs.TREND_LIMIT == fw.TREND_LIMIT


async def test_cost_fixture_matches_what_the_script_writes():
    """このモジュールの手書きの成果物を、**producer と結び付ける。**

    _COST は手で書いた辞書で、script が書く形と何のテストでも結ばれていなかった。
    share_label や avg_tokens が script 側で変われば、両方緑のまま画面だけ壊れる
    (09 章で実際に起きた「tag_intent が効いていないのにテストは緑」と同じ形)。

    上流にも Langfuse にも触らない純関数だけを呼ぶ。
    """
    from scripts import cost_by_intent as cost

    produced = cost.report({"配送": {"count": 3, "tokens": 15548}}, 11, 4, 7)

    assert set(produced) == set(_COST), "成果物の項目が script と食い違っている"
    assert set(produced["rows"][0]) == set(_COST["rows"][0])
    # 行の中身の作り方(平均と割合はここで確定させる)も同じであること
    assert cost.build_rows({"配送": {"count": 3, "tokens": 15548}})[0]["avg_tokens"] == 5182
    assert cost.build_rows({"配送": {"count": 3, "tokens": 15548}})[0]["share_label"] == "100%"


async def test_calibration_fixture_matches_what_the_script_writes():
    """校正の成果物も producer と結び付ける(scan / recommended_threshold / weak_separation)。"""
    from scripts import calibrate_confidence as cal

    dists = [d for d in (cal.build_distribution("答えられる(ABC)", [0.7, 0.8]),
                         cal.build_distribution("断るべき(D)", [0.1])) if d]
    produced = cal.build_report(dists, cal.build_scan([0.7, 0.8], [0.1]))

    assert set(produced) == set(_CALIBRATION)
    assert set(produced["distributions"][0]) == set(_CALIBRATION["distributions"][0])
    assert set(produced["scan"][0]) == set(_CALIBRATION["scan"][0])
    assert produced["strategy"] == _CALIBRATION["strategy"]
    assert isinstance(produced["weak_separation"], bool)


# ---------------------------------------------------------------------------
# 5. 推奨しきい値と実際の設定
# ---------------------------------------------------------------------------


async def test_recommendation_out_of_sync_with_the_running_setting(
    artifacts, no_runs, monkeypatch
):
    """校正の結論が設定に反映されていなければ、そう言うこと。

    レポートを出しただけで .env を直し忘れる、が最も起きやすい取りこぼしなので、
    画面が黙って両方の数字を並べるだけにはしない。
    """
    artifacts("calibration", _CALIBRATION)
    monkeypatch.setattr(settings, "evidence_confidence_threshold", 0.60)

    cal = (await _overview())["calibration"]
    assert cal["recommended_threshold"] == 0.45
    assert cal["active_threshold"] == 0.60
    assert cal["in_sync"] is False

    monkeypatch.setattr(settings, "evidence_confidence_threshold", 0.45)
    cal = (await _overview())["calibration"]
    assert cal["in_sync"] is True
    assert cal["scan"] == _CALIBRATION["scan"], "走査の行を作り替えている"
    assert cal["distributions"] == _CALIBRATION["distributions"]
    assert cal["best_j"] == 0.667


# ---------------------------------------------------------------------------
# 6. 壊れた成果物
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("broken", [
    pytest.param('{"generated_at": "2026-09-10", "rows": [', id="truncated"),
    pytest.param("", id="empty-file"),
    pytest.param('{"rows": "壊れている"}', id="rows-is-not-a-list"),
    pytest.param("[1, 2, 3]", id="not-an-object"),
    pytest.param("\x00\x01", id="binary-garbage"),
])
async def test_half_written_artifacts_are_treated_as_not_run(artifacts, no_runs, broken):
    """途中で死んだ実行の書きかけで 500 にしない。

    成果物は script が上書きで書くので、実行が途中で落ちれば必ずこの形になる。
    画面が 500 になると、**壊れていることを画面から知る手段が無くなる**。
    """
    artifacts("cost", broken)
    artifacts("calibration", broken)
    body = await _overview()

    for key in ("cost", "calibration"):
        sec = body[key]
        assert sec["present"] is False, key
        assert sec["status"] == "not_run", key
        assert sec["message"], key
        assert sec["rows" if key == "cost" else "scan"] == [], key
    # 未実行と壊れているのは文言で見分けられること(次にすることが違う)
    assert body["cost"]["message"] != obs.COST_NOT_RUN


# ---------------------------------------------------------------------------
# 7. section ごとに閉じる
# ---------------------------------------------------------------------------


async def test_a_dead_database_only_breaks_the_trend_section(artifacts, monkeypatch):
    """MySQL が落ちてもコストと校正は通常どおり返る。

    トレンドだけが自分の失敗を名乗る。ここをまとめて 1 つの try で包むと、
    MySQL が落ちている間は可観測性の画面ごと読めなくなる。
    """

    async def boom(limit: int = 10):
        raise RuntimeError("MySQL が停止している")

    monkeypatch.setattr(repository, "list_eval_runs", boom)
    artifacts("cost", _COST)
    artifacts("calibration", _CALIBRATION)

    body = await _overview()

    assert body["trend"]["status"] == "error"
    assert body["trend"]["present"] is False
    assert body["trend"]["runs"] == []
    assert "RuntimeError" in body["trend"]["message"], "型名すら出ないと切り分けができない"
    # 例外本文は載せない。SQLAlchemy の接続エラーは URL ごとパスワードを含む
    assert "MySQL が停止している" not in body["trend"]["message"]

    assert body["cost"]["status"] == "ok"
    assert body["cost"]["rows"] == _COST["rows"]
    assert body["calibration"]["status"] == "ok"
    assert body["calibration"]["recommended_threshold"] == 0.45
