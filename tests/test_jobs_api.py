"""ジョブランナーとその HTTP 出口のテスト。

この経路は「ブラウザからコマンド実行を起こせる」ため、ここのテストは他より重い意味を持つ。
特に test_unknown_job_names_are_rejected_before_spawning は、ホワイトリストを外したときに
確実に赤くなること(= 変異テスト)を確認したうえで残している。

DB にも Milvus にも触れないため、このモジュールはすべて同期テストで書く。
"""

import pathlib
import subprocess
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core import jobs
from app.main import app

MAKEFILE = pathlib.Path(__file__).resolve().parent.parent / "Makefile"


class _FakeProc:
    """poll() が None を返し続ける = 動いているプロセスのふり。"""

    def __init__(self, code: int | None = None) -> None:
        self.pid = 424242
        self._code = code

    def poll(self) -> int | None:
        return self._code

    def wait(self, timeout: float | None = None) -> int:
        return self._code or 0


@pytest.fixture()
def sandbox(monkeypatch, tmp_path):
    """本物の make を起動しないジョブランナー。起動しようとした argv を記録する。

    _spawn を差し替えるのは「起きなかったこと」を証明するため。例外を投げる
    スパイにすると、拒否されるべき名前が通ったときに 500 として現れてしまい、
    「拒否された」のか「起動して失敗した」のかがステータスコードから読めない。
    argv のリストが空のままであることを見るほうが、主張が正確になる。
    """
    spawned: list[list[str]] = []

    def fake_spawn(argv, log_file):
        spawned.append(list(argv))
        return _FakeProc()

    monkeypatch.setattr(jobs, "_spawn", fake_spawn)
    monkeypatch.setattr(jobs, "_runs", {})
    # 実物の log/acceptance/*.log を切り詰めないよう、書き先も逃がす
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    return spawned


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# ホワイトリスト(この経路の要)
# ---------------------------------------------------------------------------

# 「引数として渡せてしまわないか」を試す名前。パス片に入りうる形だけを並べる。
_HOSTILE_NAMES = [
    "kb-buildx",
    "clean",
    "kb-build --always-make",
    "kb-build;kb-reset",
    "kb-build&&kb-reset",
    "kb-build|kb-reset",
    "-f/tmp/Makefile",
    "kb-build $(whoami)",
    "kb-build`whoami`",
    "",
    " ",
    "KB-BUILD",
    "kb_build",
]


def test_unknown_job_names_are_rejected_before_spawning(sandbox, client):
    """一覧に無い名前は 404 で、プロセスは 1 つも起きない。

    これはホワイトリストそのもののテスト。app/core/jobs.py の start() から
    `if spec is None: raise UnknownJob` を外すと、名前がそのまま make の
    ターゲットになり、ここは赤くなる(変異テストで確認済み)。
    """
    for name in _HOSTILE_NAMES:
        assert name not in jobs.JOBS
        with pytest.raises(jobs.UnknownJob):
            jobs.start(name)
        # 名前ごとにその場で確認する。ループの最後にまとめて見ると、途中の名前が
        # 起動まで進んだ場合に別の例外(不正なファイル名など)で先に落ち、
        # 「プロセスが起きた」という本題が報告されなくなる
        assert sandbox == [], f"{name!r} でプロセスが起動した"
        assert jobs._runs == {}, f"{name!r} が実行中として登録された"

    # HTTP 経由でも同じ。URL に載る形の名前で確認する
    for name in ("kb-buildx", "clean", "kb-build%20--always-make", "kb-build;kb-reset"):
        resp = client.post(f"/api/jobs/{name}")
        assert resp.status_code == 404, name
    assert sandbox == []


def test_registered_job_argv_contains_only_make_and_target(sandbox):
    """起動する argv は [make の実体, ターゲット] の 2 要素だけ。

    リクエスト由来の文字列は 1 つも混ざらない(混ざる余地が無いことを形で固定する)。
    """
    jobs.start("kb-preview")
    assert len(sandbox) == 1
    argv = sandbox[0]
    assert len(argv) == 2
    assert argv[0] == jobs.resolve_make()
    assert argv[1] == "kb-preview"


def test_same_job_twice_returns_409(sandbox, client):
    jobs._runs["kb-preview"] = jobs._Run(
        proc=_FakeProc(), started_at=datetime.now(), argv=["make", "kb-preview"]
    )
    resp = client.post("/api/jobs/kb-preview")
    assert resp.status_code == 409
    assert sandbox == [], "実行中なのに 2 つ目のプロセスが起きた"

    # 終了済み(poll が終了コードを返す)なら再実行できる
    jobs._runs["kb-preview"] = jobs._Run(
        proc=_FakeProc(0), started_at=datetime.now(), argv=["make", "kb-preview"]
    )
    assert client.post("/api/jobs/kb-preview").status_code == 200
    assert len(sandbox) == 1


def test_every_registered_target_exists_in_makefile():
    """ボタンが存在しないレシピを指していないこと。

    Makefile を実際に読んで突き合わせる。JOBS 側の名前を変えたのに Makefile を
    直し忘れると、画面のボタンは 200 を返し、ログにだけ
    "No rule to make target" が出る(押した人には成功に見える)。
    """
    targets = _makefile_targets(MAKEFILE)
    assert targets, "Makefile からターゲットを 1 つも読み取れなかった"
    for name, spec in jobs.JOBS.items():
        assert spec.target in targets, f"{name} の make ターゲット {spec.target} が Makefile に無い"


def _makefile_targets(path: pathlib.Path) -> set[str]:
    """Makefile のターゲット名を読み取る。レシピ行(タブ始まり)と .PHONY は除く。"""
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0] in "\t #":
            continue
        if ":" not in line or "=" in line.split(":", 1)[0]:
            continue
        head = line.split(":", 1)[0].strip()
        if head.startswith("."):
            continue
        out.update(head.split())
    return out


def test_registry_is_exactly_the_agreed_set():
    """登録内容を固定する。ここを増やすことは「ブラウザから叩けるコマンドを増やす」ことに等しい。"""
    assert set(jobs.JOBS) == {
        "kb-preview", "kb-build", "kb-repatch", "kb-vectorize", "kb-mine",
        "kb-reset", "seed-conv", "eval-retrieval", "eval-mining", "eval-rag",
    }
    assert {n for n, s in jobs.JOBS.items() if s.heavy} == {"kb-mine", "kb-reset", "eval-rag"}


# ---------------------------------------------------------------------------
# 状態・ログ・停止
# ---------------------------------------------------------------------------


def test_list_jobs_returns_registry_with_status(client):
    body = client.get("/api/jobs").json()
    assert {j["name"] for j in body["jobs"]} == set(jobs.JOBS)
    for j in body["jobs"]:
        assert set(j) >= {"name", "target", "heavy", "running", "exit_code", "started_at"}


def test_status_of_unknown_job_is_404(client):
    assert client.get("/api/jobs/kb-buildx").status_code == 404
    assert client.post("/api/jobs/kb-buildx/stop").status_code == 404


def test_stop_when_not_running_is_409(sandbox, client):
    assert client.post("/api/jobs/kb-preview/stop").status_code == 409


def test_status_includes_log_tail(sandbox, client, monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    (tmp_path / "kb-preview.log").write_text(
        "\n".join(f"行 {i}" for i in range(200)), encoding="utf-8"
    )
    body = client.get("/api/jobs/kb-preview?lines=5").json()
    assert body["log"].splitlines() == ["行 195", "行 196", "行 197", "行 198", "行 199"]
    assert body["running"] is False


def test_tail_of_missing_log_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    assert jobs.tail("kb-preview") == ""
    with pytest.raises(jobs.UnknownJob):
        jobs.tail("kb-buildx")


# ---------------------------------------------------------------------------
# make が見つからない場合
# ---------------------------------------------------------------------------


def test_missing_make_reports_japanese_reason(monkeypatch, sandbox, client):
    """MAKE_BIN が解決できないときは FileNotFoundError ではなく、直し方の分かる 503。"""
    monkeypatch.setattr(settings, "make_bin", "make-that-does-not-exist-xyz")
    with pytest.raises(jobs.MakeNotFound) as exc:
        jobs.resolve_make()
    assert "MAKE_BIN" in str(exc.value)

    resp = client.post("/api/jobs/kb-preview")
    assert resp.status_code == 503
    assert "MAKE_BIN" in resp.json()["detail"]
    assert sandbox == []


def test_configured_make_bin_resolves_on_this_machine():
    """MAKE_BIN の設定が実際に解決できること。

    実測: アプリのプロセスからは shutil.which("make") が None になる。
    そのため settings.make_bin を経由しない実装(argv に素の "make" を置く等)は
    このマシンで必ず FileNotFoundError になる。
    """
    resolved = jobs.resolve_make()
    assert pathlib.Path(resolved).is_file()
    assert subprocess.run(
        [resolved, "--version"], capture_output=True, check=False
    ).returncode == 0
