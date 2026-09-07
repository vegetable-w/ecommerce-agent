"""/kb の再実行ボタンから make ターゲットを起動するジョブランナー。

ブラウザからコマンド実行を引き起こす経路なので、設計はこの 2 行に尽きる:

1. **argv はこのモジュールの中で確定する。** フロントエンドが送るのはジョブ名だけで、
   引数も、オプションも、対象ファイル名も送れない。
2. **ジョブ名はホワイトリストとの完全一致でしか通さない。** 一覧に無い名前は起動前に
   例外にする(下の start() の最初の 2 行)。

シェルは経由しない(shell=False の Popen に固定の argv リストを渡す)。したがって
仮に名前の検査を抜けても `;` や `&&` は解釈されないが、それは二重の防御であって、
ホワイトリストを外してよい理由にはならない。
"""

import logging
import os
import pathlib
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime

from app.config import settings

logger = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LOG_DIR = REPO_ROOT / "log" / "acceptance"


class JobError(Exception):
    """ジョブ操作の失敗。API 層がこの系統を HTTP ステータスへ翻訳する。"""


class UnknownJob(JobError):
    pass


class JobAlreadyRunning(JobError):
    pass


class JobNotRunning(JobError):
    pass


class MakeNotFound(JobError):
    pass


@dataclass(frozen=True)
class JobSpec:
    target: str
    heavy: bool
    label: str
    description: str


# ジョブ名 → make ターゲット。ここに無い名前は起動できない。
# heavy=True は「上流 LLM を呼ぶ」または「破壊的」なジョブで、画面側で確認を挟むための印。
JOBS: dict[str, JobSpec] = {
    "kb-preview": JobSpec(
        "kb-preview", False, "チャンク分割プレビュー",
        "data/kb の資料を分割した結果を集計する。DB にも Milvus にも書かない",
    ),
    "kb-build": JobSpec(
        "kb-build", False, "ナレッジ構築",
        "data/kb の資料を chunk 化して knowledge_chunks へ pending として登録する",
    ),
    "kb-repatch": JobSpec(
        "kb-repatch", False, "パッチ式の再取り込み",
        "data/kb の変更点だけを既存の chunk へ当て直す。変わっていない chunk は"
        "埋め込み直さず、chunk id も変えない",
    ),
    "kb-vectorize": JobSpec(
        "kb-vectorize", False, "ベクトル化",
        "pending の chunk を埋め込み、Milvus へ upsert する(冪等・再実行可能)",
    ),
    "kb-mine": JobSpec(
        "kb-mine", True, "会話からの知識抽出",
        "過去会話を LLM で問答に抽出する。上流 API を実際に呼ぶため課金される",
    ),
    "kb-reset": JobSpec(
        "kb-reset", True, "ナレッジ全消去",
        "knowledge_chunks と qa_extraction_staging を空にし、Milvus の collection を削除する",
    ),
    "seed-conv": JobSpec(
        "seed-conv", False, "会話サンプル投入",
        "sql/03-seed.sql を support へ流し込む",
    ),
    "eval-retrieval": JobSpec(
        "eval-retrieval", False, "検索の評価",
        "ラベル付きサンプルに対する検索の再現率を測る",
    ),
    "eval-mining": JobSpec(
        "eval-mining", False, "抽出の評価",
        "抽出済みサンプルに対する重複排除の挙動を測る",
    ),
    "eval-rag": JobSpec(
        "eval-rag", True, "RAG 評価（4戦略比較）",
        "300 問 × 4 戦略で埋め込み・リランク・生成・判定の上流を実際に呼ぶ。"
        "Milvus と構築済みナレッジが必要で、数十分かかり課金される",
    ),
}


@dataclass
class _Run:
    proc: subprocess.Popen
    started_at: datetime
    argv: list[str]


_runs: dict[str, _Run] = {}
# start/stop は状態辞書を読んでから書くまでの間に外部プロセスを起こす。
# TestClient は別スレッドでハンドラを回すため、ここは素の dict 操作では守れない。
_lock = threading.Lock()


def log_path(name: str) -> pathlib.Path:
    return LOG_DIR / f"{name}.log"


def resolve_make() -> str:
    """make の実体パスを返す。見つからなければ日本語で理由を述べて落とす。

    実測: このアプリのプロセスからは shutil.which("make") が None になる
    (winget で入れた make は Packages 配下に実体があるだけで PATH に載っていない)。
    素の Popen に "make" を渡すと FileNotFoundError という英語の裸の例外になり、
    画面には 500 しか出ない。設定 (MAKE_BIN) を見に行ったうえで、直せる形の
    メッセージにするのがこの関数の役目。
    """
    found = shutil.which(settings.make_bin)
    if found:
        return found
    # which は PATH と PATHEXT の探索なので拡張子付きの絶対パスでも通るはずだが、
    # 実行ビットの判定などで漏れる場合に備えて実ファイルの存在も見る
    p = pathlib.Path(settings.make_bin)
    if p.is_file():
        return str(p)
    raise MakeNotFound(
        f"make の実行ファイルが見つかりません(MAKE_BIN={settings.make_bin!r})。"
        ".env の MAKE_BIN に make.exe の実体パスを設定してください"
    )


def _spawn(argv: list[str], log_file) -> subprocess.Popen:
    """make を**独立したプロセスグループ**で起こす。

    make は uv を、uv は python を起こす。子だけを殺すと孫が残り、
    「止めたはずのジョブが DB を触り続けている」状態になる。
    グループにしておけば stop() でまとめて畳める。
    """
    kwargs = {}
    if sys.platform == "win32":
        # Windows には killpg が無いので、停止は taskkill /T に任せる。
        # CREATE_NEW_PROCESS_GROUP は、このサーバ自身への Ctrl-C が make まで
        # 伝播して勝手に止まるのを防ぐために付ける。
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        argv,
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,  # 標準出力と標準エラーを 1 本のログにまとめる
        stdin=subprocess.DEVNULL,  # 対話プロンプトで無言のまま固まらないように塞ぐ
        **kwargs,
    )


def is_running(name: str) -> bool:
    run = _runs.get(name)
    return run is not None and run.proc.poll() is None


def start(name: str) -> dict:
    """ジョブを起動する。名前は JOBS との完全一致のみ。

    この関数の最初の分岐がホワイトリストである。ここを外すと、URL のパス片が
    そのまま make のターゲット名になり、ブラウザから任意のターゲットを起動できる。
    """
    spec = JOBS.get(name)
    if spec is None:
        raise UnknownJob(f"未登録のジョブ名: {name!r}")
    make = resolve_make()  # 見つからなければ MakeNotFound(日本語)
    with _lock:
        if is_running(name):
            raise JobAlreadyRunning(f"ジョブ {name} は実行中です")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = log_path(name)
        # argv はここで組み立てる。外から来た値は spec.target の**選択**にしか使われず、
        # 文字列としては 1 文字も混ざらない。
        argv = [make, spec.target]
        started_at = datetime.now()
        # 追記ではなく毎回切り詰める。ログは「このジョブの最後の 1 回」を表す。
        # 追記にすると tail が前回の実行と混ざり、成功したのか失敗したのか読めなくなる。
        log_file = open(path, "w", encoding="utf-8", errors="replace")
        try:
            log_file.write(f"=== {name} 開始 {started_at:%Y-%m-%d %H:%M:%S} :: {argv} ===\n")
            log_file.flush()
            proc = _spawn(argv, log_file)
        finally:
            # 親側の fd は閉じてよい。子は複製した fd を持っている
            log_file.close()
        _runs[name] = _Run(proc=proc, started_at=started_at, argv=argv)
    logger.info("ジョブを起動 name=%s pid=%s", name, proc.pid)
    return status(name)


def status(name: str) -> dict:
    spec = JOBS.get(name)
    if spec is None:
        raise UnknownJob(f"未登録のジョブ名: {name!r}")
    base = {
        "name": name, "target": spec.target, "heavy": spec.heavy,
        "label": spec.label, "description": spec.description,
    }
    run = _runs.get(name)
    if run is None:
        return {**base, "running": False, "exit_code": None, "started_at": None, "pid": None}
    code = run.proc.poll()
    return {
        **base,
        "running": code is None,
        "exit_code": code,
        "started_at": run.started_at.isoformat(timespec="seconds"),
        "pid": run.proc.pid,
    }


def status_all() -> list[dict]:
    return [status(n) for n in JOBS]


def tail(name: str, lines: int = 60) -> str:
    if name not in JOBS:
        raise UnknownJob(f"未登録のジョブ名: {name!r}")
    path = log_path(name)
    if not path.is_file():
        return ""
    # ログは 1 ジョブ 1 実行分しか無く数百 KB を超えないので素直に全部読む。
    # errors="replace": make 自身のメッセージは UTF-8 とは限らない(cp932 の
    # コンソールメッセージが混じる)。ここで落ちると画面がログを一切出せなくなる。
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def stop(name: str) -> dict:
    """プロセス**グループごと**止める。make から uv、uv から python と続く孫を残さない。"""
    if name not in JOBS:
        raise UnknownJob(f"未登録のジョブ名: {name!r}")
    with _lock:
        run = _runs.get(name)
        if run is None or run.proc.poll() is not None:
            raise JobNotRunning(f"ジョブ {name} は実行中ではありません")
        pid = run.proc.pid
        if sys.platform == "win32":
            # taskkill /T は親子関係をたどって子孫を落とす。Popen.kill() は
            # make だけを落とし、uv と python が孤児として走り続ける(実測)。
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, check=False,
            )
        else:
            os.killpg(os.getpgid(pid), 9)
        try:
            run.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("ジョブ停止後もプロセスが残っている name=%s pid=%s", name, pid)
    logger.info("ジョブを停止 name=%s pid=%s", name, pid)
    return status(name)
