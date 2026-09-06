"""ジョブ操作の HTTP 出口。

ここはブラウザから届いた**ジョブ名だけ**を app.core.jobs へ渡す。パスパラメータを
make の引数として組み立てたり、クエリから追加の引数を受け取ったりしないこと。
argv を決めるのは app/core/jobs.py の側だけ、という一点でこの経路の安全性が決まっている。
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from app.core import jobs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
async def list_jobs() -> dict:
    return {"jobs": jobs.status_all()}


@router.post("/{name}")
async def start_job(name: str) -> dict:
    try:
        return jobs.start(name)
    except jobs.UnknownJob:
        # 一覧に無い名前。ここで 404 にすることが、この画面唯一の入力検査になる
        raise HTTPException(status_code=404, detail="そのジョブは登録されていません")
    except jobs.JobAlreadyRunning:
        raise HTTPException(status_code=409, detail=f"ジョブ {name} は実行中です。完了を待つか停止してください")
    except jobs.MakeNotFound as exc:
        # 設定の不足なので、直し方が分かる形でそのまま見せる(秘密は含まれない)
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/{name}")
async def job_status(name: str, lines: int = Query(default=60, ge=1, le=1000)) -> dict:
    try:
        state = jobs.status(name)
        return {**state, "log": jobs.tail(name, lines)}
    except jobs.UnknownJob:
        raise HTTPException(status_code=404, detail="そのジョブは登録されていません")


@router.post("/{name}/stop")
async def stop_job(name: str) -> dict:
    try:
        return jobs.stop(name)
    except jobs.UnknownJob:
        raise HTTPException(status_code=404, detail="そのジョブは登録されていません")
    except jobs.JobNotRunning:
        raise HTTPException(status_code=409, detail=f"ジョブ {name} は実行中ではありません")
