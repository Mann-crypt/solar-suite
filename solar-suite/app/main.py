import asyncio
import json
import time
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from models import loss_correction, rt_correction, aeromal

app = FastAPI(title="Solar Suite API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten once you know your frontend's origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory stores. Fine for an internal team; swap for Redis if this ever
# needs to survive restarts or scale past a single process.
# ---------------------------------------------------------------------------
FILES: dict[str, bytes] = {}          # file_id -> raw excel bytes
FILE_TS: dict[str, float] = {}        # file_id -> upload time, for cleanup
JOBS: dict[str, dict] = {}            # job_id -> {"status", "result"|"error"}

FILE_TTL_SECONDS = 60 * 60 * 4        # drop cached files after 4 hours


def _cleanup_files():
    now = time.time()
    expired = [fid for fid, ts in FILE_TS.items() if now - ts > FILE_TTL_SECONDS]
    for fid in expired:
        FILES.pop(fid, None)
        FILE_TS.pop(fid, None)


async def _run_job(job_id: str, fn, *args, **kwargs):
    JOBS[job_id]["status"] = "running"
    try:
        result = await asyncio.to_thread(fn, *args, **kwargs)
        JOBS[job_id] = {"status": "done", "result": result}
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}


def _parse_rows(edited_rows: Optional[str]):
    if not edited_rows:
        return None
    return json.loads(edited_rows)


# ---------------------------------------------------------------------------
# File upload (shared by Loss Correction)
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_excel(file: UploadFile = File(...)):
    _cleanup_files()
    contents = await file.read()
    file_id = str(uuid.uuid4())
    FILES[file_id] = contents
    FILE_TS[file_id] = time.time()
    try:
        detected = loss_correction.detect_workbook(contents)
    except Exception as e:
        raise HTTPException(400, f"Could not read workbook: {e}")
    return {"file_id": file_id, **detected}


def _get_file(file_id: str) -> bytes:
    contents = FILES.get(file_id)
    if contents is None:
        raise HTTPException(404, "file_id not found or expired — re-upload the file")
    return contents


# ---------------------------------------------------------------------------
# Loss Correction
# ---------------------------------------------------------------------------

@app.post("/api/loss-correction/fixed")
async def lc_fixed(file_id: str = Form(...), edited_rows: Optional[str] = Form(None)):
    contents = _get_file(file_id)
    rows = _parse_rows(edited_rows)
    try:
        result = await asyncio.to_thread(loss_correction.run_fixed, contents, rows)
    except Exception as e:
        raise HTTPException(400, str(e))
    return result


@app.post("/api/loss-correction/tracking/optimize")
async def lc_tracking_optimize(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    edited_rows: Optional[str] = Form(None),
):
    contents = _get_file(file_id)
    rows = _parse_rows(edited_rows)
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued"}
    background_tasks.add_task(_run_job, job_id, loss_correction.optimize_tracking, contents, rows)
    return {"job_id": job_id}


@app.post("/api/loss-correction/tracking/recalculate")
async def lc_tracking_recalculate(
    file_id: str = Form(...),
    params: str = Form(...),
    edited_rows: Optional[str] = Form(None),
):
    contents = _get_file(file_id)
    rows = _parse_rows(edited_rows)
    try:
        parsed_params = json.loads(params)
        result = await asyncio.to_thread(
            loss_correction.recalculate_tracking, contents, parsed_params, rows
        )
    except Exception as e:
        raise HTTPException(400, str(e))
    return result


# ---------------------------------------------------------------------------
# RT Correction
# ---------------------------------------------------------------------------

@app.post("/api/rt-correction/optimize")
async def rt_optimize(background_tasks: BackgroundTasks, actual: str = Form(...), trend: str = Form(...)):
    actual_rows = json.loads(actual)
    trend_rows = json.loads(trend)
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued"}
    background_tasks.add_task(_run_job, job_id, rt_correction.optimize, actual_rows, trend_rows)
    return {"job_id": job_id}


@app.post("/api/rt-correction/recalculate")
async def rt_recalculate(actual: str = Form(...), trend: str = Form(...), params: str = Form(...)):
    actual_rows = json.loads(actual)
    trend_rows = json.loads(trend)
    parsed_params = json.loads(params)
    result = await asyncio.to_thread(rt_correction.recalculate, actual_rows, trend_rows, parsed_params)
    return result


# ---------------------------------------------------------------------------
# Aeromal (password-gated)
# ---------------------------------------------------------------------------

@app.post("/api/aeromal/login")
async def aeromal_login(password: str = Form(...)):
    if not aeromal.check_password(password):
        raise HTTPException(403, "Incorrect password")
    return {"ok": True}


@app.post("/api/aeromal/curtailment")
async def aeromal_curtailment(
    password: str = Form(...),
    power: str = Form(...),
    peak_cap: float = Form(...),
    target_width: float = Form(...),
    window_length: int = Form(...),
    power_availability: float = Form(100),
    shift: Optional[int] = Form(None),
):
    if not aeromal.check_password(password):
        raise HTTPException(403, "Incorrect password")
    power_values = json.loads(power)
    try:
        result = await asyncio.to_thread(
            aeromal.run_curtailment, power_values, peak_cap, target_width, window_length,
            power_availability, shift,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


@app.post("/api/aeromal/no-curtailment")
async def aeromal_no_curtailment(
    password: str = Form(...),
    power: str = Form(...),
    window_length: int = Form(11),
    power_availability: float = Form(100),
    shift: Optional[int] = Form(None),
):
    if not aeromal.check_password(password):
        raise HTTPException(403, "Incorrect password")
    power_values = json.loads(power)
    try:
        result = await asyncio.to_thread(
            aeromal.run_no_curtailment, power_values, window_length, power_availability, shift,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


# ---------------------------------------------------------------------------
# Job status polling (used by Loss Correction / RT Correction optimize calls)
# ---------------------------------------------------------------------------

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve the frontend last so /api/* routes above take priority
app.mount("/", StaticFiles(directory="/app/frontend", html=True), name="frontend")
