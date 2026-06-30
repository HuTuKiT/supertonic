import os
import re
import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional, TypedDict

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


TTS_URL = _env("TTS_URL", "http://127.0.0.1:7788/v1/tts")
BASE_PUBLIC_URL = _env("BASE_PUBLIC_URL", "http://127.0.0.1:7799").rstrip("/")
OUTPUT_DIR = Path(_env("OUTPUT_DIR", "/data/out"))
CALLBACK_TIMEOUT_SEC = float(_env("CALLBACK_TIMEOUT_SEC", "10"))
MAX_CONCURRENCY = int(_env("MAX_CONCURRENCY", "2"))
QUEUE_MAXSIZE = int(_env("QUEUE_MAXSIZE", "500"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TTS Gateway", version="1.0.0")

_queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
_sem = asyncio.Semaphore(MAX_CONCURRENCY)


class Job(TypedDict, total=False):
    job_id: str
    user_id: str
    created_at: float
    started_at: float
    finished_at: float
    status: str  # queued|running|done|failed
    output_filename: str
    output_url: str
    error: str
    bytes: int
    request: Dict[str, Any]


_jobs: Dict[str, Job] = {}


def _safe_user_id(user_id: str) -> str:
    # giữ đơn giản: chỉ cho a-zA-Z0-9_- . Nếu khác thì thay bằng _
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
    if not cleaned:
        raise ValueError("user_id is empty")
    return cleaned[:128]


class GatewayTTSRequest(BaseModel):
    user_id: str = Field(..., description="Id người dùng")
    text: str = Field(..., min_length=1, description="Nội dung văn bản TTS")
    voice: str = Field("M1", description="Giọng đọc")
    speed: float = Field(1.0, ge=0.7, le=2.0, description="Tốc độ")
    lang: str = Field("vi", description="Ngôn ngữ")
    steps: int = Field(8, ge=5, le=16, description="Số steps của model")
    response_format: Literal["wav"] = Field("wav", description="Định dạng output (hiện hỗ trợ wav)")
    callback_url: Optional[str] = Field(
        None,
        description="Nếu có, gateway sẽ POST trạng thái + output_url về backend sau khi generate xong",
    )


class GatewayTTSResponse(BaseModel):
    job_id: str
    user_id: str
    output_filename: str
    output_url: str
    status: Literal["queued"]


class JobStatusResponse(BaseModel):
    job_id: str
    user_id: str
    status: Literal["queued", "running", "done", "failed"]
    output_filename: str
    output_url: str
    error: Optional[str] = None
    bytes: Optional[int] = None
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


def _output_paths(job_id: str, fmt: str) -> tuple[str, Path]:
    filename = f"{job_id}.{fmt}"
    return filename, (OUTPUT_DIR / filename)


def _post_callback(callback_url: str, payload: dict) -> None:
    try:
        requests.post(callback_url, json=payload, timeout=CALLBACK_TIMEOUT_SEC)
    except Exception:
        # Không raise để tránh làm fail request chính
        return


def _sync_generate_and_save(req_dict: Dict[str, Any], out_path: Path) -> int:
    r = requests.post(
        TTS_URL,
        json={
            "text": req_dict["text"],
            "voice": req_dict["voice"],
            "lang": req_dict["lang"],
            "steps": req_dict["steps"],
            "speed": req_dict["speed"],
            "response_format": req_dict["response_format"],
        },
        timeout=300,
    )
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path.stat().st_size


async def _run_one_job(job_id: str) -> None:
    job = _jobs.get(job_id)
    if not job:
        return

    async with _sem:
        job["status"] = "running"
        job["started_at"] = time.time()

        if job.get("request", {}).get("callback_url"):
            _post_callback(
                job["request"]["callback_url"],
                {
                    "job_id": job_id,
                    "user_id": job["user_id"],
                    "status": "started",
                    "output_url": job["output_url"],
                    "output_filename": job["output_filename"],
                },
            )

        try:
            _, out_path = _output_paths(job_id, job["request"]["response_format"])
            bytes_written = await asyncio.to_thread(_sync_generate_and_save, job["request"], out_path)
            job["bytes"] = int(bytes_written)
            job["status"] = "done"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
        finally:
            job["finished_at"] = time.time()

        if job.get("request", {}).get("callback_url"):
            payload: Dict[str, Any] = {
                "job_id": job_id,
                "user_id": job["user_id"],
                "status": job["status"],
                "output_url": job["output_url"],
                "output_filename": job["output_filename"],
            }
            if job["status"] == "done":
                payload["bytes"] = job.get("bytes")
            else:
                payload["error"] = job.get("error")
            _post_callback(job["request"]["callback_url"], payload)


async def _worker_loop() -> None:
    while True:
        job_id = await _queue.get()
        try:
            await _run_one_job(job_id)
        finally:
            _queue.task_done()


@app.get("/health")
def health():
    return {"ok": True}


@app.on_event("startup")
async def _startup() -> None:
    # tạo số worker = MAX_CONCURRENCY để xử lý queue
    for _ in range(max(1, MAX_CONCURRENCY)):
        asyncio.create_task(_worker_loop())


@app.post("/v1/gateway/tts", response_model=GatewayTTSResponse)
async def gateway_tts(req: GatewayTTSRequest):
    try:
        _ = _safe_user_id(req.user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = uuid.uuid4().hex
    filename, _ = _output_paths(job_id, req.response_format)
    output_url = f"{BASE_PUBLIC_URL}/files/{filename}"

    job: Job = {
        "job_id": job_id,
        "user_id": req.user_id,
        "created_at": time.time(),
        "status": "queued",
        "output_filename": filename,
        "output_url": output_url,
        "request": req.model_dump(),
    }
    _jobs[job_id] = job

    # "Trước khi generate sẽ có 1 đường link" → trả output_url ngay lập tức
    try:
        _queue.put_nowait(job_id)
    except asyncio.QueueFull:
        job["status"] = "failed"
        job["error"] = "queue_full"
        job["finished_at"] = time.time()
        raise HTTPException(status_code=429, detail="too many requests (queue full)")

    return GatewayTTSResponse(
        job_id=job_id,
        user_id=req.user_id,
        output_filename=filename,
        output_url=output_url,
        status="queued",
    )


@app.get("/v1/gateway/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(
        job_id=job_id,
        user_id=job["user_id"],
        status=job["status"],  # type: ignore[arg-type]
        output_filename=job["output_filename"],
        output_url=job["output_url"],
        error=job.get("error"),
        bytes=job.get("bytes"),
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
    )


@app.get("/files/{filename}")
def download_file(filename: str):
    # chặn path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)

