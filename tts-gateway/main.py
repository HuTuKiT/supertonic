import os
import re
from pathlib import Path
from typing import Literal, Optional

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


TTS_URL = _env("TTS_URL", "http://127.0.0.1:7788/v1/tts")
BASE_PUBLIC_URL = _env("BASE_PUBLIC_URL", "http://127.0.0.1:7799").rstrip("/")
OUTPUT_DIR = Path(_env("OUTPUT_DIR", "/data/out"))
CALLBACK_TIMEOUT_SEC = float(_env("CALLBACK_TIMEOUT_SEC", "10"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TTS Gateway", version="1.0.0")


def _safe_user_id(user_id: str) -> str:
    # giữ đơn giản: chỉ cho a-zA-Z0-9_- . Nếu khác thì thay bằng _
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
    if not cleaned:
        raise ValueError("user_id is empty")
    return cleaned[:128]


class GatewayTTSRequest(BaseModel):
    user_id: str = Field(..., description="Id người dùng (dùng làm tên file output)")
    text: str = Field(..., min_length=1, description="Nội dung văn bản TTS")
    voice: str = Field("M1", description="Giọng đọc")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Tốc độ")
    lang: str = Field("en", description="Ngôn ngữ")
    steps: int = Field(8, ge=1, le=64, description="Số steps của model")
    response_format: Literal["wav"] = Field("wav", description="Định dạng output (hiện hỗ trợ wav)")
    callback_url: Optional[str] = Field(
        None,
        description="Nếu có, gateway sẽ POST trạng thái + output_url về backend sau khi generate xong",
    )


class GatewayTTSResponse(BaseModel):
    user_id: str
    output_filename: str
    output_url: str
    status: Literal["queued"]


def _output_paths(user_id: str, fmt: str) -> tuple[str, Path]:
    safe = _safe_user_id(user_id)
    filename = f"{safe}.{fmt}"
    return filename, (OUTPUT_DIR / filename)


def _post_callback(callback_url: str, payload: dict) -> None:
    try:
        requests.post(callback_url, json=payload, timeout=CALLBACK_TIMEOUT_SEC)
    except Exception:
        # Không raise để tránh làm fail request chính
        return


def _run_tts_job(req: GatewayTTSRequest) -> None:
    filename, out_path = _output_paths(req.user_id, req.response_format)
    output_url = f"{BASE_PUBLIC_URL}/files/{filename}"

    if req.callback_url:
        _post_callback(
            req.callback_url,
            {
                "user_id": req.user_id,
                "status": "started",
                "output_url": output_url,
                "output_filename": filename,
            },
        )

    try:
        r = requests.post(
            TTS_URL,
            json={
                "text": req.text,
                "voice": req.voice,
                "lang": req.lang,
                "steps": req.steps,
                "speed": req.speed,
                "response_format": req.response_format,
            },
            timeout=300,
        )
        r.raise_for_status()
    except Exception as e:
        if req.callback_url:
            _post_callback(
                req.callback_url,
                {
                    "user_id": req.user_id,
                    "status": "failed",
                    "output_url": output_url,
                    "output_filename": filename,
                    "error": str(e),
                },
            )
        return

    try:
        out_path.write_bytes(r.content)
    except Exception as e:
        if req.callback_url:
            _post_callback(
                req.callback_url,
                {
                    "user_id": req.user_id,
                    "status": "failed",
                    "output_url": output_url,
                    "output_filename": filename,
                    "error": f"write_failed: {e}",
                },
            )
        return

    if req.callback_url:
        _post_callback(
            req.callback_url,
            {
                "user_id": req.user_id,
                "status": "done",
                "output_url": output_url,
                "output_filename": filename,
                "bytes": out_path.stat().st_size,
            },
        )


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/gateway/tts", response_model=GatewayTTSResponse)
def gateway_tts(req: GatewayTTSRequest, bg: BackgroundTasks):
    try:
        filename, _ = _output_paths(req.user_id, req.response_format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    output_url = f"{BASE_PUBLIC_URL}/files/{filename}"

    # "Trước khi generate sẽ có 1 đường link" → trả output_url ngay lập tức
    bg.add_task(_run_tts_job, req)

    return GatewayTTSResponse(
        user_id=req.user_id,
        output_filename=filename,
        output_url=output_url,
        status="queued",
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

