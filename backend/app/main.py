import asyncio
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import settings

app = FastAPI(title="Clusture MRI Segmentation Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

root = Path(settings.DATA_ROOT).expanduser().resolve()
root.mkdir(parents=True, exist_ok=True)

# In-memory store is suitable only for a single-process demo. Use persistent storage
# and a queue for a multi-worker or production deployment.
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
gpu_lock = threading.Lock()

MODALITIES = ("flair", "t1", "t1ce", "t2")
ALLOWED_SUFFIXES = (".nii", ".nii.gz")

class JobSummary(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    message: str | None = None
    prediction_available: bool = False

def now():
    return datetime.now(timezone.utc).isoformat()

def check_api_key(x_api_key: str | None):
    configured = settings.API_KEY
    if not configured or configured == "REPLACE_WITH_A_LONG_RANDOM_SECRET":
        raise HTTPException(503, "Backend API key has not been configured.")
    if not x_api_key or not __import__("hmac").compare_digest(x_api_key, configured):
        raise HTTPException(401, "Invalid API key.")

def valid_nifti_name(filename: str | None) -> bool:
    if not filename:
        return False
    lower = filename.lower()
    return any(lower.endswith(s) for s in ALLOWED_SUFFIXES)

async def save_upload(upload: UploadFile, destination: Path):
    total = 0
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    with destination.open("wb") as out:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                out.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(413, f"{upload.filename} exceeds the {settings.MAX_UPLOAD_MB} MB limit.")
            out.write(chunk)
    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, f"{upload.filename} is empty.")

def update_job(job_id: str, **fields):
    with jobs_lock:
        jobs[job_id].update(fields)
        jobs[job_id]["updated_at"] = now()

def run_inference(job_id: str, job_dir: Path):
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "queued"
            jobs[job_id]["updated_at"] = now()

        command_template = settings.command_template
        if not command_template:
            raise RuntimeError(
                "Inference is not configured. Set NNUNET_COMMAND_TEMPLATE in backend/.env "
                "to the exact Docker command already verified on your server."
            )

        # One job at a time per backend process to reduce GPU contention.
        with gpu_lock:
            update_job(job_id, status="running", message="Inference is running on the GPU server.")
            replacements = {
                "{input_dir}": str(input_dir),
                "{output_dir}": str(output_dir),
                "{case_id}": job_id,
            }
            command = []
            for arg in command_template:
                for placeholder, value in replacements.items():
                    arg = arg.replace(placeholder, value)
                command.append(arg)

            # No shell=True: prevents shell injection. Only administrators configure the command.
            completed = subprocess.run(
                command,
                cwd=str(job_dir),
                capture_output=True,
                text=True,
                timeout=settings.INFERENCE_TIMEOUT_SECONDS,
                check=False,
            )
            (job_dir / "stdout.log").write_text(completed.stdout[-20000:], encoding="utf-8")
            (job_dir / "stderr.log").write_text(completed.stderr[-20000:], encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Inference command exited with code {completed.returncode}. "
                    f"Inspect server-side stderr.log for details."
                )

        candidates = sorted(
            p for p in output_dir.rglob("*")
            if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))
        )
        if not candidates:
            raise RuntimeError("Inference finished but no .nii/.nii.gz prediction was found in the output directory.")

        # If the image creates multiple NIfTI files, this demo selects the first.
        # Change this to the exact expected output filename once the command is confirmed.
        prediction = candidates[0]
        update_job(
            job_id,
            status="completed",
            message="Inference completed. Review the result with a qualified radiologist.",
            prediction_path=str(prediction),
            prediction_filename=prediction.name,
        )
    except subprocess.TimeoutExpired:
        update_job(job_id, status="failed", message="Inference timed out. Check server logs and configured timeout.")
    except Exception as exc:
        # Avoid exposing command lines or environment secrets to the browser.
        update_job(job_id, status="failed", message=str(exc)[:500])

@app.get("/health")
def health():
    return {"status": "ok", "service": "clusture-mri-segmentation-demo"}

@app.post("/api/jobs", response_model=JobSummary)
async def create_job(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    check_api_key(x_api_key)
    uploads = {"flair": flair, "t1": t1, "t1ce": t1ce, "t2": t2}
    for name, upload in uploads.items():
        if not valid_nifti_name(upload.filename):
            raise HTTPException(400, f"{name} must be a .nii or .nii.gz file.")

    job_id = uuid.uuid4().hex
    job_dir = root / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    output_dir = job_dir / "output"
    output_dir.mkdir()

    try:
        # Save with stable, expected modality names. Adapt naming if the verified
        # Docker command requires a different dataset convention.
        for name, upload in uploads.items():
            suffix = ".nii.gz" if (upload.filename or "").lower().endswith(".nii.gz") else ".nii"
            await save_upload(upload, input_dir / f"{job_id}_{name}{suffix}")
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    record = {
        "job_id": job_id, "status": "queued", "created_at": now(), "updated_at": now(),
        "message": "Upload accepted; waiting for inference.", "job_dir": str(job_dir),
        "prediction_path": None, "prediction_filename": None,
    }
    with jobs_lock:
        jobs[job_id] = record

    threading.Thread(target=run_inference, args=(job_id, job_dir), daemon=True).start()
    return JobSummary(**{k: record.get(k) for k in JobSummary.model_fields})

@app.get("/api/jobs/{job_id}", response_model=JobSummary)
def get_job(job_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    check_api_key(x_api_key)
    with jobs_lock:
        record = jobs.get(job_id)
        if not record:
            raise HTTPException(404, "Job not found. The demo backend may have restarted or the job ID is invalid.")
        copy = dict(record)
    return JobSummary(
        job_id=copy["job_id"], status=copy["status"], created_at=copy["created_at"],
        updated_at=copy["updated_at"], message=copy.get("message"),
        prediction_available=bool(copy.get("prediction_path") and Path(copy["prediction_path"]).is_file()),
    )

@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    check_api_key(x_api_key)
    with jobs_lock:
        record = jobs.get(job_id)
        if not record:
            raise HTTPException(404, "Job not found.")
        path = record.get("prediction_path")
        filename = record.get("prediction_filename") or "prediction.nii.gz"
        status = record.get("status")
    if status != "completed" or not path or not Path(path).is_file():
        raise HTTPException(409, "Prediction is not ready.")
    return FileResponse(path, filename=filename, media_type="application/octet-stream")

@app.get("/api/jobs")
def list_jobs(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    check_api_key(x_api_key)
    with jobs_lock:
        return [
            JobSummary(
                job_id=j["job_id"], status=j["status"], created_at=j["created_at"],
                updated_at=j["updated_at"], message=j.get("message"),
                prediction_available=bool(j.get("prediction_path") and Path(j["prediction_path"]).is_file()),
            )
            for j in sorted(jobs.values(), key=lambda x: x["created_at"], reverse=True)
        ]
