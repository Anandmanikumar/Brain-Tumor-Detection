
import hmac
import shutil
import threading
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import settings
from .services.validation import (
    NiftiValidationError,
    validate_mri_study,
)
from .services.inference import (
    InferenceError,
    create_inference_engine,
)


app = FastAPI(
    title="Clusture MRI Segmentation Demo",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# --------------------------------------------------
# STORAGE AND JOB MANAGEMENT
# --------------------------------------------------

root = Path(settings.DATA_ROOT).expanduser().resolve()
root.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict[str, Any]] = {}

jobs_lock = threading.Lock()
gpu_lock = threading.Lock()

MODALITIES = ("FLAIR", "T1", "T1ce", "T2")

ALLOWED_SUFFIXES = (".nii", ".nii.gz")


class JobSummary(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    message: str | None = None
    prediction_available: bool = False
    result_kind: str | None = None


def now():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

def check_api_key(x_api_key: str | None):

    configured = settings.API_KEY

    if (
        not configured
        or configured == "REPLACE_WITH_A_LONG_RANDOM_SECRET"
    ):
        raise HTTPException(
            503,
            "Backend API key has not been configured.",
        )

    if (
        not x_api_key
        or not hmac.compare_digest(x_api_key, configured)
    ):
        raise HTTPException(
            401,
            "Invalid API key.",
        )


# --------------------------------------------------
# UPLOAD VALIDATION AND STORAGE
# --------------------------------------------------

def valid_nifti_name(filename: str | None) -> bool:

    if not filename:
        return False

    lower = filename.lower()

    return any(
        lower.endswith(suffix)
        for suffix in ALLOWED_SUFFIXES
    )


async def save_upload(
    upload: UploadFile,
    destination: Path,
):

    total = 0

    limit = settings.MAX_UPLOAD_MB * 1024 * 1024

    try:
        with destination.open("wb") as output:

            while True:

                chunk = await upload.read(1024 * 1024)

                if not chunk:
                    break

                total += len(chunk)

                if total > limit:
                    raise HTTPException(
                        413,
                        f"{upload.filename} exceeds the "
                        f"{settings.MAX_UPLOAD_MB} MB limit.",
                    )

                output.write(chunk)

        if total == 0:
            raise HTTPException(
                400,
                f"{upload.filename} is empty.",
            )

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        await upload.close()


def update_job(job_id: str, **fields):

    with jobs_lock:

        if job_id not in jobs:
            return

        jobs[job_id].update(fields)
        jobs[job_id]["updated_at"] = now()


# --------------------------------------------------
# BACKGROUND INFERENCE
# --------------------------------------------------

def run_inference(
    job_id: str,
    job_dir: Path,
):

    input_dir = job_dir / "input"
    output_dir = job_dir / "output"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mode = settings.INFERENCE_MODE.strip().lower()

    try:

        update_job(
            job_id,
            status="processing",
            message="Preparing inference engine.",
        )

        engine = create_inference_engine(
            mode=mode,
            command_template=settings.command_template,
        )

        # One inference job at a time per backend process.
        with gpu_lock:

            update_job(
                job_id,
                status="processing",
                message="Inference is running.",
            )

            result_path = engine.predict(
                input_dir=input_dir,
                output_dir=output_dir,
                case_id=job_id,
            )

        if not result_path.is_file():
            raise InferenceError(
                "Inference returned a missing result file."
            )

        # Mock inference is never treated as a segmentation.
        if mode == "mock":

            update_job(
                job_id,
                status="completed",
                message=(
                    "Workflow simulation completed. "
                    "No tumor segmentation was performed."
                ),
                result_kind="mock",
                prediction_path=None,
                prediction_filename=None,
                artifact_path=str(result_path),
            )

        elif mode == "nnunet":

            if not (
                result_path.name.lower().endswith(".nii")
                or result_path.name.lower().endswith(".nii.gz")
            ):
                raise InferenceError(
                    "The inference result is not a NIfTI file."
                )

            update_job(
                job_id,
                status="completed",
                message=(
                    "Segmentation inference completed. "
                    "Review the result with a qualified radiologist."
                ),
                result_kind="segmentation",
                prediction_path=str(result_path),
                prediction_filename=result_path.name,
                artifact_path=None,
            )

        else:
            raise InferenceError(
                f"Unsupported inference mode: {mode}"
            )

    except Exception as exc:

        # Detailed errors remain on the backend server.
        try:
            (job_dir / "error.log").write_text(
                str(exc),
                encoding="utf-8",
            )
        except OSError:
            pass

        update_job(
            job_id,
            status="failed",
            message=(
                "Inference failed. Check the backend server logs."
            ),
            error=str(exc)[:2000],
            prediction_path=None,
            prediction_filename=None,
            artifact_path=None,
        )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "clusture-mri-segmentation-demo",
        "inference_mode": settings.INFERENCE_MODE,
    }


# --------------------------------------------------
# CREATE MRI JOB
# --------------------------------------------------

@app.post(
    "/api/jobs",
    response_model=JobSummary,
)
async def create_job(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
):

    check_api_key(x_api_key)

    uploads = {
        "FLAIR": flair,
        "T1": t1,
        "T1ce": t1ce,
        "T2": t2,
    }

    for name, upload in uploads.items():

        if not valid_nifti_name(upload.filename):

            raise HTTPException(
                400,
                f"{name} must be a .nii or .nii.gz file.",
            )

    job_id = uuid.uuid4().hex

    job_dir = root / job_id

    input_dir = job_dir / "input"
    output_dir = job_dir / "output"

    input_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    output_dir.mkdir()

    try:

        # Store files using consistent internal modality names.
        for name, upload in uploads.items():

            original_name = upload.filename or ""

            suffix = (
                ".nii.gz"
                if original_name.lower().endswith(".nii.gz")
                else ".nii"
            )

            destination = input_dir / f"{name}{suffix}"

            await save_upload(
                upload,
                destination,
            )

        # Validate all four NIfTI files before accepting the job.
        modality_paths = {
            name: next(
                path
                for path in (
                    input_dir / f"{name}.nii",
                    input_dir / f"{name}.nii.gz",
                )
                if path.is_file()
            )
            for name in MODALITIES
        }

        validation_report = validate_mri_study(
            modality_paths
        )

    except NiftiValidationError as exc:

        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )

        raise HTTPException(
            400,
            f"MRI validation failed: {exc}",
        )

    except Exception:

        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )

        raise

    record = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now(),
        "updated_at": now(),
        "message": validation_report["message"],
        "job_dir": str(job_dir),
        "prediction_path": None,
        "prediction_filename": None,
        "result_kind": None,
        "artifact_path": None,
    }

    with jobs_lock:
        jobs[job_id] = record

    worker = threading.Thread(
        target=run_inference,
        args=(job_id, job_dir),
        daemon=True,
    )

    worker.start()

    return JobSummary(
        **{
            key: record[key]
            for key in (
                "job_id",
                "status",
                "created_at",
                "updated_at",
                "message",
                "result_kind",
            )
        }
    )


# --------------------------------------------------
# GET JOB STATUS
# --------------------------------------------------

@app.get(
    "/api/jobs/{job_id}",
    response_model=JobSummary,
)
def get_job(
    job_id: str,
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
):

    check_api_key(x_api_key)

    with jobs_lock:

        record = jobs.get(job_id)

        if not record:
            raise HTTPException(
                404,
                "Job not found. The backend may have restarted.",
            )

        copy = dict(record)

    prediction_available = bool(
        copy.get("prediction_path")
        and Path(copy["prediction_path"]).is_file()
    )

    return JobSummary(
        job_id=copy["job_id"],
        status=copy["status"],
        created_at=copy["created_at"],
        updated_at=copy["updated_at"],
        message=copy.get("message"),
        prediction_available=prediction_available,
        result_kind=copy.get("result_kind"),
    )


# --------------------------------------------------
# DOWNLOAD SEGMENTATION
# --------------------------------------------------

@app.get("/api/jobs/{job_id}/result")
def get_result(
    job_id: str,
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
):

    check_api_key(x_api_key)

    with jobs_lock:

        record = jobs.get(job_id)

        if not record:
            raise HTTPException(
                404,
                "Job not found.",
            )

        copy = dict(record)

    if copy["status"] != "completed":
        raise HTTPException(
            409,
            "Inference has not completed.",
        )

    if copy.get("result_kind") == "mock":
        raise HTTPException(
            409,
            "Mock inference does not produce a segmentation mask.",
        )

    path = copy.get("prediction_path")

    if not path or not Path(path).is_file():
        raise HTTPException(
            404,
            "Segmentation result is unavailable.",
        )

    return FileResponse(
        path,
        filename=copy["prediction_filename"],
        media_type="application/octet-stream",
    )


# --------------------------------------------------
# LIST JOBS
# --------------------------------------------------

@app.get("/api/jobs")
def list_jobs(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
):

    check_api_key(x_api_key)

    with jobs_lock:

        records = sorted(
            jobs.values(),
            key=lambda item: item["created_at"],
            reverse=True,
        )

        results = []

        for record in records:

            prediction_available = bool(
                record.get("prediction_path")
                and Path(record["prediction_path"]).is_file()
            )

            results.append(
                JobSummary(
                    job_id=record["job_id"],
                    status=record["status"],
                    created_at=record["created_at"],
                    updated_at=record["updated_at"],
                    message=record.get("message"),
                    prediction_available=prediction_available,
                    result_kind=record.get("result_kind"),
                )
            )

    return results
