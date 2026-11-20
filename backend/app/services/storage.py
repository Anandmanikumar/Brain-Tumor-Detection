
import shutil
import uuid
from pathlib import Path
from typing import Dict

from fastapi import UploadFile


# backend/data is the default storage location.
BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = BACKEND_DIR / "data"

INPUT_DIR_NAME = "input"
OUTPUT_DIR_NAME = "output"

ALLOWED_MODALITIES = {"FLAIR", "T1", "T1ce", "T2"}


class StorageError(Exception):
    """Raised when a storage operation fails."""
    pass


def initialize_storage() -> None:
    """Create the storage root directory."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def create_job_directory(job_id: str) -> Dict[str, Path]:
    """Create isolated input and output directories for a job."""

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError as exc:
        raise StorageError("Invalid job ID.") from exc

    initialize_storage()

    job_dir = DATA_ROOT / str(job_uuid)
    input_dir = job_dir / INPUT_DIR_NAME
    output_dir = job_dir / OUTPUT_DIR_NAME

    input_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)

    return {
        "job_dir": job_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
    }


def validate_nifti_filename(filename: str) -> str:
    """Validate and normalize a NIfTI filename."""

    if not filename:
        raise StorageError("Filename is missing.")

    # Ignore any directory components supplied by the client.
    safe_name = Path(filename.replace("\\", "/")).name

    if not (
        safe_name.endswith(".nii")
        or safe_name.endswith(".nii.gz")
    ):
        raise StorageError(
            "Only .nii and .nii.gz files are accepted."
        )

    if safe_name in {".nii", ".nii.gz"}:
        raise StorageError("Invalid NIfTI filename.")

    return safe_name


async def save_mri_upload(
    upload: UploadFile,
    destination_dir: Path,
    modality: str,
    max_size_bytes: int,
) -> Path:
    """Save one MRI modality with a size limit."""

    if modality not in ALLOWED_MODALITIES:
        raise StorageError(
            f"Unsupported modality: {modality}"
        )

    original_name = validate_nifti_filename(
        upload.filename or ""
    )

    suffix = (
        ".nii.gz"
        if original_name.endswith(".nii.gz")
        else ".nii"
    )

    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / f"{modality}{suffix}"

    if destination.exists():
        raise StorageError(
            f"File already exists for modality {modality}."
        )

    total_bytes = 0

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > max_size_bytes:
                    raise StorageError(
                        f"{modality} exceeds the upload size limit."
                    )

                output.write(chunk)

        if total_bytes == 0:
            raise StorageError(
                f"{modality} file is empty."
            )

        return destination

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        await upload.close()


def get_job_directory(job_id: str) -> Dict[str, Path]:
    """Resolve an existing job directory safely."""

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError as exc:
        raise StorageError("Invalid job ID.") from exc

    job_dir = DATA_ROOT / str(job_uuid)

    if not job_dir.is_dir():
        raise StorageError("Job not found.")

    return {
        "job_dir": job_dir,
        "input_dir": job_dir / INPUT_DIR_NAME,
        "output_dir": job_dir / OUTPUT_DIR_NAME,
    }


def find_segmentation_result(job_id: str) -> Path:
    """Find the generated NIfTI segmentation result."""

    directories = get_job_directory(job_id)
    output_dir = directories["output_dir"]

    if not output_dir.is_dir():
        raise StorageError("Output directory does not exist.")

    results = sorted(
        list(output_dir.glob("*.nii.gz"))
        + list(output_dir.glob("*.nii"))
    )

    if len(results) == 0:
        raise StorageError("No segmentation result found.")

    if len(results) > 1:
        raise StorageError(
            "Multiple segmentation results found. "
            "An explicit result selection is required."
        )

    return results[0]


def delete_job_files(job_id: str) -> None:
    """Delete all files belonging to one job."""

    directories = get_job_directory(job_id)

    shutil.rmtree(directories["job_dir"])
