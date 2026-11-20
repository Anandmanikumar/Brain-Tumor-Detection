
from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    VALIDATING = "validating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResultKind(str, Enum):
    MOCK = "mock"
    SEGMENTATION = "segmentation"


class ModalityInfo(BaseModel):
    modality: str
    filename: str
    size_bytes: int


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(default=0, ge=0, le=100)
    message: str
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None


class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    result_kind: ResultKind
    message: str
    segmentation_available: bool = False
    segmentation_filename: Optional[str] = None
    modalities: List[str] = Field(default_factory=list)
