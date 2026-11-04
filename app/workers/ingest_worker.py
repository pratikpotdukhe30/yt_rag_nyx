import contextvars
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.services.ingest_service import IngestService, get_ingest_service

logger = get_logger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


ACTIVE = {JobStatus.PENDING, JobStatus.RUNNING}


@dataclass
class Job:
    id: str
    video_id: str
    language: str | None
    status: JobStatus = JobStatus.PENDING
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None     # same {code, message} shape as API errors


class JobStore:
    """In-memory now; Redis later behind the same methods."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def get_or_create_active(self, video_id: str, language: str | None) -> tuple[Job, bool]:
        # Check + create under ONE lock → atomic. Two simultaneous requests can't both create a job.
        with self._lock:
            for job in self._jobs.values():
                if job.video_id == video_id and job.status in ACTIVE:
                    return job, False
            job = Job(id=uuid.uuid4().hex, video_id=video_id, language=language)
            self._jobs[job.id] = job
            return job, True

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for k, v in changes.items():
                setattr(job, k, v)


class IngestWorker:
    def __init__(self, service: IngestService, store: JobStore, max_workers: int):
        self._service = service
        self._store = store
        # Bounded pool = backpressure. Threads are fine: torch releases the GIL during embedding.
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest")

    def submit(self, video_id: str, language: str | None) -> Job:
        job, created = self._store.get_or_create_active(video_id, language)
        if created:
            # Run inside a COPY of the current context → worker logs keep the request_id
            ctx = contextvars.copy_context()
            self._executor.submit(ctx.run, self._run, job.id)
            logger.info("Ingest job queued", extra={"extra_fields": {"job_id": job.id, "video_id": video_id}})
        else:
            logger.info("Ingest job deduplicated", extra={"extra_fields": {"job_id": job.id, "video_id": video_id}})
        return job

    def get(self, job_id: str) -> Job | None:
        return self._store.get(job_id)

    def _run(self, job_id: str) -> None:
        job = self._store.get(job_id)
        self._store.update(job_id, status=JobStatus.RUNNING)
        try:
            result = self._service.ingest(job.video_id, job.language)
            self._store.update(job_id, status=JobStatus.COMPLETED, result=asdict(result))
            logger.info("Ingest job completed", extra={"extra_fields": {"job_id": job_id}})
        except AppError as e:
            # Expected domain failure (no transcript, YouTube down...) → stored as data
            self._store.update(job_id, status=JobStatus.FAILED, error={"code": e.code, "message": e.message})
            logger.warning("Ingest job failed", extra={"extra_fields": {"job_id": job_id, "code": e.code}})
        except Exception:
            # A bug: full trace in logs, generic error for the client
            self._store.update(
                job_id, status=JobStatus.FAILED,
                error={"code": "internal_error", "message": "Ingestion failed unexpectedly"},
            )
            logger.exception("Ingest job crashed", extra={"extra_fields": {"job_id": job_id}})

    def shutdown(self) -> None:
        # Called at app shutdown: stop accepting work, drop queued jobs, don't hang the process
        self._executor.shutdown(wait=False, cancel_futures=True)


@lru_cache
def get_ingest_worker() -> IngestWorker:
    return IngestWorker(get_ingest_service(), JobStore(), get_settings().ingest_max_workers)