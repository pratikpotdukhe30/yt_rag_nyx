from fastapi import APIRouter, Response

from app.api.deps import IngestWorkerDep
from app.core.exceptions import JobNotFoundError
from app.schemas.common import error_responses
from app.schemas.ingest import IngestJobResponse, IngestRequest
from app.workers.ingest_worker import Job

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _to_response(job: Job) -> IngestJobResponse:
    return IngestJobResponse(
        job_id=job.id, video_id=job.video_id, status=job.status.value,
        result=job.result, error=job.error,
    )


@router.post("", status_code=202, response_model=IngestJobResponse, responses=error_responses(422, 429))
async def ingest(req: IngestRequest, response: Response, worker: IngestWorkerDep) -> IngestJobResponse:
    # Now `async def`: the route only ENQUEUES (microseconds); heavy work happens in the worker.
    job = worker.submit(req.video_id, req.language)
    response.headers["Location"] = f"/v1/ingest/{job.id}"   # REST convention: where to poll
    return _to_response(job)


@router.get("/{job_id}", response_model=IngestJobResponse, responses=error_responses(404))
async def get_job(job_id: str, worker: IngestWorkerDep) -> IngestJobResponse:
    job = worker.get(job_id)
    if job is None:
        raise JobNotFoundError("Job not found", details={"job_id": job_id})
    return _to_response(job)