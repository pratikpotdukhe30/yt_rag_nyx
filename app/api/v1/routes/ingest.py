from dataclasses import asdict

from fastapi import APIRouter

from app.api.deps import IngestServiceDep
from app.schemas.common import error_responses
from app.schemas.ingest import IngestRequest, IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse, responses=error_responses(404, 422, 429, 502))
def ingest(req: IngestRequest, service: IngestServiceDep) -> IngestResponse:
    # Plain `def` on purpose: embedding is CPU-bound → FastAPI runs it in a thread pool.
    return IngestResponse(**asdict(service.ingest(req.video_id, req.language)))