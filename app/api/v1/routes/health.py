from fastapi import APIRouter

from app.rag.vectorstore import get_vector_store

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict:
    # Readiness: dependencies usable. LLM pings get added here in the observability step.
    get_vector_store().has_video("___________")   # cheap query proves Chroma + embeddings work
    return {"status": "ready"}