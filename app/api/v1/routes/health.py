from fastapi import APIRouter, Request

from app.rag.llm import get_openai_breaker
from app.rag.vectorstore import get_vector_store

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request) -> dict:
    get_vector_store().has_video("___________")          # live, cheap, local
    return {
        "status": "ready",
        "llm": request.app.state.llm_status,               # from the startup check, no token cost
        "openai_breaker": get_openai_breaker().state.value, # live: closed / open / half_open
    }