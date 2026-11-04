from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from app.api.errors import register_exception_handlers
from app.api.v1.middleware import request_context_middleware
from app.api.v1.routes import chat, health, ingest
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.rag.llm import get_llm
from app.rag.vectorstore import get_vector_store
from app.workers.ingest_worker import get_ingest_worker
from app.observability.tracing import configure_tracing
from app.rag.llm import check_llms, get_llm
from app.rag.embeddings import get_embeddings

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging()
    configure_tracing(settings)
    get_vector_store()
    get_embeddings().embed_query("warm up")   # ← new line
    get_llm()

    app.state.llm_status = await check_llms()
    logger.info("LLM status", extra={"extra_fields": app.state.llm_status})
    if not any(app.state.llm_status.values()):
        raise RuntimeError("No LLM provider reachable")

    logger.info("Application started", extra={"extra_fields": {"env": settings.env}})
    yield
    get_ingest_worker().shutdown()
    logger.info("Application stopped")

def create_app() -> FastAPI:
    # App FACTORY: tests can build a fresh app with overrides; no global side effects at import
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

    app.middleware("http")(request_context_middleware)
    register_exception_handlers(app)

    app.include_router(health.router)                 # unversioned infra endpoints
    v1 = APIRouter(prefix="/v1")
    v1.include_router(ingest.router)
    v1.include_router(chat.router)
    app.include_router(v1)
    return app


app = create_app()