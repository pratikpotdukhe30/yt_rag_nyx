from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from app.api.errors import register_exception_handlers
from app.api.v1.middleware import request_context_middleware
from app.api.v1.routes import chat, health, ingest
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.rag.llm import get_llm
from app.rag.vectorstore import get_vector_store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE at startup, before any request is accepted
    setup_logging()
    get_vector_store()   # loads the embedding model now, not on the first user's request
    get_llm()            # builds LLM clients + connection pools
    logger.info("Application started", extra={"extra_fields": {"env": get_settings().env}})
    yield
    # Runs once at shutdown: place to close connections, flush metrics
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