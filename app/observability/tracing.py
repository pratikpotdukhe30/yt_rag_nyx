import os

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def configure_tracing(settings: Settings) -> None:
    # Bridge: our typed settings → the env vars LangChain reads. Called once at startup.
    if not settings.langsmith_tracing or settings.langsmith_api_key is None:
        os.environ["LANGSMITH_TRACING"] = "false"
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    logger.info("LangSmith tracing enabled", extra={"extra_fields": {"project": settings.langsmith_project}})