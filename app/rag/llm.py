from functools import lru_cache

from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from app.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError, with_circuit_breaker
from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Only TRANSIENT failures trigger fallback. Auth/bad-request errors must surface as bugs.
_FALLBACK_ON = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


def _build_primary(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,          # SecretStr accepted directly
        temperature=0,                            # deterministic, grounded answers
        timeout=settings.llm_timeout_seconds,     # never wait forever
        max_retries=settings.llm_max_retries,     # SDK retries with backoff BEFORE we fall back
        base_url=settings.openai_base_url,
    )


def _build_fallback(settings: Settings) -> ChatGroq:
    # Same chat interface as ChatOpenAI → the chain can't tell which one answered
    return ChatGroq(
        model=settings.groq_fallback_model,
        api_key=settings.groq_api_key,          # SecretStr accepted directly, no unwrapping
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
@lru_cache  # build clients once; they hold HTTP connection pools
@lru_cache
def get_llm() -> Runnable:
    settings = get_settings()
    breaker = CircuitBreaker(
        "openai",
        failure_threshold=settings.breaker_failure_threshold,
        recovery_seconds=settings.breaker_recovery_seconds,
    )
    primary = with_circuit_breaker(_build_primary(settings), breaker, _FALLBACK_ON)
    logger.info(
        "LLM configured",
        extra={"extra_fields": {"primary": settings.openai_model, "fallback": settings.groq_fallback_model}},
    )
    return primary.with_fallbacks(
        [_build_fallback(settings)],
        exceptions_to_handle=_FALLBACK_ON + (CircuitOpenError,),   # open circuit → straight to Groq
    )