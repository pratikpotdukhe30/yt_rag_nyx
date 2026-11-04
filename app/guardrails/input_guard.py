from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.exceptions import GuardrailViolationError, ServiceUnavailableError
from app.core.logging import get_logger
from app.observability.metrics import GUARD_BLOCKS

logger = get_logger(__name__)


class InputGuard:
    def __init__(self, classifier: BaseChatModel, threshold: float, fail_open: bool):
        self._classifier = classifier     # injected → tests use a fake that returns any score
        self._threshold = threshold
        self._fail_open = fail_open

    async def check(self, question: str) -> None:
        """Returns None if safe. Raises GuardrailViolationError if malicious."""
        try:
            msg = await self._classifier.ainvoke(question)
            score = float(str(msg.content).strip())   # Prompt Guard replies with a probability
        except Exception as e:
            # Broad catch is deliberate: this is an OPTIONAL safety layer at a system boundary.
            # Network error, rate limit, unexpected output → same decision: fail open or closed.
            return self._on_guard_failure(e)

        if score >= self._threshold:
            # details (score) go to logs only; user gets a neutral message (don't teach attackers)
            GUARD_BLOCKS.labels("input").inc()
            raise GuardrailViolationError(
                "This question can't be processed.", details={"guard_score": round(score, 4)}
            )

    def _on_guard_failure(self, error: Exception) -> None:
        logger.warning("Input guard unavailable", extra={"extra_fields": {"error": type(error).__name__}})
        if not self._fail_open:
            raise ServiceUnavailableError("Safety check unavailable, try again later") from error


@lru_cache
def get_input_guard() -> InputGuard:
    s = get_settings()
    classifier = ChatGroq(
        model=s.prompt_guard_model,
        api_key=s.groq_api_key,
        timeout=5,          # a guard must be FAST; never let it dominate request latency
        max_retries=1,
    )
    return InputGuard(classifier, s.prompt_guard_threshold, s.guard_fail_open)