import threading
import time
from enum import Enum
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from app.core.logging import get_logger

logger = get_logger(__name__)


class CircuitOpenError(Exception):
    """Internal signal: 'don't even try this dependency right now'. Triggers the fallback."""


class State(str, Enum):
    CLOSED = "closed"        # normal
    OPEN = "open"            # failing fast
    HALF_OPEN = "half_open"  # testing recovery


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int, recovery_seconds: float):
        self.name = name
        self._threshold = failure_threshold
        self._recovery = recovery_seconds
        self._state = State.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        # Requests run concurrently (thread pool + event loop) → protect shared state
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        return self._state

    def before_call(self) -> None:
        with self._lock:
            if self._state is State.OPEN:
                if time.monotonic() - self._opened_at >= self._recovery:
                    self._state = State.HALF_OPEN          # cooldown over → allow one probe
                    logger.info("Circuit half-open", extra={"extra_fields": {"breaker": self.name}})
                else:
                    raise CircuitOpenError(f"{self.name} circuit is open")

    def record_success(self) -> None:
        with self._lock:
            if self._state is not State.CLOSED:
                logger.info("Circuit closed", extra={"extra_fields": {"breaker": self.name}})
            self._state, self._failures = State.CLOSED, 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            # A failed probe re-opens immediately; otherwise open after N failures in a row
            if self._state is State.HALF_OPEN or self._failures >= self._threshold:
                self._state = State.OPEN
                self._opened_at = time.monotonic()           # monotonic: immune to clock changes
                logger.warning(
                    "Circuit opened",
                    extra={"extra_fields": {"breaker": self.name, "failures": self._failures}},
                )


def with_circuit_breaker(runnable: Runnable, breaker: CircuitBreaker, trip_on: tuple[type[Exception], ...]) -> Runnable:
    """Wrap any Runnable so it stays composable in LCEL (works with |, .with_fallbacks, ainvoke...)."""

    def _call(inp: Any, config: RunnableConfig) -> Any:
        breaker.before_call()
        try:
            out = runnable.invoke(inp, config)
        except trip_on:                 # only transient errors count, same rule as fallback
            breaker.record_failure()
            raise
        breaker.record_success()
        return out

    async def _acall(inp: Any, config: RunnableConfig) -> Any:
        breaker.before_call()
        try:
            out = await runnable.ainvoke(inp, config)
        except trip_on:
            breaker.record_failure()
            raise
        breaker.record_success()
        return out

    return RunnableLambda(_call, afunc=_acall, name=f"{breaker.name}_with_breaker")