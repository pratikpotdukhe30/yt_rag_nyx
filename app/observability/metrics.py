import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from prometheus_client import Counter, Gauge, Histogram

# ---------- HTTP ----------
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency",
    ["method", "handler", "status"],                      # handler = endpoint name → low cardinality
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

# ---------- RAG pipeline ----------
CACHE_LOOKUPS = Counter("answer_cache_lookups_total", "Answer cache lookups", ["result"])   # hit | miss
CHAT_OUTCOMES = Counter("chat_outcomes_total", "Chat results", ["outcome"])               # answered | not_covered | ungrounded
GUARD_BLOCKS = Counter("guardrail_blocks_total", "Requests blocked by guardrails", ["guard"])
INGEST_JOBS = Counter("ingest_jobs_total", "Finished ingest jobs", ["status"])            # completed | failed

# ---------- LLM (filled by the callback below) ----------
LLM_CALLS = Counter("llm_calls_total", "LLM calls", ["model", "status"])                  # success | error
LLM_LATENCY = Histogram("llm_call_duration_seconds", "LLM call latency", ["model"],
                        buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32))
LLM_TOKENS = Counter("llm_tokens_total", "Tokens used", ["model"])

# ---------- Resilience ----------
OPENAI_CIRCUIT_OPEN = Gauge("openai_circuit_open", "1 if the OpenAI circuit breaker is open")


class LLMMetricsCallback(BaseCallbackHandler):
    """LangChain callback: sees EVERY chat-model call (OpenAI or Groq) without touching the chain."""

    run_inline = True   # cheap work → run in the caller's thread, no executor hop

    def __init__(self) -> None:
        self._runs: dict[UUID, tuple[str, float]] = {}

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id: UUID,
                            metadata: dict | None = None, **kwargs: Any) -> None:
        model = (metadata or {}).get("ls_model_name", "unknown")   # LangChain fills ls_model_name
        self._runs[run_id] = (model, time.perf_counter())

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        model, start = self._runs.pop(run_id, ("unknown", time.perf_counter()))
        LLM_CALLS.labels(model, "success").inc()
        LLM_LATENCY.labels(model).observe(time.perf_counter() - start)
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
            LLM_TOKENS.labels(model).inc(usage.get("total_tokens", 0))
        except (AttributeError, IndexError):
            pass   # some providers don't report usage; never fail a request over metrics

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        model, _ = self._runs.pop(run_id, ("unknown", 0.0))
        LLM_CALLS.labels(model, "error").inc()


llm_metrics_callback = LLMMetricsCallback()   # one shared instance; Prometheus counters are thread-safe