import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from app.api.errors import error_body
from app.core.logging import get_logger, request_id_ctx
from app.observability.metrics import HTTP_LATENCY
logger = get_logger(__name__)

# Infra endpoints hit every few seconds by Docker/Prometheus → never log or measure them
_SILENT_PATHS = ("/health", "/ready", "/metrics")
async def request_context_middleware(request: Request, call_next):
    # Reuse the caller's ID if present (lets you trace across services), else create one
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_ctx.set(request_id)   # every log line in this request now carries it
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        # Last safety net: a real bug. Full trace in logs, nothing internal leaked to the client.
        logger.exception("Unhandled error")
        response = JSONResponse(status_code=500, content=error_body("internal_error", "Internal server error"))

    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    response.headers["X-Request-ID"] = request_id

    if not request.url.path.startswith(_SILENT_PATHS):
        endpoint = request.scope.get("endpoint")
        handler = endpoint.__name__ if endpoint else "unmatched"
        HTTP_LATENCY.labels(request.method, handler, str(response.status_code)).observe(duration_ms / 1000)
        logger.info(
            "Request completed",
            extra={"extra_fields": {
                "method": request.method, "path": request.url.path,
                "status": response.status_code, "duration_ms": duration_ms,
            }},
        )

    request_id_ctx.reset(token)
    return response