import time
from functools import lru_cache

from fastapi import Request, Response
from limits import RateLimitItem, parse
from limits.storage import storage_from_string
from limits.strategies import MovingWindowRateLimiter

from app.core.config import get_settings
from app.core.exceptions import RateLimitExceededError


@lru_cache
def _limiter() -> MovingWindowRateLimiter:
    # Storage chosen by config: memory:// (one process) or redis:// (shared by all instances)
    return MovingWindowRateLimiter(storage_from_string(get_settings().rate_limit_storage))


class RateLimit:
    """FastAPI dependency. Usage: dependencies=[Depends(RateLimit("chat"))]"""

    def __init__(self, scope: str):
        self.scope = scope                      # limit read lazily from settings: rate_limit_<scope>

    @property
    def item(self) -> RateLimitItem:
        return parse(getattr(get_settings(), f"rate_limit_{self.scope}"))

    async def __call__(self, request: Request, response: Response) -> None:
        item = self.item
        client = request.client.host if request.client else "unknown"
        limiter = _limiter()

        allowed = limiter.hit(item, self.scope, client)      # count this request, check the window
        stats = limiter.get_window_stats(item, self.scope, client)

        # On success: tell the client its budget so it can slow down BEFORE being blocked
        response.headers["X-RateLimit-Limit"] = str(item.amount)
        response.headers["X-RateLimit-Remaining"] = str(stats.remaining)

        if not allowed:
            retry_after = max(1, int(stats.reset_time - time.time()))
            raise RateLimitExceededError(
                "Too many requests, please slow down",
                details={"client": client, "scope": self.scope},     # logs only
                headers={"Retry-After": str(retry_after)},           # HTTP-standard back-off hint
            )