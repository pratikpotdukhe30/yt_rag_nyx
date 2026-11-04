from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# Reusable typed constraints: validated by FastAPI BEFORE any of our code runs
VideoId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{11}$", examples=["TjPPIf-M6KE"])]
Language = Literal["it", "en"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    fields: list[dict[str, Any]] | None = None   # only present on validation errors


class ErrorResponse(BaseModel):
    error: ErrorDetail


def error_responses(*codes: int) -> dict:
    # Each route declares which errors it can return → docs = real contract
    desc = {
        404: "Transcript or video not found",
        409: "Video not indexed yet: call /v1/ingest first",
        422: "Invalid input",
        429: "Rate limited",
        502: "Upstream service (LLM / YouTube) failed",
        503: "Service temporarily unavailable",
    }
    return {c: {"model": ErrorResponse, "description": desc[c]} for c in codes}