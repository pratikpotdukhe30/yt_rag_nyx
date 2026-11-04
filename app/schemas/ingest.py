from pydantic import BaseModel

from app.schemas.common import Language, VideoId


class IngestRequest(BaseModel):
    video_id: VideoId
    language: Language | None = None


class IngestResponse(BaseModel):
    video_id: str
    already_indexed: bool
    chunks: int
    language: str | None

from typing import Literal


class IngestJobResponse(BaseModel):
    job_id: str
    video_id: str
    status: Literal["pending", "running", "completed", "failed"]
    result: IngestResponse | None = None
    error: dict[str, str] | None = None