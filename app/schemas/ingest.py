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