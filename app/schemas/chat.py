from pydantic import BaseModel, Field

from app.schemas.common import Language, VideoId, error_responses


class ChatRequest(BaseModel):
    video_id: VideoId
    question: str = Field(min_length=1, max_length=500)  # caps cost + abuse at the boundary
    language: Language = "en"


class SourceOut(BaseModel):
    start: float
    end: float
    url: str


class ChatResponse(BaseModel):
    answer: str
    covered: bool
    sources: list[SourceOut]