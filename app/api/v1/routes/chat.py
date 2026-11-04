from dataclasses import asdict

from fastapi import APIRouter

from app.api.deps import ChatServiceDep
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import error_responses

router = APIRouter(prefix="/chat", tags=["chat"])



@router.post("", response_model=ChatResponse, responses=error_responses(409, 422, 429, 502, 503))
async def chat(req: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    # async: waiting on the LLM frees the event loop to serve other users
    result = await service.ask(req.video_id, req.question, req.language)
    return ChatResponse.model_validate(asdict(result))