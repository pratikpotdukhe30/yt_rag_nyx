from dataclasses import dataclass, field
from functools import lru_cache

from app.core.exceptions import VideoNotIndexedError
from app.core.logging import get_logger
from app.rag.chain import build_rag_chain
from app.rag.prompts import PROMPT_VERSION
from app.rag.vectorstore import VectorStoreRepository, get_vector_store

logger = get_logger(__name__)

# User-facing text lives in the service, per language. The chain only returns None.
NOT_COVERED = {
    "it": "Questo video non sembra trattare questo argomento.",
    "en": "This video doesn't seem to cover that topic.",
}


@dataclass(frozen=True)
class Source:
    start: float
    end: float
    url: str          # deep link straight to that moment in the video


@dataclass(frozen=True)
class ChatResult:
    answer: str
    covered: bool
    sources: list[Source] = field(default_factory=list)


class ChatService:
    def __init__(self, store: VectorStoreRepository):
        self._store = store

    async def ask(self, video_id: str, question: str, language: str = "en") -> ChatResult:
        # Pipeline order (slots marked for upcoming steps):
        # [input guardrail] → [cache lookup] → retrieve + generate → [output guardrail] → [cache store]

        if not self._store.has_video(video_id):
            raise VideoNotIndexedError("Video not indexed yet", details={"video_id": video_id})

        chain = build_rag_chain(video_id)          # cheap: just wires Runnables, no model loading
        result = await chain.ainvoke(question)     # async → the server keeps serving others while waiting

        if result["answer"] is None:
            return ChatResult(answer=NOT_COVERED.get(language, NOT_COVERED["en"]), covered=False)

        sources = [
            Source(
                start=d.metadata["start"],
                end=d.metadata["end"],
                url=f"https://youtu.be/{video_id}?t={int(d.metadata['start'])}",
            )
            for d in result["docs"]
        ]
        logger.info(
            "Question answered",
            extra={"extra_fields": {"video_id": video_id, "sources": len(sources), "prompt": PROMPT_VERSION}},
        )
        return ChatResult(answer=result["answer"], covered=True, sources=sources)


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(get_vector_store())