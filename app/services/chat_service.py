from dataclasses import dataclass, field
from functools import lru_cache

from app.core.exceptions import VideoNotIndexedError
from app.core.logging import get_logger
from app.rag.chain import build_rag_chain
from app.rag.prompts import PROMPT_VERSION
from app.rag.vectorstore import VectorStoreRepository, get_vector_store
from app.guardrails.input_guard import InputGuard, get_input_guard
from app.guardrails.output_guard import OutputGuard

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
    def __init__(self, store: VectorStoreRepository, input_guard: InputGuard, output_guard: OutputGuard):
        self._store = store
        self._input_guard = input_guard
        self._output_guard = output_guard

    async def ask(self, video_id: str, question: str, language: str = "en") -> ChatResult:
        not_covered = ChatResult(answer=NOT_COVERED.get(language, NOT_COVERED["en"]), covered=False)

        # Cheapest checks first: local lookup → network classifier → retrieval + LLM
        if not self._store.has_video(video_id):
            raise VideoNotIndexedError("Video not indexed yet", details={"video_id": video_id})

        await self._input_guard.check(question)

        result = await build_rag_chain(video_id).ainvoke(question)
        if result["answer"] is None:
            return not_covered

        if not self._output_guard.is_grounded(result["answer"], result["docs"]):
            return not_covered               # abstain rather than risk a hallucination

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
    return ChatService(get_vector_store(), get_input_guard(), OutputGuard())