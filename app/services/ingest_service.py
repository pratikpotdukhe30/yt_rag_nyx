from dataclasses import dataclass
from functools import lru_cache

from app.core.logging import get_logger
from app.ingestion.chunker import TranscriptChunker
from app.ingestion.transcript import TranscriptFetcher
from app.rag.vectorstore import VectorStoreRepository, get_vector_store

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestResult:
    video_id: str
    already_indexed: bool
    chunks: int = 0
    language: str | None = None


class IngestService:
    # Constructor injection: tests pass fakes, prod passes real ones
    def __init__(self, fetcher: TranscriptFetcher, chunker: TranscriptChunker, store: VectorStoreRepository):
        self._fetcher = fetcher
        self._chunker = chunker
        self._store = store

    def ingest(self, video_id: str, language: str | None = None) -> IngestResult:
        # Idempotency: cheap check first → skip fetch + chunk + embed entirely
        if self._store.has_video(video_id):
            logger.info("Video already indexed", extra={"extra_fields": {"video_id": video_id}})
            return IngestResult(video_id=video_id, already_indexed=True)

        transcript = self._fetcher.fetch(video_id, language)   # raises domain errors on failure
        docs = self._chunker.split(transcript)
        self._store.add(docs)                                    # upsert → safe under concurrent ingests

        return IngestResult(
            video_id=video_id, already_indexed=False, chunks=len(docs), language=transcript.language
        )


@lru_cache  # wiring lives here: one place decides which concrete classes are used
def get_ingest_service() -> IngestService:
    return IngestService(TranscriptFetcher(), TranscriptChunker(), get_vector_store())