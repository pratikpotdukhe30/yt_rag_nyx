from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embeddings import get_embeddings

logger = get_logger(__name__)

#COLLECTION_NAME = "youtube_transcripts"


class VectorStoreRepository:
    """The ONLY place that knows we use Chroma. Everything else talks to this class."""

    def __init__(self, store: Chroma):
        self._store = store

    def has_video(self, video_id: str, language: str) -> bool:
        result = self._store.get(
            where={"$and": [{"video_id": video_id}, {"language": language}]},
            limit=1,
            include=[],
        )
        return len(result["ids"]) > 0

    def add(self, docs: list[Document]) -> None:
        if not docs:
            return
        # Explicit ids = deterministic chunk ids → upsert, never duplicates
        self._store.add_documents(docs, ids=[d.id for d in docs])
        logger.info(
            "Chunks stored",
            extra={"extra_fields": {"video_id": docs[0].metadata["video_id"], "count": len(docs)}},
        )

    def delete_video(self, video_id: str) -> None:
        # Needed for re-indexing or a "forget this video" admin action
        self._store.delete(where={"video_id": video_id})

    def as_retriever(self, video_id: str) -> VectorStoreRetriever:
        settings = get_settings()
        # Filter by video only, not language: an Italian question must find English chunks.
        return self._store.as_retriever(
            search_type="similarity_score_threshold",   # drops weak matches → enables "not covered" guardrail
            search_kwargs={
                "k": settings.top_k,
                "score_threshold": settings.similarity_threshold,
                "filter": {"video_id": video_id},
            },
        )


@lru_cache  # one client per process; Chroma connections are expensive to create
def get_vector_store() -> VectorStoreRepository:
    settings = get_settings()
    # "BAAI/bge-m3" → "yt_BAAI_bge-m3": each embedding model gets its own index, never mixed
    collection_name = "yt_" + settings.embedding_model.replace("/", "_")
    store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
        collection_metadata={"hnsw:space": "cosine"},   # cosine distance → score = cosine similarity
    )
    return VectorStoreRepository(store)