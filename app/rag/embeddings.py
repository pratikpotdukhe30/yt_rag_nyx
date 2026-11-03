from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embeddings() -> Embeddings:
    settings = get_settings()
    logger.info(
        "Loading embedding model",
        extra={"extra_fields": {"model": settings.embedding_model, "device": settings.embedding_device}},
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        # Used by embed_documents() → chunks at ingestion time
        encode_kwargs={
            "normalize_embeddings": True,
            "prompt": "passage: ",      # e5 training convention for documents
            "batch_size": 32,
        },
        # Used by embed_query() → user questions at chat time
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": "query: ",        # e5 training convention for questions
        },
    )