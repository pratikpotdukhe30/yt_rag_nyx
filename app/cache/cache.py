import hashlib
import json
import re
import time
from functools import lru_cache
from typing import Any

from langchain_core.stores import BaseStore, InMemoryByteStore

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class AnswerCache:
    def __init__(self, store: BaseStore[str, bytes], ttl_seconds: int, namespace: str, enabled: bool = True):
        self._store = store            # InMemoryByteStore now, RedisStore later to be done
        self._ttl = ttl_seconds
        self._namespace = namespace    # prompt/model versions → changing them invalidates old keys
        self._enabled = enabled

    @staticmethod
    def _normalize(question: str) -> str:
        q = re.sub(r"\s+", " ", question.lower().strip())
        return q.rstrip("?!. ")

    def make_key(self, video_id: str, question: str, language: str) -> str:
        raw = "|".join([self._namespace, video_id, language, self._normalize(question)])
        # Hash → fixed-length key, no weird characters, no user text stored in the key itself
        return "answer:" + hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        try:
            raw = self._store.mget([key])[0]
            if raw is None:
                return None
            entry = json.loads(raw)
            if entry["expires_at"] < time.time():
                self._store.mdelete([key])     # lazy expiry: cleaned when touched
                return None
            return entry["value"]
        except Exception as e:
            # Cache is an optimization: a broken cache = a miss, never a failed request
            logger.warning("Cache read failed", extra={"extra_fields": {"error": type(e).__name__}})
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            entry = {"expires_at": time.time() + self._ttl, "value": value}
            self._store.mset([(key, json.dumps(entry).encode())])
        except Exception as e:
            logger.warning("Cache write failed", extra={"extra_fields": {"error": type(e).__name__}})


@lru_cache
def get_answer_cache() -> AnswerCache:
    s = get_settings()
    from app.rag.prompts import PROMPT_VERSION   # wiring only; the cache class itself knows nothing about RAG
    namespace = f"{PROMPT_VERSION}:{s.openai_model}:{s.embedding_model}"
    return AnswerCache(InMemoryByteStore(), s.cache_ttl_seconds, namespace, s.cache_enabled)