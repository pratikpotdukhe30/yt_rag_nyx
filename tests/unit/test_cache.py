from langchain_core.stores import InMemoryByteStore

from app.cache.cache import AnswerCache


def make_cache(**kwargs) -> AnswerCache:
    return AnswerCache(InMemoryByteStore(), ttl_seconds=60, namespace="v1", **kwargs)


def test_normalized_questions_share_a_key():
    c = make_cache()
    assert c.make_key("vid", "Com era all inizio?", "it") == c.make_key("vid", "  com era   ALL inizio ", "it")


def test_namespace_change_invalidates_keys():
    a = AnswerCache(InMemoryByteStore(), 60, namespace="rag-v1:gpt-4o-mini")
    b = AnswerCache(InMemoryByteStore(), 60, namespace="rag-v2:gpt-4o-mini")
    assert a.make_key("vid", "q", "en") != b.make_key("vid", "q", "en")


def test_roundtrip_and_ttl_expiry(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("app.cache.cache.time.time", lambda: now[0])   # control the clock
    c = make_cache()
    c.set("k", {"answer": "x"})
    assert c.get("k") == {"answer": "x"}
    now[0] += 61                                                        # jump past the TTL
    assert c.get("k") is None


def test_disabled_cache_never_returns():
    c = make_cache(enabled=False)
    c.set("k", {"a": 1})
    assert c.get("k") is None


def test_broken_store_is_a_miss_not_a_crash():
    class BrokenStore(InMemoryByteStore):
        def mget(self, keys):
            raise ConnectionError("redis down")

    c = AnswerCache(BrokenStore(), 60, "v1")
    assert c.get("k") is None      # cache failure must never fail the request