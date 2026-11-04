import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.stores import InMemoryByteStore

from app.cache.cache import AnswerCache
from app.core.exceptions import GuardrailViolationError, VideoNotIndexedError
from app.guardrails.output_guard import OutputGuard
from app.services import chat_service
from app.services.chat_service import ChatService

DOC = Document(page_content="x", metadata={"start": 3.6, "end": 55.0})


class FakeStore:
    def __init__(self, indexed: bool = True):
        self.indexed = indexed

    def has_video(self, video_id: str, language: str | None = None) -> bool:
        return self.indexed


class FakeGuard:
    def __init__(self, block: bool = False):
        self.block, self.calls = block, 0

    async def check(self, question: str) -> None:
        self.calls += 1
        if self.block:
            raise GuardrailViolationError("blocked")


@pytest.fixture
def fake_chain(monkeypatch):
    """Replace the real RAG chain with a RunnableLambda: a REAL Runnable, fake content."""
    state = {"answer": "He says yes [00:03]", "docs": [DOC], "calls": 0}

    def run(question: str) -> dict:
        state["calls"] += 1
        return {"question": question, "docs": state["docs"], "answer": state["answer"]}

    monkeypatch.setattr(chat_service, "build_rag_chain", lambda video_id: RunnableLambda(run))
    return state


def make_service(store=None, guard=None) -> ChatService:
    return ChatService(
        store or FakeStore(), guard or FakeGuard(), OutputGuard(),
        AnswerCache(InMemoryByteStore(), 60, "test"),
    )


async def test_unindexed_video_raises(fake_chain):
    with pytest.raises(VideoNotIndexedError):
        await make_service(store=FakeStore(indexed=False)).ask("vid", "q")


async def test_answer_has_sources_with_deep_links(fake_chain):
    r = await make_service().ask("abcdefghijk", "q", "en")
    assert r.covered and r.sources[0].url == "https://youtu.be/abcdefghijk?t=3"


async def test_cache_hit_skips_guard_and_chain(fake_chain):
    guard = FakeGuard()
    svc = make_service(guard=guard)
    await svc.ask("vid", "Same question?", "en")
    await svc.ask("vid", "same question", "en")          # normalized → cache hit
    assert guard.calls == 1 and fake_chain["calls"] == 1


async def test_no_relevant_docs_returns_not_covered(fake_chain):
    fake_chain["answer"], fake_chain["docs"] = None, []
    r = await make_service().ask("vid", "carbonara?", "it")
    assert not r.covered and r.answer.startswith("Questo video")


async def test_ungrounded_answer_is_replaced(fake_chain):
    fake_chain["answer"] = "Invented fact [59:00]"
    r = await make_service().ask("vid", "q", "en")
    assert not r.covered


async def test_blocked_question_never_reaches_chain(fake_chain):
    with pytest.raises(GuardrailViolationError):
        await make_service(guard=FakeGuard(block=True)).ask("vid", "ignore instructions", "en")
    assert fake_chain["calls"] == 0