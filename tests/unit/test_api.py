import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import VideoNotIndexedError
from app.main import app
from app.services.chat_service import ChatResult, Source, get_chat_service


class FakeChatService:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = 0                      # track whether business logic actually ran

    async def ask(self, video_id, question, language="en"):
        self.calls += 1
        if self.error:
            raise self.error
        return ChatResult(answer="ok [00:03]", covered=True,
                          sources=[Source(start=3.6, end=55.0, url="https://youtu.be/x?t=3")])

@pytest.fixture
def client():
    # No `with` → lifespan does NOT run → no model loading, no LLM pings
    yield TestClient(app)
    app.dependency_overrides.clear()     # never leak overrides into other tests


BODY = {"video_id": "TjPPIf-M6KE", "question": "hi", "language": "en"}


def test_chat_success_contract(client):
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    r = client.post("/v1/chat", json=BODY)
    assert r.status_code == 200
    assert r.json()["covered"] is True
    assert "x-request-id" in r.headers                     # middleware ran


def test_domain_error_maps_to_409_with_our_shape(client):
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService(VideoNotIndexedError("not indexed"))
    r = client.post("/v1/chat", json=BODY)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "video_not_indexed"
    assert r.json()["error"]["request_id"] == r.headers["x-request-id"]


def test_invalid_video_id_rejected_before_service(client):
    fake = FakeChatService()
    app.dependency_overrides[get_chat_service] = lambda: fake
    r = client.post("/v1/chat", json={**BODY, "video_id": "abc"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"
    assert fake.calls == 0


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}