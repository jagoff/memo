import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from memo.chat.http import build_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):

    from tests.test_chat_pipeline import _FakeMemory

    memory = _FakeMemory(tmp_path)
    app = build_app(memory)
    return TestClient(app)


def test_ask_stream_sse(client) -> None:
    with client.stream("POST", "/api/ask/stream", json={"q": "hola", "history": []}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in resp.iter_text())
    frames = [json.loads(line[5:]) for line in body.split("\n\n") if line.startswith("data:")]
    kinds = [f["type"] for f in frames]
    assert "context" in kinds and "done" in kinds


def test_ask_non_stream(client) -> None:
    resp = client.post("/api/ask", json={"q": "hola"})
    assert resp.status_code == 200
    assert resp.json()["type"] == "done"


def test_feedback_source_roundtrip(client) -> None:
    resp = client.post(
        "/api/feedback/source", json={"source_id": "m1", "query": "hola", "rating": "up"}
    )
    assert resp.status_code == 200 and resp.json()["ok"] is True


def test_deferred_endpoints_501(client) -> None:
    assert client.post("/api/memory/delete", json={}).status_code == 501
    assert client.post("/api/insight/capture", json={}).status_code == 501


def test_sessions_endpoints(client) -> None:
    client.post("/api/ask", json={"q": "hola", "chat_session_id": "s1"})
    sessions = client.get("/api/sessions").json()["sessions"]
    assert any(s["id"] == "s1" for s in sessions)
    assert client.post("/api/sessions/delete", json={"session_id": "s1"}).json()["ok"] is True
