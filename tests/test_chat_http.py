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


def test_ask_malformed_json_returns_400(client) -> None:
    resp = client.post(
        "/api/ask", content=b"not json", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_feedback_source_roundtrip(client) -> None:
    resp = client.post(
        "/api/feedback/source", json={"source_id": "m1", "query": "hola", "rating": "up"}
    )
    assert resp.status_code == 200 and resp.json()["ok"] is True


def test_feedback_source_malformed_json_returns_400(client) -> None:
    resp = client.post(
        "/api/feedback/source", content=b"not json", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_deferred_endpoints_501(client) -> None:
    assert client.post("/api/memory/delete", json={}).status_code == 501
    assert client.post("/api/insight/capture", json={}).status_code == 501


def test_sessions_endpoints(client) -> None:
    client.post("/api/ask", json={"q": "hola", "chat_session_id": "s1"})

    # web-chat/src/types.ts SessionListItem expects {session_id, turn_count,
    # label, first_ts?, last_ts?} — not the internal {id, updated, turns,
    # first_query} shape.
    sessions = client.get("/api/sessions").json()["sessions"]
    row = next(s for s in sessions if s["session_id"] == "s1")
    assert row["turn_count"] == 2
    assert row["label"] == "hola"

    # web-chat/src/types.ts SessionHistory expects {session_id, turns:
    # [{role, text, at?}]} — not {id, turns: [{role, text, ts}]}.
    history = client.get("/api/sessions/s1").json()
    assert history["session_id"] == "s1"
    assert history["turns"][0]["role"] == "user"
    assert history["turns"][0]["text"] == "hola"
    assert "at" in history["turns"][0]

    assert client.post("/api/sessions/delete", json={"session_id": "s1"}).json()["ok"] is True


def test_suggestions_returns_chip_objects(client) -> None:
    client.post("/api/ask", json={"q": "hola de nuevo", "chat_session_id": "s2"})

    # web-chat/src/api.ts fetchSuggestions() reads body.chips (SuggestionChip[]
    # = {label, query}), not body.suggestions (a list of raw strings).
    body = client.get("/api/suggestions").json()
    assert "chips" in body
    chip = next(c for c in body["chips"] if c["query"] == "hola de nuevo")
    assert chip["label"] == "hola de nuevo"

    client.post("/api/sessions/delete", json={"session_id": "s2"})


def test_ask_stream_generates_session_id_when_missing(client) -> None:
    # App.tsx never generates chat_session_id client-side — it only ever
    # adopts one FROM the context/done events (App.tsx:276-277, 341-342).
    # Without server-side generation, sessions never persist in real UI use.
    with client.stream("POST", "/api/ask/stream", json={"q": "hola sin session"}) as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())
    frames = [json.loads(line[5:]) for line in body.split("\n\n") if line.startswith("data:")]
    context = next(f for f in frames if f["type"] == "context")
    done = next(f for f in frames if f["type"] == "done")
    assert context.get("chat_session_id")
    assert context["chat_session_id"] == done["chat_session_id"]

    generated_id = done["chat_session_id"]
    sessions = client.get("/api/sessions").json()["sessions"]
    row = next(s for s in sessions if s["session_id"] == generated_id)
    assert row["turn_count"] == 2

    client.post("/api/sessions/delete", json={"session_id": generated_id})


def test_ask_stream_persists_under_given_session_id(client) -> None:
    with client.stream(
        "POST",
        "/api/ask/stream",
        json={"q": "hola con session", "chat_session_id": "given-1"},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())
    frames = [json.loads(line[5:]) for line in body.split("\n\n") if line.startswith("data:")]
    context = next(f for f in frames if f["type"] == "context")
    done = next(f for f in frames if f["type"] == "done")
    assert context.get("chat_session_id") == "given-1"
    assert done.get("chat_session_id") == "given-1"

    history = client.get("/api/sessions/given-1").json()
    assert history["session_id"] == "given-1"
    assert len(history["turns"]) == 2

    client.post("/api/sessions/delete", json={"session_id": "given-1"})
