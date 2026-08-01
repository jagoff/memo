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


def test_ask_non_stream_survives_invalid_session_id(client) -> None:
    # SessionStore._path raises ValueError on an invalid session id — the
    # stream path already guards append_turn against it; /api/ask must too.
    resp = client.post("/api/ask", json={"q": "hola", "chat_session_id": "bad id!"})
    assert resp.status_code == 200


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


def _make_save_spy_memory(tmp_path):
    """`_FakeMemory` subclass that records `save()` kwargs for the capture endpoint."""
    from types import SimpleNamespace

    from tests.test_chat_pipeline import _FakeMemory

    class _SaveSpyMemory(_FakeMemory):
        def __init__(self, tmp_path):
            super().__init__(tmp_path)
            self.save_calls: list[dict] = []

        def save(self, **kwargs):
            self.save_calls.append(kwargs)
            return SimpleNamespace(id="deadbeef1234")

    return _SaveSpyMemory(tmp_path)


def test_insight_capture_valid_candidate_saves_and_logs(tmp_path) -> None:
    memory = _make_save_spy_memory(tmp_path)
    app = build_app(memory)
    client = TestClient(app)

    candidate = {
        "title": "Acordamos migrar el backend",
        "body": "Cuerpo de la memoria capturada desde el chat.",
        "tags": ["decision"],
        "score": 95,
        "suggested_type": "decision",
        "chat_session_id": "s1",
    }
    resp = client.post("/api/insight/capture", json={"candidate": candidate})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["memoria_id"] == "deadbeef1234"

    assert memory.save_calls == [
        {
            "content": candidate["body"],
            "title": candidate["title"],
            "type_": "decision",
            "tags": ["decision", "chat-capture"],
        }
    ]

    captures_path = tmp_path / "chat" / "insights" / "captures.jsonl"
    lines = captures_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["memoria_id"] == "deadbeef1234"
    assert logged["title"] == candidate["title"]
    assert logged["score"] == 95
    assert logged["chat_session_id"] == "s1"
    assert "captured_at" in logged


def test_insight_capture_empty_title_or_body_returns_400(tmp_path) -> None:
    memory = _make_save_spy_memory(tmp_path)
    app = build_app(memory)
    client = TestClient(app)

    resp = client.post("/api/insight/capture", json={"candidate": {"title": "", "body": ""}})
    assert resp.status_code == 400
    assert "error" in resp.json()
    assert memory.save_calls == []


def test_insight_capture_low_score_adds_uncertain_tag(tmp_path) -> None:
    memory = _make_save_spy_memory(tmp_path)
    app = build_app(memory)
    client = TestClient(app)

    candidate = {
        "title": "Nota de baja confianza",
        "body": "Cuerpo suficientemente largo para pasar la validacion basica.",
        "tags": [],
        "score": 55,
        "suggested_type": "note",
    }
    resp = client.post("/api/insight/capture", json={"candidate": candidate})
    assert resp.status_code == 200
    assert memory.save_calls[0]["tags"] == ["chat-capture", "_uncertain"]


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


def test_ask_normalizes_ui_history_before_rewrite(tmp_path) -> None:
    # web-chat/src/types.ts sends history turns as {role, text} (types.ts:54-57)
    # but rewrite._history_topic reads turn.get("content") — without
    # normalizing text->content the follow-up rewrite silently no-ops and
    # retrieval runs on the literal follow-up text, never the resolved topic.
    from tests.test_chat_pipeline import _FakeMemory

    class _SpyMemory(_FakeMemory):
        def __init__(self, tmp_path):
            super().__init__(tmp_path)
            self.queries: list[str] = []

        def search(self, query, *, limit=None, mode="hybrid", **kw):
            self.queries.append(query)
            return super().search(query, limit=limit, mode=mode, **kw)

    memory = _SpyMemory(tmp_path)
    app = build_app(memory)
    client = TestClient(app)

    history = [
        {"role": "user", "text": "qué sabés del proyecto memo daemon"},
        {"role": "assistant", "text": "Memo daemon es ..."},
    ]
    resp = client.post("/api/ask", json={"q": "resumime eso", "history": history})
    assert resp.status_code == 200

    assert memory.queries, "search was never called"
    assert "memo" in memory.queries[0] and "daemon" in memory.queries[0]


def test_ask_stream_normalizes_ui_history_before_rewrite(tmp_path) -> None:
    from tests.test_chat_pipeline import _FakeMemory

    class _SpyMemory(_FakeMemory):
        def __init__(self, tmp_path):
            super().__init__(tmp_path)
            self.queries: list[str] = []

        def search(self, query, *, limit=None, mode="hybrid", **kw):
            self.queries.append(query)
            return super().search(query, limit=limit, mode=mode, **kw)

    memory = _SpyMemory(tmp_path)
    app = build_app(memory)
    client = TestClient(app)

    history = [
        {"role": "user", "text": "qué sabés del proyecto memo daemon"},
        {"role": "assistant", "text": "Memo daemon es ..."},
    ]
    with client.stream(
        "POST", "/api/ask/stream", json={"q": "resumime eso", "history": history}
    ) as resp:
        assert resp.status_code == 200
        "".join(chunk for chunk in resp.iter_text())

    assert memory.queries, "search was never called"
    assert "memo" in memory.queries[0] and "daemon" in memory.queries[0]


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
