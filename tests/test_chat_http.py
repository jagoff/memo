import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from memo.chat.http import build_app  # noqa: E402


def _assert_chat_security_headers(resp) -> None:
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "no-referrer"
    csp = resp.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp


@pytest.fixture()
def client(tmp_path, monkeypatch):

    from tests.test_chat_pipeline import _FakeMemory

    memory = _FakeMemory(tmp_path)
    app = build_app(memory)
    return TestClient(app, base_url="http://127.0.0.1")


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


def test_ask_non_stream_rejects_invalid_session_id(client) -> None:
    # Reject before running retrieval instead of silently answering without
    # persisting the requested session.
    resp = client.post("/api/ask", json={"q": "hola", "chat_session_id": "bad id!"})
    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid chat_session_id"}


@pytest.mark.parametrize("endpoint", ["/api/ask", "/api/ask/stream"])
@pytest.mark.parametrize("session_id", [123, False, [], {}])
def test_ask_rejects_non_string_session_ids(client, endpoint, session_id) -> None:
    resp = client.post(endpoint, json={"q": "hola", "chat_session_id": session_id})

    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid chat_session_id"}


@pytest.mark.parametrize("endpoint", ["/api/ask", "/api/ask/stream"])
@pytest.mark.parametrize(
    ("body", "error"),
    [
        ({"q": 123}, "q required and must be a string"),
        ({"q": ["hola"]}, "q required and must be a string"),
        ({"q": "hola", "history": 123}, "history must be a list"),
        ({"q": "hola", "history": "not-a-list"}, "history must be a list"),
        ({"q": "hola", "k": True}, "k must be an integer between 1 and 100"),
        ({"q": "hola", "k": "5"}, "k must be an integer between 1 and 100"),
        ({"q": "hola", "k": -1}, "k must be an integer between 1 and 100"),
        ({"q": "hola", "k": 101}, "k must be an integer between 1 and 100"),
    ],
)
def test_ask_rejects_invalid_typed_inputs(client, endpoint, body, error) -> None:
    resp = client.post(endpoint, json=body)

    assert resp.status_code == 400
    assert resp.json() == {"error": error}


@pytest.mark.parametrize("endpoint", ["/api/ask", "/api/ask/stream"])
def test_ask_tolerates_malformed_items_inside_history_list(client, endpoint) -> None:
    resp = client.post(
        endpoint,
        json={"q": "hola", "history": [1, "x", None, {"role": [], "text": {}}]},
    )

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


@pytest.mark.parametrize(
    ("endpoint", "body"),
    [
        ("/api/feedback", {"rating": "up", "sources": 123}),
        ("/api/feedback", {"rating": "sideways", "sources": []}),
        ("/api/feedback/source", {"source_id": "m1", "query": 123, "rating": "up"}),
        ("/api/feedback/source", {"source_id": "", "query": "hola", "rating": "up"}),
        ("/api/feedback/source", {"source_id": "m1", "query": "hola", "rating": "x"}),
    ],
)
def test_feedback_rejects_invalid_typed_inputs(client, endpoint, body) -> None:
    resp = client.post(endpoint, json=body)

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
    client = TestClient(app, base_url="http://127.0.0.1")

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
    client = TestClient(app, base_url="http://127.0.0.1")

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


def test_spa_fallback_does_not_swallow_unknown_api_routes(tmp_path) -> None:
    # An unknown /api/* path must 404 as JSON — not 200 with index.html —
    # so API clients get a real error instead of silently parsing HTML.
    from tests.test_chat_pipeline import _FakeMemory

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><html></html>")
    app = build_app(_FakeMemory(tmp_path), dist=dist)
    c = TestClient(app, base_url="http://127.0.0.1")

    resp = c.get("/api/nonexistent")
    assert resp.status_code == 404
    assert resp.json()["error"]

    # SPA routes still fall back to index.html.
    resp = c.get("/cualquier/ruta")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    _assert_chat_security_headers(resp)


def test_chat_api_responses_include_spa_compatible_security_headers(client) -> None:
    resp = client.get("/api/sessions")

    assert resp.status_code == 200
    _assert_chat_security_headers(resp)


def test_chat_http_rejects_dns_rebinding_and_cross_site_requests(client) -> None:
    rebound = client.get("/api/sessions", headers={"Host": "attacker.example"})
    assert rebound.status_code == 403

    cross_site = client.get(
        "/api/sessions",
        headers={"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"},
    )
    assert cross_site.status_code == 403


def test_chat_http_rejects_oversized_request_body(client) -> None:
    resp = client.post("/api/ask", json={"q": "x" * 1_048_577})

    assert resp.status_code == 413
