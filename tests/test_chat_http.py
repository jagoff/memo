import asyncio
import json
import subprocess
import sys
import threading

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from memo.chat.http import build_app  # noqa: E402


def _local_test_client(app) -> TestClient:
    return TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )


def _post_json_document(client, endpoint: str, body: dict) -> object:
    payload = json.dumps(body, ensure_ascii=True).encode()
    return client.post(endpoint, content=payload, headers={"Content-Type": "application/json"})


def _assert_chat_security_headers(resp) -> None:
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "no-referrer"
    assert resp.headers["content-security-policy"] == (
        "default-src 'self'; script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; connect-src 'self'; "
        "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
    )


@pytest.fixture()
def client(tmp_path, monkeypatch):

    from tests.test_chat_pipeline import _FakeMemory

    memory = _FakeMemory(tmp_path)
    app = build_app(memory)
    return _local_test_client(app)


def test_ask_stream_sse(client) -> None:
    with client.stream("POST", "/api/ask/stream", json={"q": "hola", "history": []}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        _assert_chat_security_headers(resp)
        body = "".join(chunk for chunk in resp.iter_text())
    frames = [json.loads(line[5:]) for line in body.split("\n\n") if line.startswith("data:")]
    kinds = [f["type"] for f in frames]
    assert "context" in kinds and "done" in kinds


def test_ask_stream_escapes_model_surrogate_and_releases_capacity(tmp_path, monkeypatch) -> None:
    import memo.chat.http as chat_http
    from tests.test_chat_pipeline import _FakeMemory

    def surrogate_stream(*args, **kwargs):
        yield {"type": "done", "answer": "bad-\ud800-answer", "sources": []}

    monkeypatch.setattr(chat_http, "_MAX_HEAVY_IN_FLIGHT", 1)
    monkeypatch.setattr("memo.chat.pipeline.chat_stream", surrogate_stream)
    app = build_app(_FakeMemory(tmp_path))
    local_client = _local_test_client(app)

    first = local_client.post("/api/ask/stream", json={"q": "first"})
    second = local_client.post("/api/ask/stream", json={"q": "second"})
    non_stream = local_client.post("/api/ask", json={"q": "third"})

    assert first.status_code == 200
    assert "\\ud800" in first.text
    assert second.status_code == 200
    assert non_stream.status_code == 200
    assert "\\ud800" in non_stream.text
    frame = json.loads(first.text.removeprefix("data: ").strip())
    session_id = frame["chat_session_id"]
    # Reject the invalid generated answer before writing either half of the exchange.
    assert local_client.get(f"/api/sessions/{session_id}").json()["turns"] == []


def test_ask_non_stream(client) -> None:
    resp = client.post("/api/ask", json={"q": "hola"})
    assert resp.status_code == 200
    assert resp.json()["type"] == "done"


def test_ask_non_stream_rejects_invalid_session_id(client) -> None:
    # Reject before running retrieval instead of silently answering without
    # persisting the requested session.
    resp = client.post("/api/ask", json={"q": "hola", "chat_session_id": "bad id!"})
    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid chat request"}


@pytest.mark.parametrize("endpoint", ["/api/ask", "/api/ask/stream"])
@pytest.mark.parametrize("session_id", [123, False, [], {}])
def test_ask_rejects_non_string_session_ids(client, endpoint, session_id) -> None:
    resp = client.post(endpoint, json={"q": "hola", "chat_session_id": session_id})

    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid chat request"}


@pytest.mark.parametrize("endpoint", ["/api/ask", "/api/ask/stream"])
@pytest.mark.parametrize(
    "body",
    [
        {"q": 123},
        {"q": ["hola"]},
        {"q": "hola", "history": 123},
        {"q": "hola", "history": "not-a-list"},
        {"q": "hola", "k": True},
        {"q": "hola", "k": "5"},
        {"q": "hola", "k": -1},
        {"q": "hola", "k": 101},
    ],
)
def test_ask_rejects_invalid_typed_inputs(client, endpoint, body) -> None:
    resp = client.post(endpoint, json=body)

    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid chat request"}


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


@pytest.mark.parametrize("endpoint", ["/api/ask", "/api/ask/stream"])
@pytest.mark.parametrize(
    "body",
    [
        {"q": "\ud800"},
        {"q": "hola", "history": [{"role": "\ud800", "text": "ok"}]},
        {"q": "hola", "history": [{"role": "user", "text": "\ud800"}]},
    ],
)
def test_ask_rejects_isolated_unicode_surrogates(client, endpoint, body) -> None:
    resp = _post_json_document(client, endpoint, body)

    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid chat request"}


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


@pytest.mark.parametrize("endpoint", ["/api/feedback", "/api/feedback/source"])
@pytest.mark.parametrize("rating", [None, True, [], {}])
def test_feedback_rejects_non_string_ratings(client, endpoint, rating) -> None:
    if endpoint.endswith("/source"):
        body = {"source_id": "m1", "query": "hola", "rating": rating}
    else:
        body = {"sources": [], "rating": rating}

    resp = client.post(endpoint, json=body)

    assert resp.status_code == 400
    assert resp.json() == {"error": "rating must be 'up' or 'down'"}


@pytest.mark.parametrize(
    ("field", "body"),
    [
        ("chat_session_id", {"rating": "up", "sources": []}),
        ("turn_id", {"rating": "up", "sources": []}),
        ("query", {"rating": "up", "sources": []}),
        ("answer", {"rating": "up", "sources": []}),
        ("correction_text", {"rating": "up", "sources": []}),
    ],
)
def test_feedback_rejects_surrogates_in_every_persisted_text_field(client, field, body) -> None:
    body[field] = "\ud800"

    resp = _post_json_document(client, "/api/feedback", body)

    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid feedback request"}


def test_feedback_rejects_surrogate_source_id(client) -> None:
    resp = _post_json_document(
        client,
        "/api/feedback",
        {"rating": "up", "sources": [{"id": "\ud800"}]},
    )

    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid feedback request"}


@pytest.mark.parametrize("field", ["query", "source_id"])
def test_source_feedback_rejects_surrogates_in_persisted_fields(client, field) -> None:
    body = {"source_id": "m1", "query": "hola", "rating": "up"}
    body[field] = "\ud800"

    resp = _post_json_document(client, "/api/feedback/source", body)

    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid source feedback request"}


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


@pytest.mark.parametrize("session_id", [None, 1, False, [], {}])
def test_delete_session_rejects_non_string_ids(client, session_id) -> None:
    resp = client.post("/api/sessions/delete", json={"session_id": session_id})

    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid session id"}


@pytest.mark.parametrize("endpoint", ["/api/sessions", "/api/suggestions"])
@pytest.mark.parametrize("limit", [0, -1, 101])
def test_list_endpoints_reject_out_of_range_limits(client, endpoint, limit) -> None:
    resp = client.get(endpoint, params={"limit": limit})

    assert resp.status_code == 400
    assert resp.json() == {"error": "limit must be between 1 and 100"}


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
    client = _local_test_client(app)

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
    client = _local_test_client(app)

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
    c = _local_test_client(app)

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


def test_chat_http_rejects_remote_peer_even_with_loopback_host(tmp_path) -> None:
    from tests.test_chat_pipeline import _FakeMemory

    app = build_app(_FakeMemory(tmp_path))
    remote_client = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("192.0.2.10", 50000),
    )

    response = remote_client.get("/api/sessions")

    assert response.status_code == 403


def test_chat_guard_rejections_do_not_exhaust_rate_limit(client) -> None:
    for _ in range(301):
        rejected = client.get(
            "/api/sessions",
            headers={"Origin": "https://attacker.example"},
        )
        assert rejected.status_code == 403

    allowed = client.get("/api/sessions")

    assert allowed.status_code == 200


def test_chat_http_rejects_oversized_request_body(client) -> None:
    resp = client.post("/api/ask", json={"q": "x" * 1_048_577})

    assert resp.status_code == 413


def test_chat_http_module_imports_without_fastapi_or_starlette() -> None:
    code = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'fastapi', 'starlette'}:
        raise ImportError(f'blocked optional dependency: {name}')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import memo.chat.http
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_ask_reads_only_recent_session_history_once(tmp_path, monkeypatch) -> None:
    from memo.chat.config import ChatConfig
    from memo.chat.sessions import SessionStore
    from tests.test_chat_pipeline import _FakeMemory

    calls: list[tuple[str, int]] = []
    original_get_recent = SessionStore.get_recent

    def spy_get_recent(self, session_id, limit=12):
        calls.append((session_id, limit))
        return original_get_recent(self, session_id, limit)

    monkeypatch.setattr(SessionStore, "get_recent", spy_get_recent)
    memory = _FakeMemory(tmp_path)
    cfg = ChatConfig.load(memory.cfg.state_dir)
    SessionStore(cfg.sessions_dir).append_turn("once", "user", "primera")
    app = build_app(memory)
    local_client = _local_test_client(app)

    response = local_client.post("/api/ask", json={"q": "seguimiento", "chat_session_id": "once"})

    assert response.status_code == 200
    assert calls == [("once", 12)]


def test_session_endpoints_skip_corrupt_jsonl_turns(tmp_path) -> None:
    from memo.chat.config import ChatConfig
    from memo.chat.sessions import SessionStore
    from tests.test_chat_pipeline import _FakeMemory

    memory = _FakeMemory(tmp_path)
    cfg = ChatConfig.load(memory.cfg.state_dir)
    store = SessionStore(cfg.sessions_dir)
    store.append_turn("legacy", "user", "valid")
    with store._path("legacy").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "user", "ts": "bad"}) + "\n")
        fh.write(json.dumps({"role": "user", "text": "\ud800", "ts": 1}) + "\n")

    app = build_app(memory)
    local_client = _local_test_client(app)
    listing = local_client.get("/api/sessions")
    history = local_client.get("/api/sessions/legacy")

    assert listing.status_code == 200
    assert listing.json()["sessions"][0]["turn_count"] == 1
    assert history.status_code == 200
    assert [turn["text"] for turn in history.json()["turns"]] == ["valid"]


@pytest.mark.asyncio
async def test_source_feedback_shares_chat_capacity_and_slots_are_reused(
    tmp_path, monkeypatch
) -> None:
    import httpx

    import memo.chat.http as chat_http
    from tests.test_chat_pipeline import _FakeMemory

    started = threading.Event()
    release = threading.Event()
    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def blocking_run(self, question, *, session_id, history, k):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            if state["active"] == 4:
                started.set()
        release.wait(timeout=5)
        with lock:
            state["active"] -= 1
        return [{"type": "done", "answer": question, "chat_session_id": session_id}]

    monkeypatch.setattr(chat_http._ChatApi, "_run", blocking_run)
    app = build_app(_FakeMemory(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as async_client:
        requests = [
            asyncio.create_task(async_client.post("/api/ask", json={"q": f"q-{index}"}))
            for index in range(4)
        ]
        assert await asyncio.to_thread(started.wait, 2), "four requests never entered chat work"

        overflow = await async_client.post(
            "/api/feedback/source",
            json={"source_id": "m1", "query": "overflow", "rating": "up"},
        )

        assert overflow.status_code == 503
        assert overflow.headers["retry-after"] == "1"
        assert state["peak"] == 4
        release.set()
        completed = await asyncio.gather(*requests)
        assert all(response.status_code == 200 for response in completed)
        recovered = await async_client.post("/api/ask", json={"q": "recovered"})
        assert recovered.status_code == 200


@pytest.mark.asyncio
async def test_heavy_slot_is_recovered_after_worker_exception(tmp_path, monkeypatch) -> None:
    import httpx

    import memo.chat.http as chat_http
    from tests.test_chat_pipeline import _FakeMemory

    monkeypatch.setattr(chat_http, "_MAX_HEAVY_IN_FLIGHT", 1)
    calls = 0

    def flaky_run(self, question, *, session_id, history, k):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return [{"type": "done", "answer": question, "chat_session_id": session_id}]

    monkeypatch.setattr(chat_http._ChatApi, "_run", flaky_run)
    app = build_app(_FakeMemory(tmp_path))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as async_client:
        failed = await async_client.post("/api/ask", json={"q": "fails"})
        recovered = await async_client.post("/api/ask", json={"q": "works"})

    assert failed.status_code == 500
    assert recovered.status_code == 200


@pytest.mark.asyncio
async def test_stream_slot_is_recovered_if_response_construction_fails(
    tmp_path, monkeypatch
) -> None:
    import httpx

    import memo.chat.http as chat_http
    from tests.test_chat_pipeline import _FakeMemory

    monkeypatch.setattr(chat_http, "_MAX_HEAVY_IN_FLIGHT", 1)
    calls = 0

    class BrokenStreamingResponse:
        def __init__(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("cannot construct stream")

    monkeypatch.setattr("fastapi.responses.StreamingResponse", BrokenStreamingResponse)
    app = build_app(_FakeMemory(tmp_path))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as async_client:
        first = await async_client.post("/api/ask/stream", json={"q": "first"})
        second = await async_client.post("/api/ask/stream", json={"q": "second"})

    assert first.status_code == 500
    assert second.status_code == 500
    assert calls == 2


@pytest.mark.asyncio
async def test_heavy_slot_is_recovered_after_request_cancellation(tmp_path, monkeypatch) -> None:
    import httpx

    import memo.chat.http as chat_http
    from tests.test_chat_pipeline import _FakeMemory

    monkeypatch.setattr(chat_http, "_MAX_HEAVY_IN_FLIGHT", 1)
    started = asyncio.Event()

    async def cancellable_run_sync(function, *args, **kwargs):
        if function.__name__ == "_run":
            started.set()
            await asyncio.sleep(30)
        return function(*args, **kwargs)

    monkeypatch.setattr("fastapi.concurrency.run_in_threadpool", cancellable_run_sync)
    app = build_app(_FakeMemory(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as async_client:
        request = asyncio.create_task(async_client.post("/api/ask", json={"q": "cancel"}))
        await asyncio.wait_for(started.wait(), timeout=2)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        monkeypatch.setattr(chat_http._ChatApi, "_run", lambda *args, **kwargs: [])
        recovered = await async_client.post("/api/ask", json={"q": "works"})

    assert recovered.status_code == 200


@pytest.mark.asyncio
async def test_same_session_asks_are_serialized_and_exchanges_stay_adjacent(
    tmp_path, monkeypatch
) -> None:
    import httpx

    from tests.test_chat_pipeline import _FakeMemory

    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()

    def ordered_stream(_memory, question, **_kwargs):
        if question == "first":
            first_started.set()
            assert release_first.wait(timeout=5)
        else:
            second_started.set()
        yield {"type": "done", "answer": f"answer-{question}", "sources": []}

    monkeypatch.setattr("memo.chat.pipeline.chat_stream", ordered_stream)
    app = build_app(_FakeMemory(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        first = asyncio.create_task(
            client.post("/api/ask", json={"q": "first", "chat_session_id": "shared"})
        )
        assert await asyncio.to_thread(first_started.wait, 2)
        second = asyncio.create_task(
            client.post("/api/ask", json={"q": "second", "chat_session_id": "shared"})
        )
        await asyncio.sleep(0.05)
        assert not second_started.is_set()

        release_first.set()
        responses = await asyncio.gather(first, second)
        history = (await client.get("/api/sessions/shared")).json()["turns"]

    assert all(response.status_code == 200 for response in responses)
    assert [(turn["role"], turn["text"]) for turn in history] == [
        ("user", "first"),
        ("assistant", "answer-first"),
        ("user", "second"),
        ("assistant", "answer-second"),
    ]


@pytest.mark.asyncio
async def test_delete_waits_for_same_session_ask_and_cannot_be_recreated(
    tmp_path, monkeypatch
) -> None:
    import httpx

    from tests.test_chat_pipeline import _FakeMemory

    started = threading.Event()
    release = threading.Event()

    def blocking_stream(_memory, question, **_kwargs):
        started.set()
        assert release.wait(timeout=5)
        yield {"type": "done", "answer": f"answer-{question}", "sources": []}

    monkeypatch.setattr("memo.chat.pipeline.chat_stream", blocking_stream)
    app = build_app(_FakeMemory(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        ask = asyncio.create_task(
            client.post("/api/ask", json={"q": "question", "chat_session_id": "race"})
        )
        assert await asyncio.to_thread(started.wait, 2)
        delete = asyncio.create_task(
            client.post("/api/sessions/delete", json={"session_id": "race"})
        )
        await asyncio.sleep(0.05)
        assert not delete.done()

        release.set()
        ask_response, delete_response = await asyncio.gather(ask, delete)
        history = (await client.get("/api/sessions/race")).json()["turns"]

    assert ask_response.status_code == 200
    assert delete_response.json() == {"ok": True}
    assert history == []


@pytest.mark.asyncio
async def test_delete_all_waits_for_in_flight_asks_and_cannot_be_recreated(
    tmp_path, monkeypatch
) -> None:
    import httpx

    from tests.test_chat_pipeline import _FakeMemory

    started = threading.Event()
    release = threading.Event()

    def blocking_stream(_memory, question, **_kwargs):
        started.set()
        assert release.wait(timeout=5)
        yield {"type": "done", "answer": f"answer-{question}", "sources": []}

    monkeypatch.setattr("memo.chat.pipeline.chat_stream", blocking_stream)
    app = build_app(_FakeMemory(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        ask = asyncio.create_task(
            client.post("/api/ask", json={"q": "question", "chat_session_id": "race-all"})
        )
        assert await asyncio.to_thread(started.wait, 2)
        delete = asyncio.create_task(client.post("/api/sessions/delete-all", json={}))
        await asyncio.sleep(0.05)
        assert not delete.done()

        release.set()
        ask_response, delete_response = await asyncio.gather(ask, delete)
        history = (await client.get("/api/sessions/race-all")).json()["turns"]

    assert ask_response.status_code == 200
    assert delete_response.json() == {"ok": True, "deleted": 1}
    assert history == []


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_mode", ["disconnect", "send_failure"])
async def test_stream_capacity_is_released_when_asgi_exits_before_iteration(
    tmp_path, monkeypatch, exit_mode
) -> None:
    import httpx

    import memo.chat.http as chat_http
    from tests.test_chat_pipeline import _FakeMemory

    monkeypatch.setattr(chat_http, "_MAX_HEAVY_IN_FLIGHT", 1)
    app = build_app(_FakeMemory(tmp_path))
    payload = json.dumps({"q": "abandoned"}).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/ask/stream",
        "raw_path": b"/api/ask/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"127.0.0.1"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 80),
        "state": {},
    }
    incoming = [{"type": "http.request", "body": payload, "more_body": False}]
    if exit_mode == "disconnect":
        incoming.append({"type": "http.disconnect"})
    never = asyncio.Event()

    async def receive():
        if incoming:
            return incoming.pop(0)
        await never.wait()
        raise AssertionError("unreachable")

    async def send(message):
        if exit_mode == "send_failure" and message["type"] == "http.response.start":
            raise RuntimeError("transport failed")

    if exit_mode == "send_failure":
        with pytest.raises(RuntimeError, match="transport failed"):
            await asyncio.wait_for(app(scope, receive, send), timeout=2)
    else:
        await asyncio.wait_for(app(scope, receive, send), timeout=2)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        recovered = await client.post("/api/ask", json={"q": "recovered"})

    assert recovered.status_code == 200
