"""Tests for the shared-embedder socket protocol + client.

These tests run a real `_RecallServer` on a tmp Unix socket but stub
the underlying `MLXEmbedder.embed` / `embed_query` methods so no MLX
load happens. Each test gets a fresh server thread + tmp state_dir, so
parallel pytest runs do not collide.
"""

from __future__ import annotations

import json
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import memo.embedder_client as embedder_client
from memo.embed_protocol import MAX_LINE_BYTES
from memo.recall_server import (
    _RecallServer,
    _socket_path,
    connect_and_send,
)

# -- helpers ---------------------------------------------------------------


class _StubEmbedder:
    """Deterministic embedder stub — no MLX, no I/O."""

    dims = 8

    def embed(self, texts):
        # Return a tiny deterministic vector per text so test assertions
        # can verify routing/ordering without depending on MLX.
        return [[float((len(t) + i) % 7) for i in range(self.dims)] for t in texts]

    def embed_query(self, text):
        return [float((len(text) + i) % 5) for i in range(self.dims)]


class _StubCfg:
    embedder_model = "stub-embedder/test-0.0"
    embedder_dims = _StubEmbedder.dims

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir


class _StubMemory:
    def __init__(self, embedder) -> None:
        self.embedder = embedder


def _send_raw(state_dir: Path, payload: dict[str, Any], timeout: float = 2.0) -> dict[str, Any]:
    sock_path = _socket_path(state_dir)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(sock_path))
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode("utf-8").strip())


@pytest.fixture
def daemon_server() -> Iterator[tuple[Path, _RecallServer]]:
    """Spin up a real `_RecallServer` thread on a tmp Unix socket.

    macOS limits AF_UNIX paths to ~104 chars, well shorter than the
    pytest tmp_path. Use `/tmp/memo-test-*` directly so the bind never
    hits ENAMETOOLONG.
    """
    state_dir = Path(tempfile.mkdtemp(prefix="memo-test-daemon-", dir="/tmp"))
    sock_path = _socket_path(state_dir)
    embedder = _StubEmbedder()
    cfg = _StubCfg(state_dir)
    mem = _StubMemory(embedder)
    server = _RecallServer(str(sock_path), cfg, mem)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deadline = time.time() + 2.0
    while time.time() < deadline and not sock_path.exists():
        time.sleep(0.01)
    try:
        yield state_dir, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        shutil.rmtree(state_dir, ignore_errors=True)


@pytest.fixture
def daemon(daemon_server: tuple[Path, _RecallServer]) -> Path:
    return daemon_server[0]


@pytest.fixture(autouse=True)
def _reset_inproc_cache(monkeypatch):
    """Clear the embedder_client in-process singleton between tests so
    monkeypatching propagates and ordering between tests does not matter.
    """
    monkeypatch.setattr(embedder_client, "_inproc_embedder", None)
    monkeypatch.setattr(embedder_client, "_cached_state_dir", None)
    yield


# -- protocol-level tests --------------------------------------------------


def test_ping_returns_model_and_dims(daemon: Path):
    resp = _send_raw(daemon, {"op": "ping"})
    assert resp["ok"] is True
    assert resp["model"] == "stub-embedder/test-0.0"
    assert resp["dims"] == _StubEmbedder.dims
    # `started_at` + `uptime_s` were added when the daemon got stats; both
    # must round-trip so peers (`memo embed-daemon status`) can report uptime.
    assert isinstance(resp.get("started_at"), (int, float))
    assert isinstance(resp.get("uptime_s"), int)


def test_embed_query_returns_vector(daemon: Path):
    resp = _send_raw(daemon, {"op": "embed_query", "text": "hola mundo"})
    assert resp["dim"] == _StubEmbedder.dims
    assert resp["model"] == "stub-embedder/test-0.0"
    assert resp["vector"] == _StubEmbedder().embed_query("hola mundo")


def test_embed_batch_preserves_order(daemon: Path):
    texts = ["one", "two-doc", "tres documentos"]
    resp = _send_raw(daemon, {"op": "embed_batch", "texts": texts})
    assert resp["dim"] == _StubEmbedder.dims
    assert resp["vectors"] == _StubEmbedder().embed(texts)


def test_embed_query_empty_text_is_error(daemon: Path):
    resp = _send_raw(daemon, {"op": "embed_query", "text": "  "})
    assert "error" in resp
    assert "empty" in resp["error"]


def test_embed_batch_rejects_non_string_elements(daemon: Path):
    resp = _send_raw(daemon, {"op": "embed_batch", "texts": ["ok", 42]})
    assert "error" in resp
    assert "string" in resp["error"]


def test_unknown_op_returns_error(daemon: Path):
    resp = _send_raw(daemon, {"op": "warp_drive"})
    assert "error" in resp
    assert "warp_drive" in resp["error"]


def test_legacy_recall_request_still_works(daemon: Path):
    # `_recall_logic` calls into mem.search/etc which the stub doesn't
    # implement; instead verify the dispatch path: a recall request with
    # empty prompt returns `{}` BEFORE hitting search, as documented.
    resp = _send_raw(daemon, {"prompt": "", "cwd": ""})
    assert resp == {}


# -- client adapter tests --------------------------------------------------


def test_client_embed_query_via_socket(daemon: Path):
    vec = embedder_client.embed_query("hola mundo", state_dir=daemon)
    assert vec == _StubEmbedder().embed_query("hola mundo")


def test_client_embed_batch_via_socket(daemon: Path):
    texts = ["a", "bb", "ccc"]
    vecs = embedder_client.embed(texts, state_dir=daemon)
    assert vecs == _StubEmbedder().embed(texts)


def test_client_rejects_daemon_with_wrong_model_identity(daemon: Path, monkeypatch):
    """Never stamp vectors from a stale daemon as belonging to a new model pin."""
    sentinel = _StubEmbedder()
    sentinel.embed_query = lambda _text: [99.0] * sentinel.dims
    monkeypatch.setattr(embedder_client, "_inproc", lambda: sentinel)

    vec = embedder_client.embed_query(
        "identity-sensitive",
        state_dir=daemon,
        expected_model="stub-embedder/test-0.0@exact-revision",
        expected_dims=_StubEmbedder.dims,
    )

    assert vec == [99.0] * sentinel.dims


def test_client_rejects_daemon_with_wrong_dimensions(daemon: Path, monkeypatch):
    sentinel = _StubEmbedder()
    sentinel.embed = lambda texts: [[77.0] * 16 for _ in texts]
    monkeypatch.setattr(embedder_client, "_inproc", lambda: sentinel)

    vecs = embedder_client.embed(
        ["dimension-sensitive"],
        state_dir=daemon,
        expected_model="stub-embedder/test-0.0",
        expected_dims=16,
    )

    assert vecs == [[77.0] * 16]


def test_daemon_compatibility_requires_exact_model_and_dimensions():
    info = {"model": "model@example-revision", "dims": 2560}

    assert embedder_client.daemon_is_compatible(
        info, expected_model="model@example-revision", expected_dims=2560
    )
    assert not embedder_client.daemon_is_compatible(
        info, expected_model="model", expected_dims=2560
    )
    assert not embedder_client.daemon_is_compatible(
        info, expected_model="model@example-revision", expected_dims=1024
    )
    assert not embedder_client.daemon_is_compatible(
        None, expected_model="model@example-revision", expected_dims=2560
    )


def test_client_embed_rejects_bare_str(daemon: Path):
    with pytest.raises(TypeError, match="Sequence"):
        embedder_client.embed("oops", state_dir=daemon)  # type: ignore[arg-type]


def test_client_embed_query_empty_raises(daemon: Path):
    with pytest.raises(ValueError, match="empty"):
        embedder_client.embed_query("   ", state_dir=daemon)


def test_client_embed_empty_list_returns_empty(daemon: Path):
    assert embedder_client.embed([], state_dir=daemon) == []


def test_client_ping_via_socket(daemon: Path):
    resp = embedder_client.ping(state_dir=daemon)
    assert resp is not None
    assert resp["ok"] is True
    assert resp["dims"] == _StubEmbedder.dims


def test_client_ping_returns_none_when_daemon_missing(tmp_path: Path):
    assert embedder_client.ping(state_dir=tmp_path) is None


def test_client_falls_back_in_process_when_daemon_missing(tmp_path: Path, monkeypatch):
    """No daemon → lazy MLXEmbedder load. We monkeypatch the lazy loader
    to return a stub so the test doesn't actually load MLX.
    """
    sentinel = _StubEmbedder()
    monkeypatch.setattr(embedder_client, "_inproc", lambda: sentinel)

    vec = embedder_client.embed_query("fallback hello", state_dir=tmp_path)
    assert vec == sentinel.embed_query("fallback hello")

    vecs = embedder_client.embed(["a", "b"], state_dir=tmp_path)
    assert vecs == sentinel.embed(["a", "b"])


def test_client_require_daemon_raises_when_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON", "1")
    with pytest.raises(RuntimeError, match="daemon unreachable"):
        embedder_client.embed_query("hello", state_dir=tmp_path)
    with pytest.raises(RuntimeError, match="daemon unreachable"):
        embedder_client.embed(["a"], state_dir=tmp_path)


def test_client_falls_back_when_daemon_returns_error(daemon: Path, monkeypatch):
    """If the daemon responds with `{"error": ...}` (malformed input),
    the client should fall through to in-process rather than crash.
    """
    sentinel = _StubEmbedder()
    monkeypatch.setattr(embedder_client, "_inproc", lambda: sentinel)
    # Force the daemon to error by patching the handler to always reject.
    # Simpler: send an op the daemon doesn't know via the public client by
    # patching `_try_socket` to simulate a daemon error response.
    monkeypatch.setattr(
        embedder_client,
        "_try_socket",
        lambda *_args, **_kw: None,
    )
    vec = embedder_client.embed_query("via fallback", state_dir=daemon)
    assert vec == sentinel.embed_query("via fallback")


def test_send_request_helper_round_trips(daemon: Path):
    raw = connect_and_send(daemon, {"op": "ping"}, timeout=2.0)
    assert raw is not None
    assert json.loads(raw)["ok"] is True


@pytest.mark.parametrize(
    ("wire", "expected_errors"),
    [
        pytest.param(b"", 0, id="eof"),
        pytest.param(b"x" * MAX_LINE_BYTES, 1, id="oversized"),
        pytest.param(b"{\n", 1, id="invalid-json"),
        pytest.param(b"[]\n", 1, id="non-object-json"),
    ],
)
def test_invalid_frames_are_answered_and_accounted(
    daemon_server: tuple[Path, _RecallServer], wire: bytes, expected_errors: int
):
    state_dir, server = daemon_server
    before = server._stats.snapshot()["ops"].get("parse", {"count": 0, "errors": 0})

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        sock.connect(str(_socket_path(state_dir)))
        if wire:
            sock.sendall(wire)
        sock.shutdown(socket.SHUT_WR)
        response = b""
        while b"\n" not in response:
            response += sock.recv(65536)

    assert json.loads(response) == {}
    after = server._stats.snapshot()["ops"]["parse"]
    assert after["count"] == before["count"] + 1
    assert after["errors"] == before["errors"] + expected_errors


# -- stats + status -------------------------------------------------------


def test_stats_op_returns_snapshot(daemon: Path):
    # Drive a few embed_query requests so counts are non-zero.
    for _ in range(3):
        _send_raw(daemon, {"op": "embed_query", "text": "hola"})
    _send_raw(daemon, {"op": "ping"})

    snap = _send_raw(daemon, {"op": "stats"})
    assert snap["model"] == "stub-embedder/test-0.0"
    assert snap["dims"] == _StubEmbedder.dims
    assert isinstance(snap["uptime_s"], int)

    ops = snap["ops"]
    assert ops["embed_query"]["count"] == 3
    assert ops["embed_query"]["errors"] == 0
    assert ops["embed_query"]["samples"] == 3
    # The stats op itself is recorded too (this one is the 4th caller).
    assert ops["ping"]["count"] == 1
    # Each per-op p50/p95/p99 are non-None once at least one sample exists.
    for pct in ("p50_ms", "p95_ms", "p99_ms"):
        assert ops["embed_query"][pct] is not None


def test_stats_counts_errors(daemon: Path):
    # `unknown` op increments the count under the literal op key with errors=1.
    _send_raw(daemon, {"op": "warp_drive"})
    snap = _send_raw(daemon, {"op": "stats"})
    assert snap["ops"]["warp_drive"]["count"] == 1
    assert snap["ops"]["warp_drive"]["errors"] == 1


@pytest.mark.concurrency
@pytest.mark.resource_hygiene
def test_response_is_not_visible_before_request_stats(
    daemon_server: tuple[Path, _RecallServer], monkeypatch: pytest.MonkeyPatch
):
    state_dir, server = daemon_server
    record_entered = threading.Event()
    release_record = threading.Event()
    original_record = server._stats.record

    def blocking_record(op: str, latency_ms: float, *, error: bool = False) -> None:
        record_entered.set()
        assert release_record.wait(timeout=2.0)
        original_record(op, latency_ms, error=error)

    monkeypatch.setattr(server._stats, "record", blocking_record)

    sock_path = _socket_path(state_dir)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        sock.connect(str(sock_path))
        sock.sendall(b'{"op":"warp_drive"}\n')
        assert record_entered.wait(timeout=2.0)
        try:
            sock.settimeout(0.05)
            with pytest.raises(socket.timeout):
                sock.recv(1)
        finally:
            release_record.set()

        sock.settimeout(2.0)
        response = b""
        while b"\n" not in response:
            response += sock.recv(65536)

    assert "unknown op" in json.loads(response)["error"]


def test_client_stats_via_socket(daemon: Path):
    _send_raw(daemon, {"op": "embed_query", "text": "warm"})
    info = embedder_client.stats(state_dir=daemon)
    assert info is not None
    assert info["model"] == "stub-embedder/test-0.0"
    assert info["ops"]["embed_query"]["count"] >= 1


def test_client_status_via_socket(daemon: Path):
    info = embedder_client.status(state_dir=daemon)
    assert info is not None
    assert info["ok"] is True
    assert info["model"] == "stub-embedder/test-0.0"
    assert isinstance(info["uptime_s"], int)


def test_client_stats_returns_none_when_daemon_missing(tmp_path: Path):
    assert embedder_client.stats(state_dir=tmp_path) is None


def test_client_status_returns_none_when_daemon_missing(tmp_path: Path):
    assert embedder_client.status(state_dir=tmp_path) is None


# -- percentile helper ----------------------------------------------------


def test_percentile_empty_returns_none():
    from memo.recall_server import _percentile

    assert _percentile([], 50) is None


def test_percentile_single_value():
    from memo.recall_server import _percentile

    assert _percentile([7.0], 99) == 7.0


def test_percentile_known_distribution():
    """100 samples 1..100 → p50=50.5, p99≈99.01."""
    from memo.recall_server import _percentile

    xs = sorted(float(i) for i in range(1, 101))
    assert _percentile(xs, 50) == pytest.approx(50.5)
    assert _percentile(xs, 99) == pytest.approx(99.01, abs=0.05)


# -- per-op socket timeouts -------------------------------------------------


def _capture_send_timeouts(monkeypatch) -> list[float]:
    """Patch `connect_and_send` to record the timeout of each call and
    simulate an unreachable daemon (returns None → fallback path)."""
    captured: list[float] = []

    def fake(_state_dir, _payload, timeout):
        captured.append(timeout)
        return None

    monkeypatch.setattr(embedder_client, "connect_and_send", fake)
    return captured


def test_embed_query_uses_query_timeout_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MEMO_EMBEDDER_CLIENT_TIMEOUT", raising=False)
    captured = _capture_send_timeouts(monkeypatch)
    monkeypatch.setattr(embedder_client, "_inproc", lambda: _StubEmbedder())
    embedder_client.embed_query("q", state_dir=tmp_path)
    assert captured == [embedder_client._QUERY_TIMEOUT_S]


def test_embed_batch_uses_batch_timeout_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MEMO_EMBEDDER_CLIENT_TIMEOUT", raising=False)
    captured = _capture_send_timeouts(monkeypatch)
    monkeypatch.setattr(embedder_client, "_inproc", lambda: _StubEmbedder())
    embedder_client.embed(["a", "b"], state_dir=tmp_path)
    assert captured == [embedder_client._BATCH_TIMEOUT_S]
    assert embedder_client._BATCH_TIMEOUT_S > embedder_client._QUERY_TIMEOUT_S


def test_ping_and_stats_use_control_timeout_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MEMO_EMBEDDER_CLIENT_TIMEOUT", raising=False)
    captured = _capture_send_timeouts(monkeypatch)
    embedder_client.ping(state_dir=tmp_path)
    embedder_client.stats(state_dir=tmp_path)
    assert captured == [
        embedder_client._CONTROL_TIMEOUT_S,
        embedder_client._CONTROL_TIMEOUT_S,
    ]
    assert embedder_client._CONTROL_TIMEOUT_S < embedder_client._QUERY_TIMEOUT_S


def test_timeout_flag_overrides_all_ops(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_EMBEDDER_CLIENT_TIMEOUT", "42.5")
    captured = _capture_send_timeouts(monkeypatch)
    monkeypatch.setattr(embedder_client, "_inproc", lambda: _StubEmbedder())
    embedder_client.embed_query("q", state_dir=tmp_path)
    embedder_client.embed(["a"], state_dir=tmp_path)
    embedder_client.ping(state_dir=tmp_path)
    assert captured == [42.5, 42.5, 42.5]
