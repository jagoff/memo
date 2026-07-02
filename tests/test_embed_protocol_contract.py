"""Frozen-contract test for the shared embedder wire protocol.

`embed_protocol.py` is vendored byte-identically into peer repos (memflow).
There is no shared import to keep them in sync, so each repo pins the SAME
golden wire bytes + field constants here: if either copy drifts, its own copy
of this test fails. Keep this file identical across repos alongside the module.
"""

from __future__ import annotations

import json
import os
import socketserver
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from memo import embed_protocol as ep

# -- frozen constants ------------------------------------------------------


def test_op_names_frozen() -> None:
    assert ep.OP_PING == "ping"
    assert ep.OP_EMBED_QUERY == "embed_query"
    assert ep.OP_EMBED_BATCH == "embed_batch"


def test_response_field_names_frozen() -> None:
    assert ep.FIELD_VECTOR == "vector"
    assert ep.FIELD_VECTORS == "vectors"
    assert ep.FIELD_DIM == "dim"
    assert ep.FIELD_DIMS == "dims"
    assert ep.FIELD_MODEL == "model"
    assert ep.FIELD_OK == "ok"
    assert ep.FIELD_ERROR == "error"


def test_limits_frozen() -> None:
    assert ep.MAX_LINE_BYTES == 16 * 1024 * 1024
    assert ep.DEFAULT_TIMEOUT_S == 5.0
    assert ep.PING_TIMEOUT_S == 0.5


# -- frozen framing (golden bytes) -----------------------------------------


def test_encode_request_golden_bytes() -> None:
    assert ep.encode_request("ping") == b'{"op": "ping"}\n'
    assert ep.encode_request("embed_query", text="hola") == (
        b'{"op": "embed_query", "text": "hola"}\n'
    )


def test_encode_payload_golden_bytes() -> None:
    assert ep.encode_payload({"op": "embed_batch", "texts": ["a", "b"]}) == (
        b'{"op": "embed_batch", "texts": ["a", "b"]}\n'
    )


def test_encode_request_is_newline_terminated_utf8_json() -> None:
    raw = ep.encode_request("embed_query", text="acción")
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8")) == {"op": "embed_query", "text": "acción"}


# -- live socket round-trip ------------------------------------------------


@pytest.fixture
def echo_socket(tmp_path: Path) -> Iterator[tuple[Path, dict]]:
    """A real AF_UNIX server that records the request and replies with a
    canned JSON line. Exercises the actual send/recv framing path."""
    sock_path = Path("/tmp") / f"memo-ep-{os.getpid()}-{uuid.uuid4().hex}.sock"
    captured: dict = {}

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            line = self.rfile.readline(ep.MAX_LINE_BYTES)
            captured["request"] = json.loads(line.decode("utf-8").strip())
            reply = captured.get("reply", '{"ok": true}')
            self.wfile.write((reply + "\n").encode("utf-8"))

    server = socketserver.ThreadingUnixStreamServer(str(sock_path), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield sock_path, captured
    finally:
        server.shutdown()
        server.server_close()
        sock_path.unlink(missing_ok=True)


def test_send_request_parses_dict(echo_socket: tuple[Path, dict]) -> None:
    sock_path, captured = echo_socket
    captured["reply"] = json.dumps({"vector": [0.1, 0.2], "dim": 2, "dims": 2, "model": "x"})
    resp = ep.send_request(sock_path, {"op": "embed_query", "text": "q"}, timeout=2.0)
    assert resp == {"vector": [0.1, 0.2], "dim": 2, "dims": 2, "model": "x"}
    assert captured["request"] == {"op": "embed_query", "text": "q"}


def test_send_request_line_returns_raw(echo_socket: tuple[Path, dict]) -> None:
    sock_path, captured = echo_socket
    captured["reply"] = '{"hookSpecificOutput": {"a": 1}}'
    line = ep.send_request_line(sock_path, {"op": "recall", "prompt": "p"}, timeout=2.0)
    assert line == '{"hookSpecificOutput": {"a": 1}}'


def test_missing_socket_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "nope.sock"
    assert ep.send_request(missing, {"op": "ping"}, timeout=0.5) is None
    assert ep.send_request_line(missing, {"op": "ping"}, timeout=0.5) is None


def test_default_socket_path_uses_short_path_for_deep_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep = tmp_path
    for n in range(12):
        deep = deep / f"very-long-directory-name-{n:02d}"
    monkeypatch.delenv("MEMFLOW_EMBED_SOCKET", raising=False)
    monkeypatch.setenv("MEMO_STATE_DIR", str(deep))

    sock = ep.default_socket_path()

    assert sock.name.endswith(".sock")
    assert sock != deep / "recall.sock"
    assert len(str(sock)) < 104


def test_non_json_response_returns_none(echo_socket: tuple[Path, dict]) -> None:
    sock_path, captured = echo_socket
    captured["reply"] = "not json at all"
    assert ep.send_request(sock_path, {"op": "ping"}, timeout=2.0) is None
