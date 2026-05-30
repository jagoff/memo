"""Orderly-shutdown lifecycle for the recall daemon.

`_serve_until_shutdown` runs `serve_forever()` on a worker thread and shuts the
server down from the main thread when an event fires — replacing the old
`os._exit(0)` signal-handler workaround (which skipped finally/WAL cleanup).
These tests pin that contract without loading MLX (importing recall_server is
cheap; the MLX imports live inside `run_server`).
"""

from __future__ import annotations

import contextlib
import threading
import time

from memo.recall_server import _serve_until_shutdown


class _FakeServer:
    """Mimics the socketserver surface `_serve_until_shutdown` touches."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self.serve_started = False
        self.shutdown_called = False
        self.closed = False

    def serve_forever(self) -> None:
        self.serve_started = True
        # Block like a real serve_forever() until shutdown() releases it.
        self._stop.wait(timeout=5.0)

    def shutdown(self) -> None:
        self.shutdown_called = True
        self._stop.set()

    def server_close(self) -> None:
        self.closed = True


def test_serve_until_shutdown_runs_then_shuts_down_in_order() -> None:
    server = _FakeServer()
    event = threading.Event()
    cleaned: list[str] = []

    runner = threading.Thread(
        target=lambda: _serve_until_shutdown(
            server,
            event,
            on_shutdown=lambda: cleaned.append("cleanup"),
            poll_interval=0.02,
            join_timeout=2.0,
        ),
    )
    runner.start()

    # serve_forever() should be running on its worker thread.
    deadline = time.time() + 2
    while time.time() < deadline and not server.serve_started:
        time.sleep(0.01)
    assert server.serve_started
    assert runner.is_alive()  # still blocked, waiting for the event

    # Request shutdown; the helper must return promptly and in order.
    event.set()
    runner.join(timeout=3.0)
    assert not runner.is_alive()
    assert server.shutdown_called
    assert server.closed
    assert cleaned == ["cleanup"]


def test_serve_until_shutdown_without_cleanup_callback() -> None:
    server = _FakeServer()
    event = threading.Event()
    event.set()  # immediate shutdown request

    # Must not raise when on_shutdown is omitted.
    _serve_until_shutdown(server, event, poll_interval=0.02, join_timeout=2.0)

    assert server.shutdown_called
    assert server.closed


def test_serve_until_shutdown_shuts_down_even_if_cleanup_raises() -> None:
    server = _FakeServer()
    event = threading.Event()
    event.set()

    def _boom() -> None:
        raise RuntimeError("cleanup failed")

    # cleanup raising must not leave the server un-shutdown; shutdown/close run
    # in the finally block before on_shutdown.
    with contextlib.suppress(RuntimeError):
        _serve_until_shutdown(server, event, on_shutdown=_boom, join_timeout=2.0)

    assert server.shutdown_called
    assert server.closed
