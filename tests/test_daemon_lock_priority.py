"""Recall-daemon lock priority + contention observability guards.

Root cause of the p95 latency tail these pin:

1. `embed_query` acquired the shared lock at priority=1 — the SAME priority as
   the interactive `recall` op — with a 60s timeout. A burst of embed_query
   requests (memflow vec indexing, eval runs, grounding scoring) therefore
   queued AT recall's priority, an arriving recall got no precedence, burned
   its 2500ms budget behind them and bailed empty (hook fell back to the slow
   subprocess path). Only `recall` may hold priority=1 now.
2. The first request after a daemon restart paid the multi-second cold MLX
   load while HOLDING the lock. `run_server` now warms the embedder BEFORE
   binding the socket, so no client can even connect (let alone queue) during
   the load.
3. A recall that waits >500ms or bails on lock-busy emits one structured
   stderr line (`recall_lock_wait` / `recall_lock_bail`) with `wait_ms` and
   the label of the op that held the lock.

All tests run without MLX (embedders are stubs).
"""

from __future__ import annotations

import io
import json
import threading
import time
from types import SimpleNamespace

from memo.recall_socket import (
    _LOCK_WAIT_LOG_MS,
    BUSY_RESPONSE,
    PriorityLock,
    _RecallHandler,
    _SimpleLockWrapper,
    _warmup_embedder,
)

# -- helpers ---------------------------------------------------------------


class _Handler(_RecallHandler):
    """Run `handle()` against in-memory pipes, skipping socketserver setup."""

    def __init__(self, server: SimpleNamespace, payload: dict) -> None:
        self.server = server  # type: ignore[assignment]
        self.rfile = io.BytesIO((json.dumps(payload) + "\n").encode("utf-8"))
        self.wfile = io.BytesIO()


def _serve(server: SimpleNamespace, payload: dict) -> str:
    h = _Handler(server, payload)
    h.handle()
    return h.wfile.getvalue().decode("utf-8").strip()


class _RecordingLock:
    """Priority-lock stub that records every acquire call."""

    def __init__(self, result: bool = True, delay: float = 0.0, holder: str | None = None) -> None:
        self.calls: list[dict] = []
        self._result = result
        self._delay = delay
        self.holder = holder

    def acquire(
        self, priority: int = 0, timeout: float | None = None, label: str | None = None
    ) -> bool:
        self.calls.append({"priority": priority, "timeout": timeout, "label": label})
        if self._delay:
            time.sleep(self._delay)
        return self._result

    def release(self) -> None:
        pass


def _server(lock, state_dir=None) -> SimpleNamespace:
    embedder = SimpleNamespace(
        embed_query=lambda text: [0.1, 0.2, 0.3, 0.4],
        embed=lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts],
    )
    return SimpleNamespace(
        _priority_lock=lock,
        _mem=SimpleNamespace(embedder=embedder, search=lambda *a, **k: []),
        _cfg=SimpleNamespace(embedder_model="stub", embedder_dims=4, state_dir=state_dir),
        _micro_embedder=None,
    )


# -- priority semantics (real PriorityLock) ---------------------------------


def test_queued_recall_acquires_before_queued_embed_queries() -> None:
    """With the lock held, N queued embed_query waiters (priority 0) plus one
    queued recall (priority 1): on release the recall must acquire FIRST."""
    lock = PriorityLock()
    assert lock.acquire(label="embed_batch")  # simulate an in-flight batch chunk

    order: list[str] = []

    def waiter(name: str, prio: int) -> None:
        assert lock.acquire(priority=prio, timeout=10.0, label=name)
        order.append(name)
        lock.release()

    embed_threads = [
        threading.Thread(target=waiter, args=(f"embed_query-{i}", 0)) for i in range(4)
    ]
    for t in embed_threads:
        t.start()
    time.sleep(0.05)  # let the embed waiters park first
    recall_thread = threading.Thread(target=waiter, args=("recall", 1))
    recall_thread.start()

    # The recall must be REGISTERED as a high-priority waiter before the
    # holder releases — poll the counter instead of sleeping blind.
    deadline = time.time() + 5.0
    while time.time() < deadline and lock._high_priority_waiters < 1:
        time.sleep(0.005)
    assert lock._high_priority_waiters == 1

    lock.release()
    recall_thread.join(timeout=5.0)
    for t in embed_threads:
        t.join(timeout=5.0)

    assert order[0] == "recall", f"recall must outrank queued embeds, got order {order}"
    assert sorted(order[1:]) == sorted(f"embed_query-{i}" for i in range(4))


def test_priority_lock_tracks_holder_label() -> None:
    lock = PriorityLock()
    assert lock.holder is None
    assert lock.acquire(label="embed_batch")
    assert lock.holder == "embed_batch"
    lock.release()
    assert lock.holder is None


def test_simple_lock_wrapper_accepts_label_and_tracks_holder() -> None:
    wrapper = _SimpleLockWrapper(threading.Lock())
    assert wrapper.holder is None
    assert wrapper.acquire(priority=1, timeout=1.0, label="recall")
    assert wrapper.holder == "recall"
    wrapper.release()
    assert wrapper.holder is None


# -- handler wiring: which priority each op acquires at ----------------------


def test_recall_op_acquires_at_priority_one(monkeypatch) -> None:
    monkeypatch.setattr("memo.recall_socket._recall_logic", lambda *a, **k: ("{}", None))
    lock = _RecordingLock()
    out = _serve(_server(lock), {"op": "recall", "prompt": "hola que tal daemon"})
    assert out == "{}"
    assert lock.calls == [{"priority": 1, "timeout": 2.5, "label": "recall"}]


def test_embed_query_op_acquires_at_priority_zero() -> None:
    """The tail regression: embed_query at priority=1 queued AT recall's
    priority with a 60s timeout, starving interactive recalls."""
    lock = _RecordingLock()
    out = _serve(_server(lock), {"op": "embed_query", "text": "hola"})
    assert json.loads(out)["dims"] == 4
    assert lock.calls[0]["priority"] == 0
    assert lock.calls[0]["label"] == "embed_query"
    assert lock.calls[0]["timeout"] == 60.0


def test_embed_query_prefers_store_model_identity_over_config() -> None:
    """Wire responses expose the exact identity that owns stored vectors."""
    lock = _RecordingLock()
    server = _server(lock)
    server._mem.store = SimpleNamespace(embedder_model="stub/exact@revision")

    out = _serve(server, {"op": "embed_query", "text": "hola"})

    assert json.loads(out)["model"] == "stub/exact@revision"


def test_search_op_acquires_at_priority_zero(tmp_path) -> None:
    lock = _RecordingLock()
    out = _serve(_server(lock, state_dir=tmp_path), {"op": "search", "prompt": "hola"})
    assert json.loads(out)["results"] == []
    assert lock.calls[0]["priority"] == 0
    assert lock.calls[0]["label"] == "search"


def test_embed_batch_op_acquires_at_priority_zero() -> None:
    lock = _RecordingLock()
    out = _serve(_server(lock), {"op": "embed_batch", "texts": ["a", "b"]})
    assert json.loads(out)["dims"] == 4
    assert all(c["priority"] == 0 and c["label"] == "embed_batch" for c in lock.calls)


# -- lock-busy bail: "{}" within budget + structured line --------------------


def test_lock_busy_recall_bails_empty_within_budget(monkeypatch, capsys) -> None:
    monkeypatch.setenv("MEMO_RECALL_LOCK_TIMEOUT_MS", "200")
    # Lock consumes the full (tiny) timeout then reports busy.
    lock = _RecordingLock(result=False, delay=0.2, holder="embed_query")
    t0 = time.monotonic()
    out = _serve(_server(lock), {"op": "recall", "prompt": "hola que tal"})
    elapsed_s = time.monotonic() - t0

    assert out == BUSY_RESPONSE, "lock-busy recall must emit the busy marker"
    assert json.loads(out) == {"busy": True}  # frozen wire shape the hook client parses
    assert elapsed_s < 1.0, f"bail must respect the ms budget, took {elapsed_s:.2f}s"
    assert lock.calls[0]["timeout"] == 0.2

    err = capsys.readouterr().err
    line = next(ln for ln in err.splitlines() if "recall_lock_bail" in ln)
    row = json.loads(line)
    assert row["event"] == "recall_lock_bail"
    assert row["op"] == "recall"
    assert row["wait_ms"] >= 150
    assert row["held_by"] == "embed_query"


def test_recall_bails_fast_while_warming(capsys) -> None:
    """While the background model load runs (warm event unset), a recall op
    returns the busy marker immediately — distinguishable from a legit empty
    recall ("{}") so the hook client falls back to subprocess instead of
    queueing behind the cold load."""
    srv = _server(_RecordingLock(result=True))
    srv._warm_event = threading.Event()  # unset = warming
    t0 = time.monotonic()
    out = _serve(srv, {"op": "recall", "prompt": "hola que tal"})
    assert out == BUSY_RESPONSE
    assert json.loads(out) == {"busy": True}
    assert time.monotonic() - t0 < 0.5
    err = capsys.readouterr().err
    row = json.loads(next(ln for ln in err.splitlines() if "recall_warming" in ln))
    assert row["event"] == "recall_warming"
    assert row["held_by"] == "warmup"


def test_slow_lock_wait_logs_structured_line(monkeypatch, capsys) -> None:
    monkeypatch.setattr("memo.recall_socket._recall_logic", lambda *a, **k: ("{}", None))
    lock = _RecordingLock(
        result=True, delay=(_LOCK_WAIT_LOG_MS + 100) / 1000.0, holder="embed_batch"
    )
    out = _serve(_server(lock), {"op": "recall", "prompt": "hola que tal"})
    assert out == "{}"

    err = capsys.readouterr().err
    line = next(ln for ln in err.splitlines() if "recall_lock_wait" in ln)
    row = json.loads(line)
    assert row["wait_ms"] > _LOCK_WAIT_LOG_MS
    assert row["held_by"] == "embed_batch"


def test_fast_lock_acquire_logs_nothing(monkeypatch, capsys) -> None:
    monkeypatch.setattr("memo.recall_socket._recall_logic", lambda *a, **k: ("{}", None))
    lock = _RecordingLock(result=True)
    _serve(_server(lock), {"op": "recall", "prompt": "hola que tal"})
    assert "recall_lock" not in capsys.readouterr().err


# -- warm-up at daemon start --------------------------------------------------


def test_warmup_embedder_forces_load_and_reports_ms() -> None:
    calls: list[list[str]] = []

    def embed(texts):
        calls.append(list(texts))
        return [[0.0] for _ in texts]

    ms = _warmup_embedder(SimpleNamespace(embedder=SimpleNamespace(embed=embed)))
    assert ms is not None and ms >= 0.0
    assert len(calls) == 1  # one forced forward pass = model loaded


def test_warmup_embedder_failure_is_nonfatal(capsys) -> None:
    def boom(texts):
        raise RuntimeError("no MLX on this box")

    ms = _warmup_embedder(SimpleNamespace(embedder=SimpleNamespace(embed=boom)))
    assert ms is None
    assert "warm-up failed" in capsys.readouterr().err


def test_run_server_binds_first_then_warms_in_background(monkeypatch, tmp_path) -> None:
    """The socket must bind IMMEDIATELY (so `memo recall-daemon start`'s 2s
    probe succeeds on a cold start) and the cold model load runs in a
    background thread; the server's warm event is unset during the load and
    set when it finishes. Recall ops bail "{}" while it's unset (covered by
    test_recall_bails_fast_while_warming), so the load never counts against a
    queued recall's lock budget."""
    from memo import recall_socket

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(state_dir))

    events: list[str] = []
    warm_started = threading.Event()
    release_warmup = threading.Event()

    def slow_embed(texts):
        warm_started.set()
        assert release_warmup.wait(timeout=5.0)
        events.append("warmup")
        return [[0.0] for _ in texts]

    stub_mem = SimpleNamespace(embedder=SimpleNamespace(embed=slow_embed))
    monkeypatch.setattr("memo.memory.Memory", lambda cfg: stub_mem)

    servers: list[object] = []

    class _FakeServer:
        def __init__(self, sock_path: str, cfg, mem) -> None:
            events.append("bind")
            self._stats = None
            servers.append(self)

    monkeypatch.setattr(recall_socket, "_RecallServer", _FakeServer)

    def _fake_serve(*a, **k):
        # Serving starts with the bind done and the warm-up still in flight.
        assert events[0] == "bind"
        assert warm_started.wait(timeout=5.0)
        srv = servers[0]
        assert not srv._warm_event.is_set(), "warm event must be unset during the load"
        release_warmup.set()
        assert srv._warm_event.wait(timeout=5.0), "warm event must be set when load ends"

    monkeypatch.setattr(recall_socket, "_serve_until_shutdown", _fake_serve)
    monkeypatch.setattr(recall_socket, "_stats_persister", lambda *a, **k: None)
    monkeypatch.setattr(recall_socket.signal, "signal", lambda *a, **k: None)

    recall_socket.run_server(state_dir)

    assert events == ["bind", "warmup"], f"bind must precede the warm-up, got {events}"
