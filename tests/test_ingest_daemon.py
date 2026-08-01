"""Ingest worker daemon + client + flag-gated fallback.

Runs a real `_IngestServer` on a tmp Unix socket with an INJECTED fake job
runner, so the daemon path is exercised end-to-end without MLX or real git
clones. Also asserts the graceful-degradation contract: with the flag off,
or the daemon down, `Memory.repo_index` runs in-process exactly as before.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from memo import ingest_client
from memo.ingest_ledger import IngestFailureLedger
from memo.ingest_server import _IngestServer, _JobBook, _socket_path


def _wait_done(job_id: str, *, state_dir: Path, timeout: float = 3.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = ingest_client.status(job_id, state_dir=state_dir)
        assert st is not None
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


@pytest.fixture
def ingest_daemon(tmp_path: Path) -> Iterator[tuple[Path, list]]:
    """Start an _IngestServer on a tmp socket with a recording fake runner."""
    runs: list[tuple[str, dict]] = []

    def fake_runner(kind: str, payload: dict) -> dict:
        runs.append((kind, payload))
        return {"ok": True, "indexed": payload.get("url"), "files": 3}

    book = _JobBook(fake_runner)
    server = _IngestServer(str(_socket_path(tmp_path)), book)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield tmp_path, runs
    finally:
        server.shutdown()
        server.server_close()
        book.shutdown()


def test_ping(ingest_daemon: tuple[Path, list]) -> None:
    state_dir, _ = ingest_daemon
    resp = ingest_client.ping(state_dir=state_dir)
    assert resp is not None
    assert resp["ok"] is True
    assert resp["kind"] == "ingest"


def test_enqueue_runs_job_and_returns_result(ingest_daemon: tuple[Path, list]) -> None:
    state_dir, runs = ingest_daemon
    job_id = ingest_client.enqueue(
        "repo", {"url": "https://x/y.git", "force": True}, state_dir=state_dir
    )
    assert job_id is not None
    st = _wait_done(job_id, state_dir=state_dir)
    assert st["state"] == "done"
    assert st["result"] == {"ok": True, "indexed": "https://x/y.git", "files": 3}
    assert runs == [("repo", {"url": "https://x/y.git", "force": True})]


def test_unknown_kind_is_rejected(ingest_daemon: tuple[Path, list]) -> None:
    state_dir, _ = ingest_daemon
    # enqueue() returns None on an error response (daemon rejected the kind).
    assert ingest_client.enqueue("bogus", {"url": "x"}, state_dir=state_dir) is None


def test_runner_exception_surfaces_as_error_state(tmp_path: Path) -> None:
    def boom(kind: str, payload: dict) -> dict:
        raise RuntimeError("clone failed")

    book = _JobBook(boom)
    server = _IngestServer(str(_socket_path(tmp_path)), book)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        job_id = ingest_client.enqueue("repo", {"url": "x"}, state_dir=tmp_path)
        assert job_id is not None
        st = _wait_done(job_id, state_dir=tmp_path)
        assert st["state"] == "error"
        assert "clone failed" in st["error"]
    finally:
        server.shutdown()
        server.server_close()
        book.shutdown()


def test_job_dedupe_returns_active_job_once() -> None:
    release = threading.Event()
    calls: list[dict] = []

    def slow(kind: str, payload: dict) -> dict:
        calls.append(payload)
        release.wait(timeout=2)
        return {"ok": True}

    book = _JobBook(slow)
    try:
        first = book.enqueue_receipt("repo", {"url": "same"})
        second = book.enqueue_receipt("repo", {"url": "same"})
        assert second["job_id"] == first["job_id"]
        assert second["deduplicated"] is True
        release.set()
        deadline = time.time() + 2
        while time.time() < deadline:
            status = book.status(first["job_id"])
            if status and status["state"] == "done":
                break
            time.sleep(0.01)
        assert calls == [{"url": "same"}]
    finally:
        release.set()
        book.shutdown()


def test_fatal_failures_are_ledgers_then_quarantined(tmp_path: Path) -> None:
    ledger = IngestFailureLedger(tmp_path / "jobs.jsonl")

    def crash(kind: str, payload: dict) -> dict:
        raise SystemExit("fatal worker crash")

    book = _JobBook(crash, ledger=ledger, quarantine_threshold=3)
    try:
        for _ in range(3):
            job_id = book.enqueue("repo", {"url": "crash"})
            deadline = time.time() + 2
            while time.time() < deadline:
                status = book.status(job_id)
                if status and status["state"] == "error":
                    break
                time.sleep(0.01)
            assert status is not None and status["fatal"] is True
        quarantined = book.enqueue_receipt("repo", {"url": "crash"})
        assert quarantined["state"] == "quarantined"
        status = book.status(quarantined["job_id"])
        assert status is not None and status["state"] == "quarantined"
        assert book.health()["worker_alive"] is True
        assert ledger.health()["chain_valid"] is True
    finally:
        book.shutdown()


def test_client_returns_none_when_daemon_absent(tmp_path: Path) -> None:
    # No socket at all → every call degrades to None so callers fall back.
    assert ingest_client.ping(state_dir=tmp_path) is None
    assert ingest_client.enqueue("repo", {"url": "x"}, state_dir=tmp_path) is None
    assert ingest_client.status("deadbeef", state_dir=tmp_path) is None


class _FakeCorpus:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def index(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return {"ok": True, "via": "in-process", "url": url}


def _patch_corpus(mock_memory, monkeypatch) -> _FakeCorpus:
    fake = _FakeCorpus()
    monkeypatch.setattr(mock_memory, "_repo_corpus", lambda: fake)
    return fake


def test_repo_index_runs_in_process_when_flag_off(mock_memory, monkeypatch) -> None:
    monkeypatch.delenv("MEMO_INGEST_VIA_DAEMON", raising=False)
    fake = _patch_corpus(mock_memory, monkeypatch)
    # Even if a daemon were up, the flag being off keeps it in-process.
    monkeypatch.setattr("memo.ingest_client.enqueue", lambda *a, **k: "SHOULD_NOT_BE_USED")
    result = mock_memory.repo_index("https://x/y.git")
    assert result["via"] == "in-process"
    assert len(fake.calls) == 1


def test_repo_index_routes_to_daemon_when_flag_on_and_reachable(mock_memory, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_INGEST_VIA_DAEMON", "1")
    fake = _patch_corpus(mock_memory, monkeypatch)
    monkeypatch.setattr("memo.ingest_client.enqueue", lambda kind, payload, **k: "job123")
    result = mock_memory.repo_index("https://x/y.git", force=True)
    assert result == {"queued": True, "job_id": "job123", "via": "ingest-daemon"}
    assert fake.calls == []  # never ran in-process


def test_repo_index_falls_back_in_process_when_flag_on_but_daemon_down(
    mock_memory, monkeypatch
) -> None:
    monkeypatch.setenv("MEMO_INGEST_VIA_DAEMON", "1")
    fake = _patch_corpus(mock_memory, monkeypatch)
    monkeypatch.setattr("memo.ingest_client.enqueue", lambda *a, **k: None)  # daemon unreachable
    result = mock_memory.repo_index("https://x/y.git")
    assert result["via"] == "in-process"
    assert len(fake.calls) == 1


def test_repo_index_with_progress_stays_in_process_even_with_flag(mock_memory, monkeypatch) -> None:
    # progress callbacks can't cross the socket → always in-process.
    monkeypatch.setenv("MEMO_INGEST_VIA_DAEMON", "1")
    fake = _patch_corpus(mock_memory, monkeypatch)
    monkeypatch.setattr("memo.ingest_client.enqueue", lambda *a, **k: "SHOULD_NOT_BE_USED")
    result = mock_memory.repo_index("https://x/y.git", progress=lambda *a: None)
    assert result["via"] == "in-process"
    assert len(fake.calls) == 1


# -- `ingest-daemon start` readiness probe ---------------------------------


def _start_env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path),
    }


def test_start_fails_on_stale_socket_when_child_dies(tmp_path: Path, monkeypatch) -> None:
    """Regression: a stale socket file left by a crashed daemon must not make
    `ingest-daemon start` report success for a child that failed to boot. The
    connect-based probe + proc.poll fail-fast rejects it even though the file
    (which a bare exists() check would trust) is present."""
    import subprocess

    from click.testing import CliRunner

    from memo.cli import cli

    stale = _socket_path(tmp_path)
    stale.touch()  # leftover from a crash — nothing is listening

    class _DeadChild:
        pid = 4242

        def poll(self) -> int:
            return 1  # child exited immediately (failed to boot)

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _DeadChild())
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # keep logs out of ~/Library

    result = CliRunner().invoke(cli, ["ingest-daemon", "start"], env=_start_env(tmp_path))

    assert result.exit_code == 1
    assert "failed to start" in result.stderr
    # Parent must NOT unlink the socket — the child owns that under its start
    # flock (a parent-side unlink can orphan a live concurrent daemon).
    assert stale.exists()


def test_start_succeeds_when_child_binds_fresh_socket(tmp_path: Path, monkeypatch) -> None:
    """The connect-based probe reports success only once the (fake) child
    actually answers a ping on a freshly bound socket — a stale leftover
    socket file must not break a genuine boot either."""
    import subprocess

    from click.testing import CliRunner

    from memo.cli import cli

    _socket_path(tmp_path).touch()  # stale leftover from a crash

    class _LiveChild:
        pid = 4242

        def poll(self) -> None:
            return None  # still running

    started: list[tuple[_IngestServer, _JobBook]] = []

    def fake_popen(*a: object, **k: object) -> _LiveChild:
        # Simulate the real child: under its own start flock, run_server unlinks
        # any stale socket before binding a fresh one.
        _socket_path(tmp_path).unlink(missing_ok=True)
        book = _JobBook(lambda kind, payload: {"ok": True})
        server = _IngestServer(str(_socket_path(tmp_path)), book)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append((server, book))
        return _LiveChild()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    try:
        result = CliRunner().invoke(cli, ["ingest-daemon", "start"], env=_start_env(tmp_path))
        assert result.exit_code == 0, result.output
        assert "started" in result.stderr
    finally:
        for server, book in started:
            server.shutdown()
            server.server_close()
            book.shutdown()


def test_start_does_not_unlink_live_daemon_socket(tmp_path: Path, monkeypatch) -> None:
    """Regression (parent-side unlink orphan): a concurrent daemon that has
    bound its socket but not yet written its pid looks 'not running' to a second
    `start`. The parent must not unlink that live socket — the connect probe
    finds the daemon and leaves it serving."""
    import subprocess

    from click.testing import CliRunner

    from memo.cli import cli

    # A live daemon already bound (no pid file yet → _read_pid returns None).
    book = _JobBook(lambda kind, payload: {"ok": True})
    live = _IngestServer(str(_socket_path(tmp_path)), book)
    threading.Thread(target=live.serve_forever, daemon=True).start()

    class _LiveChild:
        pid = 4242

        def poll(self) -> None:
            return None

    # The faked spawn does NOT bind — the parent must reuse the already-live one.
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _LiveChild())
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    try:
        result = CliRunner().invoke(cli, ["ingest-daemon", "start"], env=_start_env(tmp_path))
        assert result.exit_code == 0, result.output
        # The live daemon is still reachable — the parent did not orphan it.
        assert ingest_client.ping(state_dir=tmp_path) is not None
    finally:
        live.shutdown()
        live.server_close()
        book.shutdown()


def test_serialized_writer_runs_jobs_one_at_a_time(tmp_path: Path) -> None:
    """The single worker drains the queue serially — concurrent enqueues never
    overlap (the 'own serialized writer' guarantee)."""
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def slow(kind: str, payload: dict) -> dict:
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)
        with lock:
            active["now"] -= 1
        return {"ok": True}

    book = _JobBook(slow)
    server = _IngestServer(str(_socket_path(tmp_path)), book)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        ids = [
            ingest_client.enqueue("repo", {"url": f"r{i}"}, state_dir=tmp_path) for i in range(4)
        ]
        for jid in ids:
            _wait_done(jid, state_dir=tmp_path)
        assert active["max"] == 1  # never two jobs writing at once
    finally:
        server.shutdown()
        server.server_close()
        book.shutdown()
