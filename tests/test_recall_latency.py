"""Recall daemon latency/productivity guards.

Two regressions these pin:

1. A cold `embed_batch` must NOT hold the shared lock for the whole batch —
   it embeds in chunks, releasing the lock between them, so a pending recall
   query-embed interleaves instead of starving (the 53s tail in recall.log).
2. `_recall_logic` defers the recall.log write to a thunk the caller invokes
   only after the response is delivered — an abandoned recall (client timed
   out → ran subprocess) must not double-log against the subprocess row.

Both run without loading MLX (the embedder is stubbed).
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from memo.dashboard import recall_log_path
from memo.memory import MemoryRecord
from memo.recall_server import _recall_logic, _RecallHandler


def _rec(id_: str, title: str, score: float) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"notes/{id_}.md",
        title=title,
        type="note",
        tags=[],
        created="2026-05-21T00:00:00+00:00",
        updated="2026-05-21T00:00:00+00:00",
        body="body " * 20,
        extra={},
        score=score,
    )


def test_embed_batch_chunks_and_releases_lock_between_chunks(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_EMBED_BATCH_CHUNK", "2")
    lock = threading.Lock()
    first_inflight = threading.Event()
    release_first = threading.Event()
    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        if len(calls) == 1:
            # Hold the lock for the duration of the FIRST chunk only.
            first_inflight.set()
            release_first.wait(timeout=2.0)
        return [[0.1, 0.2] for _ in texts]

    # The daemon serialises embeds through a PriorityLock; wrap the test's plain
    # lock with that interface so the per-chunk acquire/release path is exercised
    # exactly as production runs it (the underlying `lock` is what we assert on).
    class _PLock:
        def __init__(self, lk: threading.Lock) -> None:
            self._lk = lk

        def acquire(self, priority: int = 0, timeout: float | None = None) -> bool:
            return self._lk.acquire(timeout=timeout if timeout is not None else -1)

        def release(self) -> None:
            self._lk.release()

    server = SimpleNamespace(
        _priority_lock=_PLock(lock),
        _mem=SimpleNamespace(embedder=SimpleNamespace(embed=fake_embed)),
        _cfg=SimpleNamespace(embedder_model="stub"),
    )
    handler = SimpleNamespace(server=server)

    out: dict[str, str] = {}

    def run() -> None:
        out["res"] = _RecallHandler._embed_batch(handler, {"texts": ["a", "b", "c", "d"]})

    t = threading.Thread(target=run)
    t.start()

    # First chunk is in-flight and holds the lock — a concurrent acquire fails.
    assert first_inflight.wait(timeout=2.0)
    assert lock.acquire(timeout=0.1) is False

    # Release the first chunk; the batch must finish the second chunk in a
    # SEPARATE `with lock` block (proving the lock was released between chunks).
    release_first.set()
    t.join(timeout=2.0)
    assert not t.is_alive()

    res = json.loads(out["res"])
    assert res["dim"] == 2
    assert res["model"] == "stub"
    # 4 texts, chunk=2 → exactly two embed() calls, two chunks.
    assert calls == [["a", "b"], ["c", "d"]]


def test_recall_logic_defers_log_until_thunk_called(monkeypatch, tmp_path) -> None:
    class StubMemory:
        def search(self, query, limit, mode, recency=False, exclude_types=None):
            return [_rec("aaaaaaaa", "Surfaced", 0.91)]

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_CONTEXTUAL", "0")  # skip contextual side effects
    monkeypatch.setenv("MEMO_RECALL_EXPAND_CONTEXT", "0")

    result, log_fn = _recall_logic(
        "a genuinely real prompt about the daemon",
        cwd=None,
        mem=StubMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
        t0=time.time(),
    )

    # Response is built, but recall.log is NOT written yet — the caller logs
    # only after delivery.
    assert "additionalContext" in result
    assert callable(log_fn)
    assert not recall_log_path(tmp_path).is_file()

    log_fn()
    assert recall_log_path(tmp_path).is_file()
    row = json.loads(recall_log_path(tmp_path).read_text().splitlines()[-1])
    assert row["via"] == "daemon"
    assert row["hits"][0]["id"] == "aaaaaaaa"


def test_recall_logic_no_hits_returns_none_thunk(monkeypatch, tmp_path) -> None:
    class EmptyMemory:
        def search(self, query, limit, mode, recency=False, exclude_types=None):
            return []

    monkeypatch.setenv("MEMO_RECALL_EXPAND_CONTEXT", "0")
    result, log_fn = _recall_logic(
        "a prompt that finds nothing at all",
        cwd=None,
        mem=EmptyMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
        t0=time.time(),
    )
    assert result == "{}"
    assert log_fn is None
