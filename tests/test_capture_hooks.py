"""Capture hooks: Stop hook + incremental + watermark management tests.

Tests the hook infrastructure (Group 6):
- run_capture() — Stop hook entry point
- run_capture_incremental() — Incremental capture entry
- Watermark state management (load, save, tick_due)
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pytest

from memo.capture import _hash_assistant
from memo.capture_hooks import (
    _load_state,
    _load_watermark,
    _save_state,
    _save_watermark,
    incremental_tick_due,
    list_sessions_without_watermark,
    run_capture,
    run_capture_incremental,
)

pytestmark = pytest.mark.concurrency


def _setup_env(tmp_path: Path, monkeypatch) -> Path:
    """Isolated data/state/vault + stub embedder."""
    data = tmp_path / "data"
    vault = tmp_path / "vault"
    state = tmp_path / "state"
    data.mkdir()
    (vault / "Obsidian" / "AI" / "memory").mkdir(parents=True)
    state.mkdir()
    monkeypatch.setenv("MEMO_DATA_DIR", str(data))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(vault))
    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "0")
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, query: [1.0, 0.0, 0.0, 0.0],
    )
    return state


# ── Stop hook state ────────────────────────────────────────────────────────


def test_stop_hook_state_load_save(tmp_path: Path):
    """Stop hook state loads/saves atomically."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Load from non-existent file
    state = _load_state(state_dir)
    assert state == {}

    # Save state
    state_data = {"last_hash": "abc123", "last_save_ts": 1234567890.0}
    _save_state(state_dir, state_data)
    assert (state_dir / "last-capture.json").exists()

    # Load saved state
    loaded = _load_state(state_dir)
    assert loaded == state_data


def test_stop_hook_state_corrupted_degrades(tmp_path: Path):
    """Corrupted state file loads as empty dict."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Write invalid JSON
    (state_dir / "last-capture.json").write_text("not json", encoding="utf-8")

    # Should degrade gracefully
    state = _load_state(state_dir)
    assert state == {}


# ── Incremental watermark ──────────────────────────────────────────────────


def test_watermark_load_save(tmp_path: Path):
    """Incremental watermark loads/saves atomically."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    session_id = "sess-123"

    # Load non-existent watermark
    wm = _load_watermark(state_dir, session_id)
    assert wm == {}

    # Save watermark
    wm_data = {"session_id": session_id, "exchange_count": 5, "updated": time.time()}
    _save_watermark(state_dir, session_id, wm_data)

    # Verify file exists
    wm_file = state_dir / ".capture_watermark" / f"{session_id}.json"
    assert wm_file.exists()

    # Load saved watermark
    loaded = _load_watermark(state_dir, session_id)
    assert loaded["session_id"] == session_id
    assert loaded["exchange_count"] == 5


def test_watermark_path_rejects_session_id_traversal(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    outside = tmp_path / "outside.json"

    with pytest.raises(ValueError, match="session_id"):
        _save_watermark(state_dir, "../../outside", {"exchange_count": 1})

    assert not outside.exists()


def test_watermark_rejects_symlinked_state_directory(tmp_path: Path):
    state_dir = tmp_path / "state"
    outside = tmp_path / "outside"
    state_dir.mkdir()
    outside.mkdir()
    (state_dir / ".capture_watermark").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe watermark directory"):
        _save_watermark(state_dir, "sess-123", {"exchange_count": 1})

    assert not (outside / "sess-123.json").exists()


def test_watermark_corrupted_degrades(tmp_path: Path):
    """Corrupted watermark loads as empty dict."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    session_id = "sess-123"

    # Pre-create directory and corrupt file
    wm_dir = state_dir / ".capture_watermark"
    wm_dir.mkdir()
    (wm_dir / f"{session_id}.json").write_text("invalid json", encoding="utf-8")

    # Should degrade gracefully
    wm = _load_watermark(state_dir, session_id)
    assert wm == {}


def test_watermark_non_dict_degrades(tmp_path: Path):
    """Watermark that's a list (not dict) degrades to empty."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    session_id = "sess-123"

    wm_dir = state_dir / ".capture_watermark"
    wm_dir.mkdir()
    (wm_dir / f"{session_id}.json").write_text("[]", encoding="utf-8")

    # Should degrade gracefully
    wm = _load_watermark(state_dir, session_id)
    assert wm == {}


# ── Incremental tick throttle ──────────────────────────────────────────────


def test_incremental_tick_due_interval(tmp_path: Path):
    """Tick throttle respects interval_s."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    session_id = "sess-123"

    # First check — no watermark, always due
    assert incremental_tick_due(state_dir, session_id, interval_s=10) is True

    # Save watermark with recent timestamp
    now = time.time()
    _save_watermark(state_dir, session_id, {"updated": now, "exchange_count": 0})

    # Check again within interval — not due
    assert incremental_tick_due(state_dir, session_id, interval_s=10) is False

    # Check with 0 interval — always due
    assert incremental_tick_due(state_dir, session_id, interval_s=0) is True


def test_incremental_tick_due_old_watermark(tmp_path: Path):
    """Tick becomes due after interval elapses."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    session_id = "sess-123"

    # Save watermark with old timestamp
    old_time = time.time() - 100  # 100 seconds ago
    _save_watermark(state_dir, session_id, {"updated": old_time, "exchange_count": 0})

    # 10-second interval — should be due
    assert incremental_tick_due(state_dir, session_id, interval_s=10) is True


def test_incremental_tick_due_invalid_timestamp(tmp_path: Path):
    """Invalid timestamp degrades to due (reprocess)."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    session_id = "sess-123"

    # Save watermark with invalid timestamp
    wm_dir = state_dir / ".capture_watermark"
    wm_dir.mkdir()
    (wm_dir / f"{session_id}.json").write_text(
        json.dumps({"updated": "not a number"}), encoding="utf-8"
    )

    # Should degrade to due (safe to reprocess)
    assert incremental_tick_due(state_dir, session_id, interval_s=10) is True


# ── Session watermark filtering ────────────────────────────────────────────


def test_list_sessions_without_watermark_empty_watermark_dir(tmp_path: Path):
    """No watermark dir → all sessions are pending."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    sessions = [
        {"session_id": "sess-1", "created_at": 100},
        {"session_id": "sess-2", "created_at": 200},
    ]

    pending = list_sessions_without_watermark(state_dir, sessions, limit=10)
    assert len(pending) == 2
    assert pending[0]["session_id"] == "sess-1"


def test_list_sessions_without_watermark_filters_existing(tmp_path: Path):
    """Sessions with watermarks are filtered out."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Pre-create watermark for sess-1
    wm_dir = state_dir / ".capture_watermark"
    wm_dir.mkdir()
    (wm_dir / "sess-1.json").write_text(
        json.dumps({"session_id": "sess-1", "exchange_count": 0}),
        encoding="utf-8",
    )

    sessions = [
        {"session_id": "sess-1", "created_at": 100},
        {"session_id": "sess-2", "created_at": 200},
        {"session_id": "sess-3", "created_at": 300},
    ]

    pending = list_sessions_without_watermark(state_dir, sessions, limit=10)
    assert len(pending) == 2
    assert all(s["session_id"] != "sess-1" for s in pending)


def test_list_sessions_without_watermark_respects_limit(tmp_path: Path):
    """Respects limit parameter."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    sessions = [{"session_id": f"sess-{i}", "created_at": i * 100} for i in range(1, 11)]

    pending = list_sessions_without_watermark(state_dir, sessions, limit=3)
    assert len(pending) == 3


def test_list_sessions_without_watermark_ignores_missing_session_id(tmp_path: Path):
    """Sessions without session_id are skipped when filtering against watermarks."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Pre-create the watermark directory so filtering is active
    wm_dir = state_dir / ".capture_watermark"
    wm_dir.mkdir()
    # Pre-create a watermark for sess-2
    (wm_dir / "sess-2.json").write_text(
        json.dumps({"session_id": "sess-2", "exchange_count": 0}),
        encoding="utf-8",
    )

    sessions = [
        {"created_at": 100},  # no session_id
        {"session_id": "sess-2", "created_at": 200},  # has watermark
        {"session_id": "sess-3", "created_at": 300},  # no watermark
    ]

    pending = list_sessions_without_watermark(state_dir, sessions, limit=10)
    # Session without session_id is skipped, sess-2 has watermark so filtered out,
    # only sess-3 should be returned
    assert len(pending) == 1
    assert pending[0]["session_id"] == "sess-3"


# ── Integration: run_capture (Stop hook) ───────────────────────────────────


def test_run_capture_empty_transcript(tmp_path: Path, monkeypatch):
    """Empty transcript → no_pair status."""
    _setup_env(tmp_path, monkeypatch)
    transcript = tmp_path / "empty.jsonl"
    transcript.write_text("", encoding="utf-8")

    result = run_capture(transcript, debug=False)
    assert result["status"] == "no_pair"


def test_run_capture_no_trigger_keywords(tmp_path: Path, monkeypatch):
    """Transcript with no trigger keywords → no_trigger status."""
    _setup_env(tmp_path, monkeypatch)
    transcript = tmp_path / "notrigger.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"content": "hello"}})
        + "\n"
        + json.dumps({"type": "assistant", "message": {"content": "short"}}),
        encoding="utf-8",
    )

    result = run_capture(transcript, debug=False)
    assert result["status"] == "no_trigger"


def test_run_capture_same_session_concurrent_calls_extract_once(tmp_path: Path, monkeypatch):
    """The Stop load/check/extract/stamp cycle is exclusive per session."""
    _setup_env(tmp_path, monkeypatch)
    transcript = tmp_path / "same-session.jsonl"
    transcript.write_text("ignored\n", encoding="utf-8")
    monkeypatch.setattr(
        "memo.capture_hooks._read_recent_exchanges",
        lambda path, n: ("user", "decided to keep a durable concurrency invariant"),
    )
    monkeypatch.setattr("memo.capture_hooks._passes_prefilter", lambda text: True)

    class FakeMemory:
        def __init__(self, cfg):
            pass

        def close(self):
            pass

    monkeypatch.setattr("memo.memory.Memory", FakeMemory)

    start = threading.Barrier(3)
    entered = threading.Event()
    release = threading.Event()
    count_lock = threading.Lock()
    extract_calls = 0

    def fake_extract(*args, **kwargs):
        nonlocal extract_calls
        with count_lock:
            extract_calls += 1
            call_number = extract_calls
        if call_number == 1:
            entered.set()
            assert release.wait(timeout=3)
        return {"saved": [f"memory-{call_number}"], "save_failures": 0}

    monkeypatch.setattr("memo.capture_hooks._extract_and_save", fake_extract)

    def worker():
        start.wait(timeout=3)
        return run_capture(transcript)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker) for _ in range(2)]
        start.wait(timeout=3)
        assert entered.wait(timeout=3)
        done, _ = wait(futures, timeout=3, return_when=FIRST_COMPLETED)
        assert done
        release.set()
        results = [future.result(timeout=3) for future in futures]

    assert extract_calls == 1
    assert sorted(result["status"] for result in results) == ["locked", "ok"]


def test_run_capture_cooldown_is_per_session(tmp_path: Path, monkeypatch):
    """A recent save in session A must not throttle session B."""
    _setup_env(tmp_path, monkeypatch)
    monkeypatch.setenv("MEMO_CAPTURE_COOLDOWN_MIN", "30")
    transcripts = [tmp_path / "session-a.jsonl", tmp_path / "session-b.jsonl"]
    for transcript in transcripts:
        transcript.write_text("ignored\n", encoding="utf-8")
    monkeypatch.setattr(
        "memo.capture_hooks._read_recent_exchanges",
        lambda path, n: ("user", f"decided durable behavior for {path.stem}"),
    )
    monkeypatch.setattr("memo.capture_hooks._passes_prefilter", lambda text: True)

    class FakeMemory:
        def __init__(self, cfg):
            pass

        def close(self):
            pass

    monkeypatch.setattr("memo.memory.Memory", FakeMemory)
    monkeypatch.setattr(
        "memo.capture_hooks._extract_and_save",
        lambda *args, **kwargs: {"saved": ["memory-id"], "save_failures": 0},
    )

    assert run_capture(transcripts[0])["status"] == "ok"
    assert run_capture(transcripts[1])["status"] == "ok"


@pytest.mark.parametrize(
    ("legacy_session_id", "expected_status"),
    [(None, "ok"), ("other-session", "ok"), ("different-session", "cooldown")],
)
def test_run_capture_migrates_legacy_state_only_for_same_session(
    tmp_path: Path,
    monkeypatch,
    legacy_session_id: str | None,
    expected_status: str,
):
    """A matching legacy turn hash alone must never attribute session identity."""
    state_dir = _setup_env(tmp_path, monkeypatch)
    monkeypatch.setenv("MEMO_CAPTURE_COOLDOWN_MIN", "30")
    transcript = tmp_path / "different-session.jsonl"
    transcript.write_text("ignored\n", encoding="utf-8")
    assistant = "decided to keep legacy capture state isolated by session identity"
    legacy = {"last_hash": _hash_assistant(assistant), "last_save_ts": time.time()}
    if legacy_session_id is not None:
        legacy["session_id"] = legacy_session_id
    _save_state(state_dir, legacy)
    monkeypatch.setattr(
        "memo.capture_hooks._read_recent_exchanges",
        lambda path, n: ("user", assistant),
    )
    monkeypatch.setattr("memo.capture_hooks._passes_prefilter", lambda text: True)

    class FakeMemory:
        def __init__(self, cfg):
            pass

        def close(self):
            pass

    monkeypatch.setattr("memo.memory.Memory", FakeMemory)
    extract_calls = 0

    def extract_without_save(*args, **kwargs):
        nonlocal extract_calls
        extract_calls += 1
        return {"saved": [], "save_failures": 0}

    monkeypatch.setattr("memo.capture_hooks._extract_and_save", extract_without_save)

    result = run_capture(transcript)
    migrated = _load_state(state_dir, "different-session")

    assert result["status"] == expected_status
    if expected_status == "ok":
        assert extract_calls == 1
        assert "last_save_ts" not in migrated
    else:
        assert extract_calls == 0
        assert migrated["last_save_ts"] == legacy["last_save_ts"]


def test_run_capture_retries_after_save_failure(tmp_path: Path, monkeypatch):
    """A failed save does not stamp hash/cooldown; the next Stop retries."""
    _setup_env(tmp_path, monkeypatch)
    monkeypatch.setenv("MEMO_CAPTURE_COOLDOWN_MIN", "30")
    transcript = tmp_path / "retry-session.jsonl"
    transcript.write_text("ignored\n", encoding="utf-8")
    monkeypatch.setattr(
        "memo.capture_hooks._read_recent_exchanges",
        lambda path, n: ("user", "decided to retry transient capture write failures"),
    )
    monkeypatch.setattr("memo.capture_hooks._passes_prefilter", lambda text: True)

    class FakeMemory:
        def __init__(self, cfg):
            pass

        def close(self):
            pass

    monkeypatch.setattr("memo.memory.Memory", FakeMemory)
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {"saved": [], "save_failures": 1}
        return {"saved": ["memory-id"], "save_failures": 0}

    monkeypatch.setattr("memo.capture_hooks._extract_and_save", fail_once)

    first = run_capture(transcript)
    second = run_capture(transcript)

    assert first["status"] == "error"
    assert second["status"] == "ok"
    assert attempts == 2


def test_run_capture_reports_partial_without_stamping(tmp_path: Path, monkeypatch):
    """A mixed save result is explicit and remains retryable for dedup."""
    _setup_env(tmp_path, monkeypatch)
    transcript = tmp_path / "partial-session.jsonl"
    transcript.write_text("ignored\n", encoding="utf-8")
    monkeypatch.setattr(
        "memo.capture_hooks._read_recent_exchanges",
        lambda path, n: ("user", "decided to retry every partially saved capture batch"),
    )
    monkeypatch.setattr("memo.capture_hooks._passes_prefilter", lambda text: True)

    class FakeMemory:
        def __init__(self, cfg):
            pass

        def close(self):
            pass

    monkeypatch.setattr("memo.memory.Memory", FakeMemory)
    monkeypatch.setattr(
        "memo.capture_hooks._extract_and_save",
        lambda *args, **kwargs: {"saved": ["saved-before-failure"], "save_failures": 1},
    )

    assert run_capture(transcript)["status"] == "partial"
    assert run_capture(transcript)["status"] == "partial"


# ── Integration: run_capture_incremental ───────────────────────────────────


def test_run_capture_incremental_empty_transcript(tmp_path: Path, monkeypatch):
    """Empty transcript → no_pair status."""
    _setup_env(tmp_path, monkeypatch)
    transcript = tmp_path / "empty.jsonl"
    transcript.write_text("", encoding="utf-8")

    result = run_capture_incremental(transcript, "sess-123", debug=False)
    assert result["status"] == "no_pair"


def test_run_capture_incremental_no_new_turns(tmp_path: Path, monkeypatch):
    """If watermark is current, no_new status."""
    _setup_env(tmp_path, monkeypatch)

    # Create a transcript with 2 exchanges
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"content": "q1"}})
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": "decided to change config because we found a bug "
                    "in the reranker; the fix was to truncate text before ranking "
                    "and latency dropped three times in the hot path of the hook."
                },
            }
        )
        + "\n"
        + json.dumps({"type": "user", "message": {"content": "q2"}})
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": "decided we should use Qwen because it is faster "
                    "and more efficient for embeddings; we tested it on 1000 samples "
                    "and the latency improved by 50 percent on our warm cache tests."
                },
            }
        ),
        encoding="utf-8",
    )

    session_id = "sess-test"

    # Stub extraction so the watermark logic is exercised deterministically on
    # every backend. On Linux/CPU there is no MLX chat, so the real extractor
    # errors — which (correctly, since the fail-closed capture fix) now yields
    # status "error" instead of the old silent swallow. This test is about the
    # no-new-turns watermark, not real LLM extraction.
    monkeypatch.setattr(
        "memo.capture_hooks._extract_and_save",
        lambda *a, **k: {"saved": ["m1"], "save_failures": 0},
    )

    # First pass — should process both exchanges
    result1 = run_capture_incremental(transcript, session_id, debug=False)
    assert result1["status"] == "ok"
    assert result1["exchange_count"] == 2
    assert result1["processed_turns"] == 2

    # Second pass — no new turns, should be no_new
    result2 = run_capture_incremental(transcript, session_id, debug=False)
    assert result2["status"] == "no_new"
    assert result2["exchange_count"] == 2


@pytest.mark.parametrize(
    ("failed_saved", "expected_status"),
    [([], "error"), (["saved-before-failure"], "partial")],
)
def test_run_capture_incremental_save_failure_does_not_advance_watermark_and_retries(
    tmp_path: Path,
    monkeypatch,
    failed_saved: list[str],
    expected_status: str,
):
    """Failed incremental batches remain retryable from the prior watermark."""
    state_dir = _setup_env(tmp_path, monkeypatch)
    transcript = tmp_path / "retryable.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"content": "q1"}})
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": "decided to keep failed incremental capture batches "
                    "retryable because advancing their watermark would lose memories"
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("memo.capture_core._passes_prefilter", lambda text: True)

    class FakeMemory:
        def __init__(self, cfg):
            pass

        def close(self):
            pass

    monkeypatch.setattr("memo.memory.Memory", FakeMemory)
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {"saved": failed_saved, "save_failures": 1}
        return {"saved": ["memory-id"], "save_failures": 0}

    monkeypatch.setattr("memo.capture_hooks._extract_and_save", fail_once)
    session_id = "sess-retryable"
    clock = [1_000.0]
    monkeypatch.setattr("memo.capture_hooks.time.time", lambda: clock[0])

    first = run_capture_incremental(transcript, session_id)

    assert first["status"] == expected_status
    assert first["processed_turns"] == 1
    assert _load_watermark(state_dir, session_id) == {}

    throttled = run_capture_incremental(transcript, session_id)
    assert throttled["status"] == "backoff"
    assert attempts == 1

    clock[0] += 61.0
    second = run_capture_incremental(transcript, session_id)

    assert second["status"] == "ok"
    assert second["processed_turns"] == 1
    assert attempts == 2
    assert _load_watermark(state_dir, session_id)["exchange_count"] == 1


# ── Sidecar cleanup (bounded, throttled) ────────────────────────────────────


def test_prune_stale_sidecars_removes_old_keeps_recent(tmp_path: Path):
    """Dead-session `*.json`/`*.lock` past the TTL are pruned; recent files and
    non-matching files are left alone, and the throttle marker is stamped."""
    import os

    from memo.capture_hooks import _SIDECAR_TTL_S, _prune_stale_sidecars

    d = tmp_path / ".capture_stop"
    d.mkdir()
    old_json = d / "dead-session.json"
    old_lock = d / "dead-session.lock"
    fresh_json = d / "live-session.json"
    other = d / "keep.txt"  # non-matching suffix → never touched
    for p in (old_json, old_lock, fresh_json, other):
        p.write_text("{}", encoding="utf-8")
    old = time.time() - _SIDECAR_TTL_S - 3600
    os.utime(old_json, (old, old))
    os.utime(old_lock, (old, old))

    _prune_stale_sidecars(d)

    assert not old_json.exists()  # past TTL → pruned
    assert not old_lock.exists()  # past TTL → pruned
    assert fresh_json.exists()  # recent → kept (never unlink a live session's file)
    assert other.exists()  # wrong suffix → untouched
    assert (d / ".pruned").exists()  # throttle marker stamped


def test_prune_stale_sidecars_is_throttled(tmp_path: Path):
    """A recent `.pruned` marker skips the scan entirely (once/day)."""
    import os

    from memo.capture_hooks import _SIDECAR_TTL_S, _prune_stale_sidecars

    d = tmp_path / ".capture_watermark"
    d.mkdir()
    (d / ".pruned").touch()  # fresh marker → this call must not scan
    dead = d / "dead.json"
    dead.write_text("{}", encoding="utf-8")
    old = time.time() - _SIDECAR_TTL_S - 3600
    os.utime(dead, (old, old))

    _prune_stale_sidecars(d)

    assert dead.exists()  # throttled — not scanned this call
