"""Capture hooks: Stop hook + incremental + watermark management tests.

Tests the hook infrastructure (Group 6):
- run_capture() — Stop hook entry point
- run_capture_incremental() — Incremental capture entry
- Watermark state management (load, save, tick_due)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

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

    # First pass — should process both exchanges
    result1 = run_capture_incremental(transcript, session_id, debug=False)
    assert result1["status"] == "ok"
    assert result1["exchange_count"] == 2
    assert result1["processed_turns"] == 2

    # Second pass — no new turns, should be no_new
    result2 = run_capture_incremental(transcript, session_id, debug=False)
    assert result2["status"] == "no_new"
    assert result2["exchange_count"] == 2
