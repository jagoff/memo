"""Tests for session-level recall dedup helpers (get_recalled_ids / mark_ids_recalled).

Also covers an integration test verifying that the subprocess recall-hook path
emits a short reference line on the second turn when a memory was already
injected on the first turn.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memo.session import get_recalled_ids, mark_ids_recalled


# ---------------------------------------------------------------------------
# Unit tests for get_recalled_ids / mark_ids_recalled
# ---------------------------------------------------------------------------


def test_get_recalled_ids_empty(tmp_path: Path) -> None:
    """New session (no file) returns empty dict."""
    result = get_recalled_ids(tmp_path, "nonexistent-session-xyz")
    assert result == {}


def test_mark_and_get_recalled_ids(tmp_path: Path) -> None:
    """mark_ids_recalled persists, get_recalled_ids reads back correctly."""
    sid = "test-session-abc"
    mark_ids_recalled(tmp_path, sid, {"aabbccdd1122": 1, "eeff00112233": 1})
    result = get_recalled_ids(tmp_path, sid)
    assert result == {"aabbccdd1122": 1, "eeff00112233": 1}


def test_mark_ids_recalled_is_idempotent(tmp_path: Path) -> None:
    """Marking the same ID twice keeps the first turn value."""
    sid = "test-session-idem"
    mark_ids_recalled(tmp_path, sid, {"aabbccddee11": 1})
    mark_ids_recalled(tmp_path, sid, {"aabbccddee11": 5})  # should not overwrite
    result = get_recalled_ids(tmp_path, sid)
    assert result["aabbccddee11"] == 1  # first-seen turn preserved


def test_mark_ids_recalled_swallows_exceptions(tmp_path: Path) -> None:
    """Bad state_dir (file instead of dir) must not raise."""
    bad_state = tmp_path / "file.txt"
    bad_state.write_text("not a directory")
    # Should not raise even though state_dir is a file, not a dir.
    mark_ids_recalled(bad_state, "any-sid", {"id123": 1})  # no exception


def test_mark_ids_recalled_empty_noop(tmp_path: Path) -> None:
    """Empty new_ids dict returns immediately without touching the filesystem."""
    sid = "test-session-noop"
    sessions_dir = tmp_path / "sessions"
    mark_ids_recalled(tmp_path, sid, {})
    # No session file should have been created for a noop call.
    assert not (sessions_dir / f"{sid}.json").exists()


def test_mark_ids_recalled_merges_with_existing(tmp_path: Path) -> None:
    """New IDs are merged into existing recalled_ids without dropping old ones."""
    sid = "test-session-merge"
    mark_ids_recalled(tmp_path, sid, {"id1111111111": 1})
    mark_ids_recalled(tmp_path, sid, {"id2222222222": 2})
    result = get_recalled_ids(tmp_path, sid)
    assert result == {"id1111111111": 1, "id2222222222": 2}


# ---------------------------------------------------------------------------
# Integration test: subprocess recall-hook dedup path
# ---------------------------------------------------------------------------


def _make_hit(id_: str, title: str, body: str = "body text here") -> dict:
    return {
        "id": id_,
        "title": title,
        "type": "note",
        "tags": [],
        "created": "2026-01-01T00:00:00+00:00",
        "updated": "2026-01-01T00:00:00+00:00",
        "body": body,
        "extra": {},
        "score": 0.75,
        "path": f"notes/{id_}.md",
    }


def test_subprocess_path_dedup_short_ref_on_second_turn(
    tmp_cfg, monkeypatch
) -> None:
    """Second-turn call for the same session/memory emits the short reference line."""
    from click.testing import CliRunner

    from memo.cli_recall_hook import recall_hook
    from memo.memory import MemoryRecord

    hit_id = "aabbccdd11223344"
    hit = MemoryRecord(
        id=hit_id,
        path="notes/test.md",
        title="Test Memory",
        type="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body="This is the full body of the test memory, enough chars to matter.",
        extra={},
        score=0.80,
    )

    sid = "test-dedup-session-001"
    env = {
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_RECALL_DISABLE": "0",
        "MEMO_RECALL_TOKEN_BUDGET": "0",  # off so budget doesn't interfere
        "MEMO_RECALL_MIN_BODY_CHARS": "0",
        "MEMO_RECALL_MIN_SIM": "0.0",
        "MEMO_RECALL_SKIP_BELOW": "0.0",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_RECALL_EXPAND_CONTEXT": "0",
        "MEMO_RECALL_ADAPTIVE_CONTEXT": "0",
        "MEMO_RECALL_CONTEXTUAL": "0",
        "MEMO_RECALL_FEEDBACK_HINT": "0",
        "MEMO_RECALL_DIRECTIVE_ONCE": "0",
    }

    # Stub out the subprocess recall path so we control the hits returned.
    # Memory is imported inside the function via `from memo.memory import Memory`,
    # so we patch the class on the source module.
    class StubMemory:
        def __init__(self, cfg):  # noqa: ARG002
            pass

        def search(self, query, limit=5, mode="bm25", recency=False, exclude_types=None):
            return [hit]

        def close(self):
            pass

    monkeypatch.setattr("memo.memory.Memory", StubMemory)

    # Also stub dedup_hits so it's a no-op pass-through.
    monkeypatch.setattr("memo.recall_server.dedup_hits", lambda hits: hits)

    payload_first = json.dumps({
        "prompt": "what do you know about this topic",
        "session_id": sid,
        "cwd": str(tmp_cfg.data_dir),
    })
    payload_second = json.dumps({
        "prompt": "what do you know about this topic",
        "session_id": sid,
        "cwd": str(tmp_cfg.data_dir),
    })

    runner = CliRunner()

    # --- First turn: full body should appear ---
    result1 = runner.invoke(recall_hook, input=payload_first, env=env, catch_exceptions=False)
    assert result1.exit_code == 0, result1.output
    out1 = json.loads(result1.output)
    context1 = out1["hookSpecificOutput"]["additionalContext"]
    assert "Test Memory" in context1
    # Full body is present on first turn
    assert "full body" in context1

    # --- Second turn: short reference line should appear instead of full body ---
    result2 = runner.invoke(recall_hook, input=payload_second, env=env, catch_exceptions=False)
    assert result2.exit_code == 0, result2.output
    out2 = json.loads(result2.output)
    context2 = out2["hookSpecificOutput"]["additionalContext"]
    assert "Test Memory" in context2
    # Short reference line with "ya citado" — no full body
    assert "ya citado" in context2
    assert "full body" not in context2
