"""Tests for Tier 1 safety fixes (1.3 / 1.4 / 1.5).

Fix 1.3 — write_ops: user-facing WARNING when embed fails and record is
           marked embed-pending (disk-is-truth path).
Fix 1.4 — config: WARNING logged when state_dir is not writable and a
           transient device_id is returned.
Fix 1.5 — search_ops: DEBUG log when source-feedback boost is skipped
           because the query embedding is unavailable (bm25/fuzzy mode).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

# ── Fix 1.3 ─────────────────────────────────────────────────────────────────


def test_save_embed_pending_logs_warning(mock_memory, caplog):
    """When embedding fails, _save_index_pending must emit a user-facing
    WARNING that includes the record id and tells the user to run
    'memo reindex'."""
    # Make the embedder fail on every call.
    mock_memory.embedder.embed = MagicMock(side_effect=RuntimeError("embedder down"))
    mock_memory.embedder.embed_query = MagicMock(side_effect=RuntimeError("embedder down"))

    with caplog.at_level(logging.WARNING):
        rec = mock_memory.save(content="hello world", title="Embed Fail Test")

    # The record is still returned (disk-is-truth).
    assert rec is not None
    assert rec.id is not None
    # The embed-pending marker must be set.
    assert rec.extra.get("_memo_embed_pending") is True

    # The user-facing warning must be present.
    messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    user_msg = next(
        (m for m in messages if "memo reindex" in m),
        None,
    )
    assert user_msg is not None, f"Expected a WARNING containing 'memo reindex', got: {messages}"
    # The record id (8-char prefix) should appear in the message.
    assert rec.id[:8] in user_msg


# ── Fix 1.4 ─────────────────────────────────────────────────────────────────


def test_device_id_transient_logs_warning(tmp_path, caplog):
    """When state_dir is not writable, Config.device_id should log a
    WARNING mentioning 'transient' and 'not writable'."""
    from memo.config import Config

    data = tmp_path / "data"
    data.mkdir()
    state = tmp_path / "state"
    state.mkdir()

    cfg = Config(data_dir=data, state_dir=state)

    # Patch Path.mkdir and Path.write_text on the id_path to simulate a
    # write failure while keeping the directory itself reachable.
    with (
        patch("pathlib.Path.write_text", side_effect=OSError("read-only fs")),
        caplog.at_level(logging.WARNING, logger="memo.config"),
    ):
        device_id = cfg.device_id

    assert device_id.startswith("transient-")
    warning_messages = [
        r.message
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "memo.config"
    ]
    assert any("transient" in m for m in warning_messages), (
        f"Expected a warning about transient device_id, got: {warning_messages}"
    )
    assert any("not writable" in m for m in warning_messages), (
        f"Expected a warning mentioning 'not writable', got: {warning_messages}"
    )


# ── Fix 1.5 ─────────────────────────────────────────────────────────────────


def test_source_feedback_skipped_logs_debug_in_bm25_mode(mock_memory, caplog):
    """In bm25 mode, no query embedding is computed, so feedback boost is
    skipped. The fix must emit a DEBUG log mentioning mode=bm25."""
    mock_memory.save(content="alpha beta gamma", title="BM25 Feedback Test", tags=["t"])

    with caplog.at_level(logging.DEBUG):
        results = mock_memory.search("alpha", mode="bm25", load_bodies=False)

    assert results  # sanity: bm25 still returns results

    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    skipped_msg = next(
        (m for m in debug_messages if "feedback" in m.lower() and "skipped" in m.lower()),
        None,
    )
    assert skipped_msg is not None, (
        f"Expected a DEBUG message about feedback boost being skipped, got: {debug_messages}"
    )
    assert "bm25" in skipped_msg.lower(), (
        f"Expected mode=bm25 in the skipped message, got: {skipped_msg}"
    )
