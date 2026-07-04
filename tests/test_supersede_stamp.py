"""C6: supersede stamps superseded_by + close-date into the archived
loser's extra bag (inactive/<file>.md frontmatter)."""

from __future__ import annotations

import frontmatter


def test_archive_memory_stamps_superseded_by(mock_memory):
    loser = mock_memory.save(content="El puerto es 8080", title="Puerto (viejo)")
    winner = mock_memory.save(content="El puerto es 8765", title="Puerto (nuevo)")

    ok = mock_memory.lifecycle.archive_memory(loser.id, superseded_by=winner.id)
    assert ok

    inactive = mock_memory.cfg.memory_dir / "inactive"
    files = list(inactive.glob("*.md"))
    assert len(files) == 1
    post = frontmatter.loads(files[0].read_text(encoding="utf-8"))
    extra = post.get("extra") or {}
    assert extra["superseded_by"] == winner.id
    assert extra["superseded_at"]  # ISO close-date present


def test_archive_memory_without_kwarg_stays_unstamped(mock_memory):
    rec = mock_memory.save(content="dato viejo", title="Stale note")
    assert mock_memory.lifecycle.archive_memory(rec.id)
    files = list((mock_memory.cfg.memory_dir / "inactive").glob("*.md"))
    post = frontmatter.loads(files[0].read_text(encoding="utf-8"))
    extra = post.get("extra") or {}
    assert "superseded_by" not in extra
