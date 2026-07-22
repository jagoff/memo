"""Save-path defaults for record-level bi-temporal validity (Task 3).

A save with no explicit ``valid_at`` defaults the world-validity start to the
record's learned-time ``created``; ``invalid_at`` stays open (None). Both mirror
into the on-disk markdown frontmatter (markdown stays source of truth). An
explicit ``valid_at`` is never overwritten by the default.
"""

from __future__ import annotations

from memo.memory import Memory


def test_save_defaults_valid_at_to_created(mock_memory: Memory) -> None:
    rec = mock_memory.save(content="prod db is postgres", type_="fact")
    assert rec.valid_at == rec.created
    assert rec.invalid_at is None
    # Frontmatter mirror: valid_at lands on disk alongside created/updated.
    md = (mock_memory.cfg.memory_dir / rec.path).read_text(encoding="utf-8")
    assert "valid_at:" in md


def test_save_does_not_override_explicit_valid_at(mock_memory: Memory) -> None:
    rec = mock_memory.save(
        content="prod db is postgres",
        type_="fact",
        valid_at="2020-01-01T00:00:00",
    )
    assert rec.valid_at == "2020-01-01T00:00:00"
    assert rec.valid_at != rec.created
    assert rec.invalid_at is None
    md = (mock_memory.cfg.memory_dir / rec.path).read_text(encoding="utf-8")
    assert "valid_at: '2020-01-01T00:00:00'" in md or "valid_at: 2020-01-01T00:00:00" in md
