"""Regression: a memory deleted while reindex walks the directory is a benign
race, not a parse error.

Found running the live sweep: `watch.err.log` had grown to 15 MB, almost
entirely

    reindex: skipping <name>.md (parse error): [Errno 2] No such file or
    directory: '.../<name>.md'

`memo watch` reindexes on every change, so a file the nightly GC or a
consolidation merge had just removed was still in the glob when the read ran.
Calling that a "parse error" at warning level both misnames the cause and
floods the log for an entirely expected sequence.
"""

from __future__ import annotations

import logging

import pytest


def test_a_vanished_file_is_not_reported_as_a_parse_error(mock_memory, caplog, monkeypatch) -> None:
    """The file is listed by the walk and deleted before the read — exactly
    what `memo watch` hits when the GC removes a memory mid-reindex."""
    from pathlib import Path

    record = mock_memory.save(content="body", title="about to vanish", type_="note")
    doomed = (mock_memory.cfg.memory_dir / record.path).resolve()
    original_read_text = Path.read_text

    def vanishing_read_text(self, *args, **kwargs):
        if self.resolve() == doomed:
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", vanishing_read_text)

    with caplog.at_level(logging.DEBUG, logger="memo.memory.maintain_ops"):
        mock_memory.reindex()

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not [w for w in warnings if "parse error" in w], warnings


def test_a_real_parse_error_is_still_reported(mock_memory, caplog) -> None:
    mock_memory.save(content="body", title="valid one", type_="note")
    broken = mock_memory.cfg.memory_dir / "broken.md"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"---\n: : not: valid: yaml: [\n---\n\nbody\n")

    with caplog.at_level(logging.WARNING, logger="memo.memory.maintain_ops"):
        mock_memory.reindex()

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert [w for w in warnings if "broken.md" in w], warnings


def test_rebuild_preflight_still_fails_loudly_on_a_real_parse_error(mock_memory) -> None:
    """A corrupt file must abort a --rebuild: the derived index would silently
    lose that memory otherwise."""
    from memo.errors import StorageError

    mock_memory.save(content="body", title="valid one", type_="note")
    broken = mock_memory.cfg.memory_dir / "broken.md"
    broken.write_bytes(b"---\n: : not: valid: yaml: [\n---\n\nbody\n")

    with pytest.raises(StorageError, match="preflight"):
        mock_memory.reindex(rebuild=True)
