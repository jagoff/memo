"""P4 cross-client inevitability: mandate writer + expanded silent-gap set."""
from __future__ import annotations

from pathlib import Path

from memo import dashboard
from memo.cli_mandate import _MARKER, MANDATE_TEXT, _write_mandate, write_mandates_for_clients


def test_expected_consumers_covers_non_hook_clients() -> None:
    # Only always-on daemons/hooks belong in EXPECTED_CONSUMERS — they are
    # flagged "silent" when absent. claude-code (recall-hook) + synapse/memflow
    # (always-on services) + codex qualify.
    for c in ("claude-code", "synapse", "memflow", "codex"):
        assert c in dashboard.EXPECTED_CONSUMERS
    # On-demand tools the user invokes explicitly (not continuous daemons) must
    # NOT be flagged silent when idle: devin/opencode/devin-desktop appear as
    # readers if/when they query, but absence isn't a gap.
    for c in ("devin", "opencode", "devin-desktop"):
        assert c not in dashboard.EXPECTED_CONSUMERS


def test_write_mandate_creates_and_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    assert _write_mandate(target, dry_run=False) == "written"
    assert _MARKER in target.read_text(encoding="utf-8")
    # second write → idempotent skip, no duplication
    assert _write_mandate(target, dry_run=False) == "already present (skip)"
    assert target.read_text(encoding="utf-8").count(_MARKER) == 1


def test_write_mandate_appends_to_existing(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text("# Existing project rules\n\nstuff\n", encoding="utf-8")
    _write_mandate(target, dry_run=False)
    body = target.read_text(encoding="utf-8")
    assert "Existing project rules" in body  # preserved
    assert _MARKER in body


def test_write_mandate_nested_path(tmp_path: Path) -> None:
    target = tmp_path / ".cursor" / "rules" / "memo.md"
    assert _write_mandate(target, dry_run=False) == "written"
    assert target.is_file()


def test_write_mandate_dry_run_no_write(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    assert _write_mandate(target, dry_run=True) == "would write"
    assert not target.exists()


def test_mandate_text_mentions_source_attribution() -> None:
    assert "source=" in MANDATE_TEXT
    assert "memo_unified_briefing" in MANDATE_TEXT


def test_write_mandates_for_clients_deduplicates_shared_files(tmp_path: Path) -> None:
    results = write_mandates_for_clients(
        ["devin", "opencode", "devin-desktop"], cwd=tmp_path, dry_run=False
    )
    assert results[0][0] == "AGENTS.md"
    assert len([path for path, _status in results if path == "AGENTS.md"]) == 1
    assert all(path != ".cursor/rules/memo.md" for path, _status in results)
