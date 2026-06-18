"""P4 cross-client inevitability: mandate writer + expanded silent-gap set."""
from __future__ import annotations

from pathlib import Path

from memo import dashboard
from memo.cli_mandate import _MARKER, MANDATE_TEXT, _write_mandate, write_mandates_for_clients


def test_expected_consumers_covers_non_hook_clients() -> None:
    for c in ("claude-code", "synapse", "memflow", "codex", "devin", "opencode"):
        assert c in dashboard.EXPECTED_CONSUMERS
    # "windsurf" retired (now Devin Desktop); devin-desktop is a GUI app that
    # can't be driven headless, so it's not flagged as a silent gap.
    assert "windsurf" not in dashboard.EXPECTED_CONSUMERS
    assert "devin-desktop" not in dashboard.EXPECTED_CONSUMERS


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
    assert "memory_unified_briefing" in MANDATE_TEXT


def test_write_mandates_for_clients_deduplicates_shared_files(tmp_path: Path) -> None:
    results = write_mandates_for_clients(["devin", "opencode", "windsurf"], cwd=tmp_path, dry_run=False)
    assert results[0][0] == "AGENTS.md"
    assert len([path for path, _status in results if path == "AGENTS.md"]) == 1
    assert any(path == ".windsurfrules" for path, _status in results)
