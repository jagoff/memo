"""`memo export` destination handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from memo import cli_export
from memo.import_export import ExportResult


class _StubImportExport:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def export_to(self, output_path: Path, format: str) -> ExportResult:
        self.calls.append((output_path, format))
        output_path.write_text("[]", encoding="utf-8")
        return ExportResult(exported_count=2, output_path=str(output_path), format=format)


class _StubMemory:
    def __init__(self) -> None:
        self.import_export = _StubImportExport()


@pytest.mark.parametrize(
    "command",
    ["json", "passport", "csv", "markdown-bundle"],
)
def test_export_missing_parent_directory_is_a_clean_error(command: str, tmp_path: Path) -> None:
    missing_parent = tmp_path / "exports"
    target = missing_parent / "brain.json"

    result = CliRunner().invoke(cli_export.export_group, [command, str(target)])

    assert result.exit_code == 1
    assert not isinstance(result.exception, OSError)
    assert "output directory does not exist" in result.output
    # A typo in the destination must not silently materialize the tree it names.
    assert not missing_parent.exists()


def test_export_rejects_an_existing_directory_as_the_destination(tmp_path: Path) -> None:
    target = tmp_path / "already-a-dir"
    target.mkdir()

    result = CliRunner().invoke(cli_export.export_group, ["json", str(target)])

    assert result.exit_code == 1
    assert not isinstance(result.exception, OSError)
    assert "not a file" in result.output


def test_export_json_writes_when_the_parent_exists(monkeypatch, tmp_path: Path) -> None:
    memory = _StubMemory()
    monkeypatch.setattr(cli_export, "_get_memory", lambda _cfg: memory)
    target = tmp_path / "brain.json"

    result = CliRunner().invoke(cli_export.export_group, ["json", str(target)])

    assert result.exit_code == 0, result.output
    assert memory.import_export.calls == [(target, "json")]
    assert target.read_text(encoding="utf-8") == "[]"
