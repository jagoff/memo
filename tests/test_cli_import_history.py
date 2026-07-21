"""CLI wiring — `memo import codex|opencode|chatgpt|claude-export|mem0|zep`."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_import_codex_renders_summary_and_passes_args(tmp_path: Path, monkeypatch):
    seen: dict = {}

    def _fake(root=None, since_days=None, file_limit=None, dry_run=False):
        seen.update(root=root, since_days=since_days, file_limit=file_limit, dry_run=dry_run)
        return {
            "status": "ok",
            "root": "/x",
            "files_total": 2,
            "files_processed": 1,
            "files_skipped": 1,
            "candidates": 3,
            "saved": ["abc123"],
            "skipped_dup": 2,
            "dry_run": False,
        }

    monkeypatch.setattr("memo.history_importers.run_codex_import", _fake)

    res = CliRunner().invoke(cli, ["import", "codex", "--since", "30"], env=_env(tmp_path))

    assert res.exit_code == 0, res.output
    assert "Saved: 1" in res.output
    assert seen["since_days"] == 30


def test_import_chatgpt_dry_run_json(tmp_path: Path, monkeypatch):
    export = tmp_path / "conversations.json"
    export.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        "memo.history_importers.run_file_import",
        lambda exchanges, dry_run=False, source_name="": {
            "status": "ok",
            "dry_run": dry_run,
            "candidates": 0,
            "saved": [],
            "skipped_dup": 0,
        },
    )

    res = CliRunner().invoke(
        cli, ["import", "chatgpt", str(export), "--dry-run", "--json"], env=_env(tmp_path)
    )

    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["dry_run"] is True


def test_import_opencode_and_claude_export_registered(tmp_path: Path):
    res = CliRunner().invoke(cli, ["import", "--help"], env=_env(tmp_path))
    assert res.exit_code == 0
    for cmd in ("codex", "opencode", "chatgpt", "claude-export", "mem0", "zep"):
        assert cmd in res.output


def _stub_embed(self, inputs):
    out = []
    for s in inputs:
        h = sum(ord(c) for c in (s or "")) % 4
        v = [0.0] * 4
        v[h] = 1.0
        out.append(v)
    return out


def test_import_mem0_end_to_end(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    monkeypatch.setattr("memo.embedder.MLXEmbedder.__init__", lambda self, **kw: None)
    dump = tmp_path / "mem0.json"
    dump.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "memory": "prefers dark mode",
                        "categories": ["prefs"],
                        "created_at": "2025-11-02T10:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    env = dict(_env(tmp_path), MEMO_EMBEDDER_DIMS="4")

    res = CliRunner().invoke(cli, ["import", "mem0", str(dump)], env=env)

    assert res.exit_code == 0, res.output
    assert "Imported: 1" in res.output


# -- `memo import json|csv|markdown-bundle` failure surfacing --------------------


def test_import_json_all_records_failed_exits_nonzero_with_error_sample(tmp_path: Path):
    """errors>0 must exit non-zero and print a sample of the collected
    per-record error messages. Regression: `memo import json bad.json && rm
    bad.json` reported 'Errors: 500', exited 0, and never showed why."""
    from unittest.mock import MagicMock, patch

    from memo.import_export import ImportResult

    export = tmp_path / "export.json"
    export.write_text("[]", encoding="utf-8")
    fake_result = ImportResult(
        imported_count=0,
        skipped_count=0,
        errors=[f"Record {i}: missing required field 'content'" for i in range(7)],
    )
    mock_mem = MagicMock()
    mock_mem.import_export.import_from.return_value = fake_result

    with patch("memo.cli_import._get_memory", return_value=mock_mem):
        res = CliRunner().invoke(cli, ["import", "json", str(export)], env=_env(tmp_path))

    assert res.exit_code == 1, res.output
    assert "Errors: 7" in res.output
    assert "Record 0: missing required field 'content'" in res.output
    assert "and 2 more" in res.output


def test_import_csv_partial_errors_exit_nonzero(tmp_path: Path):
    from unittest.mock import MagicMock, patch

    from memo.import_export import ImportResult

    export = tmp_path / "export.csv"
    export.write_text("title,content\n", encoding="utf-8")
    mock_mem = MagicMock()
    mock_mem.import_export.import_from.return_value = ImportResult(
        imported_count=3, skipped_count=0, errors=["Record 2: bad row"]
    )

    with patch("memo.cli_import._get_memory", return_value=mock_mem):
        res = CliRunner().invoke(cli, ["import", "csv", str(export)], env=_env(tmp_path))

    assert res.exit_code == 1, res.output
    assert "Imported: 3" in res.output
    assert "Record 2: bad row" in res.output


def test_import_json_clean_run_exits_zero(tmp_path: Path):
    from unittest.mock import MagicMock, patch

    from memo.import_export import ImportResult

    export = tmp_path / "export.json"
    export.write_text("[]", encoding="utf-8")
    mock_mem = MagicMock()
    mock_mem.import_export.import_from.return_value = ImportResult(
        imported_count=2, skipped_count=1, errors=[]
    )

    with patch("memo.cli_import._get_memory", return_value=mock_mem):
        res = CliRunner().invoke(cli, ["import", "json", str(export)], env=_env(tmp_path))

    assert res.exit_code == 0, res.output


def test_import_json_missing_path_is_usage_error(tmp_path: Path):
    """click.Path(exists=True) rejects a nonexistent input before any work."""
    res = CliRunner().invoke(
        cli, ["import", "json", str(tmp_path / "nope.json")], env=_env(tmp_path)
    )
    assert res.exit_code == 2, res.output
    assert "does not exist" in res.output
