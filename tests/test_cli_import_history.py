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
