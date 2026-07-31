"""Tests for vault re-ingestion orchestration (ported from synapse)."""

from pathlib import Path

from memo.vault_ingest import (
    _FIXED_VAULT_EXCLUDES,
    build_ingest_command,
    vault_label,
    vault_paths,
)


def test_vault_label():
    assert vault_label(Path("/x/Notes")) == "notes"
    assert vault_label(Path("/x/obsidian-work")) == "work"
    assert vault_label(Path("/x/obsidian-")) == "vault"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMO_VAULT_PATHS", f"{tmp_path}/v1, {tmp_path}/v2")
    assert vault_paths() == [tmp_path / "v1", tmp_path / "v2"]


def test_default_paths_only_existing_dirs(monkeypatch):
    monkeypatch.delenv("MEMO_VAULT_PATHS", raising=False)
    for p in vault_paths():
        assert p.is_dir()


def test_build_ingest_command():
    cmd = build_ingest_command("/bin/memo", Path("/v/Notes"), "notes", ["a/**", "b.md"])
    assert cmd == [
        "/bin/memo", "ingest", "/v/Notes", "--name", "notes", "--prune",
        "--exclude", "a/**", "--exclude", "b.md",
    ]


def test_fixed_excludes_present():
    for glob in ("Obsidian/Whatsapp/**", "Obsidian/AI/**", "04-Archive/**", "Archive/**", "archive/**"):
        assert glob in _FIXED_VAULT_EXCLUDES
