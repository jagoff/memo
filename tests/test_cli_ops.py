"""Tests for the `memo ops` command group."""

import json

from click.testing import CliRunner

import memo.cli_ops as cli_ops
from memo.cli_ops import ops_group


class _FakeRec:
    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


class _FakeMem:
    def __init__(self, records):
        self.records = records
        self.deleted = []

    def list(self, limit=20, type_=None):
        return [_FakeRec(r) for r in self.records]

    def delete(self, id_, *, actor=None):
        self.deleted.append(id_)
        return True


def _orphan(id_):
    return {
        "id": id_,
        "body": "x",
        "updated": "2026-01-01",
        "created": "2026-01-01",
        "extra": {"source": "vault-ingest:notes", "abs_path": "/definitely/gone.md"},
    }


def test_gc_vault_orphans_dry_run(monkeypatch):
    fake = _FakeMem([_orphan("a"), _orphan("b")])
    monkeypatch.setattr(cli_ops, "_get_memory", lambda cfg: fake)
    result = CliRunner().invoke(ops_group, ["gc-vault-orphans", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out == {"scanned": 2, "orphans": 2, "deleted": 0, "dry_run": True}
    assert fake.deleted == []


def test_gc_vault_orphans_deletes(monkeypatch):
    fake = _FakeMem([_orphan("a")])
    monkeypatch.setattr(cli_ops, "_get_memory", lambda cfg: fake)
    result = CliRunner().invoke(ops_group, ["gc-vault-orphans", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["deleted"] == 1
    assert fake.deleted == ["a"]


def test_gc_memo_duplicates_counts_dry_run(monkeypatch):
    recs = [
        {"id": "old", "body": "same", "updated": "2026-01-01", "created": "", "extra": {}},
        {"id": "new", "body": "same", "updated": "2026-02-01", "created": "", "extra": {}},
    ]
    fake = _FakeMem(recs)
    monkeypatch.setattr(cli_ops, "_get_memory", lambda cfg: fake)
    result = CliRunner().invoke(ops_group, ["gc-memo-duplicates", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    # Fidelity with the synapse original: dry-run counts would-be deletions.
    assert out == {"scanned": 2, "dup_groups": 1, "deleted": 1, "dry_run": True}
    assert fake.deleted == []


def test_exclude_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(ops_group, ["exclude", "add", "notes", "a/b.md"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(ops_group, ["exclude", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"notes": ["a/b.md"]}
    result = runner.invoke(ops_group, ["exclude", "remove", "notes", "a/b.md"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(ops_group, ["exclude", "list", "--json"])
    assert json.loads(result.output) == {}


def test_ops_registered_on_root_cli():
    from memo.cli import cli

    assert "ops" in cli.commands


def test_ops_install_chat_rejects_nonexistent_dist(tmp_path):
    # A bare str --dist means a relative path + launchd's cwd=/ silently
    # crash-loops under KeepAlive instead of failing fast at install time.
    result = CliRunner().invoke(ops_group, ["install", "chat", "--dist", str(tmp_path / "missing")])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_ops_install_chat_resolves_relative_dist_to_absolute_path(monkeypatch, tmp_path):
    import shutil

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    captured: dict = {}

    def fake_install_chat(memo_bin, home, *, port=8765, dist=None):
        captured["dist"] = dist
        return home / "Library" / "LaunchAgents" / "com.memo.chat.plist"

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/memo")
    monkeypatch.setattr("memo.ops_launchd.install_chat", fake_install_chat)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(ops_group, ["install", "chat", "--dist", "dist"])
    assert result.exit_code == 0, result.output
    assert captured["dist"] == str(dist_dir.resolve())


def test_ops_install_rejects_unknown_service():
    result = CliRunner().invoke(ops_group, ["install", "bogus-service"])
    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_ops_uninstall_rejects_unknown_service():
    result = CliRunner().invoke(ops_group, ["uninstall", "bogus-service"])
    assert result.exit_code != 0
    assert "Invalid value" in result.output
