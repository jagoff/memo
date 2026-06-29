"""F6+ — `memo sync bootstrap`: clone the memo-sync repo on a new machine and
point config.toml's data_dir at it in one step.

Covers the pure git+config orchestration (`bootstrap_clone`). No MLX: reindex /
import-signal are the caller's job and exercised in test_sync_git.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from memo.setup.config_io import load_config_file
from memo.sync_git import SyncGitError, bootstrap_clone


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """A bare memo-sync remote seeded with a memorias/ dir and one memoria."""
    r = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(r)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(r), str(seed)], check=True, capture_output=True)
    _git(seed, "config", "user.email", "t@t.t")
    _git(seed, "config", "user.name", "t")
    (seed / "memorias").mkdir()
    (seed / "memorias" / "2026-01-01-hello.md").write_text(
        "---\n"
        "id: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "title: hello\n"
        "type: note\n"
        "created: '2026-01-01T00:00:00+00:00'\n"
        "updated: '2026-01-01T00:00:00+00:00'\n"
        "---\n\n"
        "body\n"
    )
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "push", "origin", "main")
    return r


def test_bootstrap_clone_fresh(remote: Path, tmp_path: Path):
    dest = tmp_path / "memo-sync"
    cfg_path = tmp_path / "config.toml"

    out = bootstrap_clone(str(remote), dest, config_path=cfg_path)

    assert out["reused"] is False
    assert out["memories_dir"] == str(dest / "memorias")
    assert out["memories"] == 1
    # config.toml now points data_dir at the cloned memorias
    storage = (load_config_file(cfg_path) or {})["storage"]
    assert storage["data_dir"] == str(dest / "memorias")


def test_bootstrap_clone_reuse_existing(remote: Path, tmp_path: Path):
    dest = tmp_path / "memo-sync"
    cfg_path = tmp_path / "config.toml"
    bootstrap_clone(str(remote), dest, config_path=cfg_path)

    # second run: dest is already a valid clone → reuse, no error
    out = bootstrap_clone(str(remote), dest, config_path=cfg_path)
    assert out["reused"] is True
    assert out["memories_dir"] == str(dest / "memorias")
    storage = (load_config_file(cfg_path) or {})["storage"]
    assert storage["data_dir"] == str(dest / "memorias")


def test_bootstrap_clone_preserves_existing_storage_keys(remote: Path, tmp_path: Path):
    dest = tmp_path / "memo-sync"
    cfg_path = tmp_path / "config.toml"
    # pre-existing config with a vault_path + single_db the user set
    from memo.setup.config_io import write_config_file

    write_config_file(
        data_dir=tmp_path / "old",
        vault_path=tmp_path / "vault",
        single_db=True,
        path=cfg_path,
    )

    bootstrap_clone(str(remote), dest, config_path=cfg_path)

    storage = (load_config_file(cfg_path) or {})["storage"]
    assert storage["data_dir"] == str(dest / "memorias")  # repointed
    assert storage["vault_path"] == str(tmp_path / "vault")  # preserved
    assert storage["single_db"] is True  # preserved


def test_bootstrap_clone_rejects_nonempty_non_clone(tmp_path: Path):
    dest = tmp_path / "memo-sync"
    dest.mkdir()
    (dest / "junk.txt").write_text("x")  # non-empty, not a git clone
    with pytest.raises(SyncGitError, match="already exists and is not empty"):
        bootstrap_clone("file:///nonexistent", dest, config_path=tmp_path / "config.toml")


def test_bootstrap_clone_refuses_broken_clone_empty_memorias(remote: Path, tmp_path: Path):
    """A git clone whose memorias/ lost its .md must NOT be silently reused — the
    caller's `reindex --rebuild` would truncate the index against an empty disk and
    wipe it (the data-loss incident). Refuse with recovery guidance instead."""
    dest = tmp_path / "memo-sync"
    cfg_path = tmp_path / "config.toml"
    bootstrap_clone(str(remote), dest, config_path=cfg_path)
    # Simulate the incident: every .md gone but .git intact.
    for md in (dest / "memorias").rglob("*.md"):
        md.unlink()
    with pytest.raises(SyncGitError, match=r"no \.md"):
        bootstrap_clone(str(remote), dest, config_path=cfg_path)


def _stub_embed(self, inputs):
    out = []
    for s in inputs:
        v = [0.0] * 4
        v[sum(ord(c) for c in s) % 4] = 1.0
        out.append(v)
    return out


def test_sync_bootstrap_cli_end_to_end(remote: Path, tmp_path: Path, monkeypatch):
    """`memo sync bootstrap`: clone → repoint config → reindex → import-signal."""
    from click.testing import CliRunner

    from memo.cli import cli

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    dest = tmp_path / "memo-sync"
    env = {
        "MEMO_CONFIG_FILE": str(tmp_path / "config.toml"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
        "MEMO_RERANKER_ENABLED": "0",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }
    result = CliRunner().invoke(
        cli, ["sync", "bootstrap", str(remote), "--dest", str(dest), "--json"], env=env
    )

    assert result.exit_code == 0, result.output
    import json as _json

    out = _json.loads(result.output)
    assert out["reused"] is False
    assert out["memories_dir"] == str(dest / "memorias")
    assert out["reindexed"]["added"] == 1  # the seeded memoria got indexed
    # config.toml repointed at the clone
    storage = (load_config_file(tmp_path / "config.toml") or {})["storage"]
    assert storage["data_dir"] == str(dest / "memorias")
