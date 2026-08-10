"""`sync init` must not stage memo's own derived state.

``sync_init_home`` / ``sync_init_home_byo`` run ``git add -A`` at
``cfg.memory_dir.parent``. When ``state_dir`` lives under that root — which it
does whenever both are pointed at one directory, the shape every isolated test
store and scratch install uses — that sweep reaches the LIVE index: the Tantivy
segment directory and the sqlite WAL/SHM sidecars.

Two consequences, one of them observed. On 2026-08-09 a plain ``memo sync
init`` against such a store died with::

    git add -A failed: fatal: unable to stat 'state/tantivy/.tmp7nQlgN':
    No such file or directory

— Tantivy's writer had already replaced its temp file between git's readdir and
its stat. The quieter one is worse: a successful run would commit a derived,
machine-local index into the cross-machine sync repo, against memo's own
contract that markdown is the source of truth and the index is rebuildable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from memo.config import Config
from memo.sync_git import ensure_sync_gitignore


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def sync_root(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    (root / "memorias").mkdir(parents=True)
    (root / "state" / "tantivy").mkdir(parents=True)
    (root / "state" / "tantivy" / ".tmpAbCdEf").write_text("scratch", encoding="utf-8")
    (root / "state" / "memvec.db").write_text("index", encoding="utf-8")
    (root / "state" / "memvec.db-wal").write_text("wal", encoding="utf-8")
    (root / "memorias" / "note.md").write_text("# a real memory", encoding="utf-8")
    _git(root, "init", "-b", "main")
    return root


def test_derived_state_is_not_staged(sync_root: Path) -> None:
    ensure_sync_gitignore(sync_root)

    _git(sync_root, "add", "-A")
    staged = _git(sync_root, "diff", "--cached", "--name-only").splitlines()

    assert "memorias/note.md" in staged
    assert not [p for p in staged if p.startswith("state/tantivy/")]
    assert not [p for p in staged if p.endswith(("-wal", "-shm"))]


def test_the_markdown_corpus_is_still_staged(sync_root: Path) -> None:
    """The ignore must be surgical — memories are the whole point of the repo."""
    ensure_sync_gitignore(sync_root)

    _git(sync_root, "add", "-A")
    staged = _git(sync_root, "diff", "--cached", "--name-only").splitlines()

    assert "memorias/note.md" in staged


def test_it_is_idempotent(sync_root: Path) -> None:
    ensure_sync_gitignore(sync_root)
    first = (sync_root / ".gitignore").read_text(encoding="utf-8")
    ensure_sync_gitignore(sync_root)

    assert (sync_root / ".gitignore").read_text(encoding="utf-8") == first


def test_it_preserves_a_hand_written_gitignore(sync_root: Path) -> None:
    (sync_root / ".gitignore").write_text("# mine\nsecrets.txt\n", encoding="utf-8")

    ensure_sync_gitignore(sync_root)

    content = (sync_root / ".gitignore").read_text(encoding="utf-8")
    assert "secrets.txt" in content
    assert "tantivy/" in content


def test_sync_init_byo_ignores_derived_state_before_staging(
    sync_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The init path itself, not just the helper, must apply the ignore."""
    from memo import sync_git

    cfg = Config(
        data_dir=sync_root / "memorias",
        state_dir=sync_root / "state",
        reranker_enabled=False,
    )
    pushed: list[tuple[str, ...]] = []
    real_git = sync_git._git

    def fake_git(root: Path, *args: str, **kwargs: object):
        if args[:1] == ("push",) or args[:1] == ("remote",):
            pushed.append(args)
            return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="", stderr="")
        return real_git(root, *args, **kwargs)

    monkeypatch.setattr(sync_git, "_git", fake_git)
    sync_git.sync_init_home_byo(cfg, "https://github.com/example/memo-sync.git")

    tracked = _git(sync_root, "ls-files").splitlines()
    assert "memorias/note.md" in tracked
    assert not [p for p in tracked if p.startswith("state/tantivy/")]
