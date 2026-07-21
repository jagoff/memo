"""Tests for project-tag derivation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from memo.project import (
    GLOBAL_BUCKET,
    current_project_tag,
    has_project_tag,
    is_project_tag,
    project_bucket,
    slugify_project,
)


def test_slugify_project() -> None:
    assert slugify_project("my-repo") == "my-repo"
    assert slugify_project("My Cool Repo!") == "my-cool-repo"
    assert slugify_project("RAG.local-v2") == "rag-local-v2"
    assert slugify_project("---") == ""


def test_is_project_tag() -> None:
    assert is_project_tag("project:memo")
    assert is_project_tag("project:")  # empty slug still has prefix
    assert not is_project_tag("projectx")
    assert not is_project_tag("memo")


def test_has_project_tag() -> None:
    assert has_project_tag(["foo", "project:memo", "bar"])
    assert not has_project_tag(["foo", "bar"])
    assert not has_project_tag([])


def test_env_var_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEMO_PROJECT_TAG", "Hand Picked!")
    assert current_project_tag(tmp_path) == "project:hand-picked"


def test_env_var_with_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEMO_PROJECT_TAG", "project:already-prefixed")
    assert current_project_tag(tmp_path) == "project:already-prefixed"


def test_git_toplevel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMO_PROJECT_TAG", raising=False)
    repo = tmp_path / "my-cool-repo"
    nested = repo / "src" / "deep" / "tree"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    assert current_project_tag(nested) == "project:my-cool-repo"


def test_no_git_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMO_PROJECT_TAG", raising=False)
    leaf = tmp_path / "no_git_here"
    leaf.mkdir()
    assert current_project_tag(leaf) is None


def test_default_cwd_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMO_PROJECT_TAG", raising=False)
    repo = tmp_path / "default-cwd"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.chdir(repo)
    assert current_project_tag() == "project:default-cwd"


def test_project_bucket_returns_slug_of_first_project_tag() -> None:
    assert project_bucket(["note", "project:memo", "db"]) == "memo"


def test_project_bucket_untagged_is_global() -> None:
    assert project_bucket(["note", "db"]) == GLOBAL_BUCKET


def test_project_bucket_empty_tags_is_global() -> None:
    assert project_bucket([]) == "_global"


def test_project_bucket_sanitizes_traversal_tag() -> None:
    # A user-supplied tag reaches project_bucket verbatim — the derived
    # folder must never contain path separators or '..'.
    bucket = project_bucket(["project:../../../../tmp/evil"])
    assert bucket == "tmp-evil"
    assert "/" not in bucket
    assert ".." not in bucket


def test_project_bucket_fully_invalid_slug_falls_back_to_global() -> None:
    assert project_bucket(["project:../.."]) == GLOBAL_BUCKET


def test_project_bucket_avoids_lifecycle_archive_dirs() -> None:
    # A project literally named `inactive`/`archived` must NOT land in the
    # lifecycle-archive keyspace — reindex/gc skip those dirs, so a memory
    # written there would be invisible to search and unrecoverable by reindex.
    # The `_`-prefix remap is collision-free (slugify never emits a leading `_`).
    assert project_bucket(["project:inactive"]) == "_inactive"
    assert project_bucket(["project:archived"]) == "_archived"
    # A git repo basename `Inactive` slugifies to `inactive` and must remap too.
    assert project_bucket(["project:Inactive"]) == "_inactive"
    # Non-bare variants are already disjoint and stay verbatim.
    assert project_bucket(["project:archived-notes"]) == "archived-notes"


def test_global_bucket_constant_value() -> None:
    assert GLOBAL_BUCKET == "_global"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_worktree_resolves_to_main_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a linked worktree (`.git` FILE) must tag as the MAIN repo,
    not as the worktree basename — the release flow's `/tmp/rel` worktree was
    minting `project:rel` memories forever."""
    monkeypatch.delenv("MEMO_PROJECT_TAG", raising=False)
    main = tmp_path / "my-main-repo"
    main.mkdir()
    _git(main, "init", "-q")
    _git(main, "commit", "-q", "--allow-empty", "-m", "seed")
    wt = tmp_path / "rel"
    _git(main, "worktree", "add", "-q", str(wt), "HEAD")

    assert current_project_tag(wt) == "project:my-main-repo"
    nested = wt / "src" / "deep"
    nested.mkdir(parents=True)
    assert current_project_tag(nested) == "project:my-main-repo"


def test_dotgit_file_without_commondir_falls_back_to_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Submodule layout: `.git` file → gitdir with NO `commondir`. Keep the
    old own-basename behavior (a submodule IS its own project)."""
    monkeypatch.delenv("MEMO_PROJECT_TAG", raising=False)
    host = tmp_path / "host"
    module_gitdir = host / ".git" / "modules" / "sub"
    module_gitdir.mkdir(parents=True)
    sub = host / "sub-module"
    sub.mkdir()
    (sub / ".git").write_text(f"gitdir: {module_gitdir}\n", encoding="utf-8")

    assert current_project_tag(sub) == "project:sub-module"


def test_dotgit_file_garbage_falls_back_to_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MEMO_PROJECT_TAG", raising=False)
    repo = tmp_path / "weird-checkout"
    repo.mkdir()
    (repo / ".git").write_text("not a gitdir pointer\n", encoding="utf-8")

    assert current_project_tag(repo) == "project:weird-checkout"
