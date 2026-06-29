"""Tests for project-tag derivation."""

from __future__ import annotations

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


def test_global_bucket_constant_value() -> None:
    assert GLOBAL_BUCKET == "_global"
