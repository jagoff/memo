"""Regression: release helpers must operate on the checkout you are standing
in, not on the one the module happened to be imported from.

Found cutting v4.9.2: `memo release bump` was run from an isolated release
worktree, but `_resolve_repo` fell back to `Path(__file__).parents[2]` — the
*shared* working tree — so the bump landed there instead. CLAUDE.md's release
procedure exists precisely to keep releases out of the shared tree, and this
silently defeated it.

Resolution order is now: MEMO_DEV_REPO (explicit) > the memo checkout
containing the cwd > the module's own checkout.
"""

from __future__ import annotations

import pytest

from memo.cli_release import _REPO_ROOT, _resolve_repo

PYPROJECT = """\
[project]
name = "mlx-memo"
version = "9.9.9"
"""


def _make_checkout(root, *, name: str = "mlx-memo"):
    (root / "src" / "memo").mkdir(parents=True)
    (root / "pyproject.toml").write_text(PYPROJECT.replace("mlx-memo", name), encoding="utf-8")
    return root


def test_cwd_checkout_wins_over_the_module_checkout(tmp_path, monkeypatch) -> None:
    worktree = _make_checkout(tmp_path / "release-worktree")
    monkeypatch.delenv("MEMO_DEV_REPO", raising=False)
    monkeypatch.chdir(worktree)

    assert _resolve_repo() == worktree


def test_a_subdirectory_of_the_checkout_resolves_to_its_root(tmp_path, monkeypatch) -> None:
    worktree = _make_checkout(tmp_path / "release-worktree")
    nested = worktree / "docs" / "homebrew"
    nested.mkdir(parents=True)
    monkeypatch.delenv("MEMO_DEV_REPO", raising=False)
    monkeypatch.chdir(nested)

    assert _resolve_repo() == worktree


def test_explicit_dev_repo_still_wins(tmp_path, monkeypatch) -> None:
    worktree = _make_checkout(tmp_path / "release-worktree")
    explicit = _make_checkout(tmp_path / "explicit")
    monkeypatch.chdir(worktree)
    monkeypatch.setenv("MEMO_DEV_REPO", str(explicit))

    assert _resolve_repo() == explicit


def test_an_unrelated_project_never_hijacks_the_release(tmp_path, monkeypatch) -> None:
    """A pyproject.toml alone is not a memo checkout — cutting a memo release
    from some other repo's directory must not rewrite that repo."""
    other = _make_checkout(tmp_path / "other-project", name="some-other-package")
    monkeypatch.delenv("MEMO_DEV_REPO", raising=False)
    monkeypatch.chdir(other)

    assert _resolve_repo() == _REPO_ROOT


def test_falls_back_to_the_module_checkout_outside_any_repo(tmp_path, monkeypatch) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.delenv("MEMO_DEV_REPO", raising=False)
    monkeypatch.chdir(elsewhere)

    assert _resolve_repo() == _REPO_ROOT


@pytest.mark.parametrize("missing", ["pyproject.toml", "src/memo"])
def test_a_partial_checkout_is_not_accepted(tmp_path, monkeypatch, missing) -> None:
    partial = _make_checkout(tmp_path / "partial")
    target = partial / missing
    if target.is_dir():
        target.rmdir()
    else:
        target.unlink()
    monkeypatch.delenv("MEMO_DEV_REPO", raising=False)
    monkeypatch.chdir(partial)

    assert _resolve_repo() == _REPO_ROOT
