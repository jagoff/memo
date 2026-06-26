from __future__ import annotations

from pathlib import Path

from memo.runtime.freshness import check_install_freshness


def _make_pkg(root: Path, version: str, body: str) -> Path:
    """Build a fake repo at `root` with src/memo/{__init__.py,sample.py} + pyproject."""
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "mlx-memo"\nversion = "{version}"\n', encoding="utf-8"
    )
    pkg = root / "src" / "memo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("# memo\n", encoding="utf-8")
    (pkg / "sample.py").write_text(body, encoding="utf-8")
    return pkg


def test_fresh_when_version_and_content_match(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pkg = _make_pkg(repo, "1.2.3", "X = 1\n")
    # Installed dir = a byte-identical copy of the repo package.
    installed = tmp_path / "site" / "memo"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("# memo\n", encoding="utf-8")
    (installed / "sample.py").write_text("X = 1\n", encoding="utf-8")

    out = check_install_freshness(
        installed_version="1.2.3", installed_pkg_dir=installed, repo_root=repo
    )
    assert out["status"] == "fresh"


def test_stale_when_same_version_different_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_pkg(repo, "1.2.3", "X = 2\n")  # repo has new content
    installed = tmp_path / "site" / "memo"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("# memo\n", encoding="utf-8")
    (installed / "sample.py").write_text("X = 1\n", encoding="utf-8")  # old content

    out = check_install_freshness(
        installed_version="1.2.3", installed_pkg_dir=installed, repo_root=repo
    )
    assert out["status"] == "stale"
    assert "reinstall" in out["message"]


def test_repo_ahead_when_versions_differ(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_pkg(repo, "1.3.0", "X = 1\n")
    out = check_install_freshness(
        installed_version="1.2.3", installed_pkg_dir=None, repo_root=repo
    )
    assert out["status"] == "repo-ahead"


def test_skipped_when_no_repo(tmp_path: Path) -> None:
    out = check_install_freshness(
        installed_version="1.2.3", installed_pkg_dir=None, repo_root=None
    )
    assert out["status"] == "skipped"


def test_skipped_on_malformed_pyproject(tmp_path: Path) -> None:
    """Guard: invalid TOML in pyproject.toml must not raise; must return skipped."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("this is not [ valid toml !!!\n", encoding="utf-8")

    out = check_install_freshness(
        installed_version="1.2.3",
        installed_pkg_dir=None,
        repo_root=repo,
    )
    assert out["status"] == "skipped"
