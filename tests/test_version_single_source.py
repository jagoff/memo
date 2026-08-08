"""`memo.__version__` must describe the code that is actually running.

`pyproject.toml` `[project].version` is the single source of truth. An
installed distribution carries a *snapshot* of it in its `.dist-info`
metadata, and for an **editable** install that snapshot goes stale the moment
the version is bumped — nothing rewrites it until someone reinstalls. That is
how the running 4.9.3 source came to report `memo.__version__ == "4.9.2"`.

So the resolution order is: a source checkout is authoritative for its own
version; everywhere else (a real wheel in site-packages, where no pyproject
exists) the distribution metadata is.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


def _checkout_root(pkg_init: Path) -> Path | None:
    """Repo root for a `<repo>/src/memo/__init__.py` layout, else None."""
    if pkg_init.parent.parent.name != "src":
        return None
    return pkg_init.parents[2]


def test_version_matches_the_checkout_it_is_imported_from() -> None:
    """The end-to-end invariant: when `memo` is imported from a source
    checkout, `memo.__version__` equals that checkout's declared version.

    This is the drift itself — a stale editable `.dist-info` reporting the
    previous release while the source on disk is the next one.
    """
    import memo

    pkg_init = Path(memo.__file__).resolve()
    repo = _checkout_root(pkg_init)
    if repo is None:
        pytest.skip(f"memo imported from an installed distribution, not a checkout: {pkg_init}")

    declared = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]

    assert memo.__version__ == declared, (
        f"memo.__version__ is {memo.__version__!r} but {repo}/pyproject.toml "
        f"declares {declared!r} — the running code reports a different release "
        f"than the one it is"
    )


def _make_checkout(root: Path, *, version: str, name: str = "mlx-memo") -> Path:
    """Build `<root>/{pyproject.toml,src/memo/__init__.py}`; return the __init__."""
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "{version}"\n', encoding="utf-8"
    )
    pkg = root / "src" / "memo"
    pkg.mkdir(parents=True)
    init = pkg / "__init__.py"
    init.write_text("# memo\n", encoding="utf-8")
    return init


def test_checkout_version_wins_over_installed_metadata(tmp_path: Path) -> None:
    """A checkout's pyproject version is returned verbatim — not the version of
    whatever `mlx-memo` distribution happens to be installed in this venv."""
    from importlib.metadata import version as installed_version

    from memo import _checkout_version

    init = _make_checkout(tmp_path, version="99.98.97")

    resolved = _checkout_version(init)

    assert resolved == "99.98.97"
    assert resolved != installed_version("mlx-memo")


def test_site_packages_layout_defers_to_metadata(tmp_path: Path) -> None:
    """An installed wheel lives at `<site-packages>/memo/__init__.py` — no `src`
    parent, no pyproject — so the resolver must decline and let the
    distribution metadata answer."""
    from memo import _checkout_version

    pkg = tmp_path / "site-packages" / "memo"
    pkg.mkdir(parents=True)
    init = pkg / "__init__.py"
    init.write_text("# memo\n", encoding="utf-8")

    assert _checkout_version(init) is None


def test_foreign_project_with_a_src_memo_is_ignored(tmp_path: Path) -> None:
    """Some other project vendoring a `src/memo/` must not donate its version."""
    from memo import _checkout_version

    init = _make_checkout(tmp_path, version="1.0.0", name="not-mlx-memo")

    assert _checkout_version(init) is None


def test_missing_pyproject_defers_to_metadata(tmp_path: Path) -> None:
    """A `src/memo/` with no pyproject above it (e.g. a partial copy) resolves
    to None rather than raising at import time."""
    from memo import _checkout_version

    pkg = tmp_path / "src" / "memo"
    pkg.mkdir(parents=True)
    init = pkg / "__init__.py"
    init.write_text("# memo\n", encoding="utf-8")

    assert _checkout_version(init) is None


def test_malformed_pyproject_defers_to_metadata(tmp_path: Path) -> None:
    """Invalid TOML must not break `import memo`."""
    from memo import _checkout_version

    (tmp_path / "pyproject.toml").write_text(
        'name = "mlx-memo" this is not [ valid toml !!!\n', encoding="utf-8"
    )
    pkg = tmp_path / "src" / "memo"
    pkg.mkdir(parents=True)
    init = pkg / "__init__.py"
    init.write_text("# memo\n", encoding="utf-8")

    assert _checkout_version(init) is None
