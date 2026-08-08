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

import re
import tomllib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "memo"

# `version("mlx-memo")` in any spelling — `importlib.metadata.version(...)`,
# a `from importlib.metadata import version` call, or an aliased
# `_version` / `_pkg_version`. `distribution("mlx-memo")` is deliberately not
# matched: `runtime/update.py` reads `direct_url.json` off it for install
# provenance, which is metadata's job and has nothing to do with the version.
_METADATA_VERSION_CALL = re.compile(r'version\s*\(\s*["\']mlx-memo["\']\s*\)')


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
    """An installed wheel lives at `<site-packages>/memo/__init__.py`. The
    directory two levels up is *not* the wheel's own project, so its pyproject
    must never be read — only a `src` parent marks a checkout.

    A real mlx-memo pyproject is planted exactly where a missing `src` check
    would look (`pkg.parents[2]`, the layout `pip install --target
    site-packages` produces inside a checkout), so this fails if that guard is
    dropped instead of passing through the no-pyproject path.
    """
    from memo import _checkout_version

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "mlx-memo"\nversion = "99.98.97"\n', encoding="utf-8"
    )
    pkg = tmp_path / "site-packages" / "memo"
    pkg.mkdir(parents=True)
    init = pkg / "__init__.py"
    init.write_text("# memo\n", encoding="utf-8")
    assert (init.parents[2] / "pyproject.toml").is_file(), "fixture must bait the missing guard"

    assert _checkout_version(init) is None


def test_foreign_project_with_a_src_memo_is_ignored(tmp_path: Path) -> None:
    """Some other project vendoring a `src/memo/` must not donate its version."""
    from memo import _checkout_version

    init = _make_checkout(tmp_path, version="1.0.0", name="not-mlx-memo")

    assert _checkout_version(init) is None


def test_foreign_project_merely_mentioning_mlx_memo_is_ignored(tmp_path: Path) -> None:
    """Identity comes from the parsed `[project] name`, not from the bytes.

    A vendoring manifest / dependency pin / comment can carry the literal
    `name = "mlx-memo"` while `[project] name` is someone else — a raw
    substring test hands that project's version to `memo.__version__`.
    """
    from memo import _checkout_version

    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "acme-agents"\n'
        'version = "9.9.9"\n'
        "\n"
        "[[tool.acme.vendored]]\n"
        'name = "mlx-memo"\n'
        'revision = "v4.9.3"\n',
        encoding="utf-8",
    )
    pkg = tmp_path / "src" / "memo"
    pkg.mkdir(parents=True)
    init = pkg / "__init__.py"
    init.write_text("# memo\n", encoding="utf-8")

    assert _checkout_version(init) is None


def test_resolve_falls_back_to_installed_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """The branch every *installed* user takes.

    `_checkout_version` returns None off a checkout (wheel in site-packages),
    and `_resolve_version` must then answer with the distribution metadata
    rather than a placeholder. Unreachable in a source checkout otherwise, so
    the checkout probe is stubbed to the None it returns when installed.
    """
    from importlib.metadata import version as installed_version

    import memo

    monkeypatch.setattr(memo, "_checkout_version", lambda _pkg_init: None)

    assert memo._resolve_version() == installed_version("mlx-memo")


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


def test_only_the_resolver_reads_installed_metadata_for_the_version() -> None:
    """One reader of distribution metadata, and it is the fallback in
    `_resolve_version`.

    Fixing `memo.__version__` alone is not enough: every surface that answers
    "what version is this?" has to route through it, or a dev checkout reports
    4.9.3 from `memo --version` and 4.9.2 from the passport header / `/health`
    / the doctor probe standing right next to it.
    """
    offenders = [
        f"{path.relative_to(SRC.parent.parent)}:{lineno}: {line.strip()}"
        for path in sorted(SRC.rglob("*.py"))
        if path.name != "__init__.py"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _METADATA_VERSION_CALL.search(line)
    ]

    assert offenders == [], (
        "these read the installed distribution's version snapshot instead of "
        "`memo.__version__`, so they go stale on an editable install the moment "
        "the version is bumped:\n  " + "\n  ".join(offenders)
    )


def test_every_version_surface_reports_the_single_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Behavioural half of the guard above: pin `memo.__version__` to a
    sentinel and every surface must echo it.

    Only works because each call site imports the attribute at call time; a
    site reading `importlib.metadata` reports the real installed version and
    fails here.
    """
    import memo
    from memo.import_export import _generator_string
    from memo.runtime.codex_notify import memo_version_badge

    monkeypatch.setattr(memo, "__version__", "77.66.55")

    assert _generator_string() == "memo/77.66.55"
    assert memo_version_badge() == "[Memo 77.66.55]"


def test_mcp_health_route_reports_the_single_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FastMCP server's `/health` liveness payload — a separate route from
    `server_http.py`'s `/health`, and the one the version fix originally
    claimed but missed."""
    import asyncio
    import json

    pytest.importorskip("starlette")

    import memo
    from memo.server import _health_route_handler

    monkeypatch.setattr(memo, "__version__", "77.66.55")

    response = asyncio.run(_health_route_handler(None))

    assert json.loads(bytes(response.body)) == {"ok": True, "version": "77.66.55"}


def test_startup_banner_reports_the_single_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """`memo startup-banner` stamps `[Memo <ver>]` into every agent launch."""
    from click.testing import CliRunner

    import memo
    from memo.cli_banner import startup_banner_cmd

    monkeypatch.setattr(memo, "__version__", "77.66.55")

    result = CliRunner().invoke(startup_banner_cmd, [])

    assert result.exit_code == 0, result.output
    assert "[Memo 77.66.55]" in result.output
