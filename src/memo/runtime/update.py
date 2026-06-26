from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import click

from memo.cli_common import console

_PIPX_CANDIDATES = [
    Path.home() / ".local/bin/pipx",
    Path("/opt/homebrew/bin/pipx"),
    Path("/usr/local/bin/pipx"),
    Path("/usr/bin/pipx"),
]


def _find_pipx() -> str | None:
    """Return path to the pipx binary, checking PATH then common locations."""
    found = shutil.which("pipx")
    if found:
        return found
    for candidate in _PIPX_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


def _find_uv() -> str | None:
    """Return path to the uv binary, checking PATH then common locations."""
    found = shutil.which("uv")
    if found:
        return found
    for candidate in [
        Path.home() / ".local/bin/uv",
        Path.home() / ".cargo/bin/uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ]:
        if candidate.is_file():
            return str(candidate)
    return None


def _read_pipx_venv_metadata() -> dict | None:
    """Read mlx-memo's pipx metadata directly from the venv JSON (no pipx binary needed)."""
    meta_path = Path.home() / ".local/pipx/venvs/mlx-memo/pipx_metadata.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _version_ge(a: str, b: str) -> bool:
    """Return True if version a >= b, using packaging if available, else string compare."""
    try:
        from packaging.version import Version

        return Version(a) >= Version(b)
    except Exception:
        return a >= b


def _detect_install_method() -> str | None:
    """``"pipx"`` / ``"uv"`` / None — how mlx-memo's isolated runtime was installed.

    Checks ``sys.executable`` first — if the running Python lives inside the uv
    tool venv, return "uv" immediately.  This prevents a stale pipx venv (left
    over from a migration) from shadowing a live uv-managed install.
    """
    uv_tool_prefix = Path.home() / ".local" / "share" / "uv" / "tools" / "mlx-memo"
    try:
        if Path(sys.executable).is_relative_to(uv_tool_prefix):
            return "uv"
    except (TypeError, ValueError):
        pass

    uv = _find_uv()
    if uv:
        try:
            res = subprocess.run([uv, "tool", "list"], capture_output=True, text=True, timeout=10)
            if "mlx-memo" in res.stdout:
                return "uv"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    pipx = _find_pipx()
    if pipx:
        try:
            res = subprocess.run([pipx, "list", "--short"], capture_output=True, text=True, timeout=10)
            if "mlx-memo" in res.stdout:
                return "pipx"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    if _read_pipx_venv_metadata() is not None:
        return "pipx"
    return None


def _clear_update_notify() -> None:
    """Remove the pending-update notification file after a successful update."""
    try:
        from memo.config import Config
        from memo.runtime.autoupdate import _clear_notify

        _clear_notify(Config.from_env())
    except Exception:  # noqa: S110
        pass


def _prewarm_after_update() -> None:
    memo_bin = shutil.which("memo") or sys.executable
    cmd = (
        [memo_bin, "prewarm", "--download-all"]
        if memo_bin.endswith("memo")
        else [memo_bin, "-m", "memo.cli", "prewarm", "--download-all"]
    )
    subprocess.run(cmd, check=False)


@click.command(name="update")
@click.argument("stray", required=False, metavar="")
@click.option("--check", is_flag=True, help="Check for a newer version without installing.")
@click.option(
    "--to-tag",
    default=None,
    help="Install a specific git tag (e.g. v1.0.1) from the memo repo, bypassing "
    "PyPI. Used by the memo-mcp auto-update path (git installs aren't on PyPI).",
)
def self_update(stray: str | None, check: bool, to_tag: str | None) -> None:
    # `memo update` is the software self-updater. If someone passes a memory ID
    # (old muscle memory — `update` used to patch a memory), nudge them to the
    # renamed memory-edit command instead of silently self-updating.
    if stray is not None:
        raise click.ClickException(f"did you mean `memo edit {stray}`?")
    current_version = importlib.metadata.version("mlx-memo")
    console.print(f"[dim]current version:[/dim] {current_version}")

    if check and not to_tag:
        from memo.config import Config
        from memo.runtime.autoupdate import notify_if_newer

        cfg = Config.from_env()
        tag = notify_if_newer(cfg, force=True)
        if tag:
            console.print(f"[yellow]Update available:[/yellow] {current_version} → {tag}")
            console.print("[dim]Run [bold]memo update[/bold] to install.[/dim]")
        else:
            console.print("[green]memo is up to date.[/green]")
        return

    # Git-tag path: reinstall the isolated runtime straight from the tagged ref.
    # PyPI is skipped (these installs come from git+https://…/memo.git).
    if to_tag:
        from memo.flags import flag_str
        from memo.runtime.autoupdate import DEFAULT_REPO

        repo = flag_str("MEMO_AUTO_UPDATE_REPO") or DEFAULT_REPO
        spec = f"git+{repo}@{to_tag}"
        method = _detect_install_method()
        if method == "uv":
            uv = _find_uv() or "uv"
            console.print(f"[dim]Installing {to_tag} via uv tool…[/dim]")
            proc = subprocess.run(
                [uv, "tool", "install", spec, "--force", "--reinstall"], check=False
            )
        elif method == "pipx":
            pipx = _find_pipx() or "pipx"
            console.print(f"[dim]Installing {to_tag} via pipx…[/dim]")
            proc = subprocess.run([pipx, "install", "--force", spec], check=False)
        else:
            raise click.ClickException(
                "Could not detect install method (pipx/uv) for git-tag update."
            )
        if proc.returncode != 0:
            raise click.ClickException(f"git-tag install of {to_tag} failed.")
        _clear_update_notify()
        console.print(f"[green]✓[/green] updated to {to_tag}. Pre-warming MLX models…")
        _prewarm_after_update()
        return

    # Check git tags first — canonical for git-installed memo; also works for PyPI installs.
    from memo.flags import flag_str
    from memo.runtime.autoupdate import DEFAULT_REPO, is_newer, latest_remote_tag

    repo = flag_str("MEMO_AUTO_UPDATE_REPO") or DEFAULT_REPO
    latest_tag = latest_remote_tag(repo)
    if latest_tag is not None:
        console.print(f"[dim]latest tag:[/dim]           {latest_tag}")
        if not is_newer(latest_tag, current_version):
            console.print("[green]memo is already up to date.[/green]")
            return
        console.print(f"[yellow]Update available:[/yellow] {current_version} → {latest_tag}")
        spec = f"git+{repo}@{latest_tag}"
        method = _detect_install_method()
        if method == "uv":
            uv = _find_uv() or "uv"
            console.print(f"[dim]Installing {latest_tag} via uv tool…[/dim]")
            proc = subprocess.run(
                [uv, "tool", "install", spec, "--force", "--reinstall"], check=False
            )
        elif method == "pipx":
            pipx = _find_pipx() or "pipx"
            console.print(f"[dim]Installing {latest_tag} via pipx…[/dim]")
            proc = subprocess.run([pipx, "install", "--force", spec], check=False)
        else:
            raise click.ClickException("Could not detect install method (pipx/uv).")
        if proc.returncode != 0:
            raise click.ClickException(f"git-tag install of {latest_tag} failed.")
        _clear_update_notify()
        console.print(f"[green]✓[/green] updated to {latest_tag}. Pre-warming MLX models…")
        _prewarm_after_update()
        return

    # Git unreachable — fall back to PyPI.
    console.print("[dim]Could not reach git; checking PyPI…[/dim]")
    try:
        url = "https://pypi.org/pypi/mlx-memo/json"
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        latest_version = data["info"]["version"]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, OSError) as exc:
        console.print(f"[red]Could not fetch version info:[/red] {exc}")
        return

    console.print(f"[dim]latest PyPI version:[/dim]  {latest_version}")
    if _version_ge(current_version, latest_version):
        console.print("[green]memo is already up to date.[/green]")
        return
    console.print(f"[yellow]Update available:[/yellow] {current_version} → {latest_version}")

    installed_via = _detect_install_method()

    if installed_via == "pipx":
        pipx = _find_pipx() or "pipx"
        console.print("[dim]Upgrading via pipx…[/dim]")
        result = subprocess.run([pipx, "upgrade", "mlx-memo"], check=False)
        if result.returncode != 0:
            raise click.ClickException("pipx upgrade failed.")
    elif installed_via == "uv":
        uv = _find_uv() or "uv"
        console.print("[dim]Upgrading via uv tool…[/dim]")
        result = subprocess.run([uv, "tool", "upgrade", "mlx-memo"], check=False)
        if result.returncode != 0:
            raise click.ClickException("uv tool upgrade failed.")
    else:
        console.print(
            "[yellow]Could not detect install method (pipx/uv).[/yellow]\n"
            "Re-run the installer:\n"
            "  curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash\n"
            "or:\n"
            "  pipx install --force git+https://github.com/jagoff/memo.git\n"
            "  uv tool install --force git+https://github.com/jagoff/memo.git"
        )
        return

    _clear_update_notify()
    console.print("[green]✓[/green] upgrade complete. Pre-warming MLX models…")
    _prewarm_after_update()
