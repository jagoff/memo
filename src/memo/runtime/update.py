from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import urllib.error
import urllib.request

import click

from memo.cli_common import console


def _detect_install_method() -> str | None:
    """``"pipx"`` / ``"uv"`` / None — how mlx-memo's isolated runtime was installed."""
    try:
        res = subprocess.run(["pipx", "list", "--short"], capture_output=True, text=True, timeout=10)
        if "mlx-memo" in res.stdout:
            return "pipx"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        res = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True, timeout=10)
        if "mlx-memo" in res.stdout:
            return "uv"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _prewarm_after_update() -> None:
    import shutil as _shutil

    memo_bin = _shutil.which("memo") or sys.executable
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

    # Git-tag path: reinstall the isolated runtime straight from the tagged ref.
    # PyPI is skipped (these installs come from git+https://…/memo.git).
    if to_tag:
        from memo.flags import flag_str
        from memo.runtime.autoupdate import DEFAULT_REPO

        repo = flag_str("MEMO_AUTO_UPDATE_REPO") or DEFAULT_REPO
        spec = f"git+{repo}@{to_tag}"
        method = _detect_install_method()
        if method == "uv":
            console.print(f"[dim]Installing {to_tag} via uv tool…[/dim]")
            proc = subprocess.run(
                ["uv", "tool", "install", spec, "--force", "--reinstall"], check=False
            )
        elif method == "pipx":
            console.print(f"[dim]Installing {to_tag} via pipx…[/dim]")
            proc = subprocess.run(["pipx", "install", "--force", spec], check=False)
        else:
            raise click.ClickException(
                "Could not detect install method (pipx/uv) for git-tag update."
            )
        if proc.returncode != 0:
            raise click.ClickException(f"git-tag install of {to_tag} failed.")
        console.print(f"[green]✓[/green] updated to {to_tag}. Pre-warming MLX models…")
        _prewarm_after_update()
        return

    try:
        url = "https://pypi.org/pypi/mlx-memo/json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latest_version = data["info"]["version"]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, OSError) as exc:
        console.print(f"[red]Could not fetch PyPI version:[/red] {exc}")
        if check:
            return
        latest_version = None

    if latest_version:
        console.print(f"[dim]latest version:[/dim]  {latest_version}")
        if current_version == latest_version:
            console.print("[green]memo is already up to date.[/green]")
            if check:
                return
        else:
            console.print(f"[yellow]Update available:[/yellow] {current_version} → {latest_version}")
            if check:
                return

    installed_via: str | None = None
    pipx_source_url: str | None = None
    try:
        res = subprocess.run(["pipx", "list", "--json"], capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            import json as _json

            _data = _json.loads(res.stdout)
            _pkg = _data.get("venvs", {}).get("mlx-memo", {}).get("metadata", {}).get(
                "main_package", {}
            )
            if _pkg:
                installed_via = "pipx"
                _url = _pkg.get("package_or_url", "")
                if _url.startswith("git+"):
                    pipx_source_url = _url
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    if installed_via is None:
        try:
            res = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True, timeout=10)
            if "mlx-memo" in res.stdout:
                installed_via = "uv"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if installed_via == "pipx":
        if pipx_source_url:
            console.print("[dim]Re-installing from git source via pipx…[/dim]")
            result = subprocess.run(
                ["pipx", "install", "--force", pipx_source_url], check=False
            )
            if result.returncode != 0:
                raise click.ClickException("pipx install --force failed.")
        else:
            console.print("[dim]Upgrading via pipx…[/dim]")
            result = subprocess.run(["pipx", "upgrade", "mlx-memo"], check=False)
            if result.returncode != 0:
                raise click.ClickException("pipx upgrade failed.")
    elif installed_via == "uv":
        console.print("[dim]Upgrading via uv tool…[/dim]")
        result = subprocess.run(["uv", "tool", "upgrade", "mlx-memo"], check=False)
        if result.returncode != 0:
            raise click.ClickException("uv tool upgrade failed.")
    else:
        console.print(
            "[yellow]Could not detect install method (pipx/uv).[/yellow]\n"
            "Re-run the installer:\n"
            "  curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash\n"
            "or:\n"
            "  pipx install --force mlx-memo\n"
            "  uv tool install --force mlx-memo"
        )
        return

    console.print("[green]✓[/green] upgrade complete. Pre-warming MLX models…")
    import shutil as _shutil

    memo_bin = _shutil.which("memo") or sys.executable
    _prewarm_cmd = (
        [memo_bin, "prewarm", "--download-all"]
        if memo_bin.endswith("memo")
        else [memo_bin, "-m", "memo.cli", "prewarm", "--download-all"]
    )
    subprocess.run(_prewarm_cmd, check=False)
