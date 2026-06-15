from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import urllib.error
import urllib.request

import click

from memo.cli_common import console


@click.command(name="self-update")
@click.option("--check", is_flag=True, help="Check for a newer version without installing.")
def self_update(check: bool) -> None:
    current_version = importlib.metadata.version("mlx-memo")
    console.print(f"[dim]current version:[/dim] {current_version}")

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
    try:
        res = subprocess.run(["pipx", "list", "--short"], capture_output=True, text=True, timeout=10)
        if "mlx-memo" in res.stdout:
            installed_via = "pipx"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if installed_via is None:
        try:
            res = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True, timeout=10)
            if "mlx-memo" in res.stdout:
                installed_via = "uv"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if installed_via == "pipx":
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
