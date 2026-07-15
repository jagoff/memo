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
from memo.runtime.codex_notify import emit_codex_notify

_CODEX_UPDATE_NOTIFY_TITLE = "Plugin updated: memo"
_CODEX_UPDATE_NOTIFY_BODY = "Run /reload_plugins to apply"

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


def _read_direct_url() -> str | None:
    """Raw ``direct_url.json`` for the installed mlx-memo, or None.

    pip/uv record install provenance here: an editable dev install has
    ``{"dir_info": {"editable": true}}``; an isolated git/PyPI install has
    ``vcs_info``/no ``dir_info``. Indirected through this helper so tests can
    stub it without faking importlib.metadata.
    """
    try:
        return importlib.metadata.distribution("mlx-memo").read_text("direct_url.json")
    except Exception:
        return None


def _running_install_is_editable() -> bool:
    """True if the running mlx-memo is an editable/dev install (project ``.venv``)
    rather than the isolated pipx/uv runtime.

    ``memo update`` can't meaningfully update an editable checkout:
    ``_detect_install_method`` would find the SIBLING isolated tool (``uv tool
    list`` shows mlx-memo) and install a tag over THAT, leaving the running
    editable install on its stale version — the "memo update doesn't update"
    symptom. Detect it and refuse instead.
    """
    raw = _read_direct_url()
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get("dir_info", {}).get("editable"))
    except (json.JSONDecodeError, AttributeError):
        return False


def _editable_source_path() -> str:
    """Filesystem path of the editable checkout (from ``direct_url.json`` url)."""
    raw = _read_direct_url()
    if not raw:
        return "this checkout"
    try:
        url = json.loads(raw).get("url", "")
    except (json.JSONDecodeError, AttributeError):
        return "this checkout"
    if url.startswith("file://"):
        return url[len("file://") :]
    return url or "this checkout"


def _version_ge(a: str, b: str) -> bool:
    """Return True if version a >= b, using packaging if available, else string compare."""
    try:
        from packaging.version import Version

        return Version(a) >= Version(b)
    except Exception:
        try:
            return tuple(int(x) for x in a.split(".")) >= tuple(int(x) for x in b.split("."))
        except (ValueError, TypeError):
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
            res = subprocess.run(
                [pipx, "list", "--short"], capture_output=True, text=True, timeout=10
            )
            if "mlx-memo" in res.stdout:
                return "pipx"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    if _read_pipx_venv_metadata() is not None:
        return "pipx"
    return None


def _clear_update_notify() -> None:
    """Remove update notification and spawned-stamp after a successful update."""
    try:
        from memo.config import Config
        from memo.runtime.autoupdate import _SPAWNED_STAMP, _clear_notify

        cfg = Config.from_env()
        _clear_notify(cfg)
        # Clear the per-tag spawned guard so the next startup can pick up
        # a newer release without being blocked by a stale stamp.
        with __import__("contextlib").suppress(OSError):
            (cfg.state_dir / _SPAWNED_STAMP).unlink(missing_ok=True)
    except Exception:  # noqa: S110
        pass


def _notify_codex_plugin_updated() -> bool:
    """Ask Codex/Supacode to show the same top-line notice used by plugin updates."""
    return emit_codex_notify(_CODEX_UPDATE_NOTIFY_TITLE, _CODEX_UPDATE_NOTIFY_BODY)


def _devin_skill_path() -> Path:
    return Path.home() / ".config" / "devin" / "skills" / "memo" / "SKILL.md"


def _refresh_agent_artifacts() -> bool:
    """Best-effort refresh of static agent artifacts after a memo runtime update.

    OpenCode and Devin Desktop consume memo through MCP config only, so there is
    no versioned plugin/skill cache to refresh for those clients.
    """
    has_claude = shutil.which("claude") is not None
    has_codex = shutil.which("codex") is not None
    devin_skill = _devin_skill_path()
    has_devin = shutil.which("devin") is not None or devin_skill.is_file()
    if not (has_claude or has_codex or has_devin):
        return False

    try:
        from memo.runtime.codex import _codex_home, _copy_slash_skill, _install_codex_plugin
        from memo.runtime.mcp import _agent_asset_root, _run_agent_command

        root = _agent_asset_root()
    except Exception as exc:
        console.print(f"[yellow]![/yellow] agent artifact refresh skipped: {exc}")
        return False

    refreshed = False
    if has_claude:
        try:
            _run_agent_command(
                ["claude", "plugin", "marketplace", "add", root],
                dry_run=False,
                ok_errors=("already", "exists"),
                best_effort=True,
            )
            # NOT best_effort: a genuinely failed install must surface as the
            # warning below (refreshed stays False), not print "already handled".
            _run_agent_command(
                ["claude", "plugin", "install", "memo@memo", "-s", "user"],
                dry_run=False,
                ok_errors=("already", "installed", "exists"),
            )
            refreshed = True
        except Exception as exc:
            console.print(f"[yellow]![/yellow] Claude Code plugin refresh skipped: {exc}")

    if has_codex:
        try:
            _copy_slash_skill(root, _codex_home() / "skills" / "memo" / "SKILL.md", dry_run=False)
            _install_codex_plugin(root, dry_run=False)
            refreshed = True
        except Exception as exc:
            console.print(f"[yellow]![/yellow] Codex plugin refresh skipped: {exc}")

    if has_devin:
        try:
            _copy_slash_skill(root, devin_skill, dry_run=False)
            refreshed = True
        except Exception as exc:
            console.print(f"[yellow]![/yellow] Devin skill refresh skipped: {exc}")

    return refreshed


def _finish_successful_update() -> None:
    _clear_update_notify()
    _refresh_agent_artifacts()
    _notify_codex_plugin_updated()


def _prewarm_after_update() -> None:
    memo_bin = shutil.which("memo") or sys.executable
    cmd = (
        [memo_bin, "prewarm", "--download-all"]
        if memo_bin.endswith("memo")
        else [memo_bin, "-m", "memo.cli", "prewarm", "--download-all"]
    )
    try:
        subprocess.run(cmd, check=False, timeout=300)
    except subprocess.TimeoutExpired:
        console.print("[yellow]prewarm timed out (300s); skipping model warmup.[/yellow]")


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

    # An editable/dev install (project .venv) can't be updated by this command —
    # it would install a tag over the SIBLING isolated tool and leave this
    # checkout unchanged. Refuse and point at the right tool (git + editable reinstall).
    if _running_install_is_editable():
        src = _editable_source_path()
        console.print(f"[yellow]memo is running from an editable/dev install[/yellow] ({src}).")
        console.print(
            "[dim]`memo update` only manages the isolated pipx/uv runtime, so it "
            "can't update an editable checkout — it would target a sibling tool "
            "and leave this install unchanged. Update the checkout with:[/dim]"
        )
        console.print(f"  git -C {src} pull && uv pip install -e .")
        return

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
        from memo.runtime.autoupdate import DEFAULT_REPO, tag_is_on_remote_master

        repo = flag_str("MEMO_AUTO_UPDATE_REPO") or DEFAULT_REPO
        if not tag_is_on_remote_master(repo, to_tag):
            raise click.ClickException(
                f"refusing untrusted tag {to_tag}: it is not reachable from remote master"
            )
        spec = f"git+{repo}@{to_tag}"
        method = _detect_install_method()
        if method == "uv":
            uv = _find_uv() or "uv"
            console.print(f"[dim]Installing {to_tag} via uv tool…[/dim]")
            try:
                proc = subprocess.run(
                    [uv, "tool", "install", spec, "--force", "--reinstall"],
                    check=False,
                    timeout=600,
                )
            except subprocess.TimeoutExpired as exc:
                raise click.ClickException(
                    f"git-tag install of {to_tag} timed out (600s)."
                ) from exc
        elif method == "pipx":
            pipx = _find_pipx() or "pipx"
            console.print(f"[dim]Installing {to_tag} via pipx…[/dim]")
            try:
                proc = subprocess.run([pipx, "install", "--force", spec], check=False, timeout=600)
            except subprocess.TimeoutExpired as exc:
                raise click.ClickException(
                    f"git-tag install of {to_tag} timed out (600s)."
                ) from exc
        else:
            raise click.ClickException(
                "Could not detect install method (pipx/uv) for git-tag update."
            )
        if proc.returncode != 0:
            raise click.ClickException(f"git-tag install of {to_tag} failed.")
        _finish_successful_update()
        console.print(f"[green]✓[/green] updated to {to_tag}. Pre-warming MLX models…")
        _prewarm_after_update()
        return

    # Check git tags first — canonical for git-installed memo; also works for PyPI installs.
    from memo.flags import flag_str
    from memo.runtime.autoupdate import (
        DEFAULT_REPO,
        is_newer,
        latest_remote_tag,
        tag_is_on_remote_master,
    )

    repo = flag_str("MEMO_AUTO_UPDATE_REPO") or DEFAULT_REPO
    latest_tag = latest_remote_tag(repo)
    if latest_tag is not None and not tag_is_on_remote_master(repo, latest_tag):
        console.print(
            f"[red]Refusing untrusted tag {latest_tag}:[/red] not reachable from remote master."
        )
        latest_tag = None
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
            try:
                proc = subprocess.run(
                    [uv, "tool", "install", spec, "--force", "--reinstall"],
                    check=False,
                    timeout=600,
                )
            except subprocess.TimeoutExpired as exc:
                raise click.ClickException(
                    f"git-tag install of {latest_tag} timed out (600s)."
                ) from exc
        elif method == "pipx":
            pipx = _find_pipx() or "pipx"
            console.print(f"[dim]Installing {latest_tag} via pipx…[/dim]")
            try:
                proc = subprocess.run([pipx, "install", "--force", spec], check=False, timeout=600)
            except subprocess.TimeoutExpired as exc:
                raise click.ClickException(
                    f"git-tag install of {latest_tag} timed out (600s)."
                ) from exc
        else:
            raise click.ClickException("Could not detect install method (pipx/uv).")
        if proc.returncode != 0:
            raise click.ClickException(f"git-tag install of {latest_tag} failed.")
        _finish_successful_update()
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
        try:
            result = subprocess.run([pipx, "upgrade", "mlx-memo"], check=False, timeout=600)
        except subprocess.TimeoutExpired as exc:
            raise click.ClickException("pipx upgrade timed out (600s).") from exc
        if result.returncode != 0:
            raise click.ClickException("pipx upgrade failed.")
    elif installed_via == "uv":
        uv = _find_uv() or "uv"
        console.print("[dim]Upgrading via uv tool…[/dim]")
        try:
            result = subprocess.run([uv, "tool", "upgrade", "mlx-memo"], check=False, timeout=600)
        except subprocess.TimeoutExpired as exc:
            raise click.ClickException("uv tool upgrade timed out (600s).") from exc
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

    _finish_successful_update()
    console.print("[green]✓[/green] upgrade complete. Pre-warming MLX models…")
    _prewarm_after_update()
