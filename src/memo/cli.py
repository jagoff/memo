"""CLI — `memo` entry point.

A handful of operational commands so the user can interact with the
memory store from the shell without spinning up the MCP server:

- `memo save 'content here' --title 'X' --tag x --tag y`
- `memo search 'query' --limit 5`
- `memo list --limit 20 --type decision`
- `memo get <id>`
- `memo delete <id>`
- `memo stats`
- `memo doctor` — verify vault path, embedder loadable, sqlite-vec
  available, MLX present.

Output style:
- Default: rich table for list/search, panel for `get`, plain stats.
- `--json` flag (where applicable): emit raw JSON for piping.

NOTE: This file is 9000+ lines. Refactoring in progress via cli_commands.py.
See src/memo/cli_commands.py for structure plan.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import selectors
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.resources import files as package_files
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel
from rich.table import Table

from memo.cli_analytics import analytics_group
from memo.cli_as_of import as_of_group
from memo.cli_backend_native import backend_native_group
from memo.cli_backup import backup_group
from memo.cli_collaborative import collaborative_group
from memo.cli_common import _parse_as_of_date, _short, console
from memo.cli_common import get_memory as _get_memory
from memo.cli_consolidate import consolidate_group
from memo.cli_contextual import contextual_group
from memo.cli_contradict import contradict_group
from memo.cli_diag import _db_health_report, _doctor_report, _recall_daemon_health
from memo.cli_embed_daemon import embed_daemon_group
from memo.cli_encrypt import encrypt_group
from memo.cli_export import export_group
from memo.cli_feedback import feedback_group
from memo.cli_graph import graph_group
from memo.cli_import import import_group
from memo.cli_links import links_group
from memo.cli_memory import (
    ask,
    chat_ask,
    delete,
    embed_cmd,
    entities,
    entity,
    extract_entities,
    get,
    history,
    lint,
    list_cmd,
    ocr_image,
    provenance,
    reindex,
    rerank_cmd,
    restore,
    save,
    search,
    update,
)
from memo.cli_multimodal import multimodal_group
from memo.cli_profile import profile_group
from memo.cli_query import query_group
from memo.cli_recall_daemon import recall_daemon_group
from memo.cli_repo import repo_group
from memo.cli_session import session_group
from memo.cli_share import share_group
from memo.cli_sync import sync_group
from memo.cli_temporal import temporal_group
from memo.cli_version import version_group
from memo.config import Config

# Imported at module scope (not lazily) so tests can `patch("memo.cli.run_picker", ...)`.
# `run_picker` itself defers the heavy `questionary` import until called.
from memo.setup import run_picker, write_config_file


@click.group()
@click.version_option(package_name="mlx-memo")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """memo — local MCP memory backed by markdown vault, MLX-native."""
    _first_run_gate(ctx)


# Command groups extracted from this module live in cli_*.py and register here.
cli.add_command(graph_group)
cli.add_command(save)
cli.add_command(search)
cli.add_command(ask)
cli.add_command(embed_cmd)
cli.add_command(chat_ask)
cli.add_command(rerank_cmd)
cli.add_command(list_cmd)
cli.add_command(get)
cli.add_command(update)
cli.add_command(reindex)
cli.add_command(delete)
cli.add_command(history)
cli.add_command(ocr_image)
cli.add_command(provenance)
cli.add_command(extract_entities)
cli.add_command(entities)
cli.add_command(entity)
cli.add_command(lint)
cli.add_command(restore)
cli.add_command(profile_group)
cli.add_command(backend_native_group)
cli.add_command(feedback_group)
cli.add_command(repo_group)
cli.add_command(recall_daemon_group)
cli.add_command(embed_daemon_group)
cli.add_command(as_of_group)
cli.add_command(session_group)
cli.add_command(temporal_group)
cli.add_command(consolidate_group)
cli.add_command(contextual_group)
cli.add_command(links_group)
cli.add_command(version_group)
cli.add_command(query_group)
cli.add_command(backup_group)
cli.add_command(sync_group)
cli.add_command(encrypt_group)
cli.add_command(share_group)
cli.add_command(analytics_group)
cli.add_command(import_group)
cli.add_command(export_group)
cli.add_command(multimodal_group)
cli.add_command(collaborative_group)
cli.add_command(contradict_group)


# Subcommands that must NEVER trigger the first-run picker — either
# because they're part of setup/diagnostics, they don't need storage,
# or they run from non-interactive hooks (the TTY check + the
# `MEMO_NONINTERACTIVE=1` env var in `hooks.json` handle the latter,
# but listing the names here is a belt-and-suspenders defence in case
# something invokes them from an interactive shell while debugging).
_FIRST_RUN_GATE_SKIP_COMMANDS = {
    "init", "doctor", "migrate-vault",
    "mcp-command", "install-slash", "prewarm", "recall-hook", "recall-daemon",
    "capture-stop", "session", "ingest", "historia", "briefing", "mapa",
    "backend-native", "profile",
}


def _first_run_gate(ctx: click.Context) -> None:
    """If the user hasn't configured `memo` yet, run the picker first.

    Resolution: skip when invoked from hooks (MEMO_NONINTERACTIVE=1 or
    non-TTY), when an env var already configures `data_dir`, when a
    config file already exists, or when the legacy `MEMO_VAULT_PATH`
    pair is set (back-compat path). Also skip for setup/diagnostic
    subcommands so the user can always recover via `memo doctor`.
    """
    import os
    import sys as _sys

    if ctx.invoked_subcommand in (None, *_FIRST_RUN_GATE_SKIP_COMMANDS):
        return
    if os.environ.get("MEMO_NONINTERACTIVE") == "1":
        return
    # Both stdin and stdout must be a TTY for the picker to make sense.
    if not (_sys.stdin.isatty() and _sys.stdout.isatty()):
        return
    if "MEMO_DATA_DIR" in os.environ:
        return
    if "MEMO_VAULT_PATH" in os.environ and "MEMO_MEMORY_SUBDIR" in os.environ:
        return
    # Re-resolve the config file at gate-firing time (env may have
    # changed between import and invocation, e.g. in tests).
    from memo.setup.config_io import _resolve_config_path
    if _resolve_config_path().is_file():
        return
    _run_picker_and_save()


def _run_picker_and_save() -> None:
    """Drive the interactive picker → persist to TOML → return.

    Caller is expected to be the first-run gate (or `memo init`). Picker
    aborts (Ctrl-C / ESC) raise `click.exceptions.Exit(130)` so the
    surrounding CLI invocation halts cleanly.
    """
    console.print(
        "[bold]memo first-run setup[/bold] — pick where memorias should live.\n",
    )
    try:
        result = run_picker()
    except KeyboardInterrupt:
        console.print(
            "[yellow]aborted.[/yellow] Re-run any memo command to retry, "
            "or run `memo init` to configure.",
        )
        raise click.exceptions.Exit(130) from None
    cfg_path = write_config_file(
        data_dir=result.data_dir,
        vault_path=result.vault_path,
    )
    result.data_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[green]✓[/green] data_dir = {result.data_dir}",
    )
    if result.vault_path is not None:
        console.print(
            f"[green]✓[/green] vault_path = {result.vault_path}  "
            "[dim](used by `memo ingest`)[/dim]",
        )
    console.print(f"[dim]config saved: {cfg_path}[/dim]")


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _resolve_command(
    name: str,
    *,
    prefer_invoked: bool = False,
    sibling_of: Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Resolve an executable, preferring the active install when known."""
    candidates: list[Path] = []

    if prefer_invoked:
        invoked = Path(sys.argv[0])
        if invoked.name == name and invoked.exists():
            candidates.append(invoked)

    if sibling_of is not None:
        sibling = sibling_of.with_name(name)
        if sibling.exists():
            candidates.append(sibling)

    raw = shutil.which(name)
    if raw:
        candidates.append(Path(raw))

    for candidate in candidates:
        resolved = _safe_resolve(candidate)
        if resolved.exists():
            return candidate, resolved
    return None, None


def _env_root_for_bin(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.parent.name == "bin":
        return path.parent.parent
    return None


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _install_mode(root: Path | None) -> str:
    if root is None:
        return "unknown"
    parts = set(root.parts)
    root_s = str(root)
    if "pipx" in parts and "venvs" in parts:
        return "pipx"
    if "uv" in parts and "tools" in parts:
        return "uv tool"
    if "Cellar" in parts or root_s.startswith("/opt/homebrew/"):
        return "homebrew"
    if root.name in {".venv", "venv"} or (root / "pyvenv.cfg").is_file():
        return "venv"
    return "unknown"


def _runtime_install_report(cwd: Path | None = None) -> dict[str, Any]:
    """Describe how the active `memo`/`memo-mcp` installation is wired.

    The operationally safe shape is an isolated tool install (pipx, uv tool,
    Homebrew). A project-local `.venv` works for development, but using it as
    the MCP runtime couples memory state and MLX deps to whichever repo had
    the venv active when the client was configured.
    """
    cwd = _safe_resolve(cwd or Path.cwd())
    memo_cmd, memo_resolved = _resolve_command("memo", prefer_invoked=True)
    mcp_cmd, mcp_resolved = _resolve_command("memo-mcp", sibling_of=memo_resolved)
    py_resolved = _safe_resolve(Path(sys.executable))

    memo_root = _env_root_for_bin(memo_resolved)
    mcp_root = _env_root_for_bin(mcp_resolved)
    py_root = _env_root_for_bin(py_resolved)
    primary_root = memo_root or mcp_root or py_root
    mode = _install_mode(primary_root)

    warnings: list[str] = []
    if memo_resolved is None:
        warnings.append("`memo` is not on PATH")
    if mcp_resolved is None:
        warnings.append("`memo-mcp` is not on PATH; MCP clients cannot start it")
    if memo_root is not None and mcp_root is not None and memo_root != mcp_root:
        warnings.append(
            "`memo` and `memo-mcp` resolve to different environments; "
            "reinstall with `pipx install --force mlx-memo` or `uv tool install --force mlx-memo`"
        )
    if mode == "venv" and primary_root is not None:
        if _path_is_relative_to(primary_root, cwd):
            warnings.append(
                f"running from project venv {primary_root}; prefer an isolated "
                "tool install so MCP is not tied to this repo"
            )
        else:
            warnings.append(
                f"running from venv {primary_root}; verify this is memo's own "
                "dedicated environment, not another project's venv"
            )
    elif mode == "unknown":
        warnings.append(
            "install mode is unknown; recommended: `pipx install mlx-memo` "
            "or `uv tool install mlx-memo`"
        )

    return {
        "mode": mode,
        "root": str(primary_root) if primary_root else None,
        "memo_cmd": str(memo_cmd) if memo_cmd else None,
        "memo_resolved": str(memo_resolved) if memo_resolved else None,
        "mcp_cmd": str(mcp_cmd) if mcp_cmd else None,
        "mcp_resolved": str(mcp_resolved) if mcp_resolved else None,
        "python": str(py_resolved),
        "warnings": warnings,
    }


_MCP_ENV_FORWARD_KEYS = (
    "MEMO_CONFIG_FILE",
    "MEMO_DATA_DIR",
    "MEMO_STATE_DIR",
    "MEMO_VAULT_PATH",
    "MEMO_MEMORY_SUBDIR",
    "MEMO_MODEL_PROFILE",
    "MEMO_LLM_MODEL",
    "MEMO_HELPER_MODEL",
    "MEMO_EMBEDDER_MODEL",
    "MEMO_EMBEDDER_DIMS",
    "MEMO_RERANKER_ENABLED",
    "MEMO_RERANKER_MODEL",
    "MEMO_RERANKER_REVISION",
    "MEMO_RERANK_INPUT_K",
    "MEMO_RERANK_FUSION_ALPHA",
)


def _mcp_server_env() -> dict[str, str]:
    """Env vars that MCP clients should pin when launching `memo-mcp`."""
    env = {"MEMO_NONINTERACTIVE": "1"}
    for key in _MCP_ENV_FORWARD_KEYS:
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


def _env_flags(client: str, env: dict[str, str]) -> list[str]:
    opt = "--env" if client == "codex" else "-e"
    flags: list[str] = []
    for key, val in env.items():
        flags.extend([opt, f"{key}={val}"])
    return flags


def _format_command(args: Sequence[str | Path]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _mcp_server_json(memo_mcp: Path, env: dict[str, str], *, include_type: bool) -> dict[str, Any]:
    config: dict[str, Any] = {
        "command": str(memo_mcp),
        "args": [],
        "env": env,
    }
    if include_type:
        config = {"type": "stdio", **config}
    return config


def _agent_asset_root(repo: Path | None = None) -> Path:
    candidates = [_safe_resolve(repo)] if repo else [
        _safe_resolve(Path.cwd()),
        _safe_resolve(Path(__file__).resolve().parents[2]),
    ]
    if not repo:
        try:
            packaged_assets = Path(str(package_files("memo") / "agent_assets"))
        except Exception:
            packaged_assets = None
        if packaged_assets is not None:
            candidates.append(_safe_resolve(packaged_assets))
    for root in candidates:
        if (
            (root / ".claude-plugin" / "plugin.json").is_file()
            and (root / "commands" / "memo.md").is_file()
            and (root / "plugins" / "memo" / ".codex-plugin" / "plugin.json").is_file()
            and (root / "plugins" / "memo" / "skills" / "memo" / "SKILL.md").is_file()
            and (root / "skills" / "memo" / "SKILL.md").is_file()
        ):
            return root
    checked = ", ".join(str(c) for c in candidates)
    raise click.ClickException(
        "Could not find memo plugin assets. Run from the memo checkout or pass "
        f"`--repo /path/to/memo`. Checked: {checked}"
    )


def _run_agent_command(
    args: list[str | Path],
    *,
    dry_run: bool,
    ok_errors: tuple[str, ...] = (),
) -> None:
    if dry_run:
        console.print(f"[dim]$ {_format_command(args)}[/dim]")
        return
    try:
        proc = subprocess.run(
            [str(arg) for arg in args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"`{args[0]}` not found on PATH; install that client first."
        ) from exc
    combined = f"{proc.stdout}\n{proc.stderr}".lower()
    if proc.returncode != 0 and not any(token in combined for token in ok_errors):
        detail = (proc.stderr or proc.stdout or "").strip()
        raise click.ClickException(
            f"Command failed ({proc.returncode}): {_format_command(args)}"
            + (f"\n{detail}" if detail else "")
        )
    if proc.returncode == 0:
        console.print(f"[green]✓[/green] {_format_command(args)}")
    else:
        console.print(f"[dim]↷ already handled: {_format_command(args)}[/dim]")


def _mcp_add_command(client: str, memo_mcp: Path, env: dict[str, str]) -> list[str | Path]:
    if client == "codex":
        return ["codex", "mcp", "add", "memo", *_env_flags("codex", env), "--", memo_mcp]
    if client == "devin":
        return [
            "devin", "mcp", "add", "-s", "user",
            *_env_flags("devin", env), "memo", "--", memo_mcp,
        ]
    mcp_json = json.dumps(_mcp_server_json(memo_mcp, env, include_type=True), separators=(",", ":"))
    return [
        "claude", "mcp", "add-json", "-s", "user", "memo", mcp_json,
    ]


def _windsurf_mcp_config_path() -> Path:
    raw = os.environ.get("WINDSURF_MCP_CONFIG")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".codeium" / "windsurf" / "mcp_config.json"


def _install_windsurf_mcp(memo_mcp: Path, env: dict[str, str], *, dry_run: bool) -> None:
    path = _windsurf_mcp_config_path()
    server_config = _mcp_server_json(memo_mcp, env, include_type=False)
    if dry_run:
        console.print(
            f"[dim]write {path}  # mcpServers.memo = "
            f"{json.dumps(server_config, ensure_ascii=False, separators=(',', ':'))}[/dim]"
        )
        return

    data: dict[str, Any]
    if path.is_file() and path.read_text(encoding="utf-8").strip():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Windsurf MCP config is not valid JSON: {path} ({exc})"
            ) from exc
        if not isinstance(loaded, dict):
            raise click.ClickException(f"Windsurf MCP config must be a JSON object: {path}")
        data = loaded
    else:
        data = {}

    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise click.ClickException(f"`mcpServers` must be a JSON object in {path}")
    servers["memo"] = server_config

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]✓[/green] wrote Windsurf MCP config: {path}")


def _codex_home() -> Path:
    raw = os.environ.get("CODEX_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".codex"


def _copy_slash_skill(root: Path, dst: Path, *, dry_run: bool) -> None:
    src = root / "skills" / "memo" / "SKILL.md"
    if dry_run:
        console.print(f"[dim]copy {src} -> {dst}  # /memo[/dim]")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    console.print(f"[green]✓[/green] copied {src} -> {dst}  [dim](/memo)[/dim]")


_MISSING_MCP_OK_ERRORS = (
    "not found",
    "unknown",
    "no such",
    "does not exist",
    "no user-scoped mcp server found",
    "no project-scoped mcp server found",
    "no local-scoped mcp server found",
    "no mcp server found",
)


def _codex_send_app_server_request(
    proc: subprocess.Popen[str],
    *,
    request_id: int,
    method: str,
    params: dict[str, Any] | None,
) -> None:
    if proc.stdin is None:
        raise click.ClickException("Codex app-server stdin is unavailable.")
    payload: dict[str, Any] = {"id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    proc.stdin.flush()


def _codex_read_app_server_response(
    proc: subprocess.Popen[str],
    request_id: int,
    *,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    if proc.stdout is None:
        raise click.ClickException("Codex app-server stdout is unavailable.")

    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_s
    seen: list[str] = []

    try:
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if not selector.select(remaining):
                break
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            seen.append(line)
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") != request_id:
                continue
            if "error" in msg:
                err = msg["error"]
                message = err.get("message", err) if isinstance(err, dict) else err
                raise click.ClickException(f"Codex app-server {request_id} failed: {message}")
            result = msg.get("result")
            return result if isinstance(result, dict) else {}
    finally:
        selector.close()

    preview = "\n".join(seen[-5:])
    raise click.ClickException(
        f"Timed out waiting for Codex app-server response id={request_id}."
        + (f"\nLast output:\n{preview}" if preview else "")
    )


def _install_codex_plugin(root: Path, *, dry_run: bool) -> None:
    marketplace = root / ".agents" / "plugins" / "marketplace.json"
    if not marketplace.is_file():
        raise click.ClickException(
            f"Codex marketplace manifest not found: {marketplace}"
        )
    args = ["codex", "app-server", "--listen", "stdio://", "--enable", "plugins"]
    if dry_run:
        console.print(
            f"[dim]$ {_format_command(args)}  # plugin/install memo from {marketplace}[/dim]"
        )
        return

    try:
        proc = subprocess.Popen(
            args,
            cwd=str(root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(
            "`codex` not found on PATH; install Codex first."
        ) from exc

    try:
        _codex_send_app_server_request(
            proc,
            request_id=0,
            method="initialize",
            params={
                "clientInfo": {
                    "name": "memo-installer",
                    "title": "memo installer",
                    "version": "0.1",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        _codex_read_app_server_response(proc, 0)
        _codex_send_app_server_request(
            proc,
            request_id=1,
            method="plugin/install",
            params={
                "marketplacePath": str(marketplace),
                "pluginName": "memo",
            },
        )
        _codex_read_app_server_response(proc, 1)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    console.print("[green]✓[/green] codex plugin/install memo@memo")


def _print_runtime_install_report(report: dict[str, Any]) -> None:
    mode = report["mode"]
    root = report.get("root") or "(unknown)"
    if report["warnings"]:
        console.print(f"[yellow]![/yellow] install mode: {mode}  [dim]{root}[/dim]")
    else:
        console.print(f"[green]✓[/green] install mode: {mode}  [dim]{root}[/dim]")

    for key, label in (
        ("memo_cmd", "memo"),
        ("mcp_cmd", "memo-mcp"),
    ):
        raw = report.get(key)
        resolved = report.get(key.replace("_cmd", "_resolved"))
        if raw and resolved and raw != resolved:
            console.print(f"[dim]{label:14s}[/dim] {raw} -> {resolved}")
        elif raw:
            console.print(f"[dim]{label:14s}[/dim] {raw}")
        else:
            console.print(f"[dim]{label:14s}[/dim] (not found)")
    console.print(f"[dim]{'python':14s}[/dim] {report['python']}")
    for warning in report["warnings"]:
        console.print(f"[yellow]![/yellow] {warning}")


def _resolved_memo_mcp() -> Path | None:
    _, memo_resolved = _resolve_command("memo", prefer_invoked=True)
    if memo_resolved is not None:
        sibling = memo_resolved.with_name("memo-mcp")
        if sibling.exists():
            return sibling
    _, resolved = _resolve_command("memo-mcp")
    if resolved is not None:
        return resolved
    return None


@cli.command(name="init")
@click.option("--force", is_flag=True, help="Overwrite existing config without confirmation.")
def init_cmd(force: bool) -> None:
    """(Re)configure where memo stores memorias.

    Runs the interactive picker. On first run, the picker also fires
    automatically — `memo init` is for explicitly re-configuring later
    (e.g. moving to a new path, switching to/from an Obsidian vault).
    """
    from memo.setup.config_io import _resolve_config_path

    cfg_path = _resolve_config_path()
    if cfg_path.is_file() and not force and not click.confirm(
        f"Config file exists at {cfg_path}. Overwrite?",
        default=False,
    ):
        console.print("[yellow]aborted[/yellow]")
        return
    _run_picker_and_save()


@cli.command(name="migrate-vault")
@click.argument("new_data_dir", required=False, type=click.Path(file_okay=False, resolve_path=True))
@click.option("--from", "from_dir", default=None,
              type=click.Path(exists=True, file_okay=False, resolve_path=True),
              help="Source memory_dir. Defaults to current cfg.memory_dir.")
@click.option("--force", is_flag=True, help="Overwrite destination even if non-empty.")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def migrate_vault(
    new_data_dir: str | None, from_dir: str | None, force: bool, yes: bool,
) -> None:
    """Move memorias to a new data_dir; rewrites config + reindexes.

    Copies all `.md` files (preserving mtime via `shutil.copy2`),
    updates `~/.config/memo/config.toml` with the new `data_dir`,
    deletes `memvec.db`, and runs `memo reindex` from the new location.

    The original `.md` files are NOT deleted — once you've verified the
    migration with `memo search`, you can `rm -rf <old-dir>` manually.
    History DB is preserved (it's append-only audit; old paths in it
    just become historical references).
    """
    import shutil
    from pathlib import Path as _Path

    from memo.memory import Memory

    cfg = Config.from_env()

    # 1. Resolve source.
    src = _Path(from_dir).resolve() if from_dir else cfg.memory_dir
    if not src.is_dir():
        console.print(f"[red]✗[/red] source dir does not exist: {src}")
        sys.exit(1)

    # 2. Resolve destination + (optional) new vault_path.
    if new_data_dir:
        dst = _Path(new_data_dir).resolve()
        chosen_vault = cfg.vault_path
    else:
        # No arg → run the picker so user can pick (incl. an Obsidian vault).
        try:
            result = run_picker()
        except KeyboardInterrupt:
            console.print("[yellow]aborted[/yellow]")
            sys.exit(130)
        dst = result.data_dir
        chosen_vault = result.vault_path

    if dst == src:
        console.print(f"[red]✗[/red] source and destination are the same: {src}")
        sys.exit(1)

    md_files = sorted(src.rglob("*.md"))
    if dst.exists() and any(dst.iterdir()) and not force:
        console.print(
            f"[red]✗[/red] destination is non-empty: {dst}\n"
            "  Use --force to overwrite.",
        )
        sys.exit(1)

    if not yes:
        click.confirm(
            f"Copy {len(md_files)} memorias from\n  {src}\n→ {dst}\n"
            "and rebuild memvec.db. Source files will be left in place. "
            "Proceed?",
            abort=True,
        )

    # 3. Copy files (preserving mtime).
    dst.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for md in md_files:
        rel = md.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md, target)
        n_copied += 1
    console.print(f"[green]✓[/green] copied {n_copied} files → {dst}")

    # 4. Update config + drop stale DB.
    cfg_path = write_config_file(data_dir=dst, vault_path=chosen_vault)
    console.print(f"[green]✓[/green] config: {cfg_path}")
    if cfg.db_path.is_file():
        cfg.db_path.unlink()
        console.print("[green]✓[/green] removed stale memvec.db")

    # 5. Reindex from new location. Re-build Config so from_env picks up
    # the freshly-written file (env vars / explicit kwargs cleared).
    new_cfg = Config.from_env()
    mem = Memory(new_cfg)
    counts = mem.reindex()
    console.print(
        f"[green]✓[/green] reindex: checked {counts['checked']}  "
        f"added {counts['added']}  reindexed {counts['reindexed']}  "
        f"skipped {counts['skipped']}",
    )
    console.print(
        f"\n[dim]Source files at {src} were left untouched. "
        "After verifying the migration with `memo search`, you can rm them.[/dim]",
    )








@cli.command()
def stats() -> None:
    """Summary stats — total records, vault path, embedder model."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    history_errors = 0
    with contextlib.suppress(Exception):
        history_errors = int(getattr(mem.history, "error_count", 0))
    info: dict[str, Any] = {
        "total": mem.store.count(),
        "data_dir": str(mem.cfg.data_dir),
        "vault_path": str(mem.cfg.vault_path) if mem.cfg.vault_path else "(unset)",
        "db_path": str(mem.cfg.db_path),
        "model_profile": mem.cfg.model_profile,
        "embedder_model": mem.cfg.embedder_model,
        "llm_model": mem.cfg.llm_model,
        "history_errors": history_errors,
    }
    for k, v in info.items():
        console.print(f"[dim]{k:14s}[/dim] {v}")




@cli.command()
@click.option("--gc", "do_gc", is_flag=True, help="Detect orphans between store and disk.")
@click.option("--fix", is_flag=True, help="With --gc: drop orphan store rows. .md files are never deleted automatically.")
@click.option("--db", "check_db", is_flag=True, help="Run read-only integrity checks on managed sqlite DBs.")
@click.option(
    "--strict-runtime",
    is_flag=True,
    help="Exit non-zero if memo/memo-mcp are not running from an isolated tool install.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit a stable JSON health report.")
def doctor(do_gc: bool, fix: bool, check_db: bool, strict_runtime: bool, as_json: bool) -> None:
    """Self-check: vault present, sqlite-vec loadable, MLX importable, models in cache.

    `--gc` reports orphans (store rows whose `.md` is gone, `.md` files
    whose `id` isn't in the store). `--gc --fix` removes orphan store
    rows; orphan `.md` files are listed but never deleted automatically.
    """
    cfg = Config.from_env()
    if as_json:
        report = _doctor_report(
            cfg,
            check_db=check_db,
            strict_runtime=strict_runtime,
            do_gc=do_gc,
            fix=fix,
        )
        click.echo(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        sys.exit(0 if report["ok"] else 1)

    ok = True

    runtime_report = _runtime_install_report()
    _print_runtime_install_report(runtime_report)
    if strict_runtime and runtime_report["warnings"]:
        ok = False

    # 1. Data dir (memorias)
    if cfg.data_dir.is_dir():
        console.print(f"[green]✓[/green] data_dir: {cfg.data_dir}")
    else:
        # Data dir is auto-created by `ensure_dirs()`; missing here means
        # something went wrong with permissions.
        console.print(f"[red]✗[/red] data_dir missing: {cfg.data_dir}")
        ok = False
    # Optional vault_path (only relevant for `memo ingest`).
    if cfg.vault_path is not None:
        if cfg.vault_path.is_dir():
            console.print(f"[green]✓[/green] vault_path: {cfg.vault_path}")
        else:
            console.print(f"[yellow]![/yellow] vault_path set but missing: {cfg.vault_path}")

    # 2. sqlite-vec
    try:
        import sqlite3

        import sqlite_vec  # type: ignore[import-untyped]

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        console.print("[green]✓[/green] sqlite-vec loadable")
    except Exception as exc:
        console.print(f"[red]✗[/red] sqlite-vec: {exc}")
        ok = False

    # 2b. FTS5 (BM25 backbone — hybrid search silently downgrades without it)
    try:
        import sqlite3 as _sqlite3_fts

        _c = _sqlite3_fts.connect(":memory:")
        try:
            _c.execute("CREATE VIRTUAL TABLE _fts_probe USING fts5(x)")
            _c.execute("DROP TABLE _fts_probe")
        finally:
            _c.close()
        console.print("[green]✓[/green] sqlite FTS5 available")
    except Exception as exc:
        console.print(
            f"[red]✗[/red] sqlite FTS5 unavailable: {exc}  "
            "[dim](BM25/hybrid search will degrade to vec-only)[/dim]"
        )
        ok = False

    # 3. MLX importable
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401

        console.print("[green]✓[/green] mlx + mlx_lm importable")
    except Exception as exc:
        console.print(f"[red]✗[/red] mlx: {exc}")
        ok = False

    # 4. Models in HF cache
    from pathlib import Path

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for model in (cfg.embedder_model, cfg.llm_model, cfg.helper_model):
        cache_dir = hf_cache / f"models--{model.replace('/', '--')}"
        if cache_dir.is_dir():
            console.print(f"[green]✓[/green] cached: {model}")
        else:
            console.print(
                f"[yellow]![/yellow] not cached: {model}  "
                f"[dim](run `hf download {model}`)[/dim]",
            )

    # 5. Recall daemon health (best-effort, never blocks)
    daemon = _recall_daemon_health(cfg)
    if daemon.get("running"):
        console.print(
            f"[green]✓[/green] recall-daemon: running "
            f"(pid={daemon.get('pid')}, ping ok)"
        )
    elif daemon.get("pid_alive") and not daemon.get("ping_ok"):
        err = daemon.get("error") or "no ping response"
        console.print(
            f"[yellow]![/yellow] recall-daemon: pid {daemon.get('pid')} alive "
            f"but socket unresponsive ({err})"
        )
    elif daemon.get("socket_exists") and not daemon.get("pid_alive"):
        console.print(
            "[yellow]![/yellow] recall-daemon: stale socket without process — "
            "run `memo recall-daemon stop` to clean up"
        )
    else:
        console.print(
            "[dim]•[/dim] recall-daemon: not running "
            "[dim](`memo recall-daemon start` to enable warm recall)[/dim]"
        )

    if check_db:
        for db in _db_health_report(cfg):
            marker = "[green]✓[/green]" if db["ok"] else "[red]✗[/red]"
            if db["exists"]:
                console.print(
                    f"{marker} db:{db['label']} {db['status']} "
                    f"integrity={db.get('integrity_check', '-')} "
                    f"tables={db.get('table_count', 0)} "
                    f"size={db.get('size_bytes', 0)}"
                )
                if db.get("vec_dims") is not None:
                    console.print(
                        f"  dims vec={db.get('vec_dims')} "
                        f"repo_vec={db.get('repo_vec_dims')} "
                        f"expected={db.get('expected_dims')}"
                    )
                if db.get("latest_memory_update"):
                    console.print(f"  latest_memory_update={db['latest_memory_update']}")
                if db.get("latest_repo_index"):
                    console.print(f"  latest_repo_index={db['latest_repo_index']}")
            else:
                console.print(f"[yellow]![/yellow] db:{db['label']} missing: {db['path']}")
            if not db["ok"]:
                ok = False

    if do_gc:
        from memo.memory import Memory

        mem = Memory(cfg)
        report = mem.gc(fix=fix)
        n_store = len(report["orphan_store"])
        n_disk = len(report["orphan_disk"])
        if n_store == 0 and n_disk == 0:
            console.print("[green]✓[/green] no orphans")
        else:
            if n_store:
                verb = "dropped" if fix else "found"
                console.print(
                    f"[yellow]{verb} {n_store} orphan store row(s)[/yellow] "
                    f"(in store, .md missing)",
                )
                for oid in report["orphan_store"][:20]:
                    console.print(f"  · {oid}")
                if n_store > 20:
                    console.print(f"  · …and {n_store - 20} more")
            if n_disk:
                console.print(
                    f"[yellow]found {n_disk} orphan .md file(s)[/yellow] "
                    f"(on disk, not in store — try `memo reindex`)",
                )
                for p in report["orphan_disk"][:20]:
                    console.print(f"  · {p}")
                if n_disk > 20:
                    console.print(f"  · …and {n_disk - 20} more")

    sys.exit(0 if ok else 1)


@cli.command(name="mcp-command")
@click.option(
    "--client",
    type=click.Choice(["claude-code", "claude-desktop", "codex", "devin", "windsurf", "json"]),
    default="claude-code",
    show_default=True,
    help="Emit a client-specific MCP registration command or raw JSON config.",
)
def mcp_command(client: str) -> None:
    """Print MCP config pinned to the resolved `memo-mcp` executable.

    This avoids accidentally registering a `memo-mcp` from another active
    project venv. Pair with `memo doctor --strict-runtime` when debugging a
    client that starts the wrong server.
    """
    memo_mcp = _resolved_memo_mcp()
    if memo_mcp is None:
        console.print(
            "[red]memo-mcp not found.[/red] Install memo as an isolated tool: "
            "`pipx install mlx-memo` or `uv tool install mlx-memo`.",
        )
        sys.exit(1)
    env = _mcp_server_env()
    if client == "json":
        click.echo(json.dumps({
            "mcpServers": {
                "memo": _mcp_server_json(memo_mcp, env, include_type=True),
            },
        }, ensure_ascii=False, indent=2))
        return
    if client == "claude-desktop":
        click.echo(json.dumps({
            "mcpServers": {
                "memo": _mcp_server_json(memo_mcp, env, include_type=False),
            },
        }, ensure_ascii=False, indent=2))
        return
    if client == "windsurf":
        click.echo(json.dumps({
            "mcpServers": {
                "memo": _mcp_server_json(memo_mcp, env, include_type=False),
            },
        }, ensure_ascii=False, indent=2))
        return
    if client == "codex":
        click.echo(_format_command(_mcp_add_command("codex", memo_mcp, env)))
        return
    if client == "devin":
        click.echo(_format_command(_mcp_add_command("devin", memo_mcp, env)))
        return
    click.echo(_format_command(_mcp_add_command("claude-code", memo_mcp, env)))


@cli.command(name="install-slash")
@click.option(
    "--client",
    "clients",
    multiple=True,
    type=click.Choice(["all", "claude-code", "codex", "devin", "windsurf"]),
    help="Client to configure. Repeatable. Defaults to all supported agent clients.",
)
@click.option(
    "--repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to the memo checkout containing plugin/skill assets.",
)
@click.option("--dry-run", is_flag=True, help="Print the commands without changing configs.")
@click.option(
    "--best-effort",
    is_flag=True,
    help="Warn and continue if a client CLI is missing or rejects configuration.",
)
def install_slash(
    clients: tuple[str, ...],
    repo: Path | None,
    dry_run: bool,
    best_effort: bool,
) -> None:
    """Install the `/memo` skill/plugin assets for supported agent CLIs.

    This configures the client-specific skill/plugin assets and an MCP server
    named `memo` pinned to the resolved `memo-mcp` binary. Current MEMO_*
    model/storage env vars are forwarded into the MCP config so agent clients
    do not accidentally boot with a different embedder profile.
    """
    selected = set(clients or ("all",))
    if "all" in selected:
        selected.remove("all")
        selected.update({"claude-code", "codex", "devin", "windsurf"})

    needs_assets = bool(selected & {"claude-code", "codex", "devin"})
    root = _agent_asset_root(repo) if needs_assets else None
    memo_mcp = _resolved_memo_mcp()
    if memo_mcp is None:
        raise click.ClickException(
            "memo-mcp not found. Install memo as an isolated tool first: "
            "`pipx install mlx-memo` or `uv tool install mlx-memo`."
        )
    env = _mcp_server_env()

    failures: list[str] = []

    def run_client(label: str, fn) -> None:
        try:
            fn()
        except click.ClickException as exc:
            if not best_effort:
                raise
            failures.append(label)
            console.print(f"[yellow]![/yellow] {label}: {exc.message}")

    def install_codex() -> None:
        assert root is not None
        console.print("[bold]Codex[/bold]")
        _copy_slash_skill(
            root,
            _codex_home() / "skills" / "memo" / "SKILL.md",
            dry_run=dry_run,
        )
        _install_codex_plugin(root, dry_run=dry_run)
        _run_agent_command(
            ["codex", "mcp", "remove", "memo"],
            dry_run=dry_run,
            ok_errors=_MISSING_MCP_OK_ERRORS,
        )
        _run_agent_command(_mcp_add_command("codex", memo_mcp, env), dry_run=dry_run)
        console.print(
            "[yellow]![/yellow] Codex CLI's TUI slash menu currently lists only built-in "
            "slash commands. memo is installed as a model-visible skill and MCP server, "
            "but `/memo` may not appear in that menu."
        )

    def install_claude_code() -> None:
        assert root is not None
        console.print("[bold]Claude Code[/bold]")
        _run_agent_command(
            ["claude", "plugin", "marketplace", "add", root],
            dry_run=dry_run,
            ok_errors=("already", "exists"),
        )
        _run_agent_command(
            ["claude", "plugin", "install", "memo@memo", "-s", "user"],
            dry_run=dry_run,
            ok_errors=("already", "installed", "exists"),
        )
        _run_agent_command(
            ["claude", "mcp", "remove", "-s", "user", "memo"],
            dry_run=dry_run,
            ok_errors=_MISSING_MCP_OK_ERRORS,
        )
        _run_agent_command(_mcp_add_command("claude-code", memo_mcp, env), dry_run=dry_run)

    def install_devin() -> None:
        assert root is not None
        console.print("[bold]Devin[/bold]")
        _copy_slash_skill(
            root,
            Path.home() / ".config" / "devin" / "skills" / "memo" / "SKILL.md",
            dry_run=dry_run,
        )
        _run_agent_command(
            ["devin", "mcp", "remove", "-s", "user", "memo"],
            dry_run=dry_run,
            ok_errors=_MISSING_MCP_OK_ERRORS,
        )
        _run_agent_command(_mcp_add_command("devin", memo_mcp, env), dry_run=dry_run)

    def install_windsurf() -> None:
        console.print("[bold]Windsurf[/bold]")
        _install_windsurf_mcp(memo_mcp, env, dry_run=dry_run)
        console.print("[dim]Refresh MCP servers in Windsurf Cascade after editing config.[/dim]")

    if "codex" in selected:
        run_client("Codex", install_codex)
    if "claude-code" in selected:
        run_client("Claude Code", install_claude_code)
    if "devin" in selected:
        run_client("Devin", install_devin)
    if "windsurf" in selected:
        run_client("Windsurf", install_windsurf)

    if failures:
        console.print(
            "[yellow]![/yellow] agent-client install finished with skipped clients: "
            + ", ".join(failures)
        )
        console.print(
            "[dim]Install those clients, then rerun: "
            "memo install-slash --client claude-code --client codex --client windsurf[/dim]"
        )
    else:
        console.print("[green]✓[/green] agent-client install complete. Open a new agent session to reload.")


# ── Ambient memory hooks (v0.3.0) ──────────────────────────────────────────
#
# `recall-hook` and `prewarm` are designed to be wired into Claude Code's
# `UserPromptSubmit` and `SessionStart` hooks respectively (see
# `hooks/hooks.json` in this plugin / repo). They turn memo from a manual
# memory store into an **ambient** context layer: the agent automatically
# sees relevant memories before answering, with zero `/memo` invocations
# from the user.
#
# Both commands fail SILENTLY by design — a hook crash must never block
# Claude Code's prompt submission. On any error (DB locked, model load
# failure, malformed stdin) we exit 0 with empty stdout, and Claude Code
# proceeds without injection.


@cli.command(name="recall-hook")
def recall_hook() -> None:
    """UserPromptSubmit hook — inject relevant memorias as additionalContext.

    Reads a JSON object from stdin (Claude Code hook format), embeds the
    `prompt` field via the MLX embedder, runs search, and outputs
    the top-k results as `additionalContext` in `hookSpecificOutput`.

    Configure via env vars (all optional, sensible defaults for v0.3.x):

      MEMO_RECALL_DISABLE          — set to "1" to make this a no-op.
      MEMO_RECALL_TOP_K            — default 3
      MEMO_RECALL_MIN_SIM          — default 0.6. Note: the floor is
        absolute over `score`. For mode=vec, `score` is cosine ∈ [0, 1].
        For mode=hybrid+rerank, `score` is fused (typically [0.2, 0.95])
        — drop the floor to ~0.4 there or hits get over-filtered.
      MEMO_RECALL_MIN_PROMPT_CHARS — default 12 (skip very short prompts)
      MEMO_RECALL_BODY_CHARS       — default 240 (snippet length per result)
      MEMO_RECALL_SKIP_SLASH       — default "1" (skip if prompt starts with /)
      MEMO_RECALL_MODE             — default "vec". "hybrid" enables
        cross-encoder rerank on top of RRF fusion. Higher precision,
        higher latency (≤5s warm with the auto-capped pool).
      MEMO_RECALL_RERANK_INPUT_K   — default 10 (only used when MODE=hybrid).
        How many fused candidates to feed the reranker; lower for tighter
        latency, higher for better recall on diffuse queries.

    Output (stdout, JSON):
      `{}` — no injection (no results / disabled / error / silent fail)
      `{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                               "additionalContext": "<markdown>"}}`
    """
    import json as _json
    import os
    import sys as _sys

    # Always exit 0 — hooks must not block Claude Code on memo failures.
    def _bail(reason: str = "") -> None:
        if reason and os.environ.get("MEMO_RECALL_DEBUG") == "1":
            print(f"# memo recall-hook: {reason}", file=_sys.stderr)
        # Always append to recall.log so `memo logs` surfaces bails even when
        # MEMO_RECALL_DEBUG is unset. Best-effort; swallows internal errors.
        if reason:
            try:
                from memo.dashboard import append_recall_log
                append_recall_log(
                    Config.from_env().state_dir,
                    prompt="",
                    hits=[],
                    via="bail",
                    reason=reason,
                )
            except Exception:
                pass
        print("{}")
        _sys.exit(0)

    if os.environ.get("MEMO_RECALL_DISABLE") == "1":
        _bail("disabled via MEMO_RECALL_DISABLE")
        return

    # Read stdin (Claude Code passes hook input as JSON).
    try:
        raw = _sys.stdin.read()
        if not raw.strip():
            _bail("empty stdin")
            return
        payload = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        _bail(f"stdin parse fail: {exc}")
        return

    prompt = (payload.get("prompt") or "").strip()
    min_chars = int(os.environ.get("MEMO_RECALL_MIN_PROMPT_CHARS", "12"))
    if len(prompt) < min_chars:
        _bail(f"prompt too short ({len(prompt)} < {min_chars})")
        return

    # Skip slash commands by default — the user is invoking another command,
    # injecting recall context would be noise. Override with
    # MEMO_RECALL_SKIP_SLASH=0 if you want recall on /memo etc.
    if os.environ.get("MEMO_RECALL_SKIP_SLASH", "1") == "1" and prompt.startswith("/"):
        _bail("slash command, skip recall")
        return

    # Fast path: try the recall daemon socket first (<200 ms when running).
    # If the socket is not there or the connection fails, fall through to
    # the regular in-process search below. The daemon returns the same JSON
    # format as the subprocess path, so we can print-and-exit immediately.
    _t0 = time.time()
    try:
        from memo.recall_server import connect_and_recall
        _daemon_result = connect_and_recall(
            Config.from_env().state_dir,
            prompt=prompt,
            cwd=payload.get("cwd"),
            timeout=1.0,
        )
        if _daemon_result is not None:
            _latency_ms = int((time.time() - _t0) * 1000)
            if os.environ.get("MEMO_RECALL_DEBUG") == "1":
                print(f"# memo recall-hook: daemon hit ({_latency_ms} ms)", file=_sys.stderr)
            # Log to recall.log (daemon path already logs internally, but
            # update the 'via' field for hook-log observability)
            print(_daemon_result)
            _sys.exit(0)
    except Exception as _daemon_exc:
        # Don't bail — fall back to in-process search below. Telemetry only:
        # capture daemon failure so `memo logs` can surface why we paid the
        # cold-start cost.
        try:
            from memo.dashboard import append_recall_log
            append_recall_log(
                Config.from_env().state_dir,
                prompt=prompt,
                hits=[],
                via="daemon_error",
                error=f"{type(_daemon_exc).__name__}: {_daemon_exc}",
            )
        except Exception:
            pass

    top_k = int(os.environ.get("MEMO_RECALL_TOP_K", "3"))
    min_sim = float(os.environ.get("MEMO_RECALL_MIN_SIM", "0.6"))
    body_chars = int(os.environ.get("MEMO_RECALL_BODY_CHARS", "240"))
    token_budget = int(os.environ.get("MEMO_RECALL_TOKEN_BUDGET", "0") or 0)
    project_boost = float(os.environ.get("MEMO_RECALL_PROJECT_BOOST", "0.15"))

    # Read cwd from the hook payload (Claude Code passes it) so we can
    # derive the project tag the user is currently working under.
    payload_cwd = payload.get("cwd")

    # Suppress HF download progress bars on stderr — they'd contaminate
    # the hook's debug output and confuse users tailing logs. The model
    # is already downloaded for any working memo install; this only
    # silences first-run cache-check noise.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # Defer the heavy import — only paid if we get past the early-exits.
    #
    # Mode selection:
    # - `vec` (default): pure cosine similarity. `score` is in [0, 1]
    #   and the `MEMO_RECALL_MIN_SIM` floor (0.6) cuts noise reliably.
    # - `hybrid`: reciprocal rank fusion + cross-encoder rerank when
    #   the reranker is enabled in config. Higher precision but spends
    #   the full hook budget on inference. Auto-shrinks the rerank
    #   input pool (`MEMO_RECALL_RERANK_INPUT_K`, default 10) so the
    #   warm latency stays under the 5s hook timeout.
    # - `bm25`: keyword-only, useful for queries with literal tag
    #   names or filenames where the embedder under-recalls.
    mode = os.environ.get("MEMO_RECALL_MODE", "vec")
    if mode == "hybrid":
        # Hook latency budget is tight: cap the rerank pool unless the
        # user explicitly set MEMO_RERANK_INPUT_K elsewhere. Setdefault
        # respects an upstream override (CI bench, custom shell rc).
        os.environ.setdefault(
            "MEMO_RERANK_INPUT_K",
            os.environ.get("MEMO_RECALL_RERANK_INPUT_K", "10"),
        )

    # Warm-signal check: if prewarm hasn't run recently and the requested
    # mode needs the embedder, downgrade to bm25 to avoid cold-load timeout.
    # The signal file is written by `memo prewarm` after a successful warm.
    # A missing or >60 min old file means the Mac just woke / first boot.
    if mode in ("vec", "hybrid") and os.environ.get("MEMO_RECALL_FORCE_MODE") != "1":
        try:
            import time as _time_mod
            _signal = Config.from_env().state_dir / ".prewarm_ts"
            _warm = (
                _signal.exists()
                and (_time_mod.time() - float(_signal.read_text().strip())) < 3600
            )
            if not _warm:
                if os.environ.get("MEMO_RECALL_DEBUG") == "1":
                    print("# memo recall-hook: cold start — downgrading to bm25", file=_sys.stderr)
                mode = "bm25"
        except Exception:
            pass

    # Widen the pool when a project boost is active — we need enough
    # candidates so that off-project hits can be re-ranked below
    # on-project ones without starving the final top_k.
    project_tag = None
    if project_boost > 0:
        try:
            from memo.project import current_project_tag
            project_tag = current_project_tag(payload_cwd)
        except Exception:
            project_tag = None
    search_k = top_k * 3 if project_tag else top_k
    try:
        from memo.memory import Memory
        mem = Memory(Config.from_env())
        hits = mem.search(prompt, limit=search_k, mode=mode)
    except Exception as exc:
        _bail(f"search failed: {exc}")
        return

    # Apply project boost — additive on the raw score, then re-sort.
    if project_tag:
        from memo.recall_server import _apply_project_boost
        hits = _apply_project_boost(hits, project_tag, project_boost)
    # Trim back to top_k after boost-aware re-sort.
    hits = hits[:top_k]

    # Filter by similarity floor. With mode="vec", `score` is cosine
    # similarity ∈ [-1, 1] (typically [0, 1] for L2-normalised embeddings).
    # 0.6 is the empirical confidence floor on the 223-doc corpus:
    #   - "qué decidí sobre MLX vs Ollama" → 3 hits @ 0.71-0.74 (all relevant)
    #   - "how to bake apple pie" → 3 hits @ 0.51-0.56 (literal-word noise,
    #     "apple-mcp" memoria matched). Threshold 0.6 cuts these out.
    # Tune via MEMO_RECALL_MIN_SIM if your corpus has different density.
    relevant = [h for h in hits if h.score is None or h.score >= min_sim]

    # Stub filter: skip memorias with tiny bodies — they're usually fragments
    # that got auto-saved without content and don't add value as context.
    min_body_chars = int(os.environ.get("MEMO_RECALL_MIN_BODY_CHARS", "40"))
    if min_body_chars > 0:
        relevant = [h for h in relevant if len((h.body or "").strip()) >= min_body_chars]

    # Staleness suppression: memories older than MEMO_RECALL_STALENESS_DAYS
    # only pass if they score well above min_sim (1.5x). This prevents
    # the corpus from being dominated by old entries on generic queries
    # while still surfacing them for strong topic-specific matches.
    staleness_days = float(os.environ.get("MEMO_RECALL_STALENESS_DAYS", "0") or 0)
    if staleness_days > 0:
        from datetime import UTC as _UTC
        from datetime import datetime as _dt
        _now = _dt.now(_UTC)
        stale_threshold = min_sim * 1.5
        filtered: list = []
        for h in relevant:
            try:
                updated = _dt.fromisoformat(h.updated)
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=_UTC)
                days = (_now - updated).total_seconds() / 86400
                if days > staleness_days and (h.score or 0.0) < stale_threshold:
                    if os.environ.get("MEMO_RECALL_DEBUG") == "1":
                        import sys as _sys2
                        print(
                            f"# memo recall-hook: staleness filter — {h.id[:8]} "
                            f"({days:.0f}d old, score {h.score:.2f} < {stale_threshold:.2f})",
                            file=_sys2.stderr,
                        )
                    continue
            except Exception:
                pass
            filtered.append(h)
        relevant = filtered

    # Telemetry: append every recall (with or without hits) to the
    # JSONL ring buffer consumed by `memo tui`. Best-effort; failures
    # are swallowed inside the helper.
    _latency_ms_subprocess = int((time.time() - _t0) * 1000)
    try:
        from memo.dashboard import append_recall_log
        append_recall_log(
            Config.from_env().state_dir,
            prompt=prompt,
            hits=[{"id": h.id, "score": h.score, "title": h.title} for h in relevant],
            mode=mode,
            latency_ms=_latency_ms_subprocess,
            via="subprocess",
        )
    except Exception:
        pass

    if not relevant:
        _bail(f"no hits above min_sim={min_sim}")
        return

    # Format as markdown additionalContext. Be terse — context budget is
    # capped at 10k chars by Claude Code; we want each prompt to inject
    # ~500-1500 chars at most so the user's actual prompt isn't drowned.
    #
    # If MEMO_RECALL_TOKEN_BUDGET is set, pack memorias greedily by
    # score until the budget is met. Token estimate is 1 token ≈ 4 chars
    # (English/Spanish prose); good-enough rule-of-thumb that avoids a
    # tiktoken dep. Last memoria gets head-truncated to fit instead of
    # being dropped wholesale.
    header = "## Relevant memories from your past (memo)"
    footer = "_Use `/memo get <id>` to see full content._"
    lines = [header, ""]
    used_chars = 0  # chars of formatted block body, excluding header/footer

    def _est_tokens(s: str) -> int:
        return max(1, len(s) // 4)

    budget_chars = token_budget * 4 if token_budget > 0 else None

    for h in relevant:
        score_tag = f" (score {h.score:.2f})" if h.score is not None else ""
        body = (h.body or "").strip().replace("\n", " ")
        if len(body) > body_chars:
            body = body[:body_chars].rstrip() + "…"
        block_lines = [f"**[{h.id[:8]}] {h.title}**{score_tag}"]
        if h.tags:
            block_lines.append(f"_tags_: {', '.join(h.tags)}")
        if body:
            block_lines.append(f"> {body}")
        block_lines.append("")
        block = "\n".join(block_lines)

        if budget_chars is None:
            lines.extend(block_lines)
            continue

        remaining = budget_chars - used_chars
        if remaining <= 0:
            break
        if len(block) <= remaining:
            lines.extend(block_lines)
            used_chars += len(block)
        else:
            # Truncate the body in this final block to fit the budget.
            if body:
                # Reserve space for header line + tags + closing "…"
                head_len = len(block_lines[0]) + 1
                tags_len = (len(block_lines[1]) + 1) if h.tags else 0
                avail = max(0, remaining - head_len - tags_len - 3)
                if avail > 20:
                    trunc_body = body[:avail].rstrip() + "…"
                    block_lines[-2 if h.tags else -1] = f"> {trunc_body}"
                    lines.extend(block_lines)
            break

    lines.append(footer)
    if token_budget > 0 and os.environ.get("MEMO_RECALL_DEBUG") == "1":
        approx = _est_tokens("\n".join(lines))
        print(f"# memo recall-hook: ~{approx} tokens (budget {token_budget})", file=_sys.stderr)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n".join(lines),
        }
    }
    print(_json.dumps(output, ensure_ascii=False))
    _sys.exit(0)


# ── Recall daemon — persistent socket server for low-latency recall ──────────








@cli.command(name="diff")
@click.option("--from", "from_date", required=True,
              help="Start date — YYYY-MM-DD or full ISO 8601.")
@click.option("--to", "to_date", required=False, default=None,
              help="End date (default: now).")
@click.option("--json", "as_json", is_flag=True)
def diff_cmd(from_date: str, to_date: str | None, as_json: bool) -> None:
    """Diff the corpus between two snapshots.

    Shows added / removed / updated memorias plus a summary line. Useful
    for "what changed since last Monday" or "what evolved between two
    releases".
    """
    from datetime import UTC
    from datetime import datetime as _dt

    from memo.memory import Memory
    from memo.time_machine import diff as _diff

    to_iso = _dt.now(UTC).isoformat() if to_date is None else _parse_as_of_date(to_date)
    from_iso = _parse_as_of_date(from_date)

    mem = Memory(Config.from_env())
    d = _diff(mem, from_ts=from_iso, to_ts=to_iso)

    if as_json:
        click.echo(json.dumps({
            "from_ts": d.from_ts.isoformat(),
            "to_ts": d.to_ts.isoformat(),
            "added": [{"id": r.id, "title": r.title, "type": r.type} for r in d.added],
            "removed": [{"id": r.id, "title": r.title, "type": r.type} for r in d.removed],
            "updated": d.updated,
        }, ensure_ascii=False, indent=2))
        return

    console.print(Panel.fit(
        f"{d.from_ts.date().isoformat()}  →  {d.to_ts.date().isoformat()}\n"
        f"[bold]{d.summary()}[/bold]",
        title="corpus diff",
        border_style="cyan",
    ))
    if d.added:
        console.print(f"\n[green]+ added ({len(d.added)})[/green]")
        for r in d.added[:20]:
            console.print(f"  [green]+[/green] [{r.id[:8]}] {r.title}  [dim]({r.type})[/dim]")
    if d.removed:
        console.print(f"\n[red]- removed ({len(d.removed)})[/red]")
        for r in d.removed[:20]:
            console.print(f"  [red]-[/red] [{r.id[:8]}] {r.title}  [dim]({r.type})[/dim]")
    if d.updated:
        console.print(f"\n[yellow]~ updated ({len(d.updated)})[/yellow]")
        for u in d.updated[:20]:
            console.print(
                f"  [yellow]~[/yellow] [{u['id'][:8]}] {u['title']}  "
                f"[dim](fields: {', '.join(u['changed_fields'])})[/dim]",
            )


@cli.command(name="historia")
@click.argument("id_or_prefix")
@click.option("--limit", default=50, type=int, show_default=True,
              help="Max events to show.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def historia_cmd(id_or_prefix: str, limit: int, as_json: bool) -> None:
    """Show the full edit history for one memoria.

    Displays every save / update / delete event from the audit log,
    with field-level diffs on each update (title, type, tags, body_hash).
    Useful for answering "when did I change this?" or reviewing how a
    decision evolved over time.

    Examples:

      memo historia abc12345
      memo historia abc12345 --json
    """
    from memo.memory import AmbiguousIdError, Memory

    mem = Memory(Config.from_env())
    try:
        resolved = mem.resolve_id(id_or_prefix)
    except AmbiguousIdError as exc:
        console.print(f"[red]Ambiguous prefix:[/red] {exc}")
        raise SystemExit(1) from exc
    if resolved is None:
        console.print(f"[red]No record found for:[/red] {id_or_prefix!r}")
        raise SystemExit(1)

    events = mem.history.list_recent(limit=limit, record_id=resolved)
    events = list(reversed(events))  # chronological order

    if as_json:
        click.echo(json.dumps(events, ensure_ascii=False, indent=2, default=str))
        return

    r = mem.get(resolved)
    title_str = f"{r.title}" if r else resolved[:8]
    console.print(Panel.fit(
        f"[bold]{title_str}[/bold]  [dim]{resolved[:8]}[/dim]",
        title="historia",
        border_style="cyan",
    ))

    if not events:
        console.print("  [dim](no events in audit log)[/dim]")
        return

    _OP_STYLE = {"save": "green", "update": "yellow", "delete": "red"}

    for ev in events:
        op = ev.get("op", "?")
        ts = ev.get("ts", "")
        style = _OP_STYLE.get(op, "white")
        ts_short = ts[:16].replace("T", " ") if ts else "?"
        console.print(f"\n  [{style}]{op.upper():6s}[/{style}]  [dim]{ts_short}[/dim]")

        delta = ev.get("delta")
        if not delta:
            continue
        for field, pair in delta.items():
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            old_v, new_v = pair
            if field == "tags":
                old_s = ", ".join(old_v) if isinstance(old_v, list) else str(old_v)
                new_s = ", ".join(new_v) if isinstance(new_v, list) else str(new_v)
            elif field == "body_hash":
                old_s, new_s = str(old_v)[:12], str(new_v)[:12]
            else:
                old_s, new_s = str(old_v), str(new_v)
            console.print(
                f"           [dim]{field}:[/dim]  "
                f"[red]{old_s}[/red]  →  [green]{new_s}[/green]"
            )

    last_ts = events[-1].get("ts", "")
    console.print(f"\n  [dim]{len(events)} event(s) · last: {last_ts[:16].replace('T', ' ')}[/dim]")


@cli.command(name="tui")
@click.option("--refresh", type=float, default=1.0, show_default=True,
              help="Refresh interval in seconds.")
@click.option("--no-clear", is_flag=True,
              help="Don't take over the terminal screen — render inline (handy for tmux/screen).")
def tui(refresh: float, no_clear: bool) -> None:
    """Live terminal dashboard — corpus stats, recent saves/recalls, MLX warm-state,
    watcher status, top tags, 14-day sparklines. Ctrl+C to exit.

    Reads from the existing `history.db` (saves) and a JSONL recall log
    written by `memo recall-hook`. Read-only — does not modify the
    corpus.
    """
    from memo.dashboard import run_tui

    run_tui(refresh=refresh, no_clear=no_clear)


@cli.command(name="hook-log")
@click.option("--limit", default=20, type=int, show_default=True,
              help="Number of recent entries to show.")
@click.option("--follow", is_flag=True,
              help="Tail the log file (like tail -f). Ctrl+C to stop.")
def hook_log(limit: int, follow: bool) -> None:
    """Show recent recall-hook activity.

    Reads the recall log written by `memo recall-hook` and prints the
    last N entries with timestamp, mode (vec/bm25/daemon), hit count,
    and latency. Use --follow to stream new entries as they arrive.

    \b
    Fields printed per entry:
      ts       — ISO timestamp of the recall
      mode     — vec / bm25 / daemon (how the search ran)
      hits     — number of memorias injected
      latency  — round-trip latency if logged
      via      — subprocess or daemon
    """
    from memo.dashboard import read_recall_log

    cfg = Config.from_env()
    state_dir = cfg.state_dir

    def _fmt_entry(e: dict) -> str:
        ts = (e.get("ts") or "")[:19].replace("T", " ")
        mode_val = e.get("mode") or "—"
        via_val = e.get("via") or "—"
        hits_list = e.get("hits") or []
        n_hits = len(hits_list)
        latency = e.get("latency_ms")
        latency_str = f"{latency} ms" if latency is not None else "—"
        prompt = (e.get("prompt") or "").replace("\n", " ")[:60]
        return (
            f"[dim]{ts}[/dim]  "
            f"mode=[cyan]{mode_val}[/cyan]  "
            f"via=[yellow]{via_val}[/yellow]  "
            f"hits=[bold]{n_hits}[/bold]  "
            f"latency=[magenta]{latency_str}[/magenta]  "
            f"[dim]\"{prompt}\"[/dim]"
        )

    if not follow:
        entries = read_recall_log(state_dir, limit=limit)
        if not entries:
            console.print("[dim](no recall log entries yet)[/dim]")
            return
        # entries is newest-first; print oldest-first for readability
        for e in reversed(entries):
            console.print(_fmt_entry(e))
        return

    # --follow: tail the log file
    from memo.dashboard import recall_log_path
    log_path = recall_log_path(state_dir)
    console.print(f"[dim]tailing {log_path} … Ctrl+C to stop[/dim]")

    # Seek to end, then loop
    last_pos = log_path.stat().st_size if log_path.exists() else 0
    try:
        while True:
            if log_path.exists():
                new_size = log_path.stat().st_size
                if new_size < last_pos:
                    # File was rotated (truncated and rewritten) — reset position
                    last_pos = 0
                if new_size > last_pos:
                    with log_path.open("r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                e = json.loads(line)
                                console.print(_fmt_entry(e))
                            except json.JSONDecodeError:
                                pass
                    last_pos = new_size
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


@cli.command(name="logs")
@click.option(
    "--source",
    type=click.Choice(["recall", "daemon", "watcher", "all"]),
    default="all",
    show_default=True,
    help="Which log to read. 'all' interleaves by timestamp where possible.",
)
@click.option("--tail", default=40, type=int, show_default=True,
              help="Number of recent lines per source.")
@click.option("--paths", is_flag=True,
              help="Just print the log file paths (for tail -f / less).")
def logs(source: str, tail: int, paths: bool) -> None:
    """Show recent memo log activity in one place.

    \b
    Aggregates three log surfaces:
      recall   — JSONL of every recall-hook invocation
                 (bails, daemon hits, in-process fallbacks)
      daemon   — recall-daemon stdout/stderr (MLX warm-state, errors)
      watcher  — filesystem watcher stdout/stderr (launchd plist)

    Use --paths if you'd rather pipe to your own `tail -f` / `less +F`.
    """
    from memo.dashboard import read_recall_log, recall_log_path

    cfg = Config.from_env()
    state_dir = cfg.state_dir

    recall_p = recall_log_path(state_dir)
    daemon_p = Path.home() / "Library" / "Logs" / "memo" / "recall-daemon.log"
    watch_out_p = Path.home() / "Library" / "Logs" / "memo" / "watch.out.log"
    watch_err_p = Path.home() / "Library" / "Logs" / "memo" / "watch.err.log"

    if paths:
        console.print(f"recall:  {recall_p}")
        console.print(f"daemon:  {daemon_p}")
        console.print(f"watcher: {watch_out_p}  +  {watch_err_p}")
        return

    def _tail_file(p: Path, n: int) -> list[str]:
        if not p.is_file():
            return []
        try:
            return p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        except Exception as exc:  # pragma: no cover - permission edge
            return [f"# (read error: {exc})"]

    if source in ("recall", "all"):
        entries = read_recall_log(state_dir, limit=tail)
        console.print(f"[bold]recall[/bold]  [dim]{recall_p}[/dim]")
        if not entries:
            console.print("  [dim](empty)[/dim]")
        else:
            for e in reversed(entries):
                ts = (e.get("ts") or "")[:19].replace("T", " ")
                via = e.get("via") or "—"
                hits = len(e.get("hits") or [])
                latency = e.get("latency_ms")
                latency_str = f" {latency}ms" if latency is not None else ""
                reason = e.get("reason")
                error = e.get("error")
                tail_str = ""
                if reason:
                    tail_str = f"  [yellow]reason=[/yellow]{reason}"
                elif error:
                    tail_str = f"  [red]error=[/red]{error}"
                else:
                    prompt = (e.get("prompt") or "").replace("\n", " ")[:50]
                    tail_str = f'  [dim]"{prompt}"[/dim]' if prompt else ""
                console.print(
                    f"  [dim]{ts}[/dim] via=[cyan]{via}[/cyan] "
                    f"hits=[bold]{hits}[/bold]{latency_str}{tail_str}"
                )
        if source == "all":
            console.print()

    if source in ("daemon", "all"):
        console.print(f"[bold]daemon[/bold]  [dim]{daemon_p}[/dim]")
        lines = _tail_file(daemon_p, tail)
        if not lines:
            console.print("  [dim](no log — daemon never started, or logs rotated)[/dim]")
        else:
            for line in lines:
                console.print(f"  {line}")
        if source == "all":
            console.print()

    if source in ("watcher", "all"):
        console.print(f"[bold]watcher.out[/bold]  [dim]{watch_out_p}[/dim]")
        out_lines = _tail_file(watch_out_p, tail)
        if not out_lines:
            console.print("  [dim](no log — watcher inactive or never wrote)[/dim]")
        else:
            for line in out_lines:
                console.print(f"  {line}")
        err_lines = _tail_file(watch_err_p, tail)
        if err_lines:
            console.print(f"[bold]watcher.err[/bold]  [dim]{watch_err_p}[/dim]")
            for line in err_lines:
                console.print(f"  [red]{line}[/red]")


@cli.command(name="self-update")
@click.option("--check", is_flag=True, help="Check for a newer version without installing.")
def self_update(check: bool) -> None:
    """Update memo to the latest version.

    Runs `pipx upgrade mlx-memo` (or the equivalent for the active install
    method) then re-warms the MLX models. Use --check to see if an update
    is available without installing.

    Detects the active install method (pipx or uv tool) automatically.
    If memo was installed via the curl installer or another method, prints
    instructions to re-run the installer.
    """
    import importlib.metadata
    import urllib.error
    import urllib.request

    current_version = importlib.metadata.version("mlx-memo")
    console.print(f"[dim]current version:[/dim] {current_version}")

    # Fetch latest version from PyPI
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
            console.print(
                f"[yellow]Update available:[/yellow] {current_version} → {latest_version}"
            )
            if check:
                return

    # Detect install method and upgrade
    installed_via: str | None = None

    # Check pipx
    try:
        res = subprocess.run(
            ["pipx", "list", "--short"],
            capture_output=True, text=True, timeout=10,
        )
        if "mlx-memo" in res.stdout:
            installed_via = "pipx"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check uv tool
    if installed_via is None:
        try:
            res = subprocess.run(
                ["uv", "tool", "list"],
                capture_output=True, text=True, timeout=10,
            )
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


@cli.command(name="watch")
@click.option("--delay", default=2.0, type=float, show_default=True,
              help="Debounce window in seconds — coalesces bursts of edits into one reindex.")
@click.option("--debug", is_flag=True, help="Print every reindex result to stderr.")
def watch(delay: float, debug: bool) -> None:
    """Auto-reindex on `.md` change. Foreground; Ctrl+C to stop.

    Watches `cfg.memory_dir` recursively. Files saved in Obsidian (or
    any editor) trigger a debounced `Memory.reindex()` call so the
    sqlite-vec index stays in sync without manual `memo reindex` runs.

    Run as a daemon via `memo install-watcher` (launchd plist).
    """
    from memo.watcher import run_watcher

    run_watcher(delay=delay, debug=debug)


@cli.command(name="install-watcher")
@click.option("--bin", "memo_bin", default=None,
              help="Absolute path to the `memo` binary (default: auto-detect via shutil.which).")
@click.option("--no-load", is_flag=True,
              help="Write the plist but don't `launchctl bootstrap` it.")
def install_watcher(memo_bin: str | None, no_load: bool) -> None:
    """Install + load the file-watcher as a launchd daemon.

    Generates `~/Library/LaunchAgents/com.fer.memo.watch.plist`, loads
    it via `launchctl bootstrap`, and verifies it's running. Restart on
    crash is enabled (`KeepAlive=true`). Logs land in
    `~/Library/Logs/memo/`.
    """
    import shutil as _shutil
    import subprocess

    from memo.watcher import _PLIST_LABEL, install_plist

    if memo_bin is None:
        memo_bin = _shutil.which("memo") or ""
        if not memo_bin:
            raise click.ClickException(
                "Could not locate `memo` on PATH. Pass --bin /abs/path/to/memo.",
            )

    plist_path = install_plist(memo_bin)
    console.print(f"[dim]wrote:[/dim] {plist_path}")

    if no_load:
        console.print(
            "[yellow]Skipped load (--no-load). To activate manually:[/yellow]\n"
            f"  launchctl bootstrap gui/$(id -u) {plist_path}",
        )
        return

    uid = os.getuid()
    domain = f"gui/{uid}"
    target = f"{domain}/{_PLIST_LABEL}"

    # Unload first if already present, to pick up plist changes.
    subprocess.run(
        ["launchctl", "bootout", target],
        check=False, capture_output=True,
    )
    res = subprocess.run(
        ["launchctl", "bootstrap", domain, str(plist_path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise click.ClickException(
            f"launchctl bootstrap failed: {res.stderr.strip() or res.stdout.strip()}",
        )

    # Verify.
    verify = subprocess.run(
        ["launchctl", "print", target],
        capture_output=True, text=True,
    )
    if verify.returncode != 0:
        raise click.ClickException(
            "Plist loaded but `launchctl print` could not find it. "
            "Inspect `~/Library/Logs/memo/watch.err.log`.",
        )

    console.print(Panel.fit(
        f"[bold]watcher loaded[/bold]\n"
        f"[dim]label:[/dim] {_PLIST_LABEL}\n"
        f"[dim]plist:[/dim] {plist_path}\n"
        f"[dim]logs:[/dim] ~/Library/Logs/memo/watch.{{out,err}}.log",
        title="✓ install-watcher",
        border_style="green",
    ))


@cli.command(name="uninstall-watcher")
def uninstall_watcher_cmd() -> None:
    """Unload + remove the file-watcher launchd job."""
    import subprocess

    from memo.watcher import _PLIST_LABEL, uninstall_plist

    uid = os.getuid()
    target = f"gui/{uid}/{_PLIST_LABEL}"
    subprocess.run(
        ["launchctl", "bootout", target],
        check=False, capture_output=True,
    )
    existed = uninstall_plist()
    if existed:
        console.print("[green]✓ watcher uninstalled.[/green]")
    else:
        console.print("[yellow]No plist found to remove.[/yellow]")


@cli.command(name="mine-history")
@click.option("--path", "root_path", default=None,
              help="Transcripts root (default: ~/.claude/projects).")
@click.option("--since", "since_days", type=int, default=None,
              help="Only process transcripts modified in the last N days.")
@click.option("--limit", "file_limit", type=int, default=None,
              help="Cap on number of transcripts to process (newest first).")
@click.option("--dry-run", is_flag=True,
              help="Walk + extract, don't save. Useful for cost estimation.")
@click.option("--debug", is_flag=True, help="Print per-file/per-candidate info to stderr.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary instead of a panel.")
def mine_history(
    root_path: str | None, since_days: int | None, file_limit: int | None,
    dry_run: bool, debug: bool, as_json: bool,
) -> None:
    """Mine past Claude Code conversations for actionable insights.

    Walks `~/.claude/projects/<hash>/*.jsonl`, runs the same prefilter +
    helper-LLM extraction + embedding-based dedup as the live capture
    hook, and saves what's new. Resumable: per-file processed-line
    counts are tracked under `~/.local/share/memo/mine-history.json`.

    Tips:
        - First run on a long history is slow (helper LLM is the bottleneck).
          Use `--limit 10 --since 30` to start with the freshest sessions.
        - `--dry-run` reports candidate counts without writing.
    """
    from pathlib import Path as _Path

    from memo.transcript_miner import mine_transcripts

    root = _Path(root_path).expanduser() if root_path else None

    console_progress = None
    if not as_json:
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        progress.start()
        task = progress.add_task("mining transcripts", total=None)

        def cb(idx: int, total: int, p: _Path) -> None:
            progress.update(
                task, total=total, completed=idx, description=f"[{idx + 1}/{total}] {p.name}",
            )

        console_progress = (progress, task, cb)

    try:
        summary = mine_transcripts(
            root=root, since_days=since_days, file_limit=file_limit,
            dry_run=dry_run, debug=debug,
            progress_cb=console_progress[2] if console_progress else None,
        )
    finally:
        if console_progress:
            console_progress[0].stop()

    if as_json:
        click.echo(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    status = summary.get("status")
    if status == "no_files":
        console.print(f"[yellow]No transcripts found under {summary['root']}.[/yellow]")
        return

    saved = summary.get("saved", [])
    body = (
        f"[dim]root:[/dim] {summary['root']}\n"
        f"[dim]files:[/dim] {summary['files_processed']}/{summary['files_total']} processed"
        f" ([dim]{summary['files_skipped']} skipped — already mined[/dim])\n"
        f"[dim]candidates:[/dim] {summary['candidates']}\n"
        f"[bold green]saved:[/bold green] {len(saved)}"
        f"{' [yellow](dry-run)[/yellow]' if summary['dry_run'] else ''}\n"
        f"[dim]skipped duplicates:[/dim] {summary['skipped_dup']}"
    )
    console.print(Panel.fit(body, title="✓ mine-history", border_style="green"))


def _resolve_ingest_row(store, path_str):
    """Resolve the (id, existing-row) an ingest path should write to.

    The vault lives on a case-insensitive filesystem (APFS), so the same
    file can be walked under different directory casing (`notes/Foo.md`
    vs `Notes/Foo.md`). A fresh sha256(path) id would differ per casing
    and mint duplicate rows on re-ingest. So first look for an existing
    row by case-insensitive path and reuse ITS id — the upsert then
    updates that row in place instead of inserting a duplicate. This
    needs no id migration: existing rows keep their original id. Only a
    genuinely new file (no row under any casing) mints a fresh id.
    """
    import hashlib

    existing = store.get_by_path_ci(path_str)
    if existing is not None:
        return existing["id"], existing
    id_ = hashlib.sha256(path_str.encode("utf-8")).hexdigest()[:32]
    return id_, store.get(id_)


@cli.command(name="ingest")
@click.argument("vault_path", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option("--name", default=None, help="Vault label (default: dirname). Used as path prefix in store.")
@click.option("--force", is_flag=True, help="Re-embed even if body unchanged.")
@click.option("--dry-run", is_flag=True, help="Walk + report counts, don't embed/write.")
@click.option("--exclude", multiple=True, help="Glob to exclude (relative to vault). Repeat. Default: .obsidian/.git/.trash/.makemd/.smart-env/.space/99-obsidian/99-AI/")
@click.option("--ocr/--no-ocr", default=True, help="Run OCR on ![[image]] embeds inside notes (Apple Vision). Default on.")
@click.option("--chunk/--no-chunk", default=True, help="Semantically chunk markdown/PDF bodies for better retrieval precision. Default on.")
@click.option("--chunk-chars", default=1500, show_default=True, type=int, help="Target chunk size in characters.")
@click.option("--chunk-overlap", default=250, show_default=True, type=int, help="Overlap between consecutive chunks.")
@click.option("--include-pdf/--no-include-pdf", default=True, help="Extract text from .pdf via pdftotext + chunk + embed.")
@click.option("--include-orphan-images/--no-include-orphan-images", default=True, help="OCR images not referenced by any note and ingest them as standalone memorias.")
def ingest(
    vault_path: str, name: str | None, force: bool, dry_run: bool, exclude: tuple[str, ...],
    ocr: bool, chunk: bool, chunk_chars: int, chunk_overlap: int,
    include_pdf: bool, include_orphan_images: bool,
) -> None:
    """Bulk-ingest all .md from a vault into the memo index.

    Walks `<vault_path>/**/*.md`, embeds each, stores under path
    `<name>/<rel-path>`. Files with `id:` in frontmatter are skipped
    (those are curated memorias managed by `memo reindex`).

    The user's .md files are NOT modified — we synthesize ids from
    path hash and write only to `~/.local/share/memo/memvec.db`.

    Idempotent: re-running skips files whose body_hash matches the
    indexed value. Use --force to re-embed everything (e.g. after
    embedder model swap).

    Default exclusions skip Obsidian system dirs (.obsidian/, .trash/,
    etc.) and memo's own memory subtree (`99-obsidian/99-AI/`) so we
    don't double-index curated memorias. Note: sibling user content
    under `99-obsidian/` — `99-Contacts/`, `99-Forms/`, `99-Templates/`
    — IS indexed (e.g. `99-obsidian/99-Contacts/Grecia.md`).
    """
    import hashlib
    import os as _os_min
    from pathlib import Path

    import frontmatter

    from memo.chunker import chunk_markdown
    from memo.embedder import MLXEmbedder, assert_valid_embedding
    from memo.ingest_helpers import (
        IMAGE_EXTENSIONS,
        enrich_with_ocr,
        extract_pdf_text,
        find_orphan_images,
        pdftotext_available,
    )
    from memo.store import VecStore

    cfg = Config.from_env()
    cfg.ensure_dirs()

    vault = Path(vault_path).resolve()
    # `cfg.vault_path` is the user's "primary" Obsidian vault (set via
    # `memo init`'s Obsidian branch, or `MEMO_VAULT_PATH`). When we're
    # ingesting that exact vault, paths are stored without a label
    # prefix (e.g. `01-Projects/foo.md`); external vaults get a
    # `<label>/` prefix so multiple vaults coexist in one store.
    is_principal_vault = cfg.vault_path is not None and vault == cfg.vault_path
    label = "" if is_principal_vault else (name or vault.name)

    default_excludes = (
        ".obsidian", ".git", ".trash", ".makemd", ".smart-env", ".space",
        ".claude", ".devin", "99-obsidian/99-AI",
    )
    exclude_patterns = list(exclude) + list(default_excludes)

    def _excluded(rel: Path) -> bool:
        s = str(rel)
        return any(s.startswith(pat) or f"/{pat}/" in f"/{s}/" for pat in exclude_patterns)

    md_files: list[Path] = []
    pdf_files: list[Path] = []
    for p in vault.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(vault)
        if _excluded(rel):
            continue
        suffix = p.suffix.lower()
        if suffix == ".md":
            md_files.append(p)
        elif suffix == ".pdf" and include_pdf:
            pdf_files.append(p)
    md_files.sort()
    pdf_files.sort()

    pdf_supported = include_pdf and pdftotext_available()
    if include_pdf and not pdf_supported:
        console.print("[yellow]pdftotext not found on PATH — skipping PDFs[/yellow]")
        pdf_files = []

    console.print(
        f"[cyan]found[/cyan] {len(md_files)} .md, {len(pdf_files)} .pdf in {label} "
        f"(after exclusions)"
    )

    if dry_run:
        console.print("[dim](dry-run — exiting before embed/write)[/dim]")
        for p in md_files[:5]:
            console.print(f"  · {p.relative_to(vault)}")
        if len(md_files) > 5:
            console.print(f"  · …and {len(md_files) - 5} more")
        if pdf_files:
            console.print(f"  · PDFs: {len(pdf_files)}")
        return

    embedder = MLXEmbedder(model_path=cfg.embedder_model, expected_dims=cfg.embedder_dims)
    store = VecStore(cfg.db_path, dims=cfg.embedder_dims)

    skipped_id = skipped_empty = skipped_unchanged = added = updated = errors = 0
    skipped_pdf_empty = pdf_added = orphan_added = orphan_skipped = 0
    chunks_emitted = 0
    referenced_images: set[Path] = set()

    min_chars = int(_os_min.environ.get("MEMO_INGEST_MIN_CHARS", "200"))
    strict_mode = _os_min.environ.get("MEMO_INGEST_STRICT") == "1"
    debug_mode = _os_min.environ.get("MEMO_INGEST_DEBUG") == "1"

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

    def _emit_record(
        *, store_path: str, title: str, tags: list[str], body: str,
        abs_path: Path, source: str, extra_meta: dict | None = None,
    ) -> str | None:
        """Embed `title + body` (chunked if --chunk and large) and upsert
        one row per chunk. Returns "added" / "updated" / None on error.

        Single-chunk path keeps the canonical store_path so dedup +
        idempotence keep working. Multi-chunk path suffixes
        `#chunk-N` to the store_path so each chunk is its own row.
        """
        nonlocal errors, chunks_emitted
        composed_full = f"{title}\n\n{body}"
        if chunk and len(composed_full) > chunk_chars:
            pieces = chunk_markdown(composed_full, target_chars=chunk_chars, overlap_chars=chunk_overlap)
        else:
            pieces = None  # single-vector path

        if pieces is None or len(pieces) <= 1:
            composed = composed_full[: cfg.max_content_chars]
            try:
                emb = embedder.embed([composed])[0]
                assert_valid_embedding(emb, cfg.embedder_dims, context=str(abs_path))
            except Exception as exc:
                errors += 1
                if strict_mode:
                    raise
                if debug_mode:
                    console.print(f"[red]reject:[/] {exc}")
                return None
            now = datetime.now(UTC).isoformat()
            id_, existing = _resolve_ingest_row(store, store_path)
            body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
            extra: dict[str, Any] = {"source": source, "vault": label, "abs_path": str(abs_path)}
            if extra_meta:
                extra.update(extra_meta)
            store.upsert(
                id_=id_, path=store_path, title=title[:200], type_="note",
                tags=tags, created=existing["created"] if existing else now,
                updated=now, body_hash=body_hash, embedding=emb,
                extra=extra, body_text=body,
            )
            chunks_emitted += 1
            return "updated" if existing else "added"

        # Multi-chunk path. Each chunk = own meta row; parent_path lets
        # the chat-ask dedup collapse chunks back to one source.
        any_added = any_updated = False
        for piece in pieces:
            seq = piece["seq"]
            heading = piece["heading"]
            chunk_body = piece["body"]
            chunk_path = f"{store_path}#chunk-{seq}"
            id_, existing = _resolve_ingest_row(store, chunk_path)
            chunk_body_hash = hashlib.sha256(chunk_body.encode("utf-8")).hexdigest()[:16]
            if existing and existing["body_hash"] == chunk_body_hash and not force:
                continue
            chunk_composed = chunk_body[: cfg.max_content_chars]
            try:
                emb = embedder.embed([chunk_composed])[0]
                assert_valid_embedding(emb, cfg.embedder_dims, context=f"{abs_path}#chunk-{seq}")
            except Exception as exc:
                errors += 1
                if strict_mode:
                    raise
                if debug_mode:
                    console.print(f"[red]reject:[/] {exc}")
                continue
            now = datetime.now(UTC).isoformat()
            chunk_title = f"{title} (§{seq+1}/{len(pieces)})"
            if heading:
                chunk_title = f"{title} — {heading}"
            extra = {
                "source": source, "vault": label, "abs_path": str(abs_path),
                "parent_path": store_path, "chunk_seq": seq,
                "chunk_count": len(pieces), "chunk_heading": heading,
            }
            if extra_meta:
                extra.update(extra_meta)
            store.upsert(
                id_=id_, path=chunk_path, title=chunk_title[:200], type_="note",
                tags=[*tags, "chunk"], created=existing["created"] if existing else now,
                updated=now, body_hash=chunk_body_hash, embedding=emb,
                extra=extra, body_text=chunk_body,
            )
            chunks_emitted += 1
            if existing:
                any_updated = True
            else:
                any_added = True
        if any_added:
            return "added"
        if any_updated:
            return "updated"
        return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task(f"embed {label}", total=len(md_files))

        for path in md_files:
            try:
                rel = path.relative_to(vault)
                store_path = f"{label}/{rel}" if label else str(rel)

                raw = path.read_text(encoding="utf-8", errors="replace")

                try:
                    fm = frontmatter.loads(raw)
                except Exception:
                    fm = frontmatter.Post(raw)

                # Skip curated memorias (have explicit id).
                if fm.metadata.get("id"):
                    skipped_id += 1
                    continue

                body = fm.content.strip()
                if not body:
                    skipped_empty += 1
                    continue

                if len(body) < min_chars and not _is_high_signal(body, fm.metadata.get("tags")):
                    skipped_empty += 1
                    continue

                title = (
                    fm.metadata.get("title")
                    or _extract_first_h1(body)
                    or path.stem.replace("-", " ").replace("_", " ")
                )
                title = str(title).strip() or path.stem

                tags: list[str] = []
                fm_tags: Any = fm.metadata.get("tags") or []
                if isinstance(fm_tags, str):
                    fm_tags = [t.strip() for t in fm_tags.split(",")]
                for t in fm_tags:
                    if t and str(t) not in tags:
                        tags.append(str(t))
                for part in rel.parent.parts:
                    if part and part not in tags:
                        tags.append(part)

                # OCR enrichment — appends <!-- OCR: img.png -->\n<text>
                # blocks for every ![[image]] embed in the note. Tracks
                # resolved image paths so the orphan-image pass below
                # knows which files are already claimed.
                if ocr:
                    enriched, resolved, _ = enrich_with_ocr(
                        body, path, vault, cfg.state_dir,
                    )
                    referenced_images.update(resolved)
                    body = enriched

                body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
                _, existing = _resolve_ingest_row(store, store_path)
                if existing and existing["body_hash"] == body_hash and not force:
                    skipped_unchanged += 1
                    continue

                outcome = _emit_record(
                    store_path=store_path, title=title, tags=tags, body=body,
                    abs_path=path, source="vault-ingest",
                )
                if outcome == "added":
                    added += 1
                elif outcome == "updated":
                    updated += 1
            except Exception as exc:
                errors += 1
                if debug_mode:
                    console.print(f"[red]err[/] {path}: {exc}")
            finally:
                progress.advance(task_id)

        if pdf_files:
            pdf_task = progress.add_task(f"PDF {label}", total=len(pdf_files))
            for pdf_path in pdf_files:
                try:
                    rel = pdf_path.relative_to(vault)
                    text = extract_pdf_text(pdf_path).strip()
                    if not text:
                        skipped_pdf_empty += 1
                        continue
                    store_path = f"{label}/{rel}" if label else str(rel)
                    title = pdf_path.stem.replace("-", " ").replace("_", " ")
                    tags = [p for p in rel.parent.parts if p] + ["pdf"]
                    outcome = _emit_record(
                        store_path=store_path, title=title, tags=tags, body=text,
                        abs_path=pdf_path, source="vault-ingest-pdf",
                    )
                    if outcome == "added":
                        pdf_added += 1
                except Exception as exc:
                    errors += 1
                    if debug_mode:
                        console.print(f"[red]err pdf[/] {pdf_path}: {exc}")
                finally:
                    progress.advance(pdf_task)

        if include_orphan_images and ocr:
            orphans = find_orphan_images(vault, referenced_images, excluded_dirs=tuple(exclude_patterns))
            # Filter image extensions we actually OCR (Apple Vision covers png/jpg/webp/heic).
            orphans = [o for o in orphans if o.suffix.lower() in IMAGE_EXTENSIONS]
            if orphans:
                orphan_task = progress.add_task(f"OCR orphan imgs {label}", total=len(orphans))
                from memo.ocr import extract_text_cached
                cache_dir = cfg.state_dir / "ocr_cache"
                for img_path in orphans:
                    try:
                        ocr_text = (extract_text_cached(img_path, cache_dir=cache_dir) or "").strip()
                        if not ocr_text:
                            orphan_skipped += 1
                            continue
                        rel = img_path.relative_to(vault)
                        store_path = f"{label}/{rel}" if label else str(rel)
                        title = img_path.stem.replace("-", " ").replace("_", " ")
                        tags = [p for p in rel.parent.parts if p] + ["standalone-image"]
                        outcome = _emit_record(
                            store_path=store_path, title=title, tags=tags, body=ocr_text,
                            abs_path=img_path, source="vault-ingest-image",
                            extra_meta={"image_ext": img_path.suffix.lower()},
                        )
                        if outcome == "added":
                            orphan_added += 1
                    except Exception as exc:
                        errors += 1
                        if debug_mode:
                            console.print(f"[red]err img[/] {img_path}: {exc}")
                    finally:
                        progress.advance(orphan_task)

    # Bump on-disk schema version so the legacy-paths probe in
    # `Memory._maybe_warn_legacy_paths` doesn't fire for ingest-only
    # vaults (vault-ingest rows live outside `cfg.data_dir` and the
    # probe can't resolve them; setting user_version=1 marks "this DB
    # is post-init, the legacy fallback is no longer relevant").
    if added or updated or pdf_added or orphan_added:
        with contextlib.suppress(Exception):
            store.set_user_version(1)

    console.print(
        f"\n[green]done[/] "
        f"added={added} updated={updated} "
        f"skipped_unchanged={skipped_unchanged} "
        f"skipped_id={skipped_id} skipped_empty={skipped_empty} "
        f"pdf_added={pdf_added} pdf_empty={skipped_pdf_empty} "
        f"orphan_added={orphan_added} orphan_skipped={orphan_skipped} "
        f"chunks_emitted={chunks_emitted} "
        f"errors={errors}"
    )


_HIGH_SIGNAL_TAGS = frozenset({
    # Notes pinned to lookup-style facts. Lowercase compare; surface
    # forms like "Link" / "LINKS" / "Pago" all match. Spanish + English
    # variants because the vault mixes both.
    "link", "links", "url", "urls",
    "dato", "datos", "data",
    "ref", "refs", "referencia", "referencias", "reference",
    "comando", "comandos", "command", "commands", "cmd", "snippet",
    "pago", "pagos", "payment",
    "credencial", "credenciales", "credential", "credentials",
    "endpoint", "endpoints", "api",
    "telefono", "teléfono", "phone", "tel",
    "cbu", "alias", "iban",
})

# Match http(s):// URLs — anchored end on whitespace, ), >, ], or "
# (common markdown wrappers). Permissive enough to catch trailing
# punctuation cases without dragging adjacent text in.
_URL_RE = re.compile(r"https?://[^\s)>\]\"]+")


def _is_high_signal(body: str, fm_tags: Any) -> bool:
    """Short notes worth indexing despite being below MIN_CHARS.

    A note is high-signal if any of:
    - frontmatter tags include `link` / `dato` / `ref` / `comando` /
      `pago` / `endpoint` / `cbu` / etc.
    - body contains an http(s) URL
    - body contains a fenced code block (```)

    The user uses these notes as atomic-fact pins (a payment URL, a
    CBU, a one-off shell command). Filtering them by char count
    dropped them from the index even when their title perfectly
    matched a future query. Real example: `Pagar escuela Grecia.md`
    with a 67-char body containing the payment URL.
    """
    if not body:
        return False

    raw_tags: list[str] = []
    if isinstance(fm_tags, list):
        raw_tags = [str(t).strip().lower() for t in fm_tags if t]
    elif isinstance(fm_tags, str):
        raw_tags = [t.strip().lower() for t in fm_tags.split(",") if t.strip()]
    if any(t in _HIGH_SIGNAL_TAGS for t in raw_tags):
        return True

    if _URL_RE.search(body):
        return True

    return "```" in body


def _extract_first_h1(body: str) -> str | None:
    """Return text of the first `# H1` line, or None."""
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("# ") and not s.startswith("##"):
            return s[2:].strip()
        if s and not s.startswith("#"):
            # First non-heading line of content — no H1.
            return None
    return None


@cli.command(name="capture-stop")
def capture_stop() -> None:
    """Stop hook — passive auto-extract of insights from the last turn.

    Reads the Stop hook payload from stdin (Claude Code format), pulls
    the last (user, assistant) exchange from the transcript, asks the
    helper LLM (Qwen2.5-3B) to extract any actionable insights, dedups
    against the existing corpus, and saves survivors via Memory.save().

    Hook input (stdin, JSON):
      {"transcript_path": "/path/to/...jsonl", ...}

    Hook output (stdout):
      `{}`  — always. Capture is silent; the user discovers new
      memorias via `memo list` or the next ambient recall.

    Env vars:
      MEMO_CAPTURE_DISABLE  — set to "1" to make this a no-op.
      MEMO_CAPTURE_DEBUG    — set to "1" to print extraction progress
                              to stderr (helpful while tuning the
                              extraction prompt or trigger keywords).

    Failure modes are absorbed. The hook never blocks the user — at
    worst you get no auto-save for that turn.
    """
    import json as _json
    import os
    import sys as _sys
    from pathlib import Path

    if os.environ.get("MEMO_CAPTURE_DISABLE") == "1":
        print("{}")
        _sys.exit(0)

    debug = os.environ.get("MEMO_CAPTURE_DEBUG") == "1"

    try:
        raw = _sys.stdin.read()
        payload = _json.loads(raw) if raw.strip() else {}
    except _json.JSONDecodeError:
        print("{}")
        _sys.exit(0)

    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        print("{}")
        _sys.exit(0)

    try:
        from memo.capture import run_capture
        run_capture(Path(transcript_path), debug=debug)
    except Exception as exc:
        if debug:
            print(f"# memo capture-stop failed: {exc}", file=_sys.stderr)

    print("{}")
    _sys.exit(0)


@cli.command(name="prewarm")
@click.option(
    "--download-all",
    "download_all",
    is_flag=True,
    default=False,
    help="Also pre-download the chat LLM models (7B + 3B helper). Used by the installer.",
)
def prewarm(download_all: bool) -> None:
    """SessionStart hook — pre-load the MLX embedder so first recall is fast.

    Loads the embedder + optional reranker into memory so subsequent
    recall-hook calls benefit from the OS file cache (cold-load drops
    from ~2 s to ~500 ms). Writes a warm-signal file so the recall-hook
    can detect a cold start and fall back to BM25 instead of timing out.

    With --download-all, also pre-downloads the chat LLM models (7B + 3B)
    so the first `memo ask` doesn't stall on a multi-GB download. Designed
    to be called by the installer immediately after `pipx install`.

    Failures are silent when run as a hook — a hook crash must never block
    Claude Code's prompt submission.
    """
    import os
    import sys as _sys
    import time as _time

    if os.environ.get("MEMO_RECALL_DISABLE") == "1":
        _sys.exit(0)
    try:
        from memo.embedder import MLXEmbedder
        cfg = Config.from_env()
        emb = MLXEmbedder(model_path=cfg.embedder_model, expected_dims=cfg.embedder_dims)
        emb.embed(["warmup"])  # batch=1; forces MLX load + first forward pass
        # Reranker prewarm — same rationale as the embedder. Skipped
        # when disabled to keep the SessionStart hook below its
        # 30s budget on machines that opted out of rerank entirely.
        if cfg.reranker_enabled:
            from memo.reranker import MLXReranker
            r = MLXReranker(
                model_path=cfg.reranker_model,
                revision=cfg.reranker_revision,
            )
            r.warmup()
        # Write warm-signal so recall-hook knows disk cache is fresh.
        # The file's mtime is the only datum read by recall-hook.
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        warm_signal = cfg.state_dir / ".prewarm_ts"
        warm_signal.write_text(str(_time.time()))
        if download_all:
            # Pre-download the chat models so the first `memo ask` doesn't
            # stall. snapshot_download is a no-op when the model is already
            # cached, so re-running this is safe.
            try:
                from huggingface_hub import snapshot_download
                for repo in (cfg.llm_model, cfg.helper_model):
                    click.echo(f"[memo] downloading {repo}…")
                    snapshot_download(repo_id=repo)
                    click.echo(f"[memo] ready: {repo}")
            except Exception as dl_exc:
                click.echo(f"[memo] chat model download failed: {dl_exc}", err=True)
    except Exception as exc:
        if os.environ.get("MEMO_RECALL_DEBUG") == "1":
            print(f"# memo prewarm failed: {exc}", file=_sys.stderr)
    _sys.exit(0)


# ── Session checkpoints (v0.4.0) ───────────────────────────────────────────
#
# `memo session ...` — short-lived "what was I working on" snapshots, written
# on every Claude Code Stop hook. Survive a closed/crashed session so the
# next SessionStart can show a picker of recent work. Storage is sidecar
# JSON in `state_dir/sessions/`, NOT memorias (different lifecycle, different
# query pattern — looked up by recency, never by semantic similarity).




@cli.command(name="resume")
@click.argument("session_id", required=False)
@click.option("--limit", default=10, type=int, show_default=True,
              help="Max sessions to show (only used when SESSION_ID is omitted).")
@click.option("--project", default=None, help="Filter to one project basename.")
@click.option("--cwd", "cwd_filter", default=None,
              help="Filter to sessions for this exact cwd (resolved). "
                   "Used by the shell wrapper to ask 'what was open here?' "
                   "without manual path comparison.")
@click.option("--json", "as_json", is_flag=True)
def resume(
    session_id: str | None, limit: int,
    project: str | None, cwd_filter: str | None, as_json: bool,
) -> None:
    """Recent sessions to retomar — picker for the SessionStart flow.

    With no argument, prints a table of the most recent sessions
    (cwd / branch / summary / id). Pass SESSION_ID (full or unique
    prefix ≥4 chars) to inspect one session in detail.

    Storage is sidecar JSON under `~/.local/share/memo/sessions/`,
    auto-written by the Stop hook (`memo session checkpoint`) and
    LRU-capped at 50.
    """
    from memo.session import format_relative, get_session, list_sessions

    cfg = Config.from_env()

    # Detail view — one session.
    if session_id:
        snap = get_session(cfg.state_dir, session_id)
        if snap is None:
            console.print(f"[red]not found:[/red] {session_id}")
            sys.exit(1)
        if as_json:
            click.echo(json.dumps(snap, ensure_ascii=False, indent=2))
            return
        mods = snap.get("modified_files") or []
        mods_line = ", ".join(mods[:5])
        if len(mods) > 5:
            mods_line += f", …(+{len(mods) - 5})"
        sid = snap.get("session_id") or ""
        console.print(Panel.fit(
            f"[bold]{snap.get('summary') or snap.get('last_user_msg') or 'session'}[/bold]\n"
            f"[dim]session_id:[/dim] {sid}\n"
            f"[dim]project:[/dim]    {snap.get('project') or '—'}\n"
            f"[dim]cwd:[/dim]        {snap.get('cwd') or '—'}\n"
            f"[dim]branch:[/dim]     {snap.get('branch') or '—'}\n"
            f"[dim]head:[/dim]       {snap.get('head_commit') or '—'}\n"
            f"[dim]modified:[/dim]   {mods_line or '—'}\n"
            f"[dim]transcript:[/dim] {snap.get('transcript_path') or '—'}\n"
            f"[dim]created:[/dim]    {snap.get('created')}  ({format_relative(snap.get('created'))})\n"
            f"[dim]updated:[/dim]    {snap.get('updated')}  ({format_relative(snap.get('updated'))})\n"
            f"[dim]turns:[/dim]      {snap.get('turn_count')}\n\n"
            f"{snap.get('last_user_msg') or ''}",
            title="session", border_style="cyan",
        ))
        if sid:
            console.print(
                f"\n[bold green]Para retomar:[/bold green]  "
                f"[cyan]claude --resume {sid}[/cyan]\n"
                f"[dim](copy-paste; corré el comando desde "
                f"`{snap.get('cwd') or '?'}`)[/dim]",
            )
        return

    # List view — picker.
    rows = list_sessions(
        cfg.state_dir, limit=limit, project=project, cwd=cwd_filter,
    )
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        console.print("[dim]no sessions yet — run a checkpoint first[/dim]")
        return

    # When the caller passed an explicit --cwd, the list is already
    # filtered to that cwd — printing a "Última en este proyecto"
    # banner on top of a homogeneous list would be redundant.
    if cwd_filter:
        same_cwd = []
    else:
        # Bias: if there's a session for the current cwd, surface it on top
        # with the exact resume command. The whole point of the picker is
        # crash recovery — if you crashed and reopened terminal in the same
        # project, the very first thing you want to see is "click here to
        # resume", not a generic chronological list.
        import os as _os
        from pathlib import Path as _Path
        cur_cwd = str(_Path(_os.getcwd()).resolve())
        same_cwd = [r for r in rows if (r.get("cwd") or "") == cur_cwd]
    if same_cwd:
        top = same_cwd[0]
        sid = top.get("session_id") or ""
        console.print(
            f"[bold green]Última en este proyecto[/bold green]  "
            f"[dim]({format_relative(top.get('updated'))})[/dim]: "
            f"{(top.get('summary') or top.get('last_user_msg') or '—')[:80]}",
        )
        console.print(
            f"[bold green]Para retomar:[/bold green]  "
            f"[cyan]claude --resume {sid}[/cyan]\n",
        )

    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("when", width=10)
    tbl.add_column("project", width=14, overflow="fold")
    tbl.add_column("branch", width=14, overflow="fold")
    tbl.add_column("turns", justify="right", width=5)
    tbl.add_column("summary", overflow="fold")
    tbl.add_column("session_id", overflow="fold")
    for r in rows:
        tbl.add_row(
            format_relative(r.get("updated")),
            r.get("project") or "—",
            r.get("branch") or "—",
            str(r.get("turn_count") or 0),
            (r.get("summary") or r.get("last_user_msg") or "—")[:80],
            r.get("session_id") or "—",
        )
    console.print(tbl)
    console.print(
        "[dim]Detalle: `memo resume <id|prefix>`  ·  "
        "Retomar: `claude --resume <session_id>` (copy desde la tabla).[/dim]",
    )


@session_group.command(name="recent")
@click.option("--limit", default=5, type=int, show_default=True)
def session_recent(limit: int) -> None:
    """SessionStart hook entrypoint — emit `additionalContext` markdown
    listing recent sessions. Same exit-0-silent contract as recall-hook."""
    import json as _json
    import os as _os
    import sys as _sys

    if _os.environ.get("MEMO_SESSION_DISABLE") == "1":
        print("{}")
        _sys.exit(0)

    try:
        from memo.session import format_relative, list_sessions
        cfg = Config.from_env()
        rows = list_sessions(cfg.state_dir, limit=limit)
    except Exception as exc:
        if _os.environ.get("MEMO_SESSION_DEBUG") == "1":
            print(f"# memo session recent failed: {exc}", file=_sys.stderr)
        print("{}")
        _sys.exit(0)

    if not rows:
        print("{}")
        _sys.exit(0)

    from pathlib import Path as _Path
    cur_cwd = str(_Path(_os.getcwd()).resolve())
    same_cwd = [r for r in rows if (r.get("cwd") or "") == cur_cwd]
    top = same_cwd[0] if same_cwd else None

    lines: list[str] = []

    if top:
        sid = top.get("session_id") or ""
        when = format_relative(top.get("updated"))
        branch = top.get("branch") or "—"
        turns = top.get("turn_count") or 0
        # Prefer running_summary (LLM-generated arc) over plain last_user_msg.
        running_summary = top.get("running_summary")
        summary = (
            top.get("summary") or top.get("last_user_msg") or "—"
        ).replace("\n", " ")[:120]
        lines.extend([
            "## Sesión anterior detectada — ¿continuar?",
            "",
            f"Había una sesión activa en este directorio ({when}):",
            f"- **Resumen**: {summary}",
            f"- **Branch**: `{branch}`  |  **Turnos**: {turns}",
            f"- **Para retomar** (en una nueva terminal): `claude --resume {sid}`",
            "",
        ])
        if running_summary:
            lines.extend([
                "### El Briefing",
                "",
                running_summary.strip(),
                "",
            ])
        prompt_trail = top.get("prompt_trail") or []
        if prompt_trail:
            lines.append("### Loops abiertos (últimos 7 días)")
            lines.append("")
            for i, p in enumerate(reversed(prompt_trail[-3:]), 1):
                lines.append(f"{i}. {p.strip()}")
            lines.append("")
        lines.extend([
            "> **Acción**: Al iniciar esta conversación, pregunta al usuario si quiere "
            "retomar la sesión anterior (ejecutando el comando de arriba en la terminal) "
            "o continuar con esta sesión nueva.",
            "",
        ])

    if len(rows) > (1 if top else 0):
        others = [r for r in rows if r is not top][:5]
        lines.extend([
            "### Otras sesiones recientes",
            "",
            "| cuándo | proyecto | branch | resumen | id |",
            "|--------|----------|--------|---------|----|",
        ])
        for r in others:
            s = (r.get("summary") or r.get("last_user_msg") or "—").replace("|", "·").replace("\n", " ")
            lines.append(
                f"| {format_relative(r.get('updated'))} | "
                f"{(r.get('project') or '—')[:18]} | "
                f"{(r.get('branch') or '—')[:14]} | "
                f"{s[:55]} | "
                f"`{(r.get('session_id') or '')[:8]}` |"
            )
        lines.append("")
        lines.append("_`memo resume <id>` para ver detalles. `claude --resume <id>` para retomar._")

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n".join(lines),
        }
    }
    print(_json.dumps(output, ensure_ascii=False))


@session_group.command(name="prune")
@click.option("--cap", default=50, type=int, show_default=True)
def session_prune(cap: int) -> None:
    """Delete oldest sessions beyond `cap`. Idempotent."""
    from memo.session import prune_lru

    cfg = Config.from_env()
    n = prune_lru(cfg.state_dir, cap=cap)
    console.print(f"[green]✓[/green] pruned {n} session(s); cap={cap}")


# ── Shell wrapper for crash recovery (v0.4.x) ──────────────────────────────
#
# Companion to `memo resume`: when the user types `claude` (no args) in a
# project that has recent session checkpoints, the wrapper offers to retomar
# one before opening a fresh session. Designed for the post-reboot case
# where iTerm2 / Terminal restored the cwd but Claude itself is closed.
# `command claude` is used everywhere to bypass the shell function and
# avoid recursion.

_WRAPPER_SNIPPET_ZSH = r"""# >>> memo session-resume wrapper >>>
# Auto-suggest resuming recent Claude Code sessions for the current cwd.
# Generated by `memo install-shell-wrapper`. Do not edit by hand — re-run
# the install command instead. To remove: delete this file and the
# matching `source` line in your shell rc.
#
# If you previously had `alias claude='claude --flag1 --flag2'`, migrate
# to:   MEMO_CLAUDE_EXTRA_ARGS=(--flag1 --flag2)
# Those flags are forwarded to every `command claude` invocation.

# ISO-8601 → relative string (macOS date -j).
_memo_reltime() {
    local iso="${1:0:19}" then now diff
    then=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$iso" "+%s" 2>/dev/null) || { echo "?"; return; }
    now=$(date +%s); diff=$(( now - then ))
    (( diff < 60    )) && { echo "ahora"; return; }
    (( diff < 3600  )) && { printf "hace %dm" $(( diff/60   )); return; }
    (( diff < 86400 )) && { printf "hace %dh" $(( diff/3600 )); return; }
                           printf "hace %dd"  $(( diff/86400 ))
}

# One │ padded-content │ row. Reads _CW, C, R from caller scope (dynamic).
_memo_bl() {
    local padded
    padded=$(printf "%-${_CW}s" "${1:0:${_CW}}")
    printf "  ${C}│${R} ${2:-}${padded}${R} ${C}│${R}\n"
}

function claude() {
    local extra_args=("${MEMO_CLAUDE_EXTRA_ARGS[@]}")

    # Pass-through: args present, stdin not a TTY, or required tools missing.
    (( $# > 0 ))  && { command claude "${extra_args[@]}" "$@"; return; }
    [[ ! -t 0 ]] && { command claude "${extra_args[@]}"; return; }
    { command -v memo && command -v jq; } >/dev/null 2>&1 \
        || { command claude "${extra_args[@]}"; return; }

    local raw count
    raw=$(memo resume --json --limit 5 --cwd "$PWD" 2>/dev/null) || raw=""
    count=$(printf '%s' "$raw" | jq 'length' 2>/dev/null)
    [[ -z "$count" || "$count" == "0" ]] && { command claude "${extra_args[@]}"; return; }

    # ANSI colors — suppressed on dumb terminals.
    local R='' B='' D='' C='' G='' GR='' W=''
    [[ -t 1 && "${TERM:-dumb}" != "dumb" ]] && {
        R=$'\e[0m' B=$'\e[1m' D=$'\e[2m'
        C=$'\e[96m' G=$'\e[92m' GR=$'\e[90m' W=$'\e[97m'
    }

    # Box: 60 chars of content, 62-char horizontal rule.
    local _CW=60
    local _HL; _HL=$(printf '%0.s─' {1..62})

    printf "\n  ${C}╭${_HL}╮${R}\n"

    if [[ "$count" == "1" ]]; then
        local sid project branch turns when summary
        sid=$(     printf '%s' "$raw" | jq -r '.[0].session_id')
        project=$( printf '%s' "$raw" | jq -r '.[0].project    // "?"')
        branch=$(  printf '%s' "$raw" | jq -r '.[0].branch     // "?"')
        turns=$(   printf '%s' "$raw" | jq -r '.[0].turn_count // 0')
        when=$(    _memo_reltime "$(printf '%s' "$raw" | jq -r '.[0].updated')")
        summary=$( printf '%s' "$raw" | jq -r '.[0].summary // .[0].last_user_msg // "—"')

        _memo_bl ""
        _memo_bl "  ${project}  ·  ${branch}  ·  ${when}  ·  ${turns} turnos" "${D}"
        _memo_bl "  ${summary}" "${B}${W}"
        _memo_bl ""
        printf "  ${C}╰${_HL}╯${R}\n"

        printf "\n  ${G}${B}¿Continuar?${R}  ${B}Y${R}${G} retomar  ${GR}n nueva sesión${R}  "
        local ans; read -rk1 ans; printf "\n\n"

        if [[ "$ans" == [yY] || "$ans" == $'\n' || "$ans" == $'\r' || -z "$ans" ]]; then
            command claude --resume "$sid" "${extra_args[@]}"
        else
            command claude "${extra_args[@]}"
        fi

    else
        local sw=$(( _CW - 2 - 1 - 8 - 1 - 12 - 1 ))  # 35 chars for summary
        _memo_bl "  ${count} sesiones anteriores — elegí cuál retomar" "${D}"
        _memo_bl ""

        local i idx r_branch r_when r_summary pn pw pb ps
        for (( i=1; i<=count; i++ )); do
            idx=$(( i - 1 ))
            r_branch=$( printf '%s' "$raw" | jq -r ".[${idx}].branch // \"—\"")
            r_when=$(   _memo_reltime "$(printf '%s' "$raw" | jq -r ".[${idx}].updated")")
            r_summary=$(printf '%s' "$raw" | jq -r ".[${idx}].summary // .[${idx}].last_user_msg // \"—\"")

            pn=$(printf "%-2s"      "$i")
            pw=$(printf "%-8s"      "${r_when}")
            pb=$(printf "%-12s"     "${r_branch:0:12}")
            ps=$(printf "%-${sw}s"  "${r_summary:0:${sw}}")

            printf "  ${C}│${R} ${B}${pn}${R} ${D}${pw}${R} ${pb} ${W}${ps}${R} ${C}│${R}\n"
        done

        _memo_bl ""
        _memo_bl "  n  nueva sesión" "${GR}"
        _memo_bl ""
        printf "  ${C}╰${_HL}╯${R}\n"

        printf "\n  ${G}${B}Elegí${R}  ${GR}[1-${count}] retomar  n nueva${R}  "
        local choice; read choice; printf "\n"

        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= count )); then
            local sid
            sid=$(printf '%s' "$raw" | jq -r ".[$(( choice - 1 ))].session_id")
            command claude --resume "$sid" "${extra_args[@]}"
        else
            command claude "${extra_args[@]}"
        fi
    fi
}
# <<< memo session-resume wrapper <<<
"""


@cli.command(name="install-shell-wrapper")
@click.option("--print", "do_print", is_flag=True,
              help="Print the wrapper snippet to stdout. Default mode "
                   "when neither --print nor --write is set.")
@click.option("--write", "do_write", is_flag=True,
              help="Write ~/.zsh/memo-wrapper.zsh and append the matching "
                   "`source` line to ~/.zshrc (idempotent).")
@click.option("--shell", "shell_kind",
              type=click.Choice(["zsh", "bash"]), default="zsh",
              show_default=True,
              help="Target shell. zsh is the macOS default.")
@click.option("--force", is_flag=True,
              help="Overwrite ~/.zsh/memo-wrapper.zsh even if its content "
                   "differs from what we would write.")
def install_shell_wrapper(
    do_print: bool, do_write: bool, shell_kind: str, force: bool,
) -> None:
    """Install or print the `claude` shell wrapper for crash recovery.

    With no flags (or --print): emit the snippet to stdout for review.
    With --write: install to `~/.zsh/memo-wrapper.zsh` and append a
    `source` line to `~/.zshrc` if not already present.

    The wrapper makes `claude` (no args) prompt to retomar a recent
    memo session checkpoint when the current cwd has any. With args
    it falls through to the real claude. Detects a pre-existing
    `alias claude=...` and warns the user to migrate to
    `MEMO_CLAUDE_EXTRA_ARGS` so flags compose with the wrapper.
    """
    # Bash compat note: zsh's `[[ ... =~ ... ]]` and `${var:0:N}` work
    # in bash >=3, so we currently emit one snippet for both. The
    # `--shell` flag is still consumed below to pick the rc file
    # (.zshrc vs .bashrc) and is kept as an explicit dispatch point
    # for the day we need bash-specific snippet tweaks
    # (e.g. `read -k 1` → `read -n 1`).
    snippet = _WRAPPER_SNIPPET_ZSH

    # Default to --print when neither flag is set; spelled out so the
    # output flows through one path.
    if not do_write:
        click.echo(snippet)
        if not do_print:
            click.echo(
                "\n(pasale `--write` para instalar en "
                "~/.zsh/memo-wrapper.zsh + ~/.zshrc.)",
                err=True,
            )
        return

    from pathlib import Path as _Path

    home = _Path.home()
    wrapper_dir = home / ".zsh"
    wrapper_path = wrapper_dir / "memo-wrapper.zsh"
    rc_path = home / (".zshrc" if shell_kind == "zsh" else ".bashrc")
    source_line = f"[[ -f {wrapper_path} ]] && source {wrapper_path}"

    wrapper_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — write the wrapper file (idempotent + force-aware).
    if wrapper_path.is_file():
        existing = wrapper_path.read_text(encoding="utf-8")
        if existing == snippet:
            console.print(
                f"[dim]✓ {wrapper_path} ya está al día[/dim]",
            )
        elif not force:
            console.print(
                f"[red]✗[/red] {wrapper_path} existe con contenido distinto. "
                f"Pasale [bold]--force[/bold] para sobrescribir.",
            )
            sys.exit(2)
        else:
            wrapper_path.write_text(snippet, encoding="utf-8")
            console.print(f"[yellow]↻[/yellow] {wrapper_path} sobrescrito (--force)")
    else:
        wrapper_path.write_text(snippet, encoding="utf-8")
        console.print(f"[green]✓[/green] {wrapper_path} creado")

    # Step 2 — append `source` line to rc if missing.
    rc_existing = rc_path.read_text(encoding="utf-8") if rc_path.is_file() else ""
    if source_line in rc_existing:
        console.print(f"[dim]✓ {rc_path} ya tiene la línea source[/dim]")
    else:
        with rc_path.open("a", encoding="utf-8") as fh:
            if rc_existing and not rc_existing.endswith("\n"):
                fh.write("\n")
            fh.write("\n# memo session-resume wrapper\n")
            fh.write(f"{source_line}\n")
        console.print(f"[green]✓[/green] {rc_path} ← appended source line")

    # Step 3 — detect pre-existing `alias claude=...` and warn the
    # user. A shell wrapper function shadows an alias of the same
    # name, so the alias would silently lose any flags it bundled.
    # Migration target: `MEMO_CLAUDE_EXTRA_ARGS=(--flag1 --flag2)`.
    import re as _re

    try:
        rc_text = rc_path.read_text(encoding="utf-8")
    except OSError:
        rc_text = ""
    alias_match = _re.search(r"^\s*alias\s+claude\s*=.*$", rc_text, _re.MULTILINE)
    if alias_match:
        console.print(
            f"[yellow]heads-up:[/yellow] found a pre-existing `{alias_match.group(0).strip()}` "
            f"in {rc_path}.\n"
            f"  The wrapper function will shadow it. To preserve those flags,\n"
            f"  remove the alias and use [bold]MEMO_CLAUDE_EXTRA_ARGS[/bold]:\n"
            f"    [dim]export MEMO_CLAUDE_EXTRA_ARGS=(--your-flag --other-flag)[/dim]",
        )








# -- graph navigation commands ------------------------------------------------




































# -- duplicate detection (exact-ish near-dups, no LLM gate) -------------------


@cli.command(name="dedupe")
@click.option("--threshold", type=float, default=0.92,
              help="Cosine threshold for near-duplicate clustering (default: 0.92)")
@click.option("--max-clusters", type=int, default=50,
              help="Max clusters to surface (default: 50)")
@click.option("--type", "type_", help="Filter by memoria type")
@click.option("--apply", "do_apply", is_flag=True,
              help="Interactively merge each cluster (default: list-only)")
@click.option("--dry-run", is_flag=True,
              help="With --apply: show merges without writing")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def dedupe_cmd(
    threshold: float, max_clusters: int, type_: str | None,
    do_apply: bool, dry_run: bool, as_json: bool,
) -> None:
    """Find and (optionally) merge near-duplicate memorias.

    Thin wrapper over `memo consolidate` with a higher default threshold,
    aimed at obvious dups (paste-restate, double-save, etc.) — not at
    semantic clustering. Use the lower-threshold `consolidate` group
    when you want LLM synthesis across loosely-related notes.
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    clusters = mem.consolidate(
        threshold=threshold,
        max_clusters=max_clusters,
        type_=type_,
    )
    dup_clusters = [c for c in clusters if c.get("relationship") in ("duplicate", "evolution")]

    if as_json:
        click.echo(json.dumps(dup_clusters, indent=2))
        return

    if not dup_clusters:
        console.print("[green]No near-duplicate clusters found at this threshold.[/green]")
        return

    console.print(f"[bold]Found {len(dup_clusters)} duplicate-like cluster(s).[/bold]")

    if not do_apply:
        for c in dup_clusters[:20]:
            console.print()
            console.print(
                f"[cyan]cluster {c.get('cluster_id', '?')}[/cyan] · "
                f"rel={c.get('relationship')} · n={len(c.get('members', []))}"
            )
            console.print(f"  [dim]{_short(c.get('summary', ''), 200)}[/dim]")
            for m in c.get("members", []):
                console.print(f"    - {m['id'][:8]} · {_short(m.get('title', ''), 70)}")
        if len(dup_clusters) > 20:
            console.print(f"[dim]…and {len(dup_clusters) - 20} more[/dim]")
        console.print()
        console.print("[dim]Re-run with --apply to merge interactively.[/dim]")
        return

    for c in dup_clusters:
        console.print()
        console.print(
            f"[cyan]cluster {c.get('cluster_id', '?')}[/cyan] · "
            f"rel={c.get('relationship')} · n={len(c.get('members', []))}"
        )
        for m in c.get("members", []):
            console.print(f"    - {m['id'][:8]} · {_short(m.get('title', ''), 70)}")

        if not click.confirm("Propose merge for this cluster?", default=True):
            continue

        proposal = mem.consolidator.propose_merge(c)
        if proposal is None:
            console.print("[red]No merge proposal generated.[/red]")
            continue
        console.print(f"[bold]merged title:[/bold] {proposal.merged_title}")
        console.print(f"[dim]strategy={proposal.merge_strategy}[/dim]")
        console.print(f"[dim]rationale={proposal.rationale}[/dim]")

        if not click.confirm("Apply merge?", default=False):
            continue

        result = mem.consolidator.apply_merge(proposal, dry_run=dry_run)
        console.print(
            f"[green]merged →[/green] "
            f"{result.merged_id[:8] if result.merged_id else 'n/a'}  "
            f"archived={len(result.archived_ids)}"
        )


@cli.command(name="briefing")
def briefing() -> None:
    """SessionStart hook — rich context panel.

    Emits a `hookSpecificOutput` JSON with `additionalContext` markdown
    containing:
      - Last session for the current project (crash recovery)
      - Open loops: recently updated memories (in-flight decisions)
      - Memory of the day: one memory picked deterministically by date
      - Quick interaction guide

    All errors swallowed — a failed briefing is worse than no briefing
    only if it blocks the session. Exit 0 + `{}` on any failure.

    Env vars:
      MEMO_BRIEFING_DISABLE        — set to "1" to skip entirely
      MEMO_BRIEFING_LOOPS_N        — open-loop count to show (default 5)
      MEMO_BRIEFING_LOOPS_DAYS     — how recent counts as "open" (default 7)
      MEMO_BRIEFING_DEBUG          — print errors to stderr
    """
    import hashlib as _hashlib
    import json as _json
    import os as _os
    import sys as _sys
    from datetime import UTC, timedelta

    debug = _os.environ.get("MEMO_BRIEFING_DEBUG") == "1"

    def _bail(reason: str = "") -> None:
        if reason and debug:
            print(f"# memo briefing: {reason}", file=_sys.stderr)
        print("{}")
        _sys.exit(0)

    if _os.environ.get("MEMO_BRIEFING_DISABLE") == "1":
        _bail("disabled")
        return

    try:
        cfg = Config.from_env()
        from memo.memory import Memory
        mem = Memory(cfg)
    except Exception as exc:
        _bail(f"Memory init failed: {exc}")
        return

    loops_n = max(1, int(_os.environ.get("MEMO_BRIEFING_LOOPS_N", "5") or 5))
    loops_days = max(1, int(_os.environ.get("MEMO_BRIEFING_LOOPS_DAYS", "7") or 7))

    lines: list[str] = []

    # ── 1. Last session for this project ──────────────────────────────────
    try:
        from pathlib import Path as _Path

        from memo.session import format_relative, list_sessions

        cur_cwd = str(_Path(_os.getcwd()).resolve())
        all_sessions = list_sessions(cfg.state_dir, limit=20)
        same_proj = [r for r in all_sessions if (r.get("cwd") or "") == cur_cwd]
        if same_proj:
            top = same_proj[0]
            sid = top.get("session_id") or ""
            when = format_relative(top.get("updated"))
            summary = (
                top.get("summary") or top.get("last_user_msg") or "—"
            ).replace("\n", " ")[:120]
            lines.append("## El Briefing")
            lines.append("")
            lines.append(f"**Última sesión en este proyecto** ({when}): {summary}")
            lines.append(f"`claude --resume {sid}`")
            lines.append("")
    except Exception as exc:
        if debug:
            print(f"# memo briefing: session lookup failed: {exc}", file=_sys.stderr)
        if not lines:
            lines.append("## El Briefing")
            lines.append("")

    # ── 1b. Unified consciousness (Synapse) ───────────────────────────────
    # Pulls present_state (memflow handoffs/focus) + reality_conflicts from
    # `synapse packet`. No-op when synapse is not installed or unreachable —
    # the rest of the briefing is unaffected (graceful, opt-in boundary).
    if _os.environ.get("MEMO_BRIEFING_SYNAPSE_DISABLE") != "1":
        try:
            from memo.briefing import synapse_briefing_lines

            try:
                cur_cwd_str = _os.getcwd()
            except Exception:
                cur_cwd_str = ""
            syn_lines = synapse_briefing_lines(cur_cwd_str)
            if syn_lines:
                lines.extend(syn_lines)
        except Exception as exc:
            if debug:
                print(f"# memo briefing: synapse lookup failed: {exc}", file=_sys.stderr)

    # ── 2. Open loops: recently updated memories ──────────────────────────
    try:
        cutoff = (datetime.now(tz=UTC) - timedelta(days=loops_days)).isoformat()
        all_recent = mem.store.list_recent(limit=loops_n * 4)
        open_loops = [
            r for r in all_recent
            if (r.get("updated") or "") >= cutoff
        ][:loops_n]

        if open_loops:
            lines.append(f"### Loops abiertos (últimos {loops_days} días)")
            lines.append("")
            for i, r in enumerate(open_loops, start=1):
                tags = r.get("tags") or []
                if isinstance(tags, str):
                    import json as _j
                    try:
                        tags = _j.loads(tags)
                    except Exception:
                        tags = []
                tag_str = ", ".join(str(t) for t in tags[:3]) if tags else ""
                title = r.get("title") or "—"
                type_ = r.get("type") or "note"
                id_short = (r.get("id") or "")[:8]
                updated = r.get("updated") or ""
                try:
                    dt = datetime.fromisoformat(updated)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    delta = datetime.now(tz=UTC) - dt
                    days_ago = delta.days
                    age = f"hace {days_ago}d" if days_ago > 0 else "hoy"
                except Exception:
                    age = updated[:10]
                lines.append(
                    f"{i}. `{id_short}` **{type_}** · {title}"
                    + (f" — {age}" if age else "")
                    + (f" [{tag_str}]" if tag_str else "")
                )
            lines.append("")
    except Exception as exc:
        if debug:
            print(f"# memo briefing: open-loops failed: {exc}", file=_sys.stderr)

    # ── 3. Memory of the day (date-seeded, biased to least-recent) ────────
    try:
        # Use today's date as seed so the pick is stable within a day but
        # rotates daily. Favour memories whose `updated` is oldest (least
        # recently revisited) so the corpus gets covered over time.
        today_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        all_ids_rows = mem.store.list_recent(limit=500)
        if all_ids_rows:
            # Sort oldest-updated first so the seed picks from the back of
            # the corpus on average.
            sorted_rows = sorted(all_ids_rows, key=lambda r: r.get("updated") or "")
            seed_int = int(_hashlib.sha256(today_str.encode()).hexdigest(), 16)
            pick_row = sorted_rows[seed_int % len(sorted_rows)]
            pick_id = pick_row.get("id") or ""
            pick_rec = mem.get(pick_id) if pick_id else None
            if pick_rec:
                body_preview = (pick_rec.body or "").strip()[:200].replace("\n", " ")
                tags = pick_rec.tags or []
                tag_str = ", ".join(str(t) for t in tags[:4]) if tags else ""
                lines.append("### Memoria del día")
                lines.append("")
                lines.append(
                    f"`{pick_rec.id[:8]}` **{pick_rec.type}** · {pick_rec.title}"
                    + (f" [{tag_str}]" if tag_str else "")
                )
                if body_preview:
                    lines.append(f"> {body_preview}{'…' if len(pick_rec.body or '') > 200 else ''}")
                lines.append("")
    except Exception as exc:
        if debug:
            print(f"# memo briefing: memory-of-day failed: {exc}", file=_sys.stderr)

    # ── 4. Interaction guide ──────────────────────────────────────────────
    lines.append(
        "_Para continuar: `dame el loop N` (retoma por número) · "
        "`/memo get <id>` · `/memo ask <pregunta>`_"
    )

    if not any(ln for ln in lines if ln and not ln.startswith("#") and not ln.startswith("_")):
        _bail("nothing to show")
        return

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n".join(lines),
        }
    }
    print(_json.dumps(output, ensure_ascii=False))


@cli.command(name="mapa")
@click.option(
    "--output", "-o", default=None,
    help="Output HTML path. Default: ~/.local/share/memo/mapa.html",
)
@click.option(
    "--open/--no-open", "open_browser", default=True,
    help="Open in default browser after generating.",
)
@click.option(
    "--limit", default=500, show_default=True,
    help="Maximum number of memories to include.",
)
@click.option(
    "--animate/--no-animate", default=True,
    help="Include timeline animation slider.",
)
def mapa_cmd(output: str | None, open_browser: bool, limit: int, animate: bool) -> None:
    """Generate an interactive 2D semantic map of the memory corpus.

    Projects all memory embeddings (stored in memvec.db) to 2D space using
    UMAP when available, falling back to PCA via numpy. Renders a
    self-contained HTML file with Plotly — hover for preview, click to copy ID.

    Requirements:
      Mandatory: numpy (already a transitive dep via mlx/scipy)
      Optional:  umap-learn (pip install umap-learn) for better topology.
                 Without it the map uses PCA (fast but loses cluster structure).
    """
    import json as _json
    import sqlite3 as _sqlite3
    import struct
    import webbrowser as _wb
    from pathlib import Path as _Path

    cfg = Config.from_env()
    db_path = cfg.state_dir / "memvec.db"
    if not db_path.is_file():
        console.print(f"[red]DB not found:[/red] {db_path}. Run `memo reindex` first.")
        raise SystemExit(1)

    # ── Read embeddings + metadata directly from SQLite ───────────────────
    # We bypass VecStore to avoid loading MLX. sqlite-vec stores
    # FLOAT[N] columns as raw 4-byte little-endian blobs.
    console.print("[dim]Reading corpus from DB…[/dim]")
    try:
        import sqlite_vec as _sv  # type: ignore[import-untyped]

        conn = _sqlite3.connect(str(db_path), timeout=10.0)
        conn.enable_load_extension(True)
        _sv.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = _sqlite3.Row
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise SystemExit(1) from exc

    try:
        rows = conn.execute(
            "SELECT vec.id, vec.embedding, "
            "       meta.title, meta.type, meta.tags, "
            "       meta.created, meta.updated "
            "FROM vec JOIN meta ON meta.id = vec.id "
            "ORDER BY meta.updated DESC "
            f"LIMIT {int(limit)}"
        ).fetchall()
    except Exception as exc:
        console.print(f"[red]Query failed:[/red] {exc}")
        raise SystemExit(1) from exc
    finally:
        conn.close()

    if len(rows) < 3:
        console.print(
            f"[yellow]Not enough memories to map ({len(rows)} found, need ≥ 3).[/yellow]\n"
            "Save some memories first with `/memo save` or `memo capture-stop`."
        )
        raise SystemExit(0)

    ids: list[str] = []
    titles: list[str] = []
    types: list[str] = []
    tags_list: list[str] = []
    created_list: list[str] = []
    updated_list: list[str] = []
    raw_vecs: list[list[float]] = []

    for row in rows:
        blob = row["embedding"]
        if blob is None:
            continue
        n = len(blob) // 4
        vec = list(struct.unpack(f"<{n}f", blob))
        if not vec:
            continue
        ids.append(row["id"])
        titles.append(row["title"] or "—")
        types.append(row["type"] or "note")
        try:
            tags = _json.loads(row["tags"] or "[]")
            tags_list.append(", ".join(str(t) for t in tags[:4]) if tags else "")
        except Exception:
            tags_list.append("")
        created_list.append((row["created"] or "")[:10])
        updated_list.append((row["updated"] or "")[:10])
        raw_vecs.append(vec)

    n_pts = len(raw_vecs)
    if n_pts < 3:
        console.print(f"[yellow]Only {n_pts} memories have vectors. Run `memo reindex`.[/yellow]")
        raise SystemExit(0)

    # ── 2D projection ─────────────────────────────────────────────────────
    console.print(f"[dim]Projecting {n_pts} memories to 2D…[/dim]")
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        console.print("[red]numpy is required for mapa.[/red] Install: pip install numpy")
        raise SystemExit(1) from exc

    mat = np.array(raw_vecs, dtype=np.float32)

    xs: list[float]
    ys: list[float]
    method_name: str

    try:
        import umap  # type: ignore[import-not-found]

        n_neighbors = min(15, n_pts - 1)
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors,
            min_dist=0.1, metric="cosine", random_state=42,
        )
        coords = reducer.fit_transform(mat)
        xs = coords[:, 0].tolist()
        ys = coords[:, 1].tolist()
        method_name = "UMAP"
    except ImportError:
        # PCA fallback — fast, preserves global variance, loses local clusters.
        mat_centered = mat - mat.mean(axis=0)
        _, _, vt = np.linalg.svd(mat_centered, full_matrices=False)
        coords = mat_centered @ vt[:2].T
        xs = coords[:, 0].tolist()
        ys = coords[:, 1].tolist()
        method_name = "PCA (install umap-learn for better topology)"

    console.print(f"[green]✓[/green] Projected via {method_name}")

    # ── Build timeline frames for animation ───────────────────────────────
    # Sort unique dates; each frame shows memories up to that date.
    if animate:
        dates_sorted = sorted(set(created_list))
        frames_data: list[dict] = []
        for d in dates_sorted:
            mask = [c <= d for c in created_list]
            frames_data.append({
                "name": d,
                "x": [xs[i] for i, m in enumerate(mask) if m],
                "y": [ys[i] for i, m in enumerate(mask) if m],
                "ids": [ids[i][:8] for i, m in enumerate(mask) if m],
                "titles": [titles[i] for i, m in enumerate(mask) if m],
                "types": [types[i] for i, m in enumerate(mask) if m],
                "tags": [tags_list[i] for i, m in enumerate(mask) if m],
            })
    else:
        frames_data = []

    # ── Type → colour mapping ─────────────────────────────────────────────
    TYPE_COLORS = {
        "decision": "#4f8ef7",
        "fact": "#34d399",
        "bug": "#f87171",
        "preference": "#a78bfa",
        "feedback": "#fb923c",
        "note": "#94a3b8",
        "manual": "#e2e8f0",
    }

    # ── Emit self-contained HTML ──────────────────────────────────────────
    out_path = _Path(output) if output else cfg.state_dir / "mapa.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Encode data as JSON to embed in the HTML script block
    data_json = _json.dumps({
        "xs": xs,
        "ys": ys,
        "ids": [i[:8] for i in ids],
        "titles": titles,
        "types": types,
        "tags": tags_list,
        "created": created_list,
        "updated": updated_list,
        "frames": frames_data,
        "method": method_name,
        "n": n_pts,
        "type_colors": TYPE_COLORS,
    }, ensure_ascii=False)

    html = _MAPA_HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    out_path.write_text(html, encoding="utf-8")

    console.print(f"[green]✓[/green] Mapa saved → [bold]{out_path}[/bold]")
    console.print(
        f"[dim]{n_pts} memories · {method_name}[/dim]"
        + (" [dim]· animation enabled[/dim]" if animate and frames_data else "")
    )

    if open_browser:
        _wb.open(out_path.as_uri())


_MAPA_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>El Mapa — memo</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f172a; color: #e2e8f0; font-family: ui-monospace, 'Cascadia Code', monospace; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  #header { padding: 12px 20px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #1e293b; flex-shrink: 0; }
  #header h1 { font-size: 1rem; font-weight: 600; color: #f1f5f9; letter-spacing: 0.05em; }
  #header .meta { font-size: 0.75rem; color: #64748b; }
  #search-box { margin-left: auto; background: #1e293b; border: 1px solid #334155; color: #e2e8f0; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; outline: none; width: 220px; }
  #search-box:focus { border-color: #4f8ef7; }
  #plot { flex: 1; width: 100%; }
  #sidebar { position: fixed; right: 0; top: 0; bottom: 0; width: 320px; background: #1e293b; border-left: 1px solid #334155; padding: 20px; transform: translateX(100%); transition: transform 0.2s ease; overflow-y: auto; z-index: 10; }
  #sidebar.open { transform: translateX(0); }
  #sidebar-close { float: right; cursor: pointer; color: #64748b; font-size: 1.2rem; line-height: 1; margin-bottom: 16px; }
  #sidebar-close:hover { color: #e2e8f0; }
  #sidebar h2 { font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
  #sidebar-title { font-size: 1rem; font-weight: 600; color: #f1f5f9; margin-bottom: 6px; line-height: 1.4; }
  #sidebar-meta { font-size: 0.75rem; color: #64748b; margin-bottom: 12px; }
  #sidebar-id { font-size: 0.75rem; background: #0f172a; padding: 6px 10px; border-radius: 4px; color: #a78bfa; cursor: pointer; display: inline-block; margin-bottom: 12px; }
  #sidebar-id:hover { background: #1e293b; }
  #sidebar-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; }
  .tag-chip { font-size: 0.7rem; background: #0f172a; color: #94a3b8; padding: 2px 8px; border-radius: 10px; }
  #legend { position: fixed; bottom: 20px; left: 20px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; font-size: 0.73rem; }
  #legend h3 { color: #64748b; margin-bottom: 6px; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .legend-item { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  #toast { position: fixed; bottom: 20px; right: 20px; background: #1e293b; border: 1px solid #4f8ef7; color: #e2e8f0; padding: 8px 14px; border-radius: 6px; font-size: 0.8rem; opacity: 0; pointer-events: none; transition: opacity 0.3s; z-index: 20; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<div id="header">
  <h1>El Mapa</h1>
  <span class="meta" id="meta-label"></span>
  <input id="search-box" type="search" placeholder="Filtrar memorias…" />
</div>
<div id="plot"></div>
<div id="sidebar">
  <span id="sidebar-close" onclick="closeSidebar()">✕</span>
  <h2>Memoria</h2>
  <div id="sidebar-title"></div>
  <div id="sidebar-meta"></div>
  <div id="sidebar-id" onclick="copyId()" title="Click para copiar ID"></div>
  <div id="sidebar-tags"></div>
</div>
<div id="legend"><h3>Tipos</h3><div id="legend-items"></div></div>
<div id="toast" id="toast">ID copiado</div>

<script>
const DATA = __DATA_JSON__;
let currentId = null;

// Build legend
const li = document.getElementById('legend-items');
Object.entries(DATA.type_colors).forEach(([type, color]) => {
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = `<div class="legend-dot" style="background:${color}"></div><span>${type}</span>`;
  li.appendChild(item);
});
document.getElementById('meta-label').textContent =
  `${DATA.n} memorias · ${DATA.method}`;

// Point colours from type
const colors = DATA.types.map(t => DATA.type_colors[t] || '#94a3b8');

const hovertext = DATA.ids.map((id, i) =>
  `<b>${DATA.titles[i]}</b><br><span style="color:#94a3b8">${DATA.types[i]}</span>`
  + (DATA.tags[i] ? `<br><span style="color:#64748b">${DATA.tags[i]}</span>` : '')
  + `<br><span style="color:#475569">${DATA.created[i]}</span>`
);

const trace = {
  x: DATA.xs, y: DATA.ys,
  mode: 'markers',
  type: 'scatter',
  marker: {
    size: 8, color: colors, opacity: 0.85,
    line: { width: 0.5, color: '#1e293b' }
  },
  text: hovertext,
  hovertemplate: '%{text}<extra></extra>',
  customdata: DATA.ids,
};

const layout = {
  paper_bgcolor: '#0f172a',
  plot_bgcolor: '#0f172a',
  xaxis: { visible: false, zeroline: false },
  yaxis: { visible: false, zeroline: false },
  margin: { t: 10, l: 10, r: 10, b: 10 },
  hovermode: 'closest',
  hoverlabel: {
    bgcolor: '#1e293b', bordercolor: '#334155',
    font: { family: 'ui-monospace', size: 12, color: '#e2e8f0' }
  },
};

let frames = [];
let sliders = [];
if (DATA.frames && DATA.frames.length > 1) {
  frames = DATA.frames.map(f => ({
    name: f.name,
    data: [{
      x: f.x, y: f.y,
      text: f.ids.map((id, i) =>
        `<b>${f.titles[i]}</b><br><span style="color:#94a3b8">${f.types[i]}</span>`
      ),
      marker: { color: f.types.map(t => DATA.type_colors[t] || '#94a3b8') },
      customdata: f.ids,
    }]
  }));
  sliders = [{
    active: frames.length - 1,
    steps: DATA.frames.map((f, i) => ({
      label: f.name, method: 'animate',
      args: [[f.name], { mode: 'immediate', frame: { duration: 0 }, transition: { duration: 0 } }],
    })),
    x: 0.05, y: 0, xanchor: 'left', yanchor: 'top',
    len: 0.9,
    bgcolor: '#1e293b', bordercolor: '#334155',
    font: { color: '#64748b', size: 10 },
    currentvalue: { prefix: 'Hasta: ', font: { color: '#94a3b8', size: 11 }, xanchor: 'center' },
  }];
  layout.sliders = sliders;
}

Plotly.newPlot('plot', [trace], layout, { responsive: true, displayModeBar: false })
  .then(gd => {
    if (frames.length > 1) Plotly.addFrames(gd, frames);
  });

document.getElementById('plot').on('plotly_click', function(data) {
  const pt = data.points[0];
  const idx = pt.pointIndex;
  currentId = DATA.ids[idx];
  document.getElementById('sidebar-title').textContent = DATA.titles[idx];
  document.getElementById('sidebar-meta').textContent =
    `${DATA.types[idx]} · creado ${DATA.created[idx]} · actualizado ${DATA.updated[idx]}`;
  document.getElementById('sidebar-id').textContent = `/memo get ${currentId}`;
  const tagsEl = document.getElementById('sidebar-tags');
  tagsEl.innerHTML = '';
  (DATA.tags[idx] || '').split(',').filter(Boolean).forEach(t => {
    const chip = document.createElement('span');
    chip.className = 'tag-chip';
    chip.textContent = t.trim();
    tagsEl.appendChild(chip);
  });
  document.getElementById('sidebar').classList.add('open');
});

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
}

function copyId() {
  if (!currentId) return;
  navigator.clipboard.writeText(currentId).then(() => {
    const t = document.getElementById('toast');
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 1800);
  });
}

// Search filter
document.getElementById('search-box').addEventListener('input', function() {
  const q = this.value.toLowerCase().trim();
  if (!q) {
    Plotly.restyle('plot', { 'marker.opacity': [0.85] });
    return;
  }
  const opacities = DATA.titles.map((title, i) =>
    title.toLowerCase().includes(q) || DATA.tags[i].toLowerCase().includes(q) ||
    DATA.types[i].toLowerCase().includes(q) ? 0.9 : 0.08
  );
  Plotly.restyle('plot', { 'marker.opacity': [opacities] });
});
</script>
</body>
</html>"""


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
