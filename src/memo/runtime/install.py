"""Runtime + install plumbing for the memo CLI.

Extracted from cli.py (3a). Runtime/install-detection helpers
(_runtime_install_report, MCP wiring, codex/windsurf installers) + the
setup commands. doctor (cli.py) + cli_diag._doctor_report import
_runtime_install_report from here.
"""

from __future__ import annotations

import json
import os
import selectors
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from importlib.resources import files as package_files
from pathlib import Path
from typing import Any

import click

from memo.cli_common import console
from memo.config import Config
from memo.setup import run_picker, write_config_file


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
    candidates = (
        [_safe_resolve(repo)]
        if repo
        else [
            _safe_resolve(Path.cwd()),
            _safe_resolve(Path(__file__).resolve().parents[2]),
        ]
    )
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
            "devin",
            "mcp",
            "add",
            "-s",
            "user",
            *_env_flags("devin", env),
            "memo",
            "--",
            memo_mcp,
        ]
    mcp_json = json.dumps(_mcp_server_json(memo_mcp, env, include_type=True), separators=(",", ":"))
    return [
        "claude",
        "mcp",
        "add-json",
        "-s",
        "user",
        "memo",
        mcp_json,
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
        raise click.ClickException(f"Codex marketplace manifest not found: {marketplace}")
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
        raise click.ClickException("`codex` not found on PATH; install Codex first.") from exc

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


@click.command(name="init")
@click.option("--force", is_flag=True, help="Overwrite existing config without confirmation.")
def init_cmd(force: bool) -> None:
    """(Re)configure where memo stores memorias.

    Runs the interactive picker. On first run, the picker also fires
    automatically — `memo init` is for explicitly re-configuring later
    (e.g. moving to a new path, switching to/from an Obsidian vault).
    """
    from memo.setup.config_io import _resolve_config_path

    cfg_path = _resolve_config_path()
    if (
        cfg_path.is_file()
        and not force
        and not click.confirm(
            f"Config file exists at {cfg_path}. Overwrite?",
            default=False,
        )
    ):
        console.print("[yellow]aborted[/yellow]")
        return
    from memo.cli import _run_picker_and_save

    _run_picker_and_save()


def _consolidate_sidecar_dbs() -> None:
    """Merge the four sidecar DBs into the main memvec.db and flip MEMO_SINGLE_DB.

    For each legacy sidecar file present in state_dir, create its tables in the
    main DB (via the store constructors, so schemas/constraints are correct),
    ATTACH the legacy file, copy rows with INSERT OR IGNORE, then rename the
    legacy file to `*.db.bak`. Idempotent: absent legacy files are skipped and
    re-running is a no-op once the renames are done.
    """
    import sqlite3

    from memo.contradict import ContradictionStore
    from memo.crossref import CrossReferenceIndex
    from memo.graph import GraphStore
    from memo.history import HistoryStore

    # Read the CURRENT (pre-flip) config so the sidecar *_db properties still
    # point at the separate legacy files.
    cfg = Config.from_env()
    if cfg.single_db:
        console.print("[yellow]![/yellow] already in single_db mode — nothing to merge")
    main_db = cfg.db_path

    # Map each legacy file to the tables it owns. Listing tables dynamically
    # from the legacy file avoids hardcoding, but an explicit map documents the
    # contract and skips sqlite_* internals cleanly.
    legacy_tables: dict[Path, list[str]] = {
        cfg.state_dir / "history.db": ["events", "sync_state"],
        cfg.state_dir / "graph.db": ["entities", "entity_memoria"],
        cfg.state_dir / "contradictions.db": ["pairs"],
        cfg.state_dir / "crossref.db": ["backlinks"],
    }

    # 1. Create the sidecar tables in the MAIN db with correct DDL by pointing
    #    each store at main_db once. (Constructors run CREATE TABLE IF NOT EXISTS.)
    HistoryStore(main_db, device_id=cfg.device_id).close()
    GraphStore(main_db).close()
    ContradictionStore(main_db).close()
    CrossReferenceIndex(main_db).close()

    # 2. Copy rows from each legacy file, then rename it aside.
    merged_any = False
    conn = sqlite3.connect(str(main_db), timeout=10.0)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        for legacy, tables in legacy_tables.items():
            if not legacy.is_file():
                continue
            conn.execute("ATTACH DATABASE ? AS legacy", (str(legacy),))
            try:
                copied = 0
                present = {
                    str(r[0])
                    for r in conn.execute(
                        "SELECT name FROM legacy.sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                for tbl in tables:
                    if tbl not in present:
                        continue
                    cur = conn.execute(
                        f"INSERT OR IGNORE INTO main.{tbl} SELECT * FROM legacy.{tbl}"
                    )
                    copied += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
                conn.commit()
            finally:
                conn.execute("DETACH DATABASE legacy")
            bak = legacy.with_suffix(legacy.suffix + ".bak")
            legacy.replace(bak)
            console.print(f"[green]✓[/green] merged {legacy.name} → memvec.db, renamed → {bak.name}")
            merged_any = True
    finally:
        conn.close()

    if not merged_any:
        console.print("[dim]No legacy sidecar files found to merge.[/dim]")

    # 3. Flip the config toggle so future runs use the single DB.
    existing = Config.from_env()
    write_config_file(
        data_dir=existing.data_dir,
        vault_path=existing.vault_path,
        memories_in_vault=existing.memories_in_vault,
        single_db=True,
    )
    console.print("[green]✓[/green] set single_db=1 in config — memo now uses one DB file")


@click.command(name="migrate-vault")
@click.argument("new_data_dir", required=False, type=click.Path(file_okay=False, resolve_path=True))
@click.option(
    "--from",
    "from_dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Source memory_dir. Defaults to current cfg.memory_dir.",
)
@click.option(
    "--into-vault",
    is_flag=True,
    help="Move memorias INTO the Obsidian vault (<vault>/<SYSTEM_DIR>/AI/memory) "
    "and set memories_in_vault=1 so the vault becomes the source of truth.",
)
@click.option(
    "--rollback",
    is_flag=True,
    help="Restore the config snapshot taken by the last migration and exit. "
    "Copied files are left in place (migration never deletes anything).",
)
@click.option(
    "--consolidate-db",
    is_flag=True,
    help="Merge the sidecar DBs (history/graph/contradictions/crossref) into the "
    "main memvec.db, set MEMO_SINGLE_DB=1 in config, and rename the legacy files "
    "to *.db.bak (reversible). Idempotent. Does not move any .md files.",
)
@click.option("--force", is_flag=True, help="Overwrite destination even if non-empty.")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def migrate_vault(
    new_data_dir: str | None,
    from_dir: str | None,
    into_vault: bool,
    rollback: bool,
    consolidate_db: bool,
    force: bool,
    yes: bool,
) -> None:
    """Move memorias to a new location; rewrites config + reindexes.

    Copies all `.md` files (preserving mtime via `shutil.copy2`), updates
    `~/.config/memo/config.toml`, and runs `memo reindex` from the new
    location. The index DB (`memvec.db`) is **preserved** — it lives in
    `state_dir` (unchanged by this command) and memoria rows are keyed on a
    stable `id`, so reindex updates them in place. User-signal data
    (feedback votes, access counts, health scores) therefore survives the
    migration. (Earlier versions deleted `memvec.db`, silently wiping that
    signal — that data-loss bug is fixed.)

    With `--into-vault`, memorias move under `<vault>/<SYSTEM_DIR>/AI/memory`
    and `memories_in_vault=1` is written so the human-editable Obsidian vault
    is the source of truth and sqlite stays a rebuildable index.

    The original `.md` files are NOT deleted — once you've verified the
    migration with `memo search`, you can `rm -rf <old-dir>` manually.
    A snapshot of the prior config is written so `--rollback` can restore it.
    """
    import shutil
    from pathlib import Path as _Path

    from memo.config import AI_SUBDIR
    from memo.memory import Memory
    from memo.setup.config_io import _resolve_config_path

    # `--rollback`: restore the pre-migration config snapshot and stop.
    snapshot = _resolve_config_path().with_suffix(".toml.pre-migrate.bak")
    if rollback:
        if not snapshot.is_file():
            console.print(f"[red]✗[/red] no migration snapshot found at {snapshot}")
            sys.exit(1)
        shutil.copy2(snapshot, _resolve_config_path())
        console.print(f"[green]✓[/green] restored config from snapshot {snapshot}")
        console.print(
            "[dim]Copied memoria files were left in place; remove them manually "
            "if you no longer want them.[/dim]"
        )
        return

    if consolidate_db:
        _consolidate_sidecar_dbs()
        return

    cfg = Config.from_env()

    # 1. Resolve source.
    src = _Path(from_dir).resolve() if from_dir else cfg.memory_dir
    if not src.is_dir():
        console.print(f"[red]✗[/red] source dir does not exist: {src}")
        sys.exit(1)

    # 2. Resolve destination + (optional) new vault_path.
    if into_vault:
        chosen_vault = cfg.vault_path
        if chosen_vault is None:
            console.print(
                "[red]✗[/red] --into-vault needs a vault: set MEMO_VAULT_PATH or run "
                "`memo init` and pick an Obsidian vault first."
            )
            sys.exit(1)
        dst = (chosen_vault / AI_SUBDIR / "memory").resolve()
    elif new_data_dir:
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
            f"[red]✗[/red] destination is non-empty: {dst}\n  Use --force to overwrite.",
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

    # 4. Snapshot the existing config (for --rollback), then update it.
    #    The index DB is intentionally NOT dropped: it lives in state_dir
    #    (unchanged) and rows are keyed on a stable id, so reindex updates
    #    them in place and user-signal tables (feedback/access/health) survive.
    existing_cfg = _resolve_config_path()
    if existing_cfg.is_file():
        shutil.copy2(existing_cfg, snapshot)
        console.print(f"[green]✓[/green] config snapshot → {snapshot} (use --rollback to restore)")
    if into_vault:
        # data_dir is kept as-is; memory_dir derives from the vault when the
        # toggle is on, so the vault layout is what reindex will read.
        cfg_path = write_config_file(
            data_dir=cfg.data_dir, vault_path=chosen_vault, memories_in_vault=True
        )
    else:
        cfg_path = write_config_file(data_dir=dst, vault_path=chosen_vault)
    console.print(f"[green]✓[/green] config: {cfg_path}")

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


@click.command(name="mcp-command")
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
        click.echo(
            json.dumps(
                {
                    "mcpServers": {
                        "memo": _mcp_server_json(memo_mcp, env, include_type=True),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if client == "claude-desktop":
        click.echo(
            json.dumps(
                {
                    "mcpServers": {
                        "memo": _mcp_server_json(memo_mcp, env, include_type=False),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if client == "windsurf":
        click.echo(
            json.dumps(
                {
                    "mcpServers": {
                        "memo": _mcp_server_json(memo_mcp, env, include_type=False),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if client == "codex":
        click.echo(_format_command(_mcp_add_command("codex", memo_mcp, env)))
        return
    if client == "devin":
        click.echo(_format_command(_mcp_add_command("devin", memo_mcp, env)))
        return
    click.echo(_format_command(_mcp_add_command("claude-code", memo_mcp, env)))


@click.command(name="install-slash")
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
        console.print(
            "[green]✓[/green] agent-client install complete. Open a new agent session to reload."
        )


@click.command(name="self-update")
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
            capture_output=True,
            text=True,
            timeout=10,
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
                capture_output=True,
                text=True,
                timeout=10,
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


@click.command(name="install-shell-wrapper")
@click.option(
    "--print",
    "do_print",
    is_flag=True,
    help="Print the wrapper snippet to stdout. Default mode "
    "when neither --print nor --write is set.",
)
@click.option(
    "--write",
    "do_write",
    is_flag=True,
    help="Write ~/.zsh/memo-wrapper.zsh and append the matching "
    "`source` line to ~/.zshrc (idempotent).",
)
@click.option(
    "--shell",
    "shell_kind",
    type=click.Choice(["zsh", "bash"]),
    default="zsh",
    show_default=True,
    help="Target shell. zsh is the macOS default.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite ~/.zsh/memo-wrapper.zsh even if its content differs from what we would write.",
)
def install_shell_wrapper(
    do_print: bool,
    do_write: bool,
    shell_kind: str,
    force: bool,
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
                "\n(pasale `--write` para instalar en ~/.zsh/memo-wrapper.zsh + ~/.zshrc.)",
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
