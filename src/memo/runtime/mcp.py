from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Sequence
from importlib.resources import files as package_files
from pathlib import Path
from typing import Any

import click

from memo.cli_common import console

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


def _mcp_server_env() -> dict[str, str]:
    env = {"MEMO_NONINTERACTIVE": "1"}
    for key in _MCP_ENV_FORWARD_KEYS:
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


def _env_flags(client: str, env: dict[str, str]) -> list[str]:
    opt = "--env" if client in {"codex", "opencode"} else "-e"
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
    from memo.runtime.detect import _safe_resolve

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
    if client == "opencode":
        return ["opencode", "mcp", "add", "memo", *_env_flags("opencode", env), "--", memo_mcp]
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
    return ["claude", "mcp", "add-json", "-s", "user", "memo", mcp_json]


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
