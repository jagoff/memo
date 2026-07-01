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
    "MEMO_MCP_PROFILE",
    "MEMO_MODEL_PROFILE",
    "MEMO_LLM_MODEL",
    "MEMO_HELPER_MODEL",
    # MEMO_EMBEDDER_MODEL / MEMO_EMBEDDER_DIMS are NOT forwarded from env here —
    # they are derived from the live index (schema_meta) by _actual_embedder_config()
    # so the installed config always matches the existing index, regardless of what
    # the shell env has at install time.
    "MEMO_RERANKER_ENABLED",
    "MEMO_RERANKER_MODEL",
    "MEMO_RERANKER_REVISION",
    "MEMO_RERANK_INPUT_K",
    "MEMO_RERANK_FUSION_ALPHA",
    # auto-update: forwarded only when set at install time, so a machine opts in
    # with `MEMO_AUTO_UPDATE=1 memo install-mcp --write` (off by default).
    "MEMO_AUTO_UPDATE",
    "MEMO_AUTO_UPDATE_INTERVAL_S",
    "MEMO_AUTO_UPDATE_REPO",
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


def _actual_embedder_config() -> dict[str, str]:
    """Read embedder model+dims from the live index (schema_meta).

    Returns {} on any failure so callers can fall back gracefully.  Reading
    from the index — not from env vars — is the only way to guarantee the MCP
    config matches the vectors already on disk.
    """
    try:
        import sqlite3

        from memo.config import Config

        cfg = Config.from_env()
        db = cfg.db_path
        if not db.is_file():
            return {}
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        try:
            model_row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'embedder_model'"
            ).fetchone()
            dims_row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'embedder_dims'"
            ).fetchone()
        finally:
            conn.close()
        result: dict[str, str] = {}
        if model_row and model_row["value"]:
            result["MEMO_EMBEDDER_MODEL"] = model_row["value"]
        if dims_row and dims_row["value"]:
            result["MEMO_EMBEDDER_DIMS"] = dims_row["value"]
        return result
    except Exception:
        return {}


def _mcp_server_env() -> dict[str, str]:
    from memo.flags import flag_str

    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_MCP_PROFILE": flag_str("MEMO_MCP_PROFILE") or "agent",
        # MCP clients may inherit a dev shell's PYTHONPATH (for example `src`
        # inside this checkout). Clear it so the isolated memo-mcp shim imports
        # its installed runtime, not whichever repo happens to be the cwd.
        "PYTHONPATH": "",
        # Keep installs current: memo-mcp checks for a newer git TAG on start and
        # self-upgrades (tag-gated + throttled). This is what makes the [MEMO <ver>]
        # statusline badge follow releases automatically. Set MEMO_AUTO_UPDATE=0 in
        # the client env to opt out.
        "MEMO_AUTO_UPDATE": "1",
    }
    for key in _MCP_ENV_FORWARD_KEYS:
        val = os.environ.get(key)
        # `is not None` (not truthiness): an explicit empty value — e.g.
        # `MEMO_AUTO_UPDATE=` to opt out — must be forwarded so it overrides the
        # hardcoded default above instead of being silently dropped.
        if val is not None:
            env[key] = val
    # Derive embedder model/dims from the live index so the installed config
    # always matches existing vectors, overriding any env var set at install time.
    env.update(_actual_embedder_config())
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


def _config_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else Path.home() / p


def _strip_jsonc_comments(text: str) -> str:
    """Drop ``//`` line and ``/* */`` block comments, respecting string literals.

    A char-level scanner, so a ``//`` or ``/*`` inside a JSON string (e.g. a URL)
    is never mistaken for a comment. Trailing commas are handled separately by
    :func:`_strip_trailing_commas`.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_str = escape = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] not in "\r\n":
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    """Drop trailing commas before ``}`` / ``]``, respecting string literals.

    JSONC (VS Code, Zed) allows a comma before a closing brace/bracket, which
    ``json.loads`` rejects. String-aware so a comma inside a value is untouched.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_str = escape = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1  # drop the trailing comma
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _loads_jsonc(text: str, path: Path) -> Any:
    """Parse JSON, tolerating JSONC (comments) on fallback.

    Editor MCP configs (VS Code, Zed, Cursor) are commonly JSONC. Strict JSON is
    tried first so valid files stay untouched; only on failure are comments
    stripped and the text reparsed. Re-serialization writes plain JSON — the
    config data is preserved, but comments are not.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(_strip_trailing_commas(_strip_jsonc_comments(text)))
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"MCP config is not valid JSON/JSONC: {path} ({exc})"
            ) from exc


def _write_mcp_json(path: Path, server: Any, *, json_key: str, include_type: bool) -> str:
    data: dict[str, Any]
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if text.strip():
        loaded = _loads_jsonc(text, path)
        if not isinstance(loaded, dict):
            raise click.ClickException(f"MCP config must be a JSON object: {path}")
        data = loaded
        action = "updated"
    else:
        data = {}
        action = "created"

    servers = data.setdefault(json_key, {})
    if not isinstance(servers, dict):
        raise click.ClickException(f"`{json_key}` must be a JSON object in {path}")
    servers[server.name] = _mcp_server_json(
        Path(server.command), server.env, include_type=include_type
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return action


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
    best_effort: bool = False,
) -> None:
    if dry_run:
        click.echo(f"$ {_format_command(args)}")
        return
    try:
        proc = subprocess.run(
            [str(arg) for arg in args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"`{args[0]}` not found on PATH; install that client first."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise click.ClickException(f"Command timed out (30s): {_format_command(args)}") from exc
    combined = f"{proc.stdout}\n{proc.stderr}".lower()
    # best_effort: a non-zero exit is tolerated regardless of the message. This is
    # for the pre-`add` `mcp remove` cleanup, where a fresh machine has no prior
    # entry and the client emits an error string we can't enumerate (e.g. Claude's
    # `No MCP server named "memo" in user scope`, Devin's `not in the user config.
    # It is configured by Claude.`). The flow must still proceed to `mcp add`.
    tolerated = best_effort or any(token in combined for token in ok_errors)
    if proc.returncode != 0 and not tolerated:
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


def _devin_desktop_mcp_config_path() -> Path:
    raw = os.environ.get("DEVIN_DESKTOP_MCP_CONFIG")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".devin" / "mcp.json"


def _install_devin_desktop_mcp(
    memo_mcp: Path, env: dict[str, str], *, dry_run: bool
) -> None:
    path = _devin_desktop_mcp_config_path()
    server_config = _mcp_server_json(memo_mcp, env, include_type=True)
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
                f"Devin Desktop MCP config is not valid JSON: {path} ({exc})"
            ) from exc
        if not isinstance(loaded, dict):
            raise click.ClickException(f"Devin Desktop MCP config must be a JSON object: {path}")
        data = loaded
    else:
        data = {}

    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise click.ClickException(f"`mcpServers` must be a JSON object in {path}")
    servers["memo"] = server_config

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    console.print(f"[green]✓[/green] wrote Devin Desktop MCP config: {path}")


def _write_yaml_continue(path: Path, server: Any) -> str:
    """Write Continue's dedicated per-server block file (memo-owned; overwrite)."""
    import yaml

    doc = {
        "name": "memo",
        "version": "0.0.1",
        "schema": "v1",
        "mcpServers": [
            {"name": server.name, "command": str(server.command), "args": [], "env": dict(server.env)}
        ],
    }
    action = "updated" if path.is_file() else "created"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, path)
    return action


def _write_yaml_goose(path: Path, server: Any) -> str:
    """Merge memo into Goose's config.yaml `extensions` map (round-trip, non-clobbering).

    Uses `cmd` (not `command`) and `envs` (a value map — required so the embedder
    model/dims reach memo-mcp; `env_keys` names alone would drop the values).
    """
    import yaml

    if path.is_file() and path.read_text(encoding="utf-8").strip():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise click.ClickException(f"Goose config is not valid YAML: {path} ({exc})") from exc
        if not isinstance(loaded, dict):
            raise click.ClickException(f"Goose config must be a YAML mapping: {path}")
        data = loaded
        action = "updated"
    else:
        data = {}
        action = "created"

    exts = data.setdefault("extensions", {})
    if not isinstance(exts, dict):
        raise click.ClickException(f"`extensions` must be a mapping in {path}")
    exts["memo"] = {
        "type": "stdio",
        "name": "memo",
        "cmd": str(server.command),
        "args": [],
        "envs": dict(server.env),
        "timeout": 300,
        "enabled": True,
        "bundled": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, path)
    return action
