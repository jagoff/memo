"""Declarative setup registry for memo's first-class agent integrations.

The registry is deliberately small: it owns the end-to-end contract for the
agents memo can verify completely. Broader MCP-only compatibility remains in
``memo install-mcp`` and can migrate here adapter by adapter.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.cli_mandate import _END_MARKER, _MARKER, _write_mandate
from memo.errors import SetupError
from memo.runtime.mcp import _format_command, _mcp_add_command, _mcp_server_env


@dataclass(frozen=True)
class AgentAdapter:
    slug: str
    binary: str
    instruction_file: str
    mcp_profile: str
    protocol_mode: str
    restart_guidance: str
    verification_command: tuple[str, ...]
    remove_command: tuple[str, ...]
    rollback_limitations: str


@dataclass(frozen=True)
class SetupAction:
    agent: str
    binary: str
    detected: bool
    mcp_profile: str
    protocol_mode: str
    mcp_command: tuple[str, ...]
    remove_command: tuple[str, ...]
    shadow_remove_commands: tuple[tuple[str, ...], ...]
    instruction_path: str
    instruction_present: bool
    restart_guidance: str
    rollback_limitations: str


@dataclass(frozen=True)
class SetupPlan:
    memo_mcp: str
    cwd: str
    actions: tuple[SetupAction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "memo_mcp": self.memo_mcp,
            "cwd": self.cwd,
            "actions": [asdict(action) for action in self.actions],
        }


AGENT_REGISTRY: dict[str, AgentAdapter] = {
    "codex": AgentAdapter(
        slug="codex",
        binary="codex",
        instruction_file="AGENTS.md",
        mcp_profile="core",
        protocol_mode="compact",
        restart_guidance="Restart Codex so it opens a fresh MCP connection.",
        verification_command=("codex", "mcp", "get", "memo", "--json"),
        remove_command=("codex", "mcp", "remove", "memo"),
        rollback_limitations="The Codex CLI does not expose the previous MCP entry for restoration.",
    ),
    "claude-code": AgentAdapter(
        slug="claude-code",
        binary="claude",
        instruction_file="CLAUDE.md",
        mcp_profile="agent",
        protocol_mode="compact",
        restart_guidance="Restart Claude Code so it opens a fresh MCP connection.",
        verification_command=("claude", "mcp", "get", "memo"),
        remove_command=("claude", "mcp", "remove", "-s", "user", "memo"),
        rollback_limitations="The Claude CLI does not expose the previous MCP entry for restoration.",
    ),
}


CommandRunner = Callable[[tuple[str, ...], bool], None]


def _agent_cli_env() -> dict[str, str]:
    """Environment for non-interactive agent configuration commands.

    Agent launch wrappers may perform startup recall/capture intended only for
    interactive sessions. Bypass those hooks here so setup and doctor cannot
    deadlock behind an unrelated memory/GPU operation.
    """
    env = dict(os.environ)
    env.update(
        {
            "MEMFLOW_CAPTURE_DISABLE": "1",
            "MEMFLOW_STARTUP_BANNER": "0",
            "MEMO_NONINTERACTIVE": "1",
        }
    )
    return env


def _default_runner(argv: tuple[str, ...], best_effort: bool) -> None:
    try:
        proc = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=_agent_cli_env(),
        )
    except FileNotFoundError as exc:
        raise SetupError(f"`{argv[0]}` not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise SetupError(f"command timed out: {_format_command(argv)}") from exc
    if proc.returncode and not best_effort:
        detail = (proc.stderr or proc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise SetupError(f"command failed ({proc.returncode}): {_format_command(argv)}{suffix}")


def resolve_isolated_memo_mcp() -> Path | None:
    """Resolve a persistent runtime and reject project virtual environments."""
    home = Path.home()
    candidates = (
        home / ".local" / "bin" / "memo-mcp",
        home / ".local" / "pipx" / "venvs" / "mlx-memo" / "bin" / "memo-mcp",
        Path("/opt/homebrew/bin/memo-mcp"),
        Path("/usr/local/bin/memo-mcp"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which("memo-mcp")
    if found and ".venv" not in found:
        return Path(found)
    return None


def detected_agents(*, which: Callable[[str], str | None] | None = None) -> tuple[str, ...]:
    resolver = which or shutil.which
    return tuple(slug for slug, adapter in AGENT_REGISTRY.items() if resolver(adapter.binary))


def _expand_targets(targets: Iterable[str], *, detect: bool) -> tuple[str, ...]:
    selected = list(targets)
    if not selected or "all" in selected:
        selected = list(AGENT_REGISTRY)
    unknown = sorted(set(selected) - set(AGENT_REGISTRY))
    if unknown:
        raise SetupError(f"unsupported setup agent(s): {', '.join(unknown)}")
    if detect:
        present = set(detected_agents())
        selected = [slug for slug in selected if slug in present]
    return tuple(dict.fromkeys(selected))


def build_setup_plan(
    targets: Iterable[str],
    *,
    cwd: Path | None = None,
    detect: bool = False,
    memo_mcp: Path | None = None,
) -> SetupPlan:
    root = (cwd or Path.cwd()).resolve()
    runtime = memo_mcp or resolve_isolated_memo_mcp()
    if runtime is None:
        raise SetupError(
            "isolated memo-mcp not found; install with `uv tool install mlx-memo` "
            "or `pipx install mlx-memo` first"
        )
    if ".venv" in str(runtime):
        raise SetupError("refusing to persist a project .venv memo-mcp path")

    actions: list[SetupAction] = []
    for slug in _expand_targets(targets, detect=detect):
        adapter = AGENT_REGISTRY[slug]
        env = {
            **_mcp_server_env(),
            "MEMO_SOURCE": slug,
            "MEMO_MCP_PROFILE": adapter.mcp_profile,
        }
        mcp_argv = tuple(str(value) for value in _mcp_add_command(slug, runtime, env))
        remove_command = adapter.remove_command
        shadow_remove_commands: tuple[tuple[str, ...], ...] = ()
        if slug == "claude-code" and _project_has_memo_server(root):
            mcp_parts = list(mcp_argv)
            mcp_parts[mcp_parts.index("user")] = "project"
            mcp_argv = tuple(mcp_parts)
            remove_command = ("claude", "mcp", "remove", "-s", "project", "memo")
            # Project scope wins in this checkout. Remove a stale user entry
            # after the project entry is healthy so Claude reports no
            # conflicting scopes.
            shadow_remove_commands = (("claude", "mcp", "remove", "-s", "user", "memo"),)
        instruction_path = root / adapter.instruction_file
        instruction_present = bool(
            instruction_path.is_file() and _MARKER in instruction_path.read_text(encoding="utf-8")
        )
        actions.append(
            SetupAction(
                agent=slug,
                binary=adapter.binary,
                detected=shutil.which(adapter.binary) is not None,
                mcp_profile=adapter.mcp_profile,
                protocol_mode=adapter.protocol_mode,
                mcp_command=mcp_argv,
                remove_command=remove_command,
                shadow_remove_commands=shadow_remove_commands,
                instruction_path=str(instruction_path),
                instruction_present=instruction_present,
                restart_guidance=adapter.restart_guidance,
                rollback_limitations=adapter.rollback_limitations,
            )
        )
    return SetupPlan(memo_mcp=str(runtime), cwd=str(root), actions=tuple(actions))


def _project_has_memo_server(root: Path) -> bool:
    config = root / ".mcp.json"
    if not config.is_file():
        return False
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    return isinstance(servers, dict) and "memo" in servers


def _backup_path(path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return path.with_name(f"{path.name}.memo-backup-{stamp}")


def _verify_setup_action(
    result: dict[str, Any],
    action: SetupAction,
    plan: SetupPlan,
) -> None:
    if not result["ok"]:
        return
    verification = verify_agent(
        action.agent,
        cwd=Path(plan.cwd),
        memo_mcp=Path(plan.memo_mcp),
        probe=False,
    )
    result["verification"] = verification["checks"]
    if not verification["ok"]:
        result.update(
            ok=False,
            status="verification-failed",
            remediation=f"memo doctor --agent {action.agent}",
        )


def apply_setup_plan(
    plan: SetupPlan,
    *,
    dry_run: bool = False,
    runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    """Apply an immutable plan and return a receipt for every selected agent."""
    results: list[dict[str, Any]] = []
    for action in plan.actions:
        result: dict[str, Any] = {
            "agent": action.agent,
            "ok": True,
            "status": "would-apply" if dry_run else "applied",
            "mcp_command": list(action.mcp_command),
            "instruction_path": action.instruction_path,
            "restart_guidance": action.restart_guidance,
        }
        if dry_run:
            result["instruction"] = (
                "already-present" if action.instruction_present else "would-write"
            )
            results.append(result)
            continue

        # Both clients reject duplicate names. Removing an absent entry is safe;
        # removal remains best-effort because its wording differs by client.
        try:
            runner(action.remove_command, True)
            runner(action.mcp_command, False)
            for shadow_remove in action.shadow_remove_commands:
                runner(shadow_remove, True)
        except Exception as exc:
            result.update(
                ok=False,
                status="partial",
                error=str(exc),
                remediation=_format_command(action.mcp_command),
                rollback_limitations=action.rollback_limitations,
            )
            results.append(result)
            continue

        target = Path(action.instruction_path)
        backup: Path | None = None
        try:
            if not action.instruction_present and target.is_file():
                backup = _backup_path(target)
                shutil.copy2(target, backup)
            result["instruction"] = _write_mandate(target, dry_run=False)
            if backup is not None:
                result["backup"] = str(backup)
        except Exception as exc:
            rollback_error: str | None = None
            try:
                runner(action.remove_command, True)
            except Exception as rollback_exc:  # pragma: no cover - defensive
                rollback_error = str(rollback_exc)
            if backup is not None and backup.is_file():
                shutil.copy2(backup, target)
            result.update(
                ok=False,
                status="partial" if rollback_error else "rolled-back",
                error=str(exc),
                remediation=(
                    _format_command(action.remove_command)
                    if rollback_error
                    else "fix the instruction-file permissions and rerun memo setup"
                ),
            )
            if rollback_error:
                result["rollback_error"] = rollback_error
        _verify_setup_action(result, action, plan)
        results.append(result)

    return {
        "ok": all(result["ok"] for result in results),
        "dry_run": dry_run,
        "memo_mcp": plan.memo_mcp,
        "results": results,
    }


def _path_writable(path: Path) -> bool:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.exists() and os_access_writable(candidate)


def _isolated_runtime_smoke(
    memo_cli: Path,
    *,
    expected_version: str,
) -> tuple[bool, bool, str]:
    """Exercise config, deferred save, and BM25 search without a real vault/model."""
    with tempfile.TemporaryDirectory(prefix="memo-agent-doctor-") as tmp:
        root = Path(tmp)
        env = dict(os.environ)
        env.update(
            {
                "MEMO_NONINTERACTIVE": "1",
                "MEMO_DATA_DIR": str(root / "data"),
                "MEMO_STATE_DIR": str(root / "state"),
                "MEMO_VAULT_PATH": str(root / "vault"),
                "MEMO_CONFIG_DIR": str(root / "config"),
                "MEMO_CONFIG_FILE": str(root / "config.toml"),
                "MEMO_EMBEDDER_VIA_DAEMON": "0",
                "MEMO_MEMORIES_IN_VAULT": "0",
            }
        )

        def run(argv: list[str], *, timeout: int = 20) -> subprocess.CompletedProcess[str]:
            proc = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            if proc.returncode:
                detail = (proc.stderr or proc.stdout or "").strip()[:500]
                raise SetupError(f"{' '.join(argv[1:])} failed: {detail}")
            return proc

        try:
            version_proc = run([str(memo_cli), "--version"])
            version_ok = expected_version != "unknown" and expected_version in (
                version_proc.stdout + version_proc.stderr
            )
            run([str(memo_cli), "config", "validate"])
            saved = run(
                [
                    str(memo_cli),
                    "save",
                    "memo agent doctor isolated smoke token",
                    "--title",
                    "Agent doctor smoke",
                    "--type",
                    "fact",
                    "--defer-embed",
                    "--no-project-tag",
                    "--json",
                ]
            )
            saved_id = str(json.loads(saved.stdout)["id"])
            searched = run(
                [
                    str(memo_cli),
                    "search",
                    "agent doctor isolated smoke token",
                    "--mode",
                    "bm25",
                    "--no-rerank",
                    "--json",
                ]
            )
            hits = json.loads(searched.stdout)
            smoke_ok = any(str(row.get("id")) == saved_id for row in hits)
            return smoke_ok, version_ok, "isolated config/save/BM25 search passed"
        except (OSError, subprocess.TimeoutExpired, SetupError, ValueError, KeyError) as exc:
            return False, False, str(exc)[:500]


def _runtime_command_matches(command: str, runtime: Path | None) -> bool:
    if not command or runtime is None:
        return False
    candidate = Path(command)
    if not candidate.is_absolute():
        try:
            parsed = shlex.split(command)
        except ValueError:
            return False
        if len(parsed) != 1:
            return False
        candidate = Path(parsed[0])
    if not candidate.is_absolute():
        resolved = shutil.which(str(candidate))
        if not resolved:
            return False
        candidate = Path(resolved)
    try:
        return candidate.resolve() == runtime.resolve()
    except OSError:
        return candidate == runtime


def _runtime_detail_matches(raw_detail: str, runtime: Path | None) -> bool:
    for line in raw_detail.splitlines():
        key, separator, value = line.partition(":")
        if (
            separator
            and key.strip().lower() == "command"
            and _runtime_command_matches(value.strip(), runtime)
        ):
            return True
    return False


def _environment_segment_profile_matches(segment: str, expected_profile: str) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    assignments = [token.partition("=") for token in tokens]
    valid = bool(assignments) and all(
        separator and key.isidentifier() for key, separator, _ in assignments
    )
    return valid and any(
        key == "MEMO_MCP_PROFILE" and value == expected_profile
        for key, _separator, value in assignments
    )


def _profile_detail_matches(raw_detail: str, expected_profile: str) -> bool:
    environment_indent: int | None = None
    for raw_line in raw_detail.splitlines():
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip())
        if environment_indent is not None:
            if stripped and indent > environment_indent:
                if _environment_segment_profile_matches(stripped, expected_profile):
                    return True
                continue
            environment_indent = None
        key, separator, value = stripped.partition(":")
        if separator and key.strip().lower() in {"env", "environment"}:
            if _environment_segment_profile_matches(value.strip(), expected_profile):
                return True
            if not value.strip():
                environment_indent = indent
    return False


def _probe_agent_configuration(
    *,
    adapter: AgentAdapter,
    slug: str,
    root: Path,
    runtime: Path | None,
    probe: bool,
    binary: str | None,
) -> tuple[bool | None, bool | None, bool | None, str]:
    if not probe or not binary:
        return None, None, None, "not probed"
    try:
        proc = subprocess.run(
            adapter.verification_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=_agent_cli_env(),
            cwd=root,
        )
        config_ok = proc.returncode == 0
        raw_detail = (proc.stdout + "\n" + proc.stderr).strip()
        parsed: dict[str, Any] | None = None
        if slug == "codex" and proc.returncode == 0:
            try:
                candidate = json.loads(proc.stdout)
                parsed = candidate if isinstance(candidate, dict) else None
            except ValueError:
                parsed = None
        if parsed is None:
            config_runtime = _runtime_detail_matches(raw_detail, runtime)
            profile_current = _profile_detail_matches(raw_detail, adapter.mcp_profile)
        else:
            transport_value = parsed.get("transport")
            transport = transport_value if isinstance(transport_value, dict) else {}
            command = str(transport.get("command") or "")
            env_value = transport.get("env")
            mcp_env = env_value if isinstance(env_value, dict) else {}
            config_runtime = _runtime_command_matches(command, runtime)
            profile_current = mcp_env.get("MEMO_MCP_PROFILE") == adapter.mcp_profile
        return config_ok, config_runtime, profile_current, raw_detail[:500]
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, False, False, str(exc)


def _probe_storage_writable(*, probe: bool) -> bool | None:
    if not probe:
        return None
    try:
        from memo.config import Config

        cfg = Config.from_env()
        return _path_writable(cfg.data_dir) and _path_writable(cfg.state_dir)
    except Exception:
        return False


def verify_agent(
    slug: str,
    *,
    cwd: Path | None = None,
    memo_mcp: Path | None = None,
    probe: bool = True,
) -> dict[str, Any]:
    if slug not in AGENT_REGISTRY:
        raise SetupError(f"unsupported setup agent: {slug}")
    adapter = AGENT_REGISTRY[slug]
    root = (cwd or Path.cwd()).resolve()
    runtime = memo_mcp or resolve_isolated_memo_mcp()
    binary = shutil.which(adapter.binary)
    instruction = root / adapter.instruction_file
    try:
        instruction_text = instruction.read_text(encoding="utf-8") if instruction.is_file() else ""
    except OSError:
        instruction_text = ""
    marker = _MARKER in instruction_text
    protocol_current = marker and _END_MARKER in instruction_text
    writable_target = instruction if instruction.exists() else instruction.parent
    writable = _path_writable(writable_target)
    runtime_ok = bool(runtime and runtime.is_file() and ".venv" not in str(runtime))
    memo_cli = runtime.with_name("memo") if runtime else None
    runtime_pair = bool(
        runtime_ok
        and runtime is not None
        and memo_cli is not None
        and memo_cli.is_file()
        and memo_cli.parent == runtime.parent
    )

    # The version this process *is* — `memo doctor` prints it, and the smoke
    # probe asserts the isolated `memo` binary reports the same one. Reading
    # distribution metadata here would make `memo doctor` disagree with
    # `memo --version` in the very checkout being verified.
    from memo import __version__ as runtime_version

    config_ok, config_runtime, profile_current, config_detail = _probe_agent_configuration(
        adapter=adapter,
        slug=slug,
        root=root,
        runtime=runtime,
        probe=probe,
        binary=binary,
    )

    smoke_ok: bool | None = None
    version_match: bool | None = None
    smoke_detail = "not probed"
    if probe and memo_cli is not None and memo_cli.is_file():
        smoke_ok, version_match, smoke_detail = _isolated_runtime_smoke(
            memo_cli,
            expected_version=runtime_version,
        )

    storage_writable = _probe_storage_writable(probe=probe)

    checks = {
        "detected": binary is not None,
        "mcp_configured": config_ok,
        "mcp_runtime_current": config_runtime,
        "runtime_isolated": runtime_ok,
        "runtime_pair": runtime_pair,
        "runtime_version": runtime_version,
        "runtime_version_match": version_match,
        "runtime_smoke": smoke_ok,
        "storage_writable": storage_writable,
        "profile": adapter.mcp_profile,
        "profile_current": profile_current,
        "protocol_mode": adapter.protocol_mode,
        "instruction_marker": marker,
        "protocol_current": protocol_current,
        "instruction_writable": writable,
    }
    required = [
        checks["detected"],
        checks["runtime_isolated"],
        checks["runtime_pair"],
        marker,
        protocol_current,
        writable,
    ]
    if config_ok is not None:
        required.extend([config_ok, config_runtime, profile_current])
    if smoke_ok is not None:
        required.extend([smoke_ok, version_match])
    if storage_writable is not None:
        required.append(storage_writable)
    return {
        "agent": slug,
        "ok": all(bool(value) for value in required),
        "checks": checks,
        "config_detail": config_detail,
        "smoke_detail": smoke_detail,
        "instruction_path": str(instruction),
        "runtime": str(runtime) if runtime else None,
        "restart_guidance": adapter.restart_guidance,
    }


def os_access_writable(path: Path) -> bool:
    """Small seam kept separate so verification stays easy to isolate in tests."""
    import os

    return os.access(path, os.W_OK)
