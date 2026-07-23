"""Declarative setup registry for memo's first-class agent integrations.

The registry is deliberately small: it owns the end-to-end contract for the
agents memo can verify completely. Broader MCP-only compatibility remains in
``memo install-mcp`` and can migrate here adapter by adapter.
"""

from __future__ import annotations

import json
import os
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
        verification_command=("codex", "mcp", "get", "memo"),
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


def _default_runner(argv: tuple[str, ...], best_effort: bool) -> None:
    try:
        proc = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=30)
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
                remove_command=adapter.remove_command,
                instruction_path=str(instruction_path),
                instruction_present=instruction_present,
                restart_guidance=adapter.restart_guidance,
                rollback_limitations=adapter.rollback_limitations,
            )
        )
    return SetupPlan(memo_mcp=str(runtime), cwd=str(root), actions=tuple(actions))


def _backup_path(path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return path.with_name(f"{path.name}.memo-backup-{stamp}")


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
        if result["ok"]:
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

    try:
        from importlib.metadata import version

        runtime_version = version("mlx-memo")
    except Exception:  # pragma: no cover - editable/uninstalled source tree
        runtime_version = "unknown"

    config_ok: bool | None = None
    config_runtime: bool | None = None
    profile_current: bool | None = None
    config_detail = "not probed"
    if probe and binary:
        try:
            proc = subprocess.run(
                adapter.verification_command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            config_ok = proc.returncode == 0
            raw_detail = (proc.stdout + "\n" + proc.stderr).strip()
            config_detail = raw_detail[:500]
            config_runtime = bool(runtime and str(runtime) in raw_detail)
            profile_current = adapter.mcp_profile in raw_detail
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            config_ok = False
            config_runtime = False
            profile_current = False
            config_detail = str(exc)

    smoke_ok: bool | None = None
    version_match: bool | None = None
    smoke_detail = "not probed"
    if probe and memo_cli is not None and memo_cli.is_file():
        smoke_ok, version_match, smoke_detail = _isolated_runtime_smoke(
            memo_cli,
            expected_version=runtime_version,
        )

    storage_writable: bool | None = None
    if probe:
        try:
            from memo.config import Config

            cfg = Config.from_env()
            storage_writable = _path_writable(cfg.data_dir) and _path_writable(cfg.state_dir)
        except Exception:
            storage_writable = False

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
