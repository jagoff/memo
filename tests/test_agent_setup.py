from __future__ import annotations

import subprocess
from importlib.metadata import version
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo import cli_setup as cli_setup_module
from memo.cli_setup import setup_cmd
from memo.config import Config
from memo.runtime import agent_registry as registry


def _runtime(tmp_path: Path) -> Path:
    runtime = tmp_path / ".local" / "bin" / "memo-mcp"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime.with_name("memo").write_text("#!/bin/sh\n", encoding="utf-8")
    return runtime


def test_setup_dry_run_is_pure_and_declarative(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(registry.shutil, "which", lambda name: f"/usr/bin/{name}")
    plan = registry.build_setup_plan(["all"], cwd=tmp_path, memo_mcp=runtime)

    receipt = registry.apply_setup_plan(plan, dry_run=True)

    assert receipt["ok"] is True
    assert [action.agent for action in plan.actions] == ["codex", "claude-code"]
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / "CLAUDE.md").exists()
    assert plan.actions[0].mcp_profile == "core"
    assert plan.actions[1].protocol_mode == "compact"


def test_setup_uses_existing_claude_project_scope_and_removes_shadow_user(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _runtime(tmp_path)
    (tmp_path / ".mcp.json").write_text(
        '{"mcpServers":{"memo":{"command":"old-memo-mcp"}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/usr/bin/claude")

    plan = registry.build_setup_plan(["claude-code"], cwd=tmp_path, memo_mcp=runtime)
    action = plan.actions[0]

    assert action.remove_command == ("claude", "mcp", "remove", "-s", "project", "memo")
    assert action.mcp_command[4] == "project"
    assert action.shadow_remove_commands == (("claude", "mcp", "remove", "-s", "user", "memo"),)


def test_default_runner_bypasses_interactive_agent_wrappers(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def fake_run(*_args, **kwargs):
        seen.update(kwargs["env"])
        return subprocess.CompletedProcess(["claude"], 0, "", "")

    monkeypatch.setattr(registry.subprocess, "run", fake_run)

    registry._default_runner(("claude", "mcp", "list"), False)

    assert seen["MEMFLOW_CAPTURE_DISABLE"] == "1"
    assert seen["MEMFLOW_STARTUP_BANNER"] == "0"
    assert seen["MEMO_NONINTERACTIVE"] == "1"


def test_setup_preserves_unknown_text_backs_up_and_is_idempotent(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/usr/bin/codex")
    target = tmp_path / "AGENTS.md"
    target.write_text("# Existing\n\nkeep-me\n", encoding="utf-8")
    calls: list[tuple[tuple[str, ...], bool]] = []

    plan = registry.build_setup_plan(["codex"], cwd=tmp_path, memo_mcp=runtime)
    first = registry.apply_setup_plan(
        plan, runner=lambda argv, best_effort: calls.append((argv, best_effort))
    )
    second_plan = registry.build_setup_plan(["codex"], cwd=tmp_path, memo_mcp=runtime)
    second = registry.apply_setup_plan(
        second_plan, runner=lambda argv, best_effort: calls.append((argv, best_effort))
    )

    body = target.read_text(encoding="utf-8")
    assert "keep-me" in body
    assert body.count("<!-- memo-mandate -->") == 1
    assert first["results"][0]["instruction"] == "written"
    assert Path(first["results"][0]["backup"]).read_text(encoding="utf-8").startswith("# Existing")
    assert second["results"][0]["instruction"] == "already present (skip)"
    assert len(list(tmp_path.glob("AGENTS.md.memo-backup-*"))) == 1
    assert len(calls) == 4


def test_external_failure_has_exact_remediation_and_does_not_write(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/usr/bin/codex")
    plan = registry.build_setup_plan(["codex"], cwd=tmp_path, memo_mcp=runtime)

    def fail_add(argv: tuple[str, ...], best_effort: bool) -> None:
        if not best_effort:
            raise registry.SetupError("boom")

    receipt = registry.apply_setup_plan(plan, runner=fail_add)

    result = receipt["results"][0]
    assert result["status"] == "partial"
    assert result["remediation"].startswith("codex mcp add memo")
    assert not (tmp_path / "AGENTS.md").exists()


def test_instruction_failure_rolls_back_external_registration(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/usr/bin/codex")
    plan = registry.build_setup_plan(["codex"], cwd=tmp_path, memo_mcp=runtime)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        registry, "_write_mandate", lambda *_a, **_k: (_ for _ in ()).throw(OSError("ro"))
    )

    receipt = registry.apply_setup_plan(plan, runner=lambda argv, _best_effort: calls.append(argv))

    assert receipt["results"][0]["status"] == "rolled-back"
    assert calls[-1] == ("codex", "mcp", "remove", "memo")


def test_detect_filters_absent_agents(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(
        registry.shutil,
        "which",
        lambda name: "/usr/bin/codex" if name == "codex" else None,
    )

    plan = registry.build_setup_plan(["all"], cwd=tmp_path, detect=True, memo_mcp=runtime)

    assert [action.agent for action in plan.actions] == ["codex"]


def test_setup_cli_json_dry_run(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(registry, "resolve_isolated_memo_mcp", lambda: runtime)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/usr/bin/codex")

    with CliRunner().isolated_filesystem(temp_dir=tmp_path):
        result = CliRunner().invoke(setup_cmd, ["codex", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    assert '"dry_run": true' in result.output


def test_setup_cli_human_plan_reports_no_detected_agents(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(registry, "resolve_isolated_memo_mcp", lambda: runtime)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: None)

    with CliRunner().isolated_filesystem(temp_dir=tmp_path):
        result = CliRunner().invoke(setup_cmd, ["all", "--detect", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert f"memo plan → {runtime}" in result.output
    assert "no detected agents selected" in result.output


def test_setup_cli_human_failure_prints_receipt_details(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(registry, "resolve_isolated_memo_mcp", lambda: runtime)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/usr/bin/codex")
    monkeypatch.setattr(
        cli_setup_module,
        "apply_setup_plan",
        lambda _plan, *, dry_run: {
            "ok": False,
            "dry_run": dry_run,
            "results": [
                {
                    "ok": False,
                    "status": "partial",
                    "instruction": "unchanged",
                    "backup": "/tmp/AGENTS.md.bak",
                    "error": "registration failed",
                    "remediation": "run codex mcp add",
                }
            ],
        },
    )

    with CliRunner().isolated_filesystem(temp_dir=tmp_path):
        result = CliRunner().invoke(setup_cmd, ["codex"])

    assert result.exit_code == 1
    assert "✗ codex: partial" in result.output
    assert "MCP: codex mcp add memo" in result.output
    assert "backup: /tmp/AGENTS.md.bak" in result.output
    assert "error: registration failed" in result.output
    assert "remediation: run codex mcp add" in result.output
    assert registry.AGENT_REGISTRY["codex"].restart_guidance in result.output


def test_setup_cli_wraps_setup_errors(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise registry.SetupError("runtime unavailable")

    monkeypatch.setattr(cli_setup_module, "build_setup_plan", fail)

    result = CliRunner().invoke(setup_cmd, ["codex"])

    assert result.exit_code == 1
    assert "Error: runtime unavailable" in result.output


def test_agent_doctor_verifies_runtime_profile_protocol_and_isolated_smoke(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/usr/bin/codex")
    registry._write_mandate(tmp_path / "AGENTS.md", dry_run=False)
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_path / "vault"))
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()

    def fake_run(argv, **_kwargs):
        args = [str(value) for value in argv]
        if args[:4] == ["codex", "mcp", "get", "memo"]:
            stdout = f"command: {runtime}\nenv: MEMO_MCP_PROFILE=core\n"
        elif "--version" in args:
            stdout = f"memo, version {version('mlx-memo')}\n"
        elif "save" in args:
            stdout = '{"id":"smoke-id"}'
        elif "search" in args:
            stdout = '[{"id":"smoke-id"}]'
        else:
            stdout = "ok\n"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(registry.subprocess, "run", fake_run)

    report = registry.verify_agent("codex", cwd=tmp_path, memo_mcp=runtime)

    assert report["ok"] is True
    assert report["checks"]["mcp_runtime_current"] is True
    assert report["checks"]["profile_current"] is True
    assert report["checks"]["protocol_current"] is True
    assert report["checks"]["runtime_version_match"] is True
    assert report["checks"]["runtime_smoke"] is True
    assert report["checks"]["storage_writable"] is True


def test_codex_doctor_reads_unmasked_json_profile(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    registry._write_mandate(tmp_path / "AGENTS.md", dry_run=False)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/usr/bin/codex")

    def fake_run(argv, **_kwargs):
        payload = {
            "transport": {
                "command": str(runtime),
                "env": {"MEMO_MCP_PROFILE": "core"},
            }
        }
        return subprocess.CompletedProcess(argv, 0, stdout=registry.json.dumps(payload), stderr="")

    monkeypatch.setattr(registry.subprocess, "run", fake_run)
    report = registry.verify_agent("codex", cwd=tmp_path, memo_mcp=runtime)

    assert report["checks"]["mcp_runtime_current"] is True
    assert report["checks"]["profile_current"] is True


def test_verify_agent_reports_probe_and_storage_failures(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    registry._write_mandate(tmp_path / "AGENTS.md", dry_run=False)
    original_read_text = Path.read_text

    def unreadable_instruction(path: Path, *args, **kwargs):
        if path.name == "AGENTS.md":
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("memo probe", 10)

    def invalid_config(_cls):
        raise RuntimeError("invalid storage")

    monkeypatch.setattr(Path, "read_text", unreadable_instruction)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/usr/bin/codex")
    monkeypatch.setattr(registry.subprocess, "run", timeout)
    monkeypatch.setattr(Config, "from_env", classmethod(invalid_config))

    report = registry.verify_agent("codex", cwd=tmp_path, memo_mcp=runtime)

    assert report["ok"] is False
    assert report["checks"]["instruction_marker"] is False
    assert report["checks"]["mcp_configured"] is False
    assert report["checks"]["runtime_smoke"] is False
    assert report["checks"]["storage_writable"] is False
    assert "memo probe" in report["config_detail"]


def test_isolated_runtime_smoke_reports_command_failure(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path).with_name("memo")
    monkeypatch.setattr(
        registry.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, returncode=2, stdout="", stderr="broken runtime"
        ),
    )

    smoke_ok, version_ok, detail = registry._isolated_runtime_smoke(
        runtime, expected_version=version("mlx-memo")
    )

    assert smoke_ok is False
    assert version_ok is False
    assert "broken runtime" in detail


def test_resolve_isolated_runtime_uses_path_fallback(monkeypatch) -> None:
    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/usr/local/bin/memo-mcp")

    assert registry.resolve_isolated_memo_mcp() == Path("/usr/local/bin/memo-mcp")

    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/repo/.venv/bin/memo-mcp")
    assert registry.resolve_isolated_memo_mcp() is None


def test_verify_agent_rejects_unknown_agent() -> None:
    with pytest.raises(registry.SetupError, match="unsupported setup agent"):
        registry.verify_agent("unknown")
