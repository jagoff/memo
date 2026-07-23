from __future__ import annotations

import subprocess
from importlib.metadata import version
from pathlib import Path

from click.testing import CliRunner

from memo.cli_setup import setup_cmd
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
