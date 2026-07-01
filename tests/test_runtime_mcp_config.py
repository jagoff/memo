from __future__ import annotations

from pathlib import Path

from memo.runtime.mcp_config import (
    KNOWN_MCP_CONFIGS,
    classify_command_path,
    extract_memo_command_paths,
    repair_mcp_configs,
    scan_mcp_configs,
)


def _shim(tmp_path: Path, *names: str) -> str:
    """Create a fake stable shim dir with the given bin files; return its path."""
    shim = tmp_path / "shim"
    shim.mkdir(exist_ok=True)
    for name in names:
        (shim / name).write_text("#!/bin/sh\n", encoding="utf-8")
    return str(shim)


def test_extract_finds_memo_and_memo_mcp_paths() -> None:
    text = '{"command": "/Users/x/.local/bin/memo-mcp", "env": "/Users/x/.local/bin/memo"}'
    assert extract_memo_command_paths(text) == [
        "/Users/x/.local/bin/memo",
        "/Users/x/.local/bin/memo-mcp",
    ]


def test_classify_venv_internal() -> None:
    # The path string itself signals a venv-internal location.
    assert (
        classify_command_path("/Users/x/.local/pipx/venvs/mlx-memo/bin/memo-mcp") == "venv-internal"
    )


def test_classify_missing(tmp_path: Path) -> None:
    assert classify_command_path(str(tmp_path / "does-not-exist" / "memo-mcp")) == "missing"


def test_classify_ok_for_existing_shim(tmp_path: Path) -> None:
    binp = tmp_path / "memo-mcp"
    binp.write_text("#!/bin/sh\n", encoding="utf-8")
    assert classify_command_path(str(binp)) is None


def test_scan_reports_dead_path(tmp_path: Path) -> None:
    cfg = tmp_path / "claude.json"
    cfg.write_text(
        '{"command": "/Users/x/.local/pipx/venvs/mlx-memo/bin/memo-mcp"}', encoding="utf-8"
    )
    findings = scan_mcp_configs((str(cfg),))
    assert len(findings) == 1
    assert findings[0]["issue"] == "venv-internal"
    assert findings[0]["suggestion"].endswith("/memo-mcp")


def test_scan_reports_bare_memo_mcp_launch_command(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('[mcp_servers.memo]\ncommand = "memo-mcp"\n', encoding="utf-8")

    findings = scan_mcp_configs((str(cfg),), shim_dir=str(tmp_path / "shim"))

    assert findings == [
        {
            "config": str(cfg),
            "command": "memo-mcp",
            "issue": "path-ambiguous",
            "suggestion": str(tmp_path / "shim" / "memo-mcp"),
        }
    ]


def test_scan_reports_bare_memo_mcp_command_array(tmp_path: Path) -> None:
    cfg = tmp_path / "opencode.jsonc"
    cfg.write_text('{"mcp": {"memo": {"command": ["memo-mcp"]}}}', encoding="utf-8")

    findings = scan_mcp_configs((str(cfg),), shim_dir=str(tmp_path / "shim"))

    assert findings == [
        {
            "config": str(cfg),
            "command": "memo-mcp",
            "issue": "path-ambiguous",
            "suggestion": str(tmp_path / "shim" / "memo-mcp"),
        }
    ]


def test_scan_ignores_shell_hook_commands_with_bare_memo(tmp_path: Path) -> None:
    cfg = tmp_path / "devin.json"
    cfg.write_text(
        '{"command": "MEMO_NONINTERACTIVE=1 memo recall-hook"}',
        encoding="utf-8",
    )

    assert scan_mcp_configs((str(cfg),), shim_dir=str(tmp_path / "shim")) == []


def test_classify_uv_tools_internal() -> None:
    assert (
        classify_command_path("/Users/x/.local/share/uv/tools/mlx-memo/bin/memo-mcp")
        == "venv-internal"
    )


def test_scan_skips_missing_config(tmp_path: Path) -> None:
    assert scan_mcp_configs((str(tmp_path / "nope.json"),)) == []


def test_known_configs_cover_codex_and_devin_desktop() -> None:
    joined = "\n".join(KNOWN_MCP_CONFIGS)

    assert "~/.codex/config.toml" in joined
    assert ".devin/mcp.json" in joined


def test_repair_repoints_dead_path_and_backs_up(tmp_path: Path) -> None:
    shim = _shim(tmp_path, "memo-mcp")
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'command = "/Users/x/.local/pipx/venvs/mlx-memo/bin/memo-mcp"\n', encoding="utf-8"
    )

    repairs = repair_mcp_configs((str(cfg),), shim_dir=shim, apply=True)

    assert len(repairs) == 1
    assert repairs[0]["status"] == "repaired"
    # File now points at the stable shim; backup preserves the original.
    assert cfg.read_text(encoding="utf-8") == f'command = "{shim}/memo-mcp"\n'
    assert (tmp_path / "config.toml.bak").read_text(encoding="utf-8") == (
        'command = "/Users/x/.local/pipx/venvs/mlx-memo/bin/memo-mcp"\n'
    )


def test_repair_repoints_bare_memo_mcp_launch_command(tmp_path: Path) -> None:
    shim = _shim(tmp_path, "memo-mcp")
    cfg = tmp_path / "config.toml"
    cfg.write_text('[mcp_servers.memo]\ncommand = "memo-mcp"\n', encoding="utf-8")

    repairs = repair_mcp_configs((str(cfg),), shim_dir=shim, apply=True)

    assert len(repairs) == 1
    assert repairs[0]["status"] == "repaired"
    assert cfg.read_text(encoding="utf-8") == (
        f'[mcp_servers.memo]\ncommand = "{shim}/memo-mcp"\n'
    )


def test_repair_repoints_bare_memo_mcp_command_array(tmp_path: Path) -> None:
    shim = _shim(tmp_path, "memo-mcp")
    cfg = tmp_path / "opencode.jsonc"
    cfg.write_text('{"mcp": {"memo": {"command": ["memo-mcp"]}}}', encoding="utf-8")

    repairs = repair_mcp_configs((str(cfg),), shim_dir=shim, apply=True)

    assert len(repairs) == 1
    assert repairs[0]["status"] == "repaired"
    assert cfg.read_text(encoding="utf-8") == (
        f'{{"mcp": {{"memo": {{"command": ["{shim}/memo-mcp"]}}}}}}'
    )


def test_repair_skips_when_shim_target_missing(tmp_path: Path) -> None:
    # Shim dir exists but has no memo-mcp binary — nothing safe to repoint to.
    shim = _shim(tmp_path)
    cfg = tmp_path / "config.toml"
    original = 'command = "/Users/x/.local/pipx/venvs/mlx-memo/bin/memo-mcp"\n'
    cfg.write_text(original, encoding="utf-8")

    repairs = repair_mcp_configs((str(cfg),), shim_dir=shim, apply=True)

    assert repairs[0]["status"] == "skipped-no-target"
    assert cfg.read_text(encoding="utf-8") == original  # untouched
    assert not (tmp_path / "config.toml.bak").exists()  # no backup written


def test_repair_dry_run_does_not_write(tmp_path: Path) -> None:
    shim = _shim(tmp_path, "memo-mcp")
    cfg = tmp_path / "config.toml"
    original = 'command = "/Users/x/.local/pipx/venvs/mlx-memo/bin/memo-mcp"\n'
    cfg.write_text(original, encoding="utf-8")

    repairs = repair_mcp_configs((str(cfg),), shim_dir=shim, apply=False)

    assert repairs[0]["status"] == "would-repair"
    assert cfg.read_text(encoding="utf-8") == original  # not mutated
    assert not (tmp_path / "config.toml.bak").exists()


def test_repair_is_idempotent(tmp_path: Path) -> None:
    shim = _shim(tmp_path, "memo-mcp")
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'command = "/Users/x/.local/pipx/venvs/mlx-memo/bin/memo-mcp"\n', encoding="utf-8"
    )

    repair_mcp_configs((str(cfg),), shim_dir=shim, apply=True)
    second = repair_mcp_configs((str(cfg),), shim_dir=shim, apply=True)

    assert second == []  # already stable, nothing left to do


def test_repair_does_not_clobber_memo_when_repointing_memo_mcp(tmp_path: Path) -> None:
    # /…/bin/memo is a prefix of /…/bin/memo-mcp — a naive substring replace
    # would corrupt memo-mcp. Both must repoint cleanly to their own shim.
    shim = _shim(tmp_path, "memo", "memo-mcp")
    cfg = tmp_path / "mcp_config.json"
    cfg.write_text(
        '{"cmd": "/p/pipx/venvs/mlx-memo/bin/memo-mcp", "bin": "/p/pipx/venvs/mlx-memo/bin/memo"}',
        encoding="utf-8",
    )

    repair_mcp_configs((str(cfg),), shim_dir=shim, apply=True)

    text = cfg.read_text(encoding="utf-8")
    assert f'"cmd": "{shim}/memo-mcp"' in text
    assert f'"bin": "{shim}/memo"' in text
    assert "pipx" not in text  # no dead path survives
    assert f"{shim}/memo-mcp-mcp" not in text  # not mangled


def test_known_configs_cover_new_agents() -> None:
    from memo.runtime.mcp_config import KNOWN_MCP_CONFIGS

    joined = " ".join(KNOWN_MCP_CONFIGS)
    for fragment in (
        ".codeium/windsurf/mcp_config.json",
        ".kiro/settings/mcp.json",
        ".warp/.mcp.json",
        ".continue/mcpServers/memo.yaml",
        "goose/config.yaml",
    ):
        assert fragment in joined
