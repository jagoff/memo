"""The adapter matrix must fail on drift, not just pass on a clean tree.

Each test mutates exactly one surface of a synthetic repo and asserts the
matching check flips to fail — a gate that only ever passes proves nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_adapter_matrix():
    """Load scripts/adapter_matrix.py the way tests/test_dev_audit.py loads its
    script: by path, since scripts/ is not an installed package."""
    script = ROOT / "scripts" / "adapter_matrix.py"
    assert script.is_file(), "scripts/adapter_matrix.py must exist"
    spec = importlib.util.spec_from_file_location("memo_adapter_matrix", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_matrix = _load_adapter_matrix()
main = _matrix.main
run = _matrix.run

VERSION = "1.2.3"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal repo where every adapter-matrix check passes."""
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "mlx-memo"\nversion = "{VERSION}"\n\n[tool.ruff]\nversion = "ignored"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n## [{VERSION}] - 2026-01-01\n",
        encoding="utf-8",
    )
    _write_json(tmp_path / ".claude-plugin" / "plugin.json", {"name": "memo", "version": VERSION})
    _write_json(
        tmp_path / ".claude-plugin" / "marketplace.json",
        {"name": "memo", "plugins": [{"name": "memo", "source": "./"}]},
    )
    _write_json(
        tmp_path / "plugins" / "memo" / ".codex-plugin" / "plugin.json",
        {"name": "memo", "version": VERSION, "mcpServers": "./.mcp.json"},
    )
    _write_json(
        tmp_path / "plugins" / "memo" / ".mcp.json",
        {"mcpServers": {"memo": {"command": "memo-mcp", "env": {"MEMO_NONINTERACTIVE": "1"}}}},
    )
    _write_json(
        tmp_path / "server.json",
        {"name": "io.github.jagoff/memo", "version": VERSION, "packages": [{"version": VERSION}]},
    )
    _write_json(
        tmp_path / ".mcp.json",
        {
            "mcpServers": {
                "memo": {
                    "command": "memo-mcp",
                    "env": {
                        "MEMO_EMBEDDER_MODEL": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
                        "MEMO_EMBEDDER_DIMS": "2560",
                    },
                }
            }
        },
    )
    _write_json(
        tmp_path / "hooks" / "hooks.json",
        {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {"type": "command", "command": "MEMO_NONINTERACTIVE=1 memo prewarm"}
                        ]
                    }
                ],
                "Stop": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "MEMO_NONINTERACTIVE=1 memo sync once --quiet",
                            }
                        ]
                    }
                ],
            }
        },
    )
    return tmp_path


def _status(root: Path, check_id: str) -> str:
    for check in run(root):
        if check.check_id == check_id:
            return "skipped" if check.skipped else ("pass" if check.ok else "fail")
    raise AssertionError(f"no check named {check_id}")


def test_clean_repo_passes_every_check(repo: Path) -> None:
    assert [c.check_id for c in run(repo) if not c.ok] == []
    assert main(["--check", "--root", str(repo)]) == 0


def test_version_drift_in_one_manifest_fails(repo: Path) -> None:
    _write_json(repo / "server.json", {"version": "9.9.9", "packages": [{"version": VERSION}]})

    assert _status(repo, "version-parity") == "fail"
    assert main(["--check", "--root", str(repo)]) == 1


def test_changelog_missing_the_released_version_fails(repo: Path) -> None:
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n", encoding="utf-8")

    assert _status(repo, "version-parity") == "fail"


def test_embedder_dims_not_matching_model_size_fails(repo: Path) -> None:
    # 4B is 2560-dim; 1024 belongs to the 0.6B model. This exact mismatch
    # corrupts the vec0 table (MLX invariant 3).
    _write_json(
        repo / ".mcp.json",
        {
            "mcpServers": {
                "memo": {
                    "env": {
                        "MEMO_EMBEDDER_MODEL": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
                        "MEMO_EMBEDDER_DIMS": "1024",
                    }
                }
            }
        },
    )

    assert _status(repo, "embedder-dims-parity") == "fail"


def test_pinned_model_without_dims_fails(repo: Path) -> None:
    _write_json(
        repo / ".mcp.json",
        {"mcpServers": {"memo": {"env": {"MEMO_EMBEDDER_MODEL": "Qwen3-Embedding-8B"}}}},
    )

    assert _status(repo, "embedder-dims-parity") == "fail"


def test_config_pinning_neither_model_nor_dims_passes(repo: Path) -> None:
    # The shipped plugin config deliberately pins no model: the installed index
    # is self-describing and adopts its own profile. That must stay legal.
    _write_json(repo / ".mcp.json", {"mcpServers": {"memo": {"env": {"MEMO_NONINTERACTIVE": "1"}}}})

    assert _status(repo, "embedder-dims-parity") == "pass"


def test_hook_firing_an_unregistered_subcommand_fails(repo: Path) -> None:
    _write_json(
        repo / "hooks" / "hooks.json",
        {
            "hooks": {
                "Stop": [
                    {
                        "hooks": [
                            {"type": "command", "command": "MEMO_NONINTERACTIVE=1 memo sync onse"}
                        ]
                    }
                ]
            }
        },
    )

    assert _status(repo, "hook-commands-resolve") == "fail"


def test_hook_firing_an_unregistered_top_level_command_fails(repo: Path) -> None:
    _write_json(
        repo / "hooks" / "hooks.json",
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "memo prewarrm"}]}]}},
    )

    assert _status(repo, "hook-commands-resolve") == "fail"


def test_hook_env_prefix_and_flags_do_not_confuse_resolution(repo: Path) -> None:
    # The real graph prefixes env assignments (including a $(tty ...) subshell
    # with spaces and quotes) and appends flags; neither is a subcommand.
    _write_json(
        repo / "hooks" / "hooks.json",
        {
            "hooks": {
                "UserPromptSubmit": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    "MEMO_NONINTERACTIVE=1 "
                                    "MEMO_AGENT_TTY=${MEMO_AGENT_TTY:-$(tty 2>/dev/null | "
                                    "grep -E '^/dev/' | head -1)} "
                                    "memo session idle-maintenance --mode capture"
                                ),
                            }
                        ]
                    }
                ]
            }
        },
    )

    assert _status(repo, "hook-commands-resolve") == "pass"


def test_codex_manifest_pointing_at_a_missing_mcp_config_fails(repo: Path) -> None:
    (repo / "plugins" / "memo" / ".mcp.json").unlink()

    assert _status(repo, "referenced-paths-exist") == "fail"


def test_marketplace_source_without_a_plugin_manifest_fails(repo: Path) -> None:
    _write_json(
        repo / ".claude-plugin" / "marketplace.json",
        {"plugins": [{"name": "memo", "source": "./does-not-exist"}]},
    )

    assert _status(repo, "referenced-paths-exist") == "fail"


def test_missing_repo_root_is_a_usage_error(tmp_path: Path) -> None:
    assert main(["--check", "--root", str(tmp_path / "nope")]) == 2
