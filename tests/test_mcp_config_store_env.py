"""`memo doctor` must see a store path in an MCP client's env, not just the binary.

``scan_mcp_configs`` checks the launch COMMAND — that it is not a venv-internal
path, not a bare name. It never looked at the ``env`` block, so on 2026-08-09,
with ``~/.claude.json`` carrying ``MEMO_DATA_DIR="sweep/store_cli_2/data"`` and
every MCP tool answering from an empty corpus, doctor printed::

    ✓ mcp config paths: stable

The command path was, in fact, stable. The store the server would open was not,
and that is the failure the operator needed named: an MCP client reading a
different corpus than the CLI is invisible from inside the CLI.

Scanning stays format-agnostic (raw text, so JSON/JSONC/TOML/YAML all work) and
report-only: unlike a command path, a wrong store directory has no safe
mechanical repair — memo cannot know which corpus was meant.
"""

from __future__ import annotations

import json
from pathlib import Path

from memo.runtime.mcp_config import scan_mcp_store_env


def _write_claude_json(tmp_path: Path, env: dict[str, str]) -> Path:
    cfg = tmp_path / "claude.json"
    cfg.write_text(
        json.dumps(
            {"mcpServers": {"memo": {"command": "/Users/x/.local/bin/memo-mcp", "env": env}}}
        ),
        encoding="utf-8",
    )
    return cfg


def test_relative_store_path_is_reported(tmp_path: Path) -> None:
    cfg = _write_claude_json(tmp_path, {"MEMO_DATA_DIR": "sweep/store_cli_2/data"})

    findings = scan_mcp_store_env((str(cfg),))

    assert len(findings) == 1
    assert findings[0]["config"] == str(cfg)
    assert findings[0]["var"] == "MEMO_DATA_DIR"
    assert findings[0]["value"] == "sweep/store_cli_2/data"
    assert findings[0]["issue"] == "relative"


def test_absolute_but_missing_store_path_is_reported(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "data"
    cfg = _write_claude_json(tmp_path, {"MEMO_STATE_DIR": str(missing)})

    findings = scan_mcp_store_env((str(cfg),))

    assert [f["issue"] for f in findings] == ["missing"]
    assert findings[0]["var"] == "MEMO_STATE_DIR"


def test_absolute_existing_store_path_is_clean(tmp_path: Path) -> None:
    store = tmp_path / "memorias"
    store.mkdir()
    cfg = _write_claude_json(tmp_path, {"MEMO_DATA_DIR": str(store)})

    assert scan_mcp_store_env((str(cfg),)) == []


def test_config_without_store_env_is_clean(tmp_path: Path) -> None:
    cfg = _write_claude_json(tmp_path, {"MEMO_MCP_PROFILE": "agent"})

    assert scan_mcp_store_env((str(cfg),)) == []


def test_both_vars_are_reported_independently(tmp_path: Path) -> None:
    cfg = _write_claude_json(
        tmp_path, {"MEMO_DATA_DIR": "sweep/data", "MEMO_STATE_DIR": "sweep/state"}
    )

    findings = scan_mcp_store_env((str(cfg),))

    assert sorted(f["var"] for f in findings) == ["MEMO_DATA_DIR", "MEMO_STATE_DIR"]


def test_toml_style_assignment_is_scanned(tmp_path: Path) -> None:
    """Codex uses TOML; the scan is deliberately format-agnostic."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[mcp_servers.memo.env]\nMEMO_DATA_DIR = "sweep/store_cli_2/data"\n', encoding="utf-8"
    )

    findings = scan_mcp_store_env((str(cfg),))

    assert [f["issue"] for f in findings] == ["relative"]


def test_missing_config_file_is_skipped(tmp_path: Path) -> None:
    assert scan_mcp_store_env((str(tmp_path / "absent.json"),)) == []
