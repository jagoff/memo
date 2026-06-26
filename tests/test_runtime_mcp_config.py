from __future__ import annotations

from pathlib import Path

from memo.runtime.mcp_config import (
    classify_command_path,
    extract_memo_command_paths,
    scan_mcp_configs,
)


def test_extract_finds_memo_and_memo_mcp_paths() -> None:
    text = '{"command": "/Users/x/.local/bin/memo-mcp", "env": "/Users/x/.local/bin/memo"}'
    assert extract_memo_command_paths(text) == [
        "/Users/x/.local/bin/memo",
        "/Users/x/.local/bin/memo-mcp",
    ]


def test_classify_venv_internal(tmp_path: Path) -> None:
    p = tmp_path / "venv-like"
    # The path string itself signals a venv-internal location.
    assert classify_command_path("/Users/x/.local/pipx/venvs/mlx-memo/bin/memo-mcp") == "venv-internal"


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


def test_scan_skips_missing_config(tmp_path: Path) -> None:
    assert scan_mcp_configs((str(tmp_path / "nope.json"),)) == []
