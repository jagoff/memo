"""Tests for server_import_export MCP tool registration."""

from __future__ import annotations

import ast
import pathlib
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_server_and_tools():
    """Return a (server_mock, tools_dict) pair.

    `server.tool()` is wired so each `@server.tool()` decorated function is
    captured in `tools` by its `__name__`, without going through FastMCP.
    """
    server = MagicMock()
    tools: dict = {}

    def tool_decorator():
        def wrapper(fn):
            tools[fn.__name__] = fn
            return fn

        return wrapper

    server.tool = tool_decorator
    return server, tools


def test_register_exposes_all_import_export_tools(tmp_cfg) -> None:
    """register() must expose every expected import/export MCP tool."""
    from memo.memory import Memory
    from memo.server_import_export import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {
        "memo_import_json",
        "memo_import_csv",
        "memo_import_passport",
        "memo_export_json",
        "memo_export_csv",
        "memo_export_markdown_bundle",
        "memo_export_passport",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_import_json_calls_import_from(tmp_cfg) -> None:
    """memo_import_json must call memory.import_export.import_from(path, 'json')."""
    from memo.import_export import ImportResult
    from memo.memory import Memory
    from memo.server_import_export import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_result = ImportResult(imported_count=3, skipped_count=1, errors=[])
    mem.import_export.import_from.return_value = fake_result

    server, tools = _make_server_and_tools()
    register(server, mem)

    safe_path = Path("/fake/safe/memories.json")
    with patch("memo.server_import_export._resolve_safe_path", return_value=safe_path):
        result = tools["memo_import_json"](input_path="/fake/safe/memories.json")

    mem.import_export.import_from.assert_called_once_with(safe_path, "json")
    assert result == {"imported_count": 3, "skipped_count": 1, "errors": []}


def test_memo_import_csv_calls_import_from(tmp_cfg) -> None:
    """memo_import_csv must call memory.import_export.import_from(path, 'csv')."""
    from memo.import_export import ImportResult
    from memo.memory import Memory
    from memo.server_import_export import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_result = ImportResult(imported_count=5, skipped_count=2, errors=["row 7: bad data"])
    mem.import_export.import_from.return_value = fake_result

    server, tools = _make_server_and_tools()
    register(server, mem)

    safe_path = Path("/fake/safe/memories.csv")
    with patch("memo.server_import_export._resolve_safe_path", return_value=safe_path):
        result = tools["memo_import_csv"](input_path="/fake/safe/memories.csv")

    mem.import_export.import_from.assert_called_once_with(safe_path, "csv")
    assert result["imported_count"] == 5
    assert result["skipped_count"] == 2
    assert result["errors"] == ["row 7: bad data"]


def test_memo_export_json_calls_export_to(tmp_cfg) -> None:
    """memo_export_json must call memory.import_export.export_to(path, 'json')."""
    from memo.import_export import ExportResult
    from memo.memory import Memory
    from memo.server_import_export import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    out = "/fake/safe/out.json"
    fake_result = ExportResult(exported_count=42, output_path=out, format="json")
    mem.import_export.export_to.return_value = fake_result

    server, tools = _make_server_and_tools()
    register(server, mem)

    safe_path = Path(out)
    with patch("memo.server_import_export._resolve_safe_path", return_value=safe_path):
        result = tools["memo_export_json"](output_path=out)

    mem.import_export.export_to.assert_called_once_with(safe_path, "json")
    assert result == {"exported_count": 42, "output_path": out, "format": "json"}


def test_memo_export_csv_calls_export_to(tmp_cfg) -> None:
    """memo_export_csv must call memory.import_export.export_to(path, 'csv')."""
    from memo.import_export import ExportResult
    from memo.memory import Memory
    from memo.server_import_export import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    out = "/fake/safe/out.csv"
    fake_result = ExportResult(exported_count=10, output_path=out, format="csv")
    mem.import_export.export_to.return_value = fake_result

    server, tools = _make_server_and_tools()
    register(server, mem)

    safe_path = Path(out)
    with patch("memo.server_import_export._resolve_safe_path", return_value=safe_path):
        result = tools["memo_export_csv"](output_path=out)

    mem.import_export.export_to.assert_called_once_with(safe_path, "csv")
    assert result["exported_count"] == 10
    assert result["format"] == "csv"


def test_memo_export_markdown_bundle_calls_export_to(tmp_cfg) -> None:
    """memo_export_markdown_bundle must call export_to(path, 'markdown_bundle')."""
    from memo.import_export import ExportResult
    from memo.memory import Memory
    from memo.server_import_export import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    out = "/fake/safe/bundle.zip"
    fake_result = ExportResult(exported_count=7, output_path=out, format="markdown_bundle")
    mem.import_export.export_to.return_value = fake_result

    server, tools = _make_server_and_tools()
    register(server, mem)

    safe_path = Path(out)
    with patch("memo.server_import_export._resolve_safe_path", return_value=safe_path):
        result = tools["memo_export_markdown_bundle"](output_path=out)

    mem.import_export.export_to.assert_called_once_with(safe_path, "markdown_bundle")
    assert result["exported_count"] == 7
    assert result["format"] == "markdown_bundle"


def test_unsafe_path_is_rejected(tmp_cfg, tmp_path: Path) -> None:
    """_resolve_safe_path must reject paths outside allowed base dirs."""
    from memo.server_import_export import _resolve_safe_path

    # tmp_path is typically /private/var/folders/... — well outside allowed dirs
    fake_file = tmp_path / "secret.json"
    fake_file.write_text("[]", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="Unsafe import path"):
        _resolve_safe_path(str(fake_file), "import")


def test_resolve_safe_path_rejects_missing_import_file(tmp_cfg) -> None:
    """_resolve_safe_path with purpose='import' must raise if the file doesn't exist."""
    import pytest

    from memo.server_import_export import _ALLOWED_BASE_DIRS, _resolve_safe_path

    # Use an allowed base dir so only the existence check fails
    allowed_base = _ALLOWED_BASE_DIRS[0]
    nonexistent = allowed_base / "__memo_test_nonexistent_12345.json"

    with pytest.raises(ValueError, match="does not exist"):
        _resolve_safe_path(str(nonexistent), "import")


def test_resolve_safe_path_accepts_export_in_cwd(tmp_cfg) -> None:
    """_resolve_safe_path must accept an export path under cwd (file need not exist)."""
    from memo.server_import_export import _ALLOWED_BASE_DIRS, _resolve_safe_path

    allowed_base = _ALLOWED_BASE_DIRS[0]  # Path.cwd()
    out_path = allowed_base / "__memo_test_export_out.json"
    # Export paths don't need to exist
    resolved = _resolve_safe_path(str(out_path), "export")
    assert resolved == out_path


def test_no_module_level_mlx_imports() -> None:
    """server_import_export must not have module-level MLX imports."""
    src = pathlib.Path(__file__).parent.parent / "src" / "memo" / "server_import_export.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    violations = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mlx"):
                    violations.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mlx"):
            violations.append(f"line {node.lineno}: from {node.module} import ...")

    assert not violations, f"Module-level MLX imports found: {violations}"
