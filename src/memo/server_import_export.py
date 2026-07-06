"""MCP tools — import/export domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import WRITE, WRITE_IDEMPOTENT, annotated_tool

# Allowed base dirs for import/export file paths. The LLM can only read/write
# within these directories — path traversal to /etc/shadow is blocked.
_ALLOWED_BASE_DIRS: tuple[Path, ...] = (
    Path.cwd(),
    Path.home() / "Downloads",
    Path.home() / "Desktop",
    Path.home() / "Documents",
)


def _resolve_safe_path(raw: str, purpose: str) -> Path:
    """Resolve ``raw`` to an absolute :class:`Path`, rejecting traversal
    outside allowed base directories. Raises ``ValueError`` with a clear
    message when the path is unsafe."""
    p = Path(raw).expanduser().resolve(strict=False)
    allowed = any(
        p == base or _is_subdir(p, base) for base in _ALLOWED_BASE_DIRS
    )
    if not allowed:
        raise ValueError(
            f"Unsafe {purpose} path: {raw}. "
            f"Must be under one of: {', '.join(str(d) for d in _ALLOWED_BASE_DIRS)}."
        )
    if p.is_dir():
        raise ValueError(f"{purpose} path must be a file, not a directory: {raw}")
    if purpose == "import" and not p.exists():
        raise ValueError(f"Import file does not exist: {raw}")
    return p


def _is_subdir(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **WRITE)
    def memo_import_json(
        input_path: str,
    ) -> dict[str, Any]:
        """Import memories from JSON file.

        Imports memories from a JSON file, creating new entries
        for each item in the file.

        Args:
            input_path: Path to JSON file (under current dir, Downloads,
                Desktop, or Documents).
        """
        safe = _resolve_safe_path(input_path, "import")
        result = memory.import_export.import_from(safe, "json")
        return result.__dict__

    @annotated_tool(server, **WRITE)
    def memo_import_csv(
        input_path: str,
    ) -> dict[str, Any]:
        """Import memories from CSV file.

        Imports memories from a CSV file, creating new entries
        for each row in the file.

        Args:
            input_path: Path to CSV file (under current dir, Downloads,
                Desktop, or Documents).
        """
        safe = _resolve_safe_path(input_path, "import")
        result = memory.import_export.import_from(safe, "csv")
        return result.__dict__

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_export_json(
        output_path: str,
    ) -> dict[str, Any]:
        """Export memories to JSON file.

        Exports all memories to a JSON file with complete metadata.

        Args:
            output_path: Path to write JSON file (under current dir, Downloads,
                Desktop, or Documents).
        """
        safe = _resolve_safe_path(output_path, "export")
        result = memory.import_export.export_to(safe, "json")
        return result.__dict__

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_export_csv(
        output_path: str,
    ) -> dict[str, Any]:
        """Export memories to CSV file.

        Exports all memories to a CSV file with columns for
        id, title, body, tags, type, created, updated.

        Args:
            output_path: Path to write CSV file (under current dir, Downloads,
                Desktop, or Documents).
        """
        safe = _resolve_safe_path(output_path, "export")
        result = memory.import_export.export_to(safe, "csv")
        return result.__dict__

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_export_markdown_bundle(
        output_path: str,
    ) -> dict[str, Any]:
        """Export memories to Markdown bundle (zip).

        Exports all memories to a zip file containing individual
        .md files with frontmatter metadata.

        Args:
            output_path: Path to write zip file (under current dir, Downloads,
                Desktop, or Documents).
        """
        safe = _resolve_safe_path(output_path, "export")
        result = memory.import_export.export_to(safe, "markdown_bundle")
        return result.__dict__
