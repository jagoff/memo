"""MCP tools — import/export domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory


def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
    def memo_import_json(
        input_path: str,
    ) -> dict[str, Any]:
        """Import memories from JSON file.

        Imports memories from a JSON file, creating new entries
        for each item in the file.

        Args:
            input_path: Path to JSON file.
        """
        from pathlib import Path

        result = memory.import_export.import_from(Path(input_path), "json")
        return result.__dict__

    @server.tool()
    def memo_import_csv(
        input_path: str,
    ) -> dict[str, Any]:
        """Import memories from CSV file.

        Imports memories from a CSV file, creating new entries
        for each row in the file.

        Args:
            input_path: Path to CSV file.
        """
        from pathlib import Path

        result = memory.import_export.import_from(Path(input_path), "csv")
        return result.__dict__

    @server.tool()
    def memo_export_json(
        output_path: str,
    ) -> dict[str, Any]:
        """Export memories to JSON file.

        Exports all memories to a JSON file with complete metadata.

        Args:
            output_path: Path to write JSON file.
        """
        from pathlib import Path

        result = memory.import_export.export_to(Path(output_path), "json")
        return result.__dict__

    @server.tool()
    def memo_export_csv(
        output_path: str,
    ) -> dict[str, Any]:
        """Export memories to CSV file.

        Exports all memories to a CSV file with columns for
        id, title, body, tags, type, created, updated.

        Args:
            output_path: Path to write CSV file.
        """
        from pathlib import Path

        result = memory.import_export.export_to(Path(output_path), "csv")
        return result.__dict__

    @server.tool()
    def memo_export_markdown_bundle(
        output_path: str,
    ) -> dict[str, Any]:
        """Export memories to Markdown bundle (zip).

        Exports all memories to a zip file containing individual
        .md files with frontmatter metadata.

        Args:
            output_path: Path to write zip file.
        """
        from pathlib import Path

        result = memory.import_export.export_to(Path(output_path), "markdown_bundle")
        return result.__dict__
