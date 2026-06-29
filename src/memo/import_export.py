"""Memory import/export — interoperability with other systems.

Enables:
- Import from other formats (Notion, Obsidian export, Roam)
- Export to JSON, CSV, Markdown bundle
- Import from CSV/TSV
- Migration from other note systems
- Export with complete metadata
"""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImportResult:
    """Result of an import operation."""

    imported_count: int
    skipped_count: int
    errors: list[str]


@dataclass
class ExportResult:
    """Result of an export operation."""

    exported_count: int
    output_path: str
    format: str


class Importer:
    """Imports memories from various formats.

    Args:
        memory: The Memory instance to import into.
    """

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def import_json(self, input_path: Path) -> ImportResult:
        """Import memories from JSON.

        Args:
            input_path: Path to JSON file.

        Returns:
            ImportResult with statistics.
        """
        data = json.loads(input_path.read_text(encoding="utf-8"))
        raw_memories = (
            data
            if isinstance(data, list)
            else (data.get("memories", data.get("memorias", [])) if isinstance(data, dict) else [])
        )
        memories: list[Any] = raw_memories if isinstance(raw_memories, list) else []

        imported = 0
        skipped = 0
        errors = []

        for item in memories:
            if not isinstance(item, dict):
                skipped += 1
                continue
            try:
                self.memory.save(
                    content=item.get("content") or item.get("body") or "",
                    title=item.get("title", ""),
                    tags=item.get("tags", []),
                    type_=item.get("type", "note"),
                )
                imported += 1
            except Exception as e:
                errors.append(f"Failed to import {item.get('id', 'unknown')}: {e}")
                skipped += 1

        return ImportResult(
            imported_count=imported,
            skipped_count=skipped,
            errors=errors,
        )

    def import_csv(self, input_path: Path) -> ImportResult:
        """Import memories from CSV.

        Args:
            input_path: Path to CSV file.

        Returns:
            ImportResult with statistics.
        """
        imported = 0
        skipped = 0
        errors = []

        with open(input_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    tags = row.get("tags", "").split(",") if row.get("tags") else []
                    self.memory.save(
                        content=row.get("content") or row.get("body") or "",
                        title=row.get("title", ""),
                        tags=[t.strip() for t in tags if t.strip()],
                        type_=row.get("type", "note"),
                    )
                    imported += 1
                except Exception as e:
                    errors.append(f"Failed to import row: {e}")
                    skipped += 1

        return ImportResult(
            imported_count=imported,
            skipped_count=skipped,
            errors=errors,
        )

    def import_markdown_bundle(self, input_path: Path) -> ImportResult:
        """Import memories from a Markdown bundle (zip).

        Args:
            input_path: Path to zip file.

        Returns:
            ImportResult with statistics.
        """
        import frontmatter

        imported = 0
        skipped = 0
        errors = []

        with zipfile.ZipFile(input_path, "r") as zf:
            for filename in zf.namelist():
                if not filename.endswith(".md"):
                    continue

                try:
                    content = zf.read(filename).decode("utf-8")
                    post = frontmatter.loads(content)
                    meta = post.metadata
                    body = post.content.strip()
                    title = str(meta.get("title") or Path(filename).stem)

                    # Export writes tags comma-joined, so YAML parses them as a
                    # plain string; a hand-edited bundle may use a YAML list.
                    raw_tags = meta.get("tags")
                    if isinstance(raw_tags, list):
                        tags = [str(t).strip() for t in raw_tags if str(t).strip()]
                    elif isinstance(raw_tags, str):
                        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
                    else:
                        tags = []

                    self.memory.save(
                        content=body,
                        title=title,
                        tags=tags,
                        type_=str(meta.get("type") or "note"),
                    )
                    imported += 1
                except Exception as e:
                    errors.append(f"Failed to import {filename}: {e}")
                    skipped += 1

        return ImportResult(
            imported_count=imported,
            skipped_count=skipped,
            errors=errors,
        )


class Exporter:
    """Exports memories to various formats.

    Args:
        memory: The Memory instance to export from.
    """

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def export_json(self, output_path: Path) -> ExportResult:
        """Export memories to JSON.

        Args:
            output_path: Path to write JSON file.

        Returns:
            ExportResult with statistics.
        """
        memories = self.memory.list(limit=10000)

        data = [
            {
                "id": m.id,
                "title": m.title,
                "body": m.body,
                "tags": m.tags,
                "type": m.type,
                "created": m.created,
                "updated": m.updated,
            }
            for m in memories
        ]

        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        return ExportResult(
            exported_count=len(memories),
            output_path=str(output_path),
            format="json",
        )

    def export_csv(self, output_path: Path) -> ExportResult:
        """Export memories to CSV.

        Args:
            output_path: Path to write CSV file.

        Returns:
            ExportResult with statistics.
        """
        memories = self.memory.list(limit=10000)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "title", "body", "tags", "type", "created", "updated"])

            for m in memories:
                writer.writerow(
                    [
                        m.id,
                        m.title,
                        m.body or "",
                        ",".join(m.tags),
                        m.type,
                        m.created,
                        m.updated,
                    ]
                )

        return ExportResult(
            exported_count=len(memories),
            output_path=str(output_path),
            format="csv",
        )

    def export_markdown_bundle(self, output_path: Path) -> ExportResult:
        """Export memories to a Markdown bundle (zip).

        Args:
            output_path: Path to write zip file.

        Returns:
            ExportResult with statistics.
        """
        memories = self.memory.list(limit=10000)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for m in memories:
                # Create markdown content with frontmatter
                frontmatter = f"""---
title: {m.title}
tags: {", ".join(m.tags)}
type: {m.type}
created: {m.created}
updated: {m.updated}
---

{m.body or ""}
"""

                filename = f"{m.id}.md"
                zf.writestr(filename, frontmatter)

        return ExportResult(
            exported_count=len(memories),
            output_path=str(output_path),
            format="markdown_bundle",
        )


class ImportExportManager:
    """Manages import and export operations.

    Args:
        memory: The Memory instance.
    """

    def __init__(self, memory: Any) -> None:
        self.memory = memory
        self.importer = Importer(memory)
        self.exporter = Exporter(memory)

    def import_from(self, input_path: Path, format: str) -> ImportResult:
        """Import from file with auto-detected or specified format.

        Args:
            input_path: Path to input file.
            format: Format (json, csv, markdown_bundle). Falls back to the file extension.

        Returns:
            ImportResult with statistics.
        """
        if format == "json" or input_path.suffix == ".json":
            return self.importer.import_json(input_path)
        elif format == "csv" or input_path.suffix == ".csv":
            return self.importer.import_csv(input_path)
        elif format == "markdown_bundle" or input_path.suffix == ".zip":
            return self.importer.import_markdown_bundle(input_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_to(self, output_path: Path, format: str) -> ExportResult:
        """Export to file with specified format.

        Args:
            output_path: Path to output file.
            format: Format (json, csv, markdown_bundle).

        Returns:
            ExportResult with statistics.
        """
        if format == "json":
            return self.exporter.export_json(output_path)
        elif format == "csv":
            return self.exporter.export_csv(output_path)
        elif format == "markdown_bundle":
            return self.exporter.export_markdown_bundle(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")


__all__ = [
    "ExportResult",
    "Exporter",
    "ImportExportManager",
    "ImportResult",
    "Importer",
]
