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
    """Imports memorias from various formats.

    Args:
        memory: The Memory instance to import into.
    """

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def import_json(self, input_path: Path) -> ImportResult:
        """Import memorias from JSON.

        Args:
            input_path: Path to JSON file.

        Returns:
            ImportResult with statistics.
        """
        data = json.loads(input_path.read_text(encoding="utf-8"))
        memorias = data if isinstance(data, list) else data.get("memorias", [])

        imported = 0
        skipped = 0
        errors = []

        for item in memorias:
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
        """Import memorias from CSV.

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
        """Import memorias from a Markdown bundle (zip).

        Args:
            input_path: Path to zip file.

        Returns:
            ImportResult with statistics.
        """
        imported = 0
        skipped = 0
        errors = []

        with zipfile.ZipFile(input_path, "r") as zf:
            for filename in zf.namelist():
                if not filename.endswith(".md"):
                    continue

                try:
                    content = zf.read(filename).decode("utf-8")
                    title = Path(filename).stem

                    # Extract frontmatter if present
                    if content.startswith("---"):
                        parts = content.split("---", maxsplit=2)
                        body = parts[2].strip() if len(parts) >= 3 else content
                    else:
                        body = content

                    self.memory.save(
                        content=body,
                        title=title,
                        tags=[],
                        type_="note",
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
    """Exports memorias to various formats.

    Args:
        memory: The Memory instance to export from.
    """

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def export_json(self, output_path: Path) -> ExportResult:
        """Export memorias to JSON.

        Args:
            output_path: Path to write JSON file.

        Returns:
            ExportResult with statistics.
        """
        memorias = self.memory.list(limit=10000)

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
            for m in memorias
        ]

        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        return ExportResult(
            exported_count=len(memorias),
            output_path=str(output_path),
            format="json",
        )

    def export_csv(self, output_path: Path) -> ExportResult:
        """Export memorias to CSV.

        Args:
            output_path: Path to write CSV file.

        Returns:
            ExportResult with statistics.
        """
        memorias = self.memory.list(limit=10000)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "title", "body", "tags", "type", "created", "updated"])

            for m in memorias:
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
            exported_count=len(memorias),
            output_path=str(output_path),
            format="csv",
        )

    def export_markdown_bundle(self, output_path: Path) -> ExportResult:
        """Export memorias to a Markdown bundle (zip).

        Args:
            output_path: Path to write zip file.

        Returns:
            ExportResult with statistics.
        """
        memorias = self.memory.list(limit=10000)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for m in memorias:
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
            exported_count=len(memorias),
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
