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
import os
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

_EXPORT_PAGE_SIZE = 10_000


def _generator_string() -> str:
    """``memo/<version>`` for the passport header, or ``memo`` if unknown."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"memo/{version('mlx-memo')}"
    except PackageNotFoundError:
        return "memo"


@contextmanager
def _atomic_output_path(output_path: Path) -> Iterator[Path]:
    """Yield a sibling temporary path and publish it only after a clean write."""
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        yield tmp_path
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


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

    def import_records(self, records: list[Any]) -> ImportResult:
        """Save a list of normalized record dicts
        ({content|body, title, tags, type, created}). Shared by import_json
        and the Mem0/Zep migrators."""
        imported = 0
        skipped = 0
        errors = []
        for item in records:
            if not isinstance(item, dict):
                skipped += 1
                continue
            try:
                self.memory.save(
                    content=item.get("content") or item.get("body") or "",
                    title=item.get("title", ""),
                    tags=item.get("tags", []),
                    type_=item.get("type", "note"),
                    created=item.get("created"),
                    extra=item.get("extra") or None,
                )
                imported += 1
            except Exception as e:
                errors.append(f"Failed to import {item.get('id', 'unknown')}: {e}")
                skipped += 1
        return ImportResult(imported_count=imported, skipped_count=skipped, errors=errors)

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

        return self.import_records(memories)

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

    def import_passport(self, input_path: Path) -> ImportResult:
        """Import a versioned ``memo.passport.v1`` file (validated).

        High-fidelity: content/title/type/tags/created and the provenance +
        verification ``extra`` bag round-trip. Ids/embeddings/relations are
        rebuilt by this store (derived data). Raises ValidationError on a
        malformed / wrong-schema passport before touching the store.
        """
        from memo.passport import normalize_for_import, validate_passport

        obj = json.loads(input_path.read_text(encoding="utf-8"))
        validate_passport(obj)
        records = [normalize_for_import(e) for e in obj["memories"]]
        return self.import_records(records)

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

    def _list_all(self) -> list[Any]:
        """Expand the list window until the complete corpus is represented."""
        total_hint: int | None = None
        store = getattr(self.memory, "store", None)
        count = getattr(store, "count", None)
        if callable(count):
            total_hint = max(0, int(count()))

        requested = _EXPORT_PAGE_SIZE
        while True:
            memories = self.memory.list(limit=requested)
            if total_hint is not None:
                if requested >= total_hint:
                    return memories
            elif len(memories) < requested:
                return memories
            requested *= 2

    def export_json(self, output_path: Path) -> ExportResult:
        """Export memories to JSON.

        Args:
            output_path: Path to write JSON file.

        Returns:
            ExportResult with statistics.
        """
        memories = self._list_all()

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

        with _atomic_output_path(output_path) as tmp_path:
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        return ExportResult(
            exported_count=len(memories),
            output_path=str(output_path),
            format="json",
        )

    def export_passport(self, output_path: Path) -> ExportResult:
        """Export a versioned, vendor-neutral ``memo.passport.v1`` file.

        Higher fidelity than ``export_json``: a stable schema header (so a
        receiver can validate) plus the ``extra`` bag (provenance +
        verification state). Embeddings/relations are omitted — derived data,
        rebuilt on import. See ``passport.py`` for the fidelity contract.
        """
        from datetime import datetime

        from memo.passport import build_passport, entry_from_record

        memories = self._list_all()
        entries = [entry_from_record(m) for m in memories]
        obj = build_passport(
            entries,
            generator=_generator_string(),
            exported_at=datetime.now(UTC).isoformat(),
        )

        with _atomic_output_path(output_path) as tmp_path:
            tmp_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

        return ExportResult(
            exported_count=len(memories),
            output_path=str(output_path),
            format="passport",
        )

    def export_csv(self, output_path: Path) -> ExportResult:
        """Export memories to CSV.

        Args:
            output_path: Path to write CSV file.

        Returns:
            ExportResult with statistics.
        """
        memories = self._list_all()

        with (
            _atomic_output_path(output_path) as tmp_path,
            open(tmp_path, "w", newline="", encoding="utf-8") as f,
        ):
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
        memories = self._list_all()

        with (
            _atomic_output_path(output_path) as tmp_path,
            zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf,
        ):
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
        if format == "passport" or input_path.suffix == ".passport":
            return self.importer.import_passport(input_path)
        elif format == "json" or input_path.suffix == ".json":
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
        if format == "passport":
            return self.exporter.export_passport(output_path)
        elif format == "json":
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
