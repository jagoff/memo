"""Read-only diagnostics for persistence privacy and identity invariants."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from memo.config import Config
from memo.identity import namespace_for_index
from memo.redact import scan_secrets


def _empty_report() -> dict[str, Any]:
    return {
        "ok": False,
        "identity_constraint": "unavailable",
        "multiple_project_tag_rows": 0,
        "topic_collision_groups": 0,
        "exact_duplicate_groups": 0,
        "legacy_identity_rows": 0,
        "secret_pattern_files": 0,
        "private_marker_files": 0,
    }


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
            (name,),
        ).fetchone()
        is not None
    )


def _active_clause(columns: set[str]) -> str:
    return "WHERE deleted_at IS NULL OR deleted_at = ''" if "deleted_at" in columns else ""


def _read_identity_diagnostics(db_path: Path) -> dict[str, Any]:
    report = _empty_report()
    if not db_path.is_file():
        return report

    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        with closing(connection):
            connection.execute("PRAGMA query_only=ON")
            if not _table_exists(connection, "meta"):
                return report
            columns = {
                str(row["name"]) for row in connection.execute("PRAGMA table_info(meta)").fetchall()
            }
            active = _active_clause(columns)
            # ``active`` is selected from two internal SQL literals based only
            # on PRAGMA schema metadata; it never contains caller input.
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM meta {active}"  # noqa: S608
                ).fetchone()[0]
            )
            required = {
                "namespace",
                "topic_key",
                "normalized_title",
                "normalized_content_hash",
            }
            if not required.issubset(columns):
                report["legacy_identity_rows"] = total
                return report

            capability = None
            if _table_exists(connection, "schema_meta"):
                capability = connection.execute(
                    "SELECT value FROM schema_meta WHERE key = 'identity_topic_unique'"
                ).fetchone()
            index_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' "
                "AND name='idx_meta_active_topic_unique'"
            ).fetchone()
            status = str(capability["value"]) if capability is not None else "unavailable"
            if status == "enabled" and index_exists is None:
                status = "unavailable"
            report["identity_constraint"] = status

            active_and = (
                "AND (deleted_at IS NULL OR deleted_at = '')" if "deleted_at" in columns else ""
            )
            report["topic_collision_groups"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM (SELECT 1 FROM meta "  # noqa: S608
                    "WHERE namespace IS NOT NULL AND topic_key IS NOT NULL "
                    f"{active_and} GROUP BY namespace, topic_key HAVING COUNT(*) > 1)"
                ).fetchone()[0]
            )
            report["exact_duplicate_groups"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM (SELECT 1 FROM meta "  # noqa: S608
                    "WHERE namespace IS NOT NULL AND normalized_title IS NOT NULL "
                    f"AND normalized_content_hash IS NOT NULL {active_and} "
                    "GROUP BY namespace, type, normalized_title, normalized_content_hash "
                    "HAVING COUNT(*) > 1)"
                ).fetchone()[0]
            )
            report["legacy_identity_rows"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM meta WHERE (namespace IS NULL "  # noqa: S608
                    "OR normalized_title IS NULL OR normalized_content_hash IS NULL) " + active_and
                ).fetchone()[0]
            )

            ambiguous = 0
            for row in connection.execute(
                "SELECT path, tags FROM meta " + active  # noqa: S608
            ).fetchall():
                try:
                    raw_tags = json.loads(str(row["tags"] or "[]"))
                    tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
                except (TypeError, ValueError, json.JSONDecodeError):
                    tags = []
                if namespace_for_index(tags, path=str(row["path"] or "")) is None:
                    ambiguous += 1
            report["multiple_project_tag_rows"] = ambiguous
    except (OSError, sqlite3.Error):
        return report
    return report


def _read_markdown_privacy(memory_dir: Path) -> tuple[int, int]:
    secret_files = 0
    private_files = 0
    if not memory_dir.is_dir():
        return secret_files, private_files
    for path in memory_dir.rglob("*.md"):
        try:
            if path.is_symlink():
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if scan_secrets(text, entropy=False):
            secret_files += 1
        if "<private>" in text.casefold():
            private_files += 1
    return secret_files, private_files


def trust_preflight(cfg: Config) -> dict[str, Any]:
    """Return sanitized counts only; never mutate or expose matched content."""
    report = _read_identity_diagnostics(cfg.db_path)
    secret_files, private_files = _read_markdown_privacy(cfg.memory_dir)
    report["secret_pattern_files"] = secret_files
    report["private_marker_files"] = private_files
    report["ok"] = bool(
        report["identity_constraint"] == "enabled"
        and not any(
            int(report[key])
            for key in (
                "multiple_project_tag_rows",
                "topic_collision_groups",
                "exact_duplicate_groups",
                "legacy_identity_rows",
                "secret_pattern_files",
                "private_marker_files",
            )
        )
    )
    return report
