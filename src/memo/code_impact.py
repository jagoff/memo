"""Bounded change-impact traversal over a local CodeGraph index."""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from memo.code_evidence import codegraph_evidence, normalize_code_path
from memo.code_traceability import codegraph_repo_id

_IMPACT_EDGE_KINDS = ("calls", "instantiates", "extends", "decorates")
_MAX_SEED_SYMBOLS = 500
_MAX_IMPACT_SYMBOLS = 1_000


def _git_root(cwd: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return cwd.resolve()
    value = result.stdout.strip()
    return Path(value) if result.returncode == 0 and value else cwd.resolve()


def _chunks(values: list[str], size: int = 400) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _edge_rows(
    conn: sqlite3.Connection,
    frontier: set[str],
) -> list[sqlite3.Row]:
    out: list[sqlite3.Row] = []
    for batch in _chunks(sorted(frontier)):
        placeholders = ",".join("?" for _ in batch)
        kinds = ",".join("?" for _ in _IMPACT_EDGE_KINDS)
        out.extend(
            conn.execute(
                "SELECT source, target, kind FROM edges "  # noqa: S608
                f"WHERE kind IN ({kinds}) "
                f"AND (source IN ({placeholders}) OR target IN ({placeholders}))",
                (*_IMPACT_EDGE_KINDS, *batch, *batch),
            ).fetchall()
        )
    return out


def code_change_impact(
    cwd: str | Path,
    changed_files: list[str] | tuple[str, ...],
    *,
    depth: int = 1,
) -> dict[str, Any]:
    """Trace changed files to directly connected symbols, with hard bounds."""
    repo_root = _git_root(Path(cwd))
    db_path = repo_root / ".codegraph" / "codegraph.db"
    paths = tuple(sorted({normalize_code_path(path) for path in changed_files if path}))
    repo_id = codegraph_repo_id(repo_root)
    envelope = codegraph_evidence(
        db_path=db_path,
        repo_root=repo_root,
        repo_id=repo_id,
        paths=paths,
    ).to_dict()
    if not db_path.is_file():
        return {
            "available": False,
            "reason": "codegraph_missing",
            "repo_root": str(repo_root),
            "changed_files": list(paths),
            "symbols": [],
            "impacted_paths": list(paths),
            "code_evidence": envelope,
            "limitations": ["Change impact requires a local .codegraph/codegraph.db index."],
        }
    if not paths:
        return {
            "available": True,
            "reason": "no_changes",
            "repo_root": str(repo_root),
            "changed_files": [],
            "symbols": [],
            "impacted_paths": [],
            "code_evidence": envelope,
            "limitations": [],
        }

    bounded_depth = max(0, min(int(depth), 3))
    limitations: list[str] = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" for _ in paths)
            seed_query = (
                "SELECT id, kind, name, qualified_name, file_path, "  # noqa: S608
                "start_line, end_line "
                f"FROM nodes WHERE file_path IN ({placeholders}) "
                "ORDER BY file_path, start_line, id"
            )
            seed_rows = conn.execute(
                seed_query,
                paths,
            ).fetchall()
            if len(seed_rows) > _MAX_SEED_SYMBOLS:
                seed_rows = seed_rows[:_MAX_SEED_SYMBOLS]
                limitations.append(
                    f"Seed symbols were capped at {_MAX_SEED_SYMBOLS} for bounded traversal."
                )
            distances = {str(row["id"]): 0 for row in seed_rows}
            via: dict[str, str] = {str(row["id"]): "changed_file" for row in seed_rows}
            frontier = set(distances)
            for distance in range(1, bounded_depth + 1):
                if not frontier or len(distances) >= _MAX_IMPACT_SYMBOLS:
                    break
                next_frontier: set[str] = set()
                for edge in _edge_rows(conn, frontier):
                    source = str(edge["source"])
                    target = str(edge["target"])
                    neighbor = target if source in frontier else source
                    if neighbor in distances:
                        continue
                    distances[neighbor] = distance
                    via[neighbor] = str(edge["kind"])
                    next_frontier.add(neighbor)
                    if len(distances) >= _MAX_IMPACT_SYMBOLS:
                        limitations.append(
                            f"Impacted symbols were capped at {_MAX_IMPACT_SYMBOLS}."
                        )
                        break
                frontier = next_frontier

            node_rows: list[sqlite3.Row] = []
            for batch in _chunks(sorted(distances)):
                ids = ",".join("?" for _ in batch)
                node_query = (
                    "SELECT id, kind, name, qualified_name, file_path, "  # noqa: S608
                    f"start_line, end_line FROM nodes WHERE id IN ({ids})"
                )
                node_rows.extend(
                    conn.execute(
                        node_query,
                        batch,
                    ).fetchall()
                )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {
            "available": False,
            "reason": "codegraph_unreadable",
            "repo_root": str(repo_root),
            "changed_files": list(paths),
            "symbols": [],
            "impacted_paths": list(paths),
            "code_evidence": envelope,
            "limitations": [f"CodeGraph traversal failed: {type(exc).__name__}."],
        }

    symbols: list[dict[str, Any]] = [
        {
            "stable_symbol_id": str(row["id"]),
            "kind": str(row["kind"]),
            "name": str(row["name"]),
            "qualified_name": str(row["qualified_name"]),
            "file_path": str(row["file_path"]),
            "start_line": int(row["start_line"]),
            "end_line": int(row["end_line"]),
            "distance": distances[str(row["id"])],
            "via": via[str(row["id"])],
        }
        for row in node_rows
    ]
    symbols.sort(
        key=lambda item: (
            int(item["distance"]),
            str(item["file_path"]),
            int(item["start_line"]),
            str(item["stable_symbol_id"]),
        )
    )
    impacted_paths = sorted({*paths, *(str(symbol["file_path"]) for symbol in symbols)})
    return {
        "available": True,
        "reason": None,
        "repo_root": str(repo_root),
        "changed_files": list(paths),
        "depth": bounded_depth,
        "symbols": symbols,
        "impacted_paths": impacted_paths,
        "code_evidence": envelope,
        "limitations": limitations,
    }


__all__ = ["code_change_impact"]
