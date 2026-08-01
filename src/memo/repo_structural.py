"""Read-only CodeGraph provider for repository search.

The provider never parses source itself.  It consumes a repository-local
CodeGraph SQLite index when one exists and returns positive path evidence;
absence is reported as ``unavailable`` rather than treated as proof that no
symbol or relationship exists.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from memo.repo_index_search import _extract_query_terms, path_in_repo_scope


def search_codegraph_paths(
    clone_path: Path,
    query: str,
    *,
    scope: str = "all",
    limit: int = 40,
) -> dict[str, Any]:
    db_path = Path(clone_path) / ".codegraph" / "codegraph.db"
    if not db_path.is_file():
        return {
            "provider": "codegraph",
            "status": "unavailable",
            "reason": "index_missing",
            "db_path": str(db_path),
            "paths": [],
        }
    terms = _extract_query_terms(query)[:8]
    if not terms:
        return {
            "provider": "codegraph",
            "status": "available",
            "reason": "no_structural_terms",
            "db_path": str(db_path),
            "paths": [],
        }

    try:
        connection = sqlite3.connect(
            f"file:{db_path}?mode=ro&immutable=1",
            uri=True,
            timeout=1.0,
        )
        connection.row_factory = sqlite3.Row
        try:
            direct = _direct_nodes(connection, terms, limit=max(limit * 3, 60))
            neighbors = _neighbor_nodes(
                connection,
                [str(row["id"]) for row in direct],
                limit=max(limit * 4, 80),
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return {
            "provider": "codegraph",
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "db_path": str(db_path),
            "paths": [],
        }

    path_scores: dict[str, float] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    direct_ids = {str(row["id"]) for row in direct}
    for row in direct:
        path = _clean_path(str(row["file_path"]))
        if not path or not path_in_repo_scope(path, scope):
            continue
        score = _node_score(row, terms)
        path_scores[path] = max(path_scores.get(path, 0.0), score)
        evidence.setdefault(path, []).append(
            {
                "kind": "symbol_match",
                "symbol_id": str(row["id"]),
                "symbol": str(row["qualified_name"] or row["name"]),
                "node_kind": str(row["kind"]),
                "line_start": int(row["start_line"]),
                "line_end": int(row["end_line"]),
                "score": round(score, 6),
            }
        )

    for row in neighbors:
        path = _clean_path(str(row["file_path"]))
        if not path or not path_in_repo_scope(path, scope):
            continue
        source = str(row["source"])
        target = str(row["target"])
        matched_id = source if source in direct_ids else target
        score = 0.42
        if str(row["kind"]) in {"calls", "imports", "extends", "implements"}:
            score += 0.08
        path_scores[path] = max(path_scores.get(path, 0.0), score)
        evidence.setdefault(path, []).append(
            {
                "kind": "one_hop_relationship",
                "edge_kind": str(row["kind"]),
                "matched_symbol_id": matched_id,
                "symbol_id": str(row["node_id"]),
                "symbol": str(row["qualified_name"] or row["name"]),
                "line_start": int(row["start_line"]),
                "line_end": int(row["end_line"]),
                "score": round(score, 6),
            }
        )

    ranked = sorted(path_scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return {
        "provider": "codegraph",
        "status": "available",
        "reason": "",
        "db_path": str(db_path),
        "paths": [
            {
                "path": path,
                "score": score,
                "evidence": sorted(
                    evidence.get(path, []),
                    key=lambda item: (-float(item["score"]), str(item.get("symbol") or "")),
                )[:8],
            }
            for path, score in ranked
        ],
    }


def _direct_nodes(
    connection: sqlite3.Connection,
    terms: list[str],
    *,
    limit: int,
) -> list[sqlite3.Row]:
    # Current CodeGraph indexes expose a segment vocabulary for snake_case and
    # CamelCase identifiers. Prefix lookups use its primary key and the nodes
    # name index; this avoids one full nodes-table scan per query term. Keep the
    # LIKE fallback for older/minimal provider schemas.
    has_segment_vocab = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'name_segment_vocab'"
    ).fetchone()
    if has_segment_vocab is not None:
        segment_predicates = ["vocab.segment GLOB ?" for _term in terms]
        segment_params: list[Any] = [f"{term.lower()}*" for term in terms]
        sql = (
            "WITH matching_names AS ("  # noqa: S608 — fixed placeholders; bound terms.
            "  SELECT vocab.name, COUNT(DISTINCT vocab.segment) AS match_count "
            "  FROM name_segment_vocab AS vocab WHERE "
            + " OR ".join(segment_predicates)
            + "  GROUP BY vocab.name"
            ") "
            "SELECT nodes.id, nodes.kind, nodes.name, nodes.qualified_name, "
            "       nodes.file_path, nodes.start_line, nodes.end_line, "
            "       nodes.signature, nodes.is_exported, matching_names.match_count "
            "FROM matching_names "
            "JOIN nodes ON nodes.name = matching_names.name "
            "ORDER BY matching_names.match_count DESC, nodes.is_exported DESC, "
            "         length(nodes.name) ASC, "
            "         nodes.file_path ASC LIMIT ?"
        )
        segment_params.append(limit)
        return connection.execute(sql, segment_params).fetchall()

    predicates: list[str] = []
    params: list[Any] = []
    for term in terms:
        pattern = f"%{term.lower()}%"
        predicates.append(
            "(lower(name) LIKE ? OR lower(qualified_name) LIKE ? "
            "OR lower(COALESCE(signature, '')) LIKE ?)"
        )
        params.extend((pattern, pattern, pattern))
    sql = (
        "SELECT id, kind, name, qualified_name, file_path, start_line, end_line, "
        "       signature, is_exported "
        "FROM nodes WHERE "
        + " OR ".join(predicates)
        + " ORDER BY is_exported DESC, length(name) ASC, file_path ASC LIMIT ?"
    )
    params.append(limit)
    return connection.execute(sql, params).fetchall()


def _neighbor_nodes(
    connection: sqlite3.Connection,
    node_ids: list[str],
    *,
    limit: int,
) -> list[sqlite3.Row]:
    if not node_ids:
        return []
    placeholders = ",".join("?" for _ in node_ids)
    sql = (
        "SELECT edges.source, edges.target, edges.kind, "
        "       nodes.id AS node_id, nodes.kind AS node_kind, nodes.name, "
        "       nodes.qualified_name, nodes.file_path, nodes.start_line, nodes.end_line "
        "FROM edges "
        "JOIN nodes ON nodes.id = CASE "
        "  WHEN edges.source IN (" + placeholders + ") THEN edges.target ELSE edges.source END "
        "WHERE edges.source IN (" + placeholders + ") OR edges.target IN (" + placeholders + ") "
        "ORDER BY edges.kind, nodes.file_path LIMIT ?"
    )
    params = [*node_ids, *node_ids, *node_ids, limit]
    return connection.execute(sql, params).fetchall()


def _node_score(row: sqlite3.Row, terms: list[str]) -> float:
    name = str(row["name"] or "").lower()
    qualified = str(row["qualified_name"] or "").lower()
    score = 0.55
    for term in terms:
        if name == term:
            score = max(score, 1.0)
        elif name.startswith(term):
            score = max(score, 0.88)
        elif term in name:
            score = max(score, 0.76)
        elif term in qualified:
            score = max(score, 0.64)
    if "match_count" in row.keys():  # noqa: SIM118 — sqlite3.Row checks values.
        score = max(score, 0.55 + min(0.4, 0.12 * int(row["match_count"] or 0)))
    if int(row["is_exported"] or 0):
        score += 0.04
    return min(score, 1.0)


def _clean_path(path: str) -> str:
    clean = path.replace("\\", "/")
    while clean.startswith("./"):
        clean = clean[2:]
    return clean.lstrip("/")


__all__ = ["search_codegraph_paths"]
