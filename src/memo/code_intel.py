"""code_intel — shared read-only engine joining codegraph with memories.

Single home for the refs↔nodes verification semantics and the graph↔memory
joins consumed by recall, dream, briefing, ask-gaps, and the code-* CLI
commands. Three invariants, enforced HERE so consumers never re-implement
them:

- **Read-only.** The codegraph index is always opened ``mode=ro`` — this
  module never writes it.
- **Fail-open.** No error escapes to a consumer: a missing/broken index or a
  malformed ref degrades to the feature being absent (None / empty result),
  never to an exception on a hot path.
- **repo_id gating is central.** A ref minted against another repo's graph
  (``codegraph://<repo_id>/…``) is unverifiable against this DB — never dead.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Collection
from pathlib import Path
from typing import Any

from memo import codegraph_loader
from memo.code_traceability import codegraph_repo_id, parse_codegraph_uri

# Undirected expansion depth cap: 2 hops already covers caller-of-caller /
# callee-of-callee; deeper walks explode into whole-module neighborhoods.
_MAX_HOPS = 2


def open_graph(db_path: Path | None = None) -> tuple[sqlite3.Connection, str] | None:
    """(read-only connection, db repo_id) or None when the DB is missing/unopenable.

    The DB resolves via :func:`codegraph_loader._resolve_db` (explicit path >
    cwd discovery > ``MEMO_CODEGRAPH_DB`` > checkout default). The repo_id is
    the id of the repo the index lives in (``<repo_root>/.codegraph/…``) —
    the value :func:`ref_status` gates foreign refs against. The caller closes
    the connection.
    """
    db = codegraph_loader._resolve_db(db_path)
    if not db.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    return conn, codegraph_repo_id(db.parent.parent)


def ref_repo_claim(ref: Any) -> str:
    """The repo a ref claims to belong to; '' means "no claim" (verifiable here).

    Single extraction shared by every consumer that gates on repo_id: the
    explicit ``repo_id`` field wins when present (an empty value IS a claim of
    "none"), else the host of the ref's ``codegraph://`` uri — the only claim
    the refs minted by ``memo code-facts`` carry — else ''.
    """
    if not isinstance(ref, dict):
        return ""
    if "repo_id" in ref:
        return str(ref.get("repo_id") or "").strip()
    uri = str(ref.get("uri") or "").strip()
    if uri:
        parsed = parse_codegraph_uri(uri)
        if parsed is not None:
            return parsed[0]
    return ""


def ref_status(graph: sqlite3.Connection, ref: Any, db_repo_id: str) -> str | None:
    """'vigente' | 'desaparecido' | None (unverifiable) for one code ref.

    None when: ref is not a dict, has no file_path, was minted against another
    repo, or the query fails — an unverifiable ref must never count as dead.
    The ref's repo claim comes from :func:`ref_repo_claim`.

    Match semantics (the ONE implementation recall and dream delegate to):
    ``kind == 'file'`` or no symbol → the file_path exists in the nodes index;
    a symbol ref (label falling back to qualified_name) additionally requires
    a node whose ``name`` matches the symbol OR whose ``qualified_name``
    matches (qualified or symbol). 'vigente' is never asserted without a
    positive SELECT against the live index.
    """
    if graph is None or not isinstance(ref, dict):
        return None
    ref_repo_id = ref_repo_claim(ref)
    if ref_repo_id and ref_repo_id != db_repo_id:
        return None
    file_path = str(ref.get("file_path") or "").strip()
    if not file_path:
        return None
    kind = str(ref.get("kind") or "").strip().lower()
    label = str(ref.get("label") or "").strip()
    qualified = str(ref.get("qualified_name") or "").strip()
    symbol = label or qualified
    try:
        if kind == "file" or not symbol:
            row = graph.execute(
                "SELECT 1 FROM nodes WHERE file_path = ? LIMIT 1", (file_path,)
            ).fetchone()
        else:
            row = graph.execute(
                "SELECT 1 FROM nodes WHERE file_path = ? AND (name = ? OR qualified_name = ?) "
                "LIMIT 1",
                (file_path, symbol, qualified or symbol),
            ).fetchone()
    except sqlite3.Error:
        return None
    return "vigente" if row else "desaparecido"


def _ref_field(field: str) -> str:
    """SQL fragment extracting one field from a code_refs entry, NULL-safe.

    ``json_each`` yields scalar rows for malformed entries (a string where an
    object belongs); the CASE guard keeps ``json_extract`` off non-objects,
    where it would raise instead of matching nothing.
    """
    return f"CASE WHEN ref.type = 'object' THEN json_extract(ref.value, '$.{field}') END"


def memories_citing(
    store_conn: Any, *, paths: Collection[str] = (), symbols: Collection[str] = (), limit: int = 50
) -> list[dict[str, Any]]:
    """[{'id','title','refs'}] of non-reference memories citing those paths/symbols.

    A memory matches when any ``extra.code_refs`` entry has an exact
    ``file_path`` in ``paths`` or a ``label``/``qualified_name`` in
    ``symbols``. Pure JSON1 (``json_each`` over ``meta.extra_json``) — no MLX,
    no embeddings; corrupt ``extra_json`` rows are skipped via ``json_valid``.
    [] when no criteria are given or on any sqlite error.
    """
    wanted_paths = [str(p) for p in paths if p]
    wanted_symbols = [str(s) for s in symbols if s]
    if not wanted_paths and not wanted_symbols:
        return []
    clauses: list[str] = []
    params: list[Any] = []
    if wanted_paths:
        marks = ", ".join("?" * len(wanted_paths))
        clauses.append(f"{_ref_field('file_path')} IN ({marks})")
        params.extend(wanted_paths)
    if wanted_symbols:
        marks = ", ".join("?" * len(wanted_symbols))
        clauses.append(f"{_ref_field('label')} IN ({marks})")
        params.extend(wanted_symbols)
        clauses.append(f"{_ref_field('qualified_name')} IN ({marks})")
        params.extend(wanted_symbols)
    sql = (
        "SELECT DISTINCT m.id, m.title, m.extra_json "  # noqa: S608 — placeholders only
        "FROM meta m, json_each(m.extra_json, '$.code_refs') ref "
        "WHERE m.type != 'reference' AND json_valid(m.extra_json) "
        f"AND ({' OR '.join(clauses)}) LIMIT ?"
    )
    params.append(int(limit))
    try:
        rows = store_conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            extra = json.loads(row[2] or "{}")
        except (TypeError, json.JSONDecodeError):
            extra = {}
        refs = extra.get("code_refs")
        out.append(
            {
                "id": str(row[0]),
                "title": str(row[1] or ""),
                "refs": refs if isinstance(refs, list) else [],
            }
        )
    return out


def symbols_for_files(graph: sqlite3.Connection, files: Collection[str]) -> set[str]:
    """Names of nodes defined in those files (``kind='file'`` nodes excluded)."""
    wanted = [str(f) for f in files if f]
    if not wanted:
        return set()
    marks = ", ".join("?" * len(wanted))
    try:
        rows = graph.execute(
            f"SELECT name FROM nodes WHERE kind != 'file' AND file_path IN ({marks})",  # noqa: S608 — placeholders only
            wanted,
        ).fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[0]) for row in rows if row[0]}


def neighbors(graph: sqlite3.Connection, symbols: set[str], hops: int = 1) -> set[str]:
    """Undirected expansion of ``symbols`` over the traversable edge kinds.

    Traverses edges whose kind is in :data:`codegraph_loader._EDGE_KINDS`
    (calls/instantiates/extends/decorates — 'contains'/'imports' are
    structural noise). The seed is always included; ``hops`` is clamped to
    ``_MAX_HOPS``. On any sqlite error the expansion so far is returned.
    """
    result = {str(s) for s in symbols if s}
    frontier = set(result)
    kind_marks = ", ".join("?" * len(codegraph_loader._EDGE_KINDS))
    for _ in range(min(int(hops), _MAX_HOPS)):
        if not frontier:
            break
        marks = ", ".join("?" * len(frontier))
        sql = (
            "SELECT s.name, t.name FROM edges e "  # noqa: S608 — placeholders only
            "JOIN nodes s ON e.source = s.id "
            "JOIN nodes t ON e.target = t.id "
            f"WHERE e.kind IN ({kind_marks}) "
            f"AND (s.name IN ({marks}) OR t.name IN ({marks}))"
        )
        seed = sorted(frontier)
        try:
            rows = graph.execute(sql, (*codegraph_loader._EDGE_KINDS, *seed, *seed)).fetchall()
        except sqlite3.Error:
            return result
        reached = {str(name) for row in rows for name in row if name}
        frontier = reached - result
        result |= frontier
    return result


__all__ = [
    "memories_citing",
    "neighbors",
    "open_graph",
    "ref_repo_claim",
    "ref_status",
    "symbols_for_files",
]
