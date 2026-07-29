"""`memo code-facts` — mine architectural facts from the codegraph index.

Reads the codegraph SQLite DB (read-only, ``mode=ro``) and distills durable
architectural facts — call hubs, the API/CLI surface, and cross-package
dependencies — into ``type=fact`` memories tagged ``codegraph-derived``.
Each memory carries ``code_refs`` provenance in the exact shape
``code_traceability._explicit_references`` parses, plus a short provenance
hash used to skip facts that are already saved on re-runs.

Dry-run by default; ``--apply`` writes. Registered onto the root group in
cli.py via ``cli.add_command(code_facts_cmd)``.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.code_traceability import codegraph_repo_id, codegraph_uri
from memo.config import Config

# Cross-package dependency pairs reported (top 10 by edge count).
_DEP_PAIR_LIMIT = 10
# Representative symbols attached as code_refs on aggregate facts.
_REFS_PER_FACT = 3
# Memories scanned when collecting already-saved provenance hashes.
_EXISTING_SCAN_LIMIT = 10000

# Decorator fragments that mark a function as an exposed route/command.
_SURFACE_DECORATOR_PATTERNS = ("%route%", "%command%", "%app.%")

# Every mining query anchors file_path to src/ — architectural facts must
# never be mined from tests/ (e.g. tests/test_cli_*.py matches '%cli_%.py').
_SRC_ONLY = "file_path LIKE 'src/%'"

_NODE_COLUMNS = "id, kind, name, qualified_name, file_path, start_line, end_line"


@dataclass(frozen=True)
class CodeFact:
    """One mined architectural fact, ready to persist as a memory."""

    category: str  # "call-hub" | "api-surface" | "package-dependency"
    text: str
    code_refs: list[dict[str, Any]]

    @property
    def provenance_hash(self) -> str:
        """Short stable hash of fact text + referenced paths (dedup key)."""
        paths = sorted({str(ref.get("file_path") or "") for ref in self.code_refs})
        payload = "\n".join([self.text, *paths])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _ref(repo_id: str, row: sqlite3.Row) -> dict[str, Any]:
    """One ``code_refs`` entry in the shape ``_explicit_references`` parses."""
    label = str(row["name"] or row["file_path"] or row["id"])
    return {
        "uri": codegraph_uri(repo_id, str(row["id"])),
        "label": label,
        "kind": str(row["kind"] or "symbol"),
        "qualified_name": str(row["qualified_name"] or label),
        "file_path": str(row["file_path"] or ""),
        "start_line": int(row["start_line"]) if row["start_line"] is not None else None,
        "end_line": int(row["end_line"]) if row["end_line"] is not None else None,
    }


def _node_ref(conn: sqlite3.Connection, repo_id: str, node_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        f"SELECT {_NODE_COLUMNS} FROM nodes WHERE id = ?",  # noqa: S608
        (node_id,),
    ).fetchone()
    return None if row is None else _ref(repo_id, row)


def _node_columns(conn: sqlite3.Connection) -> set[str]:
    return {str(row[1]) for row in conn.execute("PRAGMA table_info(nodes)")}


def _mine_call_hubs(conn: sqlite3.Connection, repo_id: str, top: int) -> list[CodeFact]:
    """Top-N src/ symbols by in-degree of ``calls`` edges."""
    rows = conn.execute(
        "SELECT t.id, t.kind, t.name, t.qualified_name, t.file_path, "  # noqa: S608
        "t.start_line, t.end_line, COUNT(*) AS n "
        "FROM edges e JOIN nodes t ON e.target = t.id "
        f"WHERE e.kind = 'calls' AND t.{_SRC_ONLY} "
        "GROUP BY t.id ORDER BY n DESC, t.qualified_name LIMIT ?",
        (top,),
    ).fetchall()
    facts: list[CodeFact] = []
    for row in rows:
        qualified = str(row["qualified_name"] or row["name"])
        text = f"Code hub: {qualified} ({row['file_path']}) receives {row['n']} call edges"
        facts.append(CodeFact("call-hub", text, [_ref(repo_id, row)]))
    return facts


def _mine_api_surface(conn: sqlite3.Connection, repo_id: str, top: int) -> list[CodeFact]:
    """Decorated route/command functions; falls back to src/ cli_*.py counts."""
    facts: list[CodeFact] = []
    if "decorators" in _node_columns(conn):
        clause = " OR ".join("decorators LIKE ?" for _ in _SURFACE_DECORATOR_PATTERNS)
        rows = conn.execute(
            f"SELECT {_NODE_COLUMNS} FROM nodes "  # noqa: S608
            f"WHERE kind = 'function' AND {_SRC_ONLY} AND decorators IS NOT NULL "
            f"AND TRIM(decorators) != '' AND ({clause}) "
            "ORDER BY file_path, start_line LIMIT ?",
            (*_SURFACE_DECORATOR_PATTERNS, top),
        ).fetchall()
        for row in rows:
            qualified = str(row["qualified_name"] or row["name"])
            text = (
                f"API/CLI surface: {qualified} ({row['file_path']}) "
                "is exposed via a route/command decorator"
            )
            facts.append(CodeFact("api-surface", text, [_ref(repo_id, row)]))
        if facts:
            return facts

    # Fallback: count top-level symbols per cli_*.py module. A symbol is
    # top-level when no other function/class in the same file encloses it
    # by line range.
    rows = conn.execute(
        f"SELECT {_NODE_COLUMNS} FROM nodes n "  # noqa: S608
        "WHERE n.kind IN ('function', 'class') "
        f"AND n.{_SRC_ONLY} "
        "AND n.file_path LIKE '%cli\\_%.py' ESCAPE '\\' "
        "AND NOT EXISTS ("
        "    SELECT 1 FROM nodes p"
        "    WHERE p.file_path = n.file_path AND p.id <> n.id"
        "      AND p.kind IN ('function', 'class')"
        "      AND p.start_line IS NOT NULL AND p.end_line IS NOT NULL"
        "      AND n.start_line IS NOT NULL"
        "      AND p.start_line < n.start_line"
        "      AND p.end_line >= COALESCE(n.end_line, n.start_line)"
        ") ORDER BY n.file_path, n.start_line"
    ).fetchall()
    per_file: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        per_file.setdefault(str(row["file_path"]), []).append(row)
    ranked = sorted(per_file.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]
    for file_path, members in ranked:
        text = f"CLI surface: {file_path} defines {len(members)} top-level symbols"
        refs = [_ref(repo_id, row) for row in members[:_REFS_PER_FACT]]
        facts.append(CodeFact("api-surface", text, refs))
    return facts


def _mine_package_dependencies(conn: sqlite3.Connection, repo_id: str) -> list[CodeFact]:
    """Cross-directory (source dir → target dir) edge counts, top src/ pairs."""
    rows = conn.execute(
        "SELECT s.id AS source_id, t.id AS target_id, "  # noqa: S608
        "s.file_path AS source_path, t.file_path AS target_path "
        "FROM edges e "
        "JOIN nodes s ON e.source = s.id "
        "JOIN nodes t ON e.target = t.id "
        f"WHERE s.{_SRC_ONLY} AND t.{_SRC_ONLY}"
    ).fetchall()
    counts: dict[tuple[str, str], int] = {}
    samples: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows:
        src_dir = posixpath.dirname(str(row["source_path"]))
        tgt_dir = posixpath.dirname(str(row["target_path"]))
        if not src_dir or not tgt_dir or src_dir == tgt_dir:
            continue
        pair = (src_dir, tgt_dir)
        counts[pair] = counts.get(pair, 0) + 1
        samples.setdefault(pair, (str(row["source_id"]), str(row["target_id"])))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:_DEP_PAIR_LIMIT]
    facts: list[CodeFact] = []
    for (src_dir, tgt_dir), n in ranked:
        text = f"Package dependency: {src_dir} -> {tgt_dir} ({n} edges)"
        refs = [
            ref
            for node_id in samples[(src_dir, tgt_dir)]
            if (ref := _node_ref(conn, repo_id, node_id)) is not None
        ]
        facts.append(CodeFact("package-dependency", text, refs))
    return facts


def mine_code_facts(conn: sqlite3.Connection, *, repo_id: str, top: int) -> list[CodeFact]:
    """Mine all fact categories from an open read-only codegraph connection."""
    return [
        *_mine_call_hubs(conn, repo_id, top),
        *_mine_api_surface(conn, repo_id, top),
        *_mine_package_dependencies(conn, repo_id),
    ]


def _existing_hashes(mem: Any) -> set[str]:
    """Provenance hashes of codegraph-derived fact memories already saved."""
    hashes: set[str] = set()
    for rec in mem.list(type_="fact", limit=_EXISTING_SCAN_LIMIT):
        if "codegraph-derived" not in (rec.tags or []):
            continue
        value = (rec.extra or {}).get("provenance_hash")
        if isinstance(value, str) and value:
            hashes.add(value)
    return hashes


def _mining_target(
    db_override: Path | None, project: Path | None
) -> tuple[Path, str, tuple[str, ...]]:
    """Resolve (db, repo_id, tags) for one mining run; --project switches all three.

    Same DB default as the loader: explicit --db > project-aware discovery
    (nearest .codegraph/codegraph.db above --project when given, else above
    cwd) > memo's own checkout. --project never falls through to the env
    override — mining the wrong repo under a project:<basename> tag is worse
    than failing.

    The project tag always derives from the mined repo (--project basename,
    else the resolved DB's repo root ``<root>/.codegraph/codegraph.db``) —
    never a hardcoded name: a run whose cwd discovery lands in ANOTHER indexed
    repo must tag that repo, or recall's project boost surfaces its facts
    under the wrong project.
    """
    from memo import codegraph_loader

    if project is not None and db_override is None:
        discovered = codegraph_loader._discover_db(start=project)
        db = discovered if discovered is not None else project / ".codegraph" / "codegraph.db"
    else:
        db = codegraph_loader._resolve_db(db_override)
    if not db.is_file():
        raise click.ClickException(f"codegraph index not found at {db}")

    if project is not None:
        tags = (f"project:{project.resolve().name}", "codegraph-derived")
        return db, codegraph_repo_id(project), tags
    repo_root = db.resolve().parent.parent
    return db, codegraph_repo_id(repo_root), (f"project:{repo_root.name}", "codegraph-derived")


@click.command(name="code-facts")
@click.option(
    "--apply",
    "apply_",
    is_flag=True,
    help="Save the mined facts as memories (default: dry-run).",
)
@click.option(
    "--db",
    "db_override",
    type=click.Path(path_type=Path),
    default=None,
    help="Codegraph DB path (default: the loader's .codegraph/codegraph.db).",
)
@click.option(
    "--project",
    "project",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Mine another indexed repo: discover its .codegraph DB from this path "
    "and tag the facts project:<basename>.",
)
@click.option("--top", default=10, show_default=True, help="Max facts per category.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def code_facts_cmd(
    apply_: bool, db_override: Path | None, project: Path | None, top: int, as_json: bool
) -> None:
    """Mine architectural facts from the codegraph index into memories.

    Read-only over the codegraph DB: call hubs (most-called symbols), the
    API/CLI surface, and cross-package dependencies. Dry-run by default —
    pass --apply to save each fact as a `type=fact` memory. Re-runs skip
    facts whose provenance hash is already saved. --project PATH mines any
    other indexed repo: the DB is discovered from that path and the facts
    are tagged `project:<basename>`, with refs minted under that repo's
    codegraph:// repo id.

    Example: memo code-facts --top 5 --apply
    """
    db, repo_id, tags = _mining_target(db_override, project)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        facts = mine_code_facts(conn, repo_id=repo_id, top=top)
    except sqlite3.Error as exc:
        raise click.ClickException(f"codegraph query failed: {exc}") from exc
    finally:
        conn.close()

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    existing = _existing_hashes(mem)

    results: list[dict[str, Any]] = []
    saved = skipped = 0
    for fact in facts:
        phash = fact.provenance_hash
        if phash in existing:
            status = "skipped"
            skipped += 1
        elif apply_:
            mem.save(
                content=fact.text,
                title=fact.text[:80],
                type_="fact",
                tags=list(tags),
                extra={
                    "code_refs": fact.code_refs,
                    "provenance_hash": phash,
                    "code_facts_category": fact.category,
                },
            )
            existing.add(phash)
            status = "saved"
            saved += 1
        else:
            status = "new"
        results.append(
            {
                "category": fact.category,
                "text": fact.text,
                "provenance_hash": phash,
                "status": status,
                "code_refs": fact.code_refs,
            }
        )

    if as_json:
        click.echo(
            json.dumps(
                {
                    "dry_run": not apply_,
                    "db": str(db),
                    "facts": results,
                    "saved": saved,
                    "skipped": skipped,
                },
                indent=2,
            )
        )
        return

    if not results:
        console.print("[dim]No architectural facts mined (empty codegraph index?)[/dim]")
        return

    table = Table(title="Code facts")
    table.add_column("Category", style="yellow")
    table.add_column("Fact", style="cyan")
    table.add_column("Status", style="green")
    for item in results:
        table.add_row(item["category"], item["text"], item["status"])
    console.print(table)

    if apply_:
        console.print(f"[green]{saved} saved[/green], [dim]{skipped} skipped (already known)[/dim]")
    else:
        new = sum(1 for item in results if item["status"] == "new")
        console.print(
            f"[yellow](dry-run)[/yellow] {new} new, {skipped} skipped — pass --apply to save"
        )
