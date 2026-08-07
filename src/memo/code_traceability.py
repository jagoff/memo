"""Stable, evidence-backed links between captured memories and codegraph nodes."""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from memo.code_evidence import CodegraphEvidenceResolver, normalize_code_path


@dataclass(frozen=True)
class CodeReference:
    uri: str
    repo_id: str
    stable_symbol_id: str
    kind: str
    label: str
    qualified_name: str
    file_path: str
    start_line: int | None
    end_line: int | None
    relation: str
    confidence: float
    code_evidence: dict[str, Any] = field(default_factory=dict)


def _normalized_remote(remote: str) -> str:
    value = remote.strip().rstrip("/")
    if value.casefold().endswith(".git"):
        value = value[:-4]
    return value.casefold()


def _git_remote(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "config", "--get", "remote.origin.url"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _git_common_repo_root(repo_root: Path) -> Path | None:
    """Locate the main checkout that owns a linked worktree's shared Git dir."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        return None
    common_dir = Path(value)
    if not common_dir.is_absolute():
        common_dir = (repo_root / common_dir).resolve()
    return common_dir.parent if common_dir.name == ".git" else None


def codegraph_repo_id(repo_root: Path, *, remote: str | None = None) -> str:
    """Return a cross-worktree id from the Git remote, with a local fallback."""
    identity = _normalized_remote(remote or _git_remote(repo_root) or str(repo_root.resolve()))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def codegraph_uri(repo_id: str, stable_symbol_id: str) -> str:
    return (
        f"codegraph://{quote(repo_id.strip(), safe='')}/{quote(stable_symbol_id.strip(), safe='')}"
    )


def parse_codegraph_uri(uri: str) -> tuple[str, str] | None:
    parts = urlsplit(uri.strip())
    if parts.scheme != "codegraph" or not parts.netloc or not parts.path.strip("/"):
        return None
    return unquote(parts.netloc), unquote(parts.path.lstrip("/"))


def _captured_paths(extra: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key, relation in (("files_modified", "modified"), ("files_read", "read")):
        values = extra.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            path = str(value).strip().replace("\\", "/")
            if path:
                out.append((path, relation))
    return out


def _relative_capture(path: str, repo_root: Path) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return candidate.as_posix()
    return normalize_code_path(candidate.as_posix())


def _node_columns(conn: sqlite3.Connection) -> set[str]:
    return {str(row[1]) for row in conn.execute("PRAGMA table_info(nodes)")}


def _file_nodes(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    columns = _node_columns(conn)
    if not {"id", "kind", "name", "file_path"}.issubset(columns):
        return []
    optional = {
        "qualified_name": "qualified_name",
        "start_line": "start_line",
        "end_line": "end_line",
    }
    selects = ["id", "kind", "name", "file_path"]
    selects.extend(
        column if column in columns else f"NULL AS {alias}" for alias, column in optional.items()
    )
    return conn.execute(
        f"SELECT {', '.join(selects)} FROM nodes ORDER BY file_path, kind, id"  # noqa: S608
    ).fetchall()


def _reference_from_row(
    row: sqlite3.Row,
    *,
    repo_id: str,
    relation: str,
) -> CodeReference:
    stable_id = str(row["id"])
    label = str(row["name"] or row["file_path"] or stable_id)
    return CodeReference(
        uri=codegraph_uri(repo_id, stable_id),
        repo_id=repo_id,
        stable_symbol_id=stable_id,
        kind=str(row["kind"] or "symbol"),
        label=label,
        qualified_name=str(row["qualified_name"] or label),
        file_path=str(row["file_path"] or ""),
        start_line=int(row["start_line"]) if row["start_line"] is not None else None,
        end_line=int(row["end_line"]) if row["end_line"] is not None else None,
        relation=relation,
        confidence=0.95 if relation == "modified" else 0.85,
    )


def _explicit_references(extra: dict[str, Any]) -> list[CodeReference]:
    values = extra.get("code_refs")
    if not isinstance(values, list):
        return []
    out: list[CodeReference] = []
    for value in values:
        uri = str(value.get("uri") or "") if isinstance(value, dict) else str(value)
        parsed = parse_codegraph_uri(uri)
        if parsed is None:
            continue
        repo_id, stable_id = parsed
        metadata = value if isinstance(value, dict) else {}
        label = str(metadata.get("label") or stable_id)
        out.append(
            CodeReference(
                uri=uri,
                repo_id=repo_id,
                stable_symbol_id=stable_id,
                kind=str(metadata.get("kind") or "symbol"),
                label=label,
                qualified_name=str(metadata.get("qualified_name") or label),
                file_path=str(metadata.get("file_path") or ""),
                start_line=(
                    int(metadata["start_line"]) if metadata.get("start_line") is not None else None
                ),
                end_line=(
                    int(metadata["end_line"]) if metadata.get("end_line") is not None else None
                ),
                relation=str(metadata.get("relation") or "explicit"),
                confidence=1.0,
                code_evidence=(
                    dict(metadata["code_evidence"])
                    if isinstance(metadata.get("code_evidence"), dict)
                    else {}
                ),
            )
        )
    return out


def resolve_code_references(
    extra: dict[str, Any] | None,
    *,
    db_path: Path | None = None,
    repo_root: Path | None = None,
    repo_id: str | None = None,
) -> tuple[CodeReference, ...]:
    """Resolve capture metadata to codegraph nodes without inventing misses."""
    return CodeReferenceResolver(
        db_path=db_path,
        repo_root=repo_root,
        repo_id=repo_id,
    ).resolve(extra)


class CodeReferenceResolver:
    """Batch-friendly resolver that opens the codegraph database once."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        repo_root: Path | None = None,
        repo_id: str | None = None,
    ) -> None:
        default_db = db_path is None
        if db_path is None or repo_root is None:
            from memo import codegraph_loader

            db_path = db_path or codegraph_loader.CODEGRAPH_DB
            repo_root = repo_root or db_path.parent.parent
        assert db_path is not None
        assert repo_root is not None
        if default_db and not db_path.is_file():
            common_root = _git_common_repo_root(repo_root)
            common_db = common_root / ".codegraph/codegraph.db" if common_root is not None else None
            if common_root is not None and common_db is not None and common_db.is_file():
                db_path = common_db
                repo_root = common_root
        self.db_path = db_path
        self.repo_root = repo_root
        self.repo_id = repo_id or codegraph_repo_id(repo_root)
        self.nodes: list[sqlite3.Row] = []
        self.nodes_by_path: dict[str, list[sqlite3.Row]] = {}
        self._evidence_resolver: CodegraphEvidenceResolver | None = None
        self._evidence_by_path: dict[str, dict[str, Any]] = {}
        if db_path.is_file():
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
                try:
                    self.nodes = _file_nodes(conn)
                finally:
                    conn.close()
            except sqlite3.Error:
                self.nodes = []
        for row in self.nodes:
            self.nodes_by_path.setdefault(str(row["file_path"]), []).append(row)

    def _matching_nodes(self, relative: str) -> list[sqlite3.Row]:
        """Resolve exact and repo-prefixed paths without scanning every node."""
        parts = relative.split("/")
        candidate_paths = dict.fromkeys("/".join(parts[index:]) for index in range(len(parts)))
        return [
            row for candidate in candidate_paths for row in self.nodes_by_path.get(candidate, ())
        ]

    def _code_evidence(self, file_path: str) -> dict[str, Any]:
        normalized = normalize_code_path(file_path)
        cached = self._evidence_by_path.get(normalized)
        if cached is not None:
            return cached
        if self._evidence_resolver is None:
            self._evidence_resolver = CodegraphEvidenceResolver(
                db_path=self.db_path,
                repo_root=self.repo_root,
                repo_id=self.repo_id,
            )
        evidence = self._evidence_resolver.resolve(paths=[normalized]).to_dict()
        self._evidence_by_path[normalized] = evidence
        return evidence

    def resolve(self, extra: dict[str, Any] | None) -> tuple[CodeReference, ...]:
        payload = extra or {}
        refs = _explicit_references(payload)
        for captured, relation in _captured_paths(payload):
            relative = _relative_capture(captured, self.repo_root)
            matches = self._matching_nodes(relative)
            if not matches:
                continue
            matches.sort(
                key=lambda row: (
                    0 if str(row["kind"]) == "file" else 1,
                    -len(str(row["file_path"])),
                    str(row["id"]),
                )
            )
            refs.append(_reference_from_row(matches[0], repo_id=self.repo_id, relation=relation))
        unique = {(ref.uri, ref.relation): ref for ref in refs}
        resolved: list[CodeReference] = []
        for key in sorted(unique):
            ref = unique[key]
            if not ref.code_evidence and ref.file_path:
                ref = replace(
                    ref,
                    code_evidence=self._code_evidence(ref.file_path),
                )
            resolved.append(ref)
        return tuple(resolved)


def sync_ast_graph_links(
    graph_store: Any,
    memory_id: str,
    extra: dict[str, Any] | None,
) -> int:
    """Sync resolved AST code references into the GraphStore code_ast_relations table."""
    if graph_store is None or not memory_id or not extra:
        return 0
    refs = resolve_code_references(extra)
    count = 0
    for ref in refs:
        if ref.file_path and ref.label:
            graph_store.upsert_code_ast_link(
                memory_id=memory_id,
                file_path=ref.file_path,
                symbol_name=ref.label,
                qualified_name=ref.qualified_name or ref.label,
                relation_type=ref.relation or "refers_to",
                confidence=ref.confidence,
            )
            count += 1
    return count


__all__ = [
    "CodeReference",
    "CodeReferenceResolver",
    "codegraph_repo_id",
    "codegraph_uri",
    "parse_codegraph_uri",
    "resolve_code_references",
    "sync_ast_graph_links",
]
