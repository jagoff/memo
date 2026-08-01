"""Provider-neutral, bounded architecture context over code indexes.

The public contract is intentionally independent from CodeGraph.  CodeGraph is
the first adapter and remains the parser/index owner; Memo only projects its
records into a compact set of architecture findings with explicit evidence and
completeness claims.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from memo.code_evidence import code_path_in_scope, codegraph_evidence, normalize_code_path
from memo.code_traceability import codegraph_repo_id, codegraph_uri

CODE_CONTEXT_SCHEMA = "memo.code_context_pack.v1"
CODE_CONTEXT_MODES = ("scout", "verify", "audit")

_MAX_SCAN_NODES = 50_000
_MAX_SCAN_EDGES = 100_000
_MAX_FINDING_CHARS = 2_000
_FINDING_ORDER = {
    "hotspot": 0,
    "blast_radius": 1,
    "boundary": 2,
    "cycle": 3,
    "route": 4,
    "package": 5,
    "layer": 6,
    "cluster": 7,
}
_ARCHITECTURE_KEYS = {
    "hotspot": "hotspots",
    "blast_radius": "blast_radius",
    "boundary": "boundaries",
    "cycle": "cycles",
    "route": "routes",
    "package": "packages",
    "layer": "layers",
    "cluster": "clusters",
}


@dataclass(frozen=True)
class CodeContextRequest:
    """One provider-independent architecture request."""

    repo_root: Path
    mode: str = "scout"
    focus: str | None = None
    scope: str = "."
    scope_inferred: bool = False
    limit: int = 8
    cursor: str | None = None
    max_chars: int = 12_000


@dataclass(frozen=True)
class CodeContextFinding:
    """One normalized architecture observation."""

    kind: str
    id: str
    label: str
    score: float
    data: dict[str, Any]
    evidence_uris: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "evidence_uris": list(self.evidence_uris),
        }


@dataclass(frozen=True)
class CodeContextProviderResult:
    """Normalized output required from any code-context provider."""

    provider: str
    provider_version: str | None
    index_generation: str | None
    findings: tuple[CodeContextFinding, ...]
    code_evidence: dict[str, Any]
    records_complete: bool
    limitations: tuple[str, ...] = ()


@runtime_checkable
class CodeContextProvider(Protocol):
    """Adapter boundary for CodeGraph or a future external index."""

    name: str

    def collect(self, request: CodeContextRequest) -> CodeContextProviderResult: ...


@dataclass(frozen=True)
class _Node:
    id: str
    kind: str
    name: str
    qualified_name: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    signature: str
    docstring: str


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    kind: str


@dataclass
class _ArchitectureState:
    by_id: dict[str, _Node]
    focused: set[str]
    relevant: set[str]
    relevant_packages: set[str]
    relevant_layers: set[str]
    package_nodes: dict[str, list[_Node]]
    layer_nodes: dict[str, list[_Node]]
    degree: dict[str, int]
    incoming: dict[str, int]
    outgoing: dict[str, int]
    boundary_counts: dict[tuple[str, str], dict[str, int]]
    boundary_samples: dict[tuple[str, str], list[_Edge]]
    package_adjacency: dict[str, set[str]]
    route_edges: dict[str, list[_Edge]]


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


def _normalize_scope(value: str) -> str:
    normalized = normalize_code_path(value or ".").rstrip("/") or "."
    parts = PurePosixPath(normalized).parts
    if PurePosixPath(normalized).is_absolute() or ".." in parts:
        raise ValueError("scope must be a repo-relative path")
    return normalized


def _infer_scope(focus: str | None) -> str:
    if not focus:
        return "."
    normalized = normalize_code_path(focus)
    parts = PurePosixPath(normalized).parts
    if len(parts) < 2:
        return "."
    if parts[0] in {"apps", "packages", "services"} and len(parts) >= 3:
        return "/".join(parts[:2])
    if parts[0] in {"app", "lib", "src", "test", "tests"}:
        return parts[0]
    return "."


def _package_for(path: str) -> str:
    parent = PurePosixPath(path).parent.as_posix()
    return "." if parent in {"", "."} else parent


def _scope_like(scope: str) -> str:
    escaped = scope.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}/%"


def _layer_for(path: str) -> str:
    parts = PurePosixPath(path).parts
    if not parts:
        return "."
    if parts[0] in {"app", "lib", "src", "test", "tests"} and len(parts) > 1:
        return "/".join(parts[:2])
    if parts[0] in {"apps", "packages", "services"} and len(parts) > 1:
        return "/".join(parts[:2])
    return parts[0]


def _node_dict(node: _Node) -> dict[str, Any]:
    snippet_parts = [node.signature.strip(), node.docstring.strip()]
    snippet = "\n".join(part for part in snippet_parts if part)[:400]
    return {
        "stable_symbol_id": node.id,
        "kind": node.kind,
        "name": node.name,
        "qualified_name": node.qualified_name,
        "file_path": node.file_path,
        "language": node.language,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "snippet": snippet,
    }


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _load_nodes(
    conn: sqlite3.Connection,
    *,
    scope: str,
) -> tuple[list[_Node], bool]:
    columns = _columns(conn, "nodes")
    required = {"id", "kind", "name", "qualified_name", "file_path", "start_line", "end_line"}
    if not required.issubset(columns):
        raise sqlite3.DatabaseError("nodes schema is incompatible")
    language = "language" if "language" in columns else "'' AS language"
    signature = "signature" if "signature" in columns else "'' AS signature"
    docstring = "docstring" if "docstring" in columns else "'' AS docstring"
    where = ""
    params: tuple[Any, ...] = ()
    if scope != ".":
        where = "WHERE file_path = ? OR file_path LIKE ? ESCAPE '\\'"
        params = (scope, _scope_like(scope))
    rows = conn.execute(
        "SELECT id, kind, name, qualified_name, file_path, start_line, end_line, "  # noqa: S608
        f"{language}, {signature}, {docstring} FROM nodes {where} "
        "ORDER BY file_path, start_line, id LIMIT ?",
        (*params, _MAX_SCAN_NODES + 1),
    ).fetchall()
    complete = len(rows) <= _MAX_SCAN_NODES
    return (
        [
            _Node(
                id=str(row["id"]),
                kind=str(row["kind"]),
                name=str(row["name"]),
                qualified_name=str(row["qualified_name"]),
                file_path=normalize_code_path(str(row["file_path"])),
                language=str(row["language"] or ""),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                signature=str(row["signature"] or ""),
                docstring=str(row["docstring"] or ""),
            )
            for row in rows[:_MAX_SCAN_NODES]
        ],
        complete,
    )


def _load_edges(
    conn: sqlite3.Connection,
    node_ids: set[str],
    *,
    scope: str,
) -> tuple[list[_Edge], bool]:
    if scope == ".":
        rows = conn.execute(
            "SELECT source, target, kind FROM edges "
            "WHERE kind IN ('calls', 'decorates', 'extends', 'imports', "
            "'instantiates', 'references') ORDER BY source, target, kind LIMIT ?",
            (_MAX_SCAN_EDGES + 1,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT e.source, e.target, e.kind FROM edges e "
            "JOIN nodes source_node ON source_node.id = e.source "
            "JOIN nodes target_node ON target_node.id = e.target "
            "WHERE e.kind IN ('calls', 'decorates', 'extends', 'imports', "
            "'instantiates', 'references') "
            "AND (source_node.file_path = ? OR source_node.file_path LIKE ? ESCAPE '\\') "
            "AND (target_node.file_path = ? OR target_node.file_path LIKE ? ESCAPE '\\') "
            "ORDER BY e.source, e.target, e.kind LIMIT ?",
            (scope, _scope_like(scope), scope, _scope_like(scope), _MAX_SCAN_EDGES + 1),
        ).fetchall()
    complete = len(rows) <= _MAX_SCAN_EDGES
    edges = [
        _Edge(source=str(row["source"]), target=str(row["target"]), kind=str(row["kind"]))
        for row in rows[:_MAX_SCAN_EDGES]
        if str(row["source"]) in node_ids and str(row["target"]) in node_ids
    ]
    return edges, complete


def _focus_nodes(nodes: list[_Node], focus: str | None) -> set[str]:
    if not focus:
        return {node.id for node in nodes}
    needle = focus.strip().casefold()
    normalized = normalize_code_path(focus)
    return {
        node.id
        for node in nodes
        if needle in node.name.casefold()
        or needle in node.qualified_name.casefold()
        or code_path_in_scope(node.file_path, normalized)
        or normalized in node.file_path
    }


def _relevant_nodes(focused: set[str], edges: list[_Edge], *, has_focus: bool) -> set[str]:
    if not has_focus:
        return set(focused)
    relevant = set(focused)
    for edge in edges:
        if edge.source in focused or edge.target in focused:
            relevant.update((edge.source, edge.target))
    return relevant


def _finish_order(adjacency: dict[str, set[str]], all_nodes: set[str]) -> list[str]:
    visited: set[str] = set()
    finish_order: list[str] = []
    for node in sorted(all_nodes):
        if node in visited:
            continue
        stack: list[tuple[str, bool]] = [(node, False)]
        while stack:
            current, finished = stack.pop()
            if finished:
                finish_order.append(current)
                continue
            if current in visited:
                continue
            visited.add(current)
            stack.append((current, True))
            for neighbor in sorted(adjacency.get(current, ()), reverse=True):
                if neighbor not in visited:
                    stack.append((neighbor, False))
    return finish_order


def _components_from_finish_order(
    reverse: dict[str, set[str]],
    finish_order: list[str],
) -> list[list[str]]:
    components: list[list[str]] = []
    assigned: set[str] = set()
    for node in reversed(finish_order):
        if node in assigned:
            continue
        component: list[str] = []
        component_stack = [node]
        assigned.add(node)
        while component_stack:
            current = component_stack.pop()
            component.append(current)
            for neighbor in sorted(reverse.get(current, ()), reverse=True):
                if neighbor not in assigned:
                    assigned.add(neighbor)
                    component_stack.append(neighbor)
        components.append(sorted(component))
    return components


def _strong_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    """Return deterministic SCCs without recursion-depth risk."""
    all_nodes = set(adjacency)
    all_nodes.update(neighbor for values in adjacency.values() for neighbor in values)
    reverse: dict[str, set[str]] = defaultdict(set)
    for source, targets in adjacency.items():
        reverse.setdefault(source, set())
        for target in targets:
            reverse[target].add(source)
    return _components_from_finish_order(reverse, _finish_order(adjacency, all_nodes))


def _weak_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    undirected: dict[str, set[str]] = defaultdict(set)
    for source, targets in adjacency.items():
        undirected.setdefault(source, set())
        for target in targets:
            undirected[source].add(target)
            undirected[target].add(source)
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in sorted(undirected):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(undirected[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _bounded_component(component: list[str], separator: str) -> tuple[str, dict[str, Any]]:
    sample = component[:12]
    omitted = len(component) - len(sample)
    label = separator.join(sample)
    if omitted:
        label += f"{separator}… (+{omitted})"
    return (
        label,
        {
            "packages": sample,
            "package_count": len(component),
            "packages_truncated": omitted > 0,
        },
    )


def _architecture_state(
    nodes: list[_Node],
    edges: list[_Edge],
    focus: str | None,
) -> _ArchitectureState:
    by_id = {node.id: node for node in nodes}
    focused = _focus_nodes(nodes, focus)
    relevant = _relevant_nodes(focused, edges, has_focus=bool(focus))
    relevant_packages = {_package_for(by_id[node_id].file_path) for node_id in relevant}
    relevant_layers = {_layer_for(by_id[node_id].file_path) for node_id in relevant}
    package_nodes: dict[str, list[_Node]] = defaultdict(list)
    layer_nodes: dict[str, list[_Node]] = defaultdict(list)
    for node in nodes:
        package_nodes[_package_for(node.file_path)].append(node)
        layer_nodes[_layer_for(node.file_path)].append(node)

    degree: dict[str, int] = defaultdict(int)
    incoming: dict[str, int] = defaultdict(int)
    outgoing: dict[str, int] = defaultdict(int)
    boundary_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    boundary_samples: dict[tuple[str, str], list[_Edge]] = defaultdict(list)
    package_adjacency: dict[str, set[str]] = defaultdict(set)
    route_edges: dict[str, list[_Edge]] = defaultdict(list)
    for edge in edges:
        source = by_id[edge.source]
        target = by_id[edge.target]
        degree[source.id] += 1
        degree[target.id] += 1
        outgoing[source.id] += 1
        incoming[target.id] += 1
        source_package = _package_for(source.file_path)
        target_package = _package_for(target.file_path)
        if source.kind == "route":
            route_edges[source.id].append(edge)
        if source_package == target_package:
            continue
        key = (source_package, target_package)
        boundary_counts[key][edge.kind] += 1
        if len(boundary_samples[key]) < 3:
            boundary_samples[key].append(edge)
        package_adjacency[source_package].add(target_package)
    return _ArchitectureState(
        by_id=by_id,
        focused=focused,
        relevant=relevant,
        relevant_packages=relevant_packages,
        relevant_layers=relevant_layers,
        package_nodes=package_nodes,
        layer_nodes=layer_nodes,
        degree=degree,
        incoming=incoming,
        outgoing=outgoing,
        boundary_counts=boundary_counts,
        boundary_samples=boundary_samples,
        package_adjacency=package_adjacency,
        route_edges=route_edges,
    )


def _hotspot_findings(
    nodes: list[_Node],
    state: _ArchitectureState,
    repo_id: str,
) -> list[CodeContextFinding]:
    findings: list[CodeContextFinding] = []
    for node in nodes:
        if node.id not in state.relevant or state.degree[node.id] == 0:
            continue
        findings.append(
            CodeContextFinding(
                kind="hotspot",
                id=f"hotspot:{node.id}",
                label=node.qualified_name or node.name,
                score=float(state.degree[node.id]),
                data={
                    **_node_dict(node),
                    "degree": state.degree[node.id],
                    "incoming": state.incoming[node.id],
                    "outgoing": state.outgoing[node.id],
                },
                evidence_uris=(codegraph_uri(repo_id, node.id),),
            )
        )
    return findings


def _blast_radius_findings(
    edges: list[_Edge],
    state: _ArchitectureState,
    repo_id: str,
    focus: str | None,
) -> list[CodeContextFinding]:
    if not focus or not state.focused:
        return []
    dependent_ids = {
        edge.source
        for edge in edges
        if edge.target in state.focused and edge.source not in state.focused
    }
    dependency_ids = {
        edge.target
        for edge in edges
        if edge.source in state.focused and edge.target not in state.focused
    }
    dependent_nodes = sorted(
        (state.by_id[node_id] for node_id in dependent_ids),
        key=lambda node: (-state.degree[node.id], node.file_path, node.id),
    )
    dependency_nodes = sorted(
        (state.by_id[node_id] for node_id in dependency_ids),
        key=lambda node: (-state.degree[node.id], node.file_path, node.id),
    )
    focus_samples = sorted(
        (state.by_id[node_id] for node_id in state.focused),
        key=lambda node: (node.file_path, node.start_line, node.id),
    )[:5]
    radius_samples = [*dependent_nodes[:8], *dependency_nodes[:8]]
    return [
        CodeContextFinding(
            kind="blast_radius",
            id="blast:"
            + hashlib.sha256(
                "|".join(sorted(node.id for node in focus_samples)).encode()
            ).hexdigest()[:16],
            label=f"One-hop blast radius for {focus}",
            score=float(len(dependent_nodes) + len(dependency_nodes)),
            data={
                "focus": focus,
                "focus_symbols": [_node_dict(node) for node in focus_samples],
                "dependent_count": len(dependent_nodes),
                "dependents": [_node_dict(node) for node in dependent_nodes[:8]],
                "dependents_truncated": len(dependent_nodes) > 8,
                "dependency_count": len(dependency_nodes),
                "dependencies": [_node_dict(node) for node in dependency_nodes[:8]],
                "dependencies_truncated": len(dependency_nodes) > 8,
                "depth": 1,
            },
            evidence_uris=tuple(
                codegraph_uri(repo_id, node.id) for node in [*focus_samples, *radius_samples]
            ),
        )
    ]


def _boundary_findings(
    state: _ArchitectureState,
    repo_id: str,
    focus: str | None,
) -> list[CodeContextFinding]:
    findings: list[CodeContextFinding] = []
    for (source_package, target_package), kind_counts in state.boundary_counts.items():
        if focus and not ({source_package, target_package} & state.relevant_packages):
            continue
        samples = state.boundary_samples[(source_package, target_package)]
        count = sum(kind_counts.values())
        findings.append(
            CodeContextFinding(
                kind="boundary",
                id=f"boundary:{source_package}->{target_package}",
                label=f"{source_package} → {target_package}",
                score=float(count),
                data={
                    "source_package": source_package,
                    "target_package": target_package,
                    "edge_count": count,
                    "edge_kinds": dict(sorted(kind_counts.items())),
                    "samples": [
                        {
                            "kind": edge.kind,
                            "source": _node_dict(state.by_id[edge.source]),
                            "target": _node_dict(state.by_id[edge.target]),
                        }
                        for edge in samples
                    ],
                },
                evidence_uris=tuple(
                    codegraph_uri(repo_id, node_id)
                    for edge in samples
                    for node_id in (edge.source, edge.target)
                ),
            )
        )
    return findings


def _component_findings(
    state: _ArchitectureState,
    focus: str | None,
) -> list[CodeContextFinding]:
    findings: list[CodeContextFinding] = []
    for component in _strong_components(state.package_adjacency):
        if len(component) < 2 or (focus and not set(component) & state.relevant_packages):
            continue
        internal_edges = sum(
            sum(kind_counts.values())
            for (source, target), kind_counts in state.boundary_counts.items()
            if source in component and target in component
        )
        label, component_data = _bounded_component(component, " ↔ ")
        findings.append(
            CodeContextFinding(
                kind="cycle",
                id="cycle:" + hashlib.sha256("|".join(component).encode()).hexdigest()[:16],
                label=label,
                score=float(internal_edges),
                data={**component_data, "cross_package_edges": internal_edges},
            )
        )

    for component in _weak_components(state.package_adjacency):
        if len(component) < 2 or (focus and not set(component) & state.relevant_packages):
            continue
        label, component_data = _bounded_component(component, ", ")
        findings.append(
            CodeContextFinding(
                kind="cluster",
                id="cluster:" + hashlib.sha256("|".join(component).encode()).hexdigest()[:16],
                label=label,
                score=float(len(component)),
                data=component_data,
            )
        )
    return findings


def _route_findings(
    nodes: list[_Node],
    state: _ArchitectureState,
    repo_id: str,
) -> list[CodeContextFinding]:
    findings: list[CodeContextFinding] = []
    for node in nodes:
        if node.kind != "route" or node.id not in state.relevant:
            continue
        outgoing_edges = state.route_edges.get(node.id, [])
        findings.append(
            CodeContextFinding(
                kind="route",
                id=f"route:{node.id}",
                label=node.qualified_name or node.name,
                score=float(max(1, len(outgoing_edges))),
                data={
                    **_node_dict(node),
                    "targets": [
                        {"kind": edge.kind, **_node_dict(state.by_id[edge.target])}
                        for edge in outgoing_edges[:5]
                    ],
                },
                evidence_uris=(codegraph_uri(repo_id, node.id),),
            )
        )
    return findings


def _package_findings(
    state: _ArchitectureState,
    repo_id: str,
    focus: str | None,
) -> list[CodeContextFinding]:
    findings: list[CodeContextFinding] = []
    for package, members in state.package_nodes.items():
        if focus and package not in state.relevant_packages:
            continue
        file_paths = sorted({node.file_path for node in members})
        languages = sorted({node.language for node in members if node.language})
        package_samples = sorted(
            members,
            key=lambda node: (-state.degree[node.id], node.file_path, node.id),
        )[:3]
        findings.append(
            CodeContextFinding(
                kind="package",
                id=f"package:{package}",
                label=package,
                score=float(len(members)),
                data={
                    "package": package,
                    "file_count": len(file_paths),
                    "symbol_count": len(members),
                    "languages": languages,
                    "sample_paths": file_paths[:5],
                    "sample_symbols": [_node_dict(node) for node in package_samples],
                },
                evidence_uris=tuple(codegraph_uri(repo_id, node.id) for node in package_samples),
            )
        )
    return findings


def _layer_findings(
    state: _ArchitectureState,
    repo_id: str,
    focus: str | None,
) -> list[CodeContextFinding]:
    findings: list[CodeContextFinding] = []
    for layer, members in state.layer_nodes.items():
        if focus and layer not in state.relevant_layers:
            continue
        file_paths = sorted({node.file_path for node in members})
        packages = sorted({_package_for(node.file_path) for node in members})
        layer_samples = sorted(
            members,
            key=lambda node: (-state.degree[node.id], node.file_path, node.id),
        )[:3]
        findings.append(
            CodeContextFinding(
                kind="layer",
                id=f"layer:{layer}",
                label=layer,
                score=float(len(members)),
                data={
                    "layer": layer,
                    "packages": packages[:12],
                    "package_count": len(packages),
                    "packages_truncated": len(packages) > 12,
                    "file_count": len(file_paths),
                    "symbol_count": len(members),
                    "sample_symbols": [_node_dict(node) for node in layer_samples],
                },
                evidence_uris=tuple(codegraph_uri(repo_id, node.id) for node in layer_samples),
            )
        )
    return findings


def _interleave_findings(findings: list[CodeContextFinding]) -> list[CodeContextFinding]:
    grouped: dict[str, list[CodeContextFinding]] = defaultdict(list)
    for finding in findings:
        grouped[finding.kind].append(finding)
    for values in grouped.values():
        values.sort(key=lambda item: (-item.score, item.id))
    interleaved: list[CodeContextFinding] = []
    index = 0
    while any(index < len(values) for values in grouped.values()):
        for kind in _FINDING_ORDER:
            values = grouped.get(kind, [])
            if index < len(values):
                interleaved.append(values[index])
        index += 1
    return interleaved


def _architecture_findings(
    *,
    nodes: list[_Node],
    edges: list[_Edge],
    repo_id: str,
    focus: str | None,
) -> list[CodeContextFinding]:
    state = _architecture_state(nodes, edges, focus)
    findings = [
        *_hotspot_findings(nodes, state, repo_id),
        *_blast_radius_findings(edges, state, repo_id, focus),
        *_boundary_findings(state, repo_id, focus),
        *_component_findings(state, focus),
        *_route_findings(nodes, state, repo_id),
        *_package_findings(state, repo_id, focus),
        *_layer_findings(state, repo_id, focus),
    ]
    return _interleave_findings(findings)


def _exact_focus_path(
    conn: sqlite3.Connection,
    *,
    focus: str | None,
    scope: str,
) -> str | None:
    if not focus:
        return None
    normalized = normalize_code_path(focus)
    if not code_path_in_scope(normalized, scope):
        return None
    row = conn.execute("SELECT path FROM files WHERE path = ?", (normalized,)).fetchone()
    return normalized if row is not None else None


class CodeGraphContextProvider:
    """Project a local CodeGraph SQLite database into the normalized contract."""

    name = "codegraph"

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path

    def collect(self, request: CodeContextRequest) -> CodeContextProviderResult:
        repo_root = request.repo_root
        db_path = self.db_path or repo_root / ".codegraph" / "codegraph.db"
        repo_id = codegraph_repo_id(repo_root)
        if not db_path.is_file():
            evidence = codegraph_evidence(
                db_path=db_path,
                repo_root=repo_root,
                repo_id=repo_id,
                scopes=(request.scope,),
            ).to_dict()
            return CodeContextProviderResult(
                provider=self.name,
                provider_version=None,
                index_generation=None,
                findings=(),
                code_evidence=evidence,
                records_complete=False,
                limitations=("Architecture context requires .codegraph/codegraph.db.",),
            )

        limitations: list[str] = []
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            try:
                nodes, nodes_complete = _load_nodes(conn, scope=request.scope)
                edges, edges_complete = _load_edges(
                    conn,
                    {node.id for node in nodes},
                    scope=request.scope,
                )
                exact_path = _exact_focus_path(
                    conn,
                    focus=request.focus,
                    scope=request.scope,
                )
            finally:
                conn.close()
        except sqlite3.Error as exc:
            evidence = codegraph_evidence(
                db_path=db_path,
                repo_root=repo_root,
                repo_id=repo_id,
                scopes=(request.scope,),
            ).to_dict()
            return CodeContextProviderResult(
                provider=self.name,
                provider_version=None,
                index_generation=evidence.get("index_generation"),
                findings=(),
                code_evidence=evidence,
                records_complete=False,
                limitations=(f"CodeGraph architecture query failed: {type(exc).__name__}.",),
            )

        if not nodes_complete:
            limitations.append(f"Node scan was capped at {_MAX_SCAN_NODES}.")
        if not edges_complete:
            limitations.append(f"Edge scan was capped at {_MAX_SCAN_EDGES}.")
        evidence = codegraph_evidence(
            db_path=db_path,
            repo_root=repo_root,
            repo_id=repo_id,
            paths=(exact_path,) if exact_path else (),
            scopes=() if exact_path else (request.scope,),
        ).to_dict()
        return CodeContextProviderResult(
            provider=self.name,
            provider_version=(
                str(evidence["provider_version"]) if evidence.get("provider_version") else None
            ),
            index_generation=(
                str(evidence["index_generation"]) if evidence.get("index_generation") else None
            ),
            findings=tuple(
                _architecture_findings(
                    nodes=nodes,
                    edges=edges,
                    repo_id=repo_id,
                    focus=request.focus,
                )
            ),
            code_evidence=evidence,
            records_complete=nodes_complete and edges_complete,
            limitations=tuple(limitations),
        )


def _cursor_fingerprint(request: CodeContextRequest, result: CodeContextProviderResult) -> str:
    value = "|".join(
        (
            result.provider,
            result.index_generation or "",
            request.mode,
            request.focus or "",
            request.scope,
        )
    )
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _encode_cursor(offset: int, fingerprint: str) -> str:
    raw = json.dumps(
        {"v": 1, "offset": offset, "fingerprint": fingerprint},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(value: str | None, fingerprint: str) -> int:
    if not value:
        return 0
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        if (
            payload.get("v") != 1
            or payload.get("fingerprint") != fingerprint
            or int(payload.get("offset", -1)) < 0
        ):
            raise ValueError
        return int(payload["offset"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("cursor is invalid or belongs to another request") from exc


def _strip_snippets(value: Any) -> None:
    if isinstance(value, dict):
        if "snippet" in value:
            value["snippet"] = ""
        for child in value.values():
            _strip_snippets(child)
    elif isinstance(value, list):
        for child in value:
            _strip_snippets(child)


def _bounded_finding(finding: CodeContextFinding) -> tuple[dict[str, Any], str]:
    item = finding.to_dict()
    encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
    if len(encoded) > _MAX_FINDING_CHARS:
        item["data"]["detail_omitted"] = True
        for key in (
            "dependencies",
            "dependents",
            "focus_symbols",
            "packages",
            "sample_paths",
            "samples",
            "sample_symbols",
            "targets",
        ):
            if key in item["data"]:
                item["data"][key] = item["data"][key][:1]
        item["evidence_uris"] = item["evidence_uris"][:6]
        _strip_snippets(item["data"])
        encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
    if len(encoded) > _MAX_FINDING_CHARS:
        item["data"] = {
            key: value for key, value in item["data"].items() if not isinstance(value, (dict, list))
        }
        item["data"]["detail_omitted"] = True
        encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return item, encoded


def _bounded_page(
    findings: tuple[CodeContextFinding, ...],
    *,
    offset: int,
    limit: int,
    max_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    page: list[dict[str, Any]] = []
    used = 0
    for finding in findings[offset : offset + limit]:
        item, encoded = _bounded_finding(finding)
        if page and used + len(encoded) > max_chars:
            break
        page.append(item)
        used += len(encoded)
    return page, offset + len(page)


def _source_universe_complete(
    request: CodeContextRequest,
    provider_result: CodeContextProviderResult,
    *,
    page_exhausted: bool,
) -> bool:
    evidence = provider_result.code_evidence
    exact_path = bool(evidence.get("requested_paths")) and not evidence.get("requested_scopes")
    return bool(
        exact_path
        and provider_result.records_complete
        and page_exhausted
        and evidence.get("recording_status") == "complete"
        and evidence.get("coverage_status") == "complete"
        and evidence.get("freshness") == "current"
        and request.mode in {"verify", "audit"}
    )


def _absence_reason(
    *,
    mode: str,
    records_complete: bool,
    page_exhausted: bool,
    source_complete: bool,
) -> str | None:
    if source_complete:
        return None
    if mode == "scout":
        return "scout reports positive findings only"
    if not records_complete:
        return "provider scan was capped or incomplete"
    if not page_exhausted:
        return "additional findings remain; follow next_cursor"
    return "the requested source universe is not provably complete"


def _context_omissions(
    request: CodeContextRequest,
    result: CodeContextProviderResult,
    *,
    next_cursor: str | None,
    next_offset: int,
    focus_resolved: bool,
) -> list[dict[str, Any]]:
    omissions: list[dict[str, Any]] = []
    if next_cursor:
        omissions.append(
            {
                "kind": "pagination",
                "count": len(result.findings) - next_offset,
                "reason": "bounded_page",
            }
        )
    if not result.records_complete:
        omissions.append(
            {
                "kind": "provider_records",
                "count": None,
                "reason": "scan_cap_or_incomplete_provider",
            }
        )
    if request.focus and not focus_resolved:
        omissions.append(
            {
                "kind": "focus",
                "count": None,
                "reason": "focus_not_resolved_in_provider_records",
            }
        )
    if request.focus and request.scope != ".":
        omissions.append(
            {
                "kind": "cross_scope_edges",
                "count": None,
                "reason": "blast_radius_scope_boundary",
            }
        )
    return omissions


def _context_limitations(
    request: CodeContextRequest,
    result: CodeContextProviderResult,
    *,
    normalized_mode: str,
    source_complete: bool,
) -> list[str]:
    limitations = [*result.limitations, *result.code_evidence.get("limitations", [])]
    if request.focus and request.scope != ".":
        limitations.append(
            "Blast radius only includes provider edges whose endpoints are inside "
            f"{request.scope!r}; cross-scope callers and dependencies are omitted."
        )
    if normalized_mode == "scout":
        limitations.append("Scout mode reports positive findings only; it cannot prove absence.")
    elif not source_complete:
        limitations.append(
            "Absence claims are disabled because source-universe completeness is unproven."
        )
    return limitations


def build_code_context_pack(
    cwd: str | Path,
    *,
    focus: str | None = None,
    scope: str | None = None,
    mode: str = "scout",
    limit: int = 8,
    cursor: str | None = None,
    max_chars: int = 12_000,
    provider: CodeContextProvider | None = None,
) -> dict[str, Any]:
    """Build a bounded architecture pack without parsing source files."""
    normalized_mode = mode.strip().casefold()
    if normalized_mode not in CODE_CONTEXT_MODES:
        return {
            "schema": CODE_CONTEXT_SCHEMA,
            "available": False,
            "reason": "invalid_mode",
            "modes": list(CODE_CONTEXT_MODES),
        }
    scope_inferred = scope is None
    try:
        normalized_scope = _normalize_scope(_infer_scope(focus) if scope_inferred else scope or ".")
    except ValueError as exc:
        return {
            "schema": CODE_CONTEXT_SCHEMA,
            "available": False,
            "reason": "invalid_scope",
            "detail": str(exc),
        }

    repo_root = _git_root(Path(cwd))
    request = CodeContextRequest(
        repo_root=repo_root,
        mode=normalized_mode,
        focus=focus.strip() if focus and focus.strip() else None,
        scope=normalized_scope,
        scope_inferred=scope_inferred,
        limit=max(1, min(int(limit), 100 if normalized_mode == "audit" else 50)),
        cursor=cursor,
        max_chars=max(2_048, min(int(max_chars), 48_000)),
    )
    selected_provider = provider or CodeGraphContextProvider()
    result = selected_provider.collect(request)
    fingerprint = _cursor_fingerprint(request, result)
    try:
        offset = _decode_cursor(cursor, fingerprint)
    except ValueError as exc:
        return {
            "schema": CODE_CONTEXT_SCHEMA,
            "available": False,
            "reason": "invalid_cursor",
            "detail": str(exc),
        }
    if offset > len(result.findings):
        return {
            "schema": CODE_CONTEXT_SCHEMA,
            "available": False,
            "reason": "invalid_cursor",
            "detail": "cursor offset is outside this result set",
        }

    page_findings, next_offset = _bounded_page(
        result.findings,
        offset=offset,
        limit=request.limit,
        max_chars=request.max_chars,
    )
    exhausted = next_offset >= len(result.findings)
    next_cursor = None if exhausted else _encode_cursor(next_offset, fingerprint)
    source_complete = _source_universe_complete(
        request,
        result,
        page_exhausted=exhausted,
    )
    architecture: dict[str, list[dict[str, Any]]] = {key: [] for key in _ARCHITECTURE_KEYS.values()}
    for finding in page_findings:
        architecture[_ARCHITECTURE_KEYS[str(finding["kind"])]].append(finding)
    focus_resolved = not request.focus or any(
        finding.kind == "blast_radius" for finding in result.findings
    )
    omissions = _context_omissions(
        request,
        result,
        next_cursor=next_cursor,
        next_offset=next_offset,
        focus_resolved=focus_resolved,
    )
    available = result.code_evidence.get("recording_status") not in {"missing", "unreadable"}
    limitations = _context_limitations(
        request,
        result,
        normalized_mode=normalized_mode,
        source_complete=source_complete,
    )
    return {
        "schema": CODE_CONTEXT_SCHEMA,
        "available": available,
        "reason": None if available else "provider_unavailable",
        "provider": {
            "name": result.provider,
            "version": result.provider_version,
            "index_generation": result.index_generation,
        },
        "mode": normalized_mode,
        "request": {
            "repo_root": str(repo_root),
            "focus": request.focus,
            "scope": request.scope,
            "scope_inferred": request.scope_inferred,
            "limit": request.limit,
            "max_chars": request.max_chars,
        },
        "page": {
            "cursor": cursor,
            "next_cursor": next_cursor,
            "offset": offset,
            "returned": len(page_findings),
            "total_findings": len(result.findings),
            "exhausted": exhausted,
        },
        "architecture": architecture,
        "findings": page_findings,
        "code_evidence": result.code_evidence,
        "claims": {
            "positive_findings_only": normalized_mode == "scout",
            "provider_records_complete": result.records_complete,
            "page_exhausted": exhausted,
            "source_universe_complete": source_complete,
            "focus_resolved": focus_resolved,
            "absence_claim_allowed": source_complete,
            "absence_claim_scope": (
                {
                    "kind": "path",
                    "paths": list(result.code_evidence.get("requested_paths") or []),
                }
                if source_complete
                else None
            ),
            "architecture_absence_claim_allowed": False,
            "absence_claim_reason": _absence_reason(
                mode=normalized_mode,
                records_complete=result.records_complete,
                page_exhausted=exhausted,
                source_complete=source_complete,
            ),
        },
        "omissions": omissions,
        "limitations": list(dict.fromkeys(limitations)),
    }


__all__ = [
    "CODE_CONTEXT_MODES",
    "CODE_CONTEXT_SCHEMA",
    "CodeContextFinding",
    "CodeContextProvider",
    "CodeContextProviderResult",
    "CodeContextRequest",
    "CodeGraphContextProvider",
    "build_code_context_pack",
]
