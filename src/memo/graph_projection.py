"""Curated, deterministic serving projection for memo's raw knowledge graph.

The raw graph remains evidence.  This module decides which evidence is safe to
serve and gives accepted nodes stable namespaced identifiers.  Projection
storage and retrieval live here too so hot paths never need raw graph tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from memo.code_traceability import CodeReference
from memo.errors import MemoError
from memo.graph_canonical import fold_key

_SCALAR_RE = re.compile(
    r"^(?:true|false|null|none|nil|[-+]?\d+(?:\.\d+)?|\d{4}[-/]\d{1,2}[-/]\d{1,2})$",
    re.IGNORECASE,
)
_CODE_SHAPE_RE = re.compile(
    r"(?:^test(?:[_\s-]|$)|^assert\b|\w+\([^)]*\)|(?:==|!=|=>|::))",
    re.IGNORECASE,
)
_REFERENCE_TYPES = frozenset({"reference", "test", "repo", "repository"})
_EXTRACTOR_WEIGHT = {
    "legacy": 0.2,
    "regex": 0.45,
    "llm": 0.85,
    "explicit": 1.0,
}

_PROJECTION_DDL = """
CREATE TABLE IF NOT EXISTS graph_projection_versions (
    version             TEXT PRIMARY KEY,
    status              TEXT NOT NULL,
    built_at            TEXT NOT NULL,
    source_fingerprint  TEXT NOT NULL,
    total_memories      INTEGER NOT NULL DEFAULT 0,
    node_count          INTEGER NOT NULL DEFAULT 0,
    edge_count          INTEGER NOT NULL DEFAULT 0,
    rejected_count      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS graph_projection_nodes (
    version        TEXT NOT NULL,
    uri            TEXT NOT NULL,
    entity_id      INTEGER NOT NULL,
    entity_type    TEXT NOT NULL,
    canonical_key  TEXT NOT NULL,
    label          TEXT NOT NULL,
    doc_freq       INTEGER NOT NULL,
    degree         INTEGER NOT NULL,
    quality        REAL NOT NULL,
    is_hub         INTEGER NOT NULL,
    idf            REAL NOT NULL,
    PRIMARY KEY (version, uri)
);

CREATE TABLE IF NOT EXISTS graph_projection_memberships (
    version       TEXT NOT NULL,
    memory_id     TEXT NOT NULL,
    uri           TEXT NOT NULL,
    confidence    REAL NOT NULL,
    evidence_id   TEXT NOT NULL,
    PRIMARY KEY (version, memory_id, uri)
);

CREATE TABLE IF NOT EXISTS graph_projection_edges (
    version            TEXT NOT NULL,
    a_uri              TEXT NOT NULL,
    b_uri              TEXT NOT NULL,
    relation           TEXT NOT NULL,
    weight             REAL NOT NULL,
    confidence         REAL NOT NULL,
    evidence_count     INTEGER NOT NULL,
    first_seen         TEXT,
    last_seen          TEXT,
    evidence_ids_json  TEXT NOT NULL,
    PRIMARY KEY (version, a_uri, b_uri, relation)
);

CREATE TABLE IF NOT EXISTS graph_projection_code_links (
    version           TEXT NOT NULL,
    memory_id         TEXT NOT NULL,
    uri               TEXT NOT NULL,
    relation          TEXT NOT NULL,
    repo_id           TEXT NOT NULL,
    stable_symbol_id  TEXT NOT NULL,
    kind              TEXT NOT NULL,
    label             TEXT NOT NULL,
    qualified_name    TEXT NOT NULL,
    file_path         TEXT NOT NULL,
    start_line        INTEGER,
    end_line          INTEGER,
    confidence        REAL NOT NULL,
    evidence_id       TEXT NOT NULL,
    PRIMARY KEY (version, memory_id, uri, relation)
);

CREATE TABLE IF NOT EXISTS graph_projection_rejections (
    version        TEXT NOT NULL,
    entity_id      INTEGER NOT NULL,
    candidate_uri  TEXT NOT NULL,
    quality        REAL NOT NULL,
    reason         TEXT NOT NULL,
    PRIMARY KEY (version, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_gpn_version ON graph_projection_nodes(version);
CREATE INDEX IF NOT EXISTS idx_gpm_memory ON graph_projection_memberships(version, memory_id);
CREATE INDEX IF NOT EXISTS idx_gpm_uri ON graph_projection_memberships(version, uri);
CREATE INDEX IF NOT EXISTS idx_gpe_a ON graph_projection_edges(version, a_uri);
CREATE INDEX IF NOT EXISTS idx_gpe_b ON graph_projection_edges(version, b_uri);
CREATE INDEX IF NOT EXISTS idx_gpcl_memory ON graph_projection_code_links(version, memory_id);
CREATE INDEX IF NOT EXISTS idx_gpcl_uri ON graph_projection_code_links(version, uri);
CREATE INDEX IF NOT EXISTS idx_gpr_version ON graph_projection_rejections(version);
"""


class ProjectionBuildError(MemoError, RuntimeError):
    """A candidate projection could not be validated or activated."""


@dataclass(frozen=True)
class ProjectionMemoryState:
    id: str
    type: str
    forgotten: bool = False
    code_refs: tuple[CodeReference, ...] = ()


@dataclass(frozen=True)
class RawEntityEvidence:
    entity_id: int
    name: str
    entity_type: str
    extractors: tuple[str, ...]
    confidences: tuple[float, ...]
    memories: tuple[ProjectionMemoryState, ...]
    alias_count: int = 0
    first_seen: str | None = None
    last_seen: str | None = None


@dataclass(frozen=True)
class ProjectionBuildConfig:
    min_quality: float = 0.45
    hub_max_doc_freq_ratio: float = 0.25
    evidence_limit: int = 8


@dataclass(frozen=True)
class ProjectionDecision:
    eligible: bool
    quality: float
    reason: str | None
    uri: str


@dataclass(frozen=True)
class ProjectedNode:
    uri: str
    label: str
    entity_type: str
    canonical_key: str
    doc_freq: int
    degree: int
    quality: float
    is_hub: bool
    idf: float


@dataclass(frozen=True)
class ProjectedEdge:
    source_uri: str
    target_uri: str
    relation: str
    weight: float
    confidence: float
    evidence_ids: tuple[str, ...]
    first_seen: str | None
    last_seen: str | None


@dataclass(frozen=True)
class ProjectedCodeLink:
    memory_id: str
    uri: str
    relation: str
    repo_id: str
    stable_symbol_id: str
    kind: str
    label: str
    qualified_name: str
    file_path: str
    start_line: int | None
    end_line: int | None
    confidence: float
    evidence_id: str


@dataclass(frozen=True)
class ProjectionBuildResult:
    version: str
    activated: bool
    node_count: int
    edge_count: int
    rejected_count: int
    built_at: str
    code_node_count: int = 0
    code_link_count: int = 0


_ProjectionRow = dict[str, Any]


def _curate_entity_rows(
    raw: list[RawEntityEvidence],
    config: ProjectionBuildConfig,
) -> tuple[dict[str, _ProjectionRow], list[_ProjectionRow]]:
    accepted: dict[str, _ProjectionRow] = {}
    rejections: list[_ProjectionRow] = []
    for item in raw:
        decision = evaluate_entity(item, config)
        if not decision.eligible:
            rejections.append(
                {
                    "entity_id": item.entity_id,
                    "uri": decision.uri,
                    "quality": decision.quality,
                    "reason": decision.reason or "unknown",
                }
            )
            continue
        candidate = accepted.setdefault(
            decision.uri,
            {
                "entity_id": item.entity_id,
                "uri": decision.uri,
                "entity_type": item.entity_type,
                "canonical_key": fold_key(item.name),
                "label": item.name,
                "quality": decision.quality,
                "first_seen": item.first_seen,
                "last_seen": item.last_seen,
                "memberships": {},
            },
        )
        if decision.quality > float(candidate["quality"]):
            candidate.update(
                entity_id=item.entity_id,
                entity_type=item.entity_type,
                label=item.name,
                quality=decision.quality,
            )
        memberships: dict[str, float] = candidate["memberships"]
        for memory, confidence in zip(item.memories, item.confidences, strict=True):
            if memory.forgotten:
                continue
            memberships[memory.id] = max(
                memberships.get(memory.id, 0.0),
                max(0.0, min(1.0, confidence)),
            )
    return accepted, rejections


def _merge_code_reference_rows(
    accepted: dict[str, _ProjectionRow],
    memories: Mapping[str, ProjectionMemoryState],
) -> list[_ProjectionRow]:
    code_links: list[_ProjectionRow] = []
    for memory_id, memory in sorted(memories.items()):
        if memory.forgotten:
            continue
        for ref in memory.code_refs:
            canonical = fold_key(ref.qualified_name or ref.file_path or ref.label)
            candidate = accepted.setdefault(
                ref.uri,
                {
                    "entity_id": -(int(hashlib.sha256(ref.uri.encode()).hexdigest()[:8], 16) + 1),
                    "uri": ref.uri,
                    "entity_type": f"code:{ref.kind}",
                    "canonical_key": canonical,
                    "label": ref.label,
                    "quality": 1.0,
                    "first_seen": None,
                    "last_seen": None,
                    "memberships": {},
                },
            )
            memberships: dict[str, float] = candidate["memberships"]
            memberships[memory_id] = max(
                memberships.get(memory_id, 0.0),
                max(0.0, min(1.0, ref.confidence)),
            )
            code_links.append(
                {
                    **asdict(ref),
                    "memory_id": memory_id,
                    "evidence_id": memory_uri(memory_id),
                }
            )
    return code_links


def _materialize_memberships(
    accepted: dict[str, _ProjectionRow],
    memories: Mapping[str, ProjectionMemoryState],
    config: ProjectionBuildConfig,
) -> tuple[list[_ProjectionRow], dict[str, dict[str, float]]]:
    live_total = sum(not memory.forgotten for memory in memories.values())
    rows: list[_ProjectionRow] = []
    memory_nodes: dict[str, dict[str, float]] = defaultdict(dict)
    for node in accepted.values():
        memberships: dict[str, float] = node["memberships"]
        node["doc_freq"] = len(memberships)
        ratio = len(memberships) / live_total if live_total else 0.0
        node["is_hub"] = ratio > config.hub_max_doc_freq_ratio
        node["idf"] = max(0.0, math.log((1 + live_total) / (1 + len(memberships))))
        for memory_id, confidence in sorted(memberships.items()):
            rows.append(
                {
                    "memory_id": memory_id,
                    "uri": node["uri"],
                    "confidence": confidence,
                    "evidence_id": memory_uri(memory_id),
                }
            )
            memory_nodes[memory_id][node["uri"]] = confidence
    return rows, memory_nodes


def _edge_relation(a_uri: str, b_uri: str) -> str:
    a_is_code = a_uri.startswith("codegraph://")
    b_is_code = b_uri.startswith("codegraph://")
    if a_is_code and b_is_code:
        return "code_co_touched"
    if a_is_code or b_is_code:
        return "contextualizes_code"
    return "co_occurs"


def _materialize_edges(
    memory_nodes: Mapping[str, dict[str, float]],
    evidence_limit: int,
) -> tuple[list[_ProjectionRow], Counter[str]]:
    accumulator: dict[tuple[str, str], _ProjectionRow] = {}
    for memory_id, uri_confidences in sorted(memory_nodes.items()):
        uris = sorted(uri_confidences)
        for index, a_uri in enumerate(uris):
            for b_uri in uris[index + 1 :]:
                edge = accumulator.setdefault(
                    (a_uri, b_uri),
                    {"confidences": [], "evidence_ids": []},
                )
                edge["confidences"].append(min(uri_confidences[a_uri], uri_confidences[b_uri]))
                edge["evidence_ids"].append(memory_uri(memory_id))

    rows: list[_ProjectionRow] = []
    degrees: Counter[str] = Counter()
    for (a_uri, b_uri), values in sorted(accumulator.items()):
        evidence_ids = tuple(sorted(values["evidence_ids"]))
        confidences = values["confidences"]
        degrees[a_uri] += 1
        degrees[b_uri] += 1
        rows.append(
            {
                "a_uri": a_uri,
                "b_uri": b_uri,
                "relation": _edge_relation(a_uri, b_uri),
                "weight": float(len(evidence_ids)),
                "confidence": sum(confidences) / len(confidences),
                "evidence_count": len(evidence_ids),
                "first_seen": None,
                "last_seen": None,
                "evidence_ids": evidence_ids[: max(0, evidence_limit)],
            }
        )
    return rows, degrees


def _finalize_node_rows(
    accepted: dict[str, _ProjectionRow],
    degrees: Counter[str],
) -> list[_ProjectionRow]:
    rows: list[_ProjectionRow] = []
    for node in sorted(accepted.values(), key=lambda value: value["uri"]):
        node["degree"] = degrees[node["uri"]]
        node.pop("memberships", None)
        rows.append(node)
    return rows


class GraphReadModel:
    """Immutable in-memory view of one complete projection version."""

    def __init__(
        self,
        *,
        available: bool,
        skip_reason: str | None = None,
        version: str | None = None,
        built_at: str | None = None,
        total_memories: int = 0,
        nodes: Mapping[str, ProjectedNode] | None = None,
        memberships: Mapping[str, tuple[str, ...]] | None = None,
        edges: Mapping[str, tuple[ProjectedEdge, ...]] | None = None,
        code_links: tuple[ProjectedCodeLink, ...] = (),
    ) -> None:
        self.available = available
        self.skip_reason = skip_reason
        self.version = version
        self.built_at = built_at
        self.total_memories = total_memories
        self._nodes = dict(nodes or {})
        self._memberships = dict(memberships or {})
        self._edges = dict(edges or {})
        code_by_memory: dict[str, list[ProjectedCodeLink]] = defaultdict(list)
        code_by_uri: dict[str, list[ProjectedCodeLink]] = defaultdict(list)
        for link in code_links:
            code_by_memory[link.memory_id].append(link)
            code_by_uri[link.uri].append(link)
        self._code_by_memory = {
            key: tuple(sorted(value, key=lambda link: (link.uri, link.relation)))
            for key, value in code_by_memory.items()
        }
        self._code_by_uri = {
            key: tuple(sorted(value, key=lambda link: (link.memory_id, link.relation)))
            for key, value in code_by_uri.items()
        }

    @classmethod
    def unavailable(cls, reason: str) -> GraphReadModel:
        return cls(available=False, skip_reason=reason)

    def resolve_query_entities(self, query: str) -> tuple[ProjectedNode, ...]:
        query_lower = query.lower()
        compact_query = fold_key(query)
        found: list[ProjectedNode] = []
        for node in self._nodes.values():
            label_pattern = rf"(?<!\w){re.escape(node.label.lower())}(?!\w)"
            label_match = bool(re.search(label_pattern, query_lower))
            compact_match = len(node.canonical_key) >= 4 and node.canonical_key in compact_query
            if label_match or compact_match:
                found.append(node)
        return tuple(sorted(found, key=lambda node: (-len(node.canonical_key), node.uri)))

    def node(self, uri: str) -> ProjectedNode | None:
        return self._nodes.get(uri)

    def memory_nodes(self, memory_id: str) -> tuple[ProjectedNode, ...]:
        return tuple(
            self._nodes[uri] for uri in self._memberships.get(memory_id, ()) if uri in self._nodes
        )

    def neighbors(self, uri: str) -> tuple[ProjectedEdge, ...]:
        return self._edges.get(uri, ())

    def all_nodes(self) -> tuple[ProjectedNode, ...]:
        return tuple(self._nodes[uri] for uri in sorted(self._nodes))

    def all_edges(self) -> tuple[ProjectedEdge, ...]:
        unique: dict[tuple[str, str, str], ProjectedEdge] = {}
        for edges in self._edges.values():
            for edge in edges:
                key = (edge.source_uri, edge.target_uri, edge.relation)
                unique[key] = edge
        return tuple(unique[key] for key in sorted(unique))

    def code_links_for_memory(self, memory_id: str) -> tuple[ProjectedCodeLink, ...]:
        return self._code_by_memory.get(memory_id, ())

    def code_links_for_uri(self, uri: str) -> tuple[ProjectedCodeLink, ...]:
        return self._code_by_uri.get(uri, ())

    def resolve_code(self, query: str) -> tuple[ProjectedNode, ...]:
        needle = query.strip().casefold()
        if not needle:
            return ()
        uris = {
            uri
            for uri, links in self._code_by_uri.items()
            if uri.casefold() == needle
            or any(
                needle in value.casefold()
                for link in links
                for value in (
                    link.label,
                    link.qualified_name,
                    link.file_path,
                    link.stable_symbol_id,
                )
                if value
            )
        }
        return tuple(self._nodes[uri] for uri in sorted(uris) if uri in self._nodes)


class GraphProjectionStore:
    """Build, atomically activate, and read curated projection versions."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        tx_factory: Callable[[], AbstractContextManager[sqlite3.Connection]],
    ) -> None:
        self._conn = conn
        self._tx_factory = tx_factory
        with self._tx_factory() as cx:
            cx.executescript(_PROJECTION_DDL)

    @staticmethod
    def _state(cx: sqlite3.Connection, key: str) -> str | None:
        row = cx.execute(
            "SELECT value FROM graph_projection_state WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row["value"]) if row is not None else None

    @staticmethod
    def _set_state(cx: sqlite3.Connection, key: str, value: str) -> None:
        cx.execute(
            "INSERT INTO graph_projection_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @staticmethod
    def _load_raw_evidence(
        cx: sqlite3.Connection,
        memories: Mapping[str, ProjectionMemoryState],
    ) -> list[RawEntityEvidence]:
        alias_counts = {
            int(row["canonical_id"]): int(row["n"])
            for row in cx.execute(
                "SELECT canonical_id, COUNT(*) AS n FROM entity_aliases GROUP BY canonical_id"
            )
        }
        rows = cx.execute(
            "SELECT e.id, e.name, e.type, e.first_seen, e.last_seen, "
            "em.memory_id, em.extractor, em.confidence "
            "FROM entities e LEFT JOIN entity_memory em ON em.entity_id = e.id "
            "ORDER BY e.id, em.memory_id"
        ).fetchall()
        grouped: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            grouped[int(row["id"])].append(row)

        evidence: list[RawEntityEvidence] = []
        for entity_id, members in sorted(grouped.items()):
            first = members[0]
            linked = [row for row in members if row["memory_id"] in memories]
            evidence.append(
                RawEntityEvidence(
                    entity_id=entity_id,
                    name=str(first["name"]),
                    entity_type=str(first["type"]),
                    extractors=tuple(str(row["extractor"]) for row in linked),
                    confidences=tuple(float(row["confidence"]) for row in linked),
                    memories=tuple(memories[str(row["memory_id"])] for row in linked),
                    alias_count=alias_counts.get(entity_id, 0),
                    first_seen=first["first_seen"],
                    last_seen=first["last_seen"],
                )
            )
        return evidence

    @staticmethod
    def _source_fingerprint(
        raw: list[RawEntityEvidence],
        memories: Mapping[str, ProjectionMemoryState],
        config: ProjectionBuildConfig,
    ) -> str:
        payload = {
            "config": asdict(config),
            "evidence": [asdict(item) for item in raw],
            "code_refs": {
                memory_id: [asdict(ref) for ref in state.code_refs]
                for memory_id, state in sorted(memories.items())
                if state.code_refs and not state.forgotten
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _build_rows(
        raw: list[RawEntityEvidence],
        memories: Mapping[str, ProjectionMemoryState],
        config: ProjectionBuildConfig,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        accepted, rejections = _curate_entity_rows(raw, config)
        code_link_rows = _merge_code_reference_rows(accepted, memories)
        membership_rows, memory_nodes = _materialize_memberships(accepted, memories, config)
        edge_rows, degrees = _materialize_edges(memory_nodes, config.evidence_limit)
        node_rows = _finalize_node_rows(accepted, degrees)
        return node_rows, membership_rows, edge_rows, rejections, code_link_rows

    @staticmethod
    def _validate_version(cx: sqlite3.Connection, version: str) -> None:
        row = cx.execute(
            "SELECT node_count, edge_count, rejected_count "
            "FROM graph_projection_versions WHERE version = ?",
            (version,),
        ).fetchone()
        if row is None:
            raise ProjectionBuildError("projection version row missing")
        actual = (
            cx.execute(
                "SELECT COUNT(*) FROM graph_projection_nodes WHERE version = ?",
                (version,),
            ).fetchone()[0],
            cx.execute(
                "SELECT COUNT(*) FROM graph_projection_edges WHERE version = ?",
                (version,),
            ).fetchone()[0],
            cx.execute(
                "SELECT COUNT(*) FROM graph_projection_rejections WHERE version = ?",
                (version,),
            ).fetchone()[0],
        )
        expected = (row["node_count"], row["edge_count"], row["rejected_count"])
        if actual != expected:
            raise ProjectionBuildError(
                f"projection row-count mismatch: expected={expected}, actual={actual}"
            )
        dangling = cx.execute(
            "SELECT COUNT(*) FROM graph_projection_edges e "
            "LEFT JOIN graph_projection_nodes a "
            "ON a.version = e.version AND a.uri = e.a_uri "
            "LEFT JOIN graph_projection_nodes b "
            "ON b.version = e.version AND b.uri = e.b_uri "
            "WHERE e.version = ? AND (a.uri IS NULL OR b.uri IS NULL)",
            (version,),
        ).fetchone()[0]
        if dangling:
            raise ProjectionBuildError(f"projection has {dangling} dangling edges")
        dangling_code = cx.execute(
            "SELECT COUNT(*) FROM graph_projection_code_links link "
            "LEFT JOIN graph_projection_nodes node "
            "ON node.version = link.version AND node.uri = link.uri "
            "WHERE link.version = ? AND node.uri IS NULL",
            (version,),
        ).fetchone()[0]
        if dangling_code:
            raise ProjectionBuildError(f"projection has {dangling_code} dangling code links")

    @staticmethod
    def _retain_active_and_previous(
        cx: sqlite3.Connection,
        active: str,
        previous: str | None,
    ) -> None:
        keep = [active]
        if previous and previous != active:
            keep.append(previous)
        placeholders = ",".join("?" for _ in keep)
        for table in (
            "graph_projection_memberships",
            "graph_projection_code_links",
            "graph_projection_edges",
            "graph_projection_nodes",
            "graph_projection_rejections",
            "graph_projection_versions",
        ):
            cx.execute(
                f"DELETE FROM {table} WHERE version NOT IN ({placeholders})",  # noqa: S608
                keep,
            )

    def rebuild(
        self,
        memories: Mapping[str, ProjectionMemoryState],
        config: ProjectionBuildConfig,
        now: datetime | None = None,
    ) -> ProjectionBuildResult:
        built_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        version = uuid.uuid4().hex
        try:
            with self._tx_factory() as cx:
                raw = self._load_raw_evidence(cx, memories)
                node_rows, memberships, edge_rows, rejections, code_links = self._build_rows(
                    raw,
                    memories,
                    config,
                )
                cx.execute(
                    "INSERT INTO graph_projection_versions "
                    "(version, status, built_at, source_fingerprint, total_memories, "
                    "node_count, edge_count, rejected_count) VALUES (?, 'ready', ?, ?, ?, ?, ?, ?)",
                    (
                        version,
                        built_at,
                        self._source_fingerprint(raw, memories, config),
                        sum(not memory.forgotten for memory in memories.values()),
                        len(node_rows),
                        len(edge_rows),
                        len(rejections),
                    ),
                )
                for node in node_rows:
                    cx.execute(
                        "INSERT INTO graph_projection_nodes "
                        "(version, uri, entity_id, entity_type, canonical_key, label, "
                        "doc_freq, degree, quality, is_hub, idf) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            version,
                            node["uri"],
                            node["entity_id"],
                            node["entity_type"],
                            node["canonical_key"],
                            node["label"],
                            node["doc_freq"],
                            node["degree"],
                            node["quality"],
                            int(node["is_hub"]),
                            node["idf"],
                        ),
                    )
                for membership in memberships:
                    cx.execute(
                        "INSERT INTO graph_projection_memberships "
                        "(version, memory_id, uri, confidence, evidence_id) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            version,
                            membership["memory_id"],
                            membership["uri"],
                            membership["confidence"],
                            membership["evidence_id"],
                        ),
                    )
                for edge in edge_rows:
                    cx.execute(
                        "INSERT INTO graph_projection_edges "
                        "(version, a_uri, b_uri, relation, weight, confidence, "
                        "evidence_count, first_seen, last_seen, evidence_ids_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            version,
                            edge["a_uri"],
                            edge["b_uri"],
                            edge["relation"],
                            edge["weight"],
                            edge["confidence"],
                            edge["evidence_count"],
                            edge["first_seen"],
                            edge["last_seen"],
                            json.dumps(edge["evidence_ids"]),
                        ),
                    )
                for link in code_links:
                    cx.execute(
                        "INSERT INTO graph_projection_code_links "
                        "(version, memory_id, uri, relation, repo_id, stable_symbol_id, kind, "
                        "label, qualified_name, file_path, start_line, end_line, confidence, "
                        "evidence_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            version,
                            link["memory_id"],
                            link["uri"],
                            link["relation"],
                            link["repo_id"],
                            link["stable_symbol_id"],
                            link["kind"],
                            link["label"],
                            link["qualified_name"],
                            link["file_path"],
                            link["start_line"],
                            link["end_line"],
                            link["confidence"],
                            link["evidence_id"],
                        ),
                    )
                for rejection in rejections:
                    cx.execute(
                        "INSERT INTO graph_projection_rejections "
                        "(version, entity_id, candidate_uri, quality, reason) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            version,
                            rejection["entity_id"],
                            rejection["uri"],
                            rejection["quality"],
                            rejection["reason"],
                        ),
                    )
                self._validate_version(cx, version)
                previous = self._state(cx, "active_version")
                self._set_state(cx, "active_version", version)
                self._set_state(cx, "dirty", "0")
                self._set_state(cx, "last_success_at", built_at)
                self._set_state(cx, "last_error", "")
                self._retain_active_and_previous(cx, version, previous)
        except (
            ProjectionBuildError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            sqlite3.Error,
        ) as exc:
            failed_at = datetime.now(UTC).isoformat()
            with self._tx_factory() as cx:
                self._set_state(cx, "last_error", f"{type(exc).__name__}: {exc}")
                self._set_state(cx, "last_failed_at", failed_at)
            raise
        return ProjectionBuildResult(
            version=version,
            activated=True,
            node_count=len(node_rows),
            edge_count=len(edge_rows),
            rejected_count=len(rejections),
            built_at=built_at,
            code_node_count=sum(node["uri"].startswith("codegraph://") for node in node_rows),
            code_link_count=len(code_links),
        )

    def read_model(
        self,
        max_age_hours: int,
        now: datetime | None = None,
    ) -> GraphReadModel:
        active = self._state(self._conn, "active_version")
        if not active:
            return GraphReadModel.unavailable("projection_missing")
        try:
            version_row = self._conn.execute(
                "SELECT built_at, total_memories, status FROM graph_projection_versions "
                "WHERE version = ?",
                (active,),
            ).fetchone()
            if version_row is None or version_row["status"] != "ready":
                return GraphReadModel.unavailable("projection_missing")
            built_at = datetime.fromisoformat(str(version_row["built_at"]))
            if built_at.tzinfo is None:
                built_at = built_at.replace(tzinfo=UTC)
            age = (now or datetime.now(UTC)).astimezone(UTC) - built_at.astimezone(UTC)
            if age.total_seconds() > max(0, max_age_hours) * 3600:
                return GraphReadModel.unavailable("projection_stale")

            nodes = {
                str(row["uri"]): ProjectedNode(
                    uri=str(row["uri"]),
                    label=str(row["label"]),
                    entity_type=str(row["entity_type"]),
                    canonical_key=str(row["canonical_key"]),
                    doc_freq=int(row["doc_freq"]),
                    degree=int(row["degree"]),
                    quality=float(row["quality"]),
                    is_hub=bool(row["is_hub"]),
                    idf=float(row["idf"]),
                )
                for row in self._conn.execute(
                    "SELECT * FROM graph_projection_nodes WHERE version = ?",
                    (active,),
                )
            }
            memberships: dict[str, list[str]] = defaultdict(list)
            for row in self._conn.execute(
                "SELECT memory_id, uri FROM graph_projection_memberships "
                "WHERE version = ? ORDER BY memory_id, uri",
                (active,),
            ):
                memberships[str(row["memory_id"])].append(str(row["uri"]))
            code_links = tuple(
                ProjectedCodeLink(
                    memory_id=str(row["memory_id"]),
                    uri=str(row["uri"]),
                    relation=str(row["relation"]),
                    repo_id=str(row["repo_id"]),
                    stable_symbol_id=str(row["stable_symbol_id"]),
                    kind=str(row["kind"]),
                    label=str(row["label"]),
                    qualified_name=str(row["qualified_name"]),
                    file_path=str(row["file_path"]),
                    start_line=(int(row["start_line"]) if row["start_line"] is not None else None),
                    end_line=(int(row["end_line"]) if row["end_line"] is not None else None),
                    confidence=float(row["confidence"]),
                    evidence_id=str(row["evidence_id"]),
                )
                for row in self._conn.execute(
                    "SELECT * FROM graph_projection_code_links WHERE version = ? "
                    "ORDER BY memory_id, uri, relation",
                    (active,),
                )
            )
            edges: dict[str, list[ProjectedEdge]] = defaultdict(list)
            for row in self._conn.execute(
                "SELECT * FROM graph_projection_edges WHERE version = ? "
                "ORDER BY a_uri, b_uri, relation",
                (active,),
            ):
                evidence = json.loads(str(row["evidence_ids_json"]))
                if not isinstance(evidence, list) or not all(
                    isinstance(value, str) for value in evidence
                ):
                    raise ValueError("malformed projection edge evidence")
                edge = ProjectedEdge(
                    source_uri=str(row["a_uri"]),
                    target_uri=str(row["b_uri"]),
                    relation=str(row["relation"]),
                    weight=float(row["weight"]),
                    confidence=float(row["confidence"]),
                    evidence_ids=tuple(evidence),
                    first_seen=row["first_seen"],
                    last_seen=row["last_seen"],
                )
                edges[edge.source_uri].append(edge)
                edges[edge.target_uri].append(edge)
            return GraphReadModel(
                available=True,
                version=active,
                built_at=str(version_row["built_at"]),
                total_memories=int(version_row["total_memories"]),
                nodes=nodes,
                memberships={key: tuple(value) for key, value in memberships.items()},
                edges={key: tuple(value) for key, value in edges.items()},
                code_links=code_links,
            )
        except (KeyError, TypeError, ValueError, sqlite3.Error, json.JSONDecodeError):
            return GraphReadModel.unavailable("projection_malformed")

    def health(self, now: datetime | None = None) -> dict[str, Any]:
        active = self._state(self._conn, "active_version")
        dirty = self._state(self._conn, "dirty") != "0"
        last_error = self._state(self._conn, "last_error") or ""
        if not active:
            return {
                "active_version": None,
                "dirty": dirty,
                "built_at": None,
                "age_hours": None,
                "node_count": 0,
                "edge_count": 0,
                "rejected_count": 0,
                "rejection_reasons": {},
                "code_node_count": 0,
                "code_link_count": 0,
                "last_error": last_error,
            }
        row = self._conn.execute(
            "SELECT built_at, node_count, edge_count, rejected_count "
            "FROM graph_projection_versions WHERE version = ?",
            (active,),
        ).fetchone()
        if row is None:
            return {
                "active_version": active,
                "dirty": dirty,
                "built_at": None,
                "age_hours": None,
                "node_count": 0,
                "edge_count": 0,
                "rejected_count": 0,
                "rejection_reasons": {},
                "code_node_count": 0,
                "code_link_count": 0,
                "last_error": last_error,
            }
        built = datetime.fromisoformat(str(row["built_at"]))
        if built.tzinfo is None:
            built = built.replace(tzinfo=UTC)
        age = (now or datetime.now(UTC)).astimezone(UTC) - built.astimezone(UTC)
        reasons = {
            str(reason["reason"]): int(reason["n"])
            for reason in self._conn.execute(
                "SELECT reason, COUNT(*) AS n FROM graph_projection_rejections "
                "WHERE version = ? GROUP BY reason ORDER BY reason",
                (active,),
            )
        }
        code_node_count = self._conn.execute(
            "SELECT COUNT(*) FROM graph_projection_nodes "
            "WHERE version = ? AND uri LIKE 'codegraph://%'",
            (active,),
        ).fetchone()[0]
        code_link_count = self._conn.execute(
            "SELECT COUNT(*) FROM graph_projection_code_links WHERE version = ?",
            (active,),
        ).fetchone()[0]
        return {
            "active_version": active,
            "dirty": dirty,
            "built_at": str(row["built_at"]),
            "age_hours": max(0.0, age.total_seconds() / 3600),
            "node_count": int(row["node_count"]),
            "edge_count": int(row["edge_count"]),
            "rejected_count": int(row["rejected_count"]),
            "rejection_reasons": reasons,
            "code_node_count": int(code_node_count),
            "code_link_count": int(code_link_count),
            "last_error": last_error,
        }


def entity_uri(entity_type: str, name: str) -> str:
    """Return a stable URI for a normalized entity identity."""
    normalized_type = entity_type.strip().lower() or "concept"
    return f"entity://{quote(normalized_type, safe='')}/{quote(fold_key(name), safe='')}"


def memory_uri(memory_id: str) -> str:
    return f"memory://{quote(memory_id.strip(), safe='')}"


def fact_uri(fact_id: str) -> str:
    return f"fact://{quote(fact_id.strip(), safe='')}"


def _is_scalar_or_date(name: str) -> bool:
    return bool(_SCALAR_RE.fullmatch(name.strip()))


def _is_code_shape(name: str) -> bool:
    return bool(_CODE_SHAPE_RE.search(name.strip()))


def evaluate_entity(
    evidence: RawEntityEvidence,
    config: ProjectionBuildConfig,
) -> ProjectionDecision:
    """Apply hard rejection rules and a bounded, deterministic quality score."""
    live = tuple(memory for memory in evidence.memories if not memory.forgotten)
    key = fold_key(evidence.name)
    uri = entity_uri(evidence.entity_type, evidence.name)
    explicit = "explicit" in evidence.extractors

    if not live:
        return ProjectionDecision(False, 0.0, "no_live_memory", uri)
    if not key:
        return ProjectionDecision(False, 0.0, "empty_key", uri)
    if _is_scalar_or_date(evidence.name):
        return ProjectionDecision(False, 0.0, "scalar_or_date", uri)
    if _is_code_shape(evidence.name) and not explicit:
        return ProjectionDecision(False, 0.0, "code_shape", uri)

    confidences = tuple(max(0.0, min(1.0, value)) for value in evidence.confidences)
    average_confidence = sum(confidences) / max(1, len(confidences))
    durable_ratio = sum(
        memory.type.strip().lower() not in _REFERENCE_TYPES for memory in live
    ) / len(live)
    provenance = max(
        (_EXTRACTOR_WEIGHT.get(value, 0.0) for value in evidence.extractors),
        default=0.0,
    )
    quality = min(
        1.0,
        0.35 * average_confidence
        + 0.25 * provenance
        + 0.15 * durable_ratio
        + min(0.2, 0.05 * len(live))
        + (0.05 if evidence.entity_type != "concept" else 0.0),
    )
    if quality < config.min_quality:
        return ProjectionDecision(False, quality, "quality_below_threshold", uri)
    return ProjectionDecision(True, quality, None, uri)


__all__ = [
    "GraphProjectionStore",
    "GraphReadModel",
    "ProjectedCodeLink",
    "ProjectedEdge",
    "ProjectedNode",
    "ProjectionBuildConfig",
    "ProjectionBuildError",
    "ProjectionBuildResult",
    "ProjectionDecision",
    "ProjectionMemoryState",
    "RawEntityEvidence",
    "entity_uri",
    "evaluate_entity",
    "fact_uri",
    "memory_uri",
]
