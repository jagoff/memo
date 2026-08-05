"""Stable provenance and coverage envelopes for code-derived results.

The envelope deliberately separates three questions that were previously
collapsed into a single "indexed" claim:

* what provider generation produced the result;
* which requested paths/scopes were actually covered;
* whether the indexed bytes still match the working tree.

Callers must treat gaps and limitations as data, not as proof that unmentioned
files were covered.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CODE_EVIDENCE_SCHEMA = "memo.code_evidence.v1"
_GAP_LIMIT = 200


@dataclass(frozen=True)
class CoverageGap:
    path: str
    reason: str
    detail: str = ""
    line_start: int | None = None
    line_end: int | None = None


@dataclass(frozen=True)
class CodeEvidenceEnvelope:
    provider: str
    provider_version: str | None
    repo_id: str | None
    commit_sha: str | None
    index_generation: str | None
    indexed_at: str | None
    requested_paths: tuple[str, ...]
    requested_scopes: tuple[str, ...]
    coverage_status: str
    recording_status: str
    freshness: str
    gaps: tuple[CoverageGap, ...] = ()
    limitations: tuple[str, ...] = ()
    schema: str = CODE_EVIDENCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "requested_paths": list(self.requested_paths),
            "requested_scopes": list(self.requested_scopes),
            "gaps": [asdict(gap) for gap in self.gaps],
            "limitations": list(self.limitations),
        }


def normalize_code_path(value: str) -> str:
    """Return a stable repo-relative POSIX path without resolving symlinks."""
    normalized = Path(str(value).strip().replace("\\", "/")).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def code_path_in_scope(path: str, scope: str) -> bool:
    return scope in {"", "."} or path == scope or path.startswith(scope + "/")


def repo_index_generation(
    *,
    repo_id: str,
    commit_sha: str,
    indexed_at: str,
    include: list[str],
    exclude: list[str],
    max_file_bytes: int,
) -> str:
    payload = {
        "schema": CODE_EVIDENCE_SCHEMA,
        "repo_id": repo_id,
        "commit_sha": commit_sha,
        "indexed_at": indexed_at,
        "include": sorted(include),
        "exclude": sorted(exclude),
        "max_file_bytes": int(max_file_bytes),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"repo:{digest[:24]}"


def _git_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _metadata(conn: sqlite3.Connection) -> dict[str, str]:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(project_metadata)")}
    if not {"key", "value"}.issubset(columns):
        return {}
    return {
        str(row["key"]): str(row["value"])
        for row in conn.execute("SELECT key, value FROM project_metadata")
    }


def _indexed_at(value: str | None, db_path: Path) -> str | None:
    if value:
        try:
            number = float(value)
            if number > 10_000_000_000:
                number /= 1000
            return datetime.fromtimestamp(number, tz=UTC).isoformat()
        except (OSError, OverflowError, ValueError):
            if "T" in value:
                return value
    try:
        return datetime.fromtimestamp(db_path.stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        return None


def _file_rows(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(files)")}
    if not {"path", "content_hash"}.issubset(columns):
        return {}
    errors = "errors" if "errors" in columns else "NULL AS errors"
    return {
        normalize_code_path(str(row["path"])): row
        for row in conn.execute(
            f"SELECT path, content_hash, {errors} FROM files ORDER BY path"  # noqa: S608
        )
    }


def _has_errors(raw: Any) -> bool:
    if raw in (None, "", "[]", "{}"):
        return False
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError):
        return True
    return bool(value)


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _codegraph_freshness(
    repo_root: Path,
    files: dict[str, sqlite3.Row],
    expanded: set[str],
) -> tuple[list[CoverageGap], bool, bool]:
    gaps: list[CoverageGap] = []
    stale = False
    checked = False
    for path in sorted(expanded):
        source_path = repo_root / path
        row = files.get(path)
        if row is None:
            reason = "source_missing" if not source_path.is_file() else "not_indexed"
            gaps.append(CoverageGap(path=path, reason=reason))
            continue
        if _has_errors(row["errors"]):
            gaps.append(CoverageGap(path=path, reason="parse_partial"))
        current_hash = _sha256(source_path)
        if current_hash is None:
            gaps.append(CoverageGap(path=path, reason="source_missing"))
            continue
        checked = True
        if current_hash != str(row["content_hash"] or ""):
            stale = True
            gaps.append(CoverageGap(path=path, reason="stale_source"))
    return gaps, stale, checked


def _bounded_gaps(
    gaps: list[CoverageGap],
    limitations: list[str],
) -> tuple[list[CoverageGap], list[str]]:
    if len(gaps) <= _GAP_LIMIT:
        return gaps, limitations
    omitted = len(gaps) - _GAP_LIMIT
    limitations.append(f"{omitted} additional coverage gaps were omitted from this envelope.")
    return gaps[:_GAP_LIMIT], limitations


class CodegraphEvidenceResolver:
    """Reuse one CodeGraph metadata snapshot across many evidence requests.

    Projection rebuilds resolve thousands of memory-to-code links in one pass.
    Loading the full ``files`` table and spawning ``git rev-parse`` per link
    made that pass quadratic in I/O and created thousands of short-lived
    processes.  A resolver is intentionally short-lived: it snapshots the
    provider database and repository HEAD once, then serves bounded path/scope
    envelopes from that stable view.
    """

    def __init__(
        self,
        *,
        db_path: Path,
        repo_root: Path,
        repo_id: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.repo_root = repo_root
        self.repo_id = repo_id
        self.commit_sha = _git_head(repo_root)
        self.metadata: dict[str, str] = {}
        self.files: dict[str, sqlite3.Row] = {}
        self.schema_version = "?"
        self.load_status = "missing"
        self.load_error: str | None = None

        if not db_path.is_file():
            return
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            try:
                self.metadata = _metadata(conn)
                self.files = _file_rows(conn)
                schema_row = conn.execute("SELECT MAX(version) FROM schema_versions").fetchone()
                self.schema_version = (
                    str(schema_row[0]) if schema_row and schema_row[0] is not None else "?"
                )
            finally:
                conn.close()
        except sqlite3.Error as exc:
            self.load_status = "unreadable"
            self.load_error = type(exc).__name__
            return
        self.load_status = "loaded"

    def resolve(
        self,
        *,
        paths: tuple[str, ...] | list[str] = (),
        scopes: tuple[str, ...] | list[str] = (),
    ) -> CodeEvidenceEnvelope:
        """Build one fail-closed envelope from the captured snapshot."""
        requested_paths = tuple(sorted({normalize_code_path(path) for path in paths if path}))
        requested_scopes = tuple(
            sorted({normalize_code_path(scope).rstrip("/") for scope in scopes if scope})
        )
        if self.load_status != "loaded":
            unreadable = self.load_status == "unreadable"
            limitation = (
                f"CodeGraph metadata unreadable: {self.load_error}."
                if unreadable
                else "CodeGraph index is missing."
            )
            return CodeEvidenceEnvelope(
                provider="codegraph",
                provider_version=None,
                repo_id=self.repo_id,
                commit_sha=self.commit_sha,
                index_generation=None,
                indexed_at=None,
                requested_paths=requested_paths,
                requested_scopes=requested_scopes,
                coverage_status="unknown",
                recording_status="unreadable" if unreadable else "missing",
                freshness="unknown",
                limitations=(limitation,),
            )

        expanded = set(requested_paths)
        if requested_scopes:
            for indexed_path in self.files:
                if any(code_path_in_scope(indexed_path, scope) for scope in requested_scopes):
                    expanded.add(indexed_path)

        gaps, stale, checked_freshness = _codegraph_freshness(
            self.repo_root,
            self.files,
            expanded,
        )
        limitations: list[str] = []
        if requested_scopes:
            limitations.append(
                "Scope coverage is bounded to paths recorded by CodeGraph; excluded tracked "
                "files cannot be reconstructed from the provider database."
            )
        if not requested_paths and not requested_scopes:
            limitations.append(
                "No path or scope was requested, so freshness was not checked against source bytes."
            )
        gaps, limitations = _bounded_gaps(gaps, limitations)

        index_state = self.metadata.get("index_state", "")
        discovered = self.metadata.get("files_discovered")
        accounted = self.metadata.get("files_accounted")
        recording = (
            "complete"
            if index_state == "complete" and discovered is not None and discovered == accounted
            else "partial"
        )
        coverage = (
            "known_gaps"
            if gaps
            else ("complete" if expanded or recording == "complete" else "unknown")
        )
        freshness = "stale" if stale else ("current" if checked_freshness else "unknown")
        updated_at = self.metadata.get("updated_at")
        generation = (
            f"codegraph:{self.schema_version}:{updated_at or int(self.db_path.stat().st_mtime_ns)}"
        )
        return CodeEvidenceEnvelope(
            provider="codegraph",
            provider_version=self.metadata.get("indexed_with_version"),
            repo_id=self.repo_id,
            commit_sha=self.commit_sha,
            index_generation=generation,
            indexed_at=_indexed_at(updated_at, self.db_path),
            requested_paths=requested_paths,
            requested_scopes=requested_scopes,
            coverage_status=coverage,
            recording_status=recording,
            freshness=freshness,
            gaps=tuple(gaps),
            limitations=tuple(limitations),
        )


def codegraph_evidence(
    *,
    db_path: Path,
    repo_root: Path,
    repo_id: str | None = None,
    paths: tuple[str, ...] | list[str] = (),
    scopes: tuple[str, ...] | list[str] = (),
) -> CodeEvidenceEnvelope:
    """Build a fail-closed CodeGraph envelope for requested files/scopes."""
    return CodegraphEvidenceResolver(
        db_path=db_path,
        repo_root=repo_root,
        repo_id=repo_id,
    ).resolve(paths=paths, scopes=scopes)


def _repo_coverage_gaps(
    rows: list[dict[str, Any]],
    requested_paths: tuple[str, ...],
    requested_scopes: tuple[str, ...],
) -> list[CoverageGap]:
    selected = [
        row
        for row in rows
        if (
            (not requested_paths and not requested_scopes)
            or row["path"] in requested_paths
            or any(code_path_in_scope(row["path"], scope) for scope in requested_scopes)
        )
    ]
    return [
        CoverageGap(
            path=str(row["path"]),
            reason=str(row["reason"]),
            detail=str(row.get("detail") or ""),
            line_start=row.get("line_start"),
            line_end=row.get("line_end"),
        )
        for row in selected
    ]


def _repo_freshness(
    clone_path: Path,
    file_hashes: dict[str, dict[str, Any]],
    requested_paths: tuple[str, ...],
    gaps: list[CoverageGap],
) -> tuple[list[CoverageGap], bool, bool]:
    stale = False
    checked = False
    gap_paths = {gap.path for gap in gaps}
    for path in requested_paths:
        if path in gap_paths:
            continue
        indexed = file_hashes.get(path)
        current = clone_path / path
        if indexed is None:
            gaps.append(
                CoverageGap(
                    path=path,
                    reason="not_indexed" if current.is_file() else "source_missing",
                )
            )
            continue
        digest = _sha256(current)
        if digest is None:
            gaps.append(CoverageGap(path=path, reason="source_missing"))
            continue
        checked = True
        if digest != str(indexed["sha256"]):
            stale = True
            gaps.append(CoverageGap(path=path, reason="stale_source"))
    return gaps, stale, checked


def repo_corpus_evidence(
    *,
    store: Any,
    repo_id: str | None,
    source: dict[str, Any] | None,
    paths: list[str] | tuple[str, ...] | None = None,
    scopes: list[str] | tuple[str, ...] | None = None,
) -> CodeEvidenceEnvelope:
    """Build an envelope from Memo's persisted repo files and coverage ledger."""
    requested_paths = tuple(sorted({normalize_code_path(path) for path in (paths or ()) if path}))
    requested_scopes = tuple(
        sorted({normalize_code_path(scope).rstrip("/") for scope in (scopes or ()) if scope})
    )
    if source is None or repo_id is None:
        return CodeEvidenceEnvelope(
            provider="memo-repo",
            provider_version=None,
            repo_id=repo_id,
            commit_sha=None,
            index_generation=None,
            indexed_at=None,
            requested_paths=requested_paths,
            requested_scopes=requested_scopes,
            coverage_status="unknown",
            recording_status="missing",
            freshness="unknown",
            limitations=("Repository index is missing.",),
        )

    extra = source.get("extra") or {}
    evidence_meta = extra.get("code_evidence") or {}
    generation = evidence_meta.get("index_generation")
    recording = str(evidence_meta.get("recording_status") or "legacy_unknown")
    rows = store.list_repo_coverage(repo_id, str(generation)) if generation is not None else []
    gaps = _repo_coverage_gaps(rows, requested_paths, requested_scopes)
    file_hashes = store.repo_file_hashes(repo_id)
    clone_path = Path(str(source["clone_path"]))
    gaps, stale, checked = _repo_freshness(
        clone_path,
        file_hashes,
        requested_paths,
        gaps,
    )

    limitations: list[str] = []
    if generation is None:
        limitations.append(
            "This repository predates coverage recording; reindex it to establish coverage."
        )
    if requested_scopes:
        limitations.append(
            "Scope freshness is not byte-checked; request specific paths for byte-level freshness."
        )
    if not requested_paths and not requested_scopes:
        limitations.append(
            "No path or scope was requested, so freshness was not checked against source bytes."
        )
    gaps, limitations = _bounded_gaps(gaps, limitations)
    if source.get("status") == "indexing":
        recording = "indexing"
    coverage = "unknown" if recording != "complete" else ("known_gaps" if gaps else "complete")
    return CodeEvidenceEnvelope(
        provider="memo-repo",
        provider_version="1",
        repo_id=repo_id,
        commit_sha=str(source.get("commit_sha") or "") or None,
        index_generation=str(generation) if generation is not None else None,
        indexed_at=str(source.get("indexed_at") or "") or None,
        requested_paths=requested_paths,
        requested_scopes=requested_scopes,
        coverage_status=coverage,
        recording_status=recording,
        freshness="stale" if stale else ("current" if checked else "unknown"),
        gaps=tuple(gaps),
        limitations=tuple(limitations),
    )


__all__ = [
    "CODE_EVIDENCE_SCHEMA",
    "CodeEvidenceEnvelope",
    "CoverageGap",
    "code_path_in_scope",
    "codegraph_evidence",
    "normalize_code_path",
    "repo_corpus_evidence",
    "repo_index_generation",
]
