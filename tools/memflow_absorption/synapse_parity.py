"""Offline Synapse-to-Memo parity gate.

Synapse is deliberately not an import dependency here: the signed manifest is
the only authority describing its retired surface.  This runner exercises the
Memo methods named by that authority and reports every unmapped or divergent
fixture as an explicit blocker.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import quantiles
from typing import Literal

from memo.contracts import AnswerStatus
from memo.memory import Memory
from tools.memflow_absorption.schemas import CapabilityManifest, OperationMappingRow


@dataclass(frozen=True)
class ParityFixture:
    fixture_id: str
    source_operation: str
    query: str
    expected_status: str
    expected_source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ParityRow:
    fixture_id: str
    status: str
    latency_ms: float
    memo_source_ids: tuple[str, ...]
    provenance_ok: bool
    error: str | None


@dataclass(frozen=True)
class ParityReport:
    status: Literal["pass", "blocked"]
    rows: tuple[ParityRow, ...]
    gap_ids: tuple[str, ...]
    p50_ms: float
    p95_ms: float


def _canonical_ids(values: Sequence[object]) -> tuple[str, ...]:
    """Canonicalize identifiers before comparing provenance.

    Source IDs are opaque but case-insensitive at the retirement boundary,
    matching the bounded Synapse data importer.  Empty/non-string values are
    not evidence and therefore cannot make a fixture pass.
    """

    return tuple(
        sorted(
            {
                item.casefold()
                for value in values
                if isinstance(value, str)
                if (item := value.strip())
            }
        )
    )


def _percentile(latencies: Sequence[float], percentile: int) -> float:
    if not latencies:
        return 0.0
    if len(latencies) == 1:
        return round(latencies[0], 3)
    return round(quantiles(latencies, n=100, method="inclusive")[percentile - 1], 3)


def _mapping_for(
    manifest: CapabilityManifest, fixture: ParityFixture
) -> OperationMappingRow | None:
    direct = next(
        (
            row
            for row in manifest.operation_mappings
            if row.source_operation == fixture.source_operation
        ),
        None,
    )
    if direct is not None:
        return direct
    capability = manifest.by_name(fixture.source_operation)
    if capability is None:
        return None
    return next(
        (
            row
            for row in capability.operation_mappings
            if row.source_operation == fixture.source_operation
        ),
        None,
    )


def _admission_error(manifest: CapabilityManifest, mapping: OperationMappingRow) -> str | None:
    """Require the manifest evidence needed to absorb an admitted chat delta."""

    capability = manifest.by_name(mapping.capability)
    if capability is None:
        return "mapping capability is absent from the signed manifest"
    if capability.observed_calls < 1 or not mapping.evidence_ids:
        return "operation has no admitted usage receipt"
    if not capability.memo_target.strip():
        return "operation has no named Memo target"
    return None


def _latency_limit(manifest: CapabilityManifest, mapping: OperationMappingRow) -> float | None:
    capability = manifest.by_name(mapping.capability)
    if capability is None:
        return None
    baselines = [
        baseline
        for baseline in manifest.slo_baselines
        if baseline.baseline_id in capability.slo_baseline_ids
    ]
    if not baselines:
        return None
    return max(
        baseline.visibility_p95_ms * (1.0 + baseline.tolerance_ratio) for baseline in baselines
    )


def _fixture_expects_status(expected: str, actual: str) -> bool:
    normalized = expected.strip().casefold()
    if normalized in {"abstain", "abstained", "insufficient_evidence"}:
        return actual == AnswerStatus.INSUFFICIENT_EVIDENCE.value
    if normalized in {"answer", "answered", "pass"}:
        return actual == AnswerStatus.ANSWERED.value
    return actual == normalized


def _source_ids(value: object) -> tuple[str, ...]:
    rows = value.get("sources", ()) if isinstance(value, dict) else getattr(value, "items", ())
    if not isinstance(rows, Sequence):
        return ()
    ids: list[object] = []
    for row in rows:
        if isinstance(row, dict):
            ids.append(row.get("id"))
        else:
            ids.append(getattr(row, "id", None))
    return _canonical_ids(ids)


class _MemoFacade:
    """Small adapter over Memo's public memory operations.

    The facade keeps the runner independent from server registration while
    exercising the same native operations exposed through MCP.  Methods such
    as briefing/session/conflict/health are intentionally not synthesized:
    their manifest routes must name a callable Memo facade method, or parity is
    blocked instead of silently falling back.
    """

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def call(self, method: str, query: str) -> tuple[str, tuple[str, ...]]:
        method = {
            "memo_ask": "ask",
            "memo_search": "search",
            "memo_evidence_pack": "evidence_pack",
        }.get(method, method.removeprefix("Memory."))
        if method == "ask":
            answer = self.memory.ask(query, include_repos=False)
            return (
                AnswerStatus.ANSWERED.value
                if answer.get("sources")
                else AnswerStatus.INSUFFICIENT_EVIDENCE.value,
                _source_ids(answer),
            )
        if method == "search":
            records = self.memory.search(query, limit=8, quality_rerank=True)
            return (
                AnswerStatus.ANSWERED.value
                if records
                else AnswerStatus.INSUFFICIENT_EVIDENCE.value,
                _source_ids({"sources": records}),
            )
        if method == "evidence_pack":
            pack = self.memory.evidence_pack(query)
            return str(pack.status), _source_ids(pack)
        target = getattr(self.memory, method, None)
        if not callable(target):
            raise LookupError(f"Memo facade has no native method: {method}")
        result = target(query)
        return AnswerStatus.ANSWERED.value, _source_ids(result)


def run_synapse_parity(
    manifest: CapabilityManifest,
    memo: Memory,
    fixtures: Sequence[ParityFixture],
) -> ParityReport:
    """Compare admitted fixtures with their signed Memo-native routes.

    A report blocks if a fixture lacks a mapping, has no executable Memo
    method, returns an unexpected answer/abstention state, or loses expected
    provenance.  All failures stay in ``gap_ids``; none are masked by a
    fallback to retired Synapse code.
    """

    facade = _MemoFacade(memo)
    rows: list[ParityRow] = []
    gaps: list[str] = []
    for fixture in fixtures:
        started = time.perf_counter()
        mapping = _mapping_for(manifest, fixture)
        admission_error = _admission_error(manifest, mapping) if mapping is not None else None
        if (
            mapping is None
            or mapping.disposition not in {"memo_native", "absorb"}
            or admission_error is not None
        ):
            elapsed = (time.perf_counter() - started) * 1000
            rows.append(
                ParityRow(
                    fixture_id=fixture.fixture_id,
                    status="blocked",
                    latency_ms=round(elapsed, 3),
                    memo_source_ids=(),
                    provenance_ok=False,
                    error=admission_error or "no admitted Memo-native operation mapping",
                )
            )
            gaps.append(fixture.fixture_id)
            continue
        methods = tuple(
            method
            for route in mapping.routes
            for method in (route.memo_methods or route.memo_mcp)
        )
        if not methods:
            elapsed = (time.perf_counter() - started) * 1000
            rows.append(
                ParityRow(
                    fixture_id=fixture.fixture_id,
                    status="blocked",
                    latency_ms=round(elapsed, 3),
                    memo_source_ids=(),
                    provenance_ok=False,
                    error="mapping has no Memo method",
                )
            )
            gaps.append(fixture.fixture_id)
            continue
        try:
            status, source_ids = facade.call(methods[0], fixture.query)
            expected_ids = _canonical_ids(fixture.expected_source_ids)
            provenance_ok = expected_ids == source_ids
            status_ok = _fixture_expects_status(fixture.expected_status, status)
            error = None
            if not status_ok:
                error = f"expected status {fixture.expected_status!r}, got {status!r}"
            elif not provenance_ok:
                error = f"expected source ids {expected_ids!r}, got {source_ids!r}"
        except Exception as exc:
            status, source_ids, provenance_ok = "blocked", (), False
            error = f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter() - started) * 1000
        limit = _latency_limit(manifest, mapping)
        if error is None and limit is not None and elapsed > limit:
            error = f"latency {elapsed:.3f}ms exceeds admitted p95 limit {limit:.3f}ms"
        row_status = "pass" if error is None else "blocked"
        rows.append(
            ParityRow(
                fixture_id=fixture.fixture_id,
                status=row_status,
                latency_ms=round(elapsed, 3),
                memo_source_ids=source_ids,
                provenance_ok=provenance_ok,
                error=error,
            )
        )
        if error is not None:
            gaps.append(fixture.fixture_id)
    latencies = [row.latency_ms for row in rows]
    return ParityReport(
        status="blocked" if gaps else "pass",
        rows=tuple(rows),
        gap_ids=tuple(dict.fromkeys(gaps)),
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
    )


__all__ = ["ParityFixture", "ParityReport", "ParityRow", "run_synapse_parity"]
