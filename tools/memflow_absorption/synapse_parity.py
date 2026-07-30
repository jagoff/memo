"""Offline Synapse-to-Memo parity gate.

Synapse is deliberately not an import dependency here: the signed manifest is
the only authority describing its retired surface.  This runner exercises the
Memo methods named by that authority and reports every unmapped or divergent
fixture as an explicit blocker.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import quantiles
from typing import Any, Literal

from memo.contracts import AnswerStatus
from memo.memory import Memory
from memo.operational_roster import VerificationRoster
from tools.memflow_absorption.manifest import ManifestError, verify_capability_manifest
from tools.memflow_absorption.schemas import (
    CapabilityManifest,
    OperationMappingRow,
    OperationRoute,
)


class ParityManifestError(RuntimeError):
    """The parity authority cannot be verified before executing fixtures."""


_ASK_MODEL_ERROR_RE = re.compile(r"^\(error querying the model: [^)]+\)$")


@dataclass(frozen=True)
class ParityFixture:
    fixture_id: str
    source_operation: str
    query: str
    expected_status: str
    expected_source_ids: tuple[str, ...]
    parameters: Mapping[str, object] = field(default_factory=dict)


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
    if normalized in {"conflict", "conflicted"}:
        return actual == AnswerStatus.CONFLICTED.value
    if normalized in {"answer", "answered", "pass"}:
        return actual == AnswerStatus.ANSWERED.value
    return actual == normalized


def _source_ids(value: object) -> tuple[str, ...]:
    """Extract only native provenance/source identifiers from a result payload."""

    if isinstance(value, Mapping):
        rows = value.get("sources", value.get("items", ()))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = value
    else:
        rows = getattr(value, "items", ())
    ids: list[object] = []
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if isinstance(row, Mapping):
                ids.append(row.get("id"))
            else:
                ids.append(getattr(row, "id", None))
    if isinstance(value, Mapping):
        for row in value.get("conflicts", ()):
            if isinstance(row, Mapping):
                ids.extend(row.get("evidence_uris", ()))
        ids.extend(value.get("source_ids", ()))
    return _canonical_ids(ids)


def _native_status(value: object) -> str:
    if value is None:
        return AnswerStatus.INSUFFICIENT_EVIDENCE.value
    if isinstance(value, Mapping):
        raw = str(value.get("status") or "").casefold()
        if raw in {status.value for status in AnswerStatus}:
            return raw
        if value.get("error"):
            return AnswerStatus.ERROR.value
        if value.get("conflicts") or value.get("conflict"):
            return AnswerStatus.CONFLICTED.value
        if value.get("available") is False or value.get("found") is False:
            return AnswerStatus.INSUFFICIENT_EVIDENCE.value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return AnswerStatus.ANSWERED.value if value else AnswerStatus.INSUFFICIENT_EVIDENCE.value
    return AnswerStatus.ANSWERED.value


def _ask_status(value: Mapping[str, object]) -> str:
    """Keep Memo's explicit abstention/error semantics ahead of source count."""

    status = _native_status(value)
    if status != AnswerStatus.ANSWERED.value:
        return status
    answer = value.get("answer")
    if isinstance(answer, str) and _ASK_MODEL_ERROR_RE.fullmatch(answer.strip()):
        return AnswerStatus.ERROR.value
    abstained = value.get("abstained")
    if abstained:
        return (
            AnswerStatus.CONFLICTED.value
            if str(abstained).casefold() == "disputed"
            else AnswerStatus.INSUFFICIENT_EVIDENCE.value
        )
    return status


def _briefing_source_ids(memory: Memory, cwd: str | None) -> tuple[str, ...]:
    """Return precisely the native record IDs rendered by briefing sections."""

    ids: list[object] = []
    try:
        cutoff = (datetime.now(tz=UTC) - timedelta(days=7)).isoformat()
        recent = memory.store.list_recent(
            limit=20,
            exclude_types={"reference", "secret"},
        )
        open_loops = [
            row
            for row in recent
            if isinstance(row, Mapping) and str(row.get("updated") or "") >= cutoff
        ][:5]
        ids.extend(row.get("id") for row in open_loops)

        pool = memory.store.list_recent(
            limit=500,
            exclude_types={"reference", "secret"},
        )
        if pool:
            ordered = sorted(
                (row for row in pool if isinstance(row, Mapping)),
                key=lambda row: str(row.get("updated") or ""),
            )
            if ordered:
                seed = int(hashlib.sha256(datetime.now(tz=UTC).strftime("%Y-%m-%d").encode()).hexdigest(), 16)
                selected = ordered[seed % len(ordered)]
                selected_id = str(selected.get("id") or "")
                record = memory.get(selected_id) if selected_id else None
                if record is not None:
                    ids.append(getattr(record, "id", selected_id))
    except Exception:
        return _canonical_ids(ids)
    try:
        project = Path(cwd).resolve().name if cwd else None
        state = memory.operational.state(project=project)
        for row in list(state.get("focus", {}).values())[:3]:
            if isinstance(row, Mapping):
                ids.append(row.get("id"))
        for row in [row for row in state.get("handoffs", {}).values() if not row.get("consumed_at")][:3]:
            if isinstance(row, Mapping):
                ids.append(row.get("id"))
        for row in [row for row in state.get("attention", {}).values() if not row.get("acknowledged_at")][:3]:
            if isinstance(row, Mapping):
                ids.append(row.get("id"))
        for row in [
            row
            for row in state.get("conflicts", {}).values()
            if row.get("lifecycle_state") not in {"resolved", "archived"}
        ][:3]:
            if isinstance(row, Mapping):
                ids.append(row.get("id"))
    except Exception:
        return _canonical_ids(ids)
    return _canonical_ids(ids)


def _route_input(fixture: ParityFixture) -> dict[str, object]:
    values = dict(fixture.parameters)
    values.setdefault("query", fixture.query)
    return values


def _route_applies(route: OperationRoute, values: Mapping[str, object]) -> bool:
    for name, matcher in route.predicate.items():
        if not isinstance(matcher, Mapping):
            return False
        operator, operand = next(iter(matcher.items()))
        present = name in values and values[name] is not None
        value = values.get(name)
        if operator == "present" and present != operand:
            return False
        if operator == "eq" and (not present or value != operand):
            return False
        if operator == "in" and (not present or value not in operand):
            return False
    return True


def _route_arguments(route: OperationRoute, values: Mapping[str, object]) -> dict[str, object]:
    arguments = dict(route.defaults)
    for source_name, memo_name in route.parameter_mapping.items():
        if source_name in values:
            arguments[memo_name] = values[source_name]
    return arguments


def _verified_manifest(manifest: CapabilityManifest, memo: Memory, roster: VerificationRoster | None) -> None:
    resolved_roster = roster or getattr(memo, "capability_manifest_roster", None)
    if not isinstance(resolved_roster, VerificationRoster):
        raise ParityManifestError("a verification roster is required for Synapse parity")
    try:
        verify_capability_manifest(manifest, roster=resolved_roster)
    except ManifestError as exc:
        raise ParityManifestError(f"unverifiable capability manifest: {exc}") from exc


class _MemoFacade:
    """Small adapter over Memo's public memory operations.

    The facade keeps the runner independent from server registration while
    exercising the same native operations exposed through MCP.  It implements
    the read-only operational surfaces directly from Memo modules; retired
    Synapse is never imported as a fallback.
    """

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def call(
        self, method: str, arguments: Mapping[str, object]
    ) -> tuple[str, tuple[str, ...]]:
        method = {
            "memo_ask": "ask",
            "memo_search": "search",
            "memo_evidence_pack": "evidence_pack",
            "memo_unified_briefing": "unified_briefing",
            "memo_operational_state": "conflict",
            "memo_session_list": "session_list",
            "memo_session_get": "session_get",
            "memo_health_summary": "health",
            "memo_health_report": "health",
        }.get(method, method.removeprefix("Memory."))
        kwargs: dict[str, Any] = dict(arguments)
        if method == "ask":
            question = str(kwargs.pop("question", kwargs.pop("query", "")))
            kwargs.setdefault("include_repos", False)
            answer = self.memory.ask(question, **kwargs)
            return _ask_status(answer), _source_ids(answer)
        if method == "search":
            query = str(kwargs.pop("query", kwargs.pop("question", "")))
            records = self.memory.search(query, **kwargs)
            return (
                AnswerStatus.ANSWERED.value
                if records
                else AnswerStatus.INSUFFICIENT_EVIDENCE.value,
                _source_ids({"sources": records}),
            )
        if method == "evidence_pack":
            question = str(kwargs.pop("question", kwargs.pop("query", "")))
            pack = self.memory.evidence_pack(question, **kwargs)
            return str(pack.status), _source_ids(pack)
        if method == "unified_briefing":
            target = getattr(self.memory, "unified_briefing", None)
            if callable(target):
                result = target(**kwargs)
            else:
                from memo.briefing import (
                    compact_text,
                    memo_native_briefing_lines,
                    operational_briefing_lines,
                )

                cwd = kwargs.get("cwd")
                cwd_value = cwd if isinstance(cwd, str) else None
                lines = [
                    *memo_native_briefing_lines(self.memory),
                    *operational_briefing_lines(self.memory, cwd_value),
                ]
                result = {
                    "available": bool(lines),
                    "markdown": compact_text("\n".join(lines), max_chars=900),
                    "lines": lines,
                    "source_ids": list(_briefing_source_ids(self.memory, cwd_value)) if lines else [],
                }
            return _native_status(result), _source_ids(result)
        if method == "conflict":
            target = getattr(self.memory, "conflict", None)
            if callable(target):
                result = target(**kwargs)
            else:
                project = kwargs.get("project")
                result = self.memory.operational.state(
                    project if isinstance(project, str) else None
                )
            return _native_status(result), _source_ids(result)
        if method in {"session_list", "session_get"}:
            target = getattr(self.memory, method, None)
            if callable(target):
                result = target(**kwargs)
            else:
                from memo.session import get_session, list_sessions

                if method == "session_list":
                    result = list_sessions(self.memory.cfg.state_dir, **kwargs)
                else:
                    session_id = str(kwargs.pop("session_id", kwargs.pop("id", "")))
                    result = get_session(self.memory.cfg.state_dir, session_id)
            return _native_status(result), _source_ids(result)
        if method == "health":
            target = getattr(self.memory, "health", None)
            if callable(target):
                result = target(**kwargs)
            else:
                from memo.health_report import build_health_report

                probe = bool(kwargs.pop("probe_embedder", False))
                result = build_health_report(self.memory, probe_embedder=probe)
            return _native_status(result), _source_ids(result)
        target = getattr(self.memory, method, None)
        if not callable(target):
            raise LookupError(f"Memo facade has no native method: {method}")
        result = target(**kwargs)
        return _native_status(result), _source_ids(result)


def run_synapse_parity(
    manifest: CapabilityManifest,
    memo: Memory,
    fixtures: Sequence[ParityFixture],
    *,
    roster: VerificationRoster | None = None,
) -> ParityReport:
    """Compare admitted fixtures with their signed Memo-native routes.

    A report blocks if a fixture lacks a mapping, has no executable Memo
    method, returns an unexpected answer/abstention state, or loses expected
    provenance.  All failures stay in ``gap_ids``; none are masked by a
    fallback to retired Synapse code.
    """

    _verified_manifest(manifest, memo, roster)
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
        values = _route_input(fixture)
        route = next(
            (item for item in mapping.routes if _route_applies(item, values)),
            None,
        )
        methods = (route.memo_methods or route.memo_mcp) if route is not None else ()
        if not methods:
            elapsed = (time.perf_counter() - started) * 1000
            rows.append(
                ParityRow(
                    fixture_id=fixture.fixture_id,
                    status="blocked",
                    latency_ms=round(elapsed, 3),
                    memo_source_ids=(),
                    provenance_ok=False,
                    error=(
                        "no signed route predicate applies to fixture"
                        if route is None
                        else "mapping has no Memo method"
                    ),
                )
            )
            gaps.append(fixture.fixture_id)
            continue
        try:
            assert route is not None
            status, source_ids = facade.call(methods[0], _route_arguments(route, values))
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
