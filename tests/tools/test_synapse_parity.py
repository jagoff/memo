"""Parity gate tests for the retired Synapse surface."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner
from tools.memflow_absorption.schemas import (
    CapabilityManifest,
    CapabilityRow,
    OperationMappingRow,
    OperationRoute,
)
from tools.memflow_absorption.synapse_parity import (
    ParityFixture,
    ParityManifestError,
    run_synapse_parity,
)


@dataclass
class _Hit:
    id: str


class _Memory:
    def ask(self, question: str, **_: Any) -> dict[str, object]:
        if question == "disputed query":
            return {
                "answer": "withheld",
                "abstained": "disputed",
                "sources": [{"id": "SOURCE-A"}],
            }
        if question == "error query":
            return {
                "answer": "backend failed",
                "error": "backend failed",
                "sources": [{"id": "SOURCE-A"}],
            }
        if question == "model error query":
            return {
                "answer": "(error querying the model: RuntimeError)",
                "sources": [{"id": "SOURCE-A"}],
            }
        return {"answer": "answered", "sources": [{"id": "SOURCE-A"}]}

    def search(self, query: str, **_: Any) -> list[_Hit]:
        return [_Hit("SOURCE-A")] if query == "native query" else []

    def unified_briefing(self, **_: Any) -> dict[str, object]:
        return {"available": True, "source_ids": ["briefing-a"]}

    def conflict(self, **_: Any) -> dict[str, object]:
        return {"conflicts": [{"evidence_uris": ["conflict-a"]}]}

    def session_list(self, **_: Any) -> list[dict[str, str]]:
        return [{"id": "session-a"}]

    def health(self, **_: Any) -> dict[str, object]:
        return {"source_ids": ["health-a"]}


class _BriefingStore:
    def __init__(self, recent: list[dict[str, str]], pool: list[dict[str, str]]) -> None:
        self.recent = recent
        self.pool = pool

    def list_recent(self, *, limit: int, **_: Any) -> list[dict[str, str]]:
        return self.recent if limit == 20 else self.pool


class _BriefingMemory:
    def __init__(self, store: _BriefingStore, roster: VerificationRoster) -> None:
        self.store = store
        self.capability_manifest_roster = roster

    def get(self, memory_id: str) -> SimpleNamespace | None:
        return SimpleNamespace(id=memory_id) if memory_id else None


def fixture(name: str) -> ParityFixture:
    fixtures = {
        "native": ParityFixture(
            fixture_id="native",
            source_operation="synapse.federate.query",
            query="native query",
            expected_status="answered",
            expected_source_ids=("source-a",),
        ),
        "unmapped": ParityFixture(
            fixture_id="unmapped",
            source_operation="synapse.chat.unmapped",
            query="unmapped query",
            expected_status="answered",
            expected_source_ids=(),
        ),
        "abstain": ParityFixture(
            fixture_id="abstain",
            source_operation="synapse.federate.query",
            query="missing query",
            expected_status="abstained",
            expected_source_ids=(),
        ),
    }
    return fixtures[name]


def _resign(
    manifest: CapabilityManifest,
    keys: DeviceKeyStore,
    roster: VerificationRoster,
) -> CapabilityManifest:
    unsigned = replace(manifest, signature="")
    unsigned = replace(
        unsigned,
        operation_map_sha256=hashlib.sha256(unsigned.operation_map_bytes()).hexdigest(),
        slo_baseline_sha256=hashlib.sha256(unsigned.slo_baseline_bytes()).hexdigest(),
    )
    envelope = OperationalSigner(keys, roster_version=roster.version).sign(
        domain="memo.cutover.capability_manifest.v1",
        payload=unsigned.signed_bytes(),
        key_id=roster.local_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


def _with_route(manifest: CapabilityManifest, route: OperationRoute) -> CapabilityManifest:
    mapping = replace(manifest.operation_mappings[0], routes=(route,))
    capability = replace(manifest.capabilities[0], operation_mappings=(mapping,))
    return replace(manifest, operation_mappings=(mapping,), capabilities=(capability,))


@pytest.fixture
def authority(tmp_path) -> tuple[DeviceKeyStore, VerificationRoster]:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path / "authority",
        pin_store=AuthorityPinStore._for_test(
            tmp_path / "authority", provider=InMemoryAuthorityPinProvider()
        ),
    )
    return keys, roster


@pytest.fixture
def roster(authority) -> VerificationRoster:
    return authority[1]


@pytest.fixture
def manifest(authority, roster) -> CapabilityManifest:
    route = OperationRoute(
        route_id="federated-query",
        predicate={"query": {"present": True}},
        memo_methods=("search",),
        memo_mcp=("memo_ask",),
        memo_cli=("memo ask",),
        parameter_mapping={"query": "query"},
        defaults={},
        result_mapping={"sources": "sources"},
        error_mapping={"not_found": "insufficient_evidence"},
        transform_id="identity",
        fixture_sha256=(),
        atomic_group=None,
    )
    mapping = OperationMappingRow(
        source_operation="synapse.federate.query",
        source_commit="a" * 40,
        source_tests=("eval/native-query.json",),
        evidence_ids=("usage-1",),
        capability="federated_query",
        disposition="memo_native",
        routes=(route,),
        parity_tests=("tests/tools/test_synapse_parity.py",),
        deletion_proof=(),
    )
    capability = CapabilityRow(
        name="federated_query",
        sources=("synapse",),
        consumers=("mcp:synapse_federate_query",),
        window_started_at="2026-01-01T00:00:00Z",
        window_ended_at="2026-04-01T00:00:00Z",
        observed_calls=1,
        observed_daemon_events=0,
        machines=("device-a",),
        evidence_ids=("usage-1",),
        exclusion_counts={},
        evidence_complete=True,
        source_operations=(mapping.source_operation,),
        operation_mappings=(mapping,),
        slo_baseline_ids=(),
        dependencies=(),
        disposition="memo_native",
        memo_target="memo_search",
        parity_tests=mapping.parity_tests,
        deletion_proof=(),
    )
    unsigned = CapabilityManifest(
        schema="memo.cutover_capability_manifest.v1",
        frozen_at="2026-04-01T00:00:00Z",
        window_started_at="2026-01-01T00:00:00Z",
        window_ended_at="2026-04-01T00:00:00Z",
        machine_ids=("device-a",),
        source_receipt_sha256={},
        capabilities=(capability,),
        operation_mappings=(mapping,),
        slo_baselines=(),
        operation_map_sha256="",
        slo_baseline_sha256="",
        blockers=(),
        frozen=True,
        signer_device_id="device-a",
        signer_key_id=roster.local_key_id,
        roster_version=roster.version,
        signature="",
    )
    return _resign(unsigned, authority[0], roster)


@pytest.fixture
def memory(roster) -> _Memory:
    result = _Memory()
    result.capability_manifest_roster = roster
    return result


def test_native_surface_maps_to_memo_without_synapse_namespace(manifest) -> None:
    row = manifest.by_name("federated_query")
    assert row is not None
    assert row.disposition == "memo_native"
    assert row.operation_mappings[0].routes[0].memo_mcp == ("memo_ask",)


def test_parity_report_blocks_unmapped_admitted_operation(manifest, memory) -> None:
    report = run_synapse_parity(manifest, memory, [fixture("unmapped")])
    assert report.status == "blocked"
    assert report.gap_ids == ("unmapped",)


def test_parity_compares_canonicalized_provenance_and_latency(manifest, memory) -> None:
    report = run_synapse_parity(manifest, memory, [fixture("native")])
    assert report.status == "pass", report.rows
    assert report.rows[0].memo_source_ids == ("source-a",)
    assert report.rows[0].provenance_ok is True
    assert report.p50_ms >= 0 and report.p95_ms >= report.p50_ms


def test_parity_requires_matching_abstention(manifest, memory) -> None:
    report = run_synapse_parity(manifest, memory, [fixture("abstain")])
    assert report.status == "pass"
    assert report.rows[0].status == "pass"


def test_parity_blocks_a_route_without_an_admitted_usage_receipt(
    manifest, memory, authority, roster
) -> None:
    capability = manifest.by_name("federated_query")
    assert capability is not None
    blocked = _resign(
        replace(
        manifest,
        capabilities=(replace(capability, observed_calls=0),),
        ),
        authority[0],
        roster,
    )

    report = run_synapse_parity(blocked, memory, [fixture("native")])

    assert report.status == "blocked"
    assert "usage receipt" in (report.rows[0].error or "")


def test_parity_resolves_unified_briefing_without_synapse_import(
    manifest, memory, authority, roster
) -> None:
    route = replace(
        manifest.operation_mappings[0].routes[0],
        memo_methods=("unified_briefing",),
        parameter_mapping={},
        predicate={"query": {"present": True}},
    )
    signed = _resign(_with_route(manifest, route), authority[0], roster)
    case = replace(fixture("native"), expected_source_ids=("briefing-a",))

    report = run_synapse_parity(signed, memory, [case])

    assert report.status == "pass"


@pytest.mark.parametrize(
    ("query", "expected_status"),
    [
        ("disputed query", "conflicted"),
        ("error query", "error"),
        ("model error query", "error"),
    ],
)
def test_memo_ask_preserves_native_abstention_and_error_before_sources(
    manifest, memory, authority, roster, query, expected_status
) -> None:
    route = replace(
        manifest.operation_mappings[0].routes[0],
        memo_methods=("ask",),
        parameter_mapping={"query": "question"},
    )
    signed = _resign(_with_route(manifest, route), authority[0], roster)
    case = replace(
        fixture("native"),
        query=query,
        expected_status=expected_status,
    )

    report = run_synapse_parity(signed, memory, [case])

    assert report.status == "pass"
    assert report.rows[0].memo_source_ids == ("source-a",)


def test_unified_briefing_fallback_emits_native_structured_source_ids(
    manifest, authority, roster, mem_with_stub, monkeypatch
) -> None:
    record = mem_with_stub.save(
        title="Briefing provenance",
        content="A native briefing source.",
        auto_project=False,
    )
    mem_with_stub.capability_manifest_roster = roster
    monkeypatch.setattr(
        "memo.briefing.memo_native_briefing_lines",
        lambda *_args, **_kwargs: ["### Open loops", "- source"],
    )
    monkeypatch.setattr("memo.briefing.operational_briefing_lines", lambda *_args, **_kwargs: [])
    route = replace(
        manifest.operation_mappings[0].routes[0],
        memo_methods=("unified_briefing",),
        parameter_mapping={},
    )
    signed = _resign(_with_route(manifest, route), authority[0], roster)
    case = replace(fixture("native"), expected_source_ids=(record.id,))

    report = run_synapse_parity(signed, mem_with_stub, [case])

    assert report.status == "pass"
    assert report.rows[0].memo_source_ids == (record.id,)


def test_unified_briefing_fallback_returns_only_rendered_open_loop_and_day_ids(
    manifest, authority, roster, monkeypatch
) -> None:
    now = datetime.now(tz=UTC).isoformat()
    recent = [{"id": f"open-{index}", "updated": now} for index in range(6)]
    recent.append({"id": "stale", "updated": "2020-01-01T00:00:00+00:00"})
    pool = [
        {"id": f"pool-{index}", "updated": f"2025-01-{(index % 28) + 1:02d}T00:00:00+00:00"}
        for index in range(500)
    ]
    seed = int(
        hashlib.sha256(datetime.now(tz=UTC).strftime("%Y-%m-%d").encode()).hexdigest(),
        16,
    )
    day_id = sorted(pool, key=lambda row: row["updated"])[seed % len(pool)]["id"]
    memory = _BriefingMemory(_BriefingStore(recent, pool), roster)
    monkeypatch.setattr(
        "memo.briefing.memo_native_briefing_lines",
        lambda *_args, **_kwargs: ["### Open loops", "### Memory of the day"],
    )
    monkeypatch.setattr("memo.briefing.operational_briefing_lines", lambda *_args, **_kwargs: [])
    route = replace(
        manifest.operation_mappings[0].routes[0],
        memo_methods=("unified_briefing",),
        parameter_mapping={},
    )
    signed = _resign(_with_route(manifest, route), authority[0], roster)
    case = replace(
        fixture("native"),
        expected_source_ids=tuple([*(f"open-{index}" for index in range(5)), day_id]),
    )

    report = run_synapse_parity(signed, memory, [case])

    assert report.status == "pass", report.rows
    assert "open-5" not in report.rows[0].memo_source_ids
    assert "stale" not in report.rows[0].memo_source_ids


def test_unified_briefing_fallback_excludes_ids_from_empty_section(
    manifest, authority, roster, monkeypatch
) -> None:
    memory = _BriefingMemory(
        _BriefingStore(
            [{"id": "unrendered-durable", "updated": datetime.now(tz=UTC).isoformat()}],
            [],
        ),
        roster,
    )
    memory.operational = SimpleNamespace(
        state=lambda **_kwargs: {
            "focus": {"focus-a": {"id": "rendered-operational"}},
            "handoffs": {},
            "attention": {},
            "conflicts": {},
        }
    )
    monkeypatch.setattr("memo.briefing.memo_native_briefing_lines", lambda *_args: [])
    monkeypatch.setattr(
        "memo.briefing.operational_briefing_lines",
        lambda *_args: ["### Operational continuity", "- focus"],
    )
    route = replace(
        manifest.operation_mappings[0].routes[0],
        memo_methods=("unified_briefing",),
        parameter_mapping={},
    )
    signed = _resign(_with_route(manifest, route), authority[0], roster)
    case = replace(
        fixture("native"),
        expected_source_ids=("rendered-operational",),
    )

    report = run_synapse_parity(signed, memory, [case])

    assert report.status == "pass", report.rows
    assert report.rows[0].memo_source_ids == ("rendered-operational",)


@pytest.mark.parametrize(
    ("method", "expected_status", "expected_source_ids"),
    [
        ("conflict", "conflicted", ("conflict-a",)),
        ("session_list", "answered", ("session-a",)),
        ("health", "answered", ("health-a",)),
    ],
)
def test_parity_compares_native_operational_status_and_provenance(
    manifest, memory, authority, roster, method, expected_status, expected_source_ids
) -> None:
    route = replace(
        manifest.operation_mappings[0].routes[0],
        memo_methods=(method,),
        parameter_mapping={},
    )
    signed = _resign(_with_route(manifest, route), authority[0], roster)
    case = replace(
        fixture("native"),
        expected_status=expected_status,
        expected_source_ids=expected_source_ids,
    )

    report = run_synapse_parity(signed, memory, [case])

    assert report.status == "pass"


def test_parity_selects_route_by_predicate_and_applies_parameter_mapping(
    manifest, memory, authority, roster
) -> None:
    route = replace(
        manifest.operation_mappings[0].routes[0],
        predicate={"mode": {"eq": "native"}},
        defaults={"limit": 1},
        parameter_mapping={"query": "query"},
    )
    signed = _resign(_with_route(manifest, route), authority[0], roster)
    case = replace(fixture("native"), parameters={"mode": "native"})

    assert run_synapse_parity(signed, memory, [case]).status == "pass"
    no_route = replace(case, fixture_id="no-route", parameters={"mode": "other"})
    report = run_synapse_parity(signed, memory, [no_route])
    assert report.status == "blocked"
    assert "predicate applies" in (report.rows[0].error or "")


@pytest.mark.parametrize(
    "invalid",
    [
        lambda manifest: replace(manifest, frozen=False),
        lambda manifest: replace(manifest, operation_map_sha256="0" * 64),
        lambda manifest: replace(manifest, signature="invalid"),
    ],
)
def test_parity_rejects_unverifiable_manifest_before_any_fixture(invalid, manifest, memory) -> None:
    with pytest.raises(ParityManifestError):
        run_synapse_parity(invalid(manifest), memory, [fixture("native")])
