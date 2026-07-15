from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB_ROOT = ROOT / "tests" / "contract_stub"


def test_public_contract_stub_exercises_every_memo_integration_branch() -> None:
    script = r"""
from consciousness_contracts import (
    AgentMcpServer,
    Anomaly,
    BackendError,
    ConsciousnessEvent,
    EmbedderProfile,
    EvidenceRef,
    LedgerWriter,
    PROVENANCE_KEYS,
    TRACE_HEADER,
    WriteReceipt,
    current_trace,
    generate_anomaly_id,
    run_json,
    trace_scope,
)
from consciousness_contracts.cache import get_default_cache
from consciousness_contracts.uri import device_id, is_memo_uri, parse_uri

assert TRACE_HEADER == "x-synapse-trace-id"
with trace_scope("trace-public-contract"):
    assert current_trace() == "trace-public-contract"

profile = EmbedderProfile(model_id="stub-model", dims=8, provider="memo")
assert EmbedderProfile.from_dict(profile.to_dict()).is_compatible_with(profile)
assert profile.fingerprint()
assert PROVENANCE_KEYS
assert AgentMcpServer(name="memo", command="memo-mcp", env={}).name == "memo"
assert issubclass(BackendError, RuntimeError)
assert callable(run_json)

cache = get_default_cache()
cache.put("key", [1.0])
assert cache.get("key") == [1.0]
assert is_memo_uri("memo://memoria/abc")
assert parse_uri("memo://memoria/abc").resource_id == "abc"
assert device_id()

event = ConsciousnessEvent(
    event_id="evt-1",
    ts="2026-07-14T00:00:00+00:00",
    source="memo",
    op="save",
    subject_uri="memo://memoria/abc",
)
assert event.to_dict()["schema"] == "consciousness.event.v1"
assert LedgerWriter().emit(event)
assert EvidenceRef(source="memo", uri="memo://memoria/abc").to_dict()["uri"].endswith("abc")
assert WriteReceipt(backend="memo", receipt_id="abc").to_dict()["receipt_id"] == "abc"
anomaly_id = generate_anomaly_id("semantic_contradiction", "memo:a:b")
assert anomaly_id
assert Anomaly(
    anomaly_id=anomaly_id,
    kind="semantic_contradiction",
    state="detected",
    summary="stub",
    detected_at="2026-07-14T00:00:00+00:00",
    source_backend="memo",
).to_dict()["anomaly_id"] == anomaly_id

import memo._trace as trace_integration
import memo.consciousness_ledger as ledger_integration
import memo.contradict as anomaly_integration
import memo.embedder as cache_integration
import memo.memory.record as provenance_integration
import memo.memory.replay_ops as uri_integration
import memo.synapse_backend as backend_integration
import memo.synapse_client as subprocess_integration

assert trace_integration.HAS_TRACE_CONTEXT
assert ledger_integration._CONTRACTS_AVAILABLE
assert anomaly_integration._HAS_CONFLICT_CONTRACTS
assert cache_integration._HAS_SHARED_CACHE
assert provenance_integration._PROVENANCE_KEYS == PROVENANCE_KEYS
assert uri_integration._HAS_URI_HELPERS
assert backend_integration._CONTRACTS_AVAILABLE
assert subprocess_integration._HAS_CONTRACTS_SUBPROCESS
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(STUB_ROOT), str(ROOT / "src")))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
