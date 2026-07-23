"""Reproducible readiness gate and local journal microbenchmark."""

from __future__ import annotations

import ast
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from memo.contracts import (
    MEMO_EVENT_SCHEMA,
    MEMO_EVIDENCE_SCHEMA,
    MEMO_FEDERATION_SCHEMA,
    MEMO_OPERATIONAL_SCHEMA,
)
from memo.operation_ledger import OperationLedger

MEMO_DEFINITIVE_SCHEMA = "memo.definitive_readiness.v1"
_RETIRED_IMPORTS = {"consciousness_contracts", "memflow", "synapse"}
_RETIRED_MODULES = {
    "_trace.py",
    "cli_crossdedup.py",
    "consciousness_ledger.py",
    "receipts.py",
    "synapse_backend.py",
    "synapse_client.py",
}


def independence_audit(package_root: Path | None = None) -> dict[str, Any]:
    """Statically prove the runtime has no imports of retired memory packages."""
    root = package_root or Path(__file__).resolve().parent
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            violations.append(f"{path.name}: unreadable ({exc})")
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root_name = name.split(".", 1)[0]
                if root_name in _RETIRED_IMPORTS:
                    violations.append(
                        f"{path.relative_to(root)}:{getattr(node, 'lineno', 0)}: {name}"
                    )
    present = sorted(name for name in _RETIRED_MODULES if (root / name).exists())
    return {
        "ok": not violations and not present,
        "forbidden_imports": violations,
        "retired_modules_present": present,
    }


def definitive_check(memory: Any) -> dict[str, Any]:
    """Check the non-negotiable invariants of an independent Memo runtime."""
    journal = memory.operational.ledger.verify()
    independence = independence_audit()
    contracts = {
        "event": MEMO_EVENT_SCHEMA,
        "evidence": MEMO_EVIDENCE_SCHEMA,
        "operational": MEMO_OPERATIONAL_SCHEMA,
        "federation": MEMO_FEDERATION_SCHEMA,
    }
    capabilities = {
        "evidence_pack": callable(getattr(memory, "evidence_pack", None)),
        "outcome_feedback": callable(getattr(memory, "record_task_outcome", None)),
        "procedural_promotion": callable(getattr(memory, "promote_learning", None)),
        "federation": callable(getattr(memory.federation, "export_bundle", None)),
        "write_policy": callable(getattr(memory.write_policy, "preflight", None)),
    }
    checks = {
        "independent_runtime": bool(independence["ok"]),
        "journal_integrity": bool(journal["ok"]),
        "markdown_authority": memory.cfg.memory_dir.is_dir(),
        "native_contracts": all(value.startswith("memo.") for value in contracts.values()),
        "definitive_capabilities": all(capabilities.values()),
    }
    return {
        "schema": MEMO_DEFINITIVE_SCHEMA,
        "ok": all(checks.values()),
        "checks": checks,
        "contracts": contracts,
        "capabilities": capabilities,
        "journal": journal,
        "independence": independence,
    }


def run_journal_benchmark(
    *,
    events: int = 250,
    min_events_per_second: float = 25.0,
) -> dict[str, Any]:
    """Measure append/verify throughput with a portable deterministic workload."""
    if not 10 <= events <= 10_000:
        raise ValueError("events must be between 10 and 10000")
    if min_events_per_second <= 0:
        raise ValueError("min_events_per_second must be positive")
    latencies_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix="memo-definitive-") as temporary:
        ledger = OperationLedger(Path(temporary), device_id="benchmark-device")
        started = time.perf_counter()
        for index in range(events):
            before = time.perf_counter()
            ledger.append(
                "benchmark.append",
                subject_uri=f"memo://benchmark/{index}",
                payload={"index": index},
            )
            latencies_ms.append((time.perf_counter() - before) * 1000)
        elapsed = time.perf_counter() - started
        verify_started = time.perf_counter()
        verification = ledger.verify()
        verify_ms = (time.perf_counter() - verify_started) * 1000
    ordered = sorted(latencies_ms)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    throughput = events / max(elapsed, 1e-9)
    passed = bool(verification["ok"]) and throughput >= min_events_per_second
    return {
        "schema": "memo.definitive_benchmark.v1",
        "ok": passed,
        "events": events,
        "elapsed_ms": round(elapsed * 1000, 3),
        "events_per_second": round(throughput, 2),
        "append_mean_ms": round(statistics.fmean(latencies_ms), 4),
        "append_p95_ms": round(ordered[p95_index], 4),
        "verify_ms": round(verify_ms, 3),
        "minimum_events_per_second": min_events_per_second,
        "verification": verification,
    }


__all__ = [
    "MEMO_DEFINITIVE_SCHEMA",
    "definitive_check",
    "independence_audit",
    "run_journal_benchmark",
]
