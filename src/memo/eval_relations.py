"""Fixed-corpus gate for canonical relation candidate policy.

This evaluator measures deterministic eligibility and namespace boundaries
before embedding/model tuning. The normal recall gate continues to own general
retrieval quality.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memo.memory.record import MemoryRecord
from memo.memory.relation_ops import _allowed_candidate_namespaces, _candidate_eligible

SCHEMA = "memo.eval_relations.v1"


@dataclass(frozen=True)
class RelationEvalResult:
    cases: int
    expected: int
    predicted: int
    true_positive: int
    false_positive: int
    false_negative: int
    recall: float
    noise: float
    precision: float
    elapsed_ms: float
    fingerprint: str
    passed: bool


def _record(payload: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        id=str(payload.get("id") or "fixture"),
        path=str(payload.get("path") or "fixture.md"),
        title=str(payload.get("title") or "fixture"),
        type=str(payload.get("type") or "note"),
        tags=[str(tag) for tag in payload.get("tags") or []],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body=str(payload.get("body") or "fixture"),
    )


def pair_allowed(case: dict[str, Any]) -> bool:
    source = _record(dict(case.get("source") or {}))
    candidate = _record(dict(case.get("candidate") or {}))
    source_namespace = str(case.get("source_namespace") or "_unscoped")
    candidate_namespace = str(case.get("candidate_namespace") or "_unscoped")
    return (
        _candidate_eligible(source)
        and _candidate_eligible(candidate)
        and candidate_namespace in _allowed_candidate_namespaces(source_namespace)
    )


def evaluate(path: Path) -> RelationEvalResult:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") != SCHEMA or not isinstance(payload.get("cases"), list):
        raise ValueError(f"invalid relation label set: {path}")
    cases = payload["cases"]
    started = time.perf_counter()
    tp = fp = fn = predicted = expected = 0
    for case in cases:
        want = bool(case.get("expect_candidate"))
        got = pair_allowed(case)
        expected += int(want)
        predicted += int(got)
        tp += int(want and got)
        fp += int(not want and got)
        fn += int(want and not got)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    recall = tp / expected if expected else 1.0
    precision = tp / predicted if predicted else 1.0
    noise = fp / predicted if predicted else 0.0
    thresholds = payload.get("thresholds") or {}
    passed = recall >= float(thresholds.get("recall_min", 1.0)) and noise <= float(
        thresholds.get("noise_max", 0.0)
    )
    return RelationEvalResult(
        cases=len(cases),
        expected=expected,
        predicted=predicted,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        recall=round(recall, 4),
        noise=round(noise, 4),
        precision=round(precision, 4),
        elapsed_ms=round(elapsed_ms, 3),
        fingerprint=hashlib.sha256(raw).hexdigest()[:16],
        passed=passed,
    )
