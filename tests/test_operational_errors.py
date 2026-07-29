from __future__ import annotations

from memo.error_contract import (
    MemoErrorEnvelope,
    OperationalError,
    OperationalErrorCode,
)


def test_error_envelope_is_stable() -> None:
    err = OperationalError(
        OperationalErrorCode.SEQUENCE_GAP,
        "origin device-a expected 4",
        retryable=True,
        details={"expected": 4, "actual": 6},
    )
    out = MemoErrorEnvelope.from_error(err, runtime_version="4.4.6", epoch=0)
    assert out.schema == "memo.error.v1"
    assert out.code == "sequence_gap"
    assert out.retryable is True
    assert out.details == {"expected": 4, "actual": 6}
    assert out.to_dict()["runtime_version"] == "4.4.6"


def test_operational_error_codes_are_exact_and_stable() -> None:
    assert {code.value for code in OperationalErrorCode} == {
        "invalid_event",
        "unknown_schema",
        "sequence_gap",
        "anchor_conflict",
        "idempotency_conflict",
        "signature_invalid",
        "key_revoked",
        "expired",
        "not_found",
        "storage_unavailable",
    }
