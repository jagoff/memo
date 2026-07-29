"""Stable serialization contract for operational failures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from memo.errors import OperationalError, OperationalErrorCode


@dataclass(frozen=True)
class MemoErrorEnvelope:
    schema: Literal["memo.error.v1"]
    code: str
    message: str
    retryable: bool
    details: dict[str, Any]
    runtime_version: str
    epoch: int

    @classmethod
    def from_error(
        cls,
        error: OperationalError,
        *,
        runtime_version: str,
        epoch: int,
    ) -> MemoErrorEnvelope:
        return cls(
            schema="memo.error.v1",
            code=error.code.value,
            message=str(error),
            retryable=error.retryable,
            details=dict(error.details),
            runtime_version=str(runtime_version),
            epoch=int(epoch),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["MemoErrorEnvelope", "OperationalError", "OperationalErrorCode"]
