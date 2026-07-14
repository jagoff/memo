"""Public test double for the optional consciousness-contracts interface.

This package is deliberately outside ``src`` and is only placed on PYTHONPATH
by the compatibility test.  It keeps memo's public integration branches
testable without granting public CI access to the private implementation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

TRACE_HEADER = "x-synapse-trace-id"
PROVENANCE_KEYS: frozenset[str] = frozenset(
    {
        "synapse_trace_id",
        "synapse_route_reason",
        "synapse_write_policy_schema",
        "synapse_write_target",
        "synapse_agent_id",
        "synapse_agent_signature",
    }
)
SUPPORTED_AGENTS: tuple[str, ...] = ("claude-code", "codex")

_TRACE: ContextVar[str] = ContextVar("contract_stub_trace", default="")


def current_trace() -> str:
    return _TRACE.get()


@contextmanager
def trace_scope(trace_id: str) -> Iterator[str]:
    normalized = (trace_id or "").strip()
    token = _TRACE.set(normalized)
    try:
        yield normalized
    finally:
        _TRACE.reset(token)


class BackendError(RuntimeError):
    """Contract subprocess failure."""


def run_json(
    binary: str,
    args: Sequence[str],
    *,
    timeout: float,
    env: dict[str, str],
    backend_name: str,
    trace_id: str,
) -> Any:
    try:
        result = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BackendError(f"{backend_name}: {trace_id}: {exc}") from exc
    if result.returncode != 0:
        raise BackendError(f"{backend_name}: exit {result.returncode}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BackendError(f"{backend_name}: invalid JSON") from exc


@dataclass(frozen=True)
class EmbedderProfile:
    model_id: str
    dims: int
    normalization: str = "l2"
    max_seq_len: int | None = None
    quantization: str | None = None
    provider: str = "memo"
    schema: str = field(default="consciousness.embedder_profile.v1", init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EmbedderProfile:
        fields = {key: val for key, val in value.items() if key != "schema"}
        return cls(**fields)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def is_compatible_with(self, other: EmbedderProfile) -> bool:
        return (
            self.model_id == other.model_id
            and self.dims == other.dims
            and self.normalization == other.normalization
        )


@dataclass(frozen=True)
class AgentMcpServer:
    name: str
    command: str
    env: dict[str, str]


@dataclass(frozen=True)
class _GenericPreset:
    config_path: str
    json_key: str = "mcpServers"


def generic_preset(*, config_path: str, json_key: str = "mcpServers") -> _GenericPreset:
    return _GenericPreset(config_path=config_path, json_key=json_key)


def register_agent_mcp(
    agent: str,
    server: AgentMcpServer,
    *,
    write: bool,
    preset: _GenericPreset | None = None,
) -> dict[str, Any]:
    return {"agent": agent, "server": server.name, "write": write, "preset": preset}


class _DictModel:
    schema = ""

    def __init__(self, **values: Any) -> None:
        for key, value in values.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        result = dict(vars(self))
        if self.schema:
            result = {"schema": self.schema, **result}
        return result


class EvidenceRef(_DictModel):
    schema = "consciousness.evidence_ref.v1"


class WriteReceipt(_DictModel):
    schema = "consciousness.write_receipt.v1"


class Anomaly(_DictModel):
    schema = "consciousness.anomaly.v1"


class ConsciousnessEvent(_DictModel):
    schema = "consciousness.event.v1"


class LedgerWriter:
    def __init__(self, *, on_error: Any | None = None) -> None:
        self.on_error = on_error

    def emit(self, event: ConsciousnessEvent) -> bool:
        return bool(event.to_dict().get("event_id"))


def generate_anomaly_id(kind: str, subject: str) -> str:
    digest = hashlib.sha256(f"{kind}:{subject}".encode()).hexdigest()[:24]
    return f"anomaly-{digest}"
