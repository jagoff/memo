"""M2b: emit ConsciousnessEvent entries to the unified trinity ledger.

Best-effort. Never raises. Disabled when ``consciousness-contracts`` is
not installed or when ``MEMO_EMIT_LEDGER=0`` is set. Default ON when
contracts is available — the ledger costs nothing if no readers, and
M2d's synapse_audit_unified_replay only works when memo emits.

Wire format: ``consciousness_contracts.ConsciousnessEvent`` JSONL to
``~/.local/share/consciousness/ledger/YYYY-MM-DD.jsonl`` (override with
``CONSCIOUSNESS_LEDGER_ROOT``).
"""

from __future__ import annotations

import logging
import secrets
from typing import TYPE_CHECKING, Any

from memo.util import utc_now_iso as _utc_now_iso

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from consciousness_contracts import ConsciousnessEvent, LedgerWriter

try:
    from consciousness_contracts import ConsciousnessEvent, LedgerWriter

    _CONTRACTS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    # Names stay undefined at runtime; every use below is guarded by
    # _CONTRACTS_AVAILABLE, and the TYPE_CHECKING import above gives mypy the
    # real types without a None-fallback that would need `type: ignore`.
    _CONTRACTS_AVAILABLE = False


_writer: LedgerWriter | None = None


def _enabled() -> bool:
    if not _CONTRACTS_AVAILABLE:
        return False
    from memo.flags import flag_bool

    return flag_bool("MEMO_EMIT_LEDGER")


def _get_writer() -> Any:
    """Return a process-wide LedgerWriter (lazy)."""
    global _writer
    if _writer is None and _CONTRACTS_AVAILABLE:
        _writer = LedgerWriter(on_error=lambda exc: _log.debug("ledger write failed: %s", exc))
    return _writer


def emit_event(
    op: str,
    *,
    subject_uri: str,
    trace_id: str = "",
    actor: str = "memo",
    payload: dict[str, Any] | None = None,
    content_hash: str | None = None,
) -> bool:
    """Append one ConsciousnessEvent. Returns False on any failure (never raises).

    ``op`` must be one of the strings declared in
    ``consciousness_contracts.ledger.LedgerOp`` — typed-narrowing happens at
    the contracts package layer; here we forward the string.
    """
    if not _enabled():
        return False
    writer = _get_writer()
    if writer is None:
        return False
    try:
        event = ConsciousnessEvent(
            event_id=secrets.token_hex(16),
            ts=_utc_now_iso(),
            source="memo",
            op=op,  # type: ignore[arg-type]
            subject_uri=subject_uri,
            trace_id=trace_id or None,
            actor=actor,
            payload=dict(payload or {}),
            content_hash=content_hash,
        )
        return writer.emit(event)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("emit_event failed: %s", exc)
        return False


__all__ = ["emit_event"]
