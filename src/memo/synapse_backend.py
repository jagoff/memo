"""In-process Synapse backend adapter for memo.

Synapse (the cross-backend orchestrator at
github.com/jagoff/synapse) defines a `SynapseBackend` ABC with
`health()` / `collect()` / `remember()`. Today synapse talks to memo
via subprocess CLI (`memo search --json`, `memo save - --json`).
That works but pays subprocess fork + JSON parse + cold MLX load on
every call.

This adapter exposes the same contract **in-process**: Synapse can
import `MemoSynapseBackend` directly and reuse memo's warm
`Memory()` instance with the recall daemon + cached embedder.

Wire shape mirrors `synapse.models.{MemoryWriteRequest,
MemoryWriteReceipt, EvidenceRef, BackendHealth}` but is encoded as
plain dicts so memo never imports synapse — both projects stay
sovereign. Synapse can adapt the dicts to its dataclasses on its
side via a 3-line wrapper.

Provenance fields (`synapse_trace_id`, `synapse_route_reason`,
`synapse_write_policy_schema`, `synapse_write_target`,
`synapse_agent_id`, `synapse_agent_signature`) ride inside
`request["metadata"]` and are persisted to `meta.extra_json` +
`history.events.delta_json`. Retrievable via `Memory.provenance(id)`.

Example wiring on the synapse side::

    from memo.config import Config
    from memo.memory import Memory
    from memo.synapse_backend import MemoSynapseBackend

    memo_backend = MemoSynapseBackend(Memory(Config.from_env()))
    health = memo_backend.health()
    refs = memo_backend.collect("astor terapia", k=5, trace_id="abc")
    receipt = memo_backend.remember({
        "kind": "decision",
        "text": "...",
        "target": "memo",
        "metadata": {
            "synapse_trace_id": "abc",
            "synapse_route_reason": "deep_semantic",
            "synapse_agent_id": "claude-4-7",
            "title": "Optional override",
        },
    })
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

try:
    from consciousness_contracts import EvidenceRef, WriteReceipt
    _CONTRACTS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep, absent in CI/clean installs
    EvidenceRef = None  # type: ignore[assignment,misc]
    WriteReceipt = None  # type: ignore[assignment,misc]
    _CONTRACTS_AVAILABLE = False

from memo.memory import (
    _PROVENANCE_KEYS,
    MEMO_BACKEND_NAME,
    Memory,
    _extract_provenance,
)
from memo.util import utc_now_iso as _utc_now_iso

_log = logging.getLogger(__name__)

# The contracts constrain source/backend to a closed set; backend_name is a
# str config constant that always holds one of these at runtime.
_BackendName = Literal["memo", "memflow", "synapse"]

# Wire schema strings — pinned to the synapse.* legacy strings until synapse
# itself migrates to consciousness_contracts (then both can flip together).
# The typed objects below ensure shape stays in sync with the contract package.
_WRITE_RECEIPT_LEGACY_SCHEMA = "synapse.memory_write_receipt.v1"
_EVIDENCE_URI_PREFIX = "memo://memoria/"
_DEFAULT_TYPE = "note"
# Kinds synapse uses that don't map 1:1 to memo's frozenset of types are
# coerced to `note` and the original kind is preserved as a tag so the
# semantic intent is not lost.
_TYPE_ALIASES = {
    "task": "note",
    "idea": "note",
    "awareness": "note",
    "retraction": "note",
    "supersede": "note",
}




def _coerce_memo_type(kind: str) -> tuple[str, str | None]:
    """Map synapse `kind` → (memo type, optional kind-tag).

    Returns (memo_type, extra_tag_or_None). Unknown kinds fall back to
    `note` with the original kind preserved as a tag.
    """
    from memo.memory import _VALID_TYPES

    if kind in _VALID_TYPES:
        return kind, None
    if kind in _TYPE_ALIASES:
        return _TYPE_ALIASES[kind], f"kind:{kind}"
    return _DEFAULT_TYPE, f"kind:{kind}"


class MemoSynapseBackend:
    """In-process implementation of synapse's `SynapseBackend` contract.

    Construct with an existing warm `Memory` instance. All calls
    delegate to that instance; no subprocess, no extra MLX load.
    """

    backend_name = MEMO_BACKEND_NAME

    def __init__(self, memory: Memory) -> None:
        if not _CONTRACTS_AVAILABLE:
            raise RuntimeError(
                "MemoSynapseBackend requires the optional `consciousness-contracts` "
                "package. Install it with: uv pip install -e ../consciousness-contracts"
            )
        self.memory = memory

    # -- contract methods --------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return backend availability without mutating state.

        Shape matches `synapse.models.BackendHealth.to_dict()`.
        """
        try:
            total = self.memory.store.count()
            available = True
            status = "ready"
            detail = f"total={total} model={self.memory.cfg.embedder_model}"
        except Exception as exc:  # pragma: no cover - defensive
            available = False
            status = "unavailable"
            detail = f"{type(exc).__name__}: {exc}"
        return {
            "name": self.backend_name,
            "available": available,
            "status": status,
            "detail": detail,
        }

    def collect(
        self,
        query: str,
        *,
        k: int = 5,
        trace_id: str = "",
    ) -> list[dict[str, Any]]:
        """Top-K hybrid search → list of evidence-ref dicts.

        Shape per element matches `synapse.models.EvidenceRef.to_dict()`.
        The `metadata` bag preserves memo's full row (type, tags, path,
        provenance) so synapse can filter / re-rank / replay.
        """
        if not query or not query.strip():
            return []
        hits = self.memory.search(query, limit=max(1, k))
        refs: list[dict[str, Any]] = []
        for h in hits:
            extra = dict(h.extra or {})
            prov = _extract_provenance(extra)
            ref = EvidenceRef(
                source=cast(_BackendName, self.backend_name),
                uri=f"{_EVIDENCE_URI_PREFIX}{h.id}",
                title=h.title or "Memo memoria",
                snippet=_clip(h.body or ""),
                score=float(h.score) if h.score is not None else None,
                updated_at=h.updated or "",
                metadata={
                    "type": h.type,
                    "tags": list(h.tags),
                    "path": h.path,
                    "created_at": h.created,
                    "provenance": prov,
                    "extra": {k: v for k, v in extra.items() if k not in _PROVENANCE_KEYS},
                    "synapse_trace_id": trace_id,
                },
            )
            # Drop the "schema" field — memo's wire format for collect() never
            # carried one, and synapse parses by position not schema string.
            d = ref.to_dict()
            d.pop("schema", None)
            refs.append(d)
        return refs

    def remember(self, request: dict[str, Any]) -> dict[str, Any]:
        """Delegate a typed memory write.

        `request` shape mirrors `synapse.models.MemoryWriteRequest`:
            {
              "kind": "decision" | "fact" | ...,
              "text": "<markdown body>",
              "target": "memo" | "auto" | ...,
              "evidence_paths": [str, ...],
              "metadata": {synapse_trace_id, synapse_route_reason,
                           synapse_write_policy_schema,
                           synapse_agent_id, synapse_agent_signature,
                           title?, tags?, ...},
            }

        Returns a receipt dict matching `synapse.models.MemoryWriteReceipt`.
        """
        kind = str(request.get("kind") or _DEFAULT_TYPE)
        text = str(request.get("text") or "")
        if not text.strip():
            raise ValueError("remember(): empty text")
        target = str(request.get("target") or "auto")
        evidence_paths = list(request.get("evidence_paths") or [])
        metadata = dict(request.get("metadata") or {})

        memo_type, kind_tag = _coerce_memo_type(kind)
        title_override = str(metadata.get("title") or "").strip() or None
        tag_request = metadata.get("tags") or []
        if isinstance(tag_request, str):
            tag_request = [tag_request]
        tags = list(tag_request)
        # Always add "synapse" tag so memo users can filter the corpus
        # by writer. Adds `kind:<original>` tag too when memo coerced
        # the kind, so the synapse intent is recoverable.
        tags.append("synapse")
        if kind_tag:
            tags.append(kind_tag)

        # `extra` carries both provenance (filtered + persisted to
        # history) and any non-provenance metadata synapse sent.
        extra: dict[str, Any] = {}
        for key in _PROVENANCE_KEYS:
            if key in metadata:
                extra[key] = metadata[key]
        # Mirror what synapse already set as default target if absent.
        if "synapse_write_target" not in extra:
            extra["synapse_write_target"] = self.backend_name
        for key, value in metadata.items():
            if key in _PROVENANCE_KEYS or key in {"title", "tags"}:
                continue
            extra.setdefault(key, value)
        if evidence_paths:
            extra.setdefault("synapse_evidence_paths", list(evidence_paths))

        body = _content_with_evidence(text, evidence_paths)
        # Synapse-originated writes always carry a trace; opt them in
        # to the freeze-write check by default. Callers can disable
        # by setting `metadata.respect_synapse_freeze = False`.
        respect_freeze = metadata.get("respect_synapse_freeze")
        rec = self.memory.save(
            content=body,
            title=title_override,
            type_=memo_type,
            tags=tags,
            extra=extra,
            respect_synapse_freeze=(
                True if respect_freeze is None else bool(respect_freeze)
            ),
            # Synapse keeps its own ledger; suppress the memflow receipt
            # so the same write isn't double-counted in memflow events.
            skip_memflow_receipt=True,
        )

        receipt = WriteReceipt(
            backend=cast(_BackendName, self.backend_name),
            receipt_id=rec.id,
            trace_id=str(extra.get("synapse_trace_id") or ""),
            kind=kind,  # type: ignore[arg-type]  # synapse may send arbitrary string
            uri=f"{_EVIDENCE_URI_PREFIX}{rec.id}",
            title=rec.title,
            requested_target=target,  # type: ignore[arg-type]  # tolerated by wire
            generated_at=_utc_now_iso(),
            evidence_paths=tuple(evidence_paths),
            metadata={
                "memo_type": memo_type,
                "memoria_id": rec.id,
                "path": rec.path,
                "tags": list(rec.tags),
                "provenance": _extract_provenance(extra),
            },
        )
        # Keep legacy synapse.* schema string on the wire until synapse migrates.
        d = receipt.to_dict()
        d["schema"] = _WRITE_RECEIPT_LEGACY_SCHEMA
        return d


# -- helpers --------------------------------------------------------------


def _clip(text: str, limit: int = 320) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _content_with_evidence(text: str, evidence_paths: list[str]) -> str:
    if not evidence_paths:
        return text
    refs = "\n".join(f"- {p}" for p in evidence_paths)
    return f"{text}\n\n## Evidence paths\n\n{refs}\n"


__all__ = ["MemoSynapseBackend"]
