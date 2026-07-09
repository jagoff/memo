from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memo.errors import AmbiguousIdError, ValidationError


@dataclass(frozen=True)
class QualityCompactProposal:
    proposal_id: str
    source_ids: list[str]
    canonical_title: str
    reasons: list[str]
    scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source_ids": list(self.source_ids),
            "canonical_title": self.canonical_title,
            "reasons": list(self.reasons),
            "scope": self.scope,
        }


def _extra(record: Any) -> dict[str, Any]:
    raw = getattr(record, "extra", None)
    return raw if isinstance(raw, dict) else {}


def _is_sensitive(record: Any) -> bool:
    tags = {str(tag) for tag in (getattr(record, "tags", None) or [])}
    extra = _extra(record)
    return (
        str(getattr(record, "type", "")) == "secret"
        or bool(extra.get("secret"))
        or "secret" in tags
    )


def _scope_key(record: Any) -> str | None:
    project_tags = {
        str(tag).strip()
        for tag in (getattr(record, "tags", None) or [])
        if str(tag).strip().startswith("project:")
    }
    if not project_tags:
        return "global"
    if len(project_tags) == 1:
        return next(iter(project_tags))
    return None


def _proposal_id(scope: str, canonical_id: str) -> str:
    scope_slug = "".join(ch if ch.isalnum() else "-" for ch in scope).strip("-") or "global"
    return f"quality-compact-{scope_slug}-{canonical_id[:8]}"


def preview_quality_compaction(memory: Any, *, limit: int = 20) -> dict[str, Any]:
    """Return conservative, read-only quality-compaction proposals.

    Task 5 intentionally stays narrow: preview only, no writes, no retrieval-path
    changes, and no fuzzy clustering. We surface only explicit canonical links
    already present on records (`canonical_id` / `superseded_by`) and only when
    every participating record stays within the same scope.
    """

    if limit < 1:
        raise ValidationError("limit must be >= 1")

    rows = memory.list(limit=limit, include_forgotten=False) if hasattr(memory, "list") else []

    grouped: dict[tuple[str, str], list[Any]] = {}
    errors: list[str] = []
    seen_errors: set[str] = set()

    def _record_error(code: str) -> None:
        if code not in seen_errors:
            seen_errors.add(code)
            errors.append(code)

    for record in rows:
        if _is_sensitive(record):
            continue
        extra = _extra(record)
        canonical_id = str(extra.get("canonical_id") or extra.get("superseded_by") or "").strip()
        if not canonical_id or canonical_id == str(getattr(record, "id", "")):
            continue
        scope = _scope_key(record)
        if scope is None:
            _record_error(f"ambiguous_scope:{getattr(record, 'id', canonical_id)}")
            continue
        grouped.setdefault((scope, canonical_id), []).append(record)

    proposals: list[QualityCompactProposal] = []
    for (scope, canonical_id), sources in sorted(grouped.items()):
        try:
            target = memory.get(canonical_id) if hasattr(memory, "get") else None
        except AmbiguousIdError:
            _record_error(f"unresolved_canonical:{canonical_id}")
            continue
        if target is None:
            _record_error(f"unresolved_canonical:{canonical_id}")
            continue
        if _is_sensitive(target):
            continue
        target_scope = _scope_key(target)
        if target_scope is None:
            _record_error(f"ambiguous_scope:{canonical_id}")
            continue
        if target_scope != scope:
            continue
        canonical_title = str(getattr(target, "title", "") or f"Canonical memory {canonical_id[:8]}")

        source_ids = sorted({str(getattr(source, "id", "")) for source in sources if getattr(source, "id", "")})
        if not source_ids:
            continue

        proposals.append(
            QualityCompactProposal(
                proposal_id=_proposal_id(scope, canonical_id),
                source_ids=source_ids,
                canonical_title=canonical_title,
                reasons=["explicit_canonical_or_superseded_by"],
                scope=scope,
            )
        )

    return {
        "mode": "preview",
        "proposals": [proposal.to_dict() for proposal in proposals],
        "applied": [],
        "errors": errors,
    }
