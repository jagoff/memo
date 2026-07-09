from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memo.quality import QualityDecision, classify_quality


@dataclass(frozen=True)
class ContextPack:
    question: str
    summary: str
    current_facts: list[dict[str, Any]]
    supporting_context: list[dict[str, Any]]
    stale_or_conflicting: list[dict[str, Any]]
    omissions: str

    def to_prompt(self) -> str:
        sections = [
            f"Context summary:\n{self.summary}",
            _format_section("Current facts", self.current_facts),
            _format_section("Supporting context", self.supporting_context),
            _format_section("Stale/conflicting context", self.stale_or_conflicting),
        ]
        if self.omissions:
            sections.append(f"Omissions:\n{self.omissions}")
        return "\n\n".join(section for section in sections if section.strip())


def consult_hits_from_pack(pack: ContextPack) -> list[dict[str, Any]]:
    """Flatten a context pack into recall-log hit dicts."""

    hits: list[dict[str, Any]] = []
    for row in pack.current_facts + pack.supporting_context + pack.stale_or_conflicting:
        hits.append(
            {
                "id": row.get("id") or row.get("id_short") or "",
                "score": row.get("score"),
                "title": row.get("title") or "",
            }
        )
    return hits


def _is_sensitive(hit: Any) -> bool:
    tags = {str(tag) for tag in (getattr(hit, "tags", None) or [])}
    extra = getattr(hit, "extra", None)
    extra_dict = extra if isinstance(extra, dict) else {}
    return (
        str(getattr(hit, "type", "")) == "secret"
        or bool(extra_dict.get("secret"))
        or "secret" in tags
    )


def build_context_row(hit: Any, *, snippet_chars: int) -> dict[str, Any] | None:
    """Build one prompt-ready memory row, omitting sensitive hits."""

    if _is_sensitive(hit):
        return None
    return _snippet(hit, snippet_chars, classify_quality(hit))


def _snippet(hit: Any, snippet_chars: int, decision: QualityDecision) -> dict[str, Any]:
    body = str(getattr(hit, "body", "") or "")
    snippet = body[:snippet_chars]
    if len(body) > snippet_chars:
        snippet = snippet.rstrip() + "..."
    return {
        "source": "memory",
        "id": str(getattr(hit, "id", "")),
        "id_short": str(getattr(hit, "id", ""))[:8],
        "title": str(getattr(hit, "title", "")),
        "type": str(getattr(hit, "type", "")),
        "score": getattr(hit, "score", None),
        "snippet": snippet,
        "quality_bucket": decision.bucket,
        "quality_reasons": list(decision.reasons),
    }


def _format_section(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [f"{title}:"]
    for row in rows:
        lines.append(
            f"[{row['id_short']}] title: {row['title']} | type: {row['type']} | "
            f"quality: {row['quality_bucket']}\n{row['snippet']}"
        )
    return "\n\n".join(lines)


def _trim_to_budget(pack: ContextPack, budget_chars: int) -> ContextPack:
    if budget_chars <= 0 or len(pack.to_prompt()) <= budget_chars:
        return pack
    supporting = list(pack.supporting_context)
    stale = list(pack.stale_or_conflicting)
    current = list(pack.current_facts)
    omitted = 0
    while (
        len(ContextPack(pack.question, pack.summary, current, supporting, stale, pack.omissions).to_prompt())
        > budget_chars
    ):
        if supporting:
            supporting.pop()
            omitted += 1
        elif stale:
            stale.pop()
            omitted += 1
        elif len(current) > 1:
            current.pop()
            omitted += 1
        else:
            break
    omissions = pack.omissions
    if omitted:
        omissions = f"{omissions}; +{omitted} trimmed by budget" if omissions else f"+{omitted} trimmed by budget"
    return ContextPack(pack.question, pack.summary, current, supporting, stale, omissions)


def build_context_pack(
    question: str,
    hits: list[Any],
    *,
    snippet_chars: int,
    budget_chars: int = 4000,
) -> ContextPack:
    current: list[dict[str, Any]] = []
    supporting: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    sensitive_omitted = 0
    for index, hit in enumerate(hits):
        row = build_context_row(hit, snippet_chars=snippet_chars)
        if row is None:
            sensitive_omitted += 1
            continue
        if row["quality_bucket"] == "stale_or_conflicting":
            stale.append(row)
        elif index == 0 or not current:
            current.append(row)
        else:
            supporting.append(row)
    if current and stale:
        summary = "Current context is available; stale/conflicting memories are included only as history."
    elif current:
        summary = "Current context is available from the retrieved memories."
    elif stale:
        summary = "Only stale/conflicting context was retrieved; answer cautiously."
    else:
        summary = "No memory context was retrieved."
    omissions = ""
    if sensitive_omitted:
        noun = "memory" if sensitive_omitted == 1 else "memories"
        omissions = f"{sensitive_omitted} sensitive {noun} omitted from compacted context"
    pack = ContextPack(question, summary, current, supporting, stale, omissions)
    return _trim_to_budget(pack, budget_chars)
