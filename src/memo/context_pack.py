from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from memo.quality import QualityDecision, classify_quality

DEFAULT_BUDGET_CHARS = 4000

# --code section caps: enough neighborhood to orient without flooding the pack.
_CODE_SECTION_SYMBOL_CAP = 8
_CODE_SECTION_MEMORY_CAP = 5


@dataclass(frozen=True)
class ContextPack:
    question: str
    summary: str
    current_facts: list[dict[str, Any]]
    supporting_context: list[dict[str, Any]]
    stale_or_conflicting: list[dict[str, Any]]
    omissions: str

    def to_prompt(self) -> str:
        sections = []
        if self.summary:
            sections.append(f"Context summary:\n{self.summary}")
        sections.extend(
            [
                _format_section("Current facts", self.current_facts),
                _format_section("Supporting context", self.supporting_context),
                _format_section("Stale/conflicting context", self.stale_or_conflicting),
            ]
        )
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


def is_sensitive_memory(hit: Any) -> bool:
    """Return whether a memory must never enter generated context."""
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

    if is_sensitive_memory(hit):
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


def _omissions_text(base: str, trimmed: int) -> str:
    if not trimmed:
        return base
    extra = f"+{trimmed} trimmed by budget"
    return f"{base}; {extra}" if base else extra


def _pack_prompt(
    pack: ContextPack,
    *,
    current: list[dict[str, Any]],
    supporting: list[dict[str, Any]],
    stale: list[dict[str, Any]],
    trimmed: int,
) -> str:
    return ContextPack(
        pack.question,
        pack.summary,
        current,
        supporting,
        stale,
        _omissions_text(pack.omissions, trimmed),
    ).to_prompt()


def _truncate_snippet(snippet: str, keep_chars: int) -> str:
    if keep_chars >= len(snippet):
        return snippet
    if keep_chars <= 0:
        return ""
    if keep_chars <= 3:
        return snippet[:keep_chars]
    return snippet[: keep_chars - 3].rstrip() + "..."


def _trim_to_budget(pack: ContextPack, budget_chars: int) -> ContextPack:
    if budget_chars <= 0 or len(pack.to_prompt()) <= budget_chars:
        return pack
    supporting = list(pack.supporting_context)
    stale = list(pack.stale_or_conflicting)
    current = list(pack.current_facts)
    omitted = 0
    while (
        len(
            _pack_prompt(pack, current=current, supporting=supporting, stale=stale, trimmed=omitted)
        )
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
    if (
        current
        and len(
            _pack_prompt(pack, current=current, supporting=supporting, stale=stale, trimmed=omitted)
        )
        > budget_chars
    ):
        row = dict(current[-1])
        snippet = str(row.get("snippet", "") or "")
        best_row: dict[str, Any] | None = None
        low = 0
        high = len(snippet)
        while low <= high:
            mid = (low + high) // 2
            candidate = dict(row)
            candidate["snippet"] = _truncate_snippet(snippet, mid)
            candidate_current = [*current[:-1], candidate]
            if (
                len(
                    _pack_prompt(
                        pack,
                        current=candidate_current,
                        supporting=supporting,
                        stale=stale,
                        trimmed=omitted,
                    )
                )
                <= budget_chars
            ):
                best_row = candidate
                low = mid + 1
            else:
                high = mid - 1
        if best_row is not None:
            current[-1] = best_row
        else:
            current.pop()
            omitted += 1
    trimmed_pack = ContextPack(
        pack.question,
        pack.summary,
        current,
        supporting,
        stale,
        _omissions_text(pack.omissions, omitted),
    )
    if len(trimmed_pack.to_prompt()) <= budget_chars:
        return trimmed_pack

    no_omissions_pack = ContextPack(pack.question, pack.summary, current, supporting, stale, "")
    if len(no_omissions_pack.to_prompt()) <= budget_chars:
        return no_omissions_pack

    prefix = "Context summary:\n"
    if budget_chars <= len(prefix):
        return ContextPack(pack.question, "", [], [], [], "")
    summary = pack.summary[: budget_chars - len(prefix)]
    return ContextPack(pack.question, summary, [], [], [], "")


def _symbol_locations(graph: sqlite3.Connection, symbols: set[str]) -> list[tuple[str, str, int]]:
    """(name, file_path, start_line) for each symbol node, alphabetical, capped."""
    if not symbols:
        return []
    marks = ", ".join("?" * len(symbols))
    sql = (
        "SELECT DISTINCT name, file_path, start_line FROM nodes "  # noqa: S608 — placeholders only
        f"WHERE kind != 'file' AND name IN ({marks}) ORDER BY name, file_path LIMIT ?"
    )
    try:
        rows = graph.execute(sql, (*sorted(symbols), _CODE_SECTION_SYMBOL_CAP)).fetchall()
    except sqlite3.Error:
        return []
    return [(str(row[0]), str(row[1] or ""), int(row[2] or 0)) for row in rows]


def code_related_section(code: str, store_conn: Any) -> str:
    """'## Código relacionado' block anchoring the pack on a symbol or path.

    ``code`` containing '/' is treated as a file path (the symbols defined
    there seed the walk); anything else is a literal symbol name. The seed
    expands 1 hop over the codegraph into up to 8 symbol lines
    (``name — file_path:start_line``) plus up to 5 memories citing the
    neighborhood. Fail-open: missing index, unknown anchor, or any engine
    error → '' (the pack renders exactly as without --code).
    """
    from memo import code_intel

    anchor = str(code or "").strip()
    if not anchor:
        return ""
    opened = code_intel.open_graph()
    if opened is None:
        return ""
    graph, _db_repo_id = opened
    try:
        seeds = code_intel.symbols_for_files(graph, [anchor]) if "/" in anchor else {anchor}
        symbols = code_intel.neighbors(graph, seeds, hops=1)
        locations = _symbol_locations(graph, symbols)
    finally:
        graph.close()
    if not locations:
        return ""
    lines = ["## Código relacionado"]
    lines.extend(f"- {name} — {path}:{line}" for name, path, line in locations)
    citing = code_intel.memories_citing(store_conn, symbols=symbols, limit=_CODE_SECTION_MEMORY_CAP)
    if citing:
        lines.append("Memorias que citan este código:")
        lines.extend(f"- [{memory['id'][:8]}] {memory['title']}" for memory in citing)
    return "\n".join(lines)


def build_context_pack(
    question: str,
    hits: list[Any],
    *,
    snippet_chars: int,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
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
        elif row["quality_bucket"] == "supporting":
            supporting.append(row)
        elif index == 0 or not current:
            current.append(row)
        else:
            supporting.append(row)
    if current and stale:
        summary = (
            "Current context is available; stale/conflicting memories are included only as history."
        )
    elif current:
        summary = "Current context is available from the retrieved memories."
    elif supporting and stale:
        summary = "Supporting context was retrieved, but no canonical/current fact was identified; stale/conflicting memories are included only as history."
    elif supporting:
        summary = "Supporting context was retrieved, but no canonical/current fact was identified."
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
