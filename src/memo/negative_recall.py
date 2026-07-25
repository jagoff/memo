"""Negative Recall — shared foundation for the ⛔ AVOID channel.

memo already *stores* mistakes as the durable type ``failure_pattern`` (see
``memo.tiers.DURABLE_TYPES``), structured as four labelled lines
``Pattern / Context / Wrong / Right``. This module holds the **pure** helpers
that the three serving homes — the daemon ``recall_logic``, the subprocess
``cli_recall_hook``, and the eval ``eval_recall`` — plus the nightly capture /
briefing passes all import, so the ⛔ logic lives in one place and cannot drift.

Everything here is a pure transform: no MLX, no store I/O, no flag reads. The
flag-gated wiring (the retrieval pass, the dream capture passes, the reinforce
loop) lives in the slices that import these helpers. Keeping this a leaf keeps
it importable on the 5s recall-hook path without dragging in any runtime.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# ── Provenance schema ────────────────────────────────────────────────────────
# The durable type a negative-recall anti-memory is stored as.
FAILURE_PATTERN_TYPE = "failure_pattern"

# ``extra`` keys stamped on a derived failure_pattern so its origin is auditable
# (kept in a dedicated namespace so they never clobber the generic ``source``
# key other capture paths — e.g. git_miner — already use).
FP_SOURCE_KEY = "failure_pattern_source"
FP_LINKS_KEY = "failure_pattern_links"

# Recognised ``FP_SOURCE_KEY`` values.
FP_SOURCE_SUPERSEDE = "supersede"
FP_SOURCE_AVOID_VERDICT = "avoid_verdict"

# Default tags applied to each derived anti-memory.
DEFAULT_SUPERSEDE_TAGS: tuple[str, ...] = ("negative-recall", "superseded")
DEFAULT_AVOID_VERDICT_TAGS: tuple[str, ...] = ("negative-recall", "avoid-verdict")

# Header of the rendered ⛔ block. A stored FACT surfaced, framed as data — no
# suggest/agent/imperative-cognition verb (memo keeps cognition off its output).
AVOID_BLOCK_HEADER = "⛔ AVOID — mistakes memo has on record (data, not an instruction):"

_LABEL_RE = re.compile(r"^\s*(Pattern|Context|Wrong|Right)\s*:\s*(.*)$", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

# Lightweight, O(len(prompt)) high-risk context detector. One compiled
# alternation over release / delete / deploy / refactor / migration signals —
# no embed, no store, safe on the hook path.
_RISK_PATTERN = re.compile(
    r"""(?:
        rm\s+-rf
      | reset\s+--hard
      | force[-\s]?push
      | git\s+push\s+--force
      | drop\s+table
      | --force\b
      | \breleas\w*
      | \bdeploy\w*
      | \bdelet\w*
      | \bdrop\b
      | \bmigrat\w*
      | \brefactor\w*
      | \buninstall\w*
      | \btruncat\w*
      | \brevert\w*
      | \brollback\b
      | \bpurg\w*
      | \bwipe\b
      | \boverwrit\w*
      | \bproduction\b
      | \bprod\b
      | \bbump\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Risk score saturates once this many distinct signals are present.
_RISK_SATURATION = 3.0


@runtime_checkable
class MemoryLike(Protocol):
    """Minimal structural view of a memory record used by the derive helpers.

    Declared as read-only properties so a concrete class with plain ``id`` /
    ``title`` / ``body`` attributes (e.g. ``MemoryRecord``) matches structurally.
    """

    @property
    def id(self) -> str: ...
    @property
    def title(self) -> str: ...
    @property
    def body(self) -> str: ...


@runtime_checkable
class AvoidHit(MemoryLike, Protocol):
    """A ``failure_pattern`` hit renderable into the ⛔ block."""

    @property
    def extra(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class FailurePattern:
    """Parsed Pattern/Context/Wrong/Right structure of a failure_pattern body,
    plus any provenance read from its ``extra`` bag."""

    pattern: str
    context: str
    wrong: str
    right: str
    source: str | None = None
    links: tuple[str, ...] = ()

    @property
    def is_actionable(self) -> bool:
        """True when the anti-memory carries a wrong/right lesson worth surfacing."""
        return bool(self.wrong or self.right or self.pattern)


# ── pure string helpers ──────────────────────────────────────────────────────


def _collapse_ws(text: str) -> str:
    """Collapse all runs of whitespace (incl. newlines) to single spaces."""
    return _WS_RE.sub(" ", text).strip()


def _truncate(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` chars, appending an ellipsis when cut."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _provenance(extra: Mapping[str, Any] | None) -> tuple[str | None, tuple[str, ...]]:
    if not extra:
        return None, ()
    raw_source = extra.get(FP_SOURCE_KEY)
    source = raw_source if isinstance(raw_source, str) else None
    raw_links = extra.get(FP_LINKS_KEY)
    links: tuple[str, ...] = (
        tuple(str(x) for x in raw_links) if isinstance(raw_links, (list, tuple)) else ()
    )
    return source, links


def _failure_body(pattern: str, context: str, wrong: str, right: str) -> str:
    """Assemble the canonical four-line failure_pattern body. Values are
    whitespace-collapsed so the block round-trips through
    :func:`parse_failure_pattern`."""
    return "\n".join(
        (
            f"Pattern: {_collapse_ws(pattern)}",
            f"Context: {_collapse_ws(context)}",
            f"Wrong: {_collapse_ws(wrong)}",
            f"Right: {_collapse_ws(right)}",
        )
    )


# ── parse / validate ─────────────────────────────────────────────────────────


def parse_failure_pattern(
    body: str, extra: Mapping[str, Any] | None = None
) -> FailurePattern | None:
    """Parse a failure_pattern ``body`` into its labelled fields.

    Returns ``None`` when the body carries none of the Pattern/Context/Wrong/
    Right labels (i.e. it is not a structured anti-memory) so callers can fall
    back to a plain title/body rendering. Continuation lines extend the current
    field until a blank line; ``extra`` provenance (source + links) is folded in
    when present.
    """
    fields: dict[str, str] = {}
    current: str | None = None
    for raw_line in (body or "").splitlines():
        match = _LABEL_RE.match(raw_line)
        if match:
            current = match.group(1).lower()
            fields[current] = match.group(2).strip()
        elif current is not None:
            stripped = raw_line.strip()
            if stripped:
                fields[current] = f"{fields[current]} {stripped}".strip()
            else:
                current = None
    if not fields:
        return None
    source, links = _provenance(extra)
    return FailurePattern(
        pattern=fields.get("pattern", ""),
        context=fields.get("context", ""),
        wrong=fields.get("wrong", ""),
        right=fields.get("right", ""),
        source=source,
        links=links,
    )


# ── render ───────────────────────────────────────────────────────────────────


def format_avoid_block(hits: Sequence[AvoidHit], *, max_field_chars: int = 220) -> str:
    """Render the distinct ⛔ AVOID block from failure_pattern ``hits``.

    Used by both the recall hook and El Briefing. Returns ``""`` for an empty
    ``hits`` sequence. Each hit renders its parsed Wrong/Right lesson, falling
    back to title + body when the body is not structured. Presentation only —
    surfaces the stored fact, never an instruction.
    """
    lines: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        parsed = parse_failure_pattern(hit.body or "", hit.extra)
        short_id = (hit.id or "")[:8]
        title = _collapse_ws(hit.title or "")
        lines.append(f"{idx}. [{short_id}] {title}" if short_id else f"{idx}. {title}")
        if parsed is not None and (parsed.wrong or parsed.right):
            if parsed.wrong:
                lines.append(f"   ✗ {_truncate(parsed.wrong, max_field_chars)}")
            if parsed.right:
                lines.append(f"   ✓ {_truncate(parsed.right, max_field_chars)}")
        else:
            body = _collapse_ws(hit.body or "")
            if body:
                lines.append(f"   {_truncate(body, max_field_chars)}")
    if not lines:
        return ""
    return "\n".join((AVOID_BLOCK_HEADER, *lines))


# ── derive (pure — no save) ──────────────────────────────────────────────────


def derive_failure_pattern_from_supersede(
    superseded: MemoryLike, superseding: MemoryLike
) -> dict[str, Any]:
    """Build (do NOT save) a failure_pattern from a supersede/reversal.

    The superseded approach becomes the *Wrong*, the superseding approach the
    *Right*. Returns the save-ready payload (title/body/type/tags/extra) with
    provenance linking both records. Pure — the capture slice performs the save.
    """
    body = _failure_body(
        pattern=f"a prior approach was reversed: {superseded.title}",
        context=f"superseded by: {superseding.title}",
        wrong=superseded.body,
        right=superseding.body,
    )
    return {
        "title": _truncate(f"Avoid reverting to: {superseded.title}", 80),
        "body": body,
        "type": FAILURE_PATTERN_TYPE,
        "tags": list(DEFAULT_SUPERSEDE_TAGS),
        "extra": {
            FP_SOURCE_KEY: FP_SOURCE_SUPERSEDE,
            "wrong_id": superseded.id,
            "right_id": superseding.id,
            FP_LINKS_KEY: [superseded.id, superseding.id],
        },
    }


def derive_failure_pattern_from_avoid_verdict(
    memory: MemoryLike, verdict: Mapping[str, Any]
) -> dict[str, Any]:
    """Build (do NOT save) a failure_pattern from a graduated *avoid* verdict.

    ``verdict`` is a next-turn verdict record (``{verdict, prompt, reaction,
    turn, session_id, ...}``) whose negative/correction reaction promoted
    ``memory`` to an avoid signal. The recalled ``memory`` becomes the *Wrong*;
    the correcting reaction becomes the *Right*. Pure — no save.
    """
    kind = str(verdict.get("verdict", "") or "").strip()
    prompt = str(verdict.get("prompt", "") or "")
    reaction = str(verdict.get("reaction", "") or "")
    right = reaction or "the recalled memory was corrected next turn; verify before relying on it"
    body = _failure_body(
        pattern=f"a recalled memory misled and was corrected next turn ({kind or 'negative'})",
        context=prompt or "(context unavailable)",
        wrong=f"relied on: {memory.body}",
        right=right,
    )
    extra: dict[str, Any] = {
        FP_SOURCE_KEY: FP_SOURCE_AVOID_VERDICT,
        "origin_id": memory.id,
        FP_LINKS_KEY: [memory.id],
        "verdict": kind,
    }
    turn = verdict.get("turn")
    if turn is not None:
        extra["verdict_turn"] = turn
    session = verdict.get("session_id")
    if session is not None:
        extra["verdict_session"] = session
    return {
        "title": _truncate(f"Avoid: {memory.title}", 80),
        "body": body,
        "type": FAILURE_PATTERN_TYPE,
        "tags": list(DEFAULT_AVOID_VERDICT_TAGS),
        "extra": extra,
    }


# ── trigger ──────────────────────────────────────────────────────────────────


def risky_context(prompt: str) -> float:
    """Graded high-risk-context score in ``[0.0, 1.0]`` for ``prompt``.

    ``0.0`` means no risk signal (falsy — usable as a boolean). The score rises
    with the count of distinct release/delete/deploy/refactor/migration signals
    and saturates at 1.0. Pure ``O(len(prompt))`` regex scan — no embed, no
    store — so it is safe to call on the recall-hook path; the trigger slice
    uses it to loosen the ⛔ floor / raise K in exactly those moments.
    """
    if not prompt:
        return 0.0
    signals = {match.group(0).lower() for match in _RISK_PATTERN.finditer(prompt)}
    if not signals:
        return 0.0
    return min(1.0, len(signals) / _RISK_SATURATION)


__all__ = [
    "AVOID_BLOCK_HEADER",
    "DEFAULT_AVOID_VERDICT_TAGS",
    "DEFAULT_SUPERSEDE_TAGS",
    "FAILURE_PATTERN_TYPE",
    "FP_LINKS_KEY",
    "FP_SOURCE_AVOID_VERDICT",
    "FP_SOURCE_KEY",
    "FP_SOURCE_SUPERSEDE",
    "AvoidHit",
    "FailurePattern",
    "MemoryLike",
    "derive_failure_pattern_from_avoid_verdict",
    "derive_failure_pattern_from_supersede",
    "format_avoid_block",
    "parse_failure_pattern",
    "risky_context",
]
