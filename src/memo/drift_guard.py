"""GC-8 Drift Guard — enforce durable constraints against actual code changes.

Where ``constitution.py`` *projects* your standing rules into agent instruction
files (advisory), this *enforces* them: it scans a code diff and flags added
lines that violate a durable prohibition. Same rule source (``gather_rules``),
opposite direction — one advises the agent, one catches the code.

High-precision v1 (the documented risk is false positives, so recall is traded
for precision): a rule is enforceable only when it (a) carries a negative
marker (never / don't / avoid / nunca / evitá / …) AND (b) names the banned
pattern inside backticks or quotes. So ``Never use `git add -A` `` is caught;
``Never commit secrets`` (nothing delimited to match) is deliberately not.
Semantic drift via an LLM judge is a later, gated addition.

All functions are pure over plain strings — unit-testable with no Memory and no
git repo. The CLI (``cli_drift.py``) wires the real ``git diff`` + rule source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Negative-polarity markers (en + es). A rule must contain one to be a
# prohibition; positive rules ("always run `pytest`") are ignored.
_NEGATIVE_MARKERS = (
    "never",
    "don't",
    "do not",
    "dont",
    "avoid",
    "forbid",
    "prohib",
    "no uses",
    "nunca",
    "evitá",
    "evita",
    "jamás",
    "jamas",
)

# Backtick code spans or single/double quoted spans.
_SPAN_RE = re.compile(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'")


@dataclass(frozen=True)
class Prohibition:
    rule_id: str
    text: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class Violation:
    rule_id: str
    rule_text: str
    pattern: str
    path: str
    line: str


def code_spans(text: str) -> list[str]:
    """Backtick- or quote-delimited spans in ``text``, in order, de-duplicated."""
    out: list[str] = []
    for m in _SPAN_RE.finditer(text):
        span = next(g for g in m.groups() if g is not None).strip()
        if span and span not in out:
            out.append(span)
    return out


def _is_prohibition(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _NEGATIVE_MARKERS)


def parse_prohibitions(rules: list[tuple[str, str]]) -> list[Prohibition]:
    """Keep only rules that are prohibitions AND carry a delimited pattern."""
    out: list[Prohibition] = []
    for rid, text in rules:
        if not _is_prohibition(text):
            continue
        spans = code_spans(text)
        if not spans:
            continue
        out.append(Prohibition(rule_id=rid, text=text, patterns=tuple(spans)))
    return out


def added_lines_from_diff(diff: str) -> list[tuple[str, str]]:
    """Parse a unified diff → ``(path, added_line_text)`` pairs.

    Tracks the current file from ``+++ b/<path>`` headers and collects ``+``
    lines that are not the ``+++`` header. Removed and context lines are
    ignored — drift is about what you are *adding*.
    """
    out: list[tuple[str, str]] = []
    path = "?"
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            path = target[2:] if target.startswith("b/") else target
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            out.append((path, raw[1:]))
    return out


def scan(prohibitions: list[Prohibition], added: list[tuple[str, str]]) -> list[Violation]:
    """Flag every added line that contains a banned pattern (case-insensitive)."""
    out: list[Violation] = []
    lowered = [(path, line, line.lower()) for path, line in added]
    for pro in prohibitions:
        for pattern in pro.patterns:
            needle = pattern.lower()
            for path, line, low in lowered:
                if needle in low:
                    out.append(
                        Violation(
                            rule_id=pro.rule_id,
                            rule_text=pro.text,
                            pattern=pattern,
                            path=path,
                            line=line.strip(),
                        )
                    )
    return out
