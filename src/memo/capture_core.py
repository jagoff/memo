"""Core capture pipeline: transcript parsing → extraction → hygiene → saving.

Groups 1-5 of the capture module:
  1. Transcript state & parsing
  2. Tool activity & provenance
  3. Text extraction & sanitization
  4. Pre-filtering & hygiene
  5. Extraction, dedup & saving

Stateless utilities called by:
- run_capture() (Stop hook, in capture_hooks.py)
- run_capture_incremental() (idle daemon, in capture_hooks.py)
- extract_and_save_text() (write-time extraction, exported from main module)
"""

from __future__ import annotations

import json
import logging
import math
import re as _re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memo.config import Config

from memo.claim_support import check_claim_support
from memo.fact_extraction import FACT_EDGES_KEY, candidate_fact_edges
from memo.grounding_judge import score_grounding
from memo.util import sha256_short

_log = logging.getLogger(__name__)

# Trigger keywords — pre-filter pass. Cheap regex check before paying
# the helper LLM cost. Permissive; better to send to the LLM and have
# it return [] than to skip a real insight on a false negative.
_TRIGGER_PATTERNS = (
    "decid",
    "fix",
    "bug",
    "error",
    "issue",
    "from now on",
    "siempre",
    "nunca",
    "regla",
    "preferenc",
    "discover",
    "turns out",
    "result",
    "shippe",
    "merged",
    "deploy",
    "config",
    "instal",
    "uninstal",
    "migrate",
    "switch to",
    "use ",
    "usá",
    "uso ahora",
    "should",
    "porque",
    "because",
    "why ",
    "fail",
    "broke",
    "rompi",
    "crash",
    "perform",
    "latenc",
    "warm",
    "cold",
    "model",
    "embed",
    "rerank",
    "commit",
    "branch",
    "test",
    "regress",
)


_EXTRACT_SYSTEM_PROMPT = """You read one conversation turn between a developer and an AI coding assistant. Your job: extract ACTIONABLE INSIGHTS the developer would want to remember in their personal memory archive — and ONLY those.

EXTRACT:
- Decisions made with rationale ("we'll use X because Y")
- Bugs found with root cause + fix (not just "fixed it")
- Preferences expressed ("from now on, always X" / "never Y")
- Discoveries / non-obvious facts ("X turns out to require Y")
- Commands / config that worked ("to do X, run Y") — type "procedure"
- Recurring mistakes with the wrong and the right way ("doing X breaks Y; do Z instead") — type "failure_pattern"
- State changes as ONE insight: when the turn shows a switch ("switched from X to Y", "moved off X", "now using Y instead of X"), record it as one memory that names both the old and new state — never two separate facts. Type it "decision" or "fact" (never "state").

DO NOT extract:
- Mid-process status updates ("checking…", "looking at…", "let me…")
- Speculation ("we could…", "if we wanted…")
- Code snippets shown but not adopted
- Generic tutorials, documentation summaries
- Pleasantries, conversational filler
- One-off or volatile facts: transient task status, values that change every session, and generic language/framework knowledge the assistant already knows.

For each insight, output a JSON object with:
- "title": ≤80 chars, no period at end, descriptive of the insight
- "type": one of "decision", "bug", "preference", "fact", "note", "procedure", "failure_pattern"
- "body": 2-5 sentences. INCLUDE: what the insight is, why it matters, and how to apply it. Preserve proper nouns, file paths, version/model names, and numbers EXACTLY as written — never paraphrase or round them.
  For "failure_pattern" ONLY, structure the body as four labelled lines:
  "Pattern: <the mistake>", "Context: <when it happens>", "Wrong: <what was done>", "Right: <what to do instead>".
- "tags": 3-6 lowercase tags (project, technology, domain)
- Optional "fact_edges": 0-3 structured facts when the insight states a concrete relation. Each edge must be {"subject": "...", "predicate": "...", "object": "..."} with short stable strings. Prefer project/component names as subjects, lowercase verb predicates ("uses", "requires", "fixed_by", "supersedes"), and concrete objects. Omit if uncertain.

Output ONLY a JSON array. Empty array `[]` if nothing notable.
NO markdown fences. NO commentary. NO preamble."""


# ── GROUP 1: State & Transcript Parsing ─────────────────────────────────────


def _parse_transcript(transcript_path: Path) -> list[tuple[str, str]]:
    """Parse the full JSONL transcript into a list of (role, text) pairs.

    Returns [] if unreadable. Only keeps user/assistant entries with text.
    """
    if not transcript_path.is_file():
        return []
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        _log.debug("capture: transcript read failed (%s): %s", transcript_path, exc)
        return []
    parsed: list[tuple[str, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            # A valid-JSON non-object line (`null`, a bare string) must not
            # crash capture for the whole session.
            continue
        role = obj.get("type") or obj.get("role")
        if role not in ("user", "assistant"):
            continue
        msg = obj.get("message", obj)
        content = msg.get("content") if isinstance(msg, dict) else None
        if role == "user" and not _has_prompt_text(content):
            # tool_result carrier message (no text blocks): projecting its
            # TOOL ACTIVITY into a synthetic user turn corrupts exchange
            # pairing — the real prompt would be dropped in favour of
            # "TOOL ACTIVITY: → ok" lines.
            continue
        text = _extract_text(content)
        if text:
            parsed.append((role, text))
    return parsed


def _has_prompt_text(content: Any) -> bool:
    """True when a user message carries real prompt text, not only tool_result
    carrier blocks."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(b, dict)
            and b.get("type") == "text"
            and isinstance(b.get("text"), str)
            and b["text"].strip()
            for b in content
        )
    return False


def _read_recent_exchanges(
    transcript_path: Path,
    n: int = 3,
) -> tuple[str, str] | None:
    """Return the last N (user, assistant) pairs from the transcript as a
    single (combined_user, combined_assistant) tuple so the LLM gets
    richer context than just the last message pair.

    `n=1` reproduces the original single-exchange behaviour.
    `n=3` (default) gives the LLM the last 3 rounds of dialogue, which
    is usually enough to spot multi-step decisions that span turns (e.g.
    "let's use X" followed by "ok, I updated the config" in turn 2).

    Returns None if the transcript doesn't yield even one complete pair.
    """
    parsed = _parse_transcript(transcript_path)
    if not parsed:
        return None

    # Collect the last N user→assistant pairs, walking backwards.
    exchanges: list[tuple[str, str]] = []
    i = len(parsed) - 1
    while i >= 0 and len(exchanges) < n:
        # Find an assistant block.
        while i >= 0 and parsed[i][0] != "assistant":
            i -= 1
        if i < 0:
            break
        # Collect all contiguous assistant blocks (multi-message turns).
        a_chunks: list[str] = []
        while i >= 0 and parsed[i][0] == "assistant":
            a_chunks.insert(0, parsed[i][1])
            i -= 1
        # Find the preceding user message.
        while i >= 0 and parsed[i][0] != "user":
            i -= 1
        if i < 0:
            break
        u_text = parsed[i][1]
        a_text = "\n\n".join(a_chunks)
        exchanges.insert(0, (u_text, a_text))
        i -= 1

    if not exchanges:
        return None

    combined_user = "\n\n---\n\n".join(u for u, _ in exchanges)
    combined_assistant = "\n\n---\n\n".join(a for _, a in exchanges)
    return combined_user, combined_assistant


def _read_last_exchange(transcript_path: Path) -> tuple[str, str] | None:
    """Backward-compat alias: read only the last (user, assistant) pair."""
    return _read_recent_exchanges(transcript_path, n=1)


def _parse_exchanges(transcript_path: Path) -> list[tuple[str, str]]:
    """All (user, assistant) exchanges in chronological order.

    Groups the flat (role, text) stream into ordered turns: consecutive
    user messages are joined, then the following assistant run is joined,
    forming one exchange. Only complete exchanges (a user turn followed by
    an assistant response) are kept.

    Where `_read_recent_exchanges` walks *backward* for the Stop hook's
    last-N view, this walks *forward* so incremental capture can slice the
    NEW turns since a per-session watermark (`exchanges[watermark:]`).
    """
    parsed = _parse_transcript(transcript_path)
    exchanges: list[tuple[str, str]] = []
    i = 0
    n = len(parsed)
    while i < n:
        while i < n and parsed[i][0] != "user":
            i += 1
        if i >= n:
            break
        u_chunks: list[str] = []
        while i < n and parsed[i][0] == "user":
            u_chunks.append(parsed[i][1])
            i += 1
        a_chunks: list[str] = []
        while i < n and parsed[i][0] == "assistant":
            a_chunks.append(parsed[i][1])
            i += 1
        if a_chunks:  # complete exchange only (has an assistant response)
            exchanges.append(("\n\n".join(u_chunks), "\n\n".join(a_chunks)))
    return exchanges


# ── GROUP 2: Tool Activity & Provenance ─────────────────────────────────────


def _tool_activity(content: Any) -> str:
    """Compact projection of a message's tool stream for capture grounding.

    The durable evidence in a coding session lives in the tool calls — the
    Edit that fixed the bug, the file touched, the command + exit code, the
    test that went green — not in the narration. Feeding the extractor only
    prose ('I fixed it', 'that works now') yields vague memories; this surfaces
    the file/symbol/command tokens that both match queries and make a recalled
    memory actionable. Returns '' when there is no tool activity.
    """
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "tool_use":
            name = str(block.get("name") or "tool")
            inp = block.get("input")
            if not isinstance(inp, dict):
                inp = {}
            arg = (
                inp.get("file_path")
                or inp.get("path")
                or inp.get("command")
                or inp.get("pattern")
                or inp.get("query")
                or ""
            )
            arg = str(arg).replace("\n", " ").strip()
            parts.append(f"{name}({arg[:60]})" if arg else name)
        elif btype == "tool_result":
            is_err = bool(block.get("is_error"))
            parts.append("→ error" if is_err else "→ ok")
    return "; ".join(parts).strip()


# Tool names whose `input` names a filesystem path, split by intent. Bash is
# deliberately excluded: parsing file args out of shell commands is unreliable.
_READ_TOOLS = frozenset({"Read", "Grep", "Glob", "NotebookRead"})
_WRITE_TOOLS = frozenset({"Edit", "MultiEdit", "Write", "NotebookEdit"})


def _record_tool_file(
    block: Any,
    files_read: dict[str, None],
    files_mod: dict[str, None],
) -> None:
    """Record one path-bearing tool block in its recency-ordered bucket."""
    if not isinstance(block, dict) or block.get("type") != "tool_use":
        return
    inp = block.get("input")
    if not isinstance(inp, dict):
        return
    path = str(inp.get("file_path") or inp.get("notebook_path") or inp.get("path") or "").strip()
    if not path or "/" not in path:
        return
    name = str(block.get("name") or "")
    bucket = files_mod if name in _WRITE_TOOLS else files_read if name in _READ_TOOLS else None
    if bucket is None:
        return
    bucket.pop(path, None)  # re-insert → most-recent-last ordering
    bucket[path] = None


def collect_tool_files(transcript_path: Path, max_files: int = 10) -> dict[str, list[str]]:
    """Structured projection of the session's tool stream: files read vs modified.

    Complements `_tool_activity` (flattened text evidence, capped at 300 chars)
    with machine-readable arrays for the by-file retrieval lane — a retrieval
    key vectors handle poorly (2026-07-03 survey, Tier2 #13 / claude-mem
    findByFile). Covers the whole transcript so far, deduped, keeping the
    MOST RECENT `max_files` distinct paths per array (re-touching a file moves
    it to the end). Soft-fail: unreadable transcript → empty arrays.
    """
    files_read: dict[str, None] = {}
    files_mod: dict[str, None] = {}
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {"files_read": [], "files_modified": []}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        msg = obj.get("message", obj)
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            _record_tool_file(block, files_read, files_mod)
    return {
        "files_read": list(files_read)[-max_files:],
        "files_modified": list(files_mod)[-max_files:],
    }


def _capture_provenance(session_id: str, transcript_path: Path, turn_hash: str) -> dict[str, Any]:
    """Provenance bag merged under every captured memory's extra: the source
    session/turn (so a memory can escalate to its origin) plus, when
    MEMO_CAPTURE_TOOL_EVIDENCE is on, the tool-file arrays. Shared by the Stop
    and incremental capture paths.
    """
    prov: dict[str, Any] = {
        "session_id": session_id,
        "transcript_path": str(transcript_path),
        "turn_hash": turn_hash,
    }
    from memo.flags import flag_bool

    if flag_bool("MEMO_CAPTURE_TOOL_EVIDENCE"):
        prov.update({k: v for k, v in collect_tool_files(transcript_path).items() if v})
    return prov


# ── GROUP 3: Text Extraction & Sanitization ────────────────────────────────


def _strip_private(text: str) -> str:
    """Honor <private>…</private> spans: content inside never reaches the
    extractor. Applies to EVERY transcript read path — Stop-hook capture,
    incremental capture-tick, and mine-history all funnel through
    _extract_text. Gated by MEMO_PRIVATE_MARKERS (default on).
    """
    if not text or "<private>" not in text.lower():
        return text
    from memo.flags import flag_bool

    if not flag_bool("MEMO_PRIVATE_MARKERS"):
        return text
    from memo.redact import strip_private_spans

    return strip_private_spans(text)


def _extract_text(content: Any) -> str:
    """Pull the plain text out of a Claude Code message content. Concatenates
    text/markdown blocks, and (when MEMO_CAPTURE_TOOL_EVIDENCE is on) appends a
    compact 'TOOL ACTIVITY' projection of any tool_use/tool_result blocks so
    capture extraction is grounded in what was actually done, not just narrated.
    Image/thinking blocks are skipped.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return _strip_private(content.strip())
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                t = block.get("text", "")
                # A malformed block can carry a non-str "text" (e.g. a nested
                # dict); skip it rather than crash the whole capture/mining pass
                # on `t.strip()` ('dict' object has no attribute 'strip').
                if isinstance(t, str) and t:
                    out.append(t.strip())
        text = "\n\n".join(out).strip()
        from memo.flags import flag_bool, flag_int

        if flag_bool("MEMO_CAPTURE_TOOL_EVIDENCE"):
            activity = _tool_activity(content)
            if activity:
                cap = flag_int("MEMO_CAPTURE_TOOL_EVIDENCE_CHARS") or 300
                text = (text + f"\n\nTOOL ACTIVITY: {activity[:cap]}").strip()
        return _strip_private(text)
    return ""


# ── GROUP 4: Pre-filtering & Hygiene ───────────────────────────────────────


def _passes_prefilter(text: str, min_chars: int = 200) -> bool:
    """Cheap keyword + length check before paying the LLM cost."""
    if len(text) < min_chars:
        return False
    lower = text.lower()
    return any(p in lower for p in _TRIGGER_PATTERNS)


# Meta-commentary filter (capture hygiene) ──────────────────────────────────
#
# Process narration ("voy a…", "let me…", "I'll…") and LLM filler are the
# highest-volume junk class in ambient capture: they describe what the
# assistant is ABOUT to do, not anything durable. Regex at segment start —
# cheap, no LLM call. Gated by MEMO_CAPTURE_META_FILTER (default on).

_META_COMMENTARY_RE = _re.compile(
    r"^(?:"
    # Spanish process narration
    r"voy\s+a\b|d[eé]jame\b|primero\s+voy\b|ahora\s+voy\b|"
    r"vamos\s+a\s+(?:ver|revisar|empezar|chequear)\b|"
    # English process narration. Note: "i'll" requires the apostrophe
    # (straight or curly, ' — "Ill-formed inputs" is substance) and bare
    # "i will" is NOT narration ("I will always use X" is a preference) —
    # only "i will <process verb>".
    r"let\s+me\b|i[\u2019']ll\b|i\s+am\s+going\s+to\b|i[\u2019']m\s+going\s+to\b|"
    r"i\s+will\s+(?:now|start|begin|check)\b|"
    r"first,?\s+(?:let\s+me|i[\u2019']ll)\b|now\s+(?:i[\u2019']ll|let\s+me)\b|"
    r"next,?\s+(?:let\s+me|i[\u2019']ll)\b"
    r")",
    _re.IGNORECASE,
)

# LLM filler OPENERS: only the opener is junk, not the sentence — "Okay, the
# fix is to use flock" keeps "the fix is to use flock". "sure/okay" need the
# comma — "Sure enough, the bug was…" is a discovery, not filler.
_FILLER_OPENER_RE = _re.compile(
    r"^(?:(?:sure|okay|ok)[,!]\s+|(?:certainly|absolutely)[,!.]?(?:\s+|$)|"
    r"(?:great|good)\s+question[,!.:]?(?:\s+|$))",
    _re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = _re.compile(r"(?<=[.!?])\s+")


def _capture_flag_bool(name: str) -> bool:
    from memo.flags import flag_bool

    return flag_bool(name)


def _capture_flag_float(name: str) -> float | None:
    from memo.flags import flag_float

    return flag_float(name)


def is_meta_commentary(text: str) -> bool:
    """True when `text` opens with process narration (not mere filler)."""
    return bool(_META_COMMENTARY_RE.match(text.strip()))


def strip_meta_commentary(text: str) -> str:
    """Drop process-narration segments from `text`, keeping the substance.

    A segment is a line or a sentence within a line. A segment opening with
    process narration (`_META_COMMENTARY_RE`) is removed whole; a segment
    opening with an LLM filler (`_FILLER_OPENER_RE`) keeps everything after
    the opener ("Okay, the fix is X" → "the fix is X"). Text with no
    narration is returned byte-identical (no re-joining side effects).
    Returns '' when nothing substantive survives — the caller drops the
    candidate entirely.
    """
    lines = text.splitlines()
    changed = False
    out_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        segments = [s for s in _SENTENCE_SPLIT_RE.split(stripped) if s.strip()]
        kept: list[str] = []
        line_changed = False
        for seg in segments:
            seg_stripped = seg.strip()
            if _META_COMMENTARY_RE.match(seg_stripped):
                line_changed = True  # narration: the whole segment is junk
                continue
            opener = _FILLER_OPENER_RE.match(seg_stripped)
            if opener:
                line_changed = True  # filler: junk is the opener, not the rest
                rest = seg_stripped[opener.end() :].strip()
                if rest:
                    kept.append(rest)
                continue
            kept.append(seg)
        if not line_changed:
            out_lines.append(line)  # untouched lines round-trip verbatim
        else:
            changed = True
            if kept:
                out_lines.append(" ".join(kept))
    return "\n".join(out_lines).strip() if changed else text


# Type-classification confidence (capture hygiene) ─────────────────────────


def score_type_confidence(type_: str, text: str) -> float:
    """Heuristic 0-1 confidence that `type_` is the right classification.

    Marker-count based (no LLM call): the type patterns in
    `memo.memory.write_ops._TYPE_PATTERNS` are the corroborating evidence.
    Matches of the claimed type's own markers raise confidence; markers of a
    DIFFERENT type lower it. Types without marker patterns (e.g. 'note') have
    nothing to corroborate → neutral default, docked when another type's
    markers are present. LLM-classified candidates with zero markers land at
    the 0.5 mid default.
    """
    from memo.memory.write_ops import _TYPE_PATTERNS

    snippet = text[:600]
    own = 0
    others = 0
    has_pattern = False
    for t, pattern in _TYPE_PATTERNS:
        n = len(pattern.findall(snippet))
        if t == type_:
            has_pattern = True
            own = n
        else:
            others += n
    if not has_pattern:
        # 'note' & friends: no markers of their own to check against.
        return 0.6 if others == 0 else 0.4
    if own >= 2:
        return 0.95
    if own == 1:
        return 0.85 if others == 0 else 0.7
    return 0.5 if others == 0 else 0.35


def reweight_ambiguous_type(type_: str, text: str, weights: dict[str, float]) -> str:
    """Citation-weight tie-break at capture's genuine type-ambiguity point.

    The capture classifier is the extractor LLM's single type claim; the only
    corroborating evidence is the `write_ops._TYPE_PATTERNS` markers (the same
    signal `score_type_confidence` uses) — there is no scored candidate list to
    multiply. The genuinely AMBIGUOUS case is therefore a claim with zero own
    markers while another type's markers are present in the text (the <0.5
    confidence branches: fallback-to-note with contradicting markers, or a
    marked type claimed without its own evidence). Only there the marker-backed
    contenders are ranked by their citation weight, and the candidate is
    re-typed only when the winner's weight STRICTLY exceeds the claimed type's
    — so empty/neutral weights (no nightly signal) change nothing, and a
    corroborated or marker-less classification is never touched.
    """
    if not weights:
        return type_
    from memo.memory.write_ops import _TYPE_PATTERNS

    snippet = text[:600]
    counts = {t: len(p.findall(snippet)) for t, p in _TYPE_PATTERNS}
    if counts.get(type_, 0) > 0:
        return type_  # claim corroborated by its own markers — not ambiguous
    marked = [t for t, _ in _TYPE_PATTERNS if counts[t] > 0]
    if not marked:
        return type_  # no marker evidence at all — nothing to break the tie with
    top = max(counts[t] for t in marked)
    contenders = [t for t in marked if counts[t] == top]
    # Highest citation weight wins; pattern order breaks exact weight ties.
    winner = max(contenders, key=lambda t: weights.get(t, 1.0))
    if weights.get(winner, 1.0) > weights.get(type_, 1.0):
        return winner
    return type_


# ── GROUP 5: Extraction, Dedup & Saving ────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity (token-set) between two strings."""
    ta = set(_re.findall(r"\w+", a.lower()))
    tb = set(_re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def dedupe_batch(
    candidates: list[dict[str, Any]],
    mem: Any,
    threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    """Collapse near-duplicate candidates within one capture batch.

    Similarity is embedder cosine when available — the embedder is already
    loaded on this path (each survivor gets embedded by `find_near_duplicate`
    anyway), so no new MLX cost class is added. When embedding fails, degrade
    to token-set Jaccard: stricter than cosine, so the fallback only
    under-collapses, never over-collapses. Returns (kept, dropped_count);
    original batch order is preserved.
    """
    if len(candidates) < 2:
        return list(candidates), 0
    texts = [f"{c['title']}\n\n{c['body']}" for c in candidates]
    vecs: list[list[float]] | None
    try:
        vecs = [mem.embedder.embed_query(t) for t in texts]
    except Exception as exc:
        _log.debug("capture: batch-dedup embed failed, Jaccard fallback: %s", exc)
        vecs = None

    def _sim(i: int, j: int) -> float:
        if vecs is not None:
            return _cosine(vecs[i], vecs[j])
        return _jaccard(texts[i], texts[j])

    def _quality(i: int) -> tuple[float, int]:
        c = candidates[i]
        return (
            score_type_confidence(str(c.get("type") or "note"), texts[i]),
            len(c.get("body") or ""),
        )

    kept: list[int] = []
    dropped = 0
    for i in range(len(candidates)):
        merged = False
        for pos, k in enumerate(kept):
            if _sim(i, k) >= threshold:
                if _quality(i) > _quality(k):
                    kept[pos] = i  # keep the better twin
                dropped += 1
                merged = True
                break
        if not merged:
            kept.append(i)
    return [candidates[i] for i in sorted(kept)], dropped


# Generic openers that produce low-value, session-scoped memories.
_GENERIC_PREFIXES = (
    # Session-narrative openers — produce low-value "what happened today" summaries
    # rather than durable knowledge. "today i " is intentionally excluded here
    # because many genuine decision memories start with "Today I decided/chose/..."
    "the user ",
    "today the ",
    "we discussed ",
    "we worked on ",
    "i helped ",
    "i assisted ",
    "this session ",
    "during this ",
)


def _passes_quality(text: str, min_words: int | None = None) -> bool:
    """True if text is specific enough to be worth saving as a long-term memory.

    Env vars:
      MEMO_CAPTURE_MIN_WORDS — minimum word count (default 15; set to 0 to disable).
    """
    if min_words is None:
        from memo.flags import flag_int

        _raw = flag_int("MEMO_CAPTURE_MIN_WORDS")
        min_words = 15 if _raw is None else max(0, _raw)
    t = text.strip()
    # Too short
    if min_words > 0 and len(t.split()) < min_words:
        return False
    lower = t.lower()
    # Generic session-narrative openers
    if any(lower.startswith(p) for p in _GENERIC_PREFIXES):
        return False
    # Pure temporal noise: "on 2026-05-16 ..."
    return not _re.match(r"^(on )?\d{4}-\d{2}-\d{2}", lower)


def _hash_assistant(text: str) -> str:
    """Hash of assistant message for idempotence detection."""
    return sha256_short(text)


def extract_insights(
    helper_chat: Any,
    helper_model: str,
    user_text: str,
    assistant_text: str,
    *,
    state_dir: Path | None = None,
    on_call_failure: list[Exception] | None = None,
) -> list[dict[str, Any]]:
    """Run the helper LLM to extract insights.

    Returns a list of insight dicts; empty on parse failure, model refusal, or a
    transient LLM-call failure. A raised ``chat()`` (LLM timeout / MLX busy /
    daemon unreachable) is a TRANSIENT failure — categorically different from
    the model running and deliberately returning ``[]`` (nothing notable). When
    ``on_call_failure`` is provided it receives the exception so callers (the
    incremental capture path) can tell the two apart and retry, instead of
    advancing a watermark past turns whose insight was never actually examined.
    The helper model is configured by `cfg.helper_model`; warm latency is
    typically ~1-3s.
    """
    system_prompt = _EXTRACT_SYSTEM_PROMPT
    if state_dir is not None:
        from memo.prompt_overrides import resolve_prompt

        system_prompt = resolve_prompt("capture_extract", _EXTRACT_SYSTEM_PROMPT, state_dir)
    user_block = f"USER:\n{user_text[:4000]}\n\nASSISTANT:\n{assistant_text[:8000]}"
    try:
        resp = helper_chat.chat(
            model=helper_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_block},
            ],
            options={"temperature": 0.0, "seed": 42, "num_predict": 800},
        )
        raw = (resp.get("message") or {}).get("content") or ""
    except Exception as exc:
        # Transient extraction failure — surface it (out-param) so the caller can
        # distinguish it from a legit empty result, but still return [] to keep
        # every caller's contract unchanged.
        _log.warning("capture: helper LLM call failed: %s", exc)
        if on_call_failure is not None:
            on_call_failure.append(exc)
        return []

    raw = raw.strip()
    if not raw:
        return []
    # Handle the occasional fenced response — strip fences before parse.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json\n"):
            raw = raw[5:]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    from memo.flags import flag_bool

    do_redact = flag_bool("MEMO_REDACT_SECRETS")
    entropy = flag_bool("MEMO_REDACT_ENTROPY")
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Helper-LLM output is untrusted: any of these may come back as a
        # nested dict/list instead of a string — skip rather than crash.
        title = item.get("title") or ""
        body = item.get("body") or ""
        type_ = item.get("type") or "note"
        if not all(isinstance(v, str) for v in (title, body, type_)):
            continue
        title = title.strip()
        body = body.strip()
        type_ = type_.strip()
        tags = item.get("tags") or []
        raw_fact_edges = item.get(FACT_EDGES_KEY)
        if not title or not body:
            continue
        if not isinstance(tags, list):
            tags = []
        norm_tags = [str(t).lower().strip() for t in tags if t]
        if do_redact:
            from memo.redact import redact_secrets

            r_title = redact_secrets(title, entropy=entropy)
            r_body = redact_secrets(body, entropy=entropy)
            if r_title.found or r_body.found:
                title, body = r_title.text, r_body.text
                if "_redacted" not in norm_tags:
                    norm_tags.append("_redacted")
            redacted_edges: list[dict[str, Any]] = []
            for edge in raw_fact_edges if isinstance(raw_fact_edges, list) else []:
                if not isinstance(edge, dict):
                    continue
                redacted_edge = dict(edge)
                for key in ("subject", "predicate", "object", "object_"):
                    value = redacted_edge.get(key)
                    if isinstance(value, str):
                        result = redact_secrets(value, entropy=entropy)
                        if result.found and "_redacted" not in norm_tags:
                            norm_tags.append("_redacted")
                        redacted_edge[key] = result.text
                redacted_edges.append(redacted_edge)
            raw_fact_edges = redacted_edges if raw_fact_edges is not None else None
        out.append(
            {
                "title": title[:80],
                "type": type_,
                "body": body,
                "tags": norm_tags,
                FACT_EDGES_KEY: raw_fact_edges,
            }
        )
    return out


def find_near_duplicate(
    memory: Any,
    candidate: dict[str, Any],
    threshold: float = 0.85,
) -> dict[str, Any] | None:
    """Return the top corpus record semantically near `candidate` (>= threshold),
    as ``{"id", "score", "title"}``, else None. Pure vec search (not
    hybrid+rerank) — the dedup decision is about embedding similarity, not the
    reranker's joint judgement.
    """
    composed = f"{candidate['title']}\n\n{candidate['body']}"
    try:
        emb = memory.embedder.embed_query(composed)
        rows = memory.store.search(emb, limit=1)
    except Exception as exc:
        _log.debug("capture: near-duplicate check failed, treating as new: %s", exc)
        return None
    if not rows:
        return None
    top_score = rows[0].get("score")
    if top_score is None or top_score < threshold:
        return None
    return {"id": rows[0].get("id"), "score": top_score, "title": rows[0].get("title")}


def is_near_duplicate(
    memory: Any,
    candidate: dict[str, Any],
    threshold: float = 0.85,
) -> bool:
    """Back-compat bool wrapper around :func:`find_near_duplicate`."""
    return find_near_duplicate(memory, candidate, threshold) is not None


def _filter_meta_candidates(
    insights: list[dict[str, Any]], *, debug: bool
) -> tuple[list[dict[str, Any]], int]:
    """Strip process narration and return the retained candidates plus drop count."""

    if not _capture_flag_bool("MEMO_CAPTURE_META_FILTER"):
        return insights, 0
    hygienic: list[dict[str, Any]] = []
    skipped = 0
    for candidate in insights:
        cleaned = strip_meta_commentary(candidate["body"])
        if not cleaned or is_meta_commentary(candidate["title"]):
            skipped += 1
            _log.debug("capture: drop meta-commentary %r", candidate["title"])
            if debug:
                print(
                    f"# memo capture: drop meta-commentary '{candidate['title']}'",
                    file=sys.stderr,
                )
            continue
        hygienic.append(
            {**candidate, "body": cleaned} if cleaned != candidate["body"] else candidate
        )
    return hygienic, skipped


def _apply_claim_support(
    candidate: dict[str, Any],
    tags: list[str],
    *,
    enabled: bool,
    debug: bool,
) -> tuple[Any | None, int]:
    """Quarantine unsupported outcome claims and return uncertainty delta."""

    if not enabled:
        return None, 0
    verdict = check_claim_support(f"{candidate['title']}\n{candidate['body']}")
    if not verdict.unsupported:
        return verdict, 0
    uncertainty_delta = 0
    if "_uncertain" not in tags:
        tags.append("_uncertain")
        uncertainty_delta = 1
    if debug:
        print(
            f"# memo capture: unsupported {verdict.claim_kind} claim "
            f"'{candidate['title']}' — tagged _uncertain ({verdict.reason})",
            file=sys.stderr,
        )
    return verdict, uncertainty_delta


def _extract_and_save(
    mem: Any,
    cfg: Any,
    user_text: str,
    assistant_text: str,
    *,
    debug: bool = False,
    merge_tags: list[str] | None = None,
    auto_project: bool = True,
    extra_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract insights from one (user, assistant) blob, dedup, save.

    Shared by the Stop-hook (`run_capture`), the incremental
    (`run_capture_incremental`), and the explicit write-time extraction
    (`extract_and_save_text`) paths so all apply the same quality gate,
    near-duplicate check, and save metadata. `merge_tags` (e.g. a caller's
    `project:` tag) is appended to every saved fact's tags; default None keeps
    the hook paths unchanged. `extra_base` is merged under every saved fact's
    `extra` (provenance: session_id/transcript_path/turn_hash and, later,
    tool-file arrays) so a memory can always escalate to its source turn;
    default None keeps all callers unchanged. Returns counts; every
    per-candidate failure is absorbed (logged only in debug).
    """
    from memo.memory.record import _VALID_TYPES
    from memo.prompt_overrides import prompt_version

    _extract_prompt_version = prompt_version(
        "capture_extract", _EXTRACT_SYSTEM_PROMPT, cfg.state_dir
    )
    _helper_chat = mem._ensure_chat()
    assistant_for_extraction, _crush_hash = maybe_crush_json_capture(
        assistant_text,
        user_text,
        cfg,
    )
    _extraction_failures: list[Exception] = []
    insights = extract_insights(
        _helper_chat,
        cfg.helper_model,
        user_text,
        assistant_for_extraction,
        state_dir=cfg.state_dir,
        on_call_failure=_extraction_failures,
    )
    _ground_on = _capture_flag_bool("MEMO_GROUNDING_JUDGE")
    _ground_min = _capture_flag_float("MEMO_GROUNDING_WRITE_MIN") or 0.4
    _ground_source = f"{user_text}\n{assistant_text}"
    _claim_on = _capture_flag_bool("MEMO_CLAIM_SUPPORT")
    _claim_conf = _capture_flag_float("MEMO_CLAIM_SUPPORT_CONFIDENCE") or 0.5
    if debug:
        print(f"# memo capture: {len(insights)} candidate(s)", file=sys.stderr)
    # `candidates` reports the EXTRACTED count (pre-hygiene) — it feeds
    # extract_and_save_text's verbatim fallback (candidates == 0 means the
    # extractor produced nothing, not that hygiene filtered everything) and
    # keeps the debug/telemetry meaning stable.
    n_extracted = len(insights)

    # Meta-commentary filter: process narration never reaches the vault.
    # Segment-level — a mixed candidate keeps its substantive sentences; a
    # candidate that is ALL narration (or whose title is narration-shaped)
    # is dropped whole. Dropped candidates are logged (debug trail), never saved.
    insights, skipped_meta = _filter_meta_candidates(insights, debug=debug)

    # Dedup → reconcile band. The corpus's highest-value memories are evolving
    # decisions on hot topics — exactly the candidates MOST similar (0.85–0.97)
    # to a prior memory on the same topic. Silently dropping those (the old
    # behaviour) fossilized the first thing ever said about a topic: an old
    # "use Qwen3-0.6B" decision blocked today's "switched to 4B, +12%" from ever
    # entering, so the contradiction/supersede machinery never ran. Now only
    # near-IDENTICAL paraphrases (>= drop_threshold) are dropped; a same-topic
    # candidate below that is ADMITTED as new so the nightly contradiction/
    # evolution pass (which demotes the superseded side) can do its job.
    near_threshold = _capture_flag_float("MEMO_CAPTURE_DUP_THRESHOLD") or 0.85
    drop_threshold = _capture_flag_float("MEMO_CAPTURE_DUP_DROP_THRESHOLD") or 0.97

    # Intra-batch near-dup window (prompt-retry pattern): collapse candidates
    # that duplicate EACH OTHER before the store-level check — the store check
    # only sees already-saved memories, so batch twins would slip through as
    # bogus same-run "evolutions". Keeps the higher-confidence/longer twin.
    skipped_batch_dup = 0
    if _capture_flag_bool("MEMO_CAPTURE_BATCH_DEDUP") and len(insights) > 1:
        insights, skipped_batch_dup = dedupe_batch(insights, mem, near_threshold)
        if debug and skipped_batch_dup:
            print(
                f"# memo capture: collapsed {skipped_batch_dup} intra-batch near-dup(s)",
                file=sys.stderr,
            )

    min_confidence = _capture_flag_float("MEMO_CAPTURE_MIN_CONFIDENCE") or 0.0

    # Citation-type feedback (default OFF): weights from the nightly dream
    # capture_weights pass, consulted only at the ambiguous-classification
    # branch inside the loop. Empty weights ⇒ identical behavior to flag off.
    type_weights: dict[str, float] = {}
    if _capture_flag_bool("MEMO_CAPTURE_TYPE_FEEDBACK"):
        from memo.capture_weights import load_type_weights

        type_weights = load_type_weights(cfg)

    saved: list[str] = []
    saved_titles: list[str] = []
    saved_records: list[dict[str, str]] = []
    save_failures = 0
    facts = 0
    fact_edges_declared = 0
    fact_edges_prepared = 0
    fact_edges_structured = 0
    fact_edges_fallback = 0
    skipped_dup = 0
    reconciled = 0
    skipped_quality = 0
    uncertain = 0
    retyped = 0
    for cand in insights:
        # Coerce an out-of-vocabulary / mis-cased type from the helper LLM to a
        # valid one BEFORE save. A hallucinated type (e.g. "state", which the
        # extract prompt's state-change guidance invites) would otherwise raise
        # inside mem.save() and silently drop a mined insight (save_failures++).
        # Logged so systematic mistyping stays visible.
        _cand_type = str(cand.get("type") or "note").strip().lower()
        if _cand_type not in _VALID_TYPES:
            if debug:
                print(
                    f"# memo capture: coerce type {cand.get('type')!r}→'note' "
                    f"for '{cand.get('title')}'",
                    file=sys.stderr,
                )
            _cand_type = "note"
        if _cand_type != cand.get("type"):
            cand = {**cand, "type": _cand_type}
        # Quality gate: skip low-specificity memories before hitting the
        # embedder or disk. Threshold controlled by MEMO_CAPTURE_MIN_WORDS.
        body_for_quality = f"{cand['title']}\n\n{cand['body']}"
        if not _passes_quality(body_for_quality):
            skipped_quality += 1
            if debug:
                print(
                    f"# memo capture: skip quality '{cand['title']}'",
                    file=sys.stderr,
                )
            continue
        match = find_near_duplicate(mem, cand, threshold=near_threshold)
        if match is not None and (match.get("score") or 0.0) >= drop_threshold:
            # Near-identical paraphrase — no new information; drop.
            skipped_dup += 1
            if debug:
                print(
                    f"# memo capture: drop paraphrase '{cand['title']}' "
                    f"(sim={match.get('score'):.2f} of {(match.get('id') or '')[:8]})",
                    file=sys.stderr,
                )
            continue
        if match is not None:
            # Same topic, evolved — admit as new (don't fossilize the topic).
            reconciled += 1
            if debug:
                print(
                    f"# memo capture: reconcile '{cand['title']}' "
                    f"(sim={match.get('score'):.2f} of {(match.get('id') or '')[:8]}) — "
                    "admitting so supersede pass can run",
                    file=sys.stderr,
                )
        # Citation-type feedback: at a genuinely ambiguous classification
        # (claimed type uncorroborated while another type's markers are
        # present) re-type to the marker-backed type the grounding data cites
        # most. No-op unless MEMO_CAPTURE_TYPE_FEEDBACK is on AND weights exist.
        if type_weights:
            _claimed_type = str(cand.get("type") or "note").strip() or "note"
            _new_type = reweight_ambiguous_type(_claimed_type, body_for_quality, type_weights)
            if _new_type != _claimed_type:
                retyped += 1
                if debug:
                    print(
                        f"# memo capture: retype '{cand['title']}' "
                        f"{cand.get('type')}→{_new_type} (citation-type feedback)",
                        file=sys.stderr,
                    )
                cand = {**cand, "type": _new_type}
        # Confidence scoring: how strongly the content's own markers back the
        # type classification. Always stamped in extra (observability); below
        # MEMO_CAPTURE_MIN_CONFIDENCE (default 0.0 = gating off) the candidate
        # is still saved but tagged '_uncertain' for later review. (A true
        # pre-archived save isn't supported: lifecycle archival moves the .md
        # to inactive/ and DROPS the index rows — the memory would become
        # unsearchable, not merely recall-excluded — so the tag is the gate.)
        confidence = score_type_confidence(str(cand.get("type") or "note"), body_for_quality)
        tags = [*cand["tags"], *merge_tags] if merge_tags else list(cand["tags"])
        if min_confidence > 0.0 and confidence < min_confidence:
            uncertain += 1
            if "_uncertain" not in tags:
                tags.append("_uncertain")
            if debug:
                print(
                    f"# memo capture: low-confidence type '{cand['title']}' "
                    f"({confidence:.2f} < {min_confidence:.2f}) — tagged _uncertain",
                    file=sys.stderr,
                )
        grounding_score: float | None = None
        if _ground_on:
            grounding_score = score_grounding(
                _helper_chat,
                cfg.helper_model,
                source=_ground_source,
                claim=f"{cand['title']}\n{cand['body']}",
            )
            if grounding_score is not None and grounding_score < _ground_min:
                uncertain += 1
                if "_uncertain" not in tags:
                    tags.append("_uncertain")
                if debug:
                    print(
                        f"# memo capture: low grounding '{cand['title']}' "
                        f"({grounding_score:.2f} < {_ground_min:.2f}) — tagged _uncertain",
                        file=sys.stderr,
                    )
        claim_verdict, claim_uncertain = _apply_claim_support(
            cand, tags, enabled=_claim_on, debug=debug
        )
        uncertain += claim_uncertain
        raw_fact_edges = cand.get(FACT_EDGES_KEY)
        if isinstance(raw_fact_edges, dict):
            fact_edges_declared += 1
        elif isinstance(raw_fact_edges, list):
            fact_edges_declared += sum(1 for item in raw_fact_edges if isinstance(item, dict))
        fact_edges = candidate_fact_edges(
            title=str(cand.get("title") or ""),
            raw_edges=raw_fact_edges,
            extractor="memo.capture",
            mode="atomic-insight",
            confidence=confidence,
            provenance={"prompt_version": _extract_prompt_version},
            metadata={
                "type": str(cand.get("type") or "note"),
                "tags": tags,
                "body_preview": str(cand.get("body") or "")[:240],
            },
        )
        fact_edges_prepared += len(fact_edges)
        structured_count = sum(
            1
            for edge in fact_edges
            if not (edge.get("subject") == "memory" and edge.get("predicate") == "asserts")
        )
        fact_edges_structured += structured_count
        fact_edges_fallback += len(fact_edges) - structured_count
        if debug and fact_edges:
            print(
                f"# memo capture: prepared {len(fact_edges)} fact edge(s) "
                f"({structured_count} structured) for '{cand['title']}'",
                file=sys.stderr,
            )
        extra_for_save = {
            **(extra_base or {}),
            "capture_confidence": round(confidence, 3),
            "prompt_version": _extract_prompt_version,
        }
        if grounding_score is not None:
            extra_for_save["grounding_score"] = round(grounding_score, 3)
        if claim_verdict is not None and claim_verdict.unsupported:
            extra_for_save["claim_support_kind"] = claim_verdict.claim_kind
            extra_for_save["claim_support_reason"] = claim_verdict.reason
        if fact_edges:
            extra_for_save[FACT_EDGES_KEY] = fact_edges
        save_succeeded = False
        try:
            rec = mem.save(
                content=cand["body"],
                title=cand["title"],
                type_=cand["type"],
                tags=tags,
                auto_project=auto_project,
                extra=extra_for_save,
            )
            save_succeeded = True
            saved.append(rec.id)
            saved_titles.append(rec.title)
            saved_records.append({"id": rec.id, "title": rec.title, "type": rec.type})
            if claim_verdict is not None and claim_verdict.unsupported:
                try:
                    mem.store.set_confidence_batch([(rec.id, _claim_conf)])
                    if debug:
                        print(
                            f"# memo capture: claim downgrade '{cand['title']}' "
                            f"→ conf {_claim_conf:.2f} ({claim_verdict.reason})",
                            file=sys.stderr,
                        )
                except Exception as _exc:  # never let a downgrade break capture
                    if debug:
                        print(f"# memo capture: claim downgrade skipped: {_exc}", file=sys.stderr)
            facts += len(
                mem.fact_edges.query(
                    source_record_id=rec.id,
                    include_inactive=True,
                    limit=1000,
                )
            )
            if debug:
                print(f"# memo capture: saved [{rec.id[:8]}] {rec.title}", file=sys.stderr)
        except Exception as exc:
            if not save_succeeded:
                save_failures += 1
                # Always log: a systematic failure (dims mismatch, locked DB)
                # would otherwise lose every mined insight with zero signal.
                _log.warning("capture: save failed for '%s': %s", cand.get("title"), exc)
            if debug:
                print(f"# memo capture: save failed: {exc}", file=sys.stderr)

    return {
        "candidates": n_extracted,
        "saved": saved,
        "saved_titles": saved_titles,
        "saved_records": saved_records,
        "save_failures": save_failures,
        # True only when the helper LLM call itself failed transiently (not a
        # legit empty extraction) — the incremental path holds the watermark.
        "extraction_failed": bool(_extraction_failures),
        "facts": facts,
        "extract_stats": {
            "fact_edges_declared": fact_edges_declared,
            "fact_edges_prepared": fact_edges_prepared,
            "fact_edges_structured": fact_edges_structured,
            "fact_edges_fallback": fact_edges_fallback,
            "fact_edges_saved": facts,
        },
        "skipped_dup": skipped_dup,
        "reconciled": reconciled,
        "skipped_quality": skipped_quality,
        "skipped_meta": skipped_meta,
        "skipped_batch_dup": skipped_batch_dup,
        "uncertain": uncertain,
        "retyped": retyped,
    }


def extract_and_save_text(
    mem: Any,
    cfg: Any,
    text: str,
    *,
    merge_tags: list[str] | None = None,
    title: str | None = None,
    type_: str = "note",
    debug: bool = False,
    auto_project: bool = True,
) -> dict[str, Any]:
    """Write-time fact extraction (mem0 ADD-model) for an explicit save.

    Instead of storing `text` as one opaque record, run the SAME
    extract → quality → dedup → save pipeline the capture hook uses
    (`_extract_and_save`), framing the blob as the assistant side of an
    exchange. The blob is decomposed into atomic, individually-searchable
    facts; `merge_tags` (e.g. a `project:` tag the caller passed) is added to
    every fact so explicit save context propagates.

    The prefilter is deliberately NOT applied — the caller explicitly asked to
    save this, so triggerless prose still goes to the extractor.

    Fallback — an explicit save must never silently vanish: if the extractor
    yields ZERO candidates (LLM can't atomize it, or no MLX / helper available),
    the blob is saved verbatim with the caller's `title`/`type_`/`merge_tags`.
    Candidates that ARE found but dropped as near-duplicates / low quality are
    respected (the information is already in the corpus) — no verbatim re-save.

    Returns a summary dict: ``status`` ("extracted" | "verbatim") plus the
    `_extract_and_save` counts (candidates/saved/saved_titles/...).
    """
    result = _extract_and_save(
        mem, cfg, "", text, debug=debug, merge_tags=merge_tags, auto_project=auto_project
    )
    if result["candidates"] == 0:
        from memo.flags import flag_bool

        vb_text = text
        vb_tags = list(merge_tags) if merge_tags else []
        if flag_bool("MEMO_REDACT_SECRETS"):
            from memo.redact import redact_secrets

            res = redact_secrets(text, entropy=flag_bool("MEMO_REDACT_ENTROPY"))
            if res.found:
                vb_text = res.text
                vb_tags.append("_redacted")
        rec = mem.save(
            content=vb_text,
            title=title,
            type_=type_,
            tags=vb_tags or None,
            auto_project=auto_project,
        )
        vb_facts = len(
            mem.fact_edges.query(
                source_record_id=rec.id,
                include_inactive=True,
                limit=1000,
            )
        )
        return {
            "status": "verbatim",
            "candidates": 0,
            "saved": [rec.id],
            "saved_titles": [rec.title],
            "facts": vb_facts,
            "extract_stats": {
                "fact_edges_declared": 0,
                "fact_edges_prepared": vb_facts,
                "fact_edges_structured": 0,
                "fact_edges_fallback": vb_facts,
                "fact_edges_saved": vb_facts,
            },
            "skipped_dup": 0,
            "reconciled": 0,
            "skipped_quality": 0,
            "skipped_meta": 0,
            "skipped_batch_dup": 0,
            "uncertain": 0,
            "retyped": 0,
        }
    return {"status": "extracted", **result}


_CRUSH_TOKEN_RE = _re.compile(r"\w+", _re.UNICODE)


def _row_tokens(row: Any) -> set[str]:
    """Lowercased word-token set of a JSON row (keys + values flattened)."""
    text = row if isinstance(row, str) else json.dumps(row, ensure_ascii=False, sort_keys=True)
    return {t.lower() for t in _CRUSH_TOKEN_RE.findall(text)}


def _score_rows_by_relevance(rows: list[Any], context: str) -> list[float]:
    """Relevance score per row for the SmartCrusher: IDF-weighted distinctiveness.

    Each row is scored by the mean inverse document frequency of its tokens
    across the whole array — rows built from tokens shared by many sibling rows
    (repetitive tool-output boilerplate) score low, rows carrying rare/unique
    tokens (the actual signal) score high, regardless of their position. When
    ``context`` is non-empty, tokens a row shares with it add an IDF-weighted
    bonus capped at 25% of the base score, so context breaks close scores without
    overwhelming distinctiveness. Pure, deterministic and embedder-free — safe
    inside the capture path's latency budget.
    """
    n = len(rows)
    token_sets = [_row_tokens(r) for r in rows]
    df: dict[str, int] = {}
    for tokens in token_sets:
        for tok in tokens:
            df[tok] = df.get(tok, 0) + 1

    def _idf(tok: str) -> float:
        return math.log((n + 1) / (df.get(tok, 0) + 1)) + 1.0

    ctx_tokens = {t.lower() for t in _CRUSH_TOKEN_RE.findall(context)} if context else set()
    scores: list[float] = []
    for tokens in token_sets:
        token_count = max(len(tokens), 1)
        base = sum(_idf(tok) for tok in tokens) / token_count
        overlap = sum(_idf(tok) for tok in tokens & ctx_tokens) / token_count if ctx_tokens else 0.0
        bonus = min(overlap, base * 0.25)
        scores.append(base + bonus)
    return scores


def maybe_crush_json_capture(content: str, context: str, config: Config) -> tuple[str, str | None]:
    """Apply SmartCrusher to JSON arrays in capture content.

    Detects JSON arrays, scores rows by relevance, keeps top-K, and offloads
    the rest to cache. Returns crushed content (original if not applicable)
    and a hash for retrieval.

    Args:
        content: Captured text (may contain JSON array)
        context: Query/context for bounded relevance scoring.
        config: Config instance (for state_dir access)

    Returns:
        Tuple of (crushed_content_or_original, crush_hash_if_crushed_else_None)

    Rows are scored by :func:`_score_rows_by_relevance` (IDF-weighted
    distinctiveness, optionally steered by ``context``); the top-K survive and
    the rest are offloaded to the crush cache behind the marker row.
    """
    import hashlib

    from memo.config import Config
    from memo.flags_capture import (
        flag_crusher_cache_ttl_days,
        flag_crusher_enabled,
        flag_crusher_keep_ratio,
    )
    from memo.store.crush_cache import CrushCache, crush_marker

    if not flag_crusher_enabled():
        return content, None

    # Try to detect JSON array
    content_stripped = content.strip()
    if not (content_stripped.startswith("[") and content_stripped.endswith("]")):
        return content, None

    try:
        json_array = json.loads(content_stripped)
    except (json.JSONDecodeError, TypeError):
        return content, None

    if not isinstance(json_array, list) or len(json_array) < 10:
        # Don't crush small arrays
        return content, None

    keep_ratio = flag_crusher_keep_ratio()
    if keep_ratio >= 1.0:
        return content, None
    keep_count = max(10, int(len(json_array) * keep_ratio))

    # Score rows by IDF-weighted distinctiveness (+ optional context overlap):
    # repetitive boilerplate rows score low, the informationally-distinctive rows
    # survive — position-blind, so a trailing "answer" row is kept.
    scores = _score_rows_by_relevance(json_array, context)

    # Keep the top-K highest-scoring rows, restored to original order. sorted() is
    # stable, so ties fall back to original position (deterministic).
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep_count]
    top_indices.sort()

    crushed_array = [json_array[i] for i in top_indices]
    dropped_count = len(json_array) - len(crushed_array)
    if dropped_count < 1:
        return content, None

    hash_val = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    candidate = json.dumps(
        [*crushed_array, crush_marker(dropped_count, hash_val)],
        ensure_ascii=False,
    )
    if len(candidate.encode("utf-8")) * 100 > len(content.encode("utf-8")) * 95:
        return content, None

    if not isinstance(config, Config):
        return content, None
    try:
        cache = CrushCache(config.state_dir)
        cache.cache(hash_val, content)
        recovered = cache.retrieve(hash_val, ttl_days=flag_crusher_cache_ttl_days())
    except (OSError, TypeError, ValueError):
        return content, None
    if recovered != content:
        return content, None

    return candidate, hash_val
