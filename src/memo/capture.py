"""Save-side ambient capture — Phase B.

Hook fires on every assistant turn (Stop event), reads the just-finished
exchange from the transcript, asks the helper LLM (Qwen2.5-3B) to
extract any actionable insights, dedups against the existing corpus,
and saves the survivors with auto-derived metadata.

## Pipeline

```
Stop event
   │
   ▼  read transcript_path JSONL → last (user, assistant) exchange
   │
   ▼  pre-filter (cheap): skip empty / too-short / pure-tool turns
   │
   ▼  helper LLM extract → JSON [{title, type, body, tags}, ...]
   │
   ▼  dedup: embed each candidate, near-search, drop if max_sim > 0.85
   │
   ▼  save survivors via Memory.save()
```

## State file

`~/.local/share/memo/last-capture.json` tracks the hash of the last
processed assistant message so re-firing on the same turn (e.g. the
user runs `/clear` mid-stream, or two Stop hooks race) doesn't
double-extract.

## Why dedup with the embedder, not the title

Two memories with different titles can describe the same fact. The
embedder is the only signal that catches "same fact, different
phrasing". Threshold 0.85 is empirical: cosine sim between
near-paraphrases is typically 0.85-0.95 with Qwen3-Embedding;
genuinely-distinct memories score below 0.75 even on the same topic.

## Failure modes

All swallowed silently. Capture is opportunistic — a hook that fails
to extract is no worse than the pre-Phase-B world. Exception: if the
user explicitly sets `MEMO_CAPTURE_DEBUG=1`, errors print to stderr.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re as _re
import sys
from pathlib import Path
from typing import Any

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
- Commands / config that worked ("to do X, run Y")

DO NOT extract:
- Mid-process status updates ("checking…", "looking at…", "let me…")
- Speculation ("we could…", "if we wanted…")
- Code snippets shown but not adopted
- Generic tutorials, documentation summaries
- Pleasantries, conversational filler

For each insight, output a JSON object with:
- "title": ≤80 chars, no period at end, descriptive of the insight
- "type": one of "decision", "bug", "preference", "fact", "note"
- "body": 2-5 sentences. INCLUDE: what the insight is, why it matters, and how to apply it. Be specific (file paths, numbers, model names) when relevant.
- "tags": 3-6 lowercase tags (project, technology, domain)

Output ONLY a JSON array. Empty array `[]` if nothing notable.
NO markdown fences. NO commentary. NO preamble."""


def _state_file(state_dir: Path) -> Path:
    return state_dir / "last-capture.json"


def _load_state(state_dir: Path) -> dict[str, Any]:
    f = _state_file(state_dir)
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    # Atomic write: this file is shared by every session on the machine, so a
    # torn/partial write would clobber a sibling session's last-capture state.
    # Write to a .tmp then os.replace (mirrors session.py `_write`).
    state_dir.mkdir(parents=True, exist_ok=True)
    dest = _state_file(state_dir)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, dest)


def _parse_transcript(transcript_path: Path) -> list[tuple[str, str]]:
    """Parse the full JSONL transcript into a list of (role, text) pairs.
    Returns [] if unreadable. Only keeps user/assistant entries with text."""
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
        role = obj.get("type") or obj.get("role")
        if role not in ("user", "assistant"):
            continue
        msg = obj.get("message", obj)
        content = msg.get("content") if isinstance(msg, dict) else None
        text = _extract_text(content)
        if text:
            parsed.append((role, text))
    return parsed


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


def _strip_private(text: str) -> str:
    """Honor <private>…</private> spans: content inside never reaches the
    extractor. Applies to EVERY transcript read path — Stop-hook capture,
    incremental capture-tick, and mine-history all funnel through
    _extract_text. Gated by MEMO_PRIVATE_MARKERS (default on)."""
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
                if t:
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


def _passes_prefilter(text: str, min_chars: int = 200) -> bool:
    """Cheap keyword + length check before paying the LLM cost."""
    if len(text) < min_chars:
        return False
    lower = text.lower()
    return any(p in lower for p in _TRIGGER_PATTERNS)


# ── Meta-commentary filter (capture hygiene) ─────────────────────────────────
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
    # (straight or curly, ’ — "Ill-formed inputs" is substance) and bare
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


# ── Type-classification confidence (capture hygiene) ─────────────────────────


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


# ── Intra-batch near-dup window (capture hygiene) ────────────────────────────
#
# The store-level dedup (`find_near_duplicate`) only compares a candidate
# against ALREADY-SAVED memories. Within one capture run, prompt retries
# yield 2-3 candidates for the same fact: the first survivor gets saved,
# and its siblings then land in the 0.85-0.97 "reconcile" band against it
# and are admitted as bogus "evolutions". This pass collapses the batch
# FIRST, keeping the best (higher-confidence, then longer) of each group.


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def _jaccard(a: str, b: str) -> float:
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
    return sha256_short(text)


def extract_insights(
    helper_chat: Any,
    helper_model: str,
    user_text: str,
    assistant_text: str,
) -> list[dict[str, Any]]:
    """Run the helper LLM. Returns a list of insight dicts; empty on
    parse failure or model refusal. The helper is the small Qwen2.5-3B
    so latency is bounded (~1-3s warm).
    """
    user_block = f"USER:\n{user_text[:4000]}\n\nASSISTANT:\n{assistant_text[:8000]}"
    try:
        resp = helper_chat.chat(
            model=helper_model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ],
            options={"temperature": 0.0, "seed": 42, "num_predict": 800},
        )
        raw = (resp.get("message") or {}).get("content") or ""
    except Exception as exc:
        _log.warning("capture: helper LLM call failed: %s", exc)
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
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        body = (item.get("body") or "").strip()
        type_ = (item.get("type") or "note").strip()
        tags = item.get("tags") or []
        if not title or not body:
            continue
        if not isinstance(tags, list):
            tags = []
        out.append(
            {
                "title": title[:80],
                "type": type_,
                "body": body,
                "tags": [str(t).lower().strip() for t in tags if t],
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
    reranker's joint judgement."""
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


def _extract_and_save(
    mem: Any,
    cfg: Any,
    user_text: str,
    assistant_text: str,
    *,
    debug: bool = False,
    merge_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Extract insights from one (user, assistant) blob, dedup, save.

    Shared by the Stop-hook (`run_capture`), the incremental
    (`run_capture_incremental`), and the explicit write-time extraction
    (`extract_and_save_text`) paths so all apply the same quality gate,
    near-duplicate check, and save metadata. `merge_tags` (e.g. a caller's
    `project:` tag) is appended to every saved fact's tags; default None keeps
    the hook paths unchanged. Returns counts; every per-candidate failure is
    absorbed (logged only in debug)."""
    insights = extract_insights(
        mem._ensure_chat(),
        cfg.helper_model,
        user_text,
        assistant_text,
    )
    if debug:
        print(f"# memo capture: {len(insights)} candidate(s)", file=sys.stderr)
    # `candidates` reports the EXTRACTED count (pre-hygiene) — it feeds
    # extract_and_save_text's verbatim fallback (candidates == 0 means the
    # extractor produced nothing, not that hygiene filtered everything) and
    # keeps the debug/telemetry meaning stable.
    n_extracted = len(insights)

    from memo.flags import flag_bool, flag_float

    # Meta-commentary filter: process narration never reaches the vault.
    # Segment-level — a mixed candidate keeps its substantive sentences; a
    # candidate that is ALL narration (or whose title is narration-shaped)
    # is dropped whole. Dropped candidates are logged (debug trail), never saved.
    skipped_meta = 0
    if flag_bool("MEMO_CAPTURE_META_FILTER"):
        hygienic: list[dict[str, Any]] = []
        for cand in insights:
            cleaned = strip_meta_commentary(cand["body"])
            if not cleaned or is_meta_commentary(cand["title"]):
                skipped_meta += 1
                _log.debug("capture: drop meta-commentary %r", cand["title"])
                if debug:
                    print(
                        f"# memo capture: drop meta-commentary '{cand['title']}'",
                        file=sys.stderr,
                    )
                continue
            if cleaned != cand["body"]:
                cand = {**cand, "body": cleaned}
            hygienic.append(cand)
        insights = hygienic

    # Dedup → reconcile band. The corpus's highest-value memories are evolving
    # decisions on hot topics — exactly the candidates MOST similar (0.85–0.97)
    # to a prior memory on the same topic. Silently dropping those (the old
    # behaviour) fossilized the first thing ever said about a topic: an old
    # "use Qwen3-0.6B" decision blocked today's "switched to 4B, +12%" from ever
    # entering, so the contradiction/supersede machinery never ran. Now only
    # near-IDENTICAL paraphrases (>= drop_threshold) are dropped; a same-topic
    # candidate below that is ADMITTED as new so the nightly contradiction/
    # evolution pass (which demotes the superseded side) can do its job.
    near_threshold = flag_float("MEMO_CAPTURE_DUP_THRESHOLD") or 0.85
    drop_threshold = flag_float("MEMO_CAPTURE_DUP_DROP_THRESHOLD") or 0.97

    # Intra-batch near-dup window (prompt-retry pattern): collapse candidates
    # that duplicate EACH OTHER before the store-level check — the store check
    # only sees already-saved memories, so batch twins would slip through as
    # bogus same-run "evolutions". Keeps the higher-confidence/longer twin.
    skipped_batch_dup = 0
    if flag_bool("MEMO_CAPTURE_BATCH_DEDUP") and len(insights) > 1:
        insights, skipped_batch_dup = dedupe_batch(insights, mem, near_threshold)
        if debug and skipped_batch_dup:
            print(
                f"# memo capture: collapsed {skipped_batch_dup} intra-batch near-dup(s)",
                file=sys.stderr,
            )

    min_confidence = flag_float("MEMO_CAPTURE_MIN_CONFIDENCE") or 0.0

    # Citation-type feedback (default OFF): weights from the nightly dream
    # capture_weights pass, consulted only at the ambiguous-classification
    # branch inside the loop. Empty weights ⇒ identical behavior to flag off.
    type_weights: dict[str, float] = {}
    if flag_bool("MEMO_CAPTURE_TYPE_FEEDBACK"):
        from memo.capture_weights import load_type_weights

        type_weights = load_type_weights(cfg)

    saved: list[str] = []
    saved_titles: list[str] = []
    skipped_dup = 0
    reconciled = 0
    skipped_quality = 0
    uncertain = 0
    retyped = 0
    for cand in insights:
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
        try:
            rec = mem.save(
                content=cand["body"],
                title=cand["title"],
                type_=cand["type"],
                tags=tags,
                extra={"capture_confidence": round(confidence, 3)},
            )
            saved.append(rec.id)
            saved_titles.append(rec.title)
            if debug:
                print(f"# memo capture: saved [{rec.id[:8]}] {rec.title}", file=sys.stderr)
        except Exception as exc:
            if debug:
                print(f"# memo capture: save failed: {exc}", file=sys.stderr)

    return {
        "candidates": n_extracted,
        "saved": saved,
        "saved_titles": saved_titles,
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
    result = _extract_and_save(mem, cfg, "", text, debug=debug, merge_tags=merge_tags)
    if result["candidates"] == 0:
        rec = mem.save(
            content=text,
            title=title,
            type_=type_,
            tags=list(merge_tags) if merge_tags else None,
        )
        return {
            "status": "verbatim",
            "candidates": 0,
            "saved": [rec.id],
            "saved_titles": [rec.title],
            "skipped_dup": 0,
            "reconciled": 0,
            "skipped_quality": 0,
            "skipped_meta": 0,
            "skipped_batch_dup": 0,
            "uncertain": 0,
            "retyped": 0,
        }
    return {"status": "extracted", **result}


def run_capture(
    transcript_path: Path,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """Top-level entry: read transcript, extract, dedup, save.
    Returns a result summary dict so the CLI can print + the tests can
    assert on counts. All errors absorbed (logged to stderr in debug).

    Env vars:
      MEMO_CAPTURE_CONTEXT_TURNS  — number of recent exchanges to include
          as context for the LLM (default 3). Higher = richer context but
          longer prompt; lower = cheaper, may miss multi-turn decisions.
      MEMO_CAPTURE_COOLDOWN_MIN   — minimum minutes between captures in the
          same session (default 0 = no cooldown). Set to e.g. 30 to avoid
          flooding the corpus during a long refactoring session.
    """
    import time

    from memo.config import Config
    from memo.flags import flag_int
    from memo.memory import Memory

    cfg = Config.from_env()
    state = _load_state(cfg.state_dir)

    # Cooldown: skip if we saved too recently.
    cooldown_min = float(flag_int("MEMO_CAPTURE_COOLDOWN_MIN") or 0)
    if cooldown_min > 0:
        last_save_ts = state.get("last_save_ts", 0.0)
        elapsed_min = (time.time() - float(last_save_ts)) / 60.0
        if elapsed_min < cooldown_min:
            if debug:
                print(
                    f"# memo capture: cooldown — {elapsed_min:.1f}m elapsed, need {cooldown_min}m",
                    file=sys.stderr,
                )
            return {"status": "cooldown"}

    context_turns = max(1, flag_int("MEMO_CAPTURE_CONTEXT_TURNS") or 3)
    pair = _read_recent_exchanges(transcript_path, n=context_turns)
    if pair is None:
        return {"status": "no_pair"}
    user_text, assistant_text = pair

    # Idempotence — hash the assistant text of the LAST turn only (not all
    # turns in the context window) so multi-turn context doesn't break the
    # duplicate-detection across successive Stop firings.
    last_pair = _read_recent_exchanges(transcript_path, n=1)
    h = _hash_assistant(last_pair[1] if last_pair else assistant_text)
    if state.get("last_hash") == h:
        return {"status": "duplicate_turn"}

    if not _passes_prefilter(assistant_text):
        # Stamp the state anyway so we don't keep re-checking the same turn.
        state["last_hash"] = h
        _save_state(cfg.state_dir, state)
        return {"status": "no_trigger"}

    # Lazy heavy imports: only paid past pre-filter.
    mem = Memory(cfg)
    try:
        result = _extract_and_save(mem, cfg, user_text, assistant_text, debug=debug)
    finally:
        mem.close()

    state["last_hash"] = h
    if result["saved"]:
        state["last_save_ts"] = time.time()
    _save_state(cfg.state_dir, state)
    return {"status": "ok", **result}


# ── Incremental capture (mid-session) ───────────────────────────────────────
#
# The Stop hook only fires once, at session end. A long-running session can
# accumulate hours of durable insight that never reaches `.md` (and thus the
# local index + git sync) until Stop — and is lost entirely on a crash.
#
# Incremental capture closes that gap: a periodic, self-throttled pass mines
# only the NEW turns since a per-session watermark, reusing the exact
# extract → quality → dedup → save pipeline above. The watermark
# (`state_dir/.capture_watermark/<session_id>.json`) records how many
# (user, assistant) exchanges have been processed, so each pass is bounded to
# the turns added since the last pass and old turns are never reprocessed.


def list_sessions_without_watermark(
    state_dir: Path,
    sessions: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return sessions that have no watermark (never captured).

    Filters a list of session dicts from list_sessions() against the
    watermark directory. Returns up to `limit` sessions that have never
    been captured, sorted by most recent first.
    """
    wm_dir = state_dir / ".capture_watermark"
    if not wm_dir.is_dir():
        return sessions[:limit]

    pending: list[dict[str, Any]] = []

    for sess in sessions:
        sid = sess.get("session_id")
        if not sid:
            continue
        wm_file = wm_dir / f"{sid}.json"
        if not wm_file.is_file():
            pending.append(sess)
            if len(pending) >= limit:
                break

    return pending


def _watermark_file(state_dir: Path, session_id: str) -> Path:
    # session_id is a Claude Code UUID — filename-safe, matching how
    # session.py keys its per-session JSON files.
    return state_dir / ".capture_watermark" / f"{session_id}.json"


def _load_watermark(state_dir: Path, session_id: str) -> dict[str, Any]:
    """Per-session watermark, or {} on missing/corrupt (so a clobbered or
    hand-edited file degrades to a fresh full pass, never a crash)."""
    f = _watermark_file(state_dir, session_id)
    if not f.is_file():
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _capture_lock_file(state_dir: Path, session_id: str) -> Path:
    return state_dir / ".capture_watermark" / f"{session_id}.capture.lock"


def _save_watermark(state_dir: Path, session_id: str, watermark: dict[str, Any]) -> None:
    f = _watermark_file(state_dir, session_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write (see _save_state): never leave a torn watermark behind.
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(watermark), encoding="utf-8")
    os.replace(tmp, f)


def incremental_tick_due(state_dir: Path, session_id: str, interval_s: int) -> bool:
    """True if at least `interval_s` seconds elapsed since this session's last
    incremental pass (the watermark `updated` stamp).

    Cheap by design — a small JSON read, no transcript parse and no Memory /
    MLX — so the per-prompt hook can call it on every prompt and bail when not
    due. `interval_s <= 0` disables the throttle (always due)."""
    if interval_s <= 0:
        return True
    import time

    wm = _load_watermark(state_dir, session_id)
    try:
        last = float(wm.get("updated", 0) or 0)
    except (TypeError, ValueError):
        last = 0.0
    return (time.time() - last) >= interval_s


def run_capture_incremental(
    transcript_path: Path,
    session_id: str,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """Mine only NEW turns since this session's watermark, then advance it.

    Reuses the Stop-hook extract/dedup/save pipeline (`_extract_and_save`);
    the watermark is what makes it incremental. Bounded to the exchanges added
    since the previous pass; old turns are never reprocessed. Self-throttling
    is the caller's job (`incremental_tick_due`) — this always processes what
    is new. Soft-fail: returns a status dict, never raises.

    Statuses: ``no_pair`` (empty/unreadable transcript), ``no_new`` (watermark
    already current), ``no_trigger`` (new turns but no insight keyword), ``ok``.
    """
    import time

    from memo.config import Config
    from memo.memory import Memory

    cfg = Config.from_env()

    # Cross-process lock: the idle-capture daemon and the MCP server's
    # _auto_capture can both run this for the same session concurrently. Without
    # a lock they race the load-watermark→process→stamp cycle and double-save.
    # Hold an exclusive, non-blocking flock for the whole cycle; if another
    # process already holds it, skip this run cleanly — it advances the
    # watermark, and the next due tick picks up anything newer.
    lock_path = _capture_lock_file(cfg.state_dir, session_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_fh:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {"status": "locked"}

        exchanges = _parse_exchanges(transcript_path)
        if not exchanges:
            return {"status": "no_pair"}

        total = len(exchanges)
        wm = _load_watermark(cfg.state_dir, session_id)
        try:
            start = int(wm.get("exchange_count", 0) or 0)
        except (TypeError, ValueError):
            start = 0
        # A negative watermark is invalid — reset to beginning.
        # A watermark ahead of the transcript means nothing new; clamp to total.
        if start < 0:
            start = 0
        elif start > total:
            start = total

        def _stamp() -> None:
            _save_watermark(
                cfg.state_dir,
                session_id,
                {"session_id": session_id, "exchange_count": total, "updated": time.time()},
            )

        new = exchanges[start:]
        if not new:
            _stamp()  # refresh `updated` so the throttle clock advances
            return {"status": "no_new", "exchange_count": total}

        combined_user = "\n\n---\n\n".join(u for u, _ in new)
        combined_assistant = "\n\n---\n\n".join(a for _, a in new)

        if not _passes_prefilter(combined_assistant):
            # Advance past these triggerless turns so we don't re-scan them.
            _stamp()
            return {"status": "no_trigger", "exchange_count": total}

        mem = Memory(cfg)
        try:
            result = _extract_and_save(mem, cfg, combined_user, combined_assistant, debug=debug)
        finally:
            mem.close()
        _stamp()
        return {"status": "ok", "processed_turns": len(new), "exchange_count": total, **result}
