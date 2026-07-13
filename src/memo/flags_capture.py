"""Crusher flags for JSON array compression on ingest.

Wave 1 token economy feature: L1 (SmartCrusher) drops low-relevance JSON rows
before indexing, saving 60–92% on structured data without losing retrieval quality.
"""

from __future__ import annotations

from memo.flags_base import FlagSpec, _spec

SPECS: tuple[FlagSpec, ...] = (
    _spec(
        "MEMO_CRUSHER_ENABLED",
        "bool",
        False,
        "capture",
        "Enable JSON array crushing on ingest (L1 SmartCrusher). When on, large JSON "
        "arrays are pruned before indexing, keeping only high-relevance rows and "
        "caching originals for later retrieval. Default: OFF (opt-in) until the "
        "committed token-quality gate is intentionally promoted. Set to 1 to enable.",
    ),
    _spec(
        "MEMO_CRUSHER_ROWS_KEEP_RATIO",
        "float",
        0.2,
        "capture",
        "Fraction of JSON array rows to keep after crushing (default: 0.2 = top 20%). "
        "Clamped to [0.05, 1.0]. Value of 1.0 disables crushing (keeps all rows).",
        min_val=0.05,
        max_val=1.0,
    ),
    _spec(
        "MEMO_CRUSHER_CACHE_TTL_DAYS",
        "int",
        30,
        "capture",
        "TTL for crush cache entries in days (default: 30). Older entries are evicted "
        "during maintenance. Minimum: 1 day.",
        min_val=1,
    ),
    _spec(
        "MEMO_CAPTURE_RECEIPT",
        "bool",
        False,
        "capture",
        "Surface a multi-line receipt (titles+ids+undo/fix verbs) for auto-saved "
        "memories, instead of the muted one-liner. Visible + correctable capture.",
    ),
    _spec(
        "MEMO_GROUNDING_JUDGE",
        "bool",
        False,
        "capture",
        "At capture, score source->claim entailment with the helper LLM (temp 0, "
        "off the recall hook). A candidate scoring below MEMO_GROUNDING_WRITE_MIN "
        "is tagged '_uncertain' (already recall-excluded, see "
        "MEMO_RECALL_EXCLUDE_UNCERTAIN) and its grounding_score is stamped in the "
        ".md extra bag. Default off.",
    ),
    _spec(
        "MEMO_GROUNDING_WRITE_MIN",
        "float",
        0.4,
        "capture",
        "Grounding-score floor (0.0-1.0) at capture. A candidate whose source->claim "
        "entailment is below this is quarantined as '_uncertain'. Only active when "
        "MEMO_GROUNDING_JUDGE is on.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_CLAIM_SUPPORT",
        "bool",
        False,
        "capture",
        "At capture, an outcome claim (works/shipped/faster/secure/tested) with no "
        "evidence ref — or a commit:<sha> that does not exist locally — DOWNGRADES "
        "the saved memory's confidence to MEMO_CLAIM_SUPPORT_CONFIDENCE (never drops "
        "it). Pure/no-LLM. Default off.",
    ),
    _spec(
        "MEMO_CLAIM_SUPPORT_CONFIDENCE",
        "float",
        0.5,
        "capture",
        "Absolute confidence stamped on an unsupported outcome claim (see "
        "MEMO_CLAIM_SUPPORT) so it ranks below grounded notes (score x confidence). "
        "Only active when MEMO_CLAIM_SUPPORT is on.",
        min_val=0.1,
        max_val=1.0,
    ),
    _spec(
        "MEMO_SAVE_GATE_PRESETS",
        "str",
        "",
        "capture",
        "JSON map of memory type -> save-gate preset (strict/balanced/permissive), "
        "e.g. '{\"decision\":\"strict\",\"bug\":\"strict\"}'. A 'strict' type REFUSES "
        "a near-duplicate save (ValueError with the colliding id) instead of warning. "
        "Unset/unlisted types = 'balanced' = today's behavior. Default off (empty).",
    ),
    _spec(
        "MEMO_RECAP",
        "bool",
        True,
        "capture",
        "Emit a cross-client '※ memo recap: <goal/progress>' line (dim ANSI, mirrors "
        "Claude Code's native recap) via the pending-notification channel every "
        "client already reads (the `notification` field on memo_search/"
        "memo_ask/memo_chat_ask/memo_context/memo_unified_briefing). Sourced "
        "from the existing session snapshot summary — no extra LLM call. "
        "Default on; disable with MEMO_RECAP=0.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RECAP_EVERY_N",
        "int",
        6,
        "capture",
        "Turns between recap lines for a session (default: 6). 0 or negative "
        "disables recap regardless of MEMO_RECAP.",
        min_val=0,
    ),
)


def flag_crusher_enabled() -> bool:
    """Enable JSON array crushing on ingest (default: OFF until scorer real + gated)."""
    from memo.flags import flag_bool
    return flag_bool("MEMO_CRUSHER_ENABLED")


def flag_crusher_keep_ratio() -> float:
    """Fraction of JSON array rows to keep (default: 0.2 = top 20%).

    Returns float, clamped to [0.05, 1.0].
    """
    from memo.flags import flag_float
    val = flag_float("MEMO_CRUSHER_ROWS_KEEP_RATIO")
    return 0.2 if val is None else max(0.05, min(1.0, val))  # Clamp [0.05, 1.0]


def flag_crusher_cache_ttl_days() -> int:
    """TTL for crush cache entries (default: 30 days).

    Returns int >= 1.
    """
    from memo.flags import flag_int
    val = flag_int("MEMO_CRUSHER_CACHE_TTL_DAYS")
    return 30 if val is None else max(1, val)
