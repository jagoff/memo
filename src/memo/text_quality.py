"""Source-agnostic text-quality signal for ingest.

Generalizes the OCR confidence gate (`ocr.py`) to ANY ingested text: PDF via
pdftotext, future OCR engines, encoding-broken imports. Produces a record-level
health confidence that the indexer stamps into ``memory_health.confidence`` (the
same lever images use via :func:`ocr.image_health_confidence`), so garbled text
ranks below clean notes (search score x confidence in
``memory/search_ops.py`` ``_apply_health_scores``).

DESIGN — conservative on purpose. We gate ONLY on signals that legitimate text
never produces: Unicode replacement chars (``�`` / ``�``) and runs of
non-printable control bytes. These are unambiguous mojibake markers — the
original garbled screenshots carried them (``U�t*� St•tqSllf``).
We deliberately do NOT use dictionary / digit-in-word / "gibberish-ratio"
heuristics: a corpus scan showed those flag legit notes (Scrum courses, guitar
tabs, code, identifiers) at ~100% false-positive while catching zero real
garbage. False-positively down-weighting a good note is worse than missing
garbage the OCR/PDF layer already cleaned. Validate any change against
``eval/regression_labels.json`` — precision@K must not drop.
"""

from __future__ import annotations

import unicodedata

__all__ = [
    "garbage_ratio",
    "text_health_confidence",
    "text_quality_enabled",
]

# Unambiguous mojibake markers. U+FFFD is the replacement char; the rest are
# substitution glyphs Vision/pdftotext emit on undecodable runs.
_REPLACEMENT_CHARS = "�￼"


def text_quality_enabled() -> bool:
    """Whether ingest stamps a text-quality confidence on records. Default on;
    ``MEMO_TEXT_QUALITY=0`` disables."""
    from memo.flags import flag_bool

    return flag_bool("MEMO_TEXT_QUALITY")


def _threshold() -> float:
    """Garbage ratio at/above which a record is down-weighted. Default 0.02
    (2% replacement/control chars — far above any clean text, which is ~0).
    ``MEMO_TEXT_QUALITY_THRESHOLD`` overrides; 0 disables."""
    from memo.flags import flag_float

    value = flag_float("MEMO_TEXT_QUALITY_THRESHOLD")
    return 0.02 if value is None else max(0.0, min(1.0, value))


def garbage_ratio(text: str) -> float:
    """Fraction of chars that are unambiguous mojibake: Unicode replacement
    chars or non-printable control codes (excluding ordinary whitespace).
    Pure + FP-free — legitimate text returns ~0.0. Empty text → 0.0."""
    if not text:
        return 0.0
    bad = 0
    for ch in text:
        if ch in _REPLACEMENT_CHARS or (ch not in "\n\r\t" and unicodedata.category(ch) in {"Cc", "Co", "Cn"}):
            bad += 1
    return bad / len(text)


def text_health_confidence(text: str) -> float | None:
    """Map a record's body to a memory-health confidence, or None to leave it
    neutral (1.0). Returns ``1 - garbage_ratio`` (floored 0.1) when the ratio is
    at/above :func:`_threshold`; None otherwise. None when the gate is disabled
    or the threshold is 0."""
    if not text_quality_enabled():
        return None
    thr = _threshold()
    if thr <= 0.0:
        return None
    ratio = garbage_ratio(text)
    if ratio < thr:
        return None
    return max(0.1, 1.0 - ratio)
