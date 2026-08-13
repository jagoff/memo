"""Filename / title / heading / tag boost for hybrid retrieval scoring.

Curatorial signals the user explicitly chose when organizing notes —
filenames, frontmatter titles, headings, tags — carry more precision
per token than body text. Hybrid (vec + BM25) scoring treats them as
plain tokens, so a YouTube transcript that mentions a word 10 times
can out-rank a procedural note whose **filename** is the answer.

This module returns a multiplicative boost ≥ 1.0 to apply to the
existing hybrid score before final sort. Deterministic, no model, no
network — same query always produces the same boost.

The cap (~10×) prevents legitimate distant matches from being buried:
hits with body score 50× the top still surface.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath

__all__ = ["boost_for", "query_terms"]

# Hard cap on the multiplicative boost — keeps a metadata-perfect note from
# burying a much stronger body match.
_MAX_BOOST = 12.0

_TERM_RE = re.compile(r"[\w\-]{3,}", re.UNICODE)
_STOPWORDS_ES_EN = frozenset(
    {
        # Spanish
        "como",
        "para",
        "que",
        "los",
        "las",
        "una",
        "uno",
        "cual",
        "cuales",
        "del",
        "sobre",
        "entre",
        "donde",
        "cuando",
        "porque",
        "porqué",
        "este",
        "esta",
        "estos",
        "estas",
        "ese",
        "esa",
        "esos",
        "esas",
        "con",
        "sin",
        "por",
        "tener",
        "tiene",
        "hace",
        "hacer",
        "puedes",
        "puedo",
        "necesito",
        "quiero",
        "favor",
        # English
        "how",
        "what",
        "which",
        "where",
        "when",
        "why",
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "your",
        "their",
        "have",
        "need",
        "want",
        "please",
        "into",
        "onto",
        "than",
        "then",
    }
)


def _fold_diacritics(s: str) -> str:
    """Strip combining marks (NFKD), mirroring FTS5 unicode61 remove_diacritics=2."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def query_terms(query: str) -> list[str]:
    """Lowercased, diacritic-folded content terms from ``query`` (stopwords +
    tokens <3 chars dropped).

    Folding mirrors FTS5's ``remove_diacritics=2`` tokenizer so the boost is
    diacritic-insensitive on both sides — an accented Spanish query still earns
    the filename/title boost on the note that answers it. Folding happens before
    the stopword check, so accented and unaccented spellings of a stopword
    (``cómo`` / ``como``) are both dropped.

    Order preserved, no dedup. Matches memo's internal query-term selection
    behavior so every retrieval path agrees on what's "significant".
    """
    if not query:
        return []
    return [
        t
        for t in (_fold_diacritics(m.group(0).lower()) for m in _TERM_RE.finditer(query))
        if t not in _STOPWORDS_ES_EN and len(t) >= 3
    ]


def boost_for(
    *,
    query: str,
    filename: str = "",
    title: str = "",
    headings: list[str] | None = None,
    tags: list[str] | None = None,
) -> float:
    """Multiplicative boost ≥ 1.0 based on overlap of query terms with
    high-precision metadata fields.

    Weighting (loosely curatorial-confidence-ordered):
      - Filename (basename without extension) exact match: ×4.0
      - Filename ≥50% match: ×2.0
      - Filename any term match: ×1.3
      - Frontmatter title ≥50% match (and distinct from filename): ×1.5
      - Heading inside chunk with ≥50% match: ×1.25
      - Any tag matches a query term: ×1.4

    Returns 1.0 when query has no significant terms (e.g. all stopwords)
    or no metadata fields match. Hard-capped at ``_MAX_BOOST`` (12×) so a
    metadata-perfect note wins decisively without burying a stronger body
    match. Tests assert these ranges.
    """
    terms = query_terms(query)
    if not terms:
        return 1.0
    boost = 1.0

    fname = _fold_diacritics(PurePosixPath(filename or "").stem.lower())
    fname_hits = 0
    if fname:
        fname_hits = sum(1 for t in terms if t in fname)
        ratio = fname_hits / len(terms)
        if ratio >= 0.99:
            boost *= 4.0
        elif ratio >= 0.5:
            boost *= 2.0
        elif ratio > 0:
            boost *= 1.3

    t_lower = _fold_diacritics((title or "").lower().strip())
    if t_lower and t_lower != fname:
        ratio = sum(1 for t in terms if t in t_lower) / len(terms)
        # Scale with overlap like the filename does — a near-exact frontmatter
        # title is a strong curatorial signal that THIS note is the answer, so it
        # must win decisively, not flat ×1.5 (which tied a 50% match with a
        # full-coverage one and let a terse correct note get blended with noise).
        if ratio >= 0.99:
            boost *= 2.5
        elif ratio >= 0.75:
            boost *= 2.0
        elif ratio >= 0.5:
            boost *= 1.5

    for h in headings or []:
        h_lower = _fold_diacritics((h or "").lower())
        if not h_lower:
            continue
        ratio = sum(1 for t in terms if t in h_lower) / len(terms)
        if ratio >= 0.99:
            boost *= 1.5
            break
        if ratio >= 0.5:
            boost *= 1.25
            break

    for tag in tags or []:
        t_clean = _fold_diacritics((tag or "").lstrip("#").lower().strip())
        if not t_clean:
            continue
        if t_clean in terms or any(t_clean in t or t in t_clean for t in terms):
            boost *= 1.4
            break

    # Explicit cap: a metadata-perfect note wins decisively but never buries a
    # much stronger body match (a hit with body score >cap× the top still surfaces).
    return min(boost, _MAX_BOOST)
