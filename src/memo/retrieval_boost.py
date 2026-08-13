"""Filename / title / heading / tag boost for hybrid retrieval scoring.

Curatorial signals the user explicitly chose when organizing notes —
filenames, frontmatter titles, headings, tags — carry more precision
per token than body text. Hybrid (vec + BM25) scoring treats them as
plain tokens, so a YouTube transcript that mentions a word 10 times
can out-rank a procedural note whose **filename** is the answer.

This module returns a multiplicative boost ≥ 1.0 to apply to the
existing hybrid score before final sort. Deterministic, no model, no
network — same query always produces the same boost.

It is a *prior*, not evidence, and it is applied downstream of stages that
bound the score to 1.0 (`_rerank` fuses `alpha * P(yes) + (1 - alpha) *
rrf_bonus`, both in [0, 1]). So the cap has to stay small enough that the
retrieval evidence still decides the ranking: `_MAX_BOOST` is exactly the
score ratio a candidate needs in order to be immune to metadata alone.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath

__all__ = ["boost_for", "query_terms"]

# Hard cap on the multiplicative boost — keeps a metadata-perfect note from
# burying a much stronger body match. This number IS the guarantee: a candidate
# scoring `_MAX_BOOST`x another cannot be overtaken by curatorial metadata. At
# the historical 12.0 that guarantee was vacuous, because the score this
# multiplies is bounded to 1.0 six stages upstream, and no real body-score
# spread is 12x — the boost simply decided the ranking on its own.
_MAX_BOOST = 1.5

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
      - Filename (basename without extension) exact match: ×1.30
      - Filename ≥50% match: ×1.15
      - Filename any term match: ×1.06
      - Frontmatter title, over the terms the filename did NOT match:
        full ×1.20, ≥75% ×1.14, ≥50% ×1.08
      - Heading inside chunk: exact ×1.10, ≥50% match ×1.05
      - Any tag matches a query term: ×1.08

    The title is scored only on terms the filename missed. memo derives a
    record's title from its own body and its filename from that title, so for a
    self-titled record "filename matches" and "title matches" are one signal
    counted twice — and what that signal encodes is *entity mention*, not
    *answer relevance*. Double-counting it is what let notes whose title merely
    names the query's subject outrank the note that answers the question.

    Returns 1.0 when query has no significant terms (e.g. all stopwords)
    or no metadata fields match. Hard-capped at ``_MAX_BOOST`` so curatorial
    metadata can reorder near-ties without overriding the retrieval evidence.
    Tests assert these ranges.
    """
    terms = query_terms(query)
    if not terms:
        return 1.0
    boost = 1.0

    fname = _fold_diacritics(PurePosixPath(filename or "").stem.lower())
    fname_hits: set[str] = set()
    if fname:
        fname_hits = {t for t in terms if t in fname}
        ratio = len(fname_hits) / len(terms)
        if ratio >= 0.99:
            boost *= 1.30
        elif ratio >= 0.5:
            boost *= 1.15
        elif ratio > 0:
            boost *= 1.06

    t_lower = _fold_diacritics((title or "").lower().strip())
    if t_lower:
        # Only terms the filename left unmatched — see the docstring. Denominator
        # stays len(terms) so the title still scales with how much of the *query*
        # it newly explains, not with how little the filename happened to cover.
        novel = sum(1 for t in terms if t not in fname_hits and t in t_lower)
        ratio = novel / len(terms)
        if ratio >= 0.99:
            boost *= 1.20
        elif ratio >= 0.75:
            boost *= 1.14
        elif ratio >= 0.5:
            boost *= 1.08

    for h in headings or []:
        h_lower = _fold_diacritics((h or "").lower())
        if not h_lower:
            continue
        ratio = sum(1 for t in terms if t in h_lower) / len(terms)
        if ratio >= 0.99:
            boost *= 1.10
            break
        if ratio >= 0.5:
            boost *= 1.05
            break

    for tag in tags or []:
        t_clean = _fold_diacritics((tag or "").lstrip("#").lower().strip())
        if not t_clean:
            continue
        if t_clean in terms or any(t_clean in t or t in t_clean for t in terms):
            boost *= 1.08
            break

    # Explicit cap: curatorial metadata reorders near-ties, but a candidate
    # scoring `_MAX_BOOST`x another is immune to it.
    return min(boost, _MAX_BOOST)
