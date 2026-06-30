"""Pure entity-name canonicalization for graph de-fragmentation.

`canonical_key` normalizes a name into a comparison bucket (lower, accent-free,
no punctuation/separators) so "FastAPI" / "fast api" / "fast-api" collapse.
`fold_key` adds a tiny curated alias map for variants normalization can't catch
("postgres" vs "postgresql"). These are GROUPING keys only — display names keep
their original lower-cased spelling. Dependency-free; no imports from memo.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["canonical_key", "fold_key"]

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SEP_RE = re.compile(r"[\s\-_./]+")

# canonical_key(variant) -> canonical_key(canonical). Keep this SMALL (YAGNI):
# only high-frequency variants that pure normalization cannot fold.
_ALIAS_MAP: dict[str, str] = {
    "postgres": "postgresql",
    "postgre": "postgresql",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
}


def canonical_key(name: str) -> str:
    """Fold a name into a comparison bucket: lower, accent-stripped, no
    punctuation or separators. 'Fast-API' -> 'fastapi'."""
    s = unicodedata.normalize("NFKD", name or "")
    s = s.encode("ascii", "ignore").decode("ascii").lower().strip()
    s = _PUNCT_RE.sub("", s)
    s = _SEP_RE.sub("", s)
    return s


def fold_key(name: str) -> str:
    """canonical_key plus the curated alias collapse."""
    base = canonical_key(name)
    return _ALIAS_MAP.get(base, base)
