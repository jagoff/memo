"""Search-ranking helpers for repo_index.py: hit construction, RRF fusion, path boosting.

Extracted to keep repo_index.py under 800 lines. Import via repo_index.py — not
intended for direct use outside the repo-indexing subsystem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from memo.config import AI_SUBDIR
from memo.retrieval_boost import boost_for as _retrieval_boost_for

# ---------------------------------------------------------------------------
# Public data class (re-exported from repo_index for backward compat)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoSearchHit:
    id: str
    repo_id: str
    repo_name: str
    url: str
    ref: str
    commit_sha: str
    file_id: str
    path: str
    language: str
    line_start: int
    line_end: int
    text: str
    score: float | None
    match_type: str

    @property
    def locator(self) -> str:
        commit = self.commit_sha[:8] if self.commit_sha else "unknown"
        return f"repo:{self.repo_name}:{self.path}:{self.line_start}-{self.line_end}@{commit}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "repo_name": self.repo_name,
            "url": self.url,
            "ref": self.ref,
            "commit_sha": self.commit_sha,
            "file_id": self.file_id,
            "path": self.path,
            "language": self.language,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "text": self.text,
            "score": self.score,
            "match_type": self.match_type,
            "locator": self.locator,
        }


# ---------------------------------------------------------------------------
# Hit construction
# ---------------------------------------------------------------------------


def _hits_from_rows(rows: list[dict[str, Any]]) -> list[RepoSearchHit]:
    return [
        RepoSearchHit(
            id=r["id"],
            repo_id=r["repo_id"],
            repo_name=r["repo_name"],
            url=r["url"],
            ref=r["ref"],
            commit_sha=r["commit_sha"],
            file_id=r["file_id"],
            path=r["path"],
            language=r.get("language") or "",
            line_start=int(r["line_start"]),
            line_end=int(r["line_end"]),
            text=r.get("body_text") or "",
            score=r.get("score"),
            match_type=r.get("match_type") or "chunk",
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Boost and re-sort
# ---------------------------------------------------------------------------


def _boost_and_resort(
    hits: list[RepoSearchHit],
    *,
    query: str,
    limit: int,
) -> list[RepoSearchHit]:
    """Apply filename/title/heading boost to each hit's score, re-sort,
    and truncate to ``limit``. Hits with ``None`` score keep their order
    (no boost applied to unknown scores).

    Boost is multiplicative on the existing hybrid score, so callers
    that compare scores across queries still get monotonic-by-query
    ordering. The boost field is not surfaced on the dataclass — it's
    folded into ``score`` so downstream consumers see a unified number.
    """
    if not hits or not query:
        return hits[:limit]

    boosted: list[tuple[RepoSearchHit, float]] = []
    for h in hits:
        if h.score is None:
            boosted.append((h, -1.0))
            continue
        b = _retrieval_boost_for(
            query=query,
            filename=h.path,
            title="",
            headings=[],
            tags=[],
        )
        new_score = float(h.score) * b
        boosted.append((replace(h, score=new_score), new_score))
    boosted.sort(key=lambda pair: pair[1], reverse=True)
    return [pair[0] for pair in boosted[:limit]]


# ---------------------------------------------------------------------------
# Path-name boost helpers
# ---------------------------------------------------------------------------

# Paths that should NEVER get a filename boost — these are ingest dumps
# (e.g. raw Claude/agent transcripts) that mention canonical names many
# times but are not themselves the canonical source. Boosting them
# defeats the purpose of preferring `Contacts/Grecia.md` over a dump
# whose filename happens to also include "Grecia".
_INGEST_PATH_MARKERS = (
    f"{AI_SUBDIR}/external-ingest/",
    "external-ingest/",
)

# Lightweight stopword + token-min-length guard. We strip punctuation,
# lowercase, and drop tokens shorter than 3 chars so noise like "es",
# "de", or "?" doesn't trigger spurious path-name boosts.
_QUERY_TERM_MIN_LEN = 3
_QUERY_TERM_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "que",
        "los",
        "las",
        "una",
        "del",
        "como",
        "qué",
        "cuál",
        "quién",
        "donde",
        "cuando",
        "este",
        "esta",
    }
)


def _extract_query_terms(query: str) -> list[str]:
    """Tokenize a query into significant terms for path-name boosting.

    Lowercased, punctuation-stripped, stopwords + short tokens removed.
    Used to compare query intent against path basenames in the post-RRF
    boost — not used for FTS5 matching (FTS5 has its own tokenizer).
    """
    if not query:
        return []
    raw = re.split(r"[^\w]+", query.lower(), flags=re.UNICODE)
    return [
        tok for tok in raw if len(tok) >= _QUERY_TERM_MIN_LEN and tok not in _QUERY_TERM_STOPWORDS
    ]


def _path_name_boost(path: str, terms: list[str]) -> float:
    """Compute a boost in [0.0, 1.0] for filename-match relevance.

    1.0: basename (sans extension) matches a query term exactly.
    0.5: basename contains a query term as substring.
    0.0: no match, OR path lies under an ingest-dump prefix (noisy).
    """
    if not path or not terms:
        return 0.0
    if any(marker in path for marker in _INGEST_PATH_MARKERS):
        return 0.0
    basename = Path(path).stem.lower()  # strips dir + extension
    if not basename:
        return 0.0
    for term in terms:
        if basename == term:
            return 1.0
    for term in terms:
        if term in basename:
            return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def _rrf_fuse_repo(
    hit_lists: list[list[dict[str, Any]]],
    *,
    limit: int,
    k: int = 60,
    query_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    fused: dict[str, float] = {}
    canon: dict[str, dict[str, Any]] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits):
            rid = hit["id"]
            fused[rid] = fused.get(rid, 0.0) + 1.0 / (k + rank + 1)
            canon.setdefault(rid, hit)
    if query_terms:
        for rid, score in list(fused.items()):
            boost = _path_name_boost(canon[rid].get("path") or "", query_terms)
            if boost:
                fused[rid] = score * (1.0 + boost)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    out: list[dict[str, Any]] = []
    for rid, score in ranked:
        d = dict(canon[rid])
        d["score"] = score
        out.append(d)
    return out
