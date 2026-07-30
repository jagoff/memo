"""Search-ranking helpers for repo_index.py: hit construction, RRF fusion, path boosting.

Extracted to keep repo_index.py under 800 lines. Import via repo_index.py — not
intended for direct use outside the repo-indexing subsystem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
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
    scope: str = "production"
    channel_scores: dict[str, float] = field(default_factory=dict)
    rank_explanation: dict[str, Any] = field(default_factory=dict)
    index_generation: str = ""

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
            "scope": self.scope,
            "channel_scores": dict(self.channel_scores),
            "rank_explanation": dict(self.rank_explanation),
            "index_generation": self.index_generation,
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
            scope=str(r.get("scope") or classify_repo_path(str(r.get("path") or ""))),
            channel_scores={
                str(key): float(value)
                for key, value in dict(r.get("channel_scores") or {}).items()
            },
            rank_explanation=dict(r.get("rank_explanation") or {}),
            index_generation=str(r.get("index_generation") or ""),
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
        explanation = dict(h.rank_explanation)
        explanation["retrieval_boost"] = round(float(b), 6)
        boosted.append(
            (
                replace(h, score=new_score, rank_explanation=explanation),
                new_score,
            )
        )
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

_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__|specs?|fixtures?|testdata)(/|$)"
    r"|(^|/)(test_[^/]+|[^/]+_(test|spec)|[^/]+\.(test|spec))\.[^/]+$",
    flags=re.IGNORECASE,
)
_VENDOR_PATH_RE = re.compile(
    r"(^|/)(vendor|vendored|third[_-]?party|node_modules|dist|build|generated|"
    r"coverage|target)(/|$)",
    flags=re.IGNORECASE,
)


def classify_repo_path(path: str) -> str:
    """Classify a repository path into the supported retrieval scopes."""
    clean = str(path or "").replace("\\", "/").lstrip("/")
    if _VENDOR_PATH_RE.search(clean):
        return "vendor"
    if _TEST_PATH_RE.search(clean):
        return "tests"
    return "production"


def path_in_repo_scope(path: str, scope: str) -> bool:
    normalized = str(scope or "all").strip().lower()
    if normalized == "all":
        return True
    if normalized not in {"production", "tests", "vendor"}:
        raise ValueError(
            f"invalid repo scope {scope!r}; expected all, production, tests, or vendor"
        )
    return classify_repo_path(path) == normalized


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
    channel_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    fused: dict[str, float] = {}
    canon: dict[str, dict[str, Any]] = {}
    contributions: dict[str, dict[str, float]] = {}
    ranks: dict[str, dict[str, int]] = {}
    channel_details: dict[str, dict[str, Any]] = {}
    for channel_index, hits in enumerate(hit_lists):
        channel = (
            channel_names[channel_index]
            if channel_names is not None and channel_index < len(channel_names)
            else f"channel_{channel_index + 1}"
        )
        for rank, hit in enumerate(hits):
            rid = hit["id"]
            contribution = 1.0 / (k + rank + 1)
            fused[rid] = fused.get(rid, 0.0) + contribution
            contributions.setdefault(rid, {})[channel] = contribution
            ranks.setdefault(rid, {})[channel] = rank + 1
            detail: dict[str, Any] = {}
            if hit.get("provider_evidence"):
                detail["evidence"] = list(hit["provider_evidence"])
            if hit.get("provider_metadata"):
                detail["metadata"] = dict(hit["provider_metadata"])
            if detail:
                channel_details.setdefault(rid, {})[channel] = detail
            canon.setdefault(rid, hit)
    path_boosts: dict[str, float] = {}
    if query_terms:
        for rid, score in list(fused.items()):
            boost = _path_name_boost(canon[rid].get("path") or "", query_terms)
            if boost:
                fused[rid] = score * (1.0 + boost)
                path_boosts[rid] = boost
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    out: list[dict[str, Any]] = []
    for rid, score in ranked:
        d = dict(canon[rid])
        d["score"] = score
        d["match_type"] = "hybrid" if len(contributions[rid]) > 1 else next(
            iter(contributions[rid]), d.get("match_type") or "chunk"
        )
        d["channel_scores"] = {
            channel: round(value, 8)
            for channel, value in sorted(contributions[rid].items())
        }
        d["rank_explanation"] = {
            "fusion": "rrf",
            "rrf_k": k,
            "channel_ranks": dict(sorted(ranks[rid].items())),
            "path_name_boost": path_boosts.get(rid, 0.0),
            "channel_details": channel_details.get(rid, {}),
        }
        d["scope"] = classify_repo_path(str(d.get("path") or ""))
        out.append(d)
    return out
