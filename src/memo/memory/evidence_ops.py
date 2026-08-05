"""EvidencePack construction and deterministic abstention."""

from __future__ import annotations

import re
from typing import Any

from memo.contracts import (
    AnswerStatus,
    EvidenceItem,
    EvidencePack,
    TrustTier,
    normalize_provenance,
)
from memo.flags import flag_bool, flag_float
from memo.memory._base import _MemoryBase
from memo.memory.evidence_graph_compact import compact_by_entity_overlap

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        # English question scaffolding and function words.
        "an",
        "and",
        "are",
        "been",
        "being",
        "did",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "into",
        "is",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        # Spanish equivalents. Memo accepts multilingual questions, so these
        # must not dilute coverage of the actual subject terms.
        "al",
        "como",
        "con",
        "cual",
        "cuando",
        "de",
        "del",
        "donde",
        "el",
        "en",
        "era",
        "es",
        "esta",
        "fue",
        "la",
        "las",
        "los",
        "para",
        "por",
        "que",
        "quien",
        "se",
        "sin",
        "son",
        "sus",
        "una",
        "unas",
        "uno",
        "unos",
    }
)
_MIN_ANSWER_CONFIDENCE = 0.4


def _tokens(value: str) -> set[str]:
    return {
        normalized
        for token in _TOKEN_RE.findall(value)
        if len(normalized := token.casefold()) > 1 and normalized not in _STOPWORDS
    }


def _calibrated_relevance(
    raw_score: float | None,
    *,
    question_tokens: set[str],
    item_tokens: set[str],
) -> float:
    """Map one hit to a fixed, query-grounded relevance scale.

    Search scores are absolute within their retrieval mode (hybrid mode uses
    RRF), so normalizing by the best hit fabricates certainty on a weak result
    set. A fixed blend retains that absolute signal while lexical coverage
    gives exact matches useful weight without depending on the other hits.
    """
    raw = min(1.0, max(0.0, float(raw_score or 0.0)))
    lexical = len(question_tokens & item_tokens) / max(1, len(question_tokens))
    return min(1.0, 0.65 * raw + 0.35 * lexical)


def _trust(extra: dict[str, Any]) -> TrustTier:
    raw = str(extra.get("trust_tier") or TrustTier.AGENT_INFERRED.value)
    try:
        return TrustTier(raw)
    except ValueError:
        return TrustTier.EXTERNAL_UNTRUSTED


def _validate_request(question: str, *, k: int, max_chars: int, min_coverage: float) -> str:
    normalized = question.strip()
    if not normalized:
        raise ValueError("question cannot be empty")
    if k < 1:
        raise ValueError("k must be >= 1")
    if max_chars < 256:
        raise ValueError("max_chars must be >= 256")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")
    return normalized


def _absorbed_coverage_tokens(hits: list[Any], compacted: list[Any]) -> set[str]:
    """Lexical tokens (title+body) of hits `compact_by_entity_overlap` removed,
    so a collapsed representative's citation doesn't silently reduce measured
    question coverage for content that was legitimately retrieved."""
    survivors = {getattr(h, "id", None) for h in compacted}
    tokens: set[str] = set()
    for hit in hits:
        if getattr(hit, "id", None) in survivors:
            continue
        body = str(hit.body or "").strip()
        title = str(hit.title or "").strip()
        tokens.update(_tokens(f"{title} {body}"))
    return tokens


def _build_items(
    hits: list[Any],
    *,
    question: str,
    max_chars: int,
    coverage_credit_tokens: set[str] | None = None,
) -> tuple[list[EvidenceItem], float, set[str]]:
    remaining = max_chars
    items: list[EvidenceItem] = []
    covered: set[str] = set()
    question_tokens = _tokens(question)
    selected_ids: set[str] = set()
    for hit in hits:
        body = str(hit.body or "").strip()
        title = str(hit.title or "").strip()
        if remaining <= 0:
            break
        snippet = (body or title)[:remaining]
        if not snippet:
            continue
        remaining -= len(snippet)
        extra = dict(hit.extra or {})
        item_tokens = _tokens(f"{title} {snippet}")
        covered.update(question_tokens & item_tokens)
        selected_ids.add(hit.id)
        items.append(
            EvidenceItem(
                id=hit.id,
                uri=f"memo://memoria/{hit.id}",
                title=title,
                snippet=snippet,
                score=round(
                    _calibrated_relevance(
                        hit.score,
                        question_tokens=question_tokens,
                        item_tokens=item_tokens,
                    ),
                    6,
                ),
                source="memory",
                type=hit.type,
                valid_at=hit.valid_at,
                invalid_at=hit.invalid_at,
                trust_tier=_trust(extra),
                provenance=normalize_provenance(extra),
            )
        )
    covered.update(question_tokens & (coverage_credit_tokens or set()))
    coverage = len(covered) / max(1, len(question_tokens))
    return items, coverage, selected_ids


def _confidence(items: list[EvidenceItem], coverage: float) -> float:
    if not items:
        return 0.0
    top_scores = [float(item.score or 0.0) for item in items[:3]]
    relevance = sum(top_scores) / len(top_scores)
    trust_weights = {
        TrustTier.HUMAN: 1.0,
        TrustTier.TOOL_OBSERVED: 0.95,
        TrustTier.AGENT_VERIFIED: 0.9,
        TrustTier.AGENT_INFERRED: 0.7,
        TrustTier.EXTERNAL_UNTRUSTED: 0.35,
    }
    top_items = items[:3]
    trust = sum(trust_weights[item.trust_tier] for item in top_items) / len(top_items)
    # Trust can strengthen relevant evidence, but must never create a confidence
    # floor for an irrelevant hit merely because its author is trusted.
    return min(1.0, relevance * (0.55 + 0.15 * trust) + 0.3 * coverage)


def _answer_status(
    *,
    conflicts: list[dict[str, Any]],
    items: list[EvidenceItem],
    coverage: float,
    min_coverage: float,
    confidence: float,
    min_confidence: float = _MIN_ANSWER_CONFIDENCE,
) -> tuple[AnswerStatus, str]:
    if not items:
        return AnswerStatus.INSUFFICIENT_EVIDENCE, "no usable evidence items were retrieved"
    if coverage < min_coverage:
        return (
            AnswerStatus.INSUFFICIENT_EVIDENCE,
            f"question coverage {coverage:.2f} is below required {min_coverage:.2f}",
        )
    if confidence < min_confidence:
        return (
            AnswerStatus.INSUFFICIENT_EVIDENCE,
            f"evidence confidence {confidence:.2f} is below required {min_confidence:.2f}",
        )
    if conflicts:
        return (
            AnswerStatus.CONFLICTED,
            "retrieved evidence contains an unresolved judged conflict",
        )
    return AnswerStatus.ANSWERED, ""


def _insufficient_evidence_pack(question: str) -> EvidencePack:
    return EvidencePack(
        question=question,
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        queries=(question,),
        abstention_reason="no relevant memories found",
    )


def _selected_conflicts(
    relations: list[dict[str, Any]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    return [
        row
        for row in relations
        if row.get("relation") == "conflicts_with"
        and str(row.get("source_id") or "") in selected_ids
        and str(row.get("target_id") or "") in selected_ids
    ]


def _claims(items: list[EvidenceItem]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "memory_id": item.id,
            "claim": item.title,
            "evidence_uris": [item.uri],
        }
        for item in items
        if item.title
    )


def _pack_from_items(
    *,
    question: str,
    items: list[EvidenceItem],
    coverage: float,
    conflicts: list[dict[str, Any]],
    min_coverage: float,
) -> EvidencePack:
    confidence = _confidence(items, coverage)
    status, reason = _answer_status(
        conflicts=conflicts,
        items=items,
        coverage=coverage,
        min_coverage=min_coverage,
        confidence=confidence,
    )
    return EvidencePack(
        question=question,
        status=status,
        items=tuple(items),
        queries=(question,),
        claims=_claims(items),
        confidence=round(confidence, 4),
        coverage=round(coverage, 4),
        token_estimate=sum(len(item.snippet) for item in items) // 4,
        abstention_reason=reason,
    )


class _EvidenceOpsMixin(_MemoryBase):
    def evidence_pack(
        self,
        question: str,
        *,
        k: int = 8,
        max_chars: int = 12_000,
        min_coverage: float = 0.2,
        type_: str | None = None,
        as_of: str | None = None,
    ) -> EvidencePack:
        """Retrieve bounded evidence and abstain when grounding is insufficient."""
        question = _validate_request(
            question,
            k=k,
            max_chars=max_chars,
            min_coverage=min_coverage,
        )

        hits = self.search(
            question,
            limit=k,
            type_=type_,
            as_of=as_of,
            recency=True,
            quality_rerank=True,
        )
        if not hits:
            return _insufficient_evidence_pack(question)

        absorbed_tokens: set[str] = set()
        if flag_bool("MEMO_EVIDENCE_GRAPH_COMPACT"):
            compacted = compact_by_entity_overlap(
                hits,
                self,
                min_idf_overlap=flag_float("MEMO_EVIDENCE_GRAPH_COMPACT_MIN_IDF") or 0.5,
            )
            absorbed_tokens = _absorbed_coverage_tokens(hits, compacted)
            hits = compacted

        items, coverage, selected_ids = _build_items(
            hits,
            question=question,
            max_chars=max_chars,
            coverage_credit_tokens=absorbed_tokens,
        )
        relations = (
            self.store.list_relations(
                status="judged",
                memory_ids=sorted(selected_ids),
                limit=min(1000, max(100, len(selected_ids) ** 2)),
            )
            if selected_ids
            else []
        )
        conflicts = _selected_conflicts(relations, selected_ids)
        return _pack_from_items(
            question=question,
            items=items,
            coverage=coverage,
            conflicts=conflicts,
            min_coverage=min_coverage,
        )


__all__ = ["_EvidenceOpsMixin"]
