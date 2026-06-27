from __future__ import annotations

from memo.flags_base import FlagSpec, _spec

SPECS: tuple[FlagSpec, ...] = (
    # search ranking
    _spec(
        "MEMO_FTS_BACKEND",
        "str",
        "auto",
        "search",
        "FTS backend: 'auto' (tantivy if installed, else fts5) | 'tantivy' | 'fts5'.",
    ),
    _spec(
        "MEMO_SEARCH_DECAY_ALPHA",
        "float",
        0.15,
        "search",
        "Recency-decay weight in hybrid ranking.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_SEARCH_DECAY_HALFLIFE",
        "int",
        0,
        "search",
        "Recency-decay half-life in days (0 = off). Consumer paths (recall/ask/chat) "
        "default to 90 when unset; raw search() stays decay-free.",
    ),
    _spec(
        "MEMO_RRF_K",
        "int",
        60,
        "search",
        "Reciprocal-rank-fusion k constant for hybrid search (Cormack default 60). "
        "Higher k softens rank dominance; lower k sharpens toward top hits.",
        min_val=1,
    ),
    _spec(
        "MEMO_RRF_ADAPTIVE",
        "bool",
        False,
        "search",
        "Adapt RRF k to result density: shrink k when ranked lists agree, grow it "
        "when they diverge. Off by default to keep the eval baseline comparable.",
    ),
    _spec(
        "MEMO_EXACT_TITLE_WEIGHT",
        "float",
        10.0,
        "search",
        "BM25 title field weight in `mode=exact` (default 10, vs 5 in normal bm25).",
        min_val=0.0,
    ),
    _spec(
        "MEMO_EXACT_TAGS_WEIGHT",
        "float",
        8.0,
        "search",
        "BM25 tags field weight in `mode=exact` (default 8, vs 3 in normal bm25).",
        min_val=0.0,
    ),
    _spec(
        "MEMO_RAG_CACHE_TTL_S",
        "int",
        300,
        "search",
        "TTL (seconds) for the session-scoped RAG context cache used by ask()/chat_ask "
        "when a session_id is supplied. Invalidated early on any corpus change.",
        min_val=0,
    ),
    _spec(
        "MEMO_FEEDBACK_HALFLIFE_DAYS",
        "float",
        180.0,
        "search",
        "Half-life (days) for temporal decay of positive feedback boosts (👍/click). "
        "0 disables decay. thumbs_down/ignore are never decayed.",
        min_val=0.0,
    ),
    _spec(
        "MEMO_DECAY_HALFLIFE_DECISION",
        "float",
        365.0,
        "search",
        "Per-type recency-decay half-life (days) for type='decision'. "
        "Decisions persist longer than the global default. "
        "Overrides MEMO_SEARCH_DECAY_HALFLIFE for this type when set. "
        "Ignored when MEMO_SEARCH_DECAY_HALFLIFE=0 (decay disabled globally).",
        min_val=0.0,
    ),
    _spec(
        "MEMO_DECAY_HALFLIFE_FEEDBACK",
        "float",
        90.0,
        "search",
        "Per-type recency-decay half-life (days) for type='feedback'. "
        "Overrides MEMO_SEARCH_DECAY_HALFLIFE for this type when set. "
        "Ignored when MEMO_SEARCH_DECAY_HALFLIFE=0 (decay disabled globally).",
        min_val=0.0,
    ),
    _spec(
        "MEMO_DECAY_HALFLIFE_NOTE",
        "float",
        30.0,
        "search",
        "Per-type recency-decay half-life (days) for type='note'. "
        "Notes are ephemeral and decay faster than the global default. "
        "Overrides MEMO_SEARCH_DECAY_HALFLIFE for this type when set. "
        "Ignored when MEMO_SEARCH_DECAY_HALFLIFE=0 (decay disabled globally).",
        min_val=0.0,
    ),
    _spec(
        "MEMO_DECAY_HALFLIFE_FACT",
        "float",
        180.0,
        "search",
        "Per-type recency-decay half-life (days) for type='fact'. "
        "Overrides MEMO_SEARCH_DECAY_HALFLIFE for this type when set. "
        "Ignored when MEMO_SEARCH_DECAY_HALFLIFE=0 (decay disabled globally).",
        min_val=0.0,
    ),
    _spec(
        "MEMO_DECAY_HALFLIFE_REFERENCE",
        "float",
        None,
        "search",
        "Per-type recency-decay half-life (days) for type='reference'. "
        "None (default) means references do not decay. "
        "Overrides MEMO_SEARCH_DECAY_HALFLIFE for this type when set. "
        "Ignored when MEMO_SEARCH_DECAY_HALFLIFE=0 (decay disabled globally).",
        min_val=0.0,
    ),
    _spec(
        "MEMO_HEALTH_SCORES_DISABLED",
        "bool",
        False,
        "search",
        "Disable health-score (confidence x roi_score) multiplier in search ranking. "
        "Defaults off — health scoring is neutral (1.0x) until Dream mode or contradiction scan have run.",
        opt_out=False,
    ),
    _spec(
        "MEMO_RERANK_ADAPTIVE_POOL",
        "bool",
        False,
        "search",
        "Dynamically size the rerank candidate pool based on vec-score standard deviation. "
        "High-variance results (stddev > 0.15) expand the pool to min(rerank_input_k * 1.5, 200); "
        "low-variance tight clusters (stddev < 0.05) shrink to max(limit + 5, 15). "
        "Default off keeps the fixed rerank_input_k pool for a stable eval baseline.",
    ),
    _spec(
        "MEMO_RERANK_SKIP_CONFIDENT_RRF",
        "bool",
        False,
        "search",
        "Skip cross-encoder rerank when the top hybrid RRF hit is already separated "
        "from the runner-up by MEMO_RERANK_SKIP_MIN_RATIO and MEMO_RERANK_SKIP_MIN_GAP. "
        "Experimental: keeps obvious searches fast while preserving rerank for ambiguous "
        "result packs. Off by default because RRF confidence can be corpus-dependent.",
    ),
    _spec(
        "MEMO_RERANK_SKIP_MIN_RATIO",
        "float",
        3.0,
        "search",
        "Minimum top/second score ratio required before confident-RRF rerank skip.",
        min_val=1.0,
    ),
    _spec(
        "MEMO_RERANK_SKIP_MIN_GAP",
        "float",
        0.05,
        "search",
        "Minimum absolute top-second score gap required before confident-RRF rerank skip.",
        min_val=0.0,
    ),
    _spec(
        "MEMO_GRAPH_EXPANSION_ENABLED",
        "bool",
        False,
        "search",
        "After primary search + rerank, follow knowledge-graph entity edges from the top-3 hits (1-hop) and append up to 3 adjacent memories scored at 0.6x the minimum primary score. Requires entities to have been extracted first (`memo extract-entities`).",
    ),
    _spec(
        "MEMO_QUERY_CACHE_SIZE",
        "int",
        256,
        "search",
        "LRU size for query embeddings (0 = off). Default 256 covers typical session query diversity with negligible RAM overhead (~few KB per cached vector).",
    ),
    _spec(
        "MEMO_CONTRADICT_PENALTY_ENABLED",
        "bool",
        True,
        "search",
        "Penalise the older side of open contradiction pairs among retrieved results. Requires `memo contradict scan` to have populated the sidecar DB.",
    ),
    _spec(
        "MEMO_CONTRADICT_PENALTY",
        "float",
        0.4,
        "search",
        "Score multiplier penalty applied to the older side of a contradiction pair.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_EVOLUTION_PENALTY",
        "float",
        0.7,
        "search",
        "Score multiplier for the older (superseded) side of an EVOLUTION pair "
        "— a later note that supersedes an earlier one. Softer than the "
        "contradiction penalty (the older fact may still hold context). Applied "
        "to pairs the temporal engine resolved as 'evolved' so known-stale "
        "memories stop ranking at full score.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_EVOLUTION_CONFIDENCE",
        "float",
        0.6,
        "search",
        "Absolute memory_health.confidence stamped on the older (superseded) "
        "side when an evolution pair is resolved. Below 1.0 so the health-score "
        "multiplier demotes it on every consumer path (default-on), not only "
        "where the contradiction-penalty pass runs. 1.0 disables this write.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_SEARCH_VEC_WEIGHT",
        "float",
        0.5,
        "search",
        "RRF fusion weight for the semantic (vec) leg in hybrid search (0.0-1.0). "
        "Default 0.5 gives equal weight to vec and BM25. Raise toward 1.0 for "
        "semantic-heavy queries; lower for precise keyword queries. "
        "If both MEMO_SEARCH_VEC_WEIGHT and MEMO_SEARCH_BM25_WEIGHT are set but "
        "do not sum to 1.0 (within 0.05), a warning is emitted.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_SEARCH_BM25_WEIGHT",
        "float",
        0.5,
        "search",
        "RRF fusion weight for the keyword (BM25) leg in hybrid search (0.0-1.0). "
        "Default 0.5 gives equal weight to vec and BM25. Raise toward 1.0 for "
        "precise technical/keyword queries; lower for semantic queries. "
        "If both MEMO_SEARCH_VEC_WEIGHT and MEMO_SEARCH_BM25_WEIGHT are set but "
        "do not sum to 1.0 (within 0.05), a warning is emitted.",
        min_val=0.0,
        max_val=1.0,
    ),
)
