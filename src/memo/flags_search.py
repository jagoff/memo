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
        "MEMO_QUALITY_RERANK",
        "bool",
        False,
        "search",
        "Enable quality-aware post-retrieval reranking for explicit search/ask paths. "
        "Demotes invalidated/superseded/contradicted hits and boosts verified/supported hits. "
        "Default off to preserve ranking baselines.",
    ),
    _spec(
        "MEMO_CONTEXT_PACK",
        "bool",
        False,
        "search",
        "Enable context-pack construction for memo ask and explicit context-pack tools. "
        "Default off; ambient recall does not use context packs.",
    ),
    _spec(
        "MEMO_CONTEXT_SURFACE",
        "bool",
        True,
        "search",
        "Enable the prompt-ready memo context/profile surface for agents.",
    ),
    _spec(
        "MEMO_CONTEXT_CACHE",
        "bool",
        True,
        "search",
        "Enable a short process-local cache for read-only context surface calls.",
    ),
    _spec(
        "MEMO_CONTEXT_CACHE_TTL",
        "int",
        60,
        "search",
        "TTL in seconds for the process-local context surface cache.",
        min_val=0,
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
        False,
        "search",
        "Penalise the older side of open contradiction pairs among retrieved results. "
        "Requires `memo contradict scan` to have populated the sidecar DB. Default OFF "
        "(2026-07-11): measured net-negative on the vec live path — it is effectively "
        "symmetric (demotes both members of a pair), costing precision@k (+0.006/+0.012 "
        "when off) with no measurable retrieval benefit. It suppresses disputed content "
        "rather than improving retrieval; opt in only if wholesale demotion of disputed "
        "pairs is desired (the dossier/disputes render flags already flag them instead).",
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
    _spec(
        "MEMO_REFERENCE_SEARCH_FLOOR",
        "float",
        0.0,
        "search",
        "Noise floor for the bulk `reference` tier in EXPLICIT retrieval "
        "(search/ask/chat — the recall hook already SQL-excludes reference via "
        "MEMO_RECALL_EXCLUDE_REFERENCE). When > 0, a reference-tier hit must "
        "have a final score >= this floor to stay in results; durable-tier "
        "hits are never affected. Compared against the final mode-dependent "
        "score (cosine in vec mode, RRF-fused in hybrid, BM25 in bm25). "
        "Skipped when the caller explicitly filters type='reference' so "
        "on-demand reference search stays intact. 0.0 (default) = off.",
        min_val=0.0,
    ),
    _spec(
        "MEMO_HYDE_ENABLED",
        "bool",
        False,
        "search",
        "Enable HyDE (Hypothetical Document Embeddings): generate a hypothetical "
        "answer doc from the query, embed it instead of the raw query. "
        "Improves recall on complex/ambiguous queries at +1 LLM call per search.",
    ),
    _spec(
        "MEMO_HYDE_MAX_TOKENS",
        "int",
        256,
        "search",
        "Max tokens for HyDE generation (LLM call to generate hypothetical doc).",
    ),
    _spec(
        "MEMO_SEARCH_CHUNK_PARENT",
        "bool",
        False,
        "search",
        "Map winning reference-tier CHUNK hits (extra.parent_id from "
        "MEMO_CHUNK_INGEST) back to their parent memory in explicit search: "
        "the parent surfaces once at the best chunk's rank/score, deduped "
        "against parents already in the result list. Skipped when the caller "
        "filters type='reference'. Recall hook unaffected (reference tier is "
        "SQL-excluded there). Off by default — eval-gated before any flip.",
    ),
    _spec(
        "MEMO_ASK_MULTI_ROUND",
        "bool",
        False,
        "search",
        "ask/chat only: after round-1 retrieval, ONE LLM sufficiency check may "
        "emit 1-3 refined queries for a single capped round-2 (at most k added "
        "hits, id-deduped). Never runs in the 5s recall hook. Default off — "
        "flip only after the eval gate + synapse eval-chat measure a win.",
    ),
    _spec(
        "MEMO_SEARCH_ADAPTIVE_LEXICAL_WEIGHT",
        "bool",
        False,
        "search",
        "Hybrid fusion: queries of <=2 tokens (identifier/tag lookups) tilt the "
        "RRF weights lexical (vec 0.35 / bm25 0.65). Explicitly-set "
        "MEMO_SEARCH_VEC_WEIGHT/BM25_WEIGHT always win. Hybrid mode only — the "
        "recall hook's default vec mode is untouched. Default off; eval-gated.",
    ),
    _spec(
        "MEMO_FACT_RETRIEVAL_ENABLED",
        "bool",
        True,
        "search",
        "Hybrid search: add a lightweight temporal fact-edge leg over "
        "subject/predicate/object text and fuse source memories through RRF.",
    ),
    _spec(
        "MEMO_FACT_RETRIEVAL_WEIGHT",
        "float",
        0.6,
        "search",
        "RRF fusion weight for temporal fact-edge candidates in hybrid search.",
        min_val=0.0,
    ),
    _spec(
        "MEMO_FACT_SURFACE_ENABLED",
        "bool",
        True,
        "search",
        "Attach query-related temporal fact edges to MemoryRecord.extra for "
        "search/ask/MCP surfaces.",
    ),
    _spec(
        "MEMO_DECLARE_DISPUTES",
        "bool",
        False,
        "search",
        "When both sides of a competing/open contradiction pair surface in the same "
        "result set, keep BOTH at full score (the dispute is declared via the hit "
        "dossier's ⚔ marker) instead of silently demoting the older side by "
        "MEMO_CONTRADICT_PENALTY. Default off = legacy silent demote.",
    ),
    _spec(
        "MEMO_VEC_QUANTIZE",
        "str",
        "int8",
        "search",
        "vec0 storage precision for the main `vec` table. 'off' = float32[dims] "
        "(4 B/dim); 'int8' = int8[dims] (1 B/dim, ~4x smaller on disk and in the "
        "base64 sync shard) via vec_quantize_int8(...,'unit'). SAFE only for "
        "L2-normalised vectors (norm-guard store/queries.py). This is a vec0 column "
        "TYPE change: it takes effect ONLY on `memo reindex --rebuild`, never at "
        "runtime. Default int8 applies to FRESH indexes only — an existing index "
        "keeps its on-disk precision (self-describing adoption in "
        "store/schema.py:_validate_vec_quant) until `memo reindex --rebuild`. "
        "Rebuild-required flag — MANUAL graduation only; never register in "
        "dream_flags.GATES. Eval-gate before flipping.",
        choices=("off", "int8"),
    ),
)
