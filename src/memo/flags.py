"""Central registry for `MEMO_*` feature/tuning environment flags.

One documented source of truth for every behavioral env var memo reads.
Storage/model config lives in `config.py` (typed `Config` model); this
module covers the ~60 *behavioral* flags that were historically read inline
via scattered `os.environ.get("MEMO_...")` calls with per-call-site defaults.

Each flag is a `FlagSpec` (kind + default + group + help). Use the typed
accessors — `flag_bool`, `flag_int`, `flag_float`, `flag_str` — or the
generic `flag(name)` which coerces by the registered kind. `validate()`
parses every *set* flag and reports misconfiguration; `active_flags()`
lists which are currently set in the environment. The `memo config`
command group surfaces both.

Rewiring every legacy call site to read through here is incremental; until
then this registry is authoritative for names/defaults/types and powers
`memo config validate`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

FlagKind = Literal["bool", "int", "float", "str"]

# Truthy spellings accepted for bool flags. Mirrors the historical mix of
# `== "1"` and `.lower() in (...)` checks across the codebase.
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class FlagSpec:
    """One `MEMO_*` flag: how to parse it, its default, and what it does."""

    name: str
    kind: FlagKind
    default: Any
    group: str
    help: str
    # Some bools are checked with inverted polarity (`!= "1"` → default-on,
    # opt-out). Recorded so `config flags` can show the real default.
    opt_out: bool = False
    # Optional inclusive bounds for numeric flags. Enforced in _coerce().
    min_val: float | None = None
    max_val: float | None = None


def _spec(
    name: str,
    kind: FlagKind,
    default: Any,
    group: str,
    help: str,
    opt_out: bool = False,
    min_val: float | None = None,
    max_val: float | None = None,
) -> FlagSpec:
    return FlagSpec(name, kind, default, group, help, opt_out, min_val, max_val)


# ── Registry ────────────────────────────────────────────────────────────────
# Grouped by subsystem. Defaults mirror the historical inline call sites.
_SPECS: tuple[FlagSpec, ...] = (
    # recall hook / daemon (UserPromptSubmit hot path — 5s budget)
    _spec("MEMO_RECALL_DISABLE", "bool", False, "recall", "Disable the recall hook entirely."),
    _spec(
        "MEMO_RECALL_DEBUG", "bool", False, "recall", "Verbose recall-hook diagnostics to stderr."
    ),
    _spec("MEMO_RECALL_MODE", "str", "vec", "recall", "Retrieval mode: vec | hybrid | bm25."),
    _spec(
        "MEMO_RECALL_FORCE_MODE",
        "bool",
        False,
        "recall",
        "Honor MEMO_RECALL_MODE even when the embedder isn't warm. Default off: a cold vec/hybrid request downgrades to bm25 to avoid blowing the recall-hook cold-load budget. Set to 1 to force the requested mode regardless of warm state.",
    ),
    _spec("MEMO_RECALL_TOP_K", "int", 3, "recall", "Number of memorias injected per prompt."),
    _spec(
        "MEMO_RECALL_MIN_SIM",
        "float",
        0.5,
        "recall",
        "Similarity floor for a hit, applied to the recency-decayed score (decay compresses raw cosine ~0.15, so 0.5 ≈ 0.65 raw; 0.6 over-filtered and caused bails). The bigger relevance lever is reference-tier exclusion.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec("MEMO_RECALL_BODY_CHARS", "int", 400, "recall", "Max body chars per injected memoria."),
    _spec(
        "MEMO_RECALL_MIN_BODY_CHARS",
        "int",
        40,
        "recall",
        "Skip memorias with bodies shorter than this.",
    ),
    _spec(
        "MEMO_RECALL_MIN_PROMPT_CHARS",
        "int",
        12,
        "recall",
        "Skip recall for prompts shorter than this.",
    ),
    _spec(
        "MEMO_RECALL_TOKEN_BUDGET",
        "int",
        0,
        "recall",
        "Token budget for injected context (0 = off).",
    ),
    _spec(
        "MEMO_RECALL_PROJECT_BOOST",
        "float",
        0.15,
        "recall",
        "Score boost for memorias tagged to the cwd project.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec("MEMO_RECALL_RERANK_INPUT_K", "int", 10, "recall", "Candidates fed to the reranker."),
    _spec(
        "MEMO_RECALL_STALENESS_DAYS",
        "int",
        0,
        "recall",
        "Down-rank memorias older than N days (0 = off).",
    ),
    _spec(
        "MEMO_RECALL_SKIP_SLASH",
        "bool",
        True,
        "recall",
        "Skip recall when the prompt starts with '/'.",
    ),
    _spec(
        "MEMO_RECALL_CONTEXTUAL",
        "bool",
        True,
        "recall",
        "Re-rank recall by learned type preferences + record what surfaces.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RECALL_EXCLUDE_REFERENCE",
        "bool",
        True,
        "recall",
        "Exclude the bulk `reference` tier (ingested vault) from auto-recall so durable knowledge isn't drowned.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RECALL_EXPAND_CONTEXT",
        "bool",
        True,
        "recall",
        "On a zero-hit recall, retry once with recent open-loop titles prepended so bare continuity prompts ('seguimos', 'qué queda pendiente') re-anchor instead of bailing. Fallback-only: queries that already recall are untouched.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RECALL_LOCK_TIMEOUT_MS",
        "int",
        2500,
        "recall",
        "Daemon: max ms a recall waits for the shared embedder/Memory lock before returning empty. Bounds the latency tail when a cold embed_batch holds the lock — recall bails fast instead of hanging tens of seconds and blowing the 5s hook budget.",
    ),
    _spec(
        "MEMO_RECALL_DAEMON_TIMEOUT_MS",
        "int",
        3500,
        "recall",
        "Client (recall-hook): ms to wait for the daemon socket before falling back to in-process subprocess search. Must sit under the hooks.json budget (12s) yet above the warm-but-slow daemon tail (~3-6s) so a slow daemon is waited on, not double-fired via subprocess.",
    ),
    _spec(
        "MEMO_EMBED_BATCH_CHUNK",
        "int",
        32,
        "recall",
        "Daemon: texts per embed chunk in embed_batch. The shared lock is released between chunks so a pending recall query-embed interleaves instead of waiting for the whole (cold) batch.",
    ),
    _spec(
        "MEMO_RECALL_GAP_THRESHOLD",
        "float",
        0.10,
        "recall",
        "If >0, reduce injected memorias to the top-1 when the score gap between rank-1 and rank-2 exceeds this value. Prevents a strong top hit from dragging in 2 weak tail hits. 0 = disabled.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RECALL_SKIP_BELOW",
        "float",
        0.45,
        "recall",
        "If >0, skip recall entirely when the best candidate's score is below this floor. Prevents low-confidence recall from injecting marginally relevant context. 0 = disabled.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RECALL_FEEDBACK_HINT",
        "bool",
        True,
        "recall",
        "Append a feedback hint comment to the recall block so the AI layer can surface memory_feedback_record to the user.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RECALL_ADAPTIVE_CONTEXT",
        "bool",
        True,
        "recall",
        "Re-weight recall results by detected prompt intent (code/decision/write → boost matching memory types). Zero extra search cost — pure score boost on returned hits.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RECALL_CLIENT",
        "str",
        "claude-code",
        "recall",
        "Front-end name stamped on recall consults for per-client telemetry.",
    ),
    _spec(
        "MEMO_GROUNDING_BUDGET_MS",
        "int",
        8000,
        "recall",
        "Time budget (ms) for grounding/citation work on the recall path.",
        min_val=0,
    ),
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
        "Recency-decay half-life in days (0 = off).",
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
        "MEMO_GRAPH_EXPANSION_ENABLED",
        "bool",
        False,
        "search",
        "After primary search + rerank, follow knowledge-graph entity edges from the top-3 hits (1-hop) and append up to 3 adjacent memorias scored at 0.6x the minimum primary score. Requires entities to have been extracted first (`memo extract-entities`).",
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
    # entity-aware retrieval + knowledge-graph expansion
    _spec(
        "MEMO_ENTITY_RETRIEVAL_ENABLED",
        "bool",
        False,
        "entity",
        "Extract entities from queries and boost results whose extra['entities'] overlap. Prerequisite for graph expansion.",
    ),
    _spec(
        "MEMO_ENTITY_GLINER",
        "bool",
        False,
        "entity",
        "Use the GLiNER zero-shot NER model for entity extraction instead of the LLM (higher recall, extra dependency).",
    ),
    _spec(
        "MEMO_ENTITY_GLINER_MODEL",
        "str",
        "urchade/gliner_medium-v2.1",
        "entity",
        "GLiNER model id used when MEMO_ENTITY_GLINER=1.",
    ),
    # session checkpoints / resume
    _spec(
        "MEMO_SESSION_DISABLE", "bool", False, "session", "Disable session checkpoint/recent hooks."
    ),
    _spec("MEMO_SESSION_DEBUG", "bool", False, "session", "Verbose session-hook diagnostics."),
    # turn capture
    _spec("MEMO_CAPTURE_DISABLE", "bool", False, "capture", "Disable Stop-hook turn capture."),
    _spec(
        "MEMO_CAPTURE_PATTERN_TYPES",
        "bool",
        True,
        "capture",
        "Zero-cost regex pre-pass in save(auto_derive=True): detect decision/preference/bug/fact type "
        "before calling the helper LLM. Set to 0 to always use the LLM for type inference.",
        opt_out=True,
    ),
    _spec("MEMO_CAPTURE_DEBUG", "bool", False, "capture", "Verbose capture diagnostics."),
    _spec(
        "MEMO_CAPTURE_MIN_WORDS", "int", 15, "capture", "Minimum words for a turn to be captured."
    ),
    _spec(
        "MEMO_CAPTURE_CONTEXT_TURNS",
        "int",
        3,
        "capture",
        "Prior turns included as capture context.",
    ),
    _spec(
        "MEMO_CAPTURE_COOLDOWN_MIN",
        "int",
        0,
        "capture",
        "Minutes between captures (0 = no cooldown).",
    ),
    # corpus maintenance (memo maintain)
    _spec(
        "MEMO_MAINTAIN_DISABLE",
        "bool",
        False,
        "maintain",
        "Disable the daily `memo maintain --if-due` auto-run.",
    ),
    _spec(
        "MEMO_MAINT_VIA_DAEMON",
        "bool",
        False,
        "maintain",
        "Route consolidation's synthesis LLM through the maintenance daemon (keeps the multi-GB model out of memo-mcp's resident set). Falls back in-process when the daemon is unreachable.",
    ),
    # emergent synthesis (memo synthesize)
    _spec(
        "MEMO_SYNTHESIS_ENABLED",
        "bool",
        True,
        "maintain",
        "Enable autonomous cross-memory synthesis pass in `memo maintain`. Set to 0 to disable.",
        opt_out=True,
    ),
    _spec(
        "MEMO_SYNTHESIS_MIN_CONFIDENCE",
        "str",
        "medium",
        "maintain",
        "Minimum LLM confidence to persist a synthesis: low | medium | high.",
    ),
    _spec(
        "MEMO_SYNTHESIS_MIN_CLUSTER",
        "int",
        5,
        "maintain",
        "Minimum cluster size for synthesis (memories per cluster). Conservative default of 5 avoids noisy low-data insights.",
        min_val=2,
        max_val=50,
    ),
    _spec(
        "MEMO_SYNTHESIS_MAX_CLUSTERS",
        "int",
        10,
        "maintain",
        "Max clusters processed per synthesis pass.",
        min_val=1,
        max_val=200,
    ),
    _spec(
        "MEMO_SYNTHESIS_THRESHOLD",
        "float",
        0.78,
        "maintain",
        "Cosine similarity threshold for synthesis clustering (looser than consolidation's 0.85).",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_CONSOLIDATE_AUTO_THRESHOLD",
        "float",
        0.95,
        "maintain",
        "Cosine floor for the LLM-free fast lane in consolidation. Clusters at this threshold or above are merged as keep_latest without calling the LLM.",
        min_val=0.0,
        max_val=1.0,
    ),
    # transcript ingest
    _spec(
        "MEMO_INGEST_MIN_CHARS",
        "int",
        200,
        "ingest",
        "Minimum chars for an ingested transcript turn.",
    ),
    _spec("MEMO_INGEST_STRICT", "bool", False, "ingest", "Strict ingest filtering."),
    _spec("MEMO_INGEST_DEBUG", "bool", False, "ingest", "Verbose ingest diagnostics."),
    _spec(
        "MEMO_INGEST_VIA_DAEMON",
        "bool",
        False,
        "ingest",
        "Route batch repo indexing through the ingest worker daemon (async, returns a job_id). Falls back to in-process when the daemon is unreachable.",
    ),
    # briefing (SessionStart panel)
    _spec(
        "MEMO_BRIEFING_DISABLE",
        "bool",
        False,
        "briefing",
        "Disable the SessionStart briefing panel.",
    ),
    _spec("MEMO_BRIEFING_DEBUG", "bool", False, "briefing", "Verbose briefing diagnostics."),
    _spec(
        "MEMO_BRIEFING_SYNAPSE_DISABLE",
        "bool",
        False,
        "briefing",
        "Skip the Synapse section of the briefing.",
        opt_out=True,
    ),
    _spec("MEMO_BRIEFING_LOOPS_N", "int", 5, "briefing", "Open-loop items shown in the briefing."),
    _spec(
        "MEMO_BRIEFING_LOOPS_DAYS", "int", 7, "briefing", "Look-back window (days) for open loops."
    ),
    # repo indexing
    _spec(
        "MEMO_REPO_MAX_FILE_BYTES", "int", None, "repo", "Skip repo files larger than this (bytes)."
    ),
    _spec(
        "MEMO_REPO_EMBED_BATCH", "int", None, "repo", "Chunks per embed batch during repo index."
    ),
    _spec("MEMO_REPO_FLUSH_BATCH", "int", None, "repo", "Rows per DB flush during repo index."),
    _spec(
        "MEMO_REPO_GIT_TIMEOUT_S", "int", None, "repo", "Timeout (s) for git operations on clone."
    ),
    # embedder daemon / client
    _spec(
        "MEMO_EMBEDDER_VIA_DAEMON",
        "bool",
        False,
        "embedder",
        "Route embeddings through the embed daemon.",
    ),
    _spec(
        "MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON",
        "bool",
        False,
        "embedder",
        "Fail instead of falling back to in-process embed.",
    ),
    _spec(
        "MEMO_EMBEDDER_CLIENT_TIMEOUT",
        "float",
        None,
        "embedder",
        "Embed-daemon client socket timeout (s).",
    ),
    _spec(
        "MEMO_EMBEDDER_STATS_INTERVAL_S",
        "float",
        None,
        "embedder",
        "Embed-daemon stats log interval (s).",
    ),
    # feedback boosting
    _spec(
        "MEMO_FEEDBACK_DISABLED",
        "bool",
        False,
        "feedback",
        "Disable relevance-feedback score boosting.",
        opt_out=True,
    ),
    _spec(
        "MEMO_FEEDBACK_SIM_THRESHOLD",
        "float",
        None,
        "feedback",
        "Similarity threshold for feedback matching.",
    ),
    _spec(
        "MEMO_FEEDBACK_BOOST_PER_VOTE", "float", None, "feedback", "Score boost added per upvote."
    ),
    _spec(
        "MEMO_FEEDBACK_BOOST_CAP", "float", None, "feedback", "Maximum cumulative feedback boost."
    ),
    # MCP transport
    _spec("MEMO_MCP_TRANSPORT", "str", "stdio", "mcp", "MCP transport: stdio | http."),
    _spec("MEMO_MCP_HOST", "str", "127.0.0.1", "mcp", "Bind host for the HTTP MCP transport."),
    _spec("MEMO_MCP_PORT", "int", 18768, "mcp", "Bind port for the HTTP MCP transport."),
    # synapse / memflow integration
    _spec(
        "MEMO_RESPECT_SYNAPSE_FREEZE",
        "bool",
        False,
        "synapse",
        "Honor a Synapse write-freeze signal.",
    ),
    _spec(
        "MEMO_SYNAPSE_EXECUTABLE", "str", "", "synapse", "Override path to the synapse executable."
    ),
    _spec(
        "MEMO_SYNAPSE_CLIENT_TIMEOUT",
        "float",
        None,
        "synapse",
        "Synapse client request timeout (s).",
    ),
    _spec("MEMO_MEMFLOW_BIN", "str", "", "synapse", "Override path to the memflow binary."),
    _spec("MEMO_EMIT_RECEIPTS", "bool", False, "synapse", "Emit operational receipts for Synapse."),
    _spec(
        "MEMO_EMIT_LEDGER",
        "bool",
        True,
        "synapse",
        "Emit consciousness-ledger entries.",
        opt_out=True,
    ),
    # cache tier (opt-in: memo as a bounded cache fronting an authoritative backing store)
    _spec(
        "MEMO_CACHE_MODE",
        "str",
        "off",
        "cache",
        "Cache tier mode: off | read_through | write_through | write_back. `off` (default) keeps memo a durable source-of-truth store with no eviction. Any other value treats the local vault as a derived cache in front of MEMO_CACHE_BACKEND.",
    ),
    _spec(
        "MEMO_CACHE_MAX_ENTRIES",
        "int",
        0,
        "cache",
        "Capacity bound for the local cache (0 = unbounded, i.e. durable behavior). When exceeded after a save, the eviction policy reclaims down to this count.",
    ),
    _spec(
        "MEMO_CACHE_EVICTION",
        "str",
        "lru",
        "cache",
        "Replacement policy when over capacity: lru (least-recently-accessed) | lfu (least-frequently-accessed) | ttl (oldest-access beyond MEMO_CACHE_TTL_DAYS first). Requires hit tracking.",
    ),
    _spec(
        "MEMO_CACHE_TTL_DAYS",
        "int",
        0,
        "cache",
        "Freshness window in days before a cached memoria is revalidated against the backing store / eligible for ttl eviction (0 = off).",
    ),
    _spec(
        "MEMO_CACHE_BACKEND",
        "str",
        "memflow",
        "cache",
        "Authoritative backing store the cache fronts: memflow (flow_* shared consciousness) | vault (remote vault path) | none. Only consulted when MEMO_CACHE_MODE != off.",
    ),
    # misc behavior
    _spec(
        "MEMO_ENCRYPTION_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the at-rest encryption vertical (EXPERIMENTAL). When off (default) the `memo encrypt` CLI group and `memory_encrypt_*` MCP tools refuse with a disabled message instead of touching the key manager; the EncryptionManager is still constructed so the facade/store wiring is unchanged.",
    ),
    _spec(
        "MEMO_OCR_ENABLED", "bool", True, "misc", "Enable OCR for image ingestion.", opt_out=True
    ),
    _spec("MEMO_PROMPT_CACHE", "bool", False, "misc", "Enable LLM prompt caching."),
    _spec(
        "MEMO_CONTEXTUAL_RETRIEVAL",
        "bool",
        False,
        "misc",
        "Enable contextual-retrieval re-ranking.",
    ),
    _spec(
        "MEMO_SAVE_DEDUP_CHECK",
        "bool",
        True,
        "misc",
        "Before saving, run a quick vec search for near-duplicates. In interactive mode, prompts to update an existing memoria instead. In non-interactive mode, logs a warning.",
        opt_out=True,
    ),
    _spec(
        "MEMO_SAVE_DEDUP_THRESHOLD",
        "float",
        0.88,
        "misc",
        "Cosine similarity floor for near-duplicate detection on save (requires MEMO_SAVE_DEDUP_CHECK=1).",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_AUTO_PROJECT_TAG",
        "bool",
        True,
        "misc",
        "Auto-tag saved memorias with the cwd project.",
        opt_out=True,
    ),
    _spec("MEMO_PROJECT_TAG", "str", "", "misc", "Pin a project tag (overrides cwd detection)."),
    _spec("MEMO_MODEL_PROFILE", "str", "", "misc", "Model profile: light | balanced | quality."),
    _spec("MEMO_NONINTERACTIVE", "bool", False, "misc", "Suppress interactive prompts (hooks/CI)."),
    _spec(
        "MEMO_SUPPRESS_LEGACY_WARN",
        "bool",
        False,
        "misc",
        "Silence legacy-config deprecation warnings.",
    ),
    _spec(
        "MEMO_ASK_FALLBACK_MSG",
        "str",
        "no encuentro la respuesta en las memorias guardadas",
        "misc",
        "Message returned by memory_ask when no relevant sources are found.",
    ),
    _spec(
        "MEMO_VAULT_SYSTEM_DIR",
        "str",
        "Obsidian",
        "misc",
        "Vault subdir holding memo's system tree (AI/, Contacts/, Whatsapp/).",
    ),
    # ROI accounting (memo roi)
    _spec(
        "MEMO_ROI_SECS_PER_GROUNDED",
        "int",
        30,
        "roi",
        "Estimated seconds saved per grounded answer.",
        min_val=0,
    ),
    _spec(
        "MEMO_ROI_SECS_PER_REASK",
        "int",
        120,
        "roi",
        "Estimated seconds cost per re-ask.",
        min_val=0,
    ),
    # WhatsApp ingest
    _spec(
        "MEMO_WHATSAPP_DB",
        "str",
        "",
        "whatsapp",
        "Path to the whatsapp-mcp bridge SQLite DB to ingest.",
    ),
    _spec(
        "MEMO_WHATSAPP_NOTES_DIR",
        "str",
        "",
        "whatsapp",
        "Override output dir for ingested WhatsApp notes (default <SYSTEM_DIR>/Whatsapp).",
    ),
)

REGISTRY: dict[str, FlagSpec] = {s.name: s for s in _SPECS}


def _coerce(spec: FlagSpec, raw: str) -> Any:
    """Parse `raw` per the spec's kind. Raises ValueError on bad input."""
    if spec.kind == "bool":
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"expected a boolean (1/0/true/false), got {raw!r}")
    if spec.kind == "int":
        vi = int(raw.strip())
        if spec.min_val is not None and vi < spec.min_val:
            raise ValueError(f"{spec.name} must be >= {spec.min_val}, got {vi}")
        if spec.max_val is not None and vi > spec.max_val:
            raise ValueError(f"{spec.name} must be <= {spec.max_val}, got {vi}")
        return vi
    if spec.kind == "float":
        vf = float(raw.strip())
        if spec.min_val is not None and vf < spec.min_val:
            raise ValueError(f"{spec.name} must be >= {spec.min_val}, got {vf}")
        if spec.max_val is not None and vf > spec.max_val:
            raise ValueError(f"{spec.name} must be <= {spec.max_val}, got {vf}")
        return vf
    return raw  # str


def flag(name: str, *, env: dict[str, str] | None = None) -> Any:
    """Return the typed, parsed value for `name`, or its default if unset.

    Unknown flags raise KeyError — every flag must be registered above.
    """
    spec = REGISTRY[name]
    src = os.environ if env is None else env
    raw = src.get(name)
    if raw is None or raw == "":
        # empty string counts as unset except for str flags whose default is ""
        if raw == "" and spec.kind == "str":
            return ""
        return spec.default
    try:
        return _coerce(spec, raw)
    except ValueError:
        return spec.default


def flag_bool(name: str, *, env: dict[str, str] | None = None) -> bool:
    return bool(flag(name, env=env))


def flag_int(name: str, *, env: dict[str, str] | None = None) -> int | None:
    v = flag(name, env=env)
    return None if v is None else int(v)


def flag_float(name: str, *, env: dict[str, str] | None = None) -> float | None:
    v = flag(name, env=env)
    return None if v is None else float(v)


def flag_str(name: str, *, env: dict[str, str] | None = None) -> str:
    v = flag(name, env=env)
    return "" if v is None else str(v)


def active_flags(env: dict[str, str] | None = None) -> dict[str, str]:
    """Registered flags currently set (non-empty) in the environment."""
    src = os.environ if env is None else env
    return {n: src[n] for n in REGISTRY if src.get(n)}


def unknown_memo_vars(env: dict[str, str] | None = None) -> list[str]:
    """`MEMO_*` env vars set but NOT in the registry (possible typos).

    Excludes storage/model vars owned by config.py.
    """
    src = os.environ if env is None else env
    owned = {
        "MEMO_DATA_DIR",
        "MEMO_STATE_DIR",
        "MEMO_VAULT_PATH",
        "MEMO_MEMORY_SUBDIR",
        "MEMO_EMBEDDER_MODEL",
        "MEMO_EMBEDDER_DIMS",
        "MEMO_LLM_MODEL",
        "MEMO_HELPER_MODEL",
        "MEMO_RERANKER_MODEL",
        "MEMO_RERANKER_ENABLED",
        "MEMO_RERANKER_REVISION",
        "MEMO_RERANK_FUSION_ALPHA",
        "MEMO_RERANK_INPUT_K",
        "MEMO_MAX_CONTENT_CHARS",
        "MEMO_SEARCH_DEFAULT_LIMIT",
        "MEMO_CONFIG_FILE",
    }
    return sorted(k for k in src if k.startswith("MEMO_") and k not in REGISTRY and k not in owned)


def validate(env: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Parse every set flag; return a list of problems (empty = all good).

    Each problem is {flag, value, error}.
    """
    src = os.environ if env is None else env
    problems: list[dict[str, str]] = []
    for name, spec in REGISTRY.items():
        raw = src.get(name)
        if raw is None or raw == "":
            continue
        try:
            _coerce(spec, raw)
        except ValueError as exc:
            problems.append({"flag": name, "value": raw, "error": str(exc)})
    for var in unknown_memo_vars(env):
        problems.append(
            {"flag": var, "value": src[var], "error": "unknown MEMO_* var (typo? not in registry)"}
        )
    return problems
