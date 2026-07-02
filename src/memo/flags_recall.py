from __future__ import annotations

from memo.flags_base import FlagSpec, _spec

SPECS: tuple[FlagSpec, ...] = (
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
    _spec("MEMO_RECALL_TOP_K", "int", 3, "recall", "Number of memories injected per prompt."),
    _spec(
        "MEMO_RECALL_MIN_SIM",
        "float",
        0.5,
        "recall",
        "Similarity floor for a hit, applied to the recency-decayed score (decay compresses raw cosine ~0.15, so 0.5 ≈ 0.65 raw; 0.6 over-filtered and caused bails). The bigger relevance lever is reference-tier exclusion.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec("MEMO_RECALL_BODY_CHARS", "int", 400, "recall", "Max body chars per injected memory."),
    _spec(
        "MEMO_RECALL_MIN_BODY_CHARS",
        "int",
        40,
        "recall",
        "Skip memories with bodies shorter than this.",
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
        600,
        "recall",
        "Token budget for injected context (0 = off).",
    ),
    _spec(
        "MEMO_RECALL_ASSOCIATIVE",
        "bool",
        True,
        "recall",
        "Append a labeled 'related via graph' nudge to recall, found by "
        "walking the entity+codegraph graph from the top-K seeds. Default on.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RECALL_SYSTEM_MESSAGE",
        "bool",
        True,
        "recall",
        "Emit a human-visible one-liner (top-level systemMessage in the hook "
        "JSON) listing what was recalled — 🧠 memo · N: titles. Claude Code "
        "shows it to the user; presence, not context. Default on.",
        opt_out=True,
    ),
    _spec("MEMO_ASSOCIATIVE_HOPS", "int", 2, "recall",
          "Graph hops to expand from recall seeds.", min_val=1, max_val=3),
    _spec("MEMO_ASSOCIATIVE_LIMIT", "int", 2, "recall",
          "Max associative memories in the recall nudge.", min_val=1),
    _spec("MEMO_ASSOCIATIVE_MIN_ACTIVATION", "float", 0.5, "recall",
          "Activation floor below which an associative hit is dropped.", min_val=0.0),
    _spec("MEMO_ASSOCIATIVE_BUDGET_MS", "int", 300, "recall",
          "Time guard for associative expansion; skip the nudge if exceeded.", min_val=0),
    _spec(
        "MEMO_RECALL_PROJECT_BOOST",
        "float",
        0.25,
        "recall",
        "Score boost for memories tagged to the cwd project. Tier-1 of the "
        "3-tier soft project ranking (current > global > other-projects).",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RECALL_GLOBAL_BOOST",
        "float",
        0.10,
        "recall",
        "Score boost for global / cross-cutting memories (no project: tag, or "
        "type preference/feedback) so they stay afloat in any project. Tier-2 of "
        "the 3-tier soft project ranking (current > global > other-projects).",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RECALL_GRAPH_PROXIMITY",
        "bool",
        False,
        "recall",
        "Boost recall candidates whose entities sit one hop from the query's "
        "entities in the materialized entity graph (Phase 2 graph-proximity "
        "rerank). Default OFF: when off the ranking is identical to today. "
        "Requires MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT > 0 to have any effect.",
    ),
    _spec(
        "MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT",
        "float",
        0.0,
        "recall",
        "Per-edge weight for the graph-proximity boost: a hit's score gains "
        "weight * (sum of entity-edge weights connecting it to a query entity). "
        "Default 0.0 (no-op even when MEMO_RECALL_GRAPH_PROXIMITY is on); the "
        "nightly tuner line-searches this knob.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec("MEMO_RECALL_RERANK_INPUT_K", "int", 10, "recall", "Candidates fed to the reranker."),
    _spec(
        "MEMO_RECALL_STALENESS_DAYS",
        "int",
        0,
        "recall",
        "Down-rank memories older than N days (0 = off).",
    ),
    _spec(
        "MEMO_RECALL_SKIP_SLASH",
        "bool",
        True,
        "recall",
        "Skip recall when the prompt starts with '/'. A slash command WITH substantive args still recalls on the arg text (see MEMO_RECALL_SLASH_MIN_ARG_CHARS); set to 0 to recall on every slash prompt unmodified.",
    ),
    _spec(
        "MEMO_RECALL_SLASH_MIN_ARG_CHARS",
        "int",
        8,
        "recall",
        "When MEMO_RECALL_SKIP_SLASH is on, recall still fires on a slash command whose argument text (after stripping the leading /command token) is at least this many chars; shorter args bail as before. Recall searches the arg text only — the /command token is dropped so it doesn't pollute the embedding.",
    ),
    _spec(
        "MEMO_RECALL_SLASH_DENYLIST",
        "str",
        "clear,compact,exit,quit,help,config,login,logout,model,resume,doctor,cost,status",
        "recall",
        "Comma-separated slash command names whose recall is always skipped even with args (pure-UI/noise verbs, and commands whose args may be secrets/paths — args reach recall.log).",
    ),
    _spec(
        "MEMO_RECALL_SHORT_EXPAND_TURNS",
        "int",
        2,
        "recall",
        "When a prompt is below MEMO_RECALL_MIN_PROMPT_CHARS and a session is active, prepend this many recent session prompts (from the prompt_trail) to re-anchor the short follow-up before bailing. 0 disables short-prompt expansion (bail as before). Gated on MEMO_RECALL_EXPAND_CONTEXT.",
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
        "On a zero-hit recall, retry once with recent open-loop titles prepended so bare continuity prompts ('continue', 'what is left') re-anchor instead of bailing. Fallback-only: queries that already recall are untouched.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RECALL_PRIORITY_ENABLED",
        "bool",
        True,
        "recall",
        "Daemon: enable priority lock. Recall requests jump the queue ahead of background batch embeds.",
        opt_out=True,
    ),
    _spec(
        "MEMO_MICRO_EMBEDDER_MODEL",
        "str",
        "",
        "recall",
        # Experimental opt-in: when set, the recall daemon uses this lighter model to
        # re-rank BM25 candidates during cold-start (main MLX embedder not yet warm).
        # A dim mismatch between this model's output and cfg.embedder_dims gracefully
        # falls back to the normal embedder path instead of returning empty recall.
        "Experimental cold-start micro-embedder. When set and the main embedder is cold, "
        "the daemon uses this lighter model to re-rank BM25 candidates. "
        "Dim mismatch with the main model falls back gracefully to the main embedder.",
    ),
    _spec(
        "MEMO_RECALL_LOCK_TIMEOUT_MS",
        "int",
        2500,
        "recall",
        "Daemon: max ms a recall waits for the shared embedder/Memory lock before returning empty. Bounds the latency tail when a cold embed_batch holds the lock — recall bails fast instead of hanging tens of seconds and blowing the 5s hook budget.",
    ),
    _spec(
        "MEMO_EMBED_LOCK_TIMEOUT_MS",
        "int",
        60000,
        "recall",
        "Daemon: max ms an embed_query waits for the shared embedder lock before falling back in-process. Unlike recall (5s hook budget), embed_query callers (save dedup, dream passes) are not latency-bound, so the default matches the 60s embed_batch hold — an embed_query never bails before the in-flight batch chunk releases, avoiding a redundant cold MLX load when a heavy job (e.g. `memo dream run`) self-contends on the daemon.",
        min_val=100,
    ),
    _spec(
        "MEMO_RECALL_DAEMON_TIMEOUT_MS",
        "int",
        2000,
        "recall",
        "Client (recall-hook): ms to wait for the daemon socket before falling back to in-process subprocess search. Default 2000 ms: worst-case daemon response is ~1-2s; 2s timeout + ~1-2s subprocess fallback stays within the 5s hook budget.",
        min_val=100,
    ),
    _spec(
        "MEMO_RECALL_DAEMON_TIMEOUT",
        "float",
        2.0,
        "recall",
        "Client (recall-hook): seconds to wait for the daemon socket (float alias for MEMO_RECALL_DAEMON_TIMEOUT_MS). When both are set, this takes precedence.",
        min_val=0.1,
        max_val=4.9,
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
        "If >0, reduce injected memories to the top-1 when the score gap between rank-1 and rank-2 exceeds this value. Prevents a strong top hit from dragging in 2 weak tail hits. 0 = disabled.",
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
        False,
        "recall",
        "Append a feedback hint comment to the recall block so the AI layer can surface memo_feedback_record to the user. Off by default (saves ~20 tokens/recall).",
    ),
    _spec(
        "MEMO_RECALL_FOOTER",
        "str",
        "full",
        "recall",
        "Footer style: 'full' (default, includes get command), 'short' (minimal), or 'none' (no footer). Short saves ~15 tokens.",
    ),
    _spec(
        "MEMO_RECALL_FORMAT",
        "str",
        "auto",
        "recall",
        "Recall format: 'auto' (default, picks based on budget/hits), 'compact' (one-line per hit), 'balanced' (title + bullets), 'full' (prose). Compact saves ~65%, balanced ~40%.",
    ),
    _spec(
        "MEMO_RECALL_DIRECTIVE_ONCE",
        "bool",
        True,
        "recall",
        "Inject the RECALL_DIRECTIVE only on the first recall turn of a session (turn=1). Subsequent turns skip it since it's already in the context window. Saves ~110 tokens per turn. Disable to always include.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RECALL_SCORE_ADAPTIVE_BODY",
        "bool",
        True,
        "recall",
        "Scale per-hit body_chars proportionally to hit score: score>=0.85 -> 1.5x, score<0.65 -> 0.5x (min 80 chars). High-confidence hits get more context; marginal hits waste fewer tokens.",
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
    _spec(
        "MEMO_RECALL_ADAPTIVE_BUDGET",
        "bool",
        True,
        "recall",
        "Scale token_budget by prompt length: longer prompts get smaller budgets (leave room). Shorter prompts get more context. Default on.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RECALL_SUMMARIZE_BODY",
        "bool",
        False,
        "recall",
        "Use LLM to summarize hit bodies that exceed body_chars before rendering. Reduces tokens at cost of extra latency. Off by default (adds ~200ms per hit).",
    ),
    _spec(
        "MEMO_RECALL_TRIVIAL_BAIL",
        "bool",
        True,
        "recall",
        "Skip recall when the prompt is ≤3 words and any word matches the built-in trivial set "
        "(yes, no, ok, sure, sí, dale, gracias, …). Saves the embed+search round-trip for "
        "pure confirmation turns. Opt-out with MEMO_RECALL_TRIVIAL_BAIL=0.",
        opt_out=True,
    ),
)
