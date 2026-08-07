from __future__ import annotations

from memo.flags_base import FlagSpec, _spec

SPECS: tuple[FlagSpec, ...] = (
    # recall hook / daemon (UserPromptSubmit hot path — 5s budget)
    _spec("MEMO_RECALL_DISABLE", "bool", False, "recall", "Disable the recall hook entirely."),
    _spec(
        "MEMO_RECALL_DEBUG", "bool", False, "recall", "Verbose recall-hook diagnostics to stderr."
    ),
    _spec(
        "MEMO_RECALL_METRICS",
        "bool",
        True,
        "recall",
        "Stamp one JSON line per recall-hook run ({ts, total_ms, path daemon|subprocess, hits}) "
        "to state_dir/recall_metrics.jsonl — latency percentile tracking surfaced by "
        "`memo stats`. Cheap append, auto-rotated at ~5000 lines. Opt-out with "
        "MEMO_RECALL_METRICS=0.",
        opt_out=True,
    ),
    _spec("MEMO_RECALL_MODE", "str", "vec", "recall", "Retrieval mode: vec | hybrid | bm25."),
    _spec(
        "MEMO_RECALL_FORCE_MODE",
        "bool",
        False,
        "recall",
        "Honor MEMO_RECALL_MODE even when the embedder isn't warm. Default off: a cold vec/hybrid request downgrades to bm25 to avoid blowing the recall-hook cold-load budget. Set to 1 to force the requested mode regardless of warm state.",
    ),
    _spec(
        "MEMO_RECALL_TOP_K",
        "int",
        3,
        "recall",
        "Number of memories injected per prompt.",
        min_val=1,
    ),
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
        "MEMO_RECALL_SESSION_MODE",
        "str",
        "",
        "recall",
        "Optional native per-session ranking preset.",
    ),
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
    _spec(
        "MEMO_RECALL_CITE_INSTRUCTION",
        "bool",
        True,
        "recall",
        "Append an instruction to the injected recall block asking the model "
        "to cite used memories inline by short id ([a1b2c3d4]) — visible "
        "attribution; cited ids also feed grounding. Default on.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RECALL_EMPTY_MARKER",
        "bool",
        True,
        "recall",
        "When a recall search actually ran in a session and nothing qualified, "
        "inject a one-line <memo-recall> marker stating memo has no record — "
        "so the agent can distinguish 'no recorded memory of X' from 'X is "
        "false'. Bails (empty stdin, short/trivial prompts, errors, session "
        "dedup) and sessionless invocations still emit nothing. ~15 tokens. "
        "Default on; opt out with MEMO_RECALL_EMPTY_MARKER=0.",
        opt_out=True,
    ),
    _spec(
        "MEMO_ASSOCIATIVE_HOPS",
        "int",
        2,
        "recall",
        "Graph hops to expand from recall seeds.",
        min_val=1,
        max_val=3,
    ),
    _spec(
        "MEMO_ASSOCIATIVE_LIMIT",
        "int",
        2,
        "recall",
        "Max associative memories in the recall nudge.",
        min_val=1,
    ),
    _spec(
        "MEMO_ASSOCIATIVE_MIN_ACTIVATION",
        "float",
        0.5,
        "recall",
        "Activation floor below which an associative hit is dropped.",
        min_val=0.0,
    ),
    _spec(
        "MEMO_ASSOCIATIVE_BUDGET_MS",
        "int",
        300,
        "recall",
        "Time guard for associative expansion; skip the nudge if exceeded.",
        min_val=0,
    ),
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
        "MEMO_RECALL_MMR_LAMBDA",
        "float",
        0.0,
        "recall",
        "Maximal-marginal-relevance diversity re-ordering of the gated recall "
        "pool: score' = lambda*relevance - (1-lambda)*max_sim_to_already_selected, "
        "similarity = token-set Jaccard over title+body (no extra embeds/IO, "
        "O(K^2) over the candidate pool only). Default 0.0 = OFF (ranking "
        "identical to today); 1.0 = pure relevance, no diversity penalty.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RECALL_SYNTHESIS_BOOST",
        "float",
        0.0,
        "recall",
        "Additive score boost for type=synthesis hits (auto-generated "
        "cross-cluster insights) so distilled knowledge surfaces above its raw "
        "sources. Composes like the project/global tier boosts. Default 0.0 = "
        "OFF (ranking identical to today).",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RECALL_ALTITUDE",
        "float",
        0.0,
        "recall",
        "Recall altitude: additive score boost for distilled hits "
        "(type=synthesis + extra.synthesis_kind=distillation) on a BROAD query, so "
        "a high-altitude summary surfaces first; suppressed on a SPECIFIC query so "
        "the raw source evidence surfaces instead. Composes with "
        "MEMO_RECALL_SYNTHESIS_BOOST. Pure render-time list boost keyed on data "
        "already on the hit (no store read, no MLX on the hook path). Default 0.0 = "
        "OFF. NOTE: like every rank_hits knob it is live only on the warm-daemon "
        "path, inert on the cli_recall_hook subprocess fallback.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RECALL_CODE_PROXIMITY_BOOST",
        "float",
        0.0,
        "recall",
        "Additive score boost for hits whose code_refs fall in the codegraph "
        "neighborhood (2 hops) of the uncommitted working-tree changes "
        "(git diff --name-only HEAD, once per render). Composes like the other "
        "rank_hits boosts. Default 0.0 = OFF: zero subprocesses and zero graph "
        "queries — ranking identical to today.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RECALL_RERANK_INPUT_K",
        "int",
        10,
        "recall",
        "Candidates fed to the reranker. Bounded 1-200 to match the pydantic "
        "`rerank_input_k` field: the recall hook injects this via cfg.model_copy "
        "(which skips validation), so an unbounded value would size a cold-start "
        "cross-encoder pool past the 5s hook budget.",
        min_val=1,
        max_val=200,
    ),
    _spec(
        "MEMO_RECALL_SKIP_SLASH",
        "bool",
        True,
        "recall",
        "Skip recall when the prompt starts with '/'. A slash command WITH substantive args still recalls on the arg text (see MEMO_RECALL_SLASH_MIN_ARG_CHARS); set to 0 to recall on every slash prompt unmodified.",
    ),
    _spec(
        "MEMO_RECALL_HOOK_BUDGET_MS",
        "int",
        10000,
        "recall",
        "Wall-clock cap on the recall-hook's in-process fallback (SIGALRM). Past it the hook yields an empty recall and logs 'hook budget exceeded' instead of blocking the prompt; measured worst case before the cap was 126s against a ~5s budget. Delivered between bytecodes, so it bounds waits that pass through the interpreter but cannot preempt a single blocking C call such as a held sqlite lock (that one is busy_timeout's job). 0 disables the cap.",
        min_val=0,
    ),
    _spec(
        "MEMO_RECALL_SKIP_MACHINE_PROMPTS",
        "bool",
        True,
        "recall",
        "Skip recall when the prompt is a harness envelope rather than a human turn (<task-notification>, <system-reminder>, command output, interrupted-request markers). Measured at 40% of hook fires on a live install, each paying a full embed + search for a machine-to-machine message. Matched on the opening marker only, so prose mentioning one still recalls; set to 0 to recall on every prompt.",
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
        "MEMO_RECALL_EXCLUDE_UNCERTAIN",
        "bool",
        True,
        "recall",
        "Exclude memories tagged '_uncertain' (low-confidence auto-captures, see "
        "MEMO_CAPTURE_MIN_CONFIDENCE) from the auto-recall hook — mirroring the "
        "reference-tier exclusion. They stay searchable on demand via "
        "memo_search/memo_ask. No-op on corpora where the capture confidence "
        "gate never tagged anything.",
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
        "MEMO_RECALL_FOOTER_AFTER",
        "str",
        "short",
        "recall",
        "Footer style used past turn 1 when MEMO_RECALL_FOOTER is unset: full|short|none.",
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
    _spec(
        "MEMO_RECALL_EPISTEMIC_LABELS",
        "bool",
        False,
        "recall",
        "Prefix each injected recall hit with its epistemic status derived from "
        "existing metadata: '⟨type · YYYY-MM⟩', '⟨~inferred · YYYY-MM⟩' for "
        "synthesis, '⟨?unverified⟩' for _uncertain captures. Pure render-layer "
        "(~5-10 tok/hit, inside the token budget); ranking untouched.",
    ),
    _spec(
        "MEMO_RECALL_OMISSIONS_TAIL",
        "bool",
        False,
        "recall",
        "When the token budget or gap-trim filter drops hits that qualified, "
        "append one budget-checked line '+N more relevant — /memo get <id>' so "
        "agents don't treat omitted as absent. Presentation-only; cannot drop "
        "or reorder hits — but it IS a new output line, so it ships opt-in "
        "(default off), same class as MEMO_RECALL_EPISTEMIC_LABELS.",
    ),
    _spec(
        "MEMO_RECALL_RECENCY_BAND_DAYS",
        "int",
        0,
        "recall",
        "khoj-style recency band: unconditionally union the newest durable "
        "memories (< N days, cap 3) into the recall candidate pool, scored at "
        "the min_sim floor so they survive the gate but never outrank genuine "
        "semantic matches. Cheap indexed SQL, no embedder. 0 = off (default); "
        "flip only after the eval gate measures a win.",
        min_val=0,
    ),
    _spec(
        "MEMO_RECALL_UNMATCHED_TERM_GATE",
        "bool",
        True,
        "recall",
        "Honest-empty gate (cipher-style): when the top recall score is under "
        "MEMO_RECALL_UNMATCHED_GATE_MAX_SCORE AND no distinctive prompt term "
        "(>=4 chars, non-stopword) appears in any candidate, inject nothing — "
        "an honest empty beats weak noise, and the zero-hit turn feeds "
        "`memo gaps`. Strong semantic (paraphrase) matches are never gated. "
        "Default ON (opt out with =0); eval-gated (no recall regression).",
    ),
    _spec(
        "MEMO_RECALL_UNMATCHED_GATE_MAX_SCORE",
        "float",
        0.55,
        "recall",
        "Score ceiling for the unmatched-term gate: at/above this the gate never "
        "fires (protects paraphrase-only semantic recall).",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RECALL_INTRA_DEDUP",
        "bool",
        True,
        "recall",
        "Collapse near-duplicate hits within a single injection (lexical Jaccard). "
        "Default ON (opt out with =0); eval-gated (no recall regression).",
    ),
    _spec(
        "MEMO_RECALL_INTRA_DEDUP_THRESHOLD",
        "float",
        0.8,
        "recall",
        "Jaccard threshold for intra-injection near-dup collapse.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RECALL_DEDUP_COLLAPSE",
        "bool",
        True,
        "recall",
        "Collapse lexical paraphrase-duplicate hits in the over-fetched pool BEFORE "
        "top-K truncation (reuses the hook-safe token-Jaccard collapse), so near-dups "
        "don't crowd out distinct results. Distinct from MEMO_RECALL_INTRA_DEDUP "
        "(post-top-K). Cosine variant deferred (hits carry no vectors on the hook). "
        "Default ON (v3.0.0): net-positive in paraphrase-crowding (fixture recall "
        "+0.333, 0 noise), provably never-negative on the regression corpus (Δ0.0).",
    ),
    _spec(
        "MEMO_RECALL_SESSION_TOKEN_BUDGET",
        "int",
        0,
        "recall",
        "Cumulative recall-token budget per session; past it, per-turn budget decays "
        "(0 = off). Conservative: only scales down, never hard-bails.",
        min_val=0,
    ),
    _spec(
        "MEMO_RECALL_PRECISION_GATE",
        "bool",
        False,
        "recall",
        "Suppress recall when the top-hit score falls in a score band that has "
        "historically never been grounded (learned from grounding.log + recall.log; "
        "bands cached to state_dir/precision_bands.json by the Stop hook). "
        "Default OFF: with this unset, recall behavior is unchanged.",
    ),
    _spec(
        "MEMO_RECALL_VERBOSITY_LEVEL",
        "int",
        0,
        "recall",
        "Verbosity steering for recall output (L4 token savings): 0 (no steering) | "
        "1 (skip preamble/postamble) | 2 (skip code restatement, reference by path+line) | "
        "3 (minimum tokens, fragments OK, no preamble unless asked). Wave 1 L4 lever; "
        "byte-stable per level for idempotency; default 0 for backward compatibility.",
        min_val=0,
        max_val=3,
    ),
    _spec(
        "MEMO_GUARD_ENABLED",
        "bool",
        False,
        "recall",
        "Flag a prior decision/preference the prompt looks to be reversing, "
        "as a ⚠ banner at the top of the recall block. Advisory, never blocks.",
    ),
    _spec(
        "MEMO_GUARD_SIM_THRESHOLD",
        "float",
        0.6,
        "recall",
        "Min recall score for a decision/preference hit to be guard-flagged.",
        min_val=0.0,
        max_val=1.0,
    ),
    # Phase 3 — INTERJECT: sharpen the guard banner. Fires ONLY when a guard
    # candidate is at HIGH calibrated confidence AND already flagged in the
    # persisted contradiction store. Deduped + budgeted per session, one-key
    # silence. Report-only (never auto-graduatable); shadow-logs regardless of
    # this flag so a human can review before flipping it.
    _spec(
        "MEMO_INTERJECT_ENABLED",
        "bool",
        False,
        "recall",
        "Render the ⚠ INTERJECT banner atop the recall block when a guard "
        "candidate (prior decision/preference the prompt looks to reverse) is at "
        "HIGH calibrated confidence AND already in the contradiction store "
        "(persisted open/competing pair). Advisory, never blocks; deduped + "
        "budget-capped per session (MEMO_INTERJECT_MAX_PER_SESSION), silenceable. "
        "Shadow-counts what it WOULD interject even when OFF (memo interject "
        "shadow). Default OFF; not auto-graduatable.",
    ),
    _spec(
        "MEMO_INTERJECT_MAX_PER_SESSION",
        "int",
        1,
        "recall",
        "Budget: max interject banners rendered per session before the "
        "per-session marker silences further fires. 0 = never render (still "
        "shadow-logs).",
        min_val=0,
    ),
    _spec(
        "MEMO_HIT_DOSSIER",
        "bool",
        False,
        "recall",
        "Render a compact per-hit trust line (type · date · conf-band · ⚔ disputed by "
        "[id]) under each recalled memory. Pure render + one cheap contradict-pairs "
        "lookup; respects MEMO_RECALL_TOKEN_BUDGET. Default off.",
    ),
    _spec(
        "MEMO_RECALL_CONFIDENCE_GATE",
        "bool",
        False,
        "recall",
        "Recall-time uncertainty gating: a hit whose CALIBRATED confidence band "
        "(confidence_calibration map over its score-band) is 'low' renders with a "
        "'⚠ unverified — consider checking' marker instead of authoritative, "
        "reusing the epistemic '?unverified' framing. Pure render-layer (no store "
        "read, no MLX on the hook path); ranking untouched. Default OFF.",
    ),
    _spec(
        "MEMO_RECALL_CODE_REFS_ENABLED",
        "bool",
        False,
        "recall",
        "Append a '  ↳ code: <path>:<line>' line under each recalled memory that "
        "carries extra.code_refs, verified live against the codegraph index "
        "(one indexed SELECT per ref over nodes by file_path [+ name]): "
        "'(vigente)' on a positive lookup, '(desaparecido)' when the DB is live "
        "but the ref no longer resolves, '(no verificado)' when the DB is "
        "unavailable or the lookup fails. One read-only sqlite connection per "
        "render, capped at 2 refs/memory and 4 lines/render (the token budget "
        "still wins). Rendered by the full and balanced formats; compact stays "
        "one-line-per-hit and never renders it. Default OFF: zero extra work "
        "on the recall hot path (the codegraph DB is never even opened).",
    ),
    # ── Negative Recall — the ⛔ AVOID channel ────────────────────────────────
    # A preemptive, high-precision pass over type=failure_pattern anti-memories,
    # rendered as a distinct ⛔ AVOID block and excluded from normal recall. The
    # pass REUSES the already-computed query embedding (LRU cache hit — no second
    # MLX forward) and is budget-gated (first optional stage to skip). All OFF by
    # default; the *_ENABLED flags declare their graduation gate in
    # dream_flags.GATES (CI-enforced by tests/test_dream_flags.py).
    _spec(
        "MEMO_NEGATIVE_RECALL_ENABLED",
        "bool",
        False,
        "recall",
        "Master switch for Negative Recall: run the preemptive ⛔ pass over "
        "type=failure_pattern anti-memories (reusing the cached query embedding), "
        "exclude failure_pattern from normal recall (no duplication), and surface "
        "the distinct ⛔ AVOID block in the recall hook + El Briefing. OFF ⇒ "
        "failure_patterns flow into normal recall exactly as today.",
    ),
    _spec(
        "MEMO_NEGATIVE_RECALL_K",
        "int",
        2,
        "recall",
        "Max ⛔ AVOID anti-memories injected per prompt.",
        min_val=1,
    ),
    _spec(
        "MEMO_NEGATIVE_RECALL_MIN_SIM",
        "float",
        0.6,
        "recall",
        "Cosine similarity floor for a ⛔ hit — high-precision, above the normal "
        "MEMO_RECALL_MIN_SIM (0.5) so only strongly-matching mistakes surface.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED",
        "bool",
        False,
        "recall",
        "Auto-derive failure_pattern anti-memories in the nightly dream passes: "
        "from supersede/reversal decisions (superseded=Wrong, superseding=Right) "
        "and from graduated avoid verdicts. Provenance recorded in extra. OFF by "
        "default.",
    ),
    _spec(
        "MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED",
        "bool",
        False,
        "recall",
        "Close the loop in the dream ROI-reconcile pass: strengthen a "
        "failure_pattern whose ⛔ was surfaced but the mistake repeated anyway "
        "(confidence/ROI up), mild-positive when heeded. Off the 5s hook path. "
        "OFF by default.",
    ),
    _spec(
        "MEMO_NEGATIVE_RECALL_TRIGGER_ENABLED",
        "bool",
        False,
        "recall",
        "Raise the ⛔ pass recall (loosen the floor / raise K) in detected "
        "high-risk contexts (release/delete/deploy/refactor/migrate) via a pure "
        "O(len(prompt)) keyword scan — never re-embeds. OFF by default.",
    ),
)


def flag_recall_verbosity_level() -> int:
    """Verbosity level for recall hook output steering (0–3, default: 0 = no steering)."""
    from memo.flags import flag_int

    val = flag_int("MEMO_RECALL_VERBOSITY_LEVEL")
    if val is None:
        return 0
    return max(0, min(3, val))  # Clamp to [0, 3]
