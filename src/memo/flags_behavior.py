from __future__ import annotations

from memo.flags_base import FlagSpec, _spec

SPECS: tuple[FlagSpec, ...] = (
    # temporal / contradiction detection
    _spec(
        "MEMO_CONTRADICTION_TIMEOUT",
        "float",
        30.0,
        "temporal",
        "Timeout (seconds) for LLM pair-classification in contradiction detection. "
        "Set higher for slow models, lower to stay within hook budgets.",
        min_val=1.0,
    ),
    _spec(
        "MEMO_SUPERSEDE_SUPPORT_GATE",
        "int",
        0,
        "temporal",
        "When > 0: `memo maintain` refuses to auto-archive the losing side of a "
        "contradiction whose memory_health.support_count >= this gate — the pair "
        "stays open for manual triage (`memo contradict list --status open`) and "
        "is reported in the receipt under flagged_for_review. 0 (default) = off.",
        min_val=0,
    ),
    _spec(
        "MEMO_BELIEF_COMPETING",
        "bool",
        False,
        "temporal",
        "When on, contradiction resolution (maintain + nightly Dream) picks the "
        "loser by trust (confidence x roi_score), not raw recency, and marks the "
        "pair 'competing' (both kept) when the two sides are within "
        "MEMO_SUPERSEDE_MARGIN. Default off = legacy recency-newer-wins.",
    ),
    _spec(
        "MEMO_SUPERSEDE_MARGIN",
        "float",
        0.15,
        "temporal",
        "Trust-score band (|conf_a x roi_a - conf_b x roi_b|) under which neither "
        "side dominates: the pair is marked 'competing' instead of superseded. "
        "Only active when MEMO_BELIEF_COMPETING is on.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_BELIEF_NWAY",
        "bool",
        False,
        "temporal",
        "When on, a connected component of 3+ mutually-contradicting memories is "
        "marked 'competing' (all pairs) instead of pairwise auto-superseded. "
        "Requires MEMO_BELIEF_COMPETING. Default off.",
    ),
    _spec(
        "MEMO_CONTRADICT_MUTABILITY",
        "bool",
        False,
        "temporal",
        "Regex mutability classes (stable/volatile/ephemeral) in the "
        "contradiction scanner: an LLM 'contradiction' verdict between two "
        "VOLATILE-class bodies (ports/versions/status) is downgraded to "
        "'evolution' — a normal update, not a conflict — so maintain demotes "
        "instead of archiving. Default off.",
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
        "MEMO_ENTITY_EXTRACT_ON_SAVE",
        "bool",
        True,
        "entity",
        "Write regex-extracted entities into extra['entities'] on EVERY save "
        "(dependency-free, no MLX). Default-on so the entity-overlap signal "
        "exists corpus-wide and entity retrieval works the moment it's enabled, "
        "without a backfill. Set 0 to skip the on-save write.",
    ),
    _spec(
        "MEMO_GRAPH_RETRIEVAL_ENABLED",
        "bool",
        False,
        "entity",
        "Include knowledge-graph candidates in hybrid search. Memories sharing entities with the query are fused via RRF.",
    ),
    _spec(
        "MEMO_GRAPH_DENSITY_BOOST",
        "float",
        0.0,
        "entity",
        "Reranking boost for well-connected memories in the knowledge graph. "
        "Multiplies graph candidates' scores by (1.0 + boost * degree). "
        "0.0 (default) = off. Higher values favor semantically central memories.",
        min_val=0.0,
    ),
    _spec(
        "MEMO_GRAPH_FALLBACK_MIN_HITS",
        "int",
        0,
        "entity",
        "Fallback seeding: if vec retrieval returns fewer than this many hits, "
        "automatically enable graph candidates to boost coverage. "
        "0 (default) = off (graph only used if MEMO_GRAPH_RETRIEVAL_ENABLED=1).",
        min_val=0,
    ),
    # NB: MEMO_GRAPH_SEMANTIC_RELATIONS lives in flags_search.py (the live
    # spec — search_ops reads it). A stale duplicate here used to shadow it
    # in REGISTRY because behavior specs are merged after search specs.
    # session checkpoints / resume
    _spec(
        "MEMO_SESSION_DISABLE", "bool", False, "session", "Disable session checkpoint/recent hooks."
    ),
    _spec("MEMO_SESSION_DEBUG", "bool", False, "session", "Verbose session-hook diagnostics."),
    _spec(
        "MEMO_SESSION_RECENT_LIMIT",
        "int",
        12,
        "session",
        "Rows fetched/shown by `memo session recent` (the SessionStart resume panel).",
    ),
    _spec(
        "MEMO_SESSION_LRU_CAP",
        "int",
        250,
        "session",
        "Max session checkpoint files retained (shared across all agents/projects).",
    ),
    _spec(
        "MEMO_RESUME_ACTIVE_WINDOW_S",
        "int",
        120,
        "session",
        "Seconds a session's transcript can be idle and still count as `active` "
        "in the cross-agent federated `memo resume --agent all` picker.",
    ),
    _spec(
        "MEMO_RESUME_SCAN_CAP",
        "int",
        150,
        "session",
        "Max transcripts a `memo resume` provider fully-parses per agent "
        "(newest-first by mtime). Bounds picker latency on machines with many "
        "sessions; older sessions beyond the cap are not surfaced.",
    ),
    _spec(
        "MEMO_EPISODIC_ENABLED",
        "bool",
        True,
        "session",
        "Episodic memory: index work sessions into a semantic index so "
        "`memo resume` searches the full history by meaning (not just recency). "
        "Off ⇒ picker is recency+substring only, no session indexing.",
    ),
    _spec(
        "MEMO_RESUME_SEMANTIC_K",
        "int",
        50,
        "session",
        "Top-k sessions the `memo resume` picker pulls from the episodic index per semantic query.",
    ),
    _spec(
        "MEMO_RESUME_INDEX_BATCH",
        "int",
        500,
        "session",
        "Max sessions embedded per `memo episodes index` backfill run "
        "(newest-first). The full history is covered over successive runs.",
    ),
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
    _spec(
        "MEMO_CAPTURE_TOOL_EVIDENCE",
        "bool",
        True,
        "capture",
        "Project the exchange's tool stream (files edited, commands+exit, test "
        "results) into a compact 'TOOL ACTIVITY' line fed to the capture "
        "extractor. Turns impressionistic prose ('I fixed it') into grounded, "
        "high-retrievability memories with real file/symbol/command tokens. "
        "Set 0 to extract from prose only.",
        opt_out=True,
    ),
    _spec(
        "MEMO_CAPTURE_TOOL_EVIDENCE_CHARS",
        "int",
        300,
        "capture",
        "Hard cap on the TOOL ACTIVITY projection per message (keeps verbose "
        "command output from swamping the extractor context).",
    ),
    _spec(
        "MEMO_VERDICT_ENABLED",
        "bool",
        False,
        "capture",
        "Classify the user's NEXT turn (heuristic regex ES+EN) as a "
        "positive/negative/correction reaction to the PRIOR turn's recalled "
        "memories, from the Stop hook. Writes implicit source_feedback "
        "(click/ignore, never thumbs_down; only_if_absent — a manual vote "
        "always wins) + verdict.log for negative tuner labels. Never runs in "
        "the 5s recall hook. Default off.",
    ),
    _spec(
        "MEMO_VERDICT_MLX",
        "bool",
        False,
        "capture",
        "When the heuristic returns no verdict, run ONE bounded MLX chat "
        "call (max_tokens=4) to classify the reaction. Stop-hook only; "
        "requires MEMO_VERDICT_ENABLED. Default off.",
    ),
    _spec(
        "MEMO_CAPTURE_DUP_THRESHOLD",
        "float",
        0.85,
        "capture",
        "Cosine similarity at/above which a capture candidate is considered "
        "'near' an existing memory (same topic).",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_CAPTURE_DUP_DROP_THRESHOLD",
        "float",
        0.97,
        "capture",
        "At/above this similarity a candidate is a near-identical paraphrase "
        "(no new info) and is dropped. Between DUP_THRESHOLD and this, the "
        "candidate is ADMITTED as new — a same-topic decision evolving — so the "
        "nightly contradiction/evolution pass can supersede the older side "
        "instead of the new fact being silently lost.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_CAPTURE_META_FILTER",
        "bool",
        True,
        "capture",
        "Drop process-narration segments ('voy a…', 'let me…', \"I'll…\") and "
        "LLM filler from capture candidates before save. Regex-based, no LLM "
        "call. A candidate that is ALL narration is dropped (logged in debug); "
        "mixed candidates keep their substantive segments. Set 0 to disable.",
        opt_out=True,
    ),
    _spec(
        "MEMO_CAPTURE_BATCH_DEDUP",
        "bool",
        True,
        "capture",
        "Intra-batch near-dup window: within one capture run, collapse "
        "candidates that are near-duplicates of EACH OTHER (the prompt-retry "
        "pattern — same fact extracted 2-3x), keeping the higher-confidence/"
        "longer one. Uses MEMO_CAPTURE_DUP_THRESHOLD; the store-level dedup "
        "only sees already-saved memories, so it can't catch these. Set 0 to "
        "disable.",
        opt_out=True,
    ),
    _spec(
        "MEMO_CAPTURE_MIN_CONFIDENCE",
        "float",
        0.0,
        "capture",
        "Type-classification confidence floor (0-1). Candidates scoring below "
        "it are still saved but tagged '_uncertain' for later review. The "
        "score itself is always stamped in extra['capture_confidence']. "
        "Default 0.0 = gating off (conservative rollout).",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_CAPTURE_TYPE_FEEDBACK",
        "bool",
        False,
        "capture",
        "Citation-type feedback: at capture's genuinely ambiguous type "
        "classifications (claimed type has no corroborating markers while "
        "another type's markers are present), re-type the candidate to the "
        "marker-backed type with the highest citation weight from "
        "state_dir/capture/type_weights.json (computed nightly by the dream "
        "capture_weights pass from grounding.log). Off (default) or no "
        "weights file = classification untouched.",
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
    _spec(
        "MEMO_QUALITY_COMPACT",
        "bool",
        False,
        "maintain",
        "Enable the quality-compaction maintenance command. Default off; preview is read-only and apply is explicit.",
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
        "MEMO_SYNTHESIS_MAX_MEMBERS",
        "int",
        0,
        "maintain",
        "Per-topic size invariant (memobase-style bound): a synthesis cluster "
        "larger than this re-clusters into subtopics at a tighter threshold "
        "before the LLM abstracts it; unsplittable clusters are sliced so the "
        "bound always holds. 0 (default) = off.",
        min_val=0,
        max_val=500,
    ),
    _spec(
        "MEMO_SYNTHESIS_BODY_MAX_CHARS",
        "int",
        0,
        "maintain",
        "Character cap for a generated synthesis body. Over the cap → one "
        "bounded re-summarize LLM call, hard truncation on failure. "
        "0 (default) = off.",
        min_val=0,
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
    _spec(
        "MEMO_CONSOLIDATE_TIMEOUT",
        "int",
        180,
        "maintain",
        "Seconds per LLM call during consolidation (merge proposal, classification, synthesis). "
        "Increase for large LLMs that cold-load slowly or generate long outputs.",
        min_val=30,
        max_val=3600,
    ),
    _spec(
        "MEMO_MAINT_SYNTHESIZE",
        "bool",
        False,
        "maintain",
        "When set, `memo maintain` runs an emergent-synthesis pass on clusters of memories "
        "added/updated since the last synthesis. Tracks state in <state_dir>/synthesis_state.json. "
        "Non-blocking: a synthesis failure logs a warning but does not abort the maintain cycle.",
    ),
    _spec(
        "MEMO_MAINT_SLEEP_CYCLE_ENABLED",
        "bool",
        False,
        "maintain",
        "Enable autonomous background maintenance (sleep cycle). Runs synthesize/consolidate when the system is idle.",
    ),
    _spec(
        "MEMO_MAINT_SLEEP_CYCLE_INTERVAL",
        "int",
        3600,
        "maintain",
        "Seconds between sleep cycle maintenance passes (default 1h).",
    ),
    _spec(
        "MEMO_MAINT_IDLE_THRESHOLD_SECS",
        "int",
        300,
        "maintain",
        "Seconds of idle time (no recall/search activity) before a sleep cycle pass is eligible to start.",
    ),
    _spec(
        "MEMO_SYNC_MEMFLOW_ENABLED",
        "bool",
        False,
        "maintain",
        "Eager Synthesis: automatically ingest .memflow session data during the sleep cycle.",
    ),
    _spec(
        "MEMO_DREAM_EDGE_VERIFY_ENABLED",
        "bool",
        False,
        "maintain",
        "Nightly Dream pass: memory↔memory knowledge-graph edges EARN their "
        "confidence from grounded co-use evidence (grounding.log turns where "
        "both endpoints were recalled AND actually used in the answer), while "
        "edges that never accumulate evidence decay gently — floored, "
        "reversible, never deleted. Curation of edge confidence only; never "
        "touches recall ranking. Default off.",
    ),
    _spec(
        "MEMO_MEMFLOW_DIR",
        "str",
        ".memflow",
        "maintain",
        "Relative or absolute path to the .memflow directory (defaults to current dir/.memflow).",
    ),
    # privacy / redaction (consumed by capture.py, cli_ingest.py)
    _spec(
        "MEMO_REDACT_SECRETS",
        "bool",
        True,
        "privacy",
        "Mask secrets (AWS/GitHub/OpenAI/Anthropic/Slack/GCP keys, PEM private-key "
        "blocks) to ****last4 before persisting capture/ingest content, tagging the "
        "memory _redacted. Pattern tier only — near-zero false positives. Default on "
        "(security fix); set 0 to store raw text.",
    ),
    _spec(
        "MEMO_REDACT_ENTROPY",
        "bool",
        False,
        "privacy",
        "Also mask long high-entropy mixed-class tokens. Opt-in: false-positive-"
        "prone; pure-hex hashes/ids are always exempt.",
    ),
    _spec(
        "MEMO_PRIVATE_MARKERS",
        "bool",
        True,
        "privacy",
        "Strip <private>...</private> spans from transcript text before capture "
        "extraction or mine-history ever see it. An unclosed <private> drops to "
        "end-of-text (fail-closed). Set 0 to disable.",
    ),
    _spec(
        "MEMO_SAVE_NORMALIZE_DATES",
        "bool",
        False,
        "behavior",
        "Annotate relative date expressions in saved content with absolute ISO "
        "dates ('ayer' -> 'ayer (2026-07-02)') using the ES+EN patterns of "
        "_normalize_relative_dates. Anchored to the `created` override when the "
        "caller back-dates (imports); durable tiers only — reference chunks are "
        "never rewritten. Default off.",
    ),
    _spec(
        "MEMO_GROUNDING_ASK_MIN",
        "float",
        0.0,
        "recall",
        "Ask-path abstention floor (0.0-1.0; 0 = off). After ask() drafts an answer, "
        "score how well the recalled sources entail it; below this floor ask abstains "
        "with MEMO_ASK_FALLBACK_MSG instead of emitting an unsupported inference. "
        "Off the recall hook (ask path only). Default 0 = off.",
        min_val=0.0,
        max_val=1.0,
    ),
    # client sampling (MCP synthesis via the caller's model)
    _spec(
        "MEMO_SAMPLING_SYNTH_ENABLED",
        "bool",
        False,
        "sampling",
        "When on, MCP synthesis tools (memo_ask / memo_chat_ask / memo_reflect / "
        "memo_synthesize_run / memo_consolidate) delegate LLM synthesis to the "
        "connected client's model via MCP sampling, falling back to local MLX on "
        "any failure. Off = local MLX only (legacy).",
    ),
    _spec(
        "MEMO_SAMPLING_TIMEOUT_S",
        "float",
        30.0,
        "sampling",
        "Per-sample timeout (seconds) for a client-sampling round trip. On "
        "timeout the request falls back to MLX for its remainder.",
        min_val=1.0,
    ),
    _spec(
        "MEMO_SAMPLING_MAX_CALLS",
        "int",
        3,
        "sampling",
        "Max client-sampling calls per MCP request; past the cap, remaining "
        "synthesis in the same request uses MLX (multi-call flows like "
        "synthesize/consolidate stay bounded).",
        min_val=1,
    ),
    _spec(
        "MEMO_SAMPLING_MAX_TOKENS",
        "int",
        2000,
        "sampling",
        "Hard cap on max_tokens requested per sample call (caller options are clamped to this).",
        min_val=64,
    ),
)
