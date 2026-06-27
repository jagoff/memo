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
        "Top-k sessions the `memo resume` picker pulls from the episodic index "
        "per semantic query.",
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
        "MEMO_MEMFLOW_DIR",
        "str",
        ".memflow",
        "maintain",
        "Relative or absolute path to the .memflow directory (defaults to current dir/.memflow).",
    ),
)
