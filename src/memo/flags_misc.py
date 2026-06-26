from __future__ import annotations

from memo.flags_base import FlagSpec, _spec

SPECS: tuple[FlagSpec, ...] = (
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
        0.85,
        "feedback",
        "Similarity threshold for feedback matching.",
        min_val=0.0,
    ),
    _spec(
        "MEMO_FEEDBACK_BOOST_PER_VOTE",
        "float",
        0.15,
        "feedback",
        "Score boost added per upvote.",
        min_val=0.0,
    ),
    _spec(
        "MEMO_FEEDBACK_BOOST_CAP",
        "float",
        0.6,
        "feedback",
        "Maximum cumulative feedback boost.",
        min_val=0.0,
    ),
    # auto-update (memo-mcp update on start, gated + throttled)
    _spec(
        "MEMO_AUTO_UPDATE",
        "bool",
        False,
        "update",
        "On memo-mcp start, check for a newer git TAG and update in the "
        "background (takes effect next start). Default off; enable per-machine.",
    ),
    _spec(
        "MEMO_AUTO_UPDATE_INTERVAL_S",
        "int",
        21600,
        "update",
        "Min seconds between auto-update checks (throttle). Default 6h.",
        min_val=0,
    ),
    _spec(
        "MEMO_AUTO_UPDATE_REPO",
        "str",
        "",
        "update",
        "Git repo URL to check tags / install from (empty → the memo default).",
    ),
    # MCP transport
    _spec("MEMO_MCP_TRANSPORT", "str", "stdio", "mcp", "MCP transport: stdio | http."),
    _spec("MEMO_MCP_HOST", "str", "127.0.0.1", "mcp", "Bind host for the HTTP MCP transport."),
    _spec("MEMO_MCP_PORT", "int", 18768, "mcp", "Bind port for the HTTP MCP transport."),
    _spec(
        "MEMO_DASHBOARD_PORT",
        "int",
        8787,
        "mcp",
        "Bind port for the local health dashboard server (memo dashboard).",
    ),
    _spec(
        "MEMO_MCP_SLIM",
        "bool",
        False,
        "mcp",
        "Expose only the 26 core inline tools (skip domain modules). "
        "Reduces tool count from ~116 to 26 for local/constrained LLMs.",
    ),
    _spec(
        "MEMO_MCP_PROFILE",
        "str",
        "agent",
        "mcp",
        "MCP surface profile: agent (default, 5 tools) | core/slim (stable core) | full/default (all tools).",
    ),
    _spec(
        "MEMO_CLI_PROFILE",
        "str",
        "default",
        "cli",
        "CLI surface profile: default/full expose every command; core/slim hide advanced and experimental commands.",
    ),
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
        # DEPRECATED: only 'off' is tested/used; smart/prefetch/aggressive modes are dead code.
        # Kept for backward compat. Any value other than 'off' has no effect.
        "Cache tier mode: off (default, durable store). Other values (read_through/write_through/write_back) "
        "are DEPRECATED and have no effect — only 'off' is wired.",
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
        "Freshness window in days before a cached memory is revalidated against the backing store / eligible for ttl eviction (0 = off).",
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
    _spec(
        "MEMO_OCR_MIN_CONFIDENCE",
        "float",
        0.4,
        "ingest",
        "Per-line OCR confidence floor: drop mojibake lines below this before indexing.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_OCR_LOW_CONF_THRESHOLD",
        "float",
        0.6,
        "ingest",
        "Mean OCR confidence below which an image is down-weighted (memory_health.confidence).",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_TEXT_QUALITY",
        "bool",
        True,
        "ingest",
        "Universal text-quality gate: down-weight garbled records (mojibake) from any source.",
        opt_out=True,
    ),
    _spec(
        "MEMO_TEXT_QUALITY_THRESHOLD",
        "float",
        0.02,
        "ingest",
        "Replacement/control-char ratio at/above which a record is down-weighted.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RETRIEVAL_BOOST",
        "bool",
        True,
        "retrieval",
        "Apply filename/title/heading/tag curatorial boost to memory-surface "
        "search ranking (a note whose metadata is the answer wins decisively). "
        "Measured: precision@5 +0.04..+0.08 across configs, noise unchanged.",
        opt_out=True,
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
        "Before saving, run a quick vec search for near-duplicates. In interactive mode, prompts to update an existing memory instead. In non-interactive mode, logs a warning.",
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
        "Auto-tag saved memories with the cwd project.",
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
        "I couldn't find an answer in the saved memories",
        "misc",
        "Message returned by memo_ask when no relevant sources are found.",
    ),
    _spec(
        "MEMO_VAULT_SYSTEM_DIR",
        "str",
        "Obsidian",
        "misc",
        "Vault subdir holding memo's system tree (AI/, Contacts/, Whatsapp/).",
    ),
    _spec(
        "MEMO_MEMORIES_IN_VAULT",
        "bool",
        False,
        "misc",
        "Store curated memory .md files INSIDE the Obsidian vault "
        "(<vault>/<SYSTEM_DIR>/AI/memory) instead of data_dir, making the "
        "vault the human-editable source of truth. Requires MEMO_VAULT_PATH. "
        "sqlite stays a rebuildable index. Ingest already excludes AI/ and "
        "id:-frontmatter files, so memories are never double-ingested as "
        "reference tier. Default off keeps existing installs untouched.",
    ),
    _spec(
        "MEMO_SINGLE_DB",
        "bool",
        False,
        "misc",
        "Consolidate the sidecar sqlite stores (history, graph, contradictions, "
        "crossref) into the single main DB file (memvec.db) instead of separate "
        "*.db files. Each store keeps its own connection to the one file (WAL "
        "allows it), so there's no shared-transaction risk. Run "
        "`memo migrate --consolidate-db` once to merge existing sidecar files. "
        "Default off keeps the historical multi-file layout.",
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
    _spec(
        "MEMO_SOURCE",
        "str",
        "",
        "roi",
        "Identity of the calling layer (synapse / memflow / devin / devin-desktop "
        "/ opencode …) used to attribute memo consults in `memo usefulness` when a "
        "read does not pass an explicit source. Set it in that tool's CLI/MCP "
        "environment; a bare developer consult leaves it empty and is not counted.",
    ),
    # The Outcome Loop — recall self-tunes from real grounding outcomes.
    _spec(
        "MEMO_OUTCOME_RANKING_ENABLED",
        "bool",
        True,
        "roi",
        "Drive memory_health.roi_score from real grounding OUTCOMES (was the "
        "surfaced memory actually used in the answer?) instead of mere access "
        "frequency. When on, `memo maintain` / `memo outcome` reconcile roi_score "
        "from recall.log+grounding.log, and the blind per-access roi boost is "
        "skipped so the outcome signal stays authoritative. Disable only if you "
        "need the legacy access-driven behaviour.",
    ),
    _spec(
        "MEMO_OUTCOME_PRIOR_N",
        "float",
        3.0,
        "roi",
        "Bayesian prior strength for per-memory utility = (grounded + "
        "prior_mean*prior_n) / (surfaced + prior_n). Higher = more surfacings "
        "needed before a memory's utility moves off the global baseline.",
        min_val=0.0,
    ),
    _spec(
        "MEMO_OUTCOME_ROI_FLOOR",
        "float",
        0.6,
        "roi",
        "Lowest roi_score the outcome loop assigns (utility 0 → floor). Demotes "
        "but never zeroes a memory that surfaces a lot yet never grounds.",
        min_val=0.0,
    ),
    _spec(
        "MEMO_OUTCOME_ROI_CAP",
        "float",
        1.5,
        "roi",
        "Highest roi_score the outcome loop assigns (utility 1 → cap).",
        min_val=1.0,
    ),
    _spec(
        "MEMO_OUTCOME_DEAD_MIN_SURFACED",
        "int",
        8,
        "roi",
        "Dead-weight archival: a memory must have been surfaced at least this "
        "many times (and never grounded) before `memo maintain` proposes "
        "archiving it as recall noise. 0 disables dead-weight archival.",
        min_val=0,
    ),
    _spec(
        "MEMO_ROI_TOKENS_PER_GROUNDED",
        "int",
        350,
        "roi",
        "Estimated model tokens saved per grounded answer — the tokens the model "
        "would have spent re-deriving the fact memo surfaced instead of being "
        "given it directly.",
        min_val=0,
    ),
    _spec(
        "MEMO_ROI_TOKENS_PER_REASK",
        "int",
        900,
        "roi",
        "Estimated model tokens saved per re-ask avoided — a full answer "
        "regeneration round-trip the user did NOT have to repeat.",
        min_val=0,
    ),
    # Git sync (memo-sync repo ↔ GitHub)
    _spec(
        "MEMO_SYNC_AUTO",
        "bool",
        True,
        "sync",
        "Enable the debounced in-session auto-sync (`memo sync auto`, wired as an "
        "async per-prompt hook). Default on; set 0 to fall back to push-on-Stop / "
        "pull-on-SessionStart only.",
    ),
    _spec(
        "MEMO_SYNC_PUSH_DEBOUNCE_S",
        "int",
        120,
        "sync",
        "Minimum seconds between auto-pushes (coalesces rapid saves into one "
        "commit+push so git isn't thrashed every prompt).",
        min_val=0,
    ),
    _spec(
        "MEMO_SYNC_PULL_INTERVAL_S",
        "int",
        300,
        "sync",
        "Minimum seconds between background auto-pulls — lets a long-running "
        "session converge on another Mac's pushes without a restart.",
        min_val=0,
    ),
    # Durable incremental capture (memo capture-tick)
    _spec(
        "MEMO_CAPTURE_INTERVAL_S",
        "int",
        600,
        "capture",
        "Minimum seconds between incremental in-session captures (`memo "
        "capture-tick`, wired as an async per-prompt hook). Mines NEW turns "
        "since a per-session watermark into durable memories so a long/crashed "
        "session's insight reaches .md mid-session instead of only at Stop. "
        "Self-throttled per session off the capture watermark; a cheap no-op "
        "when not due. 0 disables the throttle (capture every prompt).",
        min_val=0,
    ),
    _spec(
        "MEMO_SESSION_IDLE_CAPTURE_SECS",
        "int",
        10,
        "session",
        "Seconds of no new prompt before the delayed session-idle capture worker mines the current session chunk into durable memories.",
        min_val=0,
    ),
    _spec(
        "MEMO_SESSION_IDLE_REFLECT_SECS",
        "int",
        300,
        "session",
        "Seconds of no new prompt before the delayed session-idle reflect worker synthesizes the active session into durable memories.",
        min_val=0,
    ),
    _spec(
        "MEMO_AGENT_TTY",
        "str",
        "",
        "session",
        "Controlling TTY device path (e.g. /dev/ttys000) recorded by the shell "
        "shims (`memo runtime` writes it into ~/.zshrc / ~/.bashrc) so a detached "
        "worker can address the user's terminal. Set automatically per interactive "
        "session, not user-configured; registered here so `memo config validate` "
        "recognizes it instead of flagging it as an unknown MEMO_* var.",
    ),
    _spec(
        "MEMO_DEV_REPO",
        "str",
        "",
        "session",
        "Path to the memo source checkout (e.g. ~/repos/memo). When set, "
        "`memo doctor` compares the installed package against this repo and "
        "warns if they differ at the SAME version (stale build), and "
        "`memo release bump` targets this repo. Empty = checks skipped.",
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
    # dream pipeline
    _spec(
        "MEMO_DREAM_PRUNE_FLOOR",
        "float",
        0.15,
        "dream",
        "ROI score floor for quality-floor prune in `memo dream run`. "
        "Memories with roi_score below this threshold, zero access count, and age "
        "> MEMO_DREAM_PRUNE_MIN_AGE_DAYS are archived during the dream prune pass.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_DREAM_PRUNE_MIN_AGE_DAYS",
        "int",
        90,
        "dream",
        "Minimum age in days for the quality-floor prune pass in `memo dream run`. "
        "Only memories older than this are considered for archival.",
        min_val=0,
    ),
    _spec(
        "MEMO_DREAM_EVICT_MAX_COUNT",
        "int",
        0,
        "dream",
        "Corpus size ceiling for the dream eviction pass. When > 0 and the total "
        "non-reference memory count exceeds this value, the coldest LFU candidates "
        "are archived until the corpus is within budget. 0 disables eviction.",
        min_val=0,
    ),
    _spec(
        "MEMO_DREAM_COMPRESS_THRESHOLD",
        "int",
        0,
        "dream",
        "Body-length threshold (chars) for the dream verbose-compression pass. "
        "Memories whose body exceeds this length are LLM-compressed to 2-3 sentences "
        "during `memo dream run`. Disabled by default because Markdown is canonical; "
        "set an explicit positive value to opt in.",
        min_val=0,
    ),
    _spec(
        "MEMO_DREAM_PREWARM_QUERIES",
        "int",
        20,
        "dream",
        "Number of recent unique queries from recall.log to pre-embed during the "
        "dream prewarm pass, warming the LRU query-embed cache for the next session. "
        "0 disables the pass.",
        min_val=0,
    ),
    _spec(
        "MEMO_DREAM_PRESYNTHESIS_QUERIES",
        "int",
        0,
        "dream",
        "Number of top recurring queries from recall.log to pre-synthesize during "
        "the dream query-prediction pass. For each top query, memo searches for the "
        "matching memories and runs a focused synthesis pass on that cluster. "
        "0 disables the pass.",
        min_val=0,
    ),
    # graph co-recall
    _spec(
        "MEMO_GRAPH_CO_RECALL",
        "bool",
        False,
        "graph",
        "Record co-recall edges in graph.db whenever a search returns 2+ results. "
        "Pairs of co-recalled memory IDs are stored in the `co_recall` table with "
        "incrementing counts. Off by default to avoid graph DB write overhead on the "
        "recall-hook hot path.",
    ),
    # schema / embedding version check
    _spec(
        "MEMO_SKIP_MODEL_VERSION_CHECK",
        "bool",
        False,
        "misc",
        "Skip the embedder model/dims version check on VecStore open. "
        "Set to 1 in tests that use stub embedders with non-production dims/model names.",
    ),
    # strict mode — surface silent failures as exceptions for debugging
    _spec(
        "MEMO_STRICT",
        "bool",
        False,
        "misc",
        "Strict mode: raise exceptions instead of silently falling back in contextual retrieval, "
        "embedder client, reranker, and other fallback paths. Use for debugging.",
    ),
)
