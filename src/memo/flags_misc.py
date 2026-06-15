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
    _spec(
        "MEMO_MEMORIES_IN_VAULT",
        "bool",
        False,
        "misc",
        "Store curated memoria .md files INSIDE the Obsidian vault "
        "(<vault>/<SYSTEM_DIR>/AI/memory) instead of data_dir, making the "
        "vault the human-editable source of truth. Requires MEMO_VAULT_PATH. "
        "sqlite stays a rebuildable index. Ingest already excludes AI/ and "
        "id:-frontmatter files, so memorias are never double-ingested as "
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
    # schema / embedding version check
    _spec(
        "MEMO_SKIP_MODEL_VERSION_CHECK",
        "bool",
        False,
        "misc",
        "Skip the embedder model/dims version check on VecStore open. "
        "Set to 1 in tests that use stub embedders with non-production dims/model names.",
    ),
)
