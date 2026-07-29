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
    # update checks / auto-update (memo-mcp start, opt-in + throttled)
    _spec(
        "MEMO_UPDATE_CHECK_ENABLED",
        "bool",
        False,
        "update",
        "On memo-mcp start, check for a newer tagged release and record a "
        "notification. Default off; set =1 to opt in to remote checks.",
    ),
    _spec(
        "MEMO_AUTO_UPDATE",
        "bool",
        True,
        "update",
        "On memo-mcp start, check for a newer git TAG and update in the "
        "background (takes effect next start). Default ON: memo keeps itself "
        "current across every install. This is the one default-on outbound "
        "call (a throttled `git ls-remote` to the memo repo); set =0 to opt out "
        "and keep startup fully offline.",
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
    _spec(
        "MEMO_UPDATE_ENDPOINT",
        "str",
        "",
        "update",
        "HTTP endpoint for the update check. When set (and update checks are "
        "on), memo resolves the latest tag via GET <endpoint>?id=&v=&os= — a "
        "functional version check that also emits an anonymous deduped "
        "active-install heartbeat (id = sha256(device_id)[:16]; raw id never "
        "sent). Empty (default) → git ls-remote only, no heartbeat. Falls back "
        "to git on any HTTP failure.",
    ),
    _spec(
        "MEMO_STATUSLINE_SELFHEAL",
        "bool",
        False,
        "update",
        "On memo-mcp start, idempotently re-assert the [MEMO <ver>] statusLine "
        "wiring in ~/.claude/settings.json (wrapping any foreign statusline). "
        "No-op when already correct. Default off; set =1 to opt in.",
    ),
    _spec(
        "MEMO_HOOK_SELFHEAL",
        "bool",
        False,
        "update",
        "On memo-mcp start, idempotently re-assert the recall hook "
        "(UserPromptSubmit → memo recall-hook, absolute path) in "
        "~/.claude/settings.json, coexisting with foreign hooks. Makes recall "
        "survive a de-registered/clobbered plugin. Default off; set =1 to opt in.",
    ),
    _spec(
        "MEMO_STATUSLINE_ACTIVITY",
        "bool",
        True,
        "update",
        "Show today's activity (🧠 recalls · 💾 saves · ~tokens saved) in the "
        "[Memo <ver>] statusline badge, read from presence_today.json. The "
        "bash statusline reads the env var directly; this spec documents it "
        "for `memo config validate`. Default on; set =0 to opt out.",
        opt_out=True,
    ),
    _spec(
        "MEMO_PRESENCE_NOTIFY",
        "bool",
        True,
        "update",
        "Prepend a one-line activity summary (🧠 recalls · 💾 saves · ~tokens "
        "saved today) to the `notification` field of memo's MCP tool responses, "
        "so agents without a statusline (Codex, Devin, opencode, Cursor) still "
        "see memo working on every call. Default on; set =0 to silence just "
        "this channel (the statusline badge is unaffected).",
        opt_out=True,
    ),
    # MCP transport
    _spec(
        "MEMO_MCP_TRANSPORT",
        "str",
        "stdio",
        "mcp",
        "MCP transport: stdio | http | streamable-http | sse.",
        choices=("stdio", "http", "streamable-http", "sse"),
    ),
    _spec("MEMO_MCP_HOST", "str", "127.0.0.1", "mcp", "Bind host for the HTTP MCP transport."),
    _spec(
        "MEMO_MCP_PORT",
        "int",
        18768,
        "mcp",
        "Bind port for the HTTP MCP transport.",
        min_val=1,
        max_val=65535,
    ),
    _spec(
        "MEMO_MCP_ALLOW_NON_LOOPBACK",
        "bool",
        False,
        "mcp",
        "Explicitly acknowledge exposing authenticated MCP HTTP beyond loopback.",
    ),
    _spec(
        "MEMO_MCP_ALLOW_NO_AUTH",
        "bool",
        False,
        "mcp",
        "Disable MCP HTTP bearer auth for loopback-only development.",
    ),
    _spec(
        "MEMO_HTTP_HOST",
        "str",
        "127.0.0.1",
        "mcp",
        "Bind host used when loading the standalone REST app directly.",
    ),
    _spec(
        "MEMO_HTTP_ALLOW_NO_AUTH",
        "bool",
        False,
        "mcp",
        "Disable REST bearer auth for loopback-only development.",
    ),
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
        "MCP surface profile: agent (default, 30 tools) | core/slim (stable core) | full/default (all tools).",
        choices=("agent", "core", "slim", "full", "default"),
    ),
    _spec(
        "MEMO_MCP_WRITE_QUEUE_SIZE",
        "int",
        32,
        "mcp",
        "Bounded process-local FIFO for mutating MCP calls. Default 32; set 0 "
        "to disable it. The data-directory lock remains cross-process authority.",
        min_val=0,
        max_val=1024,
    ),
    _spec(
        "MEMO_RELATION_CANDIDATES_ENABLED",
        "bool",
        True,
        "relations",
        "Generate at most three canonical relation candidates after eligible saves. "
        "Default on after the fixed-corpus and save-latency gates passed; set 0 "
        "to opt out.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RELATION_ANNOTATIONS_ENABLED",
        "bool",
        True,
        "relations",
        "Attach judged canonical relations to normal retrieval results. "
        "Pending candidates are never attached. Default on; set 0 to opt out.",
        opt_out=True,
    ),
    _spec(
        "MEMO_RESOURCE_BODY_CHARS",
        "int",
        1200,
        "mcp",
        "Max body chars exposed by memo://memory/{id}; explicit memo_get still returns the full body.",
    ),
    _spec(
        "MEMO_SEARCH_JSON_BODY_CHARS",
        "int",
        280,
        "mcp",
        "Default body preview length for JSON search / recall output.",
    ),
    _spec(
        "MEMO_ASK_SNIPPET_CHARS",
        "int",
        800,
        "retrieval",
        "Default snippet length for ask/chat retrieval payloads.",
    ),
    _spec(
        "MEMO_CLI_PROFILE",
        "str",
        "default",
        "cli",
        "CLI surface profile: default/full expose every command; core/slim hide advanced and experimental commands.",
    ),
    # cache tier (opt-in: memo as a bounded cache fronting an authoritative backing store)
    _spec(
        "MEMO_CACHE_MODE",
        "str",
        "off",
        "cache",
        "Cache tier mode (opt-in; default off = durable store, nothing evicted). "
        "read_through fills from the backing store on a local miss (ask/chat); "
        "write_through pushes saves to the backing store now; write_back marks them "
        "dirty for a later flush. Only consulted when != off (see MEMO_CACHE_BACKEND).",
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
        "vault",
        "cache",
        "Memo-native backing store the cache fronts: vault | none. Only consulted when MEMO_CACHE_MODE != off.",
    ),
    # misc behavior
    _spec(
        "MEMO_GRAPH_USE_CODEGRAPH",
        "bool",
        True,
        "misc",
        "Fold the codegraph code graph (.codegraph/codegraph.db) into memo's "
        "graph navigation (path/neighbors/communities/centrality/export). "
        "Default on; set =0 to use only the entity-memory graph.",
        opt_out=True,
    ),
    _spec(
        "MEMO_CODEGRAPH_DISCOVERY",
        "bool",
        True,
        "misc",
        "Project-aware codegraph DB discovery: resolve the nearest "
        ".codegraph/codegraph.db walking up from cwd before falling back to "
        "memo's own checkout. Default on; set =0 to pin the historical "
        "checkout DB. Read raw in codegraph_loader (hot-path leaf, like "
        "MEMO_GPU_XPROC_LOCK); registered here for `memo config validate`.",
        opt_out=True,
    ),
    _spec(
        "MEMO_CODEGRAPH_MAX_EDGES",
        "int",
        300000,
        "misc",
        "Cap on traversable codegraph edges loaded per DB: over the cap, "
        "codegraph_loader.load() serves the cached graph (even stale) or "
        "raises instead of scanning every edge on the recall hot path. Read "
        "raw in codegraph_loader (hot-path leaf, like "
        "MEMO_CODEGRAPH_DISCOVERY); registered here for `memo config validate`.",
        min_val=1,
    ),
    _spec(
        "MEMO_CODEGRAPH_DB",
        "str",
        "",
        "misc",
        "Explicit path to a codegraph.db index, consulted only when cwd "
        "discovery finds no .codegraph/ upward from the working directory "
        "(launchd daemons at $HOME, pipx/uv-tool installs whose "
        "module-relative default points inside site-packages). Discovery "
        "still wins when it finds a nearer index, so project-awareness is "
        "preserved. Default '': fall back to memo's own checkout.",
    ),
    _spec(
        "MEMO_BRIEFING_GRAPH",
        "bool",
        True,
        "misc",
        "Add an entity-centric 'Knowledge map' (graph hubs + their clusters) to "
        "the SessionStart briefing and memo_unified_briefing. Default on.",
        opt_out=True,
    ),
    _spec(
        "MEMO_BRIEFING_CODE_DRIFT",
        "bool",
        True,
        "misc",
        "Surface last night's code-drift outcome (memories archived / partial / "
        "repaired) as one line in the SessionStart briefing, read from the dream "
        "receipt (state_dir/dream/last.json) — zero graph queries at "
        "SessionStart. Default on.",
        opt_out=True,
    ),
    _spec(
        "MEMO_GAPS_CODE_HUBS",
        "bool",
        True,
        "misc",
        "Flag knowledge gaps on code hubs in `memo ask-gaps`: top codegraph "
        "nodes by incoming call-edges with no memory citing them. On-demand "
        "graph query only (never the recall hook). Default on.",
        opt_out=True,
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
        "MEMO_SUPPORT_COUNT",
        "bool",
        True,
        "misc",
        "Bump memory_health.support_count when an existing memory is re-asserted "
        "(save near-dup hit, topic_key upsert, consolidation merge). Pure counter "
        "by default — no ranking effect until MEMO_SUPPORT_CONFIDENCE_LIFT > 0.",
        opt_out=True,
    ),
    _spec(
        "MEMO_SUPPORT_CONFIDENCE_LIFT",
        "float",
        0.0,
        "misc",
        "Confidence restored per corroboration bump (support_count), capped at "
        "1.0 — re-assertion undoes prior contradiction/quality penalties but "
        "never boosts above neutral. 0.0 (default) = counting only, no ranking "
        "effect. Ranking change: eval-gate before enabling.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_SAVE_ABSORB",
        "bool",
        False,
        "misc",
        "Living canonical records: when a save hits a near-duplicate "
        "(MEMO_SAVE_DEDUP_THRESHOLD), rewrite the EXISTING record via one "
        "bounded LLM call + versioned update() (rollbackable) instead of "
        "creating a near-copy; proof_count grows in extra. Requires "
        "MEMO_SAVE_DEDUP_CHECK. Skipped in derived-save scope "
        "(dream/consolidation). Never on the recall-hook path.",
    ),
    _spec(
        "MEMO_INVALIDATE_PENALTY",
        "float",
        0.3,
        "misc",
        "Confidence penalty applied per memory by `memo invalidate` "
        "(reversible bulk weakening; restored by `memo invalidate --undo`).",
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
    # MCP elicitation gate on irreversible tools (server_elicit.py)
    _spec(
        "MEMO_ELICIT_CONFIRM",
        "bool",
        True,
        "mcp",
        "MCP-side confirmation (elicitation) before irreversible tools "
        "(memo_delete, memo_synthesize_delete, memo_backup_restore, "
        "memo_feedback_clear, memo_repo_delete, memo_cache_evict). Fail-open: "
        "clients without the elicitation capability proceed unchanged. "
        "Set =0 for scripted elicitation-capable clients.",
        opt_out=True,
    ),
    _spec(
        "MEMO_ELICIT_DECLINE_SIGNAL",
        "bool",
        True,
        "mcp",
        "On an explicit elicitation decline (not cancel), save a durable "
        "type=feedback memory recording the refusal so it feeds memo's "
        "feedback loop. Fail-open: a failed signal save never blocks the abort.",
        opt_out=True,
    ),
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
        "MEMO_ASK_EXPAND_SYNTHESIS",
        "bool",
        False,
        "misc",
        "Lazy synthesis_sources expansion at ask time: when a retrieved hit is "
        "a type=synthesis abstract, pull up to 4 of its provenance memories "
        "into the ask context (store fetches only — no extra LLM call). "
        "Ask/chat path only; never the recall hook. Off by default.",
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
    _spec(
        "MEMO_STORE_BY_PROJECT",
        "bool",
        True,
        "misc",
        "Store new memory .md files in a per-project folder "
        "(memory_dir/<project>/, or _global/ when untagged) derived from the "
        "project: tag. The sqlite index globs recursively so search stays "
        "global — this is on-disk organization only. Existing flat files are "
        "untouched until `memo migrate --bucket-by-project`.",
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
        "Identity of the calling layer (codex / claude-code / devin / devin-desktop "
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
        "MEMO_OUTCOME_CITED_WEIGHT",
        "float",
        2.0,
        "roi",
        "How many grounded observations an explicitly CITED memory (the answer "
        "referenced [id]; grounding method='cited') counts as in the per-memory "
        "utility, vs 1 for mere lexical/embedding overlap — a citation is "
        "stronger evidence the recall was actually useful. 1.0 restores "
        "unweighted parity. Only read inside the outcome loop (`memo outcome` / "
        "reconcile), which ranking consumes only when "
        "MEMO_OUTCOME_RANKING_ENABLED is on.",
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
        "MEMO_TOKEN_METER_ENABLED",
        "bool",
        True,
        "misc",
        "Roll up measured per-session token usage from the transcript in the Stop hook.",
        opt_out=True,
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
    _spec(
        "MEMO_ROI_TOKENS_PER_CONSULT",
        "int",
        200,
        "roi",
        "Estimated model tokens saved per PRODUCTIVE memo consult (a search that "
        "returned >=1 hit) by a non-Claude-Code agent (codex/opencode/devin/"
        "other agents). These agents read memo over MCP/CLI/socket, so we "
        "log the consult but never see their answer — we can't ground it like a "
        "Claude Code turn. Priced BELOW MEMO_ROI_TOKENS_PER_GROUNDED (350) because "
        "the signal is weaker: memo returned relevant memory, use is unverified.",
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
    _spec(
        "MEMO_SYNC_EMBED_CACHE",
        "bool",
        True,
        "sync",
        "Carry the content-addressed embedding cache through the memo-sync repo "
        "(one embed_cache/<machine>.json shard per Mac). A pulling or "
        "bootstrapping machine imports peers' shards before the post-pull "
        "reindex, so memories saved on another Mac index with ~zero local MLX "
        "embed calls. Derived data only (vectors of already-synced .md bodies; "
        "reference/vault tier never exports) — set 0 to keep the sync payload "
        "markdown+signal only.",
    ),
    _spec(
        "MEMO_SYNC_EMBED_CACHE_MAX_ROWS",
        "int",
        1000,
        "sync",
        "Cap the embed-cache shard to the N most-recently-updated durable "
        "memories (their chunks ride along). Bounds the shard file: a 2560-dim "
        "vector is ~13.7KB in base64, so an uncapped mature corpus would put "
        "tens of MB in the sync repo. Peers that sync regularly still converge "
        "to full coverage (their local cache persists); only a fresh bootstrap "
        "re-embeds rows older than the window. 0 = no cap.",
        min_val=0,
    ),
    _spec(
        "MEMO_SYNC_SECRET_GATE",
        "bool",
        True,
        "sync",
        "Secret gate on the sync commit: scan staged .md ADDITIONS for API keys / "
        "private-key blocks before committing; on a hit the commit+push are blocked, "
        "sync_pending is stamped with the reason, and `memo sync status` / "
        "`memo doctor` surface it. Pattern tier only (no entropy heuristics). "
        "Set 0 to bypass once.",
    ),
    _spec(
        "MEMO_OUTCOME_SOURCE_FEEDBACK",
        "bool",
        False,
        "outcome",
        "Auto-mine per-query source_feedback from grounding outcomes during "
        "reconcile (implicit 'click' positives; 'ignore' negatives only when "
        "MEMO_OUTCOME_SOURCE_FEEDBACK_NEG=1). Off by default — it changes "
        "ranking from automated signal, so validate on your corpus before "
        "enabling. Never overrides a manual vote.",
    ),
    _spec(
        "MEMO_OUTCOME_SOURCE_FEEDBACK_NEG",
        "bool",
        False,
        "outcome",
        "Also write implicit-negative ('ignore') feedback for surfaced-but-"
        "unused (memory, query) pairs. Noisier than positives; default off.",
    ),
    # Durable incremental capture (memo capture-tick)
    _spec(
        "MEMO_CAPTURE_INTERVAL_S",
        "int",
        600,
        "capture",
        "Minimum seconds between incremental captures for a non-forced `memo "
        "capture-tick` (the throttle lives in `incremental_tick_due`). Mines NEW "
        "turns since a per-session watermark into durable memories so a "
        "long/crashed session's insight reaches .md mid-session instead of only "
        "at Stop. A cheap no-op when not due; 0 disables the throttle. SCOPE: "
        "this flag governs `capture-tick` only, and `--force` bypasses it — the "
        "PreCompact hook calls `capture-tick --force`, and the UserPromptSubmit "
        "incremental path (`memo session idle-maintenance --mode capture`) has "
        "its own quiet-window throttle (MEMO_SESSION_IDLE_CAPTURE_SECS). Neither "
        "live hook consults this value; it applies to a manual `capture-tick`.",
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
    _spec(
        "WHATSAPP_BOT_JID",
        "str",
        "",
        "whatsapp",
        "WhatsApp bot JID to filter from ingest (e.g., 54911xxx@s.whatsapp.net).",
    ),
    _spec(
        "WA_LISTENER_NOTES_CHAT_JID",
        "str",
        "",
        "whatsapp",
        "WhatsApp chat JID for the notes destination chat.",
    ),
    # tantivy kill-switch
    _spec(
        "MEMO_TANTIVY_ENABLED",
        "bool",
        True,
        "store",
        "Enable the Tantivy FTS index (dual-write on upsert/delete + search + "
        "startup rebuild). Set =0 to force FTS5-only — operational kill-switch, "
        "wins over MEMO_FTS_BACKEND.",
    ),
    _spec(
        "MEMO_DEDUP_EXACT",
        "bool",
        True,
        "store",
        "Compatibility setting for legacy session-pattern normalized_hash generation. "
        "Exact namespaced corroboration is a storage correctness invariant and remains "
        "enabled regardless of this value.",
    ),
    _spec(
        "MEMO_SOFT_DELETE",
        "bool",
        True,
        "store",
        "Enable soft-delete (deleted_at column) instead of hard row removal. "
        "Set =0 to hard-delete rows immediately. Requires DB migration to v3.",
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
    # dream — recall observability: label harvest + nightly retrieval eval
    _spec(
        "MEMO_DREAM_EVAL_ENABLED",
        "bool",
        True,
        "dream",
        "Enable the nightly observability passes in `memo dream run`: harvest eval "
        "labels from grounding.log into state_dir/eval/harvested_labels.json, then "
        "run a retrieval-only eval (vec mode, no reranker) over harvested + curated "
        "labels, appending the prec@K/noise@K trend to state_dir/eval/history.jsonl. "
        "Read-only + cheap; default on. Set =0 to opt out.",
        opt_out=True,
    ),
    _spec(
        "MEMO_DREAM_EVAL_MAX_LABELS",
        "int",
        200,
        "dream",
        "Max labels per nightly eval pass (curated always included; the most "
        "recently harvested labels fill the remaining room).",
        min_val=1,
    ),
    # graph co-recall
    _spec(
        "MEMO_GRAPH_CO_RECALL",
        "bool",
        False,
        "graph",
        "Record co-recall edges in graph.db whenever a search returns 2+ results, "
        "AND boost candidates frequently co-recalled with the top hit so relationally-"
        "associated memories surface together. Pairs of co-recalled memory IDs are "
        "stored in the `co_recall` table with incrementing counts. Off by default to "
        "avoid graph DB write overhead on the recall-hook hot path.",
    ),
    _spec(
        "MEMO_CO_RECALL_BOOST_WEIGHT",
        "float",
        0.1,
        "graph",
        "Max score bump added to a candidate co-recalled with the top hit, scaled by "
        "its share of the strongest co-recall edge in the result set. Only applied when "
        "MEMO_GRAPH_CO_RECALL is on.",
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
    # dream v2 — noise-quantile min_sim floor calibration, gated + reversible.
    # Runs before the min_sim tuner (below) so a co-enabled tuner searches
    # upward from the measured floor.
    _spec(
        "MEMO_FLOOR_CALIBRATION",
        "bool",
        False,
        "recall",
        "At nightly Dream, estimate the noise-quantile similarity floor from "
        "word-shuffled probes and RAISE MEMO_RECALL_MIN_SIM toward it via the tuned "
        "overlay — gated by the curated eval set, only ever raising toward the "
        "measured floor. Runs before the min_sim tuner (which then searches upward "
        "from the floor). Default off.",
    ),
    # dream v2 — self-improving recall tuner (min_sim), gated + reversible
    _spec(
        "MEMO_DREAM_TUNE_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the nightly recall self-tuner inside `memo dream run`. OFF by default; "
        "tunes MEMO_RECALL_MIN_SIM against ground-truth-by-use labels, gated by the curated "
        "regression set, reverted when a later night regresses.",
    ),
    _spec(
        "MEMO_DREAM_TUNE_MIN_COHORT",
        "int",
        20,
        "misc",
        "Phase-1 proof loop: minimum grounded-scored recalls under a newly-applied "
        "params version before its online impact is judged. Below this the tuner waits "
        "(one change per proof cycle).",
    ),
    _spec(
        "MEMO_DREAM_TUNE_ONLINE_EPS",
        "float",
        0.02,
        "misc",
        "Phase-1 proof loop: deadband on the realized online grounded-fraction delta. A "
        "change whose next-cohort fraction drops by more than this vs pre-apply is reverted.",
    ),
    _spec(
        "MEMO_DREAM_TUNE_GRADUATION_K",
        "int",
        5,
        "misc",
        "Phase-2 graduation bar: consecutive confirmed proof-loop verdicts (non-negative "
        "realized online delta) before the min_sim tuner is considered ready to enable by "
        "default. Reporting only — memo never flips MEMO_DREAM_TUNE_ENABLED automatically.",
    ),
    _spec(
        "MEMO_DREAM_TUNE_BOOST_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the nightly ONLINE-ONLY project-boost explorer inside `memo dream run`. OFF by "
        "default (separate opt-in from MEMO_DREAM_TUNE_ENABLED). Nudges MEMO_RECALL_PROJECT_BOOST "
        "and lets the online proof loop confirm/revert it against real grounding — boosts are not "
        "offline-measurable, so there is no offline gate for this knob.",
    ),
    _spec(
        "MEMO_DREAM_TUNE_BOOST_STEP",
        "float",
        0.05,
        "misc",
        "Nudge size for the online project-boost explorer (MEMO_DREAM_TUNE_BOOST_ENABLED).",
    ),
    _spec(
        "MEMO_DREAM_HYDE_TUNE_ENABLED",
        "bool",
        False,
        "misc",
        "Nightly A/B of the default-off, never-measured MEMO_HYDE_ENABLED: "
        "measures hybrid retrieval with vs without HyDE on the mined+curated "
        "labels (costs one MLX chat call per prompt — prompt count capped). "
        "Applies MEMO_HYDE_ENABLED=1 via the tuned overlay only when it wins "
        "precision without raising noise, passes the curated gate, stays within "
        "the latency headroom, AND the live recall mode is not hybrid (HyDE in "
        "the hook path would blow the 5s budget). Reversible via `memo dream "
        "tune --rollback`. Default off.",
    ),
    _spec(
        "MEMO_DREAM_RETRIEVAL_TUNE_ENABLED",
        "bool",
        False,
        "misc",
        "Deprecated inert compatibility switch. The graph-injection/expansion tuner was "
        "removed; MEMO_DREAM_TUNE_ENABLED now evaluates only the curated graph signal.",
    ),
    _spec(
        "MEMO_DREAM_RETRIEVAL_LATENCY_BUDGET_MS",
        "float",
        2500.0,
        "misc",
        "Deprecated inert compatibility value retained for old configuration files.",
        min_val=0.0,
    ),
    _spec(
        "MEMO_DREAM_FLAG_GRADUATION_ENABLED",
        "bool",
        True,
        "misc",
        "Enable the nightly dark-feature (flag) graduation pass inside `memo dream run`. "
        "Default ON (opt out with =0). Every default-off *_ENABLED flag has a declared gate in "
        "dream_flags.GATES; recall-gated flags get an ON/OFF A/B against the mined+curated "
        "labels and graduate to ON via the tuned overlay after "
        "MEMO_FLAG_GRADUATION_WIN_NIGHTS consecutive wins (latency + curated gates). "
        "Reversible: a regression vs the graduation baseline reverts the flag. Flags that "
        "never graduate within MEMO_FLAG_GRADUATION_DEADLINE_DAYS surface as cull "
        "candidates in `memo dream graduate-flags --status` (deletion stays human).",
    ),
    _spec(
        "MEMO_FLAG_GRADUATION_WIN_NIGHTS",
        "int",
        3,
        "misc",
        "Consecutive winning A/B measurements a recall-gated dark flag needs before the "
        "flag-graduation pass flips it ON via the tuned overlay.",
        min_val=1,
    ),
    _spec(
        "MEMO_FLAG_GRADUATION_MAX_PER_NIGHT",
        "int",
        3,
        "misc",
        "Max dark flags A/B-measured per flag-graduation night (cost ceiling; "
        "least-recently-measured first).",
        min_val=1,
    ),
    _spec(
        "MEMO_FLAG_GRADUATION_DEADLINE_DAYS",
        "int",
        45,
        "misc",
        "Days a dark flag may stay tracked without graduating before it is reported as a "
        "cull candidate (flip it with a real gate or delete the code path).",
        min_val=1,
    ),
    _spec(
        "MEMO_FLAG_GRADUATION_RETRY_DAYS",
        "int",
        14,
        "misc",
        "Cooldown after a flag-graduation revert before the same flag is A/B-measured again.",
        min_val=0,
    ),
    _spec(
        "MEMO_FLAG_GRADUATION_MAX_PROMPTS",
        "int",
        80,
        "misc",
        "Label-corpus cap per flag-graduation measurement (each candidate costs two "
        "retrieval evals over this many prompts).",
        min_val=1,
    ),
    _spec("MEMO_DREAM_TUNE_K", "int", 5, "misc", "K for precision@K/noise@K during dream tuning."),
    _spec(
        "MEMO_DREAM_TUNE_MAX_EVALS",
        "int",
        20,
        "misc",
        "Max eval iterations per dream tuning pass (cost ceiling).",
    ),
    _spec(
        "MEMO_DREAM_MINE_MIN_USED_SCORE",
        "float",
        0.5,
        "misc",
        "Minimum grounding used_score for a turn to be mined as a tuning label.",
    ),
    _spec(
        "MEMO_DREAM_MINE_LIMIT",
        "int",
        200,
        "misc",
        "Max labels mined from grounding.log per dream tuning pass.",
    ),
    # dream — refresh dynamic mandate rule blocks in opted-in repos
    _spec(
        "MEMO_DYNAMIC_MANDATE_SYNC_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the nightly pass that refreshes `memo mandate --dynamic` rule "
        "blocks in every opted-in repo (recorded when you run --dynamic --write). "
        "OFF by default; superseded rules retire and new ones appear on their own. "
        "No-op in a repo whose block a human deleted.",
    ),
    # dream — code-drift pass: re-verify code_refs against the codegraph index
    _spec(
        "MEMO_DREAM_CODE_DRIFT_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the nightly code-drift pass in `memo dream run`. OFF by default; "
        "re-verifies memories carrying code_refs against the live codegraph index "
        "and proposes fully-drifted ones as outdated (reversible archive, never a "
        "hard delete). Aborts when the index is missing or >24h stale.",
    ),
    # dream — code-drift auto-repair: re-point dead refs with a unique candidate
    _spec(
        "MEMO_DREAM_CODE_REPAIR_ENABLED",
        "bool",
        False,
        "misc",
        "Enable auto-repair inside the nightly code-drift pass. OFF by default; "
        "a dead code_ref with EXACTLY one rename/move candidate in the "
        "codegraph index is re-pointed in place (the old ref is preserved in "
        "extra.code_refs_history) and the memory is not archived that night. "
        "0 or >1 candidates -> archive as today.",
    ),
    # dream v2 — anticipatory pass (Phase 3): surface unmet gaps + prewarm
    _spec(
        "MEMO_DREAM_ANTICIPATE_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the nightly anticipatory pass in `memo dream run`. OFF by default; "
        "surfaces recurring knowledge gaps (detect_gaps) + hot queries into the receipt "
        "and pre-warms their embeddings. Never fabricates answers.",
    ),
    _spec(
        "MEMO_DREAM_ANTICIPATE_TOP_GAPS",
        "int",
        5,
        "misc",
        "Max recurring gaps surfaced per anticipatory pass.",
    ),
    # Phase 3 — ASK: promote AT MOST ONE recurring unmet gap per session into an
    # explicit question in the SessionStart briefing. Reads the anticipate
    # receipt (never fabricates). Deduped per session; report-only.
    _spec(
        "MEMO_ASK_GAPS_ENABLED",
        "bool",
        False,
        "misc",
        "Surface AT MOST ONE high-value recurring gap (from the nightly "
        "anticipate receipt's detect_gaps output) as an explicit question in the "
        "SessionStart briefing. Never fabricates — only re-asks what memo already "
        "failed to answer from real usage. Deduped per session. Shadow-counts "
        "what it WOULD ask even when OFF (memo ask-gaps shadow). Default OFF; not "
        "auto-graduatable.",
    ),
    # dream v2 — graph-community synthesis (spec 3): abstract knowledge clusters
    _spec(
        "MEMO_DREAM_COMMUNITIES_ENABLED",
        "bool",
        False,
        "graph",
        "Enable the `memo dream communities` pass: detect entity-graph communities "
        "and abstract each into one durable synthesis memory (synthesis_kind=community, "
        "provenance = the community's entities). OFF by default; deduped by provenance.",
    ),
    _spec(
        "MEMO_DREAM_COMMUNITIES_MIN_SIZE",
        "int",
        4,
        "graph",
        "Minimum entities in a graph community before it is synthesized.",
        min_val=2,
    ),
    # dream v2 — MinHash-blocked LLM entity canonicalization (workstream K1)
    _spec(
        "MEMO_DREAM_ENTITY_CANON_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the `memo dream entity-canon` pass: MinHash+LSH blocking over "
        "entity names proposes near-duplicate pairs, the helper LLM confirms "
        "each candidate, confirmed pairs merge via entity_aliases. Receipt "
        "reports pairs_naive vs pairs_blocked vs llm_calls (measured saving). "
        "OFF by default.",
    ),
    _spec(
        "MEMO_DREAM_ENTITY_CANON_MAX_PAIRS",
        "int",
        30,
        "misc",
        "Cap on LLM-confirmed candidate pairs per entity-canon run (bounds nightly MLX cost).",
        min_val=1,
        max_val=500,
    ),
    # dream v2 — per-folder vault abstracts (workstream K4)
    _spec(
        "MEMO_DREAM_FOLDER_ABSTRACTS_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the `memo dream folder-abstracts` pass: one synthesis memory "
        "per vault folder (synthesis_kind=folder_abstract) answering 'what is "
        "this section about'. Updated in place on membership change; skipped "
        "when unchanged. OFF by default.",
    ),
    _spec(
        "MEMO_DREAM_FOLDER_ABSTRACTS_MIN_MEMBERS",
        "int",
        5,
        "misc",
        "Minimum reference documents in a vault folder before it gets an abstract.",
        min_val=2,
    ),
    _spec(
        "MEMO_DREAM_FOLDER_ABSTRACTS_MAX",
        "int",
        5,
        "misc",
        "Max folders abstracted per nightly run (bounds MLX cost).",
        min_val=1,
        max_val=100,
    ),
    # dream v2 — Phase 2 distillation (upward re-abstraction of mature clusters)
    _spec(
        "MEMO_DREAM_DISTILL_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the `memo dream distill` pass: group MATURE, high-confidence, "
        "corroborated durable clusters and abstract each into one higher-altitude "
        "synthesis memory (synthesis_kind=distillation, provenance=source ids, "
        "confidence=corroboration-weighted). Additive + linking — never supersedes "
        "or deletes sources (fully reversible). OFF by default; deduped by provenance.",
    ),
    _spec(
        "MEMO_DREAM_DISTILL_MIN_CLUSTER",
        "int",
        3,
        "misc",
        "Minimum durable memories in a cluster before it is distilled.",
        min_val=2,
    ),
    _spec(
        "MEMO_DREAM_DISTILL_MIN_SUPPORT",
        "int",
        2,
        "misc",
        "Minimum MEAN corroboration (memory_health.support_count) across a "
        "cluster's members before it is distilled (maturity gate).",
        min_val=1,
    ),
    _spec(
        "MEMO_DREAM_DISTILL_MIN_AGE_DAYS",
        "int",
        14,
        "misc",
        "A cluster's members must be at least this many days old (by meta.created) "
        "before the cluster is distilled — distillation is upward re-abstraction of "
        "SETTLED knowledge, not fresh captures.",
        min_val=0,
    ),
    _spec(
        "MEMO_DREAM_DISTILL_MAX",
        "int",
        5,
        "misc",
        "Max clusters distilled per nightly run (bounds MLX cost).",
        min_val=1,
        max_val=100,
    ),
    _spec(
        "MEMO_DREAM_DISTILL_THRESHOLD",
        "float",
        0.78,
        "misc",
        "Cosine similarity floor for grouping durable memories into a distillable cluster.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_DREAM_DISTILL_MIN_CONFIDENCE",
        "float",
        0.5,
        "misc",
        "Minimum mean memory_health.confidence across a cluster's members before it is distilled.",
        min_val=0.0,
        max_val=1.0,
    ),
    # dream v2 — bridge / multi-hop link synthesis (spec 3, phase 3)
    _spec(
        "MEMO_DREAM_BRIDGES_ENABLED",
        "bool",
        False,
        "graph",
        "Enable the `memo dream bridges` pass: detect articulation entities that "
        "solely connect two graph regions and abstract each into one durable "
        "synthesis memory (synthesis_kind=bridge, provenance = bridge + side reps). "
        "OFF by default; deduped by provenance; runs alongside community synthesis.",
    ),
    # dream v2 — episodic→semantic consolidation (Phase 2): cross-session themes
    _spec(
        "MEMO_DREAM_CONSOLIDATE_EPISODES_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the nightly cross-session consolidation pass in `memo dream run`. "
        "OFF by default; abstracts recurring per-project work across >=N sessions into "
        "one durable synthesis memory (provenance = session ids). No episodic decay.",
    ),
    _spec(
        "MEMO_DREAM_CONSOLIDATE_MIN_SESSIONS",
        "int",
        2,
        "misc",
        "Min distinct sessions on a project before it is consolidated cross-session.",
        min_val=2,
    ),
    # dream v2 — bi-temporal validity extraction (LLM off the recall hot path)
    _spec(
        "MEMO_DREAM_VALIDITY_EXTRACT_ENABLED",
        "bool",
        True,
        "misc",
        "Enable the nightly validity-extract pass in `memo dream run`. Default ON "
        "(opt out with =0). For recent durable facts/decisions whose TEXT explicitly states "
        "a validity window ('contract runs through Q3 2026', 'valid until Dec', "
        "'as of March we use X'), an MLX LLM extracts a structured valid_at/"
        "invalid_at and writes it via store.update_validity + a frontmatter "
        "mirror. Never hallucinates — a boundary is set only when the note text "
        "explicitly supports it (the extracted year must appear verbatim in the "
        "text); records with no explicit window are skipped. Per-record results "
        "and errors land in the dream receipt.",
    ),
    _spec(
        "MEMO_DREAM_VALIDITY_EXTRACT_LIMIT",
        "int",
        50,
        "misc",
        "Max recent durable facts/decisions scanned per validity-extract pass "
        "(bounds nightly MLX cost; least-recently-updated skipped first).",
        min_val=1,
    ),
    # Tier-1 #1 (+ Tier-2 #24) — profile distillation + directive graduation
    _spec(
        "MEMO_DREAM_PROFILE_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the nightly profile-distillation pass in `memo dream run`. "
        "OFF by default; distills preference/feedback/decision/synthesis "
        "memories into char-budgeted, rewritten-in-place profile.md files "
        "(global + per-project) under memory_dir/_profile/ with memory-id "
        "provenance, plus a Standing-rules block graduated from grounding.log.",
    ),
    _spec(
        "MEMO_DREAM_PROFILE_CHAR_BUDGET",
        "int",
        4000,
        "misc",
        "Character budget per profile document (frontmatter exempt).",
        min_val=200,
    ),
    _spec(
        "MEMO_DREAM_PROFILE_MAX_PROJECTS",
        "int",
        5,
        "misc",
        "Max per-project profile documents rewritten per pass.",
        min_val=0,
    ),
    _spec(
        "MEMO_DREAM_PROFILE_DIRECTIVE_K",
        "int",
        3,
        "misc",
        "Standing-rule graduation: min DISTINCT sessions in grounding.log "
        "that must cite a memory before it renders as a standing rule.",
        min_val=2,
    ),
    _spec(
        "MEMO_DREAM_PROFILE_DIRECTIVE_MIN_USED",
        "float",
        0.5,
        "misc",
        "Standing-rule graduation: min grounding used_score for a citation to count.",
        min_val=0.0,
        max_val=1.0,
    ),
    # dream v2 — scope: project→global promotion (retag memories proven general)
    _spec(
        "MEMO_DREAM_RETAG_GLOBAL_ENABLED",
        "bool",
        False,
        "misc",
        "Enable the nightly project→global retag pass in `memo dream run`. OFF by "
        "default. Strips the `project:` tag from memories grounded (used_score >= "
        "0.6 in grounding.log) from sessions in >= MEMO_DREAM_RETAG_MIN_PROJECTS "
        "OTHER projects, via the pure-retag update path (no re-embed; reversible "
        "with `memo version rollback`). To run under the nightly LaunchAgent, add "
        "this var to its EnvironmentVariables (launchd does not inherit the shell).",
    ),
    _spec(
        "MEMO_DREAM_RETAG_MIN_PROJECTS",
        "int",
        2,
        "misc",
        "Min distinct OTHER projects whose sessions grounded a memory before the "
        "retag pass promotes it to global.",
        min_val=1,
    ),
    # dream v2 — quarantine graduation: promote _uncertain captures that earned trust
    _spec(
        "MEMO_DREAM_GRADUATION_ENABLED",
        "bool",
        True,
        "misc",
        "Enable the nightly quarantine-graduation pass in `memo dream run`: "
        "'_uncertain' auto-captures proven by grounding (used in an answer) or "
        "by corroboration (support_count) get the tag removed and re-enter "
        "auto-recall. Reversible via memo version rollback. Default ON (opt out with =0).",
    ),
    _spec(
        "MEMO_DREAM_GRADUATION_MIN_SUPPORT",
        "int",
        2,
        "misc",
        "Corroboration floor for graduation: memory_health.support_count required "
        "to promote an '_uncertain' capture without grounding evidence.",
        min_val=1,
    ),
    _spec(
        "MEMO_CROSSREF_INDEX",
        "bool",
        True,
        "links",
        "Index [[wikilinks]] and typed '- relation [[target]]' edges into the "
        "crossref backlinks table at save/update/delete/reindex, enabling "
        "cascade-aware supersede/delete warnings. Default ON (opt out with =0).",
    ),
    # chronicle dream pass (nightly human-engineering diary write)
    _spec(
        "MEMO_DREAM_CHRONICLE_ENABLED",
        "bool",
        False,
        "misc",
        "Nightly chronicle dream pass: write a human engineering diary for the "
        "day under <memory_dir>/_chronicle/, with per-id provenance. Default off.",
    ),
    _spec(
        "MEMO_CHRONICLE_WEEKLY",
        "bool",
        False,
        "misc",
        "Also regenerate the ISO-week rollup file (week-YYYY-Www.md) after each "
        "nightly chronicle write. Default off.",
    ),
    # public benchmark harness (`memo eval bench`) — offline batch, never the hook
    _spec(
        "MEMO_BENCH_JUDGE",
        "str",
        "mlx",
        "bench",
        "QA judge for `memo eval bench`: 'mlx' (local MLXChat, default) or "
        "'api' (OpenAI-compatible endpoint — local small judges are weak graders).",
    ),
    _spec(
        "MEMO_BENCH_JUDGE_MODEL",
        "str",
        "",
        "bench",
        "Judge model id. Empty = the configured llm_model for the mlx judge; "
        "required when MEMO_BENCH_JUDGE=api.",
    ),
    _spec(
        "MEMO_BENCH_JUDGE_URL",
        "str",
        "",
        "bench",
        "Base URL of the OpenAI-compatible endpoint when MEMO_BENCH_JUDGE=api "
        "(e.g. https://api.openai.com/v1).",
    ),
    _spec(
        "MEMO_BENCH_JUDGE_API_KEY_ENV",
        "str",
        "OPENAI_API_KEY",
        "bench",
        "NAME of the env var that holds the API key for the api judge. The key "
        "itself is read from that variable — it never lives in a MEMO_* flag.",
    ),
    # verification state tracking
    _spec(
        "MEMO_VERIFICATION_STATE_TRACKING",
        "bool",
        True,
        "misc",
        "Master switch for the verification-state lifecycle (UNVERIFIED/VERIFIED/"
        "STALE). When ON: (1) `memo maintain` marks VERIFIED memories STALE when "
        "their explicit `review_after` date is due, and "
        "(2) live recall multiplies each hit's score by its state decay factor "
        "(VERIFIED≈1.0, STALE 0.7, UNVERIFIED 0.8) so fresh facts outrank stale "
        "ones. No-op for an all-UNVERIFIED corpus (uniform penalty). Default ON (opt out with =0).",
    ),
    # secret storage (encrypted credentials)
    _spec(
        "MEMO_SECRET_STORAGE_ENABLED",
        "bool",
        False,
        "secret",
        "Explicitly opt in to encrypted local secret storage (passwords, tokens, SSH keys).",
    ),
    # proactive engine (statusline badge, urgent push, `memo digest`)
    _spec(
        "MEMO_PROACTIVE_ENABLED",
        "bool",
        False,
        "misc",
        "Master switch for the proactive engine (statusline badge, urgent push, "
        "`memo digest`). Default off — dark flag, graduates via dream_flags.",
    ),
    _spec(
        "MEMO_PROACTIVE_PUSH_COOLDOWN_H",
        "int",
        6,
        "misc",
        "Minimum hours between urgent pushes.",
        min_val=0,
    ),
    _spec(
        "MEMO_PROACTIVE_DAILY_CAP",
        "int",
        3,
        "misc",
        "Hard cap on proactive pushes per day.",
        min_val=0,
    ),
    _spec(
        "MEMO_PROACTIVE_MULT_FLOOR",
        "float",
        0.2,
        "misc",
        "Floor for the adaptive per-kind multiplier (reliability can never be fully muted).",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_PROACTIVE_URGENT_MIN",
        "float",
        0.7,
        "misc",
        "Minimum score for a reliability nudge to qualify for an urgent push.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_PROACTIVE_DIGEST_TOP",
        "int",
        7,
        "misc",
        "Max items shown in `memo digest`.",
        min_val=1,
    ),
)
