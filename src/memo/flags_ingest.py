from __future__ import annotations

from memo.flags_base import FlagSpec, _spec

SPECS: tuple[FlagSpec, ...] = (
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
        "MEMO_SAVE_EXTRACT",
        "bool",
        False,
        "ingest",
        "Default for write-time fact extraction (mem0 ADD-model): when on, "
        "`memo save` / `memo_save` decompose the content into atomic facts via "
        "the helper LLM and save each as its own memory instead of one opaque "
        "blob. Off by default (adds ~1-3s LLM latency); the per-call "
        "`--extract` flag / `extract=` arg always wins. Falls back to a verbatim "
        "save when nothing extractable is found.",
    ),
    _spec(
        "MEMO_CHUNK_INGEST",
        "bool",
        False,
        "ingest",
        "When enabled, the `memo reindex` pipeline splits long curated memories "
        "into heading-aware chunks before embedding, so long multi-section notes "
        "get section-level retrieval granularity. Each chunk is stored as type='reference' "
        "with extra.parent_id pointing back to the parent memory. Default off preserves "
        "the whole-note-embed behaviour. `memo ingest` has its own --chunk/--no-chunk flag "
        "and is unaffected by this env var.",
    ),
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
        "MEMO_REPO_GIT_TIMEOUT_S", "float", None, "repo", "Timeout (s) for git operations on clone."
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
    _spec(
        "MEMO_GPU_XPROC_LOCK",
        "bool",
        True,
        "embedder",
        "Serialize MLX GPU work across memo processes via a file lock "
        "(prevents the cross-process Metal SIGABRT). Set 0 to disable.",
        opt_out=True,
    ),
    _spec(
        "MEMO_GPU_LOCK_PATH",
        "str",
        "",
        "embedder",
        "Override the cross-process GPU lock file path. Default is the "
        "user-global ~/.cache/memo/memo-mlx-gpu.lock; every memo process "
        "must agree on this path to coordinate. Mainly for test isolation.",
    ),
)
