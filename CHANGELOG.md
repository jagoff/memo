# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases before `2.0.0` are archived in [docs/CHANGELOG-archive.md](docs/CHANGELOG-archive.md).

## [Unreleased]

## [2.6.4] - 2026-06-30

### Changed
- Documentation sweep for Linux/Ubuntu: the PyPI/plugin/MCP-registry package descriptions, the README (hero, hybrid-search line, diagram alt text), and `docs/reference.md` + `docs/install-new-mac.md` now state that memo runs on Apple Silicon (MLX) **or** Linux/Ubuntu (CPU `sentence-transformers`), instead of implying macOS-only. The "when not to pick memo" note no longer lists "not on Apple Silicon" as a disqualifier (memo runs standalone on Linux; only the reranker + LLM features are MLX-only).

## [2.6.3] - 2026-06-30

### Fixed
- The MLX-only cross-encoder reranker is now correctly forced **off** on non-Apple-Silicon hosts. The default `MODEL_PROFILES` set `reranker_enabled: True`, which overrode the platform-aware field default — so on Linux every hybrid search tried to load the MLX reranker, failed (`No module named 'mlx_lm'`), and fell back. `Config.from_env()` now gates it: an explicit `MEMO_RERANKER_ENABLED` wins, otherwise it is off when MLX is unavailable. Found by the new `linux-cpu-smoke` CI job running a real Ubuntu `[cpu]` install.

### Added
- `linux-cpu-smoke` CI job (ubuntu-latest): real `pip install ".[cpu]"`, sqlite-vec/FTS5 load, the actual Qwen3-Embedding-0.6B model, `memo doctor`, a save→search roundtrip, and `memo-mcp` build — the end-to-end Linux validation the (stubbed) unit suite can't give.

## [2.6.2] - 2026-06-30

### Fixed
- `memo install-mcp claude-desktop` writes to the per-OS Claude Desktop config path (`~/.config/Claude/...` on Linux, `%APPDATA%` on Windows) instead of the hardcoded macOS path.
- The MCP-config scan (`memo doctor` / mcp-config) now also checks the Linux Windsurf path (`~/.config/Windsurf/User/mcp_config.json`).

### Added
- Linux systemd **user** units in `systemd/` (`memo-dream.service` + `memo-dream.timer` for nightly maintenance, `memo-watch.service` for auto-reindex) — the counterpart of memo's macOS launchd agents. See `systemd/README.md` and `docs/ubuntu.md`.

## [2.6.1] - 2026-06-30

### Fixed
- `memo doctor` / `memo doctor --json` no longer false-fail (exit 1) on a healthy Linux/Ubuntu `[cpu]` install: the import probe is now backend-aware — it checks `sentence-transformers` on the CPU backend instead of MLX, and the model-cache report lists `st_embedder_model`.
- `memo install-watcher` / `uninstall-watcher` give a clean "launchd is macOS-only" message on Linux instead of an uncaught `FileNotFoundError` (no `launchctl`).
- `memo search --rerank` warns and skips on non-Apple-Silicon instead of trying to load the MLX-only cross-encoder.
- `RepoCorpus` (and the opt-in ingest daemon) routes the embedder through `make_embedder`, so it uses the CPU backend on Linux instead of hard-constructing `MLXEmbedder`. SessionStart prewarm likewise warms the active backend.
- Obsidian vault auto-detection reads the per-OS registry path (`~/.config/obsidian/obsidian.json` on Linux, `%APPDATA%` on Windows).

## [2.6.0] - 2026-06-30

### Added
- Linux/Ubuntu (and Intel mac) support via a CPU `sentence-transformers` embedder backend (`STEmbedder`), auto-selected off Apple Silicon. Semantic search, recall, and save work without MLX as a **standalone** corpus. New `embedder_select.make_embedder` is the single MLX-vs-CPU decision point, routed through the `Memory` facade, the `embedder_client` daemon fallback, and `memo ingest`.
- `MEMO_EMBEDDER_BACKEND` (`auto`/`mlx`/`st`) and `MEMO_ST_EMBEDDER_MODEL` (default `Qwen/Qwen3-Embedding-0.6B`, 1024-dim — same family/dims as the MLX quant, so the vec0 schema and asymmetric query prefix are unchanged).
- `pip install "mlx-memo[cpu]"` extra, `scripts/install-ubuntu.sh` (uv/pipx), and `docs/ubuntu.md`.

### Changed
- `reranker_enabled` now defaults ON only on Apple Silicon (the cross-encoder is MLX-only); on other hosts hybrid search returns fusion-ranked results without the rerank pass.
- LLM features (`ask`/`synthesize`/`dream`) raise a clear `MemoError` off Apple Silicon instead of an opaque import failure; search/recall/save are unaffected.

## [2.5.0] - 2026-06-30

### Added
- Graph-native program: associative recall nudge (spreading activation over the merged entity + codegraph graph with IDF hub-damping), entity-centric "Knowledge map" briefing section, nightly community-synthesis dream pass (off by default, `MEMO_DREAM_COMMUNITIES_ENABLED`), and `memo graph explore` / `memo_explore` knowledge exploration.
- codegraph migration: memo navigation merges colbymchenry/codegraph's symbol graph as the primary layer (`MEMO_GRAPH_USE_CODEGRAPH`).

### Fixed
- Associative recall nudge now renders on the warm-daemon (primary) recall path, not only the subprocess fallback (it was dormant in normal operation).
- `compute_centrality` rebuilt the graph adjacency O(N²) times (hung on real corpora); now builds it once.
- Storage: `get_by_path` / `get_by_path_ci` and prune/eviction candidate queries no longer return soft-deleted rows (path-collision false positives, UNIQUE constraint crashes, inflated eviction pools).
- HyDE search reuses the shared MLX chat instead of cold-loading a second copy of the LLM.
- `llm.chat_stream` no longer holds the GPU lock across the entire streamed response.
- `AmbiguousIdError` guards on `memo_related` (and the HTTP/CLI paths); restore tolerates a corrupt manifest; numerous CLI/MCP/flags/dream correctness fixes from a whole-project adversarial audit.

### Changed
- English-only user-facing strings (associative nudge); removed dead code (`find_node_fuzzy`, `auto_update_on_commit`, `get_session_metadata`); gitignore scratch output.

## [2.4.3] - 2026-06-29

### Added
- **Dream convergence guard**: `memo dream run` stamps a corpus fingerprint and, on a re-run where nothing changed (signal-gather found nothing new and the corpus is unchanged), skips the expensive contradict/synthesize/consolidate passes — a re-run on an idle corpus is now near-instant instead of redoing identical LLM work. `--force` runs every pass regardless.
- **Dream single-owner lock**: a second `dream run` (manual, or the `com.memo.dream` LaunchAgent firing while one is in flight) now skips instead of racing on the shared sidecar DBs and clobbering the receipt.

### Changed
- Dream signal-gather now uses the exact last-run timestamp for its transcript lookback instead of a day-rounded window that re-mined ~1–2 days every run.

## [2.4.2] - 2026-06-29

### Changed
- Removed the Spanish command aliases `memo historia` / `memo mapa` (use `memo record-history` / `memo map`); the CLI surface is now English-only.

### Fixed
- Silenced HuggingFace hub progress bars globally (set `HF_HUB_DISABLE_PROGRESS_BARS` at import). Model loads, prewarm, daemon startup and self-update no longer leak repeated "Fetching N files / Download complete 0.00B" noise for already-cached models.

## [2.4.1] - 2026-06-29

### Fixed
- **Split graph DB healing**: an install that ran an interim build (the table rename without a migration) ended up with both `entity_memoria` (legacy data) and an empty `entity_memory`, splitting the knowledge graph and losing historical entity links to recall. The migration now folds the legacy rows into `entity_memory` (deduped) and drops the legacy table when both are present, not only when the new table is absent.

## [2.4.0] - 2026-06-29

### Changed
- **English-only codebase** for public release: Spanish identifiers renamed to English across the code (DB table `entity_memoria`→`entity_memory`, column `memoria_id`→`memory_id`, output/API keys), remaining UI strings translated, and the `engram` name removed (module → `session_patterns`). On-disk corpus dir default `memorias/`→`memories/`.

### Fixed
- **Existing-database migrations**: pre-rename SQLite DBs are auto-migrated in place (`pairs.memoria_id_a/b`, the `entity_memoria` table + `memoria_id` column, `versions.memoria_id`) so upgrades no longer crash with `no such column` (which surfaced as a warn in `memo dream`) or silently orphan graph data.
- **Soft directory migration**: new installs use `memories/`; existing `memorias/` installs keep working — config, sync clone/bootstrap, backup/restore and import all read both names.
- 24 bugs from an exhaustive line-by-line review: `memo backup`/`memo health` crashing on their non-JSON paths, consolidation resurrecting archived duplicates, the `mem_review` SQL query, the tantivy rebuild for >5000-memory corpora, the synapse `conflicts` CLI args, and more.
- Dropped a personal locale default ("Spanish rioplatense") from the ask prompt; answers follow the question's language generically.

## [2.3.12] - 2026-06-29

### Fixed
- Dream cross-session consolidation (`memo dream consolidate-episodes`) now saves its synthesized memories — a keyword-only `save()` was being called positionally, raising `TypeError`. Regression-tested.

## [2.3.11] - 2026-06-29

### Added
- **Dream v2 — self-improving recall tuner** (off by default): nightly `MEMO_RECALL_MIN_SIM` line-search from ground-truth-by-use labels, gated by the regression set, auto-applied and auto-reverted on a later regression (`memo dream tune`).
- **Dream v2 — episodic→semantic consolidation** (off by default): abstracts recurring cross-session work into one durable synthesis memory (`memo dream consolidate-episodes`).
- **Dream v2 — anticipatory pass** (off by default): surfaces recurring knowledge gaps and hot queries into the briefing and pre-warms their embeddings; never fabricates (`memo dream anticipate`).

## [2.3.10] - 2026-06-29

### Changed
- `memo doctor` auto-repairs a stale MCP config path (a dead pipx/uv launch path is repointed to the stable `~/.local/bin` shim, with a `.bak` backup). `--check` and `--json` stay read-only.

## [2.3.9] - 2026-06-29

### Changed
- The injected recall block now carries a relevance gate and stronger prompt-injection framing — recalled memory is treated as data, never as instructions.

## [2.3.8] - 2026-06-28

### Added
- Per-project memory storage (default on): new memories are filed under `memory_dir/<project>/`, while search stays global. `memo migrate --bucket-by-project` re-buckets an existing install.
- 3-tier recall relevance: current project > global / cross-cutting > other projects.

## [2.3.7] - 2026-06-28

### Fixed
- `ask` now surfaces both sides of a contradiction instead of silently picking one, and respects `k`.
- Spanish accented queries keep their relevance boost (diacritics folded).
- The embed socket handles full batches on 4B/8B models without truncating.

## [2.3.6] - 2026-06-28

### Fixed
- The statusline no longer duplicates the `[MEMO <version>]` badge when wrapping an existing statusline.

## [2.3.5] - 2026-06-28

### Fixed
- Broad CLI hardening: corrected contradiction/temporal/stale defaults, short-id resolution in `links`, populated `version history/diff/rollback`, date validation, and crash guards across `feedback`, `import`, `eval`, `query`, and the TUI.

## [2.3.4] - 2026-06-28

### Fixed
- **CRITICAL:** archive no longer deletes memories (the `.md` is moved to an archive dir before the index row is dropped).
- **CRITICAL:** zip-slip path traversal in `memo restore` and `../` traversal in Obsidian image embeds are blocked.
- Concurrency fixes, atomic config/state writes, subprocess timeouts, and correct exit codes across the runtime.

## [2.3.3] - 2026-06-27

### Added
- The statusline self-heals its wiring on session start (gated, default on).

### Fixed
- `memo install-statusline` now **wraps** an existing statusline instead of skipping it, so the `[MEMO]` badge coexists with other tools' badges.

## [2.3.2] - 2026-06-27

### Removed
- Reverted the cross-machine version-file auto-update added in 2.2.0 — it was redundant with the existing tag-based auto-update and shipped broken. Tag-based auto-update is unaffected.

## [2.3.1] - 2026-06-27

### Added
- The `memo resume` picker now previews what changed in a session's cwd since you last worked there, plus that session's open loops.

## [2.3.0] - 2026-06-27

### Added
- `memo episodes search` (CLI) and `memo_episodes_search` (MCP) find past work sessions by meaning.

## [2.2.0] - 2026-06-27

### Added
- Semantic Resume: `memo resume` searches the full session history by meaning via a derived episode index.

## [2.1.1] - 2026-06-27

### Added
- `memo resume` opens an interactive cross-agent picker in a TTY (resumes `claude` / `codex` / …); piped or `--json` still emits the candidate list.

### Fixed
- `memo update` from an editable/dev install now refuses with clear guidance instead of silently updating the wrong runtime.

## [2.1.0] - 2026-06-27

### Added
- `memo eval harvest` mines ground-truth recall labels from real grounded outcomes.
- The outcome loop closes nightly so ranking learns from real usage; continuous entity extraction on save; the unified briefing composes from memo's own corpus.

### Changed
- The reranker sees more candidates (`rerank_input_k` 5→30; validated precision@5 0.786→0.834), with candidate bodies served from the index instead of re-reading `.md` files.

### Fixed
- Hybrid recall mode is usable (its similarity gate compares the true vec cosine).
- 34 correctness bugs across store, memory ops, sync, embedder, capture, temporal, and analytics.

## [2.0.0] - 2026-06-26

First stable release.

### Added
- **Time-machine** — rewind the corpus to any past date (`memo as-of`, `memo diff`).
- **Contradiction radar** — `memo contradict scan` / `triage` detects and resolves conflicting facts.
- **Synthesis pipeline** — `memo synthesize` and the nightly `memo dream` infer cross-cluster insights (opt-in).
- **Cross-Mac git sync** — `memo sync bootstrap <url>` shares a corpus over a private git remote (rebase-before-push, single-owner `flock`).
- **Obsidian vault as source-of-truth** — `MEMO_MEMORIES_IN_VAULT=1`; human edits win on the next reindex.
- **Knowledge graph** — entity extraction, neighbors/path/centrality/communities, backlinks/outlinks.
- **Health scoring & eval gates** — `memo health` and `memo eval recall --gate` for CI or pre-commit.
- **Multi-modal ingestion** — images and audio (macOS Vision OCR), searchable.
- **Session continuity** — open loops, session state, `memo resume`.
- **Warm daemons** — recall (<200 ms) and idle-capture for hookless MCP clients.
- 95 CLI commands and 109 MCP tools across three surface profiles (slim / agent / full).
- `install.sh` uv-first installer and `memo doctor --strict-runtime`.

### Changed
- `MEMO_AUTO_UPDATE` is on by default (tag-based self-update); the `agent` MCP profile (5 tools) is the default.

### Removed
- Purged experimental surfaces (encryption, sharing, federation) and the unused GLiNER entity backend.

---

Releases before `2.0.0`: see [docs/CHANGELOG-archive.md](docs/CHANGELOG-archive.md).
