# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.10] - 2026-06-25

### Fixed

- **`Memory.__init__` auto-detects recall daemon socket.** When the recall daemon is running, `Memory` now automatically switches to `SocketEmbedder` (via a fast ping check) without requiring `MEMO_EMBEDDER_VIA_DAEMON=1`. Eliminates per-process cold MLX loads in `idle-daemon` and other long-lived processes that share a machine with the daemon.

## [1.0.9] - 2026-06-25

### Fixed

- **Test suite: 12 failures resolved.** Fixed `XDG_DATA_HOME` `KeyError` in shims tests, architecture boundary violations, f-string syntax errors, and import ordering issues.
- **TTY guard in `recall_socket.py`.** Now checks `path.startswith("/dev/")` before calling `open()`, preventing creation of a literal `not a tty` file in the working directory.
- **hooks.json TTY filter.** Pipes `tty(1)` output through `grep '^/dev/'` so `MEMO_AGENT_TTY` is only set when a real PTY path is returned, never the string `"not a tty"`.
- **`memo update` git-installed pipx detection.** When memo was installed from a git spec (not PyPI), `pipx upgrade` fails; the updater now reads `pipx_metadata.json` directly (checking common pipx paths when `pipx` is not on `PATH`) and falls back to `pipx install --force` for git-spec installs.
- **No-downgrade version guard.** `memo update` now uses `packaging.version.Version` for comparison, refusing to install a version older than the currently running one.
- **`agent_assets/` removed from package tree.** The directory was accidentally included in the wheel via a glob, causing duplicate-inclusion build failures. Excluded explicitly in `pyproject.toml`.

## [1.0.8] - 2026-06-24

### Fixed

- **Idle auto-save notification now appears in the actual terminal.** When memo agent shims (`memo install-shims`) are active they export `MEMO_AGENT_TTY` (the terminal PTY path, captured before the agent changes it). The idle-maintenance worker now writes the `※ auto save (idle): …` notification directly to that TTY, bypassing Claude Code's hook stderr capture. Falls back to stderr when no `MEMO_AGENT_TTY` is set.

### Changed

- **Startup-banner shims (v2)** capture and export `MEMO_AGENT_TTY` at launch, so async background workers know which terminal to write notifications to.

## [1.0.7] - 2026-06-24

### Changed

- **`memo install-slash` now installs startup-banner shims automatically.** Running `memo install-slash` (or `memo install-slash --client all`) now writes bash shims to `~/.memo/bin/` for codex, devin, opencode, gemini, and blackbox, and appends an `export PATH="$HOME/.memo/bin:$PATH"` snippet to `~/.zshrc` / `~/.bashrc` (idempotent). Re-opening the shell picks up the PATH change and agents show `[MEMO ver]` at launch.
- **`memo install-shims` adds PATH snippet automatically.** The `install_path_snippet()` helper now writes the PATH line directly instead of only printing it.

## [1.0.6] - 2026-06-24

### Fixed

- **install: `mcp remove` is now best-effort before `mcp add`.** The pre-wiring cleanup step that removes any prior MCP entry is no longer fatal. Claude Code returns `No MCP server named "memo"` and Devin returns `not in the user config` on a clean machine; both caused a non-zero exit that was treated as a hard failure, skipping the client entirely and leaving memo unavailable in those agents after a fresh install. The step now logs the exit and proceeds regardless.
- **session idle-maintenance: notification now appears in the terminal.** The detached capture worker was spawned with `stderr=DEVNULL`, silently discarding the `※ auto save (idle): …` notification. Changed to `stderr=None` so the child inherits the terminal's stderr and the message actually appears.

### Added

- **`memo startup-banner --agent <name>`** — prints a `[MEMO ver] | sync …` status line to stderr at agent startup. Fast (no MLX: only `importlib.metadata` + local git).
- **`memo install-shims [--agents …] [--bin-dir …]`** — writes idempotent bash shims to `~/.memo/bin/`. Each shim wraps the next binary in PATH: if that binary is a memflow shim, it executes it directly (memflow already shows a combined memo+memflow banner); otherwise it calls `memo startup-banner` first. Disable with `MEMO_STARTUP_BANNER=0`.

## [1.0.5] - 2026-06-24

### Changed

- **CLI: renamed commands for clarity.** `memo edit <ID>` now edits a stored memory (was `memo update`), and `memo update` now updates the memo software to the latest release (was `memo upgrade` / `memo self-update`). `upgrade` and `self-update` remain as hidden back-compat aliases.

## [1.0.4] - 2026-06-24

### Changed

- **install: auto-update on by default in MCP server env.** `MEMO_AUTO_UPDATE=1` is now set in `_mcp_server_env` so `memo-mcp` self-upgrades to the latest git tag on start (tag-gated, throttled). Keeps the `[MEMO <version>]` statusline badge in sync with releases automatically. Opt out with `MEMO_AUTO_UPDATE=0` in the client env.

## [1.0.3] - 2026-06-24

### Added

- **install.sh: progress spinners + step counter.** Each install phase now shows a numbered step and an animated spinner so it's clear something is happening during slow operations.
- **install.sh: model-download explainer.** Before the download phase, install.sh prints a human-readable breakdown of what is being fetched and why, including sizes and the rationale for deferring the chat model.
- **install.sh: fast model downloads via `hf_transfer`.** `HF_HUB_ENABLE_HF_TRANSFER=1` (plus `HF_HUB_DISABLE_XET=1`) are set for the duration of install, giving 3–5× faster Hugging Face downloads. Both flags are scoped to install time and not persisted to user config.
- **install.sh: wire MCP into all supported clients via `--client all`.** Replaces the retired Windsurf-only wiring; now covers Claude Code, Cursor, Devin Desktop, Blackbox, and any other client registered in `install_mcp.py`. `--client all` is the new default for the full-install path.
- **install.sh: factory memo statusline badge.** After install, a `[MEMO <version>]` badge is written to the user's statusline config so the active version is visible at a glance in the terminal.
- **CLI: `memo upgrade` command.** Renames the former `memo self-update` to `memo upgrade` for discoverability. The old name is kept as a hidden alias for backward compatibility. `memo update <ID>` (edit a stored memory) is unchanged.

### Fixed

- **install.sh: eliminated the "0% hang".** Previously the installer triggered a synchronous download of all models, including the ~6–7 GB chat model, which appeared to stall at 0% with no feedback. Now only the required models (embedder + reranker, ~1–2 GB) are downloaded synchronously; the chat model is fetched in a detached background process and the user is told when to expect it.
- **install.sh: HF_TOKEN warning.** When `HF_TOKEN` is unset, install.sh now emits an informational warning (not a hard failure) explaining that gated models may not download.
- **install.sh: clean-reinstall path for the pipx/uv venv backend.** On a reinstall over an existing pipx-managed environment, the installer now runs `pipx uninstall` and removes the stale venv directory before reinstalling, preventing version-mismatch ghosts from the old environment.
- **install.sh: runtime check is informational, not a hard gate.** `memo doctor` output during install is now advisory; a non-zero exit from doctor no longer aborts the install script.

## [1.0.2] - 2026-06-22

### Fixed

- **sync:** `_pull_rebase` now loops over every conflict stop (was: only the first), so a Mac with multiple local commits that each conflict on `signal/*.json` reconciles instead of staying behind forever.
- **sync:** memo's git subprocesses now run with `GIT_EDITOR=true`/`GIT_SEQUENCE_EDITOR=true`, so `git rebase --continue` after auto-resolving a signal conflict no longer dies with "Terminal is dumb, EDITOR unset" in headless/daemon/SSH contexts (the failure `sync_once` was silently swallowing as "nothing to do").

## [1.0.1] - 2026-06-22

### Added

- **Auto-update on memo-mcp start (opt-in).** When `MEMO_AUTO_UPDATE=1`, memo-mcp checks (throttled, default 6h via `MEMO_AUTO_UPDATE_INTERVAL_S`) for a newer git **tag** (`vX.Y.Z`) and, if found, spawns a detached `memo self-update --to-tag <tag>` in the background — the new version takes effect on the next start. Trigger is a tag (not any commit) so un-tagged/broken pushes never propagate. Default off (public repo); `MEMO_AUTO_UPDATE=1 memo install-mcp --write` bakes it into a machine's agent configs. `memo self-update --to-tag` reinstalls the isolated runtime from `git+<repo>@<tag>` (pipx/uv), since git installs aren't on PyPI.

## [1.0.0] - 2026-06-22

### Added

- **Auto-recall fires on more real prompts (less wasted coverage).** The
  `recall-hook` previously bailed on ~50% of prompts (slash commands +
  short prompts), so memo's context never reached those turns. Now: a slash
  command **with substantive args** recalls on the arg text
  (`/plan how does memo work` → recalls on `how does memo work`), gated by
  `MEMO_RECALL_SLASH_MIN_ARG_CHARS` (default 8) and a
  `MEMO_RECALL_SLASH_DENYLIST` of pure-UI/noise verbs; and a **short follow-up
  inside an active session** ("y eso?") is re-anchored with the last N
  `prompt_trail` prompts via `MEMO_RECALL_SHORT_EXPAND_TURNS` (default 2,
  gated on `MEMO_RECALL_EXPAND_CONTEXT`) instead of bailing. All gating/rewrite
  happens once in `cli.py` before dispatch, so both the daemon and subprocess
  recall paths see the rewritten query. Defaults are backward-safe (bare and
  denylisted slash commands still bail). The recall directive now also asks the
  model to cite the `[id]` of any memoria it relies on, so grounded use is
  traceable. New `session.recent_prompts()` helper.

- **Obsidian-as-source-of-truth storage model.** The `.md` files are now treated
  as canonical and sqlite as a rebuildable index. `MEMO_MEMORIES_IN_VAULT=1`
  (with `MEMO_VAULT_PATH`) stores memorias under `<vault>/<SYSTEM_DIR>/AI/memory`
  so the human-editable vault is the source of truth; ingest already excludes
  that subtree so they're never double-indexed. `memo migrate --into-vault`
  moves an existing install there (non-destructive, `--rollback` restores the
  prior config).
- `memo reindex --rebuild` truncates only the markdown-derivable tables
  (`meta`/`vec`/`fts`) and replays from disk while **preserving** user-signal
  data (`access`, `memory_health`, `source_feedback*`) keyed on the stable id —
  the safe alternative to `rm memvec.db`. A content-addressed embedding cache
  makes a warm rebuild issue ~zero embedder calls.
- `MEMO_SINGLE_DB=1` consolidates the sidecar stores
  (history/graph/contradictions/crossref) into the single `memvec.db` file.
  `memo migrate --consolidate-db` merges existing `*.db` files (renames them
  `*.db.bak`, idempotent).
- **Idle daemon and `memo_idle_capture` MCP tool.** Background auto-capture fires
  every 10 s for agents without Claude Code hooks. Auto-starts from the
  `unified_briefing`, `search`, and `ask` MCP side-effect paths.
- **Two-tier cross-Mac git sync.** `memo sync bootstrap`, `memo sync auto`, and
  `memo sync once` replicate the memory corpus across machines via a bare git
  remote (`memo-sync`). Pull-rebase-before-push, `flock`-based single owner per
  machine, debounced async hooks on every prompt. Cross-machine deletes propagate
  via orphan-prune on pull.
- **`memo install-mcp`** — one-command MCP registration into any agent (Claude
  Code, Devin, Codex, opencode, Windsurf). Pins `MEMO_SOURCE` per-client for
  per-consumer attribution telemetry.
- **`memo stats` (formerly `memo tui`)** — production TUI with utility/verdict
  panels and `--background` HTTP server with live auto-refresh at `localhost`.
  Surfaces "¿funciona?" verdict, per-consumer grounded rate, and token-savings
  estimate.
- **`memo eval grounding`** — ground-truth calibration harness to measure whether
  memo's context is actually used in model answers (paraphrase-aware,
  knowledge-segmented).
- **Hybrid search tuning.** Exact BM25 leg + confident-RRF skip; configurable
  leg weights via `MEMO_SEARCH_VEC_WEIGHT` / `MEMO_SEARCH_BM25_WEIGHT`;
  per-type recency decay half-lives (`MEMO_RECALL_DECAY_HALF_LIFE_*`).
- **Sleep-time compute (dream pipeline improvements).** Signal gather, date
  normalization, prune floor, and orientation passes added to the nightly dream
  pipeline. Progress UI + LLM timeout/thinking hardened.
- **`memo continuity`** — native "what was I working on" command providing
  memflow-style cross-session state recall from within memo.
- **`memo_health_report` MCP tool** — corpus health metrics exposed over MCP, with
  34 covering tests.
- **Token-savings ROI** surfaced in `memo doctor` output and the stats dashboard
  (`MEMO_ROI_TOKENS_PER_*` flags).
- **`memo_idle_capture`** MCP tool for agents without Claude Code hooks to trigger
  background capture on demand.
- Zero-config repo mode for MCP marketplace distribution (self-describing index
  adopts embedder profile from DB metadata when env vars are unset).

### Changed

- **BREAKING — MCP tools renamed `memory_*` → `memo_*`.** All 116 MCP tools now
  share the `memo_` prefix matching the server name (`memo_ask`, `memo_search`,
  `memo_save`, `memo_consolidate`, …) for naming consistency. Clients that call
  tools by name (synapse `memo_backend`, memflow) were updated in lockstep.
  Non-tool identifiers (`memory_dir`, the `memory_health` table, `memory_id`,
  `memory_type`) are unchanged. Re-connect / restart any MCP client to pick up
  the new names.
- `delete()` now removes the canonical `.md` first and aborts (`StorageError`)
  if it can't, so the index never outlives its source file. `save()` no longer
  loses a memoria when indexing fails after the disk write — it marks the file
  embed-pending for `reindex` to replay.
- **Fixed** a data-loss bug: `memo migrate-vault` previously deleted `memvec.db`,
  silently wiping feedback/access/health signal. It now preserves the DB and
  reindexes in place.
- Recall token budget default raised to 600; session-dedup active in the
  subprocess recall path. `MEMO_RECALL_FEEDBACK_HINT` default changed to off
  (opt-in, saves ~20 tokens/recall).
- `cli.py` monolith split into 11+ focused modules (`cli_capture`, `cli_memory`,
  `cli_tui`, `cli_session`, `cli_runtime`, `cli_viz`, …). No command-line
  behavior change.
- Consolidation LLM timeout raised 60 s → 180 s; merge-proposal token budget
  doubled to 2048; robust JSON-parsing with sampling retry on unparseable output.
- MLX GPU work serialized across processes via a lock to prevent `SIGABRT` under
  concurrent embedder calls.
- Dashboard denominators corrected for trend and per-consumer grounded rate;
  stale numbers and false-silent consumer warnings eliminated.

### Fixed

- `rebuild_feedback_vecs` crash on reindex; embedding correctly removed on delete
  rollback.
- Config self-describing index adopts embedder profile from DB metadata when env
  unset, fixing "connection closed" in MCP clients lacking embedder env vars.
- ~80 latent bugs repaired across 3 audit passes (data-loss error swallows,
  error handling, thread safety, concurrency races).
- CI smoke uses Homebrew `python@3.13` (setup-python lacks sqlite extension
  loading; `--strict-runtime` was unsatisfiable in venv CI).
- Sync: commit-before-pull + prune orphans on pull for correct cross-Mac delete
  propagation.
- `VecStore` per-thread connections fix HTTP-transport race under FastMCP worker
  threadpool.
- Idle-maintenance never ran (async hooks receive no stdin pipe) — fixed.
- `inactivity-capture` self-cancelled every turn (keyed on wrong signal) — fixed.
- **Data-loss guard: `reindex --rebuild` refuses an empty `data_dir`.** If the
  markdown source vanished (deleted dir / half-broken clone) while the index is
  still populated, a rebuild would truncate the derivable tables and replay
  nothing — wiping the only surviving copy. It now raises `StorageError` with
  recovery steps instead. Restore the `.md` first, or use `memo reindex` (no
  `--rebuild`) to reconcile.
- **Data-loss guard: `sync bootstrap` refuses a broken clone.** A `dest` that is
  a git clone but has zero `.md` under `memorias/` is no longer "reused" (which
  then fed `reindex --rebuild` → wipe); it raises with `git restore` / re-clone
  guidance.
- `install.sh` injects `consciousness-contracts` best-effort from a local
  checkout (`MEMO_CONTRACTS_PATH`, default `~/repos/consciousness-contracts`) so
  `install-mcp` + the shared embed cache work on a fresh Mac; absent → memo runs
  with fallbacks. install.sh also hints `MEMO_SYNC_REMOTE` when no corpus is wired.

## [0.8.0] - 2026-05-21

### Changed

- The one-line installer now asks for confirmation before downloading the
  ~7 GB MLX model bundle. On an interactive terminal it prompts `[Y/n]`
  (default yes); on a piped install it defaults to yes. Override with
  `MEMO_INSTALL_DOWNLOAD_MODELS=yes|no|auto`. Models are part of memo's
  structure (embedder + reranker + chat are required for retrieval and
  ambient recall), so the default behaviour is unchanged — only the UX
  surfaced a confirmation knob.
- Marked the experimental modules (`agent`, `cognitive`, `collaborative`,
  `contextual`, `contradict`, `crossref`, `encryption`, `federation`,
  `lifecycle`, `multimodal`, `navigation`, `proactive`, `sharing`, `sync`,
  `versioning`, `chunker`) as EXPERIMENTAL in their docstrings — not
  covered by the test suite, not exposed via MCP, API may change without
  notice. Added `src/memo/experimental_index.md` listing them.
- Added a recall server (`src/memo/recall_server.py`) plus
  `tests/test_recall_hooks.py` covering the ambient recall path.
- Expanded `src/memo/cli.py` significantly (+1.5 k lines) with new
  commands, session-capture work, and the install-flow helpers.
- The one-line installer now attempts to configure memo in Claude Code, Codex,

  and Windsurf after model download and `doctor --strict-runtime`. It runs that
  agent setup in best-effort mode so a missing client CLI warns instead of
  aborting the memo install.
- Added `memo mcp-command --client windsurf` and
  `memo install-slash --client windsurf`, which update
  `~/.codeium/windsurf/mcp_config.json` with an absolute `memo-mcp` stdio
  server while preserving existing Windsurf MCP servers.
- Restored the documented `memo backup --out <zip>` portable backup shape by
  making the `backup` group invoke the portable zip path when no subcommand is
  provided.
- Added `docs/install-new-mac.md`, a fresh-Mac migration checklist covering
  install, agent registration, portable restore, synced folders, model-profile
  parity, and verification.
- Documented the supported production install shape: keep memo isolated as
  `pipx` / `uv tool` / Homebrew rather than vendoring it into a project
  virtualenv.
- Added `install.sh`, a curl-pipeable installer that installs memo through
  `pipx`, checks macOS/Apple Silicon + Python >= 3.13, removes the legacy
  `memo-mcp` pipx package if present, and runs `memo doctor --strict-runtime`.
- Changed the curl installer default to GitHub `master`, with explicit
  `MEMO_INSTALL_FROM_PYPI=1` / `MEMO_VERSION=...` knobs for PyPI installs.
- Expanded README install verification with `memo doctor --strict-runtime`,
  `memo mcp-command`, duplicate-install checks, and the model downloads
  needed by the `balanced` / `quality` profiles.
- Removed stale local-install examples that referenced a non-existent
  `[mlx]` extra; MLX is part of the normal Apple Silicon dependency set.
- Corrected Claude Code plugin install docs to use the current marketplace
  flow (`claude plugin marketplace add ...` then `claude plugin install
  memo@memo`) and noted that existing sessions need a restart to see `/memo`.
- Extended `memo mcp-command` with `--client codex`, `--client devin`,
  `--client claude-desktop`, and `--client windsurf` so the same isolated
  `memo-mcp` runtime can be registered across the main MCP-capable clients
  without hand-writing config.
- `memo mcp-command` now pins `MEMO_NONINTERACTIVE=1` and forwards current
  `MEMO_*` model/storage overrides into MCP client configs, preventing
  1024/2560 embedder-dimension drift between shell and agent sessions.
- Added `memo install-slash`, a one-shot installer for the visible exact
  `/memo` command/skill and `memo` MCP server in clients that support that UX:
  Claude Code and Devin.
- Changed `memo install-slash --client codex` to fail fast because Codex CLI
  currently installs plugins/MCP metadata but does not expose arbitrary plugin
  commands in the interactive `/` menu. Codex remains supported through
  `memo mcp-command --client codex` and the `plugins/memo` MCP plugin.
- Packaged Claude/Codex/Devin agent assets into the wheel under
  `memo/agent_assets`, so `memo install-slash` works from `pipx`, `uv tool`,
  and Homebrew installs without requiring a local repo checkout.
- Updated developer guidance and the Claude Code skill to use the current
  `mlx-memo` distribution name, absolute `memo-mcp` registration flow, and
  package-metadata-based `memo.__version__`.
- Refreshed Homebrew tap examples so they no longer hard-code old 0.5.x
  placeholders.

### Fixed

- Claude Code MCP registration now uses `claude mcp add-json`, avoiding the
  current `claude mcp add -e ...` argument parsing failure with env vars.
- `VecStore` now fails fast when an existing `memvec.db` has a vec0 embedding
  dimension different from the active config, with a direct reindex/remediation
  message instead of a late sqlite-vec query error.

## [0.7.0] - 2026-05-13

### Added — **Contradiction radar + dedupe**

memo now actively flags when two memorias disagree (especially when one
is stale and the other supersedes it) and helps you resolve duplicates.
Recall stops surfacing outdated facts as authoritative.

- **`memo contradict scan`** — corpus-wide walk. For each memoria,
  fetches vec-neighbors above a cosine floor, asks the helper LLM to
  classify the pair as contradiction / evolution / consistent /
  unrelated, and persists `contradiction` and `evolution` verdicts to
  a new sidecar DB (`contradictions.db`). Pairs already resolved by
  the user are never re-classified, so a re-scan is cheap.
- **`memo contradict list`** — show open (or any-status) pairs with
  confidence + rationale.
- **`memo contradict triage`** — interactive walker. For each open
  pair: shows both excerpts (older on top, newer below, `(stale)`
  marker if past the threshold) and applies the user's verdict:
  `f` fuse via `AdvancedConsolidator`, `n` newer wins, `o` older wins,
  `e` mark as evolved, `d` dismiss as false positive, `s` skip.
- **`memo contradict stats` / `reopen`** — corpus-level counts and a
  way to send a resolved pair back to the open queue.
- **`memo dedupe`** — higher-threshold wrapper over `consolidate`
  aimed at obvious paste-restate / double-save duplicates. With
  `--apply`, walks each cluster and offers an LLM-synthesized merge.
- **New module `memo.contradict`** with `ContradictionStore`
  (sqlite sidecar, status lifecycle `open|fused|kept_newer|kept_older|
  evolved|dismissed`), `ContradictionScanner` (corpus walker), and
  `PairRecord` / `ScanResult` dataclasses.
- **MCP tools**: `memory_contradict_scan`, `memory_contradict_list`,
  `memory_contradict_resolve`, `memory_contradict_stats`. Same
  contract over stdio.
- **Sidecar isolated**: contradictions live in their own sqlite file
  (`~/.local/share/memo/contradictions.db`) following the same
  convention as `history.db` / `graph.db`. Hot vec reads keep their
  WAL.
- **Self-cleanup**: deleting a memoria automatically drops dangling
  pairs touching it.

### Changed

- `Memory.__init__` opens the contradiction sidecar lazily; callers
  that never invoke the radar pay no extra sqlite handle.

## [0.6.0] - 2026-05-12

### Added — **Time-machine (THE differentiator)**

memo is now the only agent-memory product that lets you rewind the
corpus to any past date. Replays `history.db` events in reverse from
"now" to reconstruct a snapshot at any past timestamp.

- **`memo as-of search <query> --date YYYY-MM-DD`** — semantic search
  over the snapshot. Live embedder + post-filter to the historical
  record set.
- **`memo as-of ask <question> --date YYYY-MM-DD`** — RAG against the
  snapshot. The system prompt tells the model the corpus view is
  historical so it doesn't smuggle in later facts.
- **`memo as-of list --date YYYY-MM-DD`** — list memorias that
  existed at that date (recent-first by `updated`).
- **`memo diff --from <date> [--to <date>]`** — added / removed /
  updated between two snapshots.
- **MCP tools**: `memory_search_as_of`, `memory_ask_as_of`,
  `memory_diff` exposed over stdio so any MCP client gets the same
  contract.
- New module `memo.time_machine` with `reconstruct(memory, as_of)`
  and `diff(memory, from_ts, to_ts)`. Public API surfaces
  `CorpusSnapshot` (`.list()`, `.search()`, `.ask()`) and `CorpusDiff`
  (`added`, `removed`, `updated`, `summary()`).

### Changed

- `_now_iso()` upgraded from second to millisecond precision. Same
  ISO 8601 wire format, just sharper. Required so the reverse-replay
  can distinguish save/update/delete pairs that happen within a
  single second.

### Docs

- New SVG diagram `docs/time-machine.svg` illustrating the
  reverse-replay algorithm visually.
- README hero section now leads with time-machine as **THE**
  differentiator vs the 8 competitor projects.

### Tests

- 12 new tests in `tests/test_time_machine.py` covering snapshot
  reconstruction (empty corpus, before-save, after-delete,
  between-save-and-delete, title-revert), diff (added / removed /
  updated / summary), ISO-string parsing, and snapshot-bound search.

## [0.5.0] - 2026-05-12

PyPI dist rename + TUI follow-ups (q/ESC quit, smaller layout).

### Changed

- **PyPI distribution renamed `memo-mcp` → `mlx-memo`.** The previous
  name collided with [`milasd/memo-mcp`](https://github.com/milasd/memo-mcp)
  (a ChromaDB-backed journal MCP) and risked install ambiguity for new
  users. The Python module (`memo`), CLI binary (`memo`), MCP server
  binary (`memo-mcp`), and the GitHub repo all keep their names — only
  the PyPI dist moved. Existing `pip install memo-mcp ≤ 0.4.3` users
  keep working; new installs use `pip install mlx-memo`. README +
  CLAUDE.md updated, badges repointed.

### Added (0.4.2 + 0.4.3 follow-ups, merged into 0.5.0)

- **`q` / `ESC` exit** in `memo tui`. Background stdin reader in
  cbreak mode sets a stop event; main loop polls. Falls back to
  Ctrl+C-only when stdin isn't a TTY. Footer advertises the keys.
- **TUI layout shrunk to ~18 rows**. Removed the panel-framed hero
  (now an inline footer status line); corpus and runtime collapsed
  to single-line summaries.

### Fixed

- **Legacy-path warning** (`stored path(s) don't resolve…`) now
  silenced inside `memo tui` via `MEMO_SUPPRESS_LEGACY_WARN=1`. The
  user can't act on it from inside the alt screen anyway. Outside the
  TUI the message was rephrased ("heads-up: tu índice apunta a paths
  antiguos") and explicitly labeled as not-an-error.

## [0.4.1] - 2026-05-12

Adds a live terminal dashboard and the recall-log plumbing that feeds it.

### Added

- **`memo tui`** — live, colored Rich-Live dashboard. Six panels:
  corpus stats (totals + per-type breakdown + project count), runtime
  (MLX warm/cold flags for embedder / reranker / chat, vault size,
  watcher status from `launchctl print`), recent saves (last 10 from
  `history.db`), recent recalls (last 8 from the new recall log), top
  tags (project tags highlighted), and 14-day saves/recalls
  sparklines (`▁▂▃▄▅▆▇█`). Refresh `--refresh N` (default 1.0 s).
  Ctrl+C to exit. Zero new deps — Rich was already in.
- **Recall log JSONL** at `~/.local/share/memo/recall.log`. The
  `memo recall-hook` appends `{ts, prompt, hits[]}` per invocation
  (best-effort, failures swallowed). Auto-rotates at ~200 KB to the
  last 200 entries. Powers the TUI's recall panel; failures here can
  never affect the hook's output.
- **`/memo tui`** routing in `skills/memo/SKILL.md` (along with
  `/memo watch`, `/memo install-watcher`, `/memo mine-history`).

### Changed

- New module `memo.dashboard` (~370 lines). Public API:
  `run_tui`, `render`, `sparkline`, `append_recall_log`,
  `read_recall_log`.

## [0.4.0] - 2026-05-12

Minor release — five "gamechanger" features land alongside a major
repo-hygiene pass to make the project public-ready.

### Added

- **Project-scoped recall.** `memo save` now auto-attaches a
  `project:<repo>` tag derived from the git toplevel of the caller's
  cwd (or the `MEMO_PROJECT_TAG` env var). The recall hook reads `cwd`
  from the Claude Code hook payload and additively boosts the score of
  memorias that share the current project tag by
  `MEMO_RECALL_PROJECT_BOOST` (default `0.15`). Opt out per-call with
  `memo save --no-project-tag` or globally with
  `MEMO_AUTO_PROJECT_TAG=0`. New module: `memo.project`.
- **Token-budget-aware recall.** `MEMO_RECALL_TOKEN_BUDGET` (default
  `0` = off) packs memorias greedily by score until the budget is
  reached; the final memoria gets body-truncated to fit instead of
  being dropped wholesale. Token estimate is `len(text)//4` — no
  tiktoken dep.
- **`memo mine-history`** — bulk-mine past Claude Code transcripts
  under `~/.claude/projects/<hash>/*.jsonl` for insights, running the
  same prefilter → helper-LLM extract → embedding-dedup pipeline as
  the live capture hook. Resumable per-file via
  `~/.local/share/memo/mine-history.json`. Flags:
  `--path / --since / --limit / --dry-run / --debug / --json`.
  New module: `memo.transcript_miner`.
- **MCP resources** — `memo://recent` (top-20 by `updated` desc, with
  per-memoria `memo://memory/<id>` links) and `memo://memory/{id}`
  (full record, accepts prefix ≥4 chars). Clients can pin / drag
  memorias into context without paying tool-call overhead per access.
- **`memo watch`** — file-watcher daemon. Watches `cfg.memory_dir`
  recursively via `watchdog`/FSEvents and triggers a debounced
  `Memory.reindex()` (`--delay` configurable, default 2 s) when `.md`
  files are modified. Bundled installer:
  - `memo install-watcher` writes
    `~/Library/LaunchAgents/com.memo.watch.plist`, loads it via
    `launchctl bootstrap`, and verifies. Logs to
    `~/Library/Logs/memo/`. `KeepAlive=true`.
  - `memo uninstall-watcher` boots out the job and removes the plist.
  New module: `memo.watcher`. New dep: `watchdog>=4.0`.

### Changed

- **README rewritten for a public audience.** Cleaner hero, expanded
  alternatives matrix (mem0 / letta / cognee / supermemory / mem-vault
  / MCP-memory reference / engram) with the seven differentiators
  spelled out in plain terms, and an updated `docs/architecture.svg`
  that covers clients → MCP/CLI → core → MLX → storage rather than
  just the recall pipeline.
- **`.gitignore` expanded** to cover `dist.v*/`, `.claude/`,
  `.windsurf/`, `integrations/**/node_modules/`,
  `integrations/**/dist/`, `integrations/**/.paperclip-sdk/`, and
  common IDE dirs.
- **Test fixture default** now sets `MEMO_AUTO_PROJECT_TAG=0` so tests
  asserting exact tag sets aren't polluted by the cwd-derived
  project tag. Tests exercising the auto-tag flow opt back in
  explicitly via `monkeypatch.setenv`.

### Removed

- Stale `dist.v0.2.0/`, `dist.v0.3.0/`, `dist.v0.3.1/` build snapshots
  and accumulated `.DS_Store` files from the repo. `dist/` was already
  gitignored; the dated variants weren't.

## [0.3.3] - 2026-05-08

Patch release — install-from-git fixes surfaced while validating the
distributed-install flow on a clean machine.

### Fixed

- **`memo --version` crashed** with `RuntimeError: 'memo' is not
  installed. Try passing 'package_name' instead.` because click's
  `version_option` defaulted to `package_name="memo"` while the actual
  PyPI/wheel dist is `memo-mcp`. Pinned the lookup explicitly.
- **`DEFAULT_MEMORY_SUBDIR` pointed at the deprecated archive path**
  (`04-Archive/99-obsidian-system/AI/memory`). Updated to the
  current `Obsidian/AI/memory` location, matching the user-facing
  vault reorganization done on 2026-05-08. Existing installs that
  override via `MEMO_MEMORY_SUBDIR` are unaffected.

## [0.3.2] - 2026-05-07

Patch release — BM25 recall fix.

### Fixed

- **BM25 search wrapped query in phrase quotes** (`"foo bar"`) which
  required the words to appear consecutively. This killed recall on
  natural multi-word Spanish queries:
  - `"Astor terapia ocupacional"` did NOT match the document titled
    `"Informe Terapia Ocupacional — Astor Ferrari"` because the words
    don't appear in that exact consecutive order.

  Fix: tokenize via `\w+` (Unicode-aware), wrap each token in its own
  phrase quotes, join with whitespace (FTS5's implicit AND). Now the
  query is `"Astor" "terapia" "ocupacional"` — matches any doc
  containing all 3 words anywhere, any order.

### Verified

Same query post-fix:
- BM25: `Astor — Informe Terapia Ocupacional feb 2026` returns first hit ✓
- Hybrid (vec+BM25 RRF): same doc in top-3 ✓

Other corpus queries that benefit:
- `MLX migration` → `obsidian-rag: migración Ollama → MLX` (was missing pre-fix)
- `obsidian-rag bug fix` → `Bug pattern — sqlite3 'database is locked'` (was missing pre-fix)

## [0.3.1] - 2026-05-07

Patch release — bug fixes for the v0.3.0 ingest pipeline.

### Added

- `memo ingest <vault-path>` CLI command — bulk-ingest .md files from any
  Obsidian vault into the memo index. Synthesizes ids from path hash so
  user .md files are not modified. Idempotent. See README "Ambient memory"
  section for usage.
- `MEMO_INGEST_MIN_CHARS` env var (default 200) — skip notes shorter than
  this threshold. Tag-only stubs (`#tagA #tagB` + 1-line question)
  produce noisy embeddings near the corpus centroid that match generic
  queries with high false-positive rate. Filtering them improves recall
  precision on queries with proper nouns.

### Fixed

- **Embedder API misuse** in `recall-hook`, `prewarm`, and `ingest`:
  `MLXEmbedder.embed()` is batched (signature `Sequence[str] →
  list[list[float]]`); passing a bare string iterated per-char,
  producing variable-dim outputs (135, 512, 2465...) instead of 1024.
  Cascade Metal GPU error after several mismatches. Fix: wrap input
  in `[composed]` and take `[0]`.
- **Plugin install path** — bumping from 0.3.0 to 0.3.1 so users on the
  Claude Code plugin marketplace pick up these fixes via
  `claude plugin update memo@memo`.

## [0.3.0] - 2026-05-07

**Game-changer release**: memo turns into an *ambient* context layer.
Memorias auto-inject as `additionalContext` on every Claude Code prompt,
without the user invoking `/memo` at all.

### Added

- `memo recall-hook` CLI command — Claude Code `UserPromptSubmit` hook that
  embeds the user prompt via the MLX embedder, runs vec-only search against
  the memo index, and emits relevant memorias as `additionalContext` markdown
  on stdout. Sub-1.7s warm latency on a 223-doc corpus, well within the
  default 5s hook timeout.
- `memo prewarm` CLI command — `SessionStart` hook that pre-loads the MLX
  embedder so the first `recall-hook` invocation of the session is fast
  (warm load: ~500ms vs ~2s cold).
- `hooks/hooks.json` bundled in the plugin — auto-wires the two hooks above
  when users install via `/plugin install memo@memo`.
- 6 env vars to tune ambient memory behaviour:
  - `MEMO_RECALL_DISABLE` — kill switch (default: enabled).
  - `MEMO_RECALL_TOP_K` — default 3.
  - `MEMO_RECALL_MIN_SIM` — cosine similarity floor, default 0.6.
  - `MEMO_RECALL_MIN_PROMPT_CHARS` — default 12.
  - `MEMO_RECALL_BODY_CHARS` — snippet length, default 240.
  - `MEMO_RECALL_SKIP_SLASH` — skip recall on `/` prompts, default 1.
  - `MEMO_RECALL_DEBUG` — print failure reasons to stderr, default 0.

### Why this is the game changer

Before: user types `/memo save 'X'` to save and `/memo search 'Y'` to recall.
Friction = adoption blocker; memory only helps when remembered to invoke.

After: memo silently consults the user's past on every prompt and injects
the most relevant 3 memorias if any score above 0.6 cosine similarity.
Zero `/memo` invocations needed for the recall side. The agent "knows
your past" automatically.

### Empirical tuning

Threshold 0.6 was picked after testing the 223-doc corpus:
- "qué decidí sobre MLX vs Ollama" → 3 hits at 0.71-0.74 (all relevant).
- "how to bake apple pie" (corpus has zero food memorias) → 0 hits at 0.6
  (3 noise hits at 0.5-0.6 cut by the floor).
- "qué hice con whatsapp" → 3 hits at 0.6-0.75 (whatsapp work).

### Privacy / local-first

Hot path is 100% MLX in-process. Embedder = `Qwen3-Embedding-0.6B-4bit-DWQ`,
search = `sqlite-vec`. Zero network calls, zero cloud APIs, zero telemetry.
The hook input (your prompt) never leaves the machine.

### Save side (Phase B)

Passive memory extraction (auto-`memo save` from chat transcripts) is NOT
in 0.3.0 — recall side first to validate the architecture, save side in
0.4.0 once we have signal on whether the recall hook is helpful in real use.

## [0.2.0] - 2026-05-07

First public release. Distribution name on PyPI is `memo-mcp`
(`memo` was already taken).

### Added

- Public PyPI distribution as [`memo-mcp`](https://pypi.org/project/memo-mcp/).
- Claude Code plugin format (`.claude-plugin/plugin.json`) — installable via
  the Claude Code plugin marketplace flow.
- `.mcp.json` bundled in repo root so MCP-aware clients can auto-register.
- `skills/memo/SKILL.md` bundled in repo — slash-command UX layer for Claude
  Code CLI users (optional).
- LICENSE file (MIT).
- README polish: install, quickstart, architecture diagram, comparison vs
  `mem-vault` / `mem0` / `engram`.
- Migration script `scripts/migrate-from-mem-vault.py` (already shipped in
  0.1.0 development; documented in README for this release).

### Changed

- Bumped Development Status from `3 - Alpha` to `4 - Beta` in pyproject classifiers.
- Author name expanded from "Fer" to "Fernando Ferrari" for PyPI metadata clarity.

## [0.1.0] - 2026-04-28

Initial development release (private — not on PyPI).

### Added

- MLX-native memory MCP for Apple Silicon. Stack: `mlx-lm` + `mlx`
  (Qwen2.5-7B/3B-Instruct-4bit + Qwen3-Embedding-0.6B-4bit-DWQ),
  `sqlite-vec` for vectors, markdown files in Obsidian vault for storage.
- Tools exposed: `memory_save`, `memory_search`, `memory_list`, `memory_get`,
  `memory_update`, `memory_delete`, `memory_stats`, `memory_reindex`,
  `memory_ask` (RAG over memorias with inline citations), `memory_consolidate`
  (cluster + LLM merge proposals), graph queries (entity extraction).
- CLI (`memo`) with subcommands matching MCP tools.
- MCP server entry point (`memo-mcp`) using FastMCP framework.
- History tracking via SQLite for memory edits + accesses.
