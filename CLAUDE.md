# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

memo is a local-first semantic memory: MLX embeddings + sqlite-vec hybrid
(vec + BM25) search, reranking, a knowledge graph, temporal reasoning, an MCP
server, and a CLI. Single-user, runs offline. PyPI distribution name is `mlx-memo`; CLI binary is `memo`; MCP server binary is `memo-mcp`.

## Dev commands

```bash
# Tests (no MLX needed — MLX tests skip automatically on non-Apple Silicon)
uv run --no-sync pytest tests/                          # full suite
uv run --no-sync pytest tests/test_foo.py::test_bar -v  # single test

# Type checking
uv run --no-sync mypy src/memo/

# Lint + format
uv run --no-sync ruff check src/ && uv run --no-sync ruff format src/

# Run the CLI
uv run --no-sync memo <cmd>

# Validate MEMO_* flags (typos, unknown vars)
uv run --no-sync memo config validate
```

## Architecture

**`Memory` facade** (`src/memo/memory/facade.py`) multiply-inherits ten operation mixins — `_WriteOpsMixin`, `_UpdateOpsMixin`, `_DeleteOpsMixin`, `_SearchOpsMixin`, `_AskOpsMixin`, `_RerankOpsMixin`, `_RepoOpsMixin`, `_MaintainOpsMixin`, `_ConsolidateOpsMixin`, `_ReplayOpsMixin` — each in their own `src/memo/memory/<op>_ops.py` file. Module-level constants, prompts, and pure helpers are in `src/memo/memory/record.py`. Never import from a mixin directly; always go through `Memory`.

**MCP server** (`src/memo/server.py`) registers tools via `build_server()`. Each domain is a `server_<domain>.py` module that exports `register(server, memory)` — called once in `build_server()`. Adding a new MCP tool = create `src/memo/server_<domain>.py` + add one `register` call in `server.py`. The `_MaintainOpsMixin` method is directly callable from tests via `mock_memory.<method>()`.

**Storage** (`src/memo/store/`) is a subpackage. `VecStore` in `queries.py` is the primary interface: one sqlite-vec DB file, thread-local connections (required for FastMCP HTTP worker threadpool). Writes use `_tx()` (`BEGIN IMMEDIATE`); vectors are packed float32 blobs; WAL mode + `busy_timeout`.

**CLI** is in `src/memo/cli.py` (entry-point wiring only) + `src/memo/cli_<domain>.py` files. Each domain file exports a Click command or group imported and registered in `cli.py`.

**Flags** (`src/memo/flags.py`) is the single registry for all `MEMO_*` env vars — it aggregates `FlagSpec`s defined in the per-domain `flags_<group>.py` modules (`flags_recall/search/behavior/ingest/misc.py`, base types in `flags_base.py`). Add a new flag in the matching `flags_<group>.py`. Use `flag_bool/int/float/str(name)` — never `os.environ.get("MEMO_...")` inline. `memo config validate` parses every set flag.

**Cache** (`src/memo/embedder.py`): query embeddings use shared LRU cache from `consciousness_contracts.cache` when `MEMO_QUERY_CACHE_SIZE > 0`. This eliminates duplication with Synapse's embed cache and ensures consistent cache behavior across the trinity.

**URI helpers** (`src/memo/memory/maintain_ops.py`): backend-native replay uses shared URI parsing/validation from `consciousness_contracts.uri` when available. Fallback to manual parsing for CI/clean installs.

**Sync — two explicit tiers** (`src/memo/sync_git.py`, `src/memo/identity.py`):
- **LOCAL (intra-machine):** all sessions on a Mac share one `data_dir`/`memvec.db`
  (WAL, thread-local conns), so a save/capture is visible to sibling sessions on
  their next recall **with no git**. `sync_tier(cfg)` → `"local"` when no git
  remote is configured.
- **REMOTE (inter-machine):** the git `memo-sync` remote is the only cross-machine
  channel. `sync_tier(cfg)` → `"remote"`. ONE owner per machine: `sync_once()`
  takes an `flock` on `state_dir/.sync.lock` (concurrent same-machine sessions
  skip — their writes ride the lock holder's push) and does
  **pull-rebase-before-push** so an advanced remote rebases instead of rejecting.
  Triggers: per-prompt `memo sync auto` (debounced via `MEMO_SYNC_PUSH_DEBOUNCE_S`
  / `MEMO_SYNC_PULL_INTERVAL_S`, async hook) + `memo sync once` on Stop. A failed
  push stamps `state_dir/sync_pending` and retries next trigger.
- **Identity** (`identity.py`): stable `machine_id` (= persisted `cfg.device_id`,
  also `history.device_id`) + `hostname` (cross-tool match key) + `session_id` +
  `terminal`. Commits are attributed `[<hostname>·<session>]`. memflow can
  reference a terminal later; memo only exposes the identity.
- **Durable capture:** `memo capture-tick` (per-prompt async, self-throttled by
  `MEMO_CAPTURE_INTERVAL_S`) captures NEW turns since a per-session watermark so a
  long session's insight reaches `.md` before Stop, not only at Stop.
- `memo sync status` / `memo doctor` surface the silent no-op (not a clone) and
  stranded commits. The legacy audit-log replay (`sync.py` `SyncManager`) is the
  `--remote <path>` fallback; the `SyncCoordinator`/consciousness_contracts hook is
  not wired (the machine `flock` is the coordinator).

**Memory types** (`src/memo/tiers.py`): durable tier = `decision`, `fact`, `bug`, `feedback`, `preference`, `note`, `manual`, `synthesis`; reference tier = `reference` (bulk-ingested vault chunks, excluded from auto-recall). `synthesis` memories are auto-generated by `memo synthesize` / `MEMO_SYNTHESIS_ENABLED=1 memo maintain` — cross-cluster inferred insights with `synthesis_sources` provenance in their `extra` frontmatter bag.

## MLX invariants (do not violate)

The embedder/LLM path depends on four invariants. A diff touching
`src/memo/embedder.py`, `src/memo/llm.py`, `src/memo/memory/` (any file), or anything
that calls `.embed()` / `.chat()` must preserve all four:

1. **Asymmetric retrieval prefix on QUERIES only.** Query text gets
   `_QUERY_INSTRUCTION_PREFIX` (see `embedder.embed_query`); stored documents do
   **not**. Prefixing both (or neither) collapses cosine similarity — the model
   places prefixed and raw inputs in different regions of the space.
2. **`MLXEmbedder.embed()` takes `Sequence[str]`, never a bare `str`.** A bare
   string is iterated as characters and silently produces garbage. Always wrap:
   `embed([text])`.
3. **`MEMO_EMBEDDER_DIMS` must match the model.** 1024 / 2560 / 4096 for the
   0.6B / 4B / 8B Qwen3-Embedding models. Default model is
   `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` (1024-dim). A mismatch corrupts
   the vec0 table or fails the dims-validation guard in `store/queries.py`.
4. **`mlx` / `mlx-lm` imports stay deferred** (inside functions, not at module
   level) — module-level import drags the MLX runtime into every CLI invocation
   and blows the recall-hook cold-start budget.

When stubbing `MLXEmbedder.embed` in tests, also pin `MEMO_EMBEDDER_DIMS` to the
stub's output dim.

## Recall hook (UserPromptSubmit) — 5s budget

`memo recall-hook` is wired into Claude Code's UserPromptSubmit hook
(`hooks/hooks.json`). End-to-end it must stay under ~5s. Cold MLX load is ~2s,
leaving ~3s for embed + vec search + format. The rerank candidate pool
auto-shrinks (and `MEMO_RECALL_TOKEN_BUDGET` caps output) so warm latency stays
under the timeout. Anything added to the hook path — `hooks/hooks.json`, the
`recall-hook` command in `src/memo/cli.py`, `embedder.py`, `store/queries.py` search —
must respect that budget. A warm `memo-mcp` HTTP/recall daemon serves embeds
over a socket (`MEMO_EMBEDDER_VIA_DAEMON`) to skip cold load.

## Runtime

`memo` and `memo-mcp` must resolve from the **same isolated runtime**
(pipx / `uv tool` / Homebrew) — **not** a project `.venv`. A mixed runtime is the
usual cause of "works in CLI, broken in MCP". `memo doctor --strict-runtime`
checks this.

Runtime plumbing lives in `src/memo/runtime/`: `install.py` (runtime/install
detection + `memo doctor`/install helpers), `daemon.py` (launchctl + warm
recall-daemon lifecycle, `MEMO_EMBEDDER_VIA_DAEMON` socket), `report.py`
(install-report presentation). These were extracted from `cli.py`/`cli_runtime`;
the warm `memo-mcp` daemon they manage is what serves embeds over the socket to
keep the recall hook under its 5s budget.

## BM25 / Spanish search

FTS5's tokenizer wraps each `\w+` token in its own phrase quotes, so a
multi-token query becomes an AND-of-tokens (not a phrase match). `store/bm25_queries.py`
tokenizes, AND-joins, and falls back to OR only on zero recall. Diacritics are
folded (`unicode61 remove_diacritics 2`) so "decision" matches "decisión".

## Storage

`VecStore` (`src/memo/store/queries.py`) is sqlite-vec backed, one DB file, **one
connection per thread** (thread-local — required for the FastMCP HTTP transport's
worker threadpool). Writes go through `_tx()` (`BEGIN IMMEDIATE`); vectors are
packed float32 blobs; WAL mode + `busy_timeout` give concurrent readers + a
writer.

### Markdown is the source of truth; sqlite is a rebuildable index

The `.md` files are canonical; the sqlite index is **derived and replayable**:

- **Authority.** `save()` writes the `.md` first, then indexes — if indexing
  fails, the memoria is stamped `_memo_embed_pending` on disk and `memo reindex`
  replays it (the save never silently vanishes). `delete()` removes the `.md`
  **first** and aborts (`StorageError`) if the file can't be removed, so the
  index never outlives its truth-bearing file. A hand-edit in Obsidian wins on
  the next `reindex` (body_hash mismatch → disk overwrites the index).
- **Rebuild, don't `rm`.** Use `memo reindex --rebuild` (not `rm memvec.db`) to
  rebuild from disk. It truncates only the markdown-derivable tables
  (`meta`/`vec`/`fts`) and preserves the **user-signal** tables — `access`,
  `memory_health`, `source_feedback*` — which are PRIMARY data not present in
  markdown and re-join on the stable memoria `id`. A content-addressed embedding
  cache (`repo_embedding_cache`, keyed on `model+dims+sha256(text)`) makes a warm
  rebuild issue ~zero embedder calls.
- **Memorias in the vault.** `MEMO_MEMORIES_IN_VAULT=1` (needs `MEMO_VAULT_PATH`)
  stores memorias under `<vault>/<SYSTEM_DIR>/AI/memory` so the human-editable
  Obsidian vault is the source of truth. Ingest already excludes `AI/` and any
  `id:`-frontmatter file, so they're never double-ingested as reference tier.
  `memo migrate --into-vault` moves an existing install there (non-destructive,
  `--rollback` restores the prior config). `memo migrate` never drops `memvec.db`.
- **One DB file (opt-in).** `MEMO_SINGLE_DB=1` folds the sidecar stores
  (history/graph/contradictions/crossref) into `memvec.db` — each keeps its own
  connection to the one file (WAL allows it; no shared-transaction risk). Run
  `memo migrate --consolidate-db` once to merge existing `*.db` files (renames
  them `*.db.bak`, idempotent). Default off keeps the historical multi-file
  layout. The `cfg.*_db` path properties collapse onto `db_path` when on.

## Test isolation (see `tests/conftest.py`)

- Use the `tmp_cfg` fixture or build an isolated `Config` — never call
  `Config.from_env()` without controlling the environment.
- `CliRunner` invocations set `MEMO_NONINTERACTIVE=1`, `MEMO_DATA_DIR`, and
  `MEMO_STATE_DIR` in `env=` (conftest defaults `MEMO_NONINTERACTIVE=1`).
- Real MLX forward passes are gated by `@pytest.mark.requires_mlx` (auto-skipped
  when `mlx_lm` isn't importable).
- Never read or write the developer's real vault.

## Config & errors

- `MEMO_*` behavioral flags live in `src/memo/flags.py` (registry + typed accessors). Storage/model config lives in `src/memo/config.py` (typed `Config` dataclass). Prefer `flag_bool/int/float/str` over raw `os.environ`. `memo config validate` catches typos.
- Domain errors live in `src/memo/errors.py` (`MemoError` base). Raise/catch
  those rather than bare `Exception` in non-defensive code.

## Releasing

Bump the version in sync across **four** source-of-truth files:
`pyproject.toml` `[project].version`, `.claude-plugin/plugin.json`,
`server.json` (version + package version), and `CHANGELOG.md`
(Keep-a-Changelog). Commit / tag / push stays manual.

## Source of truth — role & contract

memo is the canonical store of **durable semantic knowledge**: decisions,
facts, preferences, learnings. It is consulted automatically (recall hook every
prompt; El Briefing at SessionStart) and stays fresh on its own (`memo
maintain` supersedes contradictions, merges duplicates, archives stale —
reversibly). The injected recall block is labelled authoritative, so treat
surfaced memorias as established facts: prefer them over assumptions, build on
them, and contradict one only explicitly.

Role split (resolve the "first place to look" overlap by role, not rivalry):

- **memo** — durable facts/decisions/preferences. Source of truth for *what is
  known*. Reads: recall hook, `memo_search`/`memo_ask`,
  `memo_unified_briefing`. Writes: `memo_save` (+ ambient capture on Stop).
- **Memflow** — live cross-agent/cross-machine working state and presence.
  Source of truth for *what is happening right now*, not durable knowledge.

Contract for any layer above memo (synapse, memflow, agents):
1. **Read first** — consult memo recall before deciding; `memo_unified_briefing`
   is the entry point.
2. **Write back** — persist durable facts to memo via `memo_save` so the next
   session inherits them.
3. **Respect freshness** — memo's contradiction/freshness state is authoritative
   for durable knowledge; don't reintroduce a fact memo has superseded.
4. **Identify yourself** — every read path is attributable, so `memo usefulness`
   proves who actually reads memo (a layer that never appears is a silent gap):
   - **MCP tools** — pass `source="<layer>"` on `memo_search` / `memo_ask` /
     `memo_chat_ask` / `memo_unified_briefing`. If omitted, `log_consult`
     falls back to `MEMO_SOURCE` then to the MCP client's handshake
     `clientInfo.name` — so agent clients (devin/opencode/windsurf) attribute
     automatically without per-call args.
   - **CLI** (synapse/memflow shell out) — `memo search/ask/chat-ask/recall`
     take `--source` or read `MEMO_SOURCE` env; a consult logs only when one is
     set (an interactive `memo search` stays out of the stats). `memo recall`
     emits `{"results":[...]}` for the memflow bridge.
   - **Warm socket** — the recall daemon's `{"op":"search","client":"<layer>"}`
     gives sub-second structured recall (no cold CLI) and attributes the client;
     this is how memflow reads memo without blowing its latency budget.

memo deliberately keeps cognition OFF its MCP surface
(`test_brain_like_mcp_tools_are_not_registered`): no `suggest`/`agent`/
`cognitive` verbs. Proactivity lives in memo's own recall/briefing output (the
"También en tu memoria" nudge), not as a brain tool — memo is the store, the
layer above is the cognition.

## CI gates

`pytest`, `mypy`, and coverage run per commit. Keep the suite green.

## Retrieval-regression discipline (every failed search → a system change, measured)

**Rule:** when a search returns wrong results, do NOT patch that one query. Make a
**systemic** change (ingest quality, ranking, a general gate) and prove it holds
across the whole regression set — never per-question.

- Growing committed corpus: `eval/regression_labels.json` (schema
  `memo.eval_recall.labels.v1`). Every incident adds a labeled prompt:
  `relevant=true` + `expect_ids` (the note that MUST surface, ≥8 hex prefix),
  and `noise_tags` / `noise_path_fragments` for records that must NOT crowd
  top-K (garbled OCR screenshots, archived/old notes).
- Gate (fast, no MLX — retrieval only, ~0.5s/prompt):
  `memo eval recall --labels eval/regression_labels.json --k 5 --force`.
  A retrieval/ingest change must keep **precision@K** high and **noise@K** low
  across ALL prompts. Baseline today: prec@5=0.2 (max for a single-answer
  prompt), noise@5=0.0. Runs against the live index (machine-local, not GitHub CI).
- Split of concerns: retrieval-class regressions (right note buried, garbage
  crowding) gate here; synthesis-class regressions (fabrication, refusal,
  wrong format) gate in synapse `eval-chat` with `require_substrings` /
  `forbid_substrings` checks.

## Workflows (Claude Code dynamic workflows)

Saved orchestration scripts in `.claude/workflows/`, invoked as `/`-commands:

- `/data-integrity-audit` — read-only health sweep over memo + memflow data (dup/near-empty chunks, stale entries, sync/heartbeat anomalies); reports exact fix commands, never mutates.
- `/demonolith-split [path]` — map a god-file and propose a clean in-repo package split (plan only). No arg → largest source file (e.g. `cli_capture.py`).

The trinity-wide `/trinity-green` and `/trinity-review` live in the synapse repo. Enable auto-orchestration for a session with `/effort ultracode`. List runs with `/workflows`.
