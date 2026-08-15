# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

memo is a local-first semantic memory: MLX embeddings + sqlite-vec hybrid
(vec + BM25) search, reranking, a knowledge graph, temporal reasoning, an MCP
server, and a CLI. Single-user, runs offline. PyPI distribution name is `mlx-memo`; CLI binary is `memo`; MCP server binary is `memo-mcp`.

## Dev commands

```bash
# One-time setup (fresh clone/worktree — pytest is an optional-dependency
# extra, not a base dep, so plain `uv sync` alone leaves pytest missing)
uv sync --extra dev          # pytest, ruff, mypy, hypothesis, etc.
uv sync --all-extras         # + http/tantivy/multimodal/ocr, needed for those test modules

# Tests (no MLX needed — MLX tests skip automatically on non-Apple Silicon)
uv run --no-sync pytest tests/                          # full suite
uv run --no-sync pytest tests/test_foo.py::test_bar -v  # single test
uv run --no-sync pytest tests/ -m "not slow"             # skip real-MLX-load tests (>1s each)

# Type checking
uv run --no-sync mypy src/memo/

# Lint + format
uv run --no-sync ruff check src/ && uv run --no-sync ruff format src/

# Run the CLI
uv run --no-sync memo <cmd>

# Validate MEMO_* flags (typos, unknown vars)
uv run --no-sync memo config validate
```

If `memo doctor` warns `memo Python package loaded from .../src/memo/..., outside
the isolated runtime`, a stray `PYTHONPATH=src` (relative to a `~/repos/memo`
cwd) is shadowing the isolated `uv tool` install with this working tree's
source — `memo --version` looks right while the code that actually runs is
stale. `unset PYTHONPATH` before verifying a fix against the real installed
binary.

## Working tree is shared (read before any git op)

This repo is frequently worked on by **concurrent agent sessions sharing one
working tree and HEAD**. Their commits land on whatever branch is checked out,
master advances underneath you, and a `git checkout` in another session moves
*your* HEAD too. So:

- **Never `git add -A` / `git commit -a` / `ruff format src/`** — they sweep in
  other sessions' in-flight files. Stage explicit paths only:
  `git add src/memo/foo.py tests/test_foo.py`. Lint/format only your files.
- **Check `git reflog` + `git status` before any `reset`/`checkout`** — you may
  un-reference a concurrent commit or move a shared HEAD.
- **Cut releases from an isolated worktree**, never `git checkout master` in the
  shared tree: `git worktree add --detach /tmp/rel origin/master`, cherry-pick
  your commit, bump versions + CHANGELOG, commit, `git push origin HEAD:master`,
  tag, `git worktree remove --force`. Check `git ls-remote --tags origin vX.Y.Z`
  first — a concurrent session may have taken your version number.
  **`master` is GitHub-branch-protected** (10 required status checks, no direct
  push — verified: `git push origin HEAD:master` fails with `GH006: Protected
  branch update failed`) — land changes via `gh pr create` + merge, not a raw
  push, even from an isolated worktree with a clean fast-forward.
- Nothing is lost on entanglement (commits live in `git reflog`, uncommitted
  work stays in the tree). Rebuild a clean branch with
  `git checkout <branch> -- <only your files>` (zsh does NOT word-split `$VAR` —
  list paths explicitly).

## Architecture

**`Memory` facade** (`src/memo/memory/facade.py`) multiply-inherits thirteen operation mixins — `_WriteOpsMixin`, `_UpdateOpsMixin`, `_DeleteOpsMixin`, `_SearchOpsMixin`, `_SearchScoringMixin`, `_AskOpsMixin`, `_ChatAskOpsMixin`, `_RerankOpsMixin`, `_RepoOpsMixin`, `_MaintainOpsMixin`, `_ConsolidateOpsMixin`, `_ReplayOpsMixin`, `_SecretOpsMixin` — each in their own `src/memo/memory/<op>_ops.py` file. Module-level constants, prompts, and pure helpers are in `src/memo/memory/record.py`. Never import from a mixin directly; always go through `Memory`.

**MCP server** (`src/memo/server.py`) registers tools via `build_server()`. Each domain is a `server_<domain>.py` module that exports `register(server, memory)` — called once in `build_server()`. Adding a new MCP tool = create `src/memo/server_<domain>.py` + add one `register` call in `server.py`. The `_MaintainOpsMixin` method is directly callable from tests via `mock_memory.<method>()`.

**Storage** (`src/memo/store/`) is a subpackage. `VecStore` in `queries.py` is the primary interface: one sqlite-vec DB file, thread-local connections (required for FastMCP HTTP worker threadpool). Writes use `_tx()` (`BEGIN IMMEDIATE`); vectors are packed float32 blobs; WAL mode + `busy_timeout`.

**CLI** is in `src/memo/cli.py` (entry-point wiring only) + `src/memo/cli_<domain>.py` files. Each domain file exports a Click command or group imported and registered in `cli.py`.

**Flags** (`src/memo/flags.py`) is the single registry for all `MEMO_*` env vars — it aggregates `FlagSpec`s defined in the per-domain `flags_<group>.py` modules (`flags_recall/search/behavior/capture/ingest/misc.py`, base types in `flags_base.py`). Add a new flag in the matching `flags_<group>.py`. Use `flag_bool/int/float/str(name)` — never `os.environ.get("MEMO_...")` inline. `memo config validate` parses every set flag.

**Cache** (`src/memo/cache.py`): query embeddings use Memo's native
thread-safe LRU cache when `MEMO_QUERY_CACHE_SIZE > 0`. The stable path never
loads a cache implementation from another package.

**Contracts and replay URIs** (`src/memo/contracts.py`,
`src/memo/memory/replay_ops.py`): Memo owns its request/result contracts and
`memo://` parser. Legacy field names are migration aliases only.

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
  push stamps `state_dir/sync_pending` and retries next trigger. The repo also
  carries per-machine **embed-cache shards** (`embed_cache/<device_id>.json`,
  `sync_embed_cache.py`, `MEMO_SYNC_EMBED_CACHE` default on): document/chunk
  embeddings of the durable corpus, imported into `repo_embedding_cache` before
  the post-pull reindex — so memories saved on another Mac (and `memo sync
  bootstrap` on a fresh one) index with ~zero local MLX embed calls. Derived
  data only; reference/vault tier never exports.
- **Identity** (`identity.py`): stable `machine_id` (= persisted `cfg.device_id`,
  also `history.device_id`) + `hostname` + `session_id` + `terminal`. Commits are
  attributed `[<hostname>·<session>]`; Memo owns and exposes the identity.
- **Durable capture:** `memo capture-tick` (per-prompt async, self-throttled by
  `MEMO_CAPTURE_INTERVAL_S`) captures NEW turns since a per-session watermark so a
  long session's insight reaches `.md` before Stop, not only at Stop.
- `memo sync status` / `memo doctor` surface the silent no-op (not a clone) and
  stranded commits. The legacy audit-log replay (`sync.py` `SyncManager`) is the
  `--remote <path>` fallback; the machine `flock` is the coordinator.

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

## Chat — native RAG chat UI (`:8765`)

`memo chat serve [--port 8765] [--dist web-chat/dist]` serves the chat SPA
(vendored from the archived synapse chat) plus its API on `127.0.0.1:<port>`
(requires the `[http]` extra — FastAPI/uvicorn; raises a `ClickException`
otherwise, not an `ImportError` — `cli_chat.py` catches the missing-extra
`ImportError` and re-raises it as a clean CLI error). Pipeline:
`src/memo/chat/pipeline.py` orchestrates retrieval (`Memory.search` /
`Memory.repo_search`, called directly — not through `_ChatAskOpsMixin`) →
dedup/fusion/fulldoc/rewrite → synthesis, streamed as SSE events
(`stage`/`context`/`token`/`done`/`error`) over `POST /api/ask/stream`.
Feedback closes the loop (`src/memo/chat/feedback.py`): per-turn 👍/👎 and
per-source votes boost a source's rank on an exact question-key match, or
(semantic fallback) cosine similarity ≥ `MEMO_CHAT_SEMANTIC_THRESHOLD` between
the vote's query embedding and the new query.

Knobs (`src/memo/chat/config.py` — env-only + built-in defaults, read directly
via `os.environ`; NOT registered in `flags.py`'s markdown-config/tuned-overlay
chain, though the 9 names are excluded from `flags.unknown_memo_vars` so
`memo config validate` doesn't flag them as typos):
`MEMO_CHAT_BASE_K` (20, retrieval pool before dedup/rerank),
`MEMO_CHAT_RELEVANCE_FLOOR` (0.25), `MEMO_CHAT_VOTE_BOOST` (1.5, multiplier on
a 👍-voted source), `MEMO_CHAT_SEMANTIC_THRESHOLD` (0.75),
`MEMO_CHAT_MULTI_QUERY` (`true`) / `MEMO_CHAT_MULTI_QUERY_N` (2, query
rewrite/expansion), `MEMO_CHAT_FULLDOC` (`true`, inline the full doc for a
dominant source group), `MEMO_CHAT_ANSWER_MAX_TOKENS` (1200),
`MEMO_CHAT_SYNTH_HEAD` (8, sources actually fed to synthesis).

`memo eval chat` runs `eval/chat_regression_corpus.json` (schema
`synapse.eval_chat.query.v1`, rescued from synapse) through the pipeline and
checks pass/fail + p50/p95 latency — the chat-side counterpart to `memo eval
recall` (see Retrieval-regression discipline below).

**Latency is retrieval-dominated, and multi-query is the biggest term.**
Measured 2026-08-05 against the live corpus (11.3k memories, 4B embedder,
30B synthesiser): retrieval ~15-17s warm, ~70s on the service's first query
(the chat process cold-loads the LLM the query rewrite needs). `memo chat ask`
in a fresh process: 77s default, 31s with `MEMO_CHAT_MULTI_QUERY=false`. Turn
it off when latency matters more than recall breadth; keep it on for the UI,
where the SPA streams stages while retrieval runs.

**Ops (launchd):** `memo ops install chat [--port 8765] [--dist <path>]` /
`memo ops uninstall chat` / `memo ops status` (`src/memo/ops_launchd.py`)
render and bootstrap a `com.memo.chat` LaunchAgent (`KeepAlive`, logs to
`~/Library/Logs/memo/chat.log`). **Post-release install** uses the released
binary, not a worktree venv: `uv tool upgrade mlx-memo && memo ops install
chat --dist <dist>` — the plist's `ProgramArguments` point at whatever `memo`
resolves to in `PATH` at install time, so bootstrapping before the chat code
ships crash-loops under `KeepAlive`.

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
  fails, the memory is stamped `_memo_embed_pending` on disk and `memo reindex`
  replays it (the save never silently vanishes). `delete()` drops the derived
  index **first** and removes the canonical `.md` **last**, rolling the store
  back (`StorageError`) if the unlink fails — so the truth-bearing file is never
  lost to a partial delete (a leftover index row is recoverable via `reindex`).
  A hand-edit in Obsidian wins on the next `reindex` (body_hash mismatch → disk
  overwrites the index).
- **Rebuild, don't `rm`.** Use `memo reindex --rebuild` (not `rm memvec.db`) to
  rebuild from disk. It truncates only the markdown-derivable tables
  (`meta`/`vec`/`fts`) and preserves the **user-signal** tables — `access`,
  `memory_health`, `source_feedback*` — which are PRIMARY data not present in
  markdown and re-join on the stable memory `id`. A content-addressed embedding
  cache (`repo_embedding_cache`, keyed on `model+dims+sha256(text)`) makes a warm
  rebuild issue ~zero embedder calls.
- **Memories in the vault.** `MEMO_MEMORIES_IN_VAULT=1` (needs `MEMO_VAULT_PATH`)
  stores memories under `<vault>/<SYSTEM_DIR>/AI/memory` so the human-editable
  Obsidian vault is the source of truth. Ingest already excludes `AI/` and any
  `id:`-frontmatter file, so they're never double-ingested as reference tier.
  `memo migrate --into-vault` moves an existing install there (non-destructive,
  `--rollback` restores the prior config). `memo migrate` never drops `memvec.db`.
- **One DB file (opt-in).** `MEMO_SINGLE_DB=1` folds the sidecar stores
  (history/graph/contradictions/crossref/fact_edges) into `memvec.db` — each keeps its own
  connection to the one file (WAL allows it; no shared-transaction risk). Run
  `memo migrate --consolidate-db` once to merge existing `*.db` files (renames
  them `*.db.bak`, idempotent). Default off keeps the historical multi-file
  layout. The `cfg.*_db` path properties collapse onto `db_path` when on.
- **Per-project folders (default on).** `MEMO_STORE_BY_PROJECT=1` writes each new
  memory's `.md` into a per-project bucket — `memory_dir/<project>/` (or `_global/`
  when untagged) — derived from the `project:` tag (the tag stays the source of
  truth; the folder is derived). The sqlite index globs recursively, so search
  stays global — on-disk organization only. Recall then applies a 3-tier soft
  boost: `MEMO_RECALL_PROJECT_BOOST` (0.25, current project) > `MEMO_RECALL_GLOBAL_BOOST`
  (0.10, no project tag or `preference`/`feedback`) > other projects (+0).
  `memo migrate --bucket-by-project` re-buckets an existing flat install
  (non-destructive, idempotent, then reindexes). Backup/restore + git sync use
  `rglob`, so they preserve the bucket layout.

## Test isolation (see `tests/conftest.py`)

- Use the `tmp_cfg` fixture or build an isolated `Config` — never call
  `Config.from_env()` without controlling the environment.
- `CliRunner` invocations set `MEMO_NONINTERACTIVE=1`, `MEMO_DATA_DIR`, and
  `MEMO_STATE_DIR` in `env=` (conftest defaults `MEMO_NONINTERACTIVE=1`).
- Real MLX forward passes are gated by `@pytest.mark.requires_mlx` (auto-skipped
  when `mlx_lm` isn't importable).
- Never read or write the developer's real vault.
- Changing hooks, daemon lifecycle, install/runtime plumbing, or migration
  paths: run the broad suite plus the focused modules —
  `tests/test_recall_hooks.py`, `tests/test_recall_server.py`,
  `tests/test_runtime_isolation.py`.

## Config & errors

- `MEMO_*` behavioral flags live in `src/memo/flags.py` (registry + typed accessors). Storage/model config lives in `src/memo/config.py` (typed `Config` dataclass). Prefer `flag_bool/int/float/str` over raw `os.environ`. `memo config validate` catches typos.
- **User-facing config is the persistent Markdown config** (`src/memo/config_md.py`): `memo config` / `memo config set <key> <value>` (dotted keys, e.g. `recall.top_k`) writes `~/.config/memo/*-config.md` — reaches daemons/MCP/hooks, unlike a per-terminal `export MEMO_*`. Flag resolution: **env var > markdown config > tuned overlay > built-in default**.
- Domain errors live in `src/memo/errors.py` (`MemoError` base). Raise/catch
  those rather than bare `Exception` in non-defensive code.

## Releasing

Bump the version in sync across **five** source-of-truth files:
`pyproject.toml` `[project].version`, `.claude-plugin/plugin.json`,
`plugins/memo/.codex-plugin/plugin.json`, `server.json` (version + package
version), and `CHANGELOG.md` (Keep-a-Changelog) — `memo release bump` edits
all five. Commit / tag / push stays manual.

`memo release check` is the gate. It validates version parity across every
versioned surface (both plugin manifests, `server.json` incl. extra packages,
the mcpb manifests + archive, install pins, the Homebrew formula, the CHANGELOG
section) **and** — via `src/memo/adapter_matrix.py` — three surfaces that carry
no version and whose drift is *silent*:

| check | catches |
|---|---|
| `hook-commands-resolve` | `hooks/hooks.json` firing a `memo` subcommand the CLI no longer registers. Hooks are soft-fail by design, so a rename stops recall/capture/sync with **no error anywhere** |
| `embedder-dims-parity` | an `.mcp.json` pinning `MEMO_EMBEDDER_MODEL` whose `MEMO_EMBEDDER_DIMS` doesn't match the model size (MLX invariant 3 — corrupts the vec0 table on first write). A config pinning *neither* stays legal: the installed index is self-describing |
| `referenced-paths-exist` | the codex manifest's `mcpServers` path or the marketplace `source` pointing at nothing |

`tests/test_adapter_matrix.py` mutates one surface per test and asserts the
check flips to fail — a gate that only ever passes proves nothing. Version
parity deliberately lives only in `cli_release`; a second, weaker opinion that
can disagree with the real gate is worse than none.

## Source of truth — role & contract

memo is the canonical store of **durable semantic knowledge**: decisions,
facts, preferences, learnings. It is consulted automatically (recall hook every
prompt; El Briefing at SessionStart) and stays fresh on its own (`memo
maintain` supersedes contradictions, merges duplicates, archives stale —
reversibly). The injected recall block is labelled authoritative, so treat
surfaced memories as established facts: prefer them over assumptions, build on
them, and contradict one only explicitly.

Memo is the source of truth for both durable knowledge and the operational
continuity needed by agents. Durable facts, decisions, preferences, focus,
handoffs, attention, conflicts, and outcome feedback all live behind Memo-owned
CLI/MCP contracts.

Contract for any agent using memo:
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
     `clientInfo.name` — so agent clients (devin/opencode/devin-desktop) attribute
     automatically without per-call args.
   - **CLI** — `memo search/ask/chat-ask/recall` take `--source` or read
     `MEMO_SOURCE`; a consult logs only when one is set (an interactive
     `memo search` stays out of the stats).
   - **Warm socket** — the recall daemon's `{"op":"search","client":"<layer>"}`
     gives sub-second structured recall (no cold CLI) and attributes the client.

memo deliberately keeps cognition OFF its MCP surface
(`test_brain_like_mcp_tools_are_not_registered`): no `suggest`/`agent`/
`cognitive` verbs. Proactivity lives in memo's own recall/briefing output (the
"Also in your memory" nudge), not as a brain tool — memo is the store, the
layer above is the cognition.

## CI gates

`pytest`, `mypy`, `ruff` (on `src/` **and** `tests/`), and coverage run per
commit. Keep the suite green.

The `runtime-independence` job proves the installed distribution does not
resolve retired sibling modules or private contract packages. The definitive
benchmark also gates journal integrity and native event throughput.

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
  across ALL prompts (the label set grows with every incident — 37 prompts as of
  2026-07; the authoritative numbers are the per-machine saved baseline below,
  not any figure hard-coded here). Runs against the live index (machine-local,
  not GitHub CI).
- **Enforced gate (machine-local, opt-in):** seed once with
  `memo eval recall --labels eval/regression_labels.json --k 5 --update-baseline`,
  then `--gate` (instead of printing) exits non-zero if precision dropped or
  noise rose vs the saved baseline — wire it into a pre-commit hook. The
  baseline lives under `state_dir/eval/` (per-machine; it tracks *your* corpus).
- Split of concerns: retrieval-class regressions (right note buried, garbage
  crowding) gate here; end-to-end grounding, abstention, and integrity gate in
  `memo definitive check` plus the relevant Memo eval command.

### Behavior eval — did the recalled memory actually steer the answer?

`memo eval recall` stops at "the right note surfaced". A memory can be top-1 and
still be ignored, re-litigated, or answered around, with every retrieval metric
green. `memo eval behavior` (`src/memo/eval_behavior.py`, corpus
`eval/behavior_scenarios.json`, schema `memo.eval_behavior.scenario.v1`) covers
the step after retrieval:

1. seeds an isolated store with the scenario's memories (real store, real
   embeddings — nothing mocked),
2. runs the **real `memo recall-hook` as a subprocess** against it and takes its
   `additionalContext` verbatim (no ranking is reimplemented, so the harness
   cannot drift from the hook),
3. feeds `[injected block + prompt]` to a model and scores the scenario's gates.

Gates are two-layered: `must_recall` / `must_not_recall` (deterministic, from
the block) and `answer_must_contain_any` / `answer_must_not_contain_any` /
`semantic` (from the answer). **A scenario with only recall-layer gates is
rejected at load** — that is retrieval, already covered above.

```bash
memo eval behavior --recall-only          # deterministic, no LLM loaded
memo eval behavior --gate                 # full, exits non-zero on any failure
memo eval behavior --only <scenario-id> --json
```

**Read a red gate correctly.** The model in step 3 is memo's own local LLM, not
Claude — a failure proves *the injected payload under-steers a competent model*
(block formatting, labelling, ordering, truncation, token budget: all memo-side),
never that a specific agent misbehaved. The answerer and judge both default to
`MEMO_HELPER_MODEL`, never the 30B generation model: the embedder is already
resident during a run, and stacking a 30B answerer + 30B judge on top is the
residency mix that has OOM'd this machine before.

## Dream — nightly self-maintenance + self-improvement

`memo dream run` is the nightly pipeline (LaunchAgent `com.memo.dream` at 03:00;
`memo dream if-due` for the >24h guard). It is **separate from `memo maintain`**.
Wiring is `src/memo/cli_dream.py` (Click) + `src/memo/cli_dream_passes.py`
(`_run_*` implementations); the receipt persists to `state_dir/dream/last.json`
(`memo dream status`). Every pass records its result there and **failures land in
`receipt["errors"]` — never silently swallowed**.

- **Maintenance passes (default-on):** orientation inventory → signal-gather
  (transcript mining → memories) → contradictions (supersede ≥0.9) → consolidate
  duplicates → archive stale → `synthesize_cross_cluster` → entities → ROI decay
  → prune-floor / LFU-evict → compress → prewarm.
- **Dream v2 — self-improvement passes (all OFF by default, opt-in per flag):**
  - **Tuner** (`dream_tune.py`, `MEMO_DREAM_TUNE_ENABLED`): mines
    ground-truth-by-use labels from `grounding.log` (reuses
    `eval_recall.harvest_labels`), line-searches `MEMO_RECALL_MIN_SIM` over the
    live index, and **auto-applies** the winner — gated by the curated
    `eval/regression_labels.json` (a change that helps mined labels but hurts the
    curated set is rejected) and **auto-reverted** when a later night regresses
    vs the saved baseline. CLI `memo dream tune`. Scope: it only moves `min_sim`
    — the one knob the *tuner* line-searches today. The blocker this used to
    name is gone: `rank_hits()` is already public and pure
    (`recall_logic.py:1388`) and is the shared ranking path for the recall hook,
    the eval harness and the A/B seam, so boosts and rerank-pool are measurable
    now with `memo eval recall`. Widening the tuner to them is unstarted work,
    not a missing extraction.
  - **Consolidate** (`dream_consolidate.py`,
    `MEMO_DREAM_CONSOLIDATE_EPISODES_ENABLED`): groups recent episodes
    (`EpisodeStore.recent`) by project and abstracts recurring cross-session work
    into one durable `type=synthesis` memory (`synthesis_kind=cross_session`,
    session ids as provenance, dedup by provenance hash). Distinct from per-turn
    signal-gather and cross-memory synthesize; no destructive episodic decay.
  - **Anticipate** (`dream_anticipate.py`, `MEMO_DREAM_ANTICIPATE_ENABLED`):
    surfaces recurring unmet gaps (`outcome.detect_gaps`) + hot queries into the
    receipt/briefing and pre-warms their embeddings. Never fabricates.
  - **Flag graduation** (`dream_flags.py`, `MEMO_DREAM_FLAG_GRADUATION_ENABLED`):
    every default-off `*_ENABLED` flag declares a gate in `dream_flags.GATES`
    (`recall` A/B via the eval `flag_overrides` seam / `tuner`-owned / `manual`
    + reason) — completeness CI-enforced by `test_dream_flags.py`, so a new
    dark flag cannot merge without declaring its gate. Winners graduate to ON
    via the overlay after `MEMO_FLAG_GRADUATION_WIN_NIGHTS` consecutive wins
    (latency + curated gates), auto-revert on regression; flags un-graduated
    past `MEMO_FLAG_GRADUATION_DEADLINE_DAYS` become cull candidates in
    `memo dream graduate-flags --status` (deletion stays human). Distinct from
    `dream_graduate.py` (quarantined-memory graduation).

**Tuned-params overlay** (`src/memo/tuned_overlay.py`) is the only place
auto-tuning touches live behavior: the tuner writes `state_dir/tuned_params.json`
and `flags.flag()` consults it, so flag resolution is **env var > markdown
config > overlay > built-in default** (an explicit `MEMO_*` env var or
`memo config set` value is never overridden). Delete the
file or `memo dream tune --rollback` to revert. The enable flags live in the
nightly LaunchAgent's `EnvironmentVariables` (launchd does not inherit the shell)
— see `~/repos/memo/launchd/com.memo.dream.plist`.

## Codegraph-first (code intelligence)

The repo is indexed by codegraph (`.codegraph/codegraph.db`, kept fresh by the
MCP watcher + git hooks + nightly sync). Use it INSTEAD of the grep+Read loop:

- **Understand a flow / architecture / bug:** ONE `codegraph_explore` call with
  2-4 **exact symbol/file names** (a bag of names, never prose — retrieval is
  lexical FTS5, no embeddings). Don't know the name? `codegraph_search` first.
- **Treat explore output as Read-equivalent:** don't re-verify with grep, don't
  delegate to file-reading subagents.
- **Before `codegraph_callers`/`codegraph_impact`:** confirm the symbol EXISTS
  with `codegraph_search` — a non-existent name silently substitutes the best
  fuzzy match (upstream #1473).
- **After editing:** a ⚠️ staleness banner means re-read with Read; in doubt,
  `codegraph_status` and check `pendingChanges`.
- **Do NOT trust the "⚠️ no covering tests" blast-radius note** (~40% false,
  upstream #1475) — verify against `tests/` with grep.
- **Overloaded names** (`search`, `save`, `recall`): pin with file/line via
  `codegraph_node`.
- **Cross-repo (memflow/synapse):** pass `projectPath` — graphs are per-repo,
  no cross-repo edges. Caveat: synapse consumes memo via subprocess/CLI, so
  0 cross-repo callers does NOT prove a CLI surface is unused.
- **Pre-refactor gate:** run `scripts/cg_impact_gate.sh <symbol>` before
  renaming or changing signatures under `src/memo/`. Success criterion for a
  rename: `codegraph_callers` on the OLD name returns 0.
- **`codegraph affected` is BROKEN for this repo's src-layout** (returns 0
  tests) — use `scripts/cg_affected_tests.sh` (SQL over the DB) instead.

### Review with codegraph

Per file in the diff: `codegraph_node` on the file → its dependents are the
review checklist (each caller appears in the diff or is justified). Per renamed
symbol: `codegraph_callers <old-name>` must return 0. The CLI's
`node --symbols-only` output truncates dependents with `+N more` — when that
marker appears, expand via the `codegraph_node` MCP tool, never trust the
truncated list as complete.

## Workflows (Claude Code dynamic workflows)

Saved orchestration scripts in `.claude/workflows/`, invoked as `/`-commands:

- `/demonolith-split [path]` — map a god-file and propose a clean in-repo package split (plan only). No arg → largest source file (e.g. `cli_capture.py`).

Use `/effort ultracode` when a change benefits from parallel in-repo
implementation and review.

<!-- memo-mandate -->
## Memory-first (memo)

Before deciding or answering anything that prior work might already cover,
consult memo FIRST:
- Start with `memo_unified_briefing` (or `memo_search` / `memo_ask`) to
  pull durable facts, decisions, and preferences.
- Already hold a memory id or a named entity and want what's connected to
  it? Reach for `memo_graph` / `memo_related` (cheap graph traversal —
  id/title only) before firing a fresh `memo_search`/`memo_ask` (full
  retrieval) for the same thing.
- Pass `source="<this-client>"` on the read tools so usage is attributed
  (e.g. `source="codex"`). A client that never appears in memo's consult log is
  flagged as a silent gap by `memo usefulness`.
- Write durable outcomes back with `memo_save` so the next session inherits
  them. memo is the source of truth for what is *known*; build on it, and
  contradict a surfaced memory only explicitly.
- Keep it honest: when a surfaced memory is stale or contradicted, correct it
  instead of silently working around it — `memo_feedback_flag(kind="outdated")`
  to retire it, or `kind="wrong"` (with `superseded_by` when a replacement
  exists). Both archive reversibly, never hard-delete.
<!-- /memo-mandate -->
