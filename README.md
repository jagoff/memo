<div align="center">

<img src="docs/banner.jpg" alt="memo — local memory for AI" width="100%" />

# memo — MCP semantic memory server with MLX embeddings

**Local-first semantic memory for AI agents — with time-travel, contradiction radar, and automatic synthesis.**

[![PyPI](https://img.shields.io/pypi/v/mlx-memo.svg)](https://pypi.org/project/mlx-memo/)
[![Downloads](https://static.pepy.tech/badge/mlx-memo)](https://pepy.tech/project/mlx-memo)
[![Python](https://img.shields.io/pypi/pyversions/mlx-memo.svg)](https://pypi.org/project/mlx-memo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-server-3b82f6.svg)](https://modelcontextprotocol.io)
[![Glama](https://img.shields.io/badge/Glama-A%20quality-22c55e)](https://glama.ai/mcp/servers/jagoff/memo)

</div>

## What is memo?

**memo** is an MCP server for semantic memory. It gives any AI agent (Claude Code, Devin, Cursor, Cline) a persistent, searchable knowledge base that:

- Runs **100% locally** (macOS + Apple Silicon via MLX, or Linux/CPU)
- Uses **hybrid search** (vector embeddings + full-text search)
- Stores memories as **plain Markdown** (version-controllable, human-readable)
- Syncs **across machines** via serverless git
- Detects & resolves **contradictions** automatically
- Time-travels to any date (audit trail)

No Ollama, no Qdrant, no cloud APIs, no keys.

<!-- mcp-name: io.github.jagoff/memo -->

`memo` gives any MCP-aware agent (Claude Code, Codex, Devin, OpenCode, Cursor, Cline, Continue, …) a long-term memory that **runs entirely on your own machine** — **macOS on Apple Silicon** via [Apple MLX](https://github.com/ml-explore/mlx), or **Linux / Ubuntu on a CPU `sentence-transformers` backend** (`pipx install "mlx-memo[cpu]"`, see [docs/ubuntu.md](docs/ubuntu.md)). Each memory is a plain Markdown file; embeddings live in a single sqlite file; the embedder, reranker, and LLM run in-process — no Ollama, no Qdrant, no cloud API, no keys. Your prompts and memories never leave the machine.

<div align="center">

<img src="docs/diagram-loop.svg" alt="Save a fact once and any later session recalls it automatically, all stored locally on your own machine." width="760" />

</div>

<div align="center">

<img src="docs/demo.gif" alt="memo in a terminal: save a fact once, then a later session recalls it automatically." width="760" />

</div>

## What makes memo different

| Capability | memo | mem0 | letta | cognee | engram | basic-memory | cipher |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100% local (no cloud API) | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ | ⚠️ |
| **Time-machine** (rewind corpus to any date) | ✅ | ❌ | ⚠️ | ❌ | ❌ | ⚠️ | ⚠️ |
| **Contradiction radar** (detect + resolve conflicts) | ✅ | ⚠️ | ⚠️ | ❌ | ⚠️ | ❌ | ❌ |
| **Synthesis pipeline** (auto-infer cross-cluster insights) | ✅ | ❌ | ✅ | ⚠️ | ❌ | ❌ | ⚠️ |
| **Cross-Mac git sync** (shared corpus, no server) | ✅ | ❌ | ⚠️ | ❌ | ✅ | ✅ | ⚠️ |
| Cloud sync (opt-in replication) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **TUI** (terminal UI) | ✅ | ❌ | ⚠️ | ❌ | ✅ | ❌ | ✅ |
| Obsidian as source-of-truth | ✅ | ❌ | ⚠️ | ❌ | ❌ | ✅ | ❌ |
| Knowledge graph + entity extraction | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ⚠️ |
| Eval regression gate (pre-commit wireable) | ✅ | ⚠️ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ |
| Multi-modal (images, audio OCR) | ✅ | ⚠️ | ⚠️ | ✅ | ❌ | ❌ | ❌ |
| MCP surface profiles (token economy) | ✅ | ❌ | ❌ | ⚠️ | ⚠️ | ✅ | ❌ |
| **Passive capture** (auto-extract from transcripts) | ✅ | ⚠️ | ✅ | ✅ | ✅ | ❌ | ⚠️ |
| Session timeline (context before/after) | ✅ | ❌ | ❌ | ❌ | ✅ | ⚠️ | ⚠️ |

<sub>✅ first-class · ⚠️ partial, config-gated, or add-on · ❌ absent. Verified mid-2026 against each project's docs/repo: [mem0](https://github.com/mem0ai/mem0), [letta](https://github.com/letta-ai/letta) (formerly MemGPT), [cognee](https://github.com/topoteretes/cognee), [engram](https://github.com/Gentleman-Programming/engram), [basic-memory](https://github.com/basicmachines-co/basic-memory), [cipher](https://github.com/campfirein/cipher). Closest comparators: **basic-memory** (local-first + Obsidian + MCP — memo's exact thesis) and **cipher** (memory layer for coding agents).</sub>

## Why it pays for itself — in tokens

memo is built to **spend fewer tokens, not more**.

- **~90% smaller MCP surface.** The default `agent` profile exposes **13 tools / ~1.2k schema tokens**, versus **124 tools / ~15k tokens** for the full surface — that overhead is paid *every session, in every client*. memo trims it to almost nothing.
- **Recall injects the answer instead of re-deriving it.** Ambient recall surfaces the top memory *before* the agent answers, on a tight **~160-token budget**. The agent stops re-explaining what it already figured out last week.

On a ~200-memory corpus, `memo roi` estimates **~80k tokens of model work avoided** per session. The number is corpus-specific; it grows as memo learns more.

`memo tokens` tracks the savings in real-time:

<img src="docs/tokens-screenshot.png" alt="memo tokens — showing 673k tokens saved all-time across 1924 memories used" width="760" />

| Technique | How to enable | Typical saving |
|---|---|---|
| Compact recall format | `export MEMO_RECALL_FORMAT=compact` | ~65% per injection |
| Trivial prompt gate | On by default | ~25% fewer injections |
| Context file compression | `memo compress-context CLAUDE.md` | 30–40% smaller context |

## Use cases

- **Continuity across sessions.** Decide "we use Postgres, not Mongo" today; tomorrow, in a fresh session, the agent recalls it on its own — recall injects the decision *before* it answers, so you never re-explain it.
- **Shared memory across agents.** Save something while working in Claude Code; Codex, Cursor, or Cline pick it up later. They all read the same local store over MCP.
- **Memory that follows you across Macs.** Start on the laptop, continue on the desktop. The corpus travels over serverless git sync and the agent starts with the same context on both.
- **Preferences and conventions that stick.** "Tests first", "commit messages in English", "don't touch the auth module" — say it once, the agent applies it every future session.
- **Contradiction radar.** Change your mind on an old decision and memo flags the now-stale version — the agent won't reintroduce what you already discarded.
- **Time-machine / audit.** "What did we know about this bug last month?" Rewind the corpus to any date and see the state of knowledge at that point.
- **Instant project onboarding.** A cold agent gets the project's durable decisions, facts, and preferences up front via the session-start briefing.
- **Fewer tokens, not more.** Instead of re-deriving what you solved last week, recall injects the answer on a tight budget — and the default MCP surface is 13 tools, not 124.

## Requirements

- **macOS on Apple Silicon** (M1–M4) — MLX is the load-bearing piece and the only path with the reranker + LLM features (ask / synthesize / dream).
- **Linux / Ubuntu / Intel Mac** — supported as a **standalone** install via a CPU `sentence-transformers` backend (search + recall + save, no MLX). One command: `pipx install "mlx-memo[cpu]"`. See **[docs/ubuntu.md](docs/ubuntu.md)** for what works and the trade-offs.
- **~8 GB** free disk for the default model set (the installer downloads it).
- *Optional:* an Obsidian vault. Without one, memo defaults to `~/Documents/memo/`.

> Python ≥ 3.13 is required if you install without uv. The `curl | bash` installer handles this automatically — it detects `uv` and uses its managed Python if no system Python ≥ 3.13 is on PATH.

## Install — one step

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash
```

The installer auto-detects **uv** (preferred) or falls back to **pipx**. It downloads MLX models, and wires memo into every agent client it finds (Claude Code, Codex, Devin, Devin Desktop, OpenCode).

Prefer a manual install? Any of these expose the same two binaries — `memo` (CLI) and `memo-mcp` (MCP server):

```bash
uv tool install mlx-memo          # recommended
pipx install mlx-memo
brew tap jagoff/memo && brew install mlx-memo
```

> Keep memo **isolated as its own tool** (uv tool / pipx / Homebrew). Don't vendor it inside another project's `.venv`. `memo doctor --strict-runtime` verifies the install.

**On Linux, or just want to try it without installing anything?** Run the Docker
image (CPU backend, cross-platform — search/recall/save; the reranker + `ask`/
`synthesize`/`dream` verbs are Apple-Silicon-only):

```bash
docker run --rm ghcr.io/jagoff/memo:latest memo doctor
```

Details in **[docs/docker.md](docs/docker.md)**.

First install downloads ~8 GB of MLX models (5–15 min); later installs hit the HuggingFace cache. Full installer knobs and "move to a new Mac" steps: **[docs/reference.md › Install](docs/reference.md#install-detail)**.

**Migrating from another Mac?** Install first, then restore your corpus:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash
memo sync bootstrap git@github.com:yourname/memo-sync.git   # restore from git
```

## Hand it to your agent

memo installs itself if you hand the repo (or just the install line) to an AI agent:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash
memo doctor --strict-runtime     # verify runtime is healthy
```

After install, tools surface as `mcp__memo__memo_*` (`memo_save`, `memo_search`, `memo_ask`, `memo_get`, `memo_graph`, `memo_unified_briefing`, plus capture/session/version helpers on the default profile). Per-client setup (Claude Desktop, Cursor, Cline, Continue, manual JSON) is in **[docs/reference.md › MCP setup](docs/reference.md#mcp-setup)**.

## Quick start

```bash
memo doctor                                            # self-check: models, vault, sqlite-vec
memo save 'MLX prefill ~30% faster than Ollama on M3 Max' --title 'MLX bench' -t mlx -t bench
memo search 'how fast was the MLX benchmark'           # search by meaning, not just keywords
memo list --limit 5                                    # most recent
memo ask 'what changed in the embedder this month?'   # RAG — cites memories by id
```

## Core features

- **Ambient recall** — every prompt silently consults memory and injects top hits as context. Warm recall daemon keeps it under **<200 ms**. No `/remember` calls.
- **Auto-capture** — a `Stop` hook extracts durable insights from each exchange through a quality gate. The corpus grows on its own.
- **Session briefing** — `SessionStart` surfaces open loops, a memory of the day, and one-line crash recovery.

## Key capabilities

### 🕰️ Time-machine

Rewind the corpus to any past date and query it as it was then:

```bash
memo as-of ask "what was the deployment strategy?" --date 2026-02-01
memo as-of search "redis config" --date 2026-01-15
memo diff --from 2026-01-01 --to 2026-03-01    # what changed
```

No other agent-memory system offers this. Full historical reconstruction via reverse-replay of `history.db`.

### ⚡ Contradiction radar

```bash
memo contradict scan                  # detect conflicting facts corpus-wide
memo contradict triage                # resolve interactively: fuse / newer-wins / dismiss
```

The LLM classifies each candidate pair. Results persist in `contradictions.db`; resolved conflicts inform future saves.

### 🔮 Synthesis & autonomous maintenance

#### Synthesis

```bash
memo synthesize                       # generate cross-cluster insights (LLM)
```

`MEMO_SYNTHESIS_ENABLED=1` runs synthesis automatically during `memo maintain`, creating type=`synthesis` memories from inferred cross-cluster patterns. Memories are cited with their source cluster.

#### Dream — 7-phase autonomous maintenance

`memo dream` runs nightly (03:00 AM by default) as a **fully autonomous pipeline** that synthesizes insights, resolves contradictions, decays old memories, and optimizes the corpus — with zero user intervention:

```bash
memo dream run                 # run once (foreground)
memo dream status              # show receipt + changes made
memo dream if-due              # no-op unless > 24h since last run (for cron)
```

The 7 phases: **(1) Orientation** (inventory corpus), **(2) Signal gather** (mine transcript labels), **(3) Heal** (resolve conflicts + consolidate duplicates), **(4) Prune** (archive 365d+ untouched, decay idle >30d), **(5) Enrich** (synthesize cross-cluster insights, extract entities), **(6) Optimize** (prune low-quality, evict if full, compress context), **(7) Prewarm** (cache top-100 query embeddings for <200ms recall next session).

Every phase logs to a **receipt** (`~/.memo-state/dream/last.json`): timestamp, duration, phase counts (resolved, consolidated, archived, synthesized, etc.), and any errors. Failures never silently vanish — they live in the receipt. A failed phase doesn't stop the pipeline; subsequent phases run anyway.

<div align="center">

<img src="docs/diagram-dream.svg" alt="memo dream: 7-phase pipeline with optional self-improvement (tuner/consolidate/anticipate)" width="100%" />

</div>

**Optional self-improvement** (all default OFF):

- **Tuner** (`MEMO_DREAM_TUNE_ENABLED=1`) — mines ground-truth labels from real usage, auto-tunes `MEMO_RECALL_MIN_SIM`, auto-reverts if eval baseline regresses. `memo dream tune`.
- **Consolidate** (`MEMO_DREAM_CONSOLIDATE_EPISODES_ENABLED=1`) — abstracts recurring cross-session work into synthesis memories. `memo dream consolidate`.
- **Anticipate** (`MEMO_DREAM_ANTICIPATE_ENABLED=1`) — surfaces unmet gaps + hot queries, pre-warms their embeddings. `memo dream anticipate`.

memo is the **only MCP memory system that self-optimizes**: auto-tuning recall quality, auto-detecting gaps, consolidating patterns — all from real usage, all nightly, all without asking.

### 🌐 Cross-Mac git sync

```bash
memo sync bootstrap git@github.com:yourname/memo-sync.git   # wire a shared corpus
memo sync once                                                # push/pull now
```

Pull-rebase-before-push. `flock`-based single owner per machine. Async debounced hooks keep the corpus current without blocking.

### 📚 Obsidian vault as source-of-truth

```bash
MEMO_MEMORIES_IN_VAULT=1 memo init                # store memories inside your vault
memo migrate --into-vault                          # non-destructive migration
```

Human edits in Obsidian win on the next `memo reindex`. The sqlite index is always rebuildable from the `.md` files.

### 🕸️ Knowledge graph

```bash
memo graph neighbors "MLX"             # what's related
memo graph path "embedder" "reranker"  # how two concepts connect
memo entities                          # list extracted entities
memo links --id abc123                 # backlinks + outlinks
```

Entity extraction uses a dependency-free regex backend. For code-heavy corpora, memo can merge a **[codegraph](https://github.com/colbymchenry/codegraph) symbol graph** as the graph's primary layer (opt-in, `MEMO_GRAPH_USE_CODEGRAPH`) — callers, callees, and imports become first-class edges, so recall and `memo graph path` reason over real code structure, not just text similarity. The merged graph also powers the `memo_graph` MCP tool and the entity-centric "Knowledge map" briefing.

### 🏥 Health scoring & eval gates

```bash
memo health                                         # grounded rate, ROI, usefulness verdict
memo eval recall --labels eval/regression_labels.json --k 5
memo eval recall --gate                             # exit non-zero if precision drops
memo eval recall --update-baseline                  # snapshot current best
```

Wire `--gate` into a pre-commit hook to catch retrieval regressions before they ship. `memo feedback` records per-source 👍/👎 votes that teach the retriever which memories to surface (or hide) for similar queries.

### 🖼️ Multi-modal ingestion

```bash
memo ocr-image screenshot.png                         # one-shot macOS Vision OCR
MEMO_VLM_CAPTION_ENABLED=1 memo ingest ~/Vault --include-orphan-images
memo ingest ~/Vault --include-audio                   # mlx-whisper, optional
memo search "whiteboard diagram"                      # finds OCR/caption/transcript text
```

The old placeholder `memo multimodal` store was removed; images and audio now
become searchable by flowing OCR, VLM captions, and whisper transcripts through
the normal text index.

### Daemons

memo runs four background daemons:

| Daemon | Command | Purpose |
|---|---|---|
| recall-daemon | `memo recall-daemon start` | Warm MLX embedder over socket (<200 ms recall) |
| idle-daemon | auto-started by `memo-mcp` | Auto-capture for MCP-only clients (Devin, OpenCode) |
| ingest-daemon | `memo ingest-daemon start` | Bulk vault ingestion |
| maint-daemon | `memo maint-daemon start` | Background cleanup + synthesis |

### All 108 visible CLI commands

<details>
<summary>Click to expand</summary>

**Core:** `save` `search` `ask` `get` `edit` `rename` `delete` `list`

**Recall & Hooks:** `recall` `recall-hook` `briefing` `continuity` `prewarm` `capture-tick` `capture-stop`

**Session & History:** `history` `as-of` `diff` `record-history` `session` `resume` `reflect` `mine-history` `episodes`

**Maintenance:** `reindex` `maintain` `dream` `consolidate` `synthesize` `dedupe` `cross-dedup` `retier` `contradict` `invalidate` `temporal` `compress-context`

**Analysis & Quality:** `health` `stats` `doctor` `lint` `analytics` `eval` `roi` `tokens` `token-savings` `usefulness` `gaps` `outcome` `profile`

**Knowledge Graph:** `graph` `entities` `entity` `extract-entities` `links` `version` `related`

**Advanced Search:** `embed` `rerank` `contextual` `chat` `chat-ask` `repo`

**Import / Export / Sync:** `import` `export` `backup` `restore` `sync` `ingest`

**Visualization:** `tui` `dashboard` `map` `logs` `hook-log`

**Setup & Config:** `init` `config` `install-mcp` `install-watcher` `uninstall-watcher` `install-slash` `install-statusline` `install-recall-hook` `install-shell-wrapper` `install-shims` `startup-banner` `migrate` `migrate-vault` `update` `watch` `release`

**Daemons:** `recall-daemon` `ingest-daemon` `maint-daemon` `embed-daemon` `idle-daemon`

**Other:** `backend-native` `collaborative` `feedback` `query` `mandate` `sleep-cycle` `ocr-image` `provenance` `mcp-command` `codex-badge` `debug-recall` `http-api` `mine-git` `token-gate`

</details>

### MCP surface profiles

| Profile | Tools | Schema tokens | Use when |
|---|---|---|---|
| `agent` (default) | 13 | ~1.2k | Standard agent work — max token economy |
| `core` / `slim` | 33 | ~2.8k | Constrained clients (Codex, OpenCode), admin-lite |
| `full` / `default` | 124 | ~15k | Power users, debugging |

Set via `MEMO_MCP_PROFILE=full` or in each client's MCP env config.

Non-MCP clients: `memo http-api` serves the same operations as a localhost REST API (plain JSON).

## Retrieval architecture

**Hybrid search:** vec leg (MLX embedding on Apple Silicon, CPU `sentence-transformers` on Linux) + BM25 leg (FTS5/Tantivy, diacritic-folding for Spanish) fused via Reciprocal Rank Fusion → optional MLX cross-encoder rerank (Apple Silicon).

<div align="center">

<img src="docs/diagram-recall.svg" alt="A prompt runs a vector search and a keyword search in parallel; results are fused, reranked, and the top memory is injected." width="820" />

</div>

**Markdown is the source of truth.** The `.md` files are canonical; sqlite is a rebuildable index. A hand-edit in Obsidian wins on the next `memo reindex`. `delete()` removes the index first, then the file — no silent data loss.

<div align="center">

<img src="docs/diagram-storage.svg" alt="Markdown files are canonical; the sqlite index is derived and can be rebuilt from them at any time." width="760" />

</div>

**Embedding models:**

| Model | Dims | Disk | Use |
|---|---|---|---|
| `Qwen3-Embedding-0.6B-4bit` | 1024 | ~0.6 GB | Default (fast, good) |
| `Qwen3-Embedding-4B-4bit` | 2560 | ~3 GB | Higher recall quality |
| `Qwen3-Embedding-8B-4bit` | 4096 | ~5 GB | Maximum quality |

Switch with `MEMO_EMBEDDER_MODEL` + `MEMO_EMBEDDER_DIMS` (requires `memo reindex --rebuild`).

## Documentation

| Topic | Where |
|---|---|
| Full install detail, installer knobs, new-Mac migration | [docs/reference.md › Install](docs/reference.md#install-detail) |
| Per-client MCP setup + the `/memo` slash command | [docs/reference.md › MCP setup](docs/reference.md#mcp-setup) |
| MCP profile tools and advanced domains | [docs/reference.md › MCP tools](docs/reference.md#mcp-tools) |
| Ambient memory, recall daemon, capture & recall tuning | [docs/reference.md › Ambient memory](docs/reference.md#ambient-memory) |
| Time-machine, session briefing, semantic map | [docs/reference.md › Surfaces](docs/reference.md#surfaces) |
| Full CLI reference + live dashboard (`memo tui`) | [docs/reference.md › CLI](docs/reference.md#cli-reference) |
| Stable/common `MEMO_*` flags, model profiles, upgrading the embedder | [docs/reference.md › Configuration](docs/reference.md#configuration) |
| Architecture, sync tiers, design notes | [docs/reference.md › Design & comparison](docs/reference.md#design-and-comparison) |

Contributors: `git clone https://github.com/jagoff/memo && cd memo && uv pip install -e '.[dev]'`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License & provenance

MIT — see [LICENSE](LICENSE). Forked philosophically from [`mem-vault`](https://github.com/jagoff/mem-vault) (storage layout + frontmatter schema); the MLX backend pieces are ported from [`obsidian-rag`](https://github.com/jagoff/rag-obsidian). memo is one of three sovereign systems in a wider stack ([Memflow](https://github.com/jagoff/memflow), Synapse) — the integration is opt-in everywhere; single-Mac users see zero behaviour change.
