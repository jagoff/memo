<!-- mcp-name: io.github.jagoff/memo -->
<!-- Banner: keep it, but make sure it loads fast (<200KB, WebP if possible) -->
[![memo — local memory for AI](https://raw.githubusercontent.com/jagoff/memo/master/docs/banner.webp)](https://raw.githubusercontent.com/jagoff/memo/master/docs/banner.webp)

# memo

**Your coding agent starts every session with amnesia. memo fixes that — 100% on your own machine.**

Persistent, searchable memory for Claude Code, Codex, Cursor, Cline, Devin, and OpenCode. No cloud, no API keys, no Ollama, no vector DB to run. And it spends *fewer* tokens, not more.

[![PyPI](https://img.shields.io/pypi/v/mlx-memo.svg)](https://pypi.org/project/mlx-memo/)
[![Downloads](https://static.pepy.tech/badge/mlx-memo)](https://pepy.tech/project/mlx-memo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jagoff/memo/blob/master/LICENSE)
[![MCP](https://img.shields.io/badge/MCP-server-3b82f6.svg)](https://modelcontextprotocol.io)
[![MCP Toplist](https://mcptoplist.com/badge/io.github.jagoff%2Fmemo.svg)](https://mcptoplist.com/server/io.github.jagoff%2Fmemo)

![Save a fact once — every later session recalls it automatically, all stored locally.](https://raw.githubusercontent.com/jagoff/memo/master/docs/demo.gif)

---

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/v4.13.2/install.sh | bash
```

<sub>Prefer a package manager? `uv tool install mlx-memo` · `pipx install mlx-memo` · `brew tap jagoff/memo && brew install mlx-memo`</sub>

Then:

```bash
memo doctor                                   # self-check
memo save 'we use Postgres, not Mongo'        # save a decision
memo search 'what database did we pick?'      # search by meaning
```

That's it. Your agents pick it up over MCP automatically — the installer wires every client it finds.

<details>
<summary>Installing on another Mac or handing setup to an agent?</summary>

New Mac:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/v4.13.2/install.sh | bash
memo sync bootstrap git@github.com:yourname/memo-sync.git
```

Agent-managed setup:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/v4.13.2/install.sh | bash
memo doctor --strict-runtime
```

</details>

**On Linux or just want to look around first?**

```bash
docker run --rm ghcr.io/jagoff/memo:latest memo doctor
```

---

## Why this saves you money

Most memory servers *add* context. memo is built to remove it.

| Profile | Tools | Schema tokens |
|---|---:|---:|
| `agent` (default) | 41 | ~9.4k |
| `core` / `slim` | 58 | ~12.9k |
| `full` / `default` | 164 | ~30.4k |

The default MCP surface is 41 tools, not 164 — 75% fewer tools, and about 69% less schema context: **41 tools / ~9.4k schema tokens** versus **164 tools / ~30.4k tokens** on the full surface — overhead paid every session, in every client.

Ambient recall injects one relevant memory before the model answers. The bundled Claude Code hook caps that injection at ~160 tokens. `memo roi` reads the real grounding and re-ask ledgers, then estimates accumulated savings with disclosed defaults (350 tokens per grounded recall and 900 per avoided re-ask).

```bash
memo roi       # value from grounded recalls and avoided re-asks
memo tokens    # usage-savings ledger
```

---

## Three things nothing else does

### 🕰️ Time-machine — query your knowledge as it was

```bash
memo as-of ask "what was the deploy strategy?" --date 2026-02-01
memo diff --from 2026-01-01 --to 2026-03-01
```

Full historical reconstruction by reverse-replaying `history.db`. Useful when you need to know *why* past-you made a call, not just what past-you decided.

### ⚡ Contradiction radar — memory that notices when you change your mind

```bash
memo contradict scan      # find conflicting facts corpus-wide
memo contradict triage    # resolve: fuse / newer-wins / dismiss
```

Change a decision and memo flags the now-stale version, so the agent stops reintroducing what you already threw out.

### 🔮 Dream — it optimizes itself while you sleep

```bash
memo dream run
```

A 7-phase nightly pipeline: inventory → mine signals → resolve conflicts → prune stale → synthesize cross-cluster insights → optimize → pre-warm the top-100 query embeddings so tomorrow's recall stays under 200 ms. Every run writes a receipt you can audit. Zero intervention.

---

## How it works

**Hybrid retrieval.** A vector leg (MLX on Apple Silicon, `sentence-transformers` on CPU) and a BM25 leg (FTS5, diacritic-folding for Spanish) run in parallel, fuse via Reciprocal Rank Fusion, then go through an optional MLX cross-encoder rerank.

![vector + keyword search in parallel, fused, reranked, top memory injected](https://raw.githubusercontent.com/jagoff/memo/master/docs/diagram-recall.svg)

**Markdown is the source of truth.** Every memory is a plain `.md` file you can read, grep, and version-control. SQLite is a derived index that rebuilds from the files at any time — hand-edit in Obsidian and your edit wins on the next `memo reindex`. Nothing is locked in a database you can't open.

**Prompts and memories stay on your machine.** Embedder, reranker, and LLM all run in-process. No telemetry. Memory travels only if *you* point `memo sync` at a git remote you own. Normal startup is fully offline; remote update checks and auto-update require an explicit opt-in. → **[Privacy and network policy](https://github.com/jagoff/memo/blob/master/PRIVACY.md)**

Also in the box: cross-agent `memo resume` (reopen any session from any agent), cross-Mac git sync, a knowledge graph with optional codegraph symbol edges, encrypted secret storage, OCR/audio ingestion, evidence packs, outcome learning, signed federation, and a local chat UI over your memory (`memo chat serve`). → **[Full feature reference](https://github.com/jagoff/memo/blob/master/docs/reference.md)**

---

## How it compares

Verified July 2026 against each project's own docs. Corrections welcome — [open an issue](https://github.com/jagoff/memo/issues) and I'll fix the table.

| | memo | [mem0](https://github.com/mem0ai/mem0) | [letta](https://github.com/letta-ai/letta) | [cognee](https://github.com/topoteretes/cognee) | [basic-memory](https://github.com/basicmachines-co/basic-memory) | [cipher](https://github.com/campfirein/cipher) |
|---|---|---|---|---|---|---|
| 100% local, no cloud API | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ |
| Time-machine (rewind to any date) | ✅ | ❌ | ⚠️ | ❌ | ⚠️ | ⚠️ |
| Contradiction detection + resolution | ✅ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ |
| Autonomous nightly maintenance | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Token-economy MCP profiles | ✅ | ❌ | ❌ | ⚠️ | ✅ | ❌ |
| Markdown / Obsidian as source of truth | ✅ | ❌ | ⚠️ | ❌ | ✅ | ❌ |

<sub>✅ first-class · ⚠️ partial, config-gated, or add-on · ❌ absent</sub>

Closest comparators are **basic-memory** (local-first + Obsidian + MCP — same thesis) and **cipher** (memory for coding agents).

---

## Requirements

| | Support |
|---|---|
| **macOS, Apple Silicon (M1–M4)** | Full — MLX embedder + reranker + `ask`/`synthesize`/`dream` |
| **Linux / Ubuntu** | Standalone CPU backend — search, recall, save. `pipx install "mlx-memo[cpu]"` · [docs/ubuntu.md](https://github.com/jagoff/memo/blob/master/docs/ubuntu.md) |
| **Intel Mac** | Unsupported — current PyTorch releases do not ship Python 3.13 wheels for this platform |
| **Docker** | Cross-platform, CPU backend · [docs/docker.md](https://github.com/jagoff/memo/blob/master/docs/docker.md) |

Python ≥ 3.13 (the installer handles this via `uv` if you don't have it). First install pulls ~8 GB of models, 5–15 min. Optional: an Obsidian vault — without one, memo uses `~/Documents/memo/`.

---

## Docs

| | |
|---|---|
| Install detail, installer knobs, new-Mac migration | [reference.md › Install](https://github.com/jagoff/memo/blob/master/docs/reference.md#install-detail) |
| Per-client MCP setup (Claude Desktop, Cursor, Cline, Continue) | [reference.md › MCP setup](https://github.com/jagoff/memo/blob/master/docs/reference.md#mcp-setup) |
| Ambient recall, capture, and tuning | [reference.md › Ambient memory](https://github.com/jagoff/memo/blob/master/docs/reference.md#ambient-memory) |
| Full CLI reference (144 commands) + `memo tui` | [reference.md › CLI](https://github.com/jagoff/memo/blob/master/docs/reference.md#cli-reference) |
| All `MEMO_*` flags and model profiles | [reference.md › Configuration](https://github.com/jagoff/memo/blob/master/docs/reference.md#configuration) |
| Architecture and design notes | [reference.md › Design](https://github.com/jagoff/memo/blob/master/docs/reference.md#design-and-comparison) |
| Privacy and network policy | [PRIVACY.md](https://github.com/jagoff/memo/blob/master/PRIVACY.md) |

### All 144 top-level CLI commands

<details>
<summary>Complete command inventory (kept here so CI detects CLI/documentation drift)</summary>

**Core:** `save` `search` `ask` `get` `edit` `rename` `delete` `list`

**Recall & Hooks:** `recall` `recall-hook` `context` `briefing` `continuity` `prewarm` `capture-tick` `capture-stop` `interject` `ask-gaps` `guard` `digest`

**Session & History:** `history` `as-of` `diff` `record-history` `session` `chat-session` `resume` `reflect` `mine-history` `episodes` `chronicle`

**Maintenance:** `reindex` `maintain` `review` `dream` `consolidate` `synthesize` `dedupe` `cross-dedup` `retier` `contradict` `coordinate` `terminal` `invalidate` `temporal` `compress-context` `ops`

**Analysis & Quality:** `health` `stats` `doctor` `journey-check` `lint` `drift` `analytics` `eval` `roi` `tokens` `token-savings` `usefulness` `gaps` `outcome` `profile` `confidence` `graduation` `hype` `definitive` `evidence`

**Knowledge Graph:** `graph` `entities` `entity` `extract-entities` `links` `version` `related`

**Advanced Search:** `embed` `rerank` `contextual` `retrieve` `context-pack` `chat` `chat-ask` `repo`

**Import / Export / Sync:** `import` `export` `backup` `restore` `sync` `ingest` `federation`

**Visualization:** `tui` `dashboard` `map` `logs` `hook-log`

**Setup & Config:** `init` `setup` `config` `install-mcp` `install-watcher` `uninstall-watcher` `install-slash` `install-statusline` `install-recall-hook` `install-shell-wrapper` `install-shims` `startup-banner` `migrate` `migrate-vault` `migrate-independence` `update` `upgrade` `self-update` `watch` `release` `onboard`

**Daemons:** `daemons` `recall-daemon` `ingest-daemon` `maint-daemon` `embed-daemon` `idle-daemon`

**Other:** `backend-native` `collaborative` `events` `feedback` `query` `mandate` `drift` `sleep-cycle` `operational` `ocr-image` `provenance` `secret` `verbatim` `mcp-command` `codex-badge` `debug-recall` `http-api` `mine-git` `token-gate` `fix` `undo` `code-facts` `code-nudge` `code-health`

</details>

---

## Contributing

```bash
git clone https://github.com/jagoff/memo && cd memo
uv pip install -e '.[dev]'
```

Issues and PRs welcome — see [CONTRIBUTING.md](https://github.com/jagoff/memo/blob/master/CONTRIBUTING.md). If memo is useful to you, a ⭐ genuinely helps other people find it.

MIT licensed. Built on [Apple MLX](https://github.com/ml-explore/mlx), [sqlite-vec](https://github.com/asg017/sqlite-vec), and [codegraph](https://github.com/colbymchenry/codegraph).
