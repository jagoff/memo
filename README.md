<div align="center">

# memo

**Persistent semantic memory for AI agents — 100% local, MLX-native, Apple Silicon.**

[![PyPI](https://img.shields.io/pypi/v/mlx-memo.svg)](https://pypi.org/project/mlx-memo/)
[![Python](https://img.shields.io/pypi/pyversions/mlx-memo.svg)](https://pypi.org/project/mlx-memo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-server-3b82f6.svg)](https://modelcontextprotocol.io)

</div>

<!-- mcp-name: io.github.jagoff/memo -->

`memo` gives any MCP-aware agent (Claude Code, Claude Desktop, Cursor, Cline, Continue, Paperclip, …) a long-term memory that **runs entirely on your Mac**. It stores each memory as a plain Markdown file inside an Obsidian-friendly folder, indexes embeddings in a single sqlite file, and runs the LLM + embedder + reranker **in-process via [Apple MLX](https://github.com/ml-explore/mlx)** — no Ollama, no Qdrant, no cloud API, no keys.

> Your prompts and memorias never leave the machine.

---

## What it does

- **Saves what your agent decides, learns, prefers** as durable Markdown files (`type`, `tags`, `title`, body).
- **Recalls** the most relevant memorias when you ask — semantic (vec), keyword (BM25), or hybrid w/ cross-encoder rerank.
- **Injects context automatically**: with the optional Claude Code plugin, every prompt silently consults memory; the agent sees the top-3 memorias *before* answering.
- **Speaks MCP** over stdio so any compliant client picks it up with one line of config.
- **Speaks shell** too: the same API ships as a `memo` CLI with ~25 commands.

## 🕰️ The unique feature: **time-machine**

memo is **the only agent-memory product that lets you rewind the corpus to any past date.** Every other store on the market (mem0, letta, cognee, supermemory, mem-vault, milasd/memo-mcp, doggybee, engram) serves *current* state only.

```bash
# What did I think about MLX vs Ollama three months ago?
memo as-of ask "MLX vs Ollama" --date 2026-02-01

# What changed in my decisions between releases?
memo diff --from 2026-03-01 --to 2026-04-30

# Search the corpus as it stood on a specific Monday
memo as-of search "auth middleware" --date 2026-03-15
```

Under the hood: `history.db` is an append-only audit log of every save/update/delete. A snapshot at any `T` is built by replaying events in reverse from "now". See [docs/time-machine.svg](docs/time-machine.svg) for the algorithm at a glance.

![time-machine algorithm](docs/time-machine.svg)

**Why this matters:**

- **Debug agent regressions.** "Claude gives a different answer now — which memoria I added last week broke it?" → `memo as-of ask "..." --date <before>` vs `--date <after>`.
- **Reproducible AI behavior.** Mount a snapshot as an alternate MCP and serve it to the agent so you can reproduce a past decision deterministically.
- **Personal audit.** "Did I already have this preference on 2026-03-01?" answered definitively from the audit log.
- **Compliance.** "What did the model know when it took action X?" — reconstruct the exact memory state at time T.

## Why memo

| Pain | What memo gives you |
|---|---|
| Cloud memory products see your private notes | **Zero network in the hot path.** Models run in-process. |
| Ollama / Qdrant / docker daemons just to remember things | **One Python install.** sqlite-vec is one file; MLX is in-process. |
| DB-only stores lock your knowledge inside an opaque blob | **Markdown is the source of truth.** Edit in Obsidian, vim, anything. |
| Cold-start latencies of 2-10s per recall | **MLX prewarm hook** → sub-second recalls after session start. |
| Hand-crafted `/remember` invocations every turn | **Ambient recall**: top-3 hits auto-injected into every prompt. |
| **No way to query past corpus state** | **Time-machine**: snapshot the corpus at any past date (see above). |
| Vendor lock | **MIT package, open stack** (sqlite-vec Apache 2.0, MLX MIT, Qwen Apache 2.0). |

## How it fits in your stack

![memo architecture](docs/architecture.svg)

Three layers, one direction of data flow:

1. **Clients** (Claude Code, Cursor, …) talk to memo over **MCP stdio** — or you talk to it directly via the **`memo` CLI**.
2. The **Memory API** runs save / search / rerank / ask against the **MLX models in-process**: embedder for semantic, optional reranker for precision, chat (Qwen2.5-7B) for `ask()`.
3. The **`.md` vault** is the storage of record; **`sqlite-vec`** is a rebuildable index. Delete the index any time — `memo reindex` rebuilds from the `.md` files.

With the Claude Code plugin installed, two extra hooks plug in:

- `SessionStart` → `memo prewarm` (warms MLX so the first recall is fast)
- `UserPromptSubmit` → `memo recall-hook` (5s budget, injects top-3 memorias as `additionalContext`)

## Stack

| Component | Choice | Why |
|---|---|---|
| LLM (chat) | [`Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) + [`3B helper`](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit) via [`mlx-lm`](https://github.com/ml-explore/mlx-lm) | Two-tier; 7B for `ask()` synthesis, 3B for cheap helpers. Both 4-bit fit comfortably. |
| Embedder | [`Qwen3-Embedding-0.6B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) | 1024-dim, ~50ms/embed, ~600 MB on disk. Upgrade path to 4B/8B is one env var. |
| Reranker | [`Qwen3-Reranker-0.6B`](https://huggingface.co/mlx-community/Qwen3-Reranker-0.6B) (optional) | Cross-encoder over top-30 from vec+BM25, then α-fusion. Bumps precision on diffuse queries. |
| Vector store | [`sqlite-vec`](https://github.com/asg017/sqlite-vec) | One file, no daemon, embedded. Reset = `rm memvec.db`. |
| Source of truth | Markdown files under `<vault>/...` with YAML frontmatter | Human-editable, syncs through iCloud/git/Syncthing/whatever. |
| MCP transport | [`fastmcp`](https://github.com/jlowin/fastmcp) | Stdio out of the box. |

## Requirements

- **macOS on Apple Silicon** (M1 / M2 / M3 / M4). MLX is the load-bearing piece.
- **Python ≥ 3.13**.
- **~4 GB** free disk for the default model set (downloaded on first use).
- *Optional:* an Obsidian vault. If you don't have one, memo defaults to `~/Documents/memo/` and creates the folder for you.

## Install

```bash
pipx install mlx-memo
# or
uv tool install mlx-memo
# or via the Homebrew tap
brew tap jagoff/memo && brew install mlx-memo
# or, only if you intentionally want it in the active Python env
pip install mlx-memo
```

Any of those expose two binaries: `memo` (CLI) and `memo-mcp` (MCP server).
For MCP clients, prefer an isolated tool install (`pipx`, `uv tool`, or
Homebrew) instead of installing into another project's `.venv`; that keeps
memo's MLX dependencies, sqlite state, and `memo-mcp` runtime independent
from whichever repo happens to be active in your shell.

> The PyPI distribution is **`mlx-memo`** as of 0.5.0. Earlier
> versions shipped as `memo-mcp` and the binary names haven't
> changed — existing MCP configs keep working.

Pre-download the MLX models so the first save/search doesn't stall on a multi-GB download:

```bash
hf download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
hf download mlx-community/Qwen2.5-3B-Instruct-4bit
hf download mlx-community/Qwen2.5-7B-Instruct-4bit
```

### Dev install (contributors)

```bash
git clone https://github.com/jagoff/memo
cd memo
uv pip install -e '.[dev]'
```

## Quick start

```bash
# Self-check (validates models, vault path, sqlite-vec)
memo doctor

# Save a memory
memo save 'Bench MLX vs Ollama: ~30% faster prefill on M3 Max' \
  --title 'MLX bench result' -t bench -t mlx

# Search by meaning (not just keywords)
memo search 'cuál fue el resultado del bench MLX'

# Recent
memo list --limit 5

# RAG — ask a question, memo cites memorias by id
memo ask 'qué cambios hice en el embedder este mes?'
```

## MCP setup

After `pip install mlx-memo`, register the MCP with your client.

### Claude Code

```bash
memo mcp-command
# then run the printed command, e.g.
claude mcp add memo -s user /Users/you/.local/pipx/venvs/mlx-memo/bin/memo-mcp
```

Or hand-edit `~/.claude.json`:

```jsonc
{
  "mcpServers": {
    "memo": {
      "type": "stdio",
      "command": "/path/to/memo-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

Restart Claude Code. Tools surface as `mcp__memo__memory_*` inside the agent.
If Claude starts the wrong server, run `memo doctor --strict-runtime`; it
will warn when `memo`/`memo-mcp` resolve from a project-local venv or from
different environments.

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```jsonc
{
  "mcpServers": {
    "memo": { "command": "/path/to/memo-mcp" }
  }
}
```

### Cursor / Cline / Continue

Each client has its own MCP config UI but the contract is the same: register a stdio server pointing at the `memo-mcp` binary.

### Paperclip

A first-party plugin under [`integrations/paperclip-plugin-memo/`](./integrations/paperclip-plugin-memo) exposes five tools (`memo_search`, `memo_save`, `memo_list`, `memo_get`, `memo_ask`) to any agent running in a Paperclip company.

## Tools exposed over MCP

| Tool | What it does |
|---|---|
| `memory_save(content, title?, type?, tags?)` | Persist a new memory; returns the full record. |
| `memory_search(query, limit?, type?, body_chars=280, mode="hybrid")` | Top-k. `hybrid` (default) fuses vec + bm25 via RRF, then optionally re-ranks. `vec` is semantic only; `bm25` is keyword (FTS5 unicode61, diacritic-stripping for Spanish). |
| `memory_list(limit?, type?)` | Recent by `updated` desc. |
| `memory_get(id)` | Full record. Accepts a unique prefix ≥4 chars (git-style); returns `{"error": "ambiguous", "matches": [...]}` on collision. |
| `memory_update(id, title?, type?, tags?, content?)` | Patches fields; re-embeds only if body changed. |
| `memory_reindex()` | Re-scan vault, re-embed entries whose `body_hash` diverged. |
| `memory_delete(id)` | Removes from vec + disk. |
| `memory_ask(question)` | RAG synthesis; cites memorias by id. |
| `memory_stats()` | Counts, paths, active models. |
| `memory_consolidate()`, `memory_extract_entities()`, `memory_entities()`, `memory_history()` | Post-v0 endpoints — see CHANGELOG. |

## Ambient memory (v0.3.0+) — recall without `/memo`

Install the bundled [Claude Code plugin](#slash-command--memo-claude-code-only) and memo silently consults your past on every prompt and injects the most relevant memorias as `additionalContext` — **the agent sees them before answering**, no manual invocation.

### How it works

- `SessionStart` hook → `memo prewarm` (async) — pre-loads the MLX embedder so the first recall is fast.
- `UserPromptSubmit` hook → `memo recall-hook` (5s timeout) — embeds your prompt, runs vec-only search, returns top-3 memorias above cosine 0.6.

Both run 100% local. Your prompt never leaves the machine.

### Tuning

| Env var | Default | Purpose |
|---|---|---|
| `MEMO_RECALL_DISABLE` | unset | Set to `1` to skip recall entirely |
| `MEMO_RECALL_TOP_K` | `3` | Max memorias to inject |
| `MEMO_RECALL_MIN_SIM` | `0.6` | Cosine similarity floor |
| `MEMO_RECALL_MIN_PROMPT_CHARS` | `12` | Skip very short prompts |
| `MEMO_RECALL_BODY_CHARS` | `240` | Snippet length per memoria |
| `MEMO_RECALL_SKIP_SLASH` | `1` | Skip recall on `/` prompts |
| `MEMO_RECALL_TOKEN_BUDGET` | `0` | When > 0, pack memorias greedily until ~N tokens; truncate tail to fit |
| `MEMO_RECALL_PROJECT_BOOST` | `0.15` | Additive score boost for memorias whose tags match the current project tag |
| `MEMO_RECALL_DEBUG` | unset | Print failure reasons to stderr |

### Empirical tuning of `MIN_SIM=0.6`

On a 223-doc corpus:
- `qué decidí sobre MLX vs Ollama` → 3 hits at 0.71–0.74 (relevant ✓)
- `how to bake apple pie` (no food memorias) → 0 hits at 0.6 ✓ (3 noise hits at 0.51–0.56 cut by the floor)

Tune lower (0.5) on sparse corpora, higher (0.7) for high-precision only.

## Slash command — `/memo` (Claude Code only)

A Claude Code [skill](https://docs.claude.com/en/docs/claude-code/skills) ships at `skills/memo/SKILL.md`. Symlink it:

```bash
ln -s "$(pwd)/skills/memo/SKILL.md" ~/.claude/skills/memo/SKILL.md
```

Or install everything (skill + MCP config + hooks) in one shot via the bundled plugin:

```bash
/plugin install memo@jagoff/memo
```

The skill routes user input to the right MCP tool:

| Input | Action |
|---|---|
| `/memo <query>` | semantic search (k=5, snippet body) |
| `/memo` | smart capture — destila el insight del turno y guarda |
| `/memo list [n]` | recent memories |
| `/memo save <text>` | save with auto-derived type/tags |
| `/memo get <id\|prefix>` | full record (prefix ≥4 chars) |
| `/memo update <id\|prefix> [flags] [body]` | patch metadata or body |
| `/memo delete <id\|prefix>` | delete (asks confirmation) |
| `/memo stats` | totals + paths + models |
| `/memo reindex` | absorb edits made directly in Obsidian |
| `/memo doctor [--gc] [--fix]` | self-check + orphan detect |

## CLI reference

```bash
memo doctor                       # self-check
memo doctor --gc                  # report orphans (store ↔ disk)
memo doctor --gc --fix            # drop orphan store rows (.md never auto-deleted)
memo save 'body markdown' --title 'X' -t mlx -t local
memo search 'query' --limit 5
memo list --limit 20 --type decision
memo get <id>
memo update <id> --title 'X2' -t mlx -t local --type decision
memo update <id> --content -      # read replacement body from stdin
memo reindex                      # absorb edits made directly in Obsidian
memo delete <id> --yes
memo stats
memo init                         # re-run first-run picker
memo migrate-vault <new-path>     # move memorias to a different folder
memo backup --out memo.zip        # backup .md files + index
memo mine-history --since 30      # backfill memorias from past Claude Code chats
memo watch                        # foreground file-watcher: auto-reindex on .md edit
memo install-watcher              # background watcher via launchd plist
memo uninstall-watcher            # remove the launchd watcher job
memo tui                          # live terminal dashboard (Ctrl+C exits)
memo as-of search 'query' --date 2026-03-01    # search a past snapshot
memo as-of ask 'question' --date 2026-03-01    # RAG on a past snapshot
memo as-of list --date 2026-03-01              # memorias that existed then
memo diff --from 2026-03-01 --to 2026-04-30    # diff between two snapshots
```

### Live dashboard — `memo tui`

![memo tui dashboard](docs/tui-dashboard.png)

Six panels, all-colored, refresh every second by default:

- **corpus** — total memorias, distinct project tags, top 3 types
- **runtime** — MLX warm/cold flags (`emb` / `rrk` / `chat`), vault size, watcher state
- **recent saves** — last 5 entries from `history.db`
- **recent recalls** — last 4 entries from the recall log (`~/.local/share/memo/recall.log`)
- **top tags** — most-frequent corpus tags (`project:*` highlighted)
- **activity** — 14-day saves/recalls sparklines (`▁▂▃▄▅▆▇█`)

Reads read-only from the existing `history.db` (saves), a JSONL recall
log written by `memo recall-hook` (auto-rotated at ~200 KB), and the
live MLX object flags (`embedder._model is not None`). Watcher state
comes from `launchctl print`. No new dependencies — Rich was already
pulled in.

Quit with `q`, `ESC`, or `Ctrl+C`.

### Backfill from past Claude Code conversations

`memo mine-history` walks `~/.claude/projects/<hash>/*.jsonl`, runs the
same prefilter + helper-LLM extract + embedding-dedup pipeline as the
live capture hook, and saves what's new. Resumable per file.

```bash
memo mine-history --since 30 --limit 20     # last 30 days, 20 newest sessions
memo mine-history --dry-run --debug         # cost estimation, no writes
```

### Auto-reindex on edit

Editing a memoria directly in Obsidian normally requires a manual
`memo reindex` to refresh embeddings. `memo watch` (foreground) or
`memo install-watcher` (background launchd job) debounces FS events
and runs `Memory.reindex()` automatically. Logs land in
`~/Library/Logs/memo/`.

### Project-scoped recall

`memo save` auto-attaches a `project:<repo>` tag derived from the git
toplevel of your cwd (or `MEMO_PROJECT_TAG`). The recall hook reads
`cwd` from the Claude Code hook payload and boosts memorias whose tags
match the current project by `MEMO_RECALL_PROJECT_BOOST` (default
`0.15`). Opt out per-call: `memo save --no-project-tag`. Disable
globally: `MEMO_AUTO_PROJECT_TAG=0`.

## First-run setup

The first time you run any `memo` command in an interactive shell, an arrow-key picker asks where memorias should live:

```
? Where should memo store your memorias?
❯ Standard macOS path: /Users/you/Documents/memo  (recommended)
  Obsidian vault: Notes  (/Users/you/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes)
  Obsidian vault: work-notes  (...)
  Custom path…
```

The choice is persisted to `~/.config/memo/config.toml`:

```toml
[storage]
data_dir = "/Users/you/Documents/memo"
# Optional — set when you pick an Obsidian vault. Used by `memo ingest`
# to bulk-index that vault's notes alongside your memorias.
vault_path = "/Users/you/Library/.../Notes"
```

Re-run the picker any time with `memo init`. To move memorias to a different location later:

```bash
memo migrate-vault ~/Documents/memo  # copies .md files, updates config, reindexes
```

Hooks (recall, prewarm, capture, session) get `MEMO_NONINTERACTIVE=1` prefixed in [`hooks/hooks.json`](./hooks/hooks.json) so they never trigger the picker.

## Configuration

All env vars are optional. Defaults aim at a fresh Apple Silicon Mac.

| Env var | Default | What |
|---|---|---|
| `MEMO_DATA_DIR` | `~/Documents/memo` | Where memoria `.md` files live |
| `MEMO_VAULT_PATH` | `(unset)` | Optional Obsidian vault for `memo ingest` |
| `MEMO_STATE_DIR` | `~/.local/share/memo` | sqlite-vec DB + state |
| `MEMO_CONFIG_FILE` | `~/.config/memo/config.toml` | Override config-file path |
| `MEMO_NONINTERACTIVE` | unset | Set to `1` in hooks to skip the first-run picker |
| `MEMO_MODEL_PROFILE` | `balanced` | Model bundle: `light`, `balanced`, or `quality` |
| `MEMO_LLM_MODEL` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Chat tier |
| `MEMO_HELPER_MODEL` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Helper tier |
| `MEMO_EMBEDDER_MODEL` | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | Embedder |
| `MEMO_EMBEDDER_DIMS` | `1024` | Embedding dim — must match the embedder |
| `MEMO_MAX_CONTENT_CHARS` | `64000` | Truncate body before embed |
| `MEMO_SEARCH_DEFAULT_LIMIT` | `10` | Default `--limit` for search |
| `MEMO_AUTO_PROJECT_TAG` | `1` | Auto-add `project:<repo>` tag from git toplevel on save. Set `0` to disable. |
| `MEMO_PROJECT_TAG` | unset | Explicit project tag (overrides git-toplevel detection) |

Resolution precedence (highest first): explicit kwargs → `MEMO_*` env vars → `~/.config/memo/config.toml` → legacy `MEMO_VAULT_PATH` + `MEMO_MEMORY_SUBDIR` (back-compat) → hardcoded defaults.

Model profiles:

- `light`: 0.6B embedder, no reranker. Best for low-latency hooks.
- `balanced`: 0.6B embedder + 0.6B reranker. Default for most users.
- `quality`: 4B embedder (2560 dims) + reranker + Qwen3 4B chat. Requires `rm ~/.local/share/memo/memvec.db && memo reindex` when switching from 1024-dim profiles.

If models are still downloading, you can save without MLX and keep keyword search available:

```bash
memo save "text to remember" --title "Short title" --defer-embed
memo search "text" --mode bm25
# later, once the embedder is cached:
memo reindex
```

## Upgrading the embedder

The default 0.6B is fast (~50 ms/embed) and small (~600 MB) but recall on diffuse queries (where the doc title doesn't lexically overlap with the query) can be noisy. For the 200–2000 memorias range, swap to the 4B variant when the noise starts to bite.

| Model | Dims | Disk | Recall | Per-embed |
|---|---|---|---|---|
| [`Qwen3-Embedding-0.6B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) *(default)* | 1024 | ~600 MB | OK | ~50 ms |
| [`Qwen3-Embedding-4B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ) | 2560 | ~3 GB | better | ~200 ms |
| [`Qwen3-Embedding-8B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-8B-4bit-DWQ) | 4096 | ~5 GB | best | ~400 ms |

To upgrade (example: 0.6B → 4B):

```bash
# 1) Pre-download.
hf download mlx-community/Qwen3-Embedding-4B-4bit-DWQ

# 2) Point memo at it.
export MEMO_EMBEDDER_MODEL=mlx-community/Qwen3-Embedding-4B-4bit-DWQ
export MEMO_EMBEDDER_DIMS=2560

# 3) Backup before destructive re-embed.
memo backup --out memo-pre-4b.zip

# 4) Wipe the index and rebuild.
rm ~/.local/share/memo/memvec.db
memo reindex
```

The dim mismatch is a hard error: `MEMO_EMBEDDER_DIMS` must match the new model's hidden size. `memo doctor` validates the dim at load.

## Design notes

- **One sqlite file, no Qdrant.** `sqlite-vec` outperforms a small Qdrant snapshot for the size of corpus memo targets (a few thousand entries, single-writer). Single file makes reset trivial: `rm memvec.db`.
- **Embed `title + body` together.** Titles carry the highest-density retrieval signal for memos with terse titles + long bodies. Prepending also protects the title from head-truncation when the body is long. Pure retag/type changes still skip the embedder.
- **`.md` is the storage of record.** Edit memories in Obsidian; the next `memo reindex` picks them up via `body_hash` mismatch.
- **Head-truncate long inputs + append EOS.** The embedder caps at 512 tokens; we head-truncate (preserves the title-like header) and explicitly append `<|im_end|>` so Qwen3-Embedding's last-token pool lands on the EOS hidden state it was fine-tuned for.
- **Asymmetric retrieval.** Queries get a `Instruct: …\nQuery: …` prefix; documents go raw. Without the prefix, cosine collapses toward 0.
- **Cosine distance metric.** The vec0 schema declares `distance_metric=cosine` so `vec.distance` is true cosine distance (1 − dot for unit vectors); `score = 1 − distance` is interpretable in [0, 1].
- **No Ollama dep, anywhere.** `pyproject.toml` does not declare it; `doctor` does not probe `:11434`. Anyone running memo with Ollama installed is just ignoring it.

## How memo differs from other agent-memory projects

A handful of projects sit in the same neighbourhood. They diverge on the things that actually matter day-to-day: where the model runs, where the data lives, how recall is wired, and whether you can read your own memory in plain text.

### Side-by-side comparison

| | **memo** | [`mem0`](https://github.com/mem0ai/mem0) | [`letta`](https://github.com/letta-ai/letta) (ex-MemGPT) | [`cognee`](https://github.com/topoteretes/cognee) | [`supermemory`](https://github.com/supermemoryai/supermemory) | [`mem-vault`](https://github.com/jagoff/mem-vault) | MCP [`memory` reference](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) | [`engram`](https://github.com/perrygeo/engram) |
|---|---|---|---|---|---|---|---|---|
| **Runtime** | MLX, in-process | Cloud API or Ollama | Postgres + LLM API | Cloud or Ollama | Cloud SaaS | Ollama daemon | Node, in-process | Python, in-process |
| **LLM/embed location** | local Mac (MLX) | OpenAI/Anthropic/Ollama | Anthropic/OpenAI/Ollama | OpenAI/Ollama/other | hosted | Ollama (`:11434`) | provider-supplied | provider-supplied |
| **Network in hot path** | **0** | yes (cloud) or `:11434` | yes (LLM API) | yes (LLM API) | always | `:11434` + `:6333` | yes (LLM API) | 0 |
| **Vector store** | sqlite-vec (one file) | Qdrant / pgvector | Postgres + pgvector | LanceDB / Qdrant / pgvector | hosted | Qdrant (server) | in-memory JSON | SQLite |
| **External daemons** | **none** | Ollama + Qdrant | Postgres | Postgres / vector DB | none (SaaS) | Ollama + Qdrant | none | none |
| **Storage of record** | **markdown files** | DB blob | DB rows | DB rows + graph | hosted DB | markdown files | JSON entity graph | DB rows |
| **Human-readable / editable** | ✅ open in Obsidian/vim | ❌ | ❌ | ❌ | ❌ | ✅ | partial (JSON dump) | ❌ |
| **MCP server (stdio)** | ✅ 13 tools | ❌ | ❌ | ❌ | ❌ | ✅ (unregistered) | ✅ (official ref) | ✅ |
| **Hybrid retrieval** | vec + BM25 + RRF | vec | vec | vec + graph | vec | vec | n/a (entity-based) | vec |
| **Cross-encoder reranker** | ✅ MLX Qwen3-Reranker | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Ambient recall (zero invoke)** | ✅ Claude Code hooks | ❌ | n/a | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Time-machine (past snapshots)** | ✅ `memo as-of ask --date …` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Apple Silicon optimisation** | ✅ first-class (MLX) | runs, no opt | runs, no opt | runs, no opt | n/a | works | n/a | works |
| **License** | MIT | Apache-2.0 | Apache-2.0 | Apache-2.0 | proprietary (SaaS) | MIT | MIT | MIT |
| **Privacy posture** | data never leaves Mac | depends on provider | depends on provider | depends on provider | hosted | local + cloud-ollama opt | depends on LLM | local |

> Notes on the table — projects move fast. The cells above reflect the public state of each repo at the time of writing. PR a correction if any is stale.

### The differentiators in plain terms

0. **🕰️ Time-machine — the ONLY agent-memory product with this.** Every other store in the table above (mem0, letta, cognee, supermemory, mem-vault, milasd/memo-mcp, doggybee, engram, MCP-memory reference) serves *current* state only. memo lets you `as-of` any past date, `diff` between two snapshots, and `ask` questions against the corpus as it stood months ago. The implementation is built on the audit log that already records every save/update/delete with field-level diffs — see [the algorithm diagram](docs/time-machine.svg). Use cases: debugging agent regressions, reproducible AI behavior, personal audit, compliance ("what did the model know when it took action X?"). **No competitor offers this and none can retrofit it without an audit-log they don't have.**

1. **100 % local hot path, no Ollama.** memo runs the LLM, embedder, and reranker **in-process via MLX**. No `localhost:11434` round-trip per call, no Docker for Qdrant, no provider key. mem0 / cognee / letta all rely on either a cloud API or a local Ollama daemon; supermemory is hosted; mem-vault needs both Ollama and Qdrant running. memo just imports MLX into the same Python process and goes.

2. **Markdown is the storage of record, not a DB blob.** Your memorias are plain `.md` files with frontmatter that you can open in Obsidian, edit in vim, sync via iCloud/git/Syncthing, and `grep` from a shell. The sqlite-vec index is rebuildable — `rm memvec.db && memo reindex`. Almost every alternative locks your knowledge inside an opaque database.

3. **Hybrid retrieval + cross-encoder reranker out of the box.** memo fuses semantic (vec) and keyword (BM25 over FTS5 with unicode61 + diacritic stripping for Spanish/Portuguese) via RRF, then optionally reranks the top-30 with a Qwen3-Reranker cross-encoder and fuses scores α-weighted. mem0 / letta / supermemory ship vec-only. cognee adds a graph but no cross-encoder. This is the single biggest precision lift for noisy or short queries.

4. **Ambient recall as a first-class feature.** With the bundled Claude Code plugin, `SessionStart` prewarms MLX and `UserPromptSubmit` consults memory on every prompt (5 s budget, top-3 above cosine 0.6, injected as `additionalContext`). The agent sees the right memorias *before* it answers — no `/memo` call from you. No alternative ships this as a turnkey hook bundle.

5. **MCP is a primary interface, not an afterthought.** memo exposes 13 tools over stdio so Claude Code, Cursor, Cline, Continue, Paperclip, and any future MCP client get the same contract on day one. mem0 and letta have no MCP server; mem-vault has one but isn't published in the registry; the official MCP `memory` reference is entity-graph-only and stores in JSON.

6. **Apple Silicon is a target, not a footnote.** Embedder, reranker, and chat are 4-bit MLX builds tuned for unified memory: ~50 ms/embed on 0.6B, sub-second first recall after prewarm, ~4 GB RAM ceiling for the default 7B chat tier. Other projects "work" on M-series Macs because Python runs there — they aren't tuned for it.

7. **No vendor lock and no telemetry.** MIT package on top of MIT/Apache-2.0 dependencies (MLX MIT, sqlite-vec Apache-2.0, Qwen weights Apache-2.0). Nothing phones home; `doctor` literally does not probe `:11434`.

### Other projects called "memo" or "memo-mcp"

A handful of unrelated repos share the name. Quick disambiguation in case you're searching:

| Project | What it is | Overlap with us |
|---|---|---|
| [`upstash/memo`](https://github.com/upstash/memo) | MCP server for **handing off conversation state** between agents (goals / pending tasks / decisions). State lives in Upstash Redis (managed cloud or self-hosted on Vercel). No embeddings, no RAG. | Different problem entirely — agent handoff, not a memory archive. We're local-first markdown + vector search; they're cloud-state with structured handoff objects. |
| [`milasd/memo-mcp`](https://github.com/milasd/memo-mcp) | Local Python MCP for **RAG over personal journal entries**. Pluggable vector backend (ChromaDB default / FAISS / in-memory), Apple-Silicon GPU embedder, no bundled LLM. | Closest competitor. Both local RAG. We diverge on: MLX-only runtime, markdown source-of-record (Obsidian-readable), sqlite-vec + FTS5 hybrid w/ RRF, cross-encoder reranker, history.db / graph.db split, ambient recall hook bundle. _PyPI name collision avoided — we ship as `mlx-memo` from 0.5.0._ |
| [`doggybee/mcp-server-memo`](https://github.com/doggybee/mcp-server-memo) | Node.js MCP for **append-only versioned session summaries**. Plain filesystem JSON, no DB, no vector store, no embedder. | Different category — flat-file versioned summaries, no semantic search. |

### When you should *not* pick memo

Pick something else when:

- You're not on Apple Silicon. MLX is the load-bearing piece — memo will not run on Linux / Windows / Intel Macs.
- You need a hosted, multi-tenant memory service across many users — `supermemory` or `mem0` cloud is what you want.
- You want a long-horizon agent runtime with explicit "core memory" vs "archival memory" tiers and an event loop around it — that's `letta`'s sweet spot.
- You want a knowledge-graph + ontology layer rather than a doc store — `cognee` is the right pick.

memo's bet is the opposite: a single user, one machine, plain markdown, MLX, and a contract small enough to remember.

## Roadmap

Ship-ready today:

- [x] Memory API: save / search / list / get / update / delete / reindex / consolidate / ask / stats
- [x] CLI: ~28 commands including `doctor`, `migrate-vault`, `backup`, `ingest`, `mine-history`, `watch`
- [x] MCP server (13 tools + `memo://recent` / `memo://memory/{id}` resources)
- [x] Hybrid search (vec + BM25 + RRF + cross-encoder rerank)
- [x] Prefix-ID lookup (git-style, ≥4 chars)
- [x] Ambient recall (Claude Code plugin)
- [x] **Project-scoped recall** (auto-tag + cwd-based boost)
- [x] **Token-budget-aware recall** packing
- [x] **Transcript miner** (`memo mine-history` over `~/.claude/projects/`)
- [x] **File-watcher daemon** (`memo watch` / `install-watcher` launchd plist)
- [x] First-run picker + migration tooling
- [x] Paperclip plugin (5 tools)

Post-v0:

- [ ] Entity graph queries over `graph.db`
- [ ] LLM-driven consolidation / dedup using the 3B helper tier
- [ ] Multi-hop `ask()` over `[[wikilinks]]`

## Provenance

Forked from [`mem-vault`](https://github.com/jagoff/mem-vault) philosophically (storage layout + frontmatter schema), not literally — the codebase is new. The MLX backend pieces (embedder pooling, chat template handling) are direct ports from [`obsidian-rag`](https://github.com/jagoff/rag-obsidian) Phase 1+2 of the MLX migration.

## License

MIT — see [LICENSE](LICENSE).
