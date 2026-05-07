# memo

> Local MCP memory backed by an Obsidian vault. **MLX-native** stack — zero Ollama, zero cloud APIs.

[![PyPI](https://img.shields.io/pypi/v/memo-mcp.svg)](https://pypi.org/project/memo-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/memo-mcp.svg)](https://pypi.org/project/memo-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`memo` is persistent semantic memory for AI agents, designed for Apple Silicon
Macs. It exposes a [Model Context Protocol](https://modelcontextprotocol.io)
server so any MCP-aware client (Claude Code, Claude Desktop, Cursor, Cline,
Continue, …) can `save` / `search` / `list` / `update` / `delete` memories,
and a CLI for the same operations from a shell.

It runs the LLM + embedder **in-process** via [Apple MLX](https://github.com/ml-explore/mlx),
indexes embeddings in [`sqlite-vec`](https://github.com/asg017/sqlite-vec) (single
file, no daemon), and stores each memory as a markdown file with frontmatter
inside an Obsidian vault — so the storage of record is human-editable and
syncs through whatever you already use to sync notes (iCloud / git / Syncthing).

**No Ollama. No Qdrant. No cloud API. No keys.**

## Why memo (vs the alternatives)

| | **memo** | [`mem-vault`](https://github.com/jagoff/mem-vault) | [`mem0`](https://github.com/mem0ai/mem0) | [engram](https://github.com/perrygeo/engram) |
|---|---|---|---|---|
| Runtime | MLX (in-process) | Ollama daemon | Cloud or Ollama | SQLite |
| Vector store | sqlite-vec (file) | Qdrant (server) | Qdrant / pgvector | SQLite |
| External daemons | none | Ollama + Qdrant | Ollama + Qdrant | none |
| Network calls | **0** (offline) | localhost:11434 + :6333 | localhost or HTTPS | 0 |
| Storage | markdown files | markdown files | DB only | DB only |
| Apple Silicon | ✅ first-class | works | works | works |
| MCP server | ✅ stdio | ✅ stdio (unregistered) | ❌ | ✅ stdio |
| Repo | this | [jagoff/mem-vault](https://github.com/jagoff/mem-vault) | [mem0ai/mem0](https://github.com/mem0ai/mem0) | [perrygeo/engram](https://github.com/perrygeo/engram) |

Trade-off: `memo` is intentionally smaller-surface than `mem0`. It does not
ship hybrid retrieval / reflection / consolidation **out of the box** — those
are post-v0 (see [Status](#status)).

## Stack

| Component | What |
|---|---|
| LLM | [`mlx-lm`](https://github.com/ml-explore/mlx-lm) loading [`Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) (chat) + [`Qwen2.5-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit) (helper) |
| Embedder | [`Qwen3-Embedding-0.6B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) — 1024-dim L2-normalised, last-token pooling |
| Vector store | [`sqlite-vec`](https://github.com/asg017/sqlite-vec) — single file, no daemon, embedded |
| Storage of record | Markdown files under `<vault>/04-Archive/99-obsidian-system/99-AI/memory/` |
| MCP transport | [`fastmcp`](https://github.com/jlowin/fastmcp) |

## Why a from-scratch replacement (not a fork of mem-vault)

`mem-vault` is built on [`mem0`](https://github.com/mem0ai/mem0), which assumes Ollama or a cloud
LLM provider as its backend. Custom-providering MLX into mem0 requires deep familiarity with
`mem0.LlmBase` plus shipping a fork of mem0 alongside. Cleaner to own the memory layer
ourselves with the contract we actually use (save / search / list / get / update / delete) and
the storage format we already standardised on (markdown + frontmatter under
`99-AI/memory/`).

The trade-off: we lose mem0's built-in consolidation / hybrid retrieval / reflection
features. v0 of `memo` is intentionally smaller-surface; we add features back as their
need is proven.

## Requirements

- macOS on Apple Silicon (M1 / M2 / M3 / M4). MLX is the load-bearing piece.
- Python ≥ 3.13.
- ~4 GB free disk for the default model set (downloaded on first use).
- Optional: an Obsidian vault. If you don't have one, memo defaults to
  `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes` and creates
  the memory subdirectory if it doesn't exist.

## Install

```bash
pip install memo-mcp
```

Or, if you prefer [`uv`](https://github.com/astral-sh/uv):

```bash
uv tool install memo-mcp
```

Both expose two commands on your PATH: `memo` (CLI) and `memo-mcp` (MCP server stdio).

Pre-download the MLX models so the first save/search doesn't stall on a
multi-GB download:

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

## CLI

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
```

## MCP setup

After `pip install memo-mcp`, register the MCP with your client.

### Claude Code

```bash
claude mcp add memo -s user $(which memo-mcp)
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

Replace `/path/to/memo-mcp` with the output of `which memo-mcp`. Restart
Claude Code. Tools surface as `mcp__memo__memory_*` inside the agent.

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```jsonc
{
  "mcpServers": {
    "memo": {
      "command": "/path/to/memo-mcp"
    }
  }
}
```

### Cursor / Cline / Continue

Each client has its own MCP config UI but the contract is the same:
register a stdio server pointing at the `memo-mcp` binary.

Tools exposed:

- `memory_save(content, title?, type?, tags?)` → returns full record
- `memory_search(query, limit?, type?, body_chars=280, mode="hybrid")` → top-k search. `mode="hybrid"` (default) fuses vec + bm25 via reciprocal rank fusion. `mode="vec"` is semantic only, `mode="bm25"` is keyword only (FTS5 over title + tags + body, unicode61 + diacritic-stripping for Spanish). `body` truncated to `body_chars`; pass a large number to disable.
- `memory_list(limit?, type?)` → recent by `updated` desc
- `memory_get(id)` → one full record. `id` accepts a unique prefix ≥4 chars; on ambiguity returns `{"error": "ambiguous", "matches": [...]}`.
- `memory_update(id, title?, type?, tags?, content?)` → patches fields, re-embeds only if body changed. Same prefix-ID + ambiguous-shape semantics as `memory_get`.
- `memory_reindex()` → re-scan vault, re-embed entries whose on-disk body diverged from the indexed `body_hash`
- `memory_delete(id)` → removes from vec + disk. Same prefix-ID semantics.
- `memory_stats()` → counts + paths + active models

## Slash command — `/memo` (Claude Code only)

A Claude Code [skill](https://docs.claude.com/en/docs/claude-code/skills) ships
in this repo at `skills/memo/SKILL.md`. Copy or symlink it into
`~/.claude/skills/memo/SKILL.md` and you get slash-command sugar over the MCP
tools:

```bash
ln -s "$(pwd)/skills/memo/SKILL.md" ~/.claude/skills/memo/SKILL.md
```

Or install everything (skill + MCP config) in one step via the bundled
[Claude Code plugin](https://docs.claude.com/en/docs/claude-code/plugins):

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
| `/memo doctor [--gc] [--fix]` | self-check + orphan detect (shell) |

## Configuration

All env vars are optional. Defaults aim at a fresh Apple Silicon Mac with Obsidian in the
default iCloud location.

| Env var | Default | What |
|---|---|---|
| `MEMO_VAULT_PATH` | `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes` | Vault root |
| `MEMO_MEMORY_SUBDIR` | `04-Archive/99-obsidian-system/99-AI/memory` | Where `.md` files land |
| `MEMO_STATE_DIR` | `~/.local/share/memo` | sqlite-vec DB + state |
| `MEMO_LLM_MODEL` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Chat tier |
| `MEMO_HELPER_MODEL` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Helper tier |
| `MEMO_EMBEDDER_MODEL` | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | Embedder |
| `MEMO_EMBEDDER_DIMS` | `1024` | Embedding dim — must match the embedder |
| `MEMO_MAX_CONTENT_CHARS` | `64000` | Truncate body before embed |
| `MEMO_SEARCH_DEFAULT_LIMIT` | `10` | Default `--limit` for search |

## Upgrading the embedder (recall vs cost)

The default [`Qwen3-Embedding-0.6B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) is fast (~50ms per embed) and small
(~600MB on disk) but the 0.6B parameter count means recall on diffuse
queries (where the doc title doesn't lexically overlap with the query)
can be noisy. For the 200-2000 memorias range, swap to the 4B variant
when the noise becomes a problem.

| Model | Dims | Disk | Recall | Per-embed |
|---|---|---|---|---|
| [`Qwen3-Embedding-0.6B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) (default) | 1024 | ~600 MB | OK | ~50 ms |
| [`Qwen3-Embedding-4B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ) | 2560 | ~3 GB | better | ~200 ms |
| [`Qwen3-Embedding-8B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-8B-4bit-DWQ) | 4096 | ~5 GB | best | ~400 ms |

**To upgrade** (example: 0.6B → 4B):

```bash
# 1) Pre-download the new weights so the first embed doesn't stall.
hf download mlx-community/Qwen3-Embedding-4B-4bit-DWQ

# 2) Set the env vars (add to your shell rc for persistence).
export MEMO_EMBEDDER_MODEL=mlx-community/Qwen3-Embedding-4B-4bit-DWQ
export MEMO_EMBEDDER_DIMS=2560

# 3) Backup before the rebuild — re-embed is destructive of the old vectors.
memo backup --out memo-pre-4b.zip

# 4) Drop the index and re-embed the entire corpus.
rm ~/.local/share/memo/memvec.db
memo reindex     # re-runs the embedder over every .md
```

The dim mismatch is a hard error — `MEMO_EMBEDDER_DIMS` must match the
new model's hidden size (1024 for 0.6B, 2560 for 4B, 4096 for 8B).
`memo doctor` validates the dim at load time.

The `body_hash` from the old store is irrelevant after `rm memvec.db`,
so a plain `memo reindex` re-embeds everything. Without the rm, use
`memo reindex --force` instead — same result.

## Design notes

- **One sqlite file, no qdrant**. `sqlite-vec` outperforms a small qdrant snapshot for
  the size of corpus memo targets (a few thousand entries, single-writer). Single-file
  also makes reset trivial: `rm ~/.local/share/memo/memvec.db`.
- **Embed `title + body` together**. Titles carry the highest-density retrieval signal
  for memos with terse titles + long bodies. Prepending also protects the title from
  head-truncation when the body is long. Pure retag/type changes still skip the embedder.
- **`.md` is the storage of record**. The user can edit memories from Obsidian and the
  next index pass picks them up via `body_hash` mismatch — `memo reindex` (or `--force`
  to re-embed even when hash matches, e.g. after composition or model change).
- **Head-truncate long inputs + append EOS**. The embedder caps at 512 tokens; we
  head-truncate (preserves the title-like header) and explicitly append `<|im_end|>` so
  Qwen3-Embedding's last-token pool lands on the EOS hidden state it was fine-tuned for.
- **Asymmetric retrieval**. Queries get a `Instruct: ...\nQuery: ...` prefix; documents
  go raw. Without the prefix, cosine collapses toward 0. See the comment in `embedder.py`
  for the empirical verification.
- **Cosine distance metric**. The vec0 schema declares `distance_metric=cosine` so
  `vec.distance` is true cosine distance (1 - dot for unit vectors); `score = 1 - distance`
  is interpretable in [0, 1].
- **No Ollama dep, anywhere**. `pyproject.toml` does not declare `ollama`; the `doctor`
  command does not probe `:11434`. Anyone running memo with Ollama installed is just
  ignoring it.

## Status

v0 — ship-ready.

- [x] Skeleton: config, MLX embedder, MLX chat, sqlite-vec store
- [x] Memory API: save / search / list / get / **update** / **reindex** / delete / **gc**
- [x] CLI: 9 commands (`save`, `search`, `list`, `get`, `update`, `reindex`, `delete`, `stats`, `doctor [--gc [--fix]]`)
- [x] MCP server with 8 tools
- [x] Stubbed unit tests (39, including MCP server snippet + ambiguous-prefix coverage) + real-MLX smoke (`tests/test_smoke_mlx.py`, 3 cases, gated by `requires_mlx`)
- [x] Prefix-ID lookup (git-style, ≥4 chars) on `get` / `update` / `delete` for both CLI and MCP
- [x] `/memo` skill — Claude Code slash command routing all verbs to the MCP

Post-v0 (not blocking the v0 ship):

- [ ] `consolidate` clustering — LLM-driven dedup/merge using `MLXChat`
- [ ] Markdown frontmatter watcher daemon (auto-`reindex` on edit, plist-installed)
- [ ] Hybrid search (vec + BM25) for short-query recall

## Provenance

Forked from `mem-vault` philosophically (storage layout + frontmatter schema), not
literally — the codebase is new. The MLX backend pieces (embedder pooling, chat
template handling) are direct ports from the work in [`obsidian-rag`](https://github.com/jagoff/rag-obsidian)
Phase 1+2 of the MLX migration (commits `aff4b8f`, `b1f163d`, `7f9a34a`).
