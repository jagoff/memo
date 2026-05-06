# mem-lmx

> Local MCP memory backed by an Obsidian vault. **MLX-native** stack — zero Ollama, zero cloud APIs.

`mem-lmx` is a from-scratch replacement for [`mem-vault`](https://github.com/jagoff/mem-vault)
that drops Ollama entirely and runs the LLM + embedder in-process via [Apple
MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

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
features. v0 of `mem-lmx` is intentionally smaller-surface; we add features back as their
need is proven.

## Install

```bash
cd ~/repositories/mem-lmx
uv pip install -e '.[dev]'
```

Pre-download the MLX models (first run downloads them automatically too, but doing it
once up-front avoids a multi-GB stall on the first save/search):

```bash
hf download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
hf download mlx-community/Qwen2.5-3B-Instruct-4bit
hf download mlx-community/Qwen2.5-7B-Instruct-4bit
```

## CLI

```bash
mem-lmx doctor                       # self-check
mem-lmx save 'body markdown' --title 'X' -t mlx -t local
mem-lmx search 'query' --limit 5
mem-lmx list --limit 20 --type decision
mem-lmx get <id>
mem-lmx delete <id> --yes
mem-lmx stats
```

## MCP

Wire it up to Claude Code:

```bash
claude mcp add mem-lmx -s user mem-lmx-mcp
```

Tools exposed:

- `memory_save(content, title?, type?, tags?)` → returns full record
- `memory_search(query, limit?, type?)` → top-k by cosine similarity
- `memory_list(limit?, type?)` → recent by `updated` desc
- `memory_get(id)` → one full record
- `memory_delete(id)` → removes from vec + disk
- `memory_stats()` → counts + paths + active models

## Configuration

All env vars are optional. Defaults aim at a fresh Apple Silicon Mac with Obsidian in the
default iCloud location.

| Env var | Default | What |
|---|---|---|
| `MEM_LMX_VAULT_PATH` | `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes` | Vault root |
| `MEM_LMX_MEMORY_SUBDIR` | `04-Archive/99-obsidian-system/99-AI/memory` | Where `.md` files land |
| `MEM_LMX_STATE_DIR` | `~/.local/share/mem-lmx` | sqlite-vec DB + state |
| `MEM_LMX_LLM_MODEL` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Chat tier |
| `MEM_LMX_HELPER_MODEL` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Helper tier |
| `MEM_LMX_EMBEDDER_MODEL` | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | Embedder |
| `MEM_LMX_EMBEDDER_DIMS` | `1024` | Embedding dim — must match the embedder |
| `MEM_LMX_MAX_CONTENT_SIZE` | `64000` | Truncate body before embed |
| `MEM_LMX_SEARCH_DEFAULT_LIMIT` | `10` | Default `--limit` for search |

## Design notes

- **One sqlite file, no qdrant**. `sqlite-vec` outperforms a small qdrant snapshot for
  the size of corpus mem-lmx targets (a few thousand entries, single-writer). Single-file
  also makes reset trivial: `rm ~/.local/share/mem-lmx/memvec.db`.
- **Embed body, not title**. Titles are short, biased toward filename-style noise.
  Body-only embed gives the user freedom to title things terselessly.
- **`.md` is the storage of record**. The user can edit memories from Obsidian and the
  next index pass picks them up via `body_hash` mismatch (planned: `mem-lmx reindex`).
- **Tail-truncate long inputs**. The embedder caps at 512 tokens; tail-truncation
  preserves the trailing summary that Qwen3-Embedding's instruction template was tuned for.
- **No Ollama dep, anywhere**. `pyproject.toml` does not declare `ollama`; the `doctor`
  command does not probe `:11434`. Anyone running mem-lmx with Ollama installed is just
  ignoring it.

## Status

- [x] Skeleton: config, MLX embedder, MLX chat, sqlite-vec store
- [x] Memory API: save / search / list / get / delete
- [x] CLI: 7 commands incl. doctor
- [x] MCP server with 6 tools
- [ ] Smoke tests (`tests/test_smoke.py` placeholder)
- [ ] `update()` patch path (post v0)
- [ ] `consolidate` clustering (post v0)
- [ ] Markdown frontmatter watcher (re-index on edit)

## Provenance

Forked from `mem-vault` philosophically (storage layout + frontmatter schema), not
literally — the codebase is new. The MLX backend pieces (embedder pooling, chat
template handling) are direct ports from the work in [`obsidian-rag`](https://github.com/jagoff/rag-obsidian)
Phase 1+2 of the MLX migration (commits `aff4b8f`, `b1f163d`, `7f9a34a`).
