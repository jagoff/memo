# memo

> Local MCP memory backed by an Obsidian vault. **MLX-native** stack — zero Ollama, zero cloud APIs.

`memo` is a from-scratch replacement for [`mem-vault`](https://github.com/jagoff/mem-vault)
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
features. v0 of `memo` is intentionally smaller-surface; we add features back as their
need is proven.

## Install

```bash
cd ~/repositories/memo
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

## MCP

Wire it up to Claude Code (user scope, absolute path to the venv entry
point — `claude mcp add` is interactive-only inside an active Claude
Code session, so just edit `~/.claude.json` and add a top-level
`mcpServers` entry):

```jsonc
// ~/.claude.json
{
  "mcpServers": {
    "memo": {
      "type": "stdio",
      "command": "/Users/<you>/repositories/memo/.venv/bin/memo-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

Restart Claude Code. Verify with `claude mcp list` from a fresh shell.
Tools surface as `mcp__memo__memory_*` inside the agent.

Tools exposed:

- `memory_save(content, title?, type?, tags?)` → returns full record
- `memory_search(query, limit?, type?, body_chars=280)` → top-k by cosine similarity. `body` is truncated to `body_chars` (set to a large number to disable). `body_truncated: true` is added when truncation kicked in.
- `memory_list(limit?, type?)` → recent by `updated` desc
- `memory_get(id)` → one full record. `id` accepts a unique prefix ≥4 chars; on ambiguity returns `{"error": "ambiguous", "matches": [...]}`.
- `memory_update(id, title?, type?, tags?, content?)` → patches fields, re-embeds only if body changed. Same prefix-ID + ambiguous-shape semantics as `memory_get`.
- `memory_reindex()` → re-scan vault, re-embed entries whose on-disk body diverged from the indexed `body_hash`
- `memory_delete(id)` → removes from vec + disk. Same prefix-ID semantics.
- `memory_stats()` → counts + paths + active models

## Slash command — `/memo`

Once the MCP is registered, the `/memo` skill at
`~/.config/devin/skills/memo/SKILL.md` (symlinked into
`~/.claude/skills/memo/`) routes user input to the right MCP tool:

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
| `MEMO_MAX_CONTENT_SIZE` | `64000` | Truncate body before embed |
| `MEMO_SEARCH_DEFAULT_LIMIT` | `10` | Default `--limit` for search |

## Design notes

- **One sqlite file, no qdrant**. `sqlite-vec` outperforms a small qdrant snapshot for
  the size of corpus memo targets (a few thousand entries, single-writer). Single-file
  also makes reset trivial: `rm ~/.local/share/memo/memvec.db`.
- **Embed body, not title**. Titles are short, biased toward filename-style noise.
  Body-only embed gives the user freedom to title things terselessly.
- **`.md` is the storage of record**. The user can edit memories from Obsidian and the
  next index pass picks them up via `body_hash` mismatch (planned: `memo reindex`).
- **Tail-truncate long inputs**. The embedder caps at 512 tokens; tail-truncation
  preserves the trailing summary that Qwen3-Embedding's instruction template was tuned for.
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
