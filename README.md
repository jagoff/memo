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
- `memory_search(query, limit?, type?, body_chars=280, mode="hybrid")` → top-k search. `mode="hybrid"` (default) fuses vec + bm25 via reciprocal rank fusion. `mode="vec"` is semantic only, `mode="bm25"` is keyword only (FTS5 over title + tags + body, unicode61 + diacritic-stripping for Spanish). `body` truncated to `body_chars`; pass a large number to disable.
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
