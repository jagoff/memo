# Registry submissions — `mlx-memo`

Ready-to-paste blocks for getting `mlx-memo` listed in the MCP
ecosystem. None of these can be fully automated; each registry needs a
manual PR or form submission. The text below is calibrated so you can
copy-paste with at most one or two edits.

---

## 1. Official MCP servers list (`modelcontextprotocol/servers`)

The community-maintained README at
[modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)
lists third-party MCP servers under
`README.md` → "Community Servers" section. Submission flow:

1. Fork [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers).
2. Find the **Community Servers** section in `README.md` (alphabetical).
3. Insert this entry under `M` (between `meilisearch` and any later `m*` entry):

```markdown
- **[memo](https://github.com/jagoff/memo)** — Local-first persistent memory for AI agents. MLX-native runtime (Apple Silicon), markdown-on-disk as the source of record, sqlite-vec + BM25 hybrid retrieval with a cross-encoder reranker, ambient-recall Claude Code hooks. Zero Ollama, zero cloud APIs.
```

4. Commit, push, open a PR titled `Add memo MCP server (local MLX-native memory)`.
5. PR body template:

```markdown
## What

`memo` is a local MCP memory server for Apple Silicon Macs. It exposes 13 tools (`memory_save`, `memory_search`, `memory_list`, `memory_ask`, etc.) plus two resources (`memo://recent`, `memo://memory/{id}`) over stdio.

## Why it's different

- 100 % local — embedder (Qwen3-Embedding), reranker (Qwen3-Reranker), and chat LLM (Qwen2.5-7B) all run in-process via Apple MLX. No Ollama, no Qdrant, no cloud key.
- Markdown is the storage of record — memorias are plain `.md` files in an Obsidian-compatible vault. The sqlite-vec index is rebuildable.
- Hybrid retrieval out of the box: vec (cosine) + BM25 (FTS5 with unicode61 + diacritic stripping) fused via RRF, then cross-encoder reranked.
- Ambient recall: a Claude Code plugin auto-injects the top-k memorias as `additionalContext` on every prompt — no manual `/memo` call.

## Repo / install

- Repo: <https://github.com/jagoff/memo>
- PyPI: <https://pypi.org/project/mlx-memo/>
- Install: `pip install mlx-memo` (Apple Silicon only)
- License: MIT
```

---

## 2. mcp.so

[mcp.so](https://mcp.so) is a community directory. It auto-crawls
GitHub repos tagged with the `mcp-server` topic — which we just set —
but a manual submission speeds things up.

Submit at <https://mcp.so/submit> with:

| Field | Value |
| --- | --- |
| Name | `memo` |
| Slug | `mlx-memo` (since `memo` is taken) |
| Repo | `https://github.com/jagoff/memo` |
| Category | `Memory & Knowledge` |
| Tags | `memory`, `mlx`, `apple-silicon`, `local-first`, `obsidian`, `rag`, `markdown` |
| One-liner | Local MLX-native persistent memory for AI agents — markdown source of record, sqlite-vec hybrid retrieval, zero cloud APIs |
| Long description | (paste the "Why it's different" block from the PR above) |
| Author | Fernando Ferrari |
| License | MIT |

---

## 3. Other places to list

- **[awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)** — same flow as #1: fork, add entry to the Memory section, PR.
- **PyPI keywords** — already set in `pyproject.toml`: `mcp`, `memory`, `obsidian`, `mlx`, `rag`, `agents`, `claude`, `local-first`, `apple-silicon`.
- **Homebrew tap** — see `docs/homebrew-tap.md` (separate flow).
- **Claude Code plugin marketplace** — already published via this repo's `.claude-plugin/` directory; users install via `/plugin install memo@jagoff/memo`.

---

## Verification

After acceptance, verify the listing renders correctly:

- mcp.so: search for "memo" → entry shows with the right tags
- modelcontextprotocol/servers: README renders the new bullet under M
- PyPI: <https://pypi.org/project/mlx-memo/> shows latest version
- GitHub: <https://github.com/jagoff/memo> shows the 13 topic tags on the right sidebar

If any registry rejects or requests changes, update this doc with the
revised submission so the next release reuses the polished copy.
