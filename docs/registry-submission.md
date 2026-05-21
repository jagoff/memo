# Registry submissions — `mlx-memo`

Ready-to-paste blocks for getting `mlx-memo` listed in the MCP ecosystem.
The official MCP Registry is automated by `.github/workflows/publish.yml`;
community directories still need a manual PR or form submission.

---

## 1. Official MCP Registry

The canonical registry entry is `server.json` with the package name
`mlx-memo` and the MCP name `io.github.jagoff/memo`. On each GitHub Release,
the `publish` workflow publishes PyPI first, waits for propagation, and then
runs:

```bash
mcp-publisher login github-oidc
mcp-publisher publish
```

The README also includes the required PyPI metadata hint:

```html
<!-- mcp-name: io.github.jagoff/memo -->
```

## 2. Official MCP servers list (`modelcontextprotocol/servers`)

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
- Install: `pipx install mlx-memo` or the one-line installer in the README (Apple Silicon only)
- License: MIT
```

---

## 3. mcp.so

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

## 4. Other places to list

- **[awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)** — same flow as #1: fork, add entry to the Memory section, PR.
- **PyPI keywords** — already set in `pyproject.toml`: `mcp`, `memory`, `obsidian`, `mlx`, `rag`, `agents`, `claude`, `local-first`, `apple-silicon`.
- **Homebrew tap** — see `docs/homebrew/README.md`.
- **Claude Code plugin marketplace** — already published via this repo's
  `.claude-plugin/` directory; users install with
  `claude plugin marketplace add jagoff/memo` and
  `claude plugin install memo@memo -s user`.
- **Codex plugin marketplace + user skill** — already packaged under
  `.agents/plugins/` and `plugins/memo/` for MCP metadata. The installer also
  copies `skills/memo/SKILL.md` to `$CODEX_HOME/skills/memo/SKILL.md` so Codex
  surfaces that expose skills as slash commands can show the exact `/memo`.
- **Windsurf / Cascade (local install)** — configured by `memo install-slash --client windsurf`,
  which writes a stdio `memo` server entry to
  `~/.codeium/windsurf/mcp_config.json`. Windsurf users then refresh MCP
  servers in Cascade. This handles per-user installs but does **not** list memo
  in the public Windsurf marketplace — see section 5 for that.

---

## 5. Windsurf marketplace (Cascade in-app + windsurf.run)

There are two Windsurf-adjacent surfaces and they have different submission
flows. Both depend on prereqs being green (PyPI `mlx-memo` at the latest
version + MCP Registry entry `io.github.jagoff/memo` published via
`.github/workflows/publish.yml`).

### 5a. Cascade in-app marketplace (official, curated)

The marketplace shown inside Windsurf (MCPs icon → Marketplace) is curated by
Codeium. There is **no public PR or form**. Inclusion is granted via direct
outreach. Recommended path:

1. Confirm the MCP Registry entry is live:
   ```bash
   curl -s "https://registry.modelcontextprotocol.io/v0/servers?search=jagoff" | jq
   ```
   Should return at least one server with name `io.github.jagoff/memo`.
2. Email `support@codeium.com` with the template below.
3. Mirror the ask on X tagging `@windsurf_ai` (public visibility helps).

Email / DM template:

```
Subject: MCP marketplace inclusion — memo (local-first MLX memory)

Hi Windsurf team,

I maintain memo, a local-first persistent-memory MCP server for Apple
Silicon. It's already published to the official MCP Registry as
`io.github.jagoff/memo` and to PyPI as `mlx-memo`. I'd like it listed in
the Cascade in-app MCP marketplace.

- Repo: https://github.com/jagoff/memo
- PyPI: https://pypi.org/project/mlx-memo/
- MCP Registry: io.github.jagoff/memo
- Transport: stdio
- License: MIT
- Platform: macOS arm64 (Apple Silicon)

Differentiators:
- 100 % local — MLX-native embedder (Qwen3), reranker (Qwen3-Reranker),
  and chat LLM (Qwen2.5). No Ollama, no cloud API.
- Markdown is the storage of record (Obsidian-compatible vault).
- Hybrid retrieval out of the box: sqlite-vec + BM25 (FTS5) fused via RRF
  with cross-encoder rerank.
- 13 MCP tools + 2 resources. Windsurf install already supported via
  `memo install-slash --client windsurf`.

Happy to provide a demo video or additional metadata. Thanks!

— Fernando Ferrari
```

### 5b. windsurf.run / cursor.directory community directory

`windsurf.run/mcp` is a read-only view backed by `pontusab/cursor.directory`.
The submission form does **not** live on `windsurf.run` (that domain only
ships listings, the `/plugins/new` route returns 404). Submit on
**`cursor.directory`** instead — the same database powers both views, so an
approved entry shows up automatically under `windsurf.run/mcp` and
`windsurf.directory`.

Submission flow:

1. Visit <https://cursor.directory/plugins/new>.
2. Sign in with GitHub.
3. Choose "Auto-detect from a GitHub repo" and paste
   `https://github.com/jagoff/memo`. The parser reads
   `.claude-plugin/plugin.json` + `.mcp.json` from the repo and pre-fills
   the form (name, description, version, repo URL, license, keywords, and
   the `mcp_server` component with `command: memo-mcp`).
4. Verify the pre-filled fields, adjust if needed, submit, wait for
   moderation.

Form values:

| Field | Value |
| --- | --- |
| Type | MCP Server |
| Name | `memo` |
| Slug | `mlx-memo` |
| Repo URL | `https://github.com/jagoff/memo` |
| Install command | `pipx install mlx-memo` |
| MCP config snippet | (see below) |
| Category / Tag | Memory · Knowledge · Local-first |
| Description | Local-first persistent memory for AI agents. MLX-native runtime (Apple Silicon), markdown-on-disk source of record, sqlite-vec + BM25 hybrid retrieval with a cross-encoder reranker, ambient-recall Claude Code hooks. Zero Ollama, zero cloud APIs. |
| Author | Fernando Ferrari |
| License | MIT |

MCP config snippet to paste in the form:

```json
{
  "mcpServers": {
    "memo": {
      "command": "memo-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

---

## Verification

After acceptance, verify the listing renders correctly:

- mcp.so: search for "memo" → entry shows with the right tags
- modelcontextprotocol/servers: README renders the new bullet under M
- PyPI: <https://pypi.org/project/mlx-memo/> shows latest version
- GitHub: <https://github.com/jagoff/memo> shows the 13 topic tags on the right sidebar
- MCP Registry: `curl -s "https://registry.modelcontextprotocol.io/v0/servers?search=jagoff"` returns `io.github.jagoff/memo`
- cursor.directory: <https://cursor.directory/mcp/mlx-memo> resolves once moderation approves
- windsurf.run: <https://windsurf.run/mcp> search for "memo" shows the entry (mirrors cursor.directory)
- Cascade in-app: Windsurf → Cascade panel → MCPs icon → Marketplace shows memo (only after Codeium approves the outreach in 5a)

If any registry rejects or requests changes, update this doc with the
revised submission so the next release reuses the polished copy.
