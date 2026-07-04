# memo — local memory for AI agents

Registers **[memo](https://github.com/jagoff/memo)** as an MCP server in VS Code's
**agent mode**, so Copilot (and any MCP-aware chat) can remember decisions, facts,
and preferences across sessions — stored **locally**, on your own machine.

- 🧠 **Long-term memory** — hybrid `sqlite-vec` + BM25 retrieval with reranking, a
  knowledge graph, and temporal ("time-machine") queries.
- 🍎 **Local & private** — MLX embeddings on Apple Silicon, CPU on Linux. Markdown
  files on disk are the source of truth. **No cloud, no API keys.**
- 🔌 **Zero config** — the extension contributes the `memo` MCP server automatically.
  Open agent mode, pick **memo**, and its tools appear.

## Requirements

memo runs as a local process. By default the extension launches it with:

```
uvx --from mlx-memo memo-mcp
```

So you need **[uv](https://docs.astral.sh/uv/getting-started/installation/)** installed
(`uvx` ships with it). `uvx` fetches the `mlx-memo` package from PyPI on first run — no
manual `pip install` needed.

Already installed memo yourself (`pipx install mlx-memo` or `uv tool install mlx-memo`)?
Turn on **`memo: Use Installed Binary`** in settings and the extension will run your
`memo-mcp` directly.

## Usage

1. Install this extension.
2. Open the Chat view and switch to **Agent** mode.
3. Click the tools icon → **memo** should be listed. Enable it.
4. Ask the agent to remember or recall something — memo persists it to disk.

Run **`memo: Setup & Requirements`** from the Command Palette for install help.

## Settings

| Setting | Description |
| --- | --- |
| `memo.useInstalledBinary` | Run an already-installed `memo-mcp` instead of `uvx --from mlx-memo`. |
| `memo.dataDir` | Directory for memo's store and index (`MEMO_DATA_DIR`). Empty = default. |
| `memo.vaultPath` | Store memories under an Obsidian vault (`MEMO_VAULT_PATH` + `MEMO_MEMORIES_IN_VAULT`). |

## About

memo is open source (MIT) and distributed on PyPI as
[`mlx-memo`](https://pypi.org/project/mlx-memo/). Source, issues, and full docs:
**https://github.com/jagoff/memo**
