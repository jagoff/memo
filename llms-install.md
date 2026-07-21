# memo — AI agent installation guide (llms-install.md)

This guide is for AI agents (Cline, Claude, etc.) installing **memo** as an MCP server. memo is a local-first persistent memory: MLX embeddings on Apple Silicon (or CPU sentence-transformers on Linux), sqlite-vec hybrid search, markdown files on disk. No cloud APIs, no Ollama.

## Requirements

- Python **3.13+**
- macOS on Apple Silicon (uses MLX) **or** Linux (uses CPU embeddings)
- `uv` (recommended) or `pipx`

## Install

Install memo as an **isolated tool** (do not add it to a project venv):

```bash
# macOS Apple Silicon (recommended)
uv tool install mlx-memo

# Linux (CPU embeddings; explicit Python avoids an older distro interpreter)
uv tool install --python 3.13 \
  --find-links https://download.pytorch.org/whl/cpu/torch/ mlx-memo

# Alternatives (Apple Silicon; Linux pipx needs the CPU-wheel args in docs/ubuntu.md)
pipx install mlx-memo
brew tap jagoff/memo && brew install mlx-memo
```

This provides two binaries: `memo` (CLI) and `memo-mcp` (MCP server, stdio).

## MCP configuration

Add to the client's MCP settings (e.g. Cline's `cline_mcp_settings.json`):

```json
{
  "mcpServers": {
    "memo": {
      "command": "memo-mcp",
      "args": []
    }
  }
}
```

No API keys required. On first run memo downloads the Qwen3-Embedding model
(about 0.4 GB for the MLX quant or 1.2 GB for Linux CPU); the first tool call
may take a minute.

If `memo-mcp` is not on PATH, use the zero-install form instead:

```json
{
  "mcpServers": {
    "memo": {
      "command": "uvx",
      "args": ["--from", "mlx-memo", "memo-mcp"]
    }
  }
}
```

(On Linux, prefer the isolated `uv tool install` path
above so the Python version and CPU dependency are resolved explicitly.)

## Optional environment variables

| Variable | Purpose |
|---|---|
| `MEMO_DATA_DIR` | Where memory `.md` files live (default `~/Documents/memo`) |
| `MEMO_VAULT_PATH` | Path to an Obsidian vault to ingest as reference knowledge |

Set them in the `env` block of the MCP server entry if needed. Defaults work out of the box.

## Verify

```bash
memo doctor        # checks runtime, model, database
memo save "hello from install test"
memo search "install test"
```

The MCP server exposes tools like `memo_save`, `memo_search`, `memo_ask`, and `memo_unified_briefing`. Full docs: <https://github.com/jagoff/memo>.
