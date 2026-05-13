# `@fer/paperclip-plugin-memo`

[Paperclip](https://github.com/paperclipai/paperclip) plugin that bridges the
local [`memo`](../..) MCP memory server into Paperclip as agent tools, so any
agent in a Paperclip company (Claude Code, Codex, OpenClaw, HTTP/webhook bots)
can recall, save, list, and ask over the operator's persistent memory store.

## What it exposes

Five agent tools, namespaced by the host as
`memo.paperclip-plugin-memo:<tool>`:

| Tool | Purpose |
| --- | --- |
| `memo_search` | Top-k hybrid (vec + bm25) search over memorias. |
| `memo_save` | Persist a new memory; auto-derives title/type/tags optionally. |
| `memo_list` | Most recent memorias by `updated` desc. |
| `memo_get` | Fetch one memory by id (git-style prefix ≥ 4 chars OK). |
| `memo_ask` | RAG synthesis with `[id]` citations using the local 7B chat model. |

Plus a small dashboard widget showing `memo stats` (total memorias, vault path,
embedder + LLM in use).

## Requirements

- Apple Silicon (memo is MLX-native).
- [`memo`](../..) installed and on `$PATH`. Verify with `memo doctor`.
- A running Paperclip instance (≥ 0.3.x) on `127.0.0.1:3100`.

## Install

```bash
# 1. Build the plugin
cd ~/repositories/memo/integrations/paperclip-plugin-memo
pnpm install
pnpm build

# 2. Make sure Paperclip is running
cd ~/repositories/paperclip
pnpm dev   # or `pnpm dev:server` for API-only

# 3. Register the plugin from its absolute path (in another shell)
~/repositories/memo/integrations/paperclip-plugin-memo/scripts/install.sh
```

The install script POSTs to `http://127.0.0.1:3100/api/plugins/install` with
`isLocalPath:true`. The host watches local-path plugins for file changes, so
`pnpm dev` rebuilds the plugin in place and the worker auto-restarts.

## Config

Operator settings (via Paperclip's plugin settings UI, or
`paperclipai plugin inspect memo.paperclip-plugin-memo`):

- `memoBinary` — path to `memo` (default `memo` on `$PATH`).
- `defaultSearchLimit` — default top-K for `memo_search` (default `5`).
- `defaultSearchMode` — `hybrid` | `vec` | `bm25` (default `hybrid`).

## Develop

```bash
pnpm dev            # esbuild watch
pnpm typecheck
pnpm test           # vitest — uses /bin/echo as a fake memo binary
pnpm build:rollup   # alternative rollup-based build
```

The scaffold snapshots `@paperclipai/plugin-sdk` and `@paperclipai/shared` from
the local Paperclip checkout at `~/repositories/paperclip/packages/plugins/sdk`.
The packed tarballs live in `.paperclip-sdk/`. Before publishing this plugin,
switch those dependencies to published npm versions.
