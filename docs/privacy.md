# Privacy and Network Policy

memo stores memories, embeddings, caches, and history locally. Normal
`memo-mcp` startup is offline by default: it makes no outbound network request
and does not modify Claude, Codex, or other agent configuration.

## Startup opt-ins

All four startup behaviors default to `0`/false:

- `MEMO_UPDATE_CHECK_ENABLED=1` permits a throttled remote tag check and records
  a local update notification.
- `MEMO_AUTO_UPDATE=1` permits that check and may install a newer tagged memo
  release in the background for the next startup.
- `MEMO_STATUSLINE_SELFHEAL=1` permits memo to repair its statusline entry in
  local Claude settings.
- `MEMO_HOOK_SELFHEAL=1` permits memo to repair its local recall-hook entries.

The two self-heal operations modify local configuration but do not require the
network. Opted-in background errors never prevent the MCP server from starting.

## Explicit network operations

The offline default does not claim that every command is offline. Network access
is expected when the user explicitly requests:

- `memo update` or installation from a remote package/git source;
- `memo sync` against a configured git remote;
- model downloads during installation or the first use of a model not already
  present in the local Hugging Face cache; or
- benchmark downloads for evaluation commands that use external corpora.

Local search, recall, save, history, briefing, and normal MCP startup do not send
prompts or memories to a hosted memo service. A separately configured model,
sync remote, HTTP integration, or external helper has its own privacy boundary.
