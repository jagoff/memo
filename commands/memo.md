---
description: Search, save, inspect, and maintain memo persistent memory.
argument-hint: "[query|stats|list|save|get|update|delete|reindex|doctor]"
---

# /memo

Route `$ARGUMENTS` to the local `memo` MCP server. Use MCP tools whenever
they are available; use the `memo` CLI only for maintenance commands that are
not exposed as tools.

## Preflight

If MCP tools named `mcp__memo__memo_*` are not available, tell the user to
register memo first:

```bash
memo install-slash --client claude-code
# or only print the MCP command:
memo mcp-command --client claude-code
# other client installers:
memo install-slash --client codex
memo install-slash --client devin-desktop
```

Then open a new Claude Code session so the `/` menu and MCP tools reload.

## Routing

- Empty arguments: smart-capture the actionable insight from the current turn
  and call `mcp__memo__memo_save`.
- `stats`: call `mcp__memo__memo_stats`.
- `list [n]`: call `mcp__memo__memo_list` with `limit=n` or `20`.
- `save <text>`: derive a short title, type, and tags, then call
  `mcp__memo__memo_save`.
- `get <id|prefix>`: call `mcp__memo__memo_get`.
- `update <id|prefix> ...`: call `mcp__memo__memo_update`.
- `delete <id|prefix>`: ask for explicit confirmation, then call
  `mcp__memo__memo_delete`.
- `reindex`: call `mcp__memo__memo_reindex`.
- `doctor [--gc] [--fix]`: run `MEMO_NONINTERACTIVE=1 memo doctor ...`.
- Anything else: semantic search via `mcp__memo__memo_search` with
  `limit=5` and `body_chars=280`.

Keep output compact. For search results, show score, type, title, updated date,
and id prefix. For saves/updates/deletes, show the resulting id and title.
