---
description: Search, save, inspect, and maintain memo persistent memory.
argument-hint: "[query|stats|list|save|get|update|delete|reindex|doctor]"
---

# /memo

Route `$ARGUMENTS` to the local `memo` MCP server. Use MCP tools whenever
the active profile exposes them; use the isolated `memo` CLI for maintenance
commands that are intentionally absent from the default 13-tool agent profile.

## Preflight

If MCP tools named `mcp__memo__memo_*` are not available, tell the user to
register memo first:

```bash
memo mcp-command --client codex
# for Devin Desktop:
memo install-slash --client devin-desktop
```

Then open a new Codex session so MCP tools reload.

## Routing

- Empty arguments: smart-capture the actionable insight from the current turn
  and call `mcp__memo__memo_save`.
- `stats`: call `mcp__memo__memo_stats` when available; otherwise run `memo stats`.
- `list [n]`: call `mcp__memo__memo_list` when available; otherwise run `memo list --limit n`.
- `save <text>`: derive a short title, type, and tags, then call
  `mcp__memo__memo_save`.
- `get <id|prefix>`: call `mcp__memo__memo_get`.
- `update <id|prefix> ...`: call `mcp__memo__memo_update` when available; otherwise run `memo edit`.
- `delete <id|prefix>`: ask for explicit confirmation, then call
  `mcp__memo__memo_delete` when available; otherwise run `memo delete --yes`.
- `reindex`: call `mcp__memo__memo_reindex` when available; otherwise run `memo reindex`.
- `doctor [--gc] [--fix]`: run `MEMO_NONINTERACTIVE=1 memo doctor ...`.
- Anything else: semantic search via `mcp__memo__memo_search` with
  `limit=5` and `body_chars=280`.

Keep output compact. For search results, show score, type, title, updated date,
and id prefix. For saves/updates/deletes, show the resulting id and title.
