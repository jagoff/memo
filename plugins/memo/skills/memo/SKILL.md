---
name: memo
description: "Route `/memo` requests to the local memo MCP memory server. Use for searching, asking, saving, listing, getting, updating, deleting, reindexing, stats, and doctor commands. Prefer `mcp__memo__memo_*` tools; use the `memo` CLI only for maintenance commands not exposed over MCP."
argument-hint: "(empty = smart capture) | <query> | ask <question> | list [n] | save <text> | get <id|prefix> | update <id|prefix> [flags] | delete <id|prefix> | stats | reindex | doctor [--gc] [--fix]"
---

# /memo

The user invoked `/memo`. Parse the text after `/memo` and route it to the
local memo MCP server.

## Preflight

If `mcp__memo__memo_*` tools are unavailable, tell the user to run:

```bash
memo install-slash --client codex
# for Devin Desktop MCP config:
memo install-slash --client devin-desktop
```

Then start a new Codex session so plugin skills and MCP tools reload, or
restart Devin Desktop after editing its MCP config.

## Routing

- Empty arguments: smart-capture one actionable insight from the recent
  conversation and call `mcp__memo__memo_save`.
- `stats`: call `mcp__memo__memo_stats`.
- `list [n]`: call `mcp__memo__memo_list` with `limit=n` or `20`.
- `ask <question>`: call `mcp__memo__memo_ask` when available; otherwise
  search first, then answer with cited memory ids.
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
and id prefix. For saves, updates, and deletes, show the resulting id and
title.
