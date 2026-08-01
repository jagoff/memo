# MCP resource subscriptions / list_changed — investigated, NOT planned

**Date:** 2026-07-28
**Status:** closed as no-build (MCP gap ranking #4)
**Evidence:** ultracode research sweep (prior-art grep with synonyms, installed
FastMCP/mcp API audit, ecosystem/client-support survey, memo resource-surface
inventory).

## Verdict

memo's MCP resources (`memo://profile/{scope}`, `memo://recent`,
`memo://memory/{id}` — `server_resources.py`) stay **poll-only**. Three
independent lines of evidence:

1. **No target client consumes it.** Claude Code closed the exact feature
   request (including the multi-agent memory use case) as `not_planned`
   (anthropics/claude-code#7252). Claude Desktop never calls `resources/read`
   spontaneously; Cursor and Goose: no support. The only real consumer
   (VS Code `mcpResourceFilesystem`) surfaces resources as editor files, not
   model context. No popular memory MCP server implements it (mem0,
   basic-memory, cognee, letta, zep — zero hits).
2. **The installed API does not offer it.** FastMCP 3.4.4 registers no
   `resources/subscribe` handler (→ "Method not found") and mcp 1.28.1
   hardcodes the `subscribe=False` capability with no knob
   (`mcp/server/lowlevel/server.py:211-213`). The low-level escape hatch would
   need a per-session registry, a capabilities override, and — over HTTP — a
   long-lived SSE stream that memo's `json_response` transport mode
   (`server.py:449-453`) deliberately avoids.
3. **The real cost is cross-process.** Invalidating writes mostly arrive from
   other processes (nightly dream, sibling CLI sessions, sync pull, Obsidian
   hand-edits), so a correct implementation needs a watcher→live-session
   bridge — for a beneficiary that does not exist today.

`notifications/resources/list_changed` is equally moot: memo's resource list
is static (3 templates), so the notification would never legitimately fire,
and the `listChanged` capability is already advertised by FastMCP itself.

## What already covers the use case

- Resource reads are never cached server-side — a poll of `memo://recent` is
  always fresh; for stdio clients the per-prompt recall hook re-injects
  freshness anyway.
- The existing internal channel piggybacks change notices onto tool
  responses: `pending_idle_notification.txt` → `notification` field on the
  core tools → `memo_pop_notification`.
- A future long-lived HTTP consumer can poll cheaply; SQLite
  `PRAGMA data_version` is available as a change-check if ever needed.

## Reevaluation triggers

- Claude Code (or another target client) announces real consumption of
  resource subscriptions.
- The mcp SDK parameterizes `subscribe` (today an upstream hardcode).
- FastMCP exposes first-class subscription APIs.
- A real long-lived HTTP consumer wants `memo://recent` pinned.

Cost of implementing late is low: the subscription spec has been stable
across MCP revisions.
