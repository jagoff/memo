# MCP surface profiles — design

**Date:** 2026-07-28
**Status:** approved (brainstorm), pending implementation plan
**Origin:** gap analysis "qué le falta a memo" (MCP-first, public-product
criterion). Ranked #1 of 5: the MCP surface exposes ~135 tools to every
client with no way to slim it.

## Problem

`build_server()` registers every `server_<domain>.py` unconditionally: ~135
tools reach every MCP client on every connection. Most clients inject all
tool schemas into model context (tens of thousands of tokens per request);
some degrade or cap beyond ~40-50 tools. The ~12 tools that matter for the
memory-first contract (`memo_unified_briefing`, `memo_search`, `memo_save`,
…) are buried among ~120 that a typical agent never calls. The
`install-mcp` agent presets (`runtime/agent_presets.py`) only choose which
client config file to write — nothing manages the tool surface itself.

Non-goals (explicitly out of scope, tracked as separate gaps from the same
analysis): meta-tool gateway / progressive disclosure (approach C, possible
phase 2), ambient-recall parity for MCP-only clients, contradiction-aware
`memo_ask`, elicitation on destructive ops, MCP subscriptions/notifications.
No dynamic post-startup surface changes (`list_changed`).

## Design

### Profiles — `src/memo/mcp_profiles.py` (new)

`PROFILES: dict[str, frozenset[str]]` holds the curated profiles (`core`,
`read`); `full` is a reserved sentinel name, not a `PROFILES` entry:

- **`core`** (~14 tools): `memo_unified_briefing`, `memo_search`,
  `memo_ask`, `memo_chat_ask`, `memo_save`, `memo_get`, `memo_list`,
  `memo_update`, `memo_forget`, `memo_context`, `memo_feedback_record`,
  `memo_feedback_flag`, `memo_stats`, `memo_health_summary`.
  `memo_delete` is deliberately NOT in core — destructive; `memo_forget`
  is the reversible path. Exact list is CI-guarded (see Testing), capped at
  20.
- **`read`** (~8 tools): read-only subset of core — `memo_unified_briefing`,
  `memo_search`, `memo_ask`, `memo_chat_ask`, `memo_get`, `memo_list`,
  `memo_context`, `memo_stats`. Every member must carry
  `readOnlyHint=True`.
- **`full`**: sentinel — no filtering, the entire surface (current
  behavior). **Default.**

`resolve_active_set() -> tuple[frozenset[str] | None, frozenset[str]]`
returns `(base, excluded)`: `base` is `None` for `full` (no allowlist).
`MEMO_MCP_INCLUDE` / `MEMO_MCP_EXCLUDE` are comma-separated tool names:
include adds to `base` (no-op under `full`); exclude applies under **any**
profile, including `full`, and wins on conflict. Gate predicate:
`(base is None or name in base) and name not in excluded`.

### Enforcement seam — `annotated_tool()`

`server_annotations.annotated_tool()` becomes profile-aware: it resolves
the active set once per process and returns a decorator that registers the
function only when `fn.__name__` is in the set (or the set is `None`).
The 158 call sites across 42 `server_*` modules are untouched — no tool
uses a `name=` override, so `fn.__name__` is the tool name everywhere.
The single bare `server.tool()` registration (`mcp_tools.py:248`, dynamic
wrapper) goes through the same gate.

MCP **resources** (`server_resources.py`: `memo://profile`,
`memo://recent`, `memo://memory/{id}`) are not filtered — three cheap
resources, orthogonal concern.

### Configuration

- New flag `MEMO_MCP_PROFILE` in `flags_misc.py` (registry + typed
  accessor), markdown-config key `mcp.profile`. Standard resolution:
  env var > markdown config > built-in default `full`.
- `MEMO_MCP_INCLUDE` / `MEMO_MCP_EXCLUDE` likewise registered flags,
  default empty.
- `memo config validate` validates the profile name against
  `PROFILES.keys() | {"full"}`.

### install-mcp integration

`memo install-mcp` gains `--profile {core,read,full}`. When given, the
generated client config includes `MEMO_MCP_PROFILE` in the server `env`
(same mechanism as the existing `MEMO_SOURCE` injection in
`agent_presets.py`). **Default stays `full`** — decided conservative: no
behavior change for regenerated configs; `core` is opt-in and recommended
in docs/README.

### Observability

- `memo_version` (MCP tool) adds `profile` and `tools_exposed` fields.
- `memo doctor` prints active profile + exposed-tool count, so "why is
  tool X missing" is answerable in one step.

## Error handling

- Unknown profile name (env or md config) → **fail-fast**: `MemoError` at
  server startup listing valid profiles. Rationale: a crash with a clear
  stderr message in the client's MCP log is more discoverable than a
  silently wrong surface; `memo config validate` catches it pre-flight.
- Unknown tool names in INCLUDE/EXCLUDE → stderr warning, non-fatal (may
  reference tools from another memo version).

## Data flow

`memo-mcp` startup → `resolve_active_set()` once → `build_server()` runs
all `register()` calls normally → gate in `annotated_tool` admits only the
active set → FastMCP only ever holds the filtered surface → client lists N
tools. No post-startup mutation.

## Testing

- **Unit** (`tests/test_mcp_profiles.py`): resolution precedence
  (env > md config > default), include/exclude application + exclude-wins,
  unknown profile raises `MemoError`, unknown include/exclude warns.
- **Integration**: `build_server()` under `MEMO_MCP_PROFILE=core` →
  `list_tools()` equals the core set exactly; under `read` → every listed
  tool has `readOnlyHint=True`; unset → full surface (count ≥ current).
- **Drift guard (CI)**: build the full server, collect real tool names,
  assert every name in every profile is a subset — a tool rename breaks
  the test, not the user. Assert `len(core) <= 20` so core stays lean.
- Existing `server_*` test modules keep stubbing `server.tool` untouched
  (the gate's fallback path preserves the zero-arg stub behavior).

## Decision record

| Decision | Choice | Alternative rejected |
|---|---|---|
| Mechanism | Curated name sets at the `annotated_tool` seam | FastMCP tags (10× churn, TypeError fallback swallows tags in test stubs); meta-tool gateway (untyped dispatch, loses destructive hints) |
| Default profile | `full` (no breakage) | `core` default for new preset installs |
| Unknown profile | Fail-fast at startup | Fail-open to full (silent wrong surface) |
| `memo_delete` in core | Excluded | Included (destructive without elicitation — separate gap) |
