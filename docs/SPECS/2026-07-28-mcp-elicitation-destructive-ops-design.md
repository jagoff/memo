# MCP elicitation on irreversible operations — design

**Date:** 2026-07-28
**Status:** draft — autonomous defaults taken (see §Decisions), revisable at PR review
**Origin:** MCP gap ranking #3: destructive MCP tools execute immediately with
no confirmation; `destructiveHint` annotations are advisory metadata only.

## Prior art (verified — broad-synonym grep, NOT rebuilt here)

- **Zero `elicit` anywhere** in src/tests/docs/hooks/skills.
- CLI has real gates: `memo delete` → `--yes` else `click.confirm(abort=True)`
  (`cli_memory.py:566-587`); `memo backup restore` →
  `@click.confirmation_option` (`cli_backup.py:238-241`); `memo feedback clear`
  → `--yes`. The MCP path has none.
- `server_annotations.py:32-51` `DESTRUCTIVE` preset: advisory, over-applied
  (14 tools incl. reversible `memo_update`/`memo_forget`).
- `skills/memo/SKILL.md:31-36`: "ask for confirmation before delete" — prose,
  unenforced.
- `MEMO_NONINTERACTIVE` never auto-confirms (`flags_misc.py:441`); daemons pass
  `--yes` explicitly. That invariant is preserved here.
- Ecosystem (gh code search, 2026-07): **no memory MCP server uses elicitation**
  (mem0/OpenMemory, cognee, zep/graphiti, claude-mem, basic-memory, official
  `servers/src/memory` — all zero hits). Elicitation appears only outside the
  niche (awslabs aws-api-mcp consent gate — fails closed; semgrep triage;
  github-mcp OAuth). memo would be the **first memory server with in-band
  destructive confirmation**.

## Scope — gate on irreversibility, not on the DESTRUCTIVE annotation

Gated (private list in server code; annotations unchanged):

| Tool | Why irreversible |
|---|---|
| `memo_delete` | `.md` unlinked, no trash (`memory/delete_ops.py:73-199`) |
| `memo_synthesize_delete` | same delete path (`server_synthesis.py:158`) |
| `memo_backup_restore` | overwrites whole store; rollback journal deleted on success (`sync.py:523-716`) |
| `memo_feedback_clear` | hard SQL DELETE of user-signal rows that `reindex` deliberately preserves (`store/feedback_store.py:208-223`) |
| `memo_repo_delete` | `shutil.rmtree` of clone, `remove_clone=True` default (`repo_index.py:684-693`) |
| `memo_cache_evict` | multi-memory `Memory.delete()`, widest single-call blast radius (`cache.py:186-250`) |

Exempt: `memo_forget` / `invalidate` / `supersede` / `update` / `rename` /
`version_rollback` (reversible by design), `cache_flush`, `pop_notification`,
`query_delete` (single JSON entry, no memory data). Over-gating would
contradict memo's reversibility architecture.

## Design

### 1. `src/memo/server_elicit.py` (new)

One helper, used by all six tools:

```python
async def confirm_destructive(ctx, *, action: str, detail: str) -> ElicitOutcome
```

`ElicitOutcome.proceed: bool` + `outcome: "accepted" | "declined" | "cancelled"
| "unsupported" | "disabled" | "error"`.

Flow (probe-verified against installed fastmcp 3.4.4 / mcp 1.28.1):

1. `MEMO_ELICIT_CONFIRM` off → `("disabled", proceed=True)`.
2. Capability check FIRST:
   `ctx.session.check_client_capability(ClientCapabilities(elicitation=ElicitationCapability()))`
   (`mcp/server/session.py:153-154`). Absent → `("unsupported", proceed=True)`
   — **fail-open**: an unguarded `ctx.elicit` raises `McpError` → ToolError
   and would brick the tool for every non-elicitation client.
3. `ctx.elicit(message, response_type=[action, "cancel"])` — list-of-choices
   form; `response_type=None` is deprecated in fastmcp 3.4.4 and hangs VS Code
   (`fastmcp/server/context.py:1165-1175`). Proceed **only** on
   `AcceptedElicitation(data=action)` exactly.
4. `except McpError` → `("error", proceed=True)` (belt and braces; also covers
   clients that advertise but misimplement).
5. Instant-cancel (headless print/SDK clients auto-cancel) → treated as
   `cancelled` abort; the fail-open path for those clients is the capability
   check, which they fail before ever reaching `elicit`.

The elicit `message` states the blast radius, computed from data the tools
already have: delete → `title (type)`, crossref `referenced_by` count, "no
trash — recovery only via backup/git-sync/versions"; restore → store
memory-count being overwritten; feedback_clear → row counts; repo_delete →
clone path; cache_evict → number of memories evicted.

### 2. Gated tools become `async def` + `Context` param

The six tools gain `ctx: Context` (FastMCP-injected, absent from the client
schema) and go async to await the helper. Abort returns a normal result
envelope `{"ok": false, "aborted": "declined"|"cancelled"}` — a user's "no"
is a valid outcome, not a protocol error.

### 3. Decline-as-signal (the exceed — no competitor distinguishes decline from cancel)

Per MCP spec three-action model (2025-11-25 elicitation): **decline** =
explicit refusal; **cancel** = no decision. On decline (and only decline),
save a durable `type=feedback` memory — "user refused <action> of <target>"
with tool + target id/title + timestamp in `extra` — so refusal itself feeds
memo's feedback loop. Cancel is a pure no-op. Flag-gated
`MEMO_ELICIT_DECLINE_SIGNAL`, default on; the write is fail-open (a failed
signal save never blocks the abort).

### 4. Flags (`flags_misc.py`, registry pattern)

- `MEMO_ELICIT_CONFIRM` — bool, default **on**. Scripted elicitation-capable
  clients can opt out.
- `MEMO_ELICIT_DECLINE_SIGNAL` — bool, default **on**.

## Transport caveats (documented, not solved here)

- HTTP daemon runs `json_response=True` for non-SSE (`server.py:449-454`) —
  no mid-call server→client stream, so elicitation is effectively
  stdio/SSE-only. Capability check keeps HTTP fail-open.
- REST `DELETE /api/memory/{id}` (`server_http.py:200`) bypasses MCP entirely —
  out of scope (see Decisions Q6).
- Primary client OK: Claude Code ≥2.1.76 ships form elicitation.

## Testing (in-process, probe-verified)

`fastmcp.Client(server, elicitation_handler=...)` over FastMCPTransport;
omitting the handler makes the client not advertise the capability — that IS
the fail-open fixture (`mcp/client/session.py:166-172`). Matrix per gated
tool (parametrized where the envelope allows):

1. **accept** — handler returns the action word → op executes.
2. **decline** — `ElicitResult(action="decline")` → aborted, nothing deleted,
   decline-signal memory written.
3. **cancel** — aborted, NO signal memory.
4. **no handler** — op executes without confirmation (must-not-brick
   regression).
5. **`MEMO_ELICIT_CONFIRM=0`** — handler never invoked.

Plus: ungated tools (`memo_forget`, `memo_update`) never elicit even with a
handler present; `MEMO_ELICIT_DECLINE_SIGNAL=0` suppresses the signal write.

## Decisions (autonomous defaults — flag any at PR review)

1. **Fail-open, no strict mode.** A `MEMO_ELICIT_REQUIRED` (refuse-with-hint
   when the client lacks the capability, AWS-style fail-closed) is a
   one-flag follow-up if wanted.
2. **Archive-first `memo_delete`** (snapshot `.md` before unlink via
   `lifecycle.archive_memory`) — real defense-in-depth, separate follow-up;
   not bundled to keep this diff reviewable.
3. **Automatic pre-restore backup** for `memo_backup_restore` — same: separate
   follow-up.
4. **No annotation split** (`DESTRUCTIVE` vs `IRREVERSIBLE`): the gate list
   lives privately in `server_elicit.py`; client-facing annotations unchanged.
5. **Decline-as-signal default on**, but flag-gated (writing a memory as a
   side effect of "no" must be switch-off-able).
6. **REST bypass out of scope** — documented above.
7. **`memo_cache_evict` gated** — weakest call in the list (policy-driven
   maintenance op); fail-open keeps daemons unaffected. Drop it from the list
   if review disagrees.
