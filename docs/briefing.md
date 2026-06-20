# El Briefing — unified session-start panel

`memo briefing` is the `SessionStart` hook output: a markdown panel
the agent (Claude Code, Cursor, …) sees at the top of every new
session. It surfaces the state of the user's work without forcing the
agent to issue any extra MCP calls.

## Sections

By default the panel composes the following, top to bottom:

1. **Última sesión en este proyecto** — last session-checkpoint that
   matches the current `cwd`, plus the `claude --resume` command.
2. **Estado actual (Synapse)** — top-3 `present_state` items from
   `synapse packet` (memflow handoffs, current focus, attention
   queue). Only present when synapse is reachable + returns data.
3. **Conflictos abiertos** — top-3 open `reality_conflicts` from the
   same packet, with severity + freeze-write flag.
4. **Loops abiertos** — recently updated memorias (`memo store
   list_recent`), windowed to the last N days.
5. **Memoria del día** — date-seeded pick from the oldest-updated
   memorias so the corpus is sampled over time.
6. **Interaction guide** — `dame el loop N` / `/memo get` /
   `/memo ask` shortcuts.

A trailing `_Synapse: ready · trace=<short>_` line is appended when
section 2 or 3 fired, so the agent can spot trace_id at a glance.

## Synapse fallback

The synapse subprocess goes through `memo.synapse_client.get_packet`,
which returns `None` on any of:

- `synapse` binary not on PATH (or `MEMO_SYNAPSE_EXECUTABLE` mis-set)
- non-zero exit code
- bad / truncated JSON
- timeout (`MEMO_SYNAPSE_CLIENT_TIMEOUT`, default 8s)

When `None` is returned, the briefing skips sections 2+3 and the rest
of the panel is identical to the pre-GC5 output. Zero regression for
single-Mac users without synapse.

Forcing the local-only fallback for one session:

```bash
MEMO_BRIEFING_SYNAPSE_DISABLE=1 memo briefing
```

## MCP surface

The same composer is exposed as `memo_unified_briefing`:

```python
result = mcp.call("memo_unified_briefing", cwd="/abs/path")
# {"available": bool, "markdown": "<lines>", "lines": [str, ...]}
```

`available=False` + empty markdown when synapse is unreachable;
callers should layer their own context after.

## Environment knobs

| Variable | Default | Purpose |
|---|---|---|
| `MEMO_BRIEFING_DISABLE` | unset | `1` skips the whole panel (emit `{}`) |
| `MEMO_BRIEFING_SYNAPSE_DISABLE` | unset | `1` skips only the synapse sections |
| `MEMO_BRIEFING_LOOPS_N` | `5` | how many open loops to show |
| `MEMO_BRIEFING_LOOPS_DAYS` | `7` | window for "open" loops |
| `MEMO_BRIEFING_DEBUG` | unset | `1` prints failures to stderr |
| `MEMO_SYNAPSE_EXECUTABLE` | unset | override the `synapse` binary path |
| `MEMO_SYNAPSE_CLIENT_TIMEOUT` | `8.0` | synapse subprocess timeout |

## Why this design

The panel is markdown, not JSON, because the agent reads it directly.
Each section is short (≤3 items) so the briefing stays under a few
hundred tokens — well below the recall-hook 5s budget shared with
prewarm and recall-daemon.

Synapse owns brain/front-door routing; memo owns the corpus +
operation receipts (see `docs/receipts.md` + `docs/contradict-loop.md`).
The briefing is a *read* — it composes synapse output into the local
panel without any orchestration logic on memo's side, which keeps the
boundary clean per the README's Experimental modules note.
