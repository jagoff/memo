# Session briefing

`memo briefing` is the native session-start view for durable knowledge and
operational continuity. It reads only Memo-owned Markdown, SQLite indexes, and
the local hash-chained operation journal.

## Sections

The full briefing can include:

- memories relevant to the current project;
- active focus and unconsumed handoffs;
- unacknowledged attention items;
- unresolved conflicts, including write freezes;
- proactive suggestions already computed by Memo.

The operational section ends with the current journal head so an agent can
verify that continuity came from the locally validated event chain.

## Commands

```bash
memo briefing
memo briefing --compact
memo operational state
memo operational verify
```

`--compact` is the latency-sensitive SessionStart surface. The full command is
intended for interactive inspection.

## MCP surface

`memo_unified_briefing` returns the same native view and accepts `source` so
consults remain attributable. Operational state is also available through
`memo_operational_state`, with focused mutation tools for focus, handoffs,
attention, and conflicts.

All of these tools operate without another daemon, repository, binary, or
private Python package.

## Failure behavior

Briefing is a read path and degrades safely: a corrupt optional section is
omitted rather than blocking session startup. Use `memo operational verify` and
`memo doctor` to diagnose the underlying journal or store.
