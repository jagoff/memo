# Contradiction loop (memo + synapse)

`memo contradict` maintains a persistent radar over the corpus — pairs
of memories that the LLM verdict marks as `contradiction` or
`evolution`, lifecycle-tracked in a sidecar SQLite DB. The triage
walker turns them into resolutions (fuse / keep newer / dismiss / etc.).

This document covers the lifecycle, the CLI, and **what the loop with
Synapse looks like** — because there is a common misconception that
memo pushes contradictions to Synapse's Reality Workbench. It does not.

## Local lifecycle

```bash
memo contradict scan [--since 2026-04-01] [--type decision]   # detect
memo contradict list [--status open] [--json]                 # inspect
memo contradict triage                                        # walk + resolve
memo contradict stats                                         # counts
memo contradict reopen <pair-id>                              # re-queue
```

Pair lifecycle: `open → fused | kept_newer | kept_older | evolved | dismissed`.
Resolutions are written to the same `contradictions.db` sidecar so the
same pair never re-litigates after a final verdict (`upsert_open()`
skips already-resolved rows).

Scanning is LLM-driven: every promising neighbor pair (vec cosine ≥
`--sim-floor`, `--min-days-apart` apart) gets classified by the helper
LLM. That makes `scan` expensive — typically a once-daily cron / `launchd`
job, not a per-prompt hook.

## Direction of the synapse loop

Synapse exposes **no** `conflicts register --source <backend>` verb
today. Its `conflict` subcommand only manages the lifecycle of conflicts
it already knows about: `list | acknowledge | resolve | archive |
escalate`.

Instead, Synapse **pulls** memo's contradictions on demand:

```
synapse packet --query Q --json
   └── synapse.conflict_sync.collect_external_conflicts()
        ├── project_memo_contradictions()
        │     └── MemoBackend.contradict_list()
        │           └── memo contradict list --json --status open
        └── project_memflow_markers()
              └── MemflowBackend.list_markers(...)
```

Each row memo returns is projected into a `RealityConflict` with
`source_backend="memo"`, `source_uri="memo://contradiction/<pair-id>"`,
and EvidenceRefs for both memories.

That projection runs *every time synapse builds a packet*. Memo just
has to keep `contradictions.db` fresh — the rest is synapse's job.

## How the other direction works (freeze)

Synapse → memo is the half memo controls. When a `RealityConflict` is
marked `freeze_write=true` (`lifecycle ∈ {detected, acknowledged}`),
synapse expects writers to refuse new memories on that topic until the
conflict is resolved.

Memo respects the freeze via the GC4 protocol:

```python
mem.save(content="...", extra={"synapse_trace_id": "..."},
         respect_synapse_freeze=True)
# → raises WriteRefused if synapse reports a blocking conflict
```

See `docs/synapse-adapter.md` for the full payload.

## Why no push

A `memo → synapse` push would need:

1. A synapse verb that accepts external signals (`conflicts register`
   or equivalent). Does not exist today.
2. Stable, dedup-safe contradiction IDs that cross the boundary
   (`memo://contradiction/<id>` already is, but synapse has to own the
   uniqueness contract).
3. Lifecycle reconciliation when memo resolves a pair locally —
   today's pull model handles this implicitly (the row disappears from
   `--status open`).

Until that verb lands on synapse's side, the pull-based model gives
identical surface in the Reality Workbench at lower coupling. If you
find yourself adding `_register_with_synapse()` to `contradict.py`,
stop — the loop is already closed.

## Environment knobs

| Variable | Default | Purpose |
|---|---|---|
| `MEMO_RESPECT_SYNAPSE_FREEZE` | unset | `1` enables synapse freeze-check on writes carrying a `synapse_trace_id` |
| `MEMO_SYNAPSE_EXECUTABLE` | unset | Override path to the `synapse` binary |
| `MEMO_SYNAPSE_CLIENT_TIMEOUT` | `8.0` | Subprocess timeout for synapse calls |
