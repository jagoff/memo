# Engram Default Activation Proof

**Date:** 2026-07-23

**Result:** complete

**Implementation:** `55ba2df9`

**Legacy-vault compatibility:** `7e1c7f5a`

## Activated defaults

| Capability | Default | Persistent local setting | Rollback |
|---|---:|---|---|
| Relation candidates | on | `relations.relation_candidates_enabled=true` | set false or `MEMO_RELATION_CANDIDATES_ENABLED=0` |
| Judged relation annotations | on | `relations.relation_annotations_enabled=true` | set false or `MEMO_RELATION_ANNOTATIONS_ENABLED=0` |
| MCP write coordinator | 32 | `mcp.write_queue_size=32` | set 0 or `MEMO_MCP_WRITE_QUEUE_SIZE=0` |

The relation flags were removed from the dark-flag and graduation registries.
Internal post-save candidate retrieval uses `_track_usage=False`, so automatic
derivation does not increment access counters or create co-recall signals.

## Compatibility issue found by dogfooding

The live vault had the experimental session-era relation table at schema
version 7:

- `id INTEGER PRIMARY KEY`
- `source_id INTEGER`
- `target_id INTEGER`

Canonical relation IDs are strings, so a real judgment initially failed with
SQLite `datatype mismatch`. Schema version 8 now transactionally rebuilds that
table with text identity columns, preserves all rows and lookup aliases,
recomputes pair keys, orphans invalid/duplicate pairs, and recreates its
indexes.

An SQLite backup of the real database proved the exact migration before it was
applied:

| Check | Before | After |
|---|---|---|
| `PRAGMA user_version` | 7 | 8 |
| relation identity types | `INTEGER` | `TEXT` |
| new candidate | rejected | `rel-05572d36c28a63a7d0175fde`, pending |

The real vault is now at schema version 8.

## End-to-end product proof

The activation outcome was saved through memo as
`b4b8c9fe6555441bac584c33988d2c67`. It was explicitly compared with the prior
Engram completion memory `7d89cd257bab4d279b9f04c89246aaaf`.

The canonical judgment is:

- relation ID: `rel-4f33c1507b91ebb91794c817`
- verb: `related`
- status: `judged`
- confidence: `1.0`

A normal hybrid retrieval returned the activation memory first and attached
that judgment under `extra.memory_relations`. Pending rows were not exposed.

A newly constructed FastMCP server over the real vault reported:

```text
write queue: enabled=true, capacity=32, depth=0, failed=0
search hit: b4b8c9fe6555441bac584c33988d2c67
annotation: rel-4f33c1507b91ebb91794c817 (related, judged)
```

The mutating middleware was also exercised through a real FastMCP client over
isolated storage. One `memo_save` changed the coordinator counters from zero to
`submitted=1`, `started=1`, `completed=1`, with `failed=0`; measured queue wait
was 0.026 ms.

The MCP process that was already connected before these settings changed keeps
its construction-time queue capacity until restart. Persistent configuration
is in place, and a new server process demonstrated all three capabilities
together.

## Verification

| Gate | Result |
|---|---|
| `ruff check src/ tests/` | passed |
| `mypy src/memo` | passed, 431 source files |
| non-slow pytest suite | 5,288 passed, 29 skipped |
| coverage | 76.09% (required 74%) |
| slow pytest suite | 7 passed, 4 skipped |
| relation policy gate | 12 cases; recall 1.000, precision 1.000, noise 0.000 |
| recall regression | 164 searches; baseline configuration remained recommended |
| config validation | 8 configured flags valid |
| runtime doctor | `memo` and `memo-mcp` resolve to the same project venv |

The doctor retains its advisory preference for an isolated tool install rather
than a repository venv. That warning does not represent a runtime mismatch and
does not affect this activation.
