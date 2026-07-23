# Native conflict lifecycle

Memo owns contradiction detection, conflict lifecycle, and write-freeze
enforcement. No external conflict ledger participates in the write path.

## Durable knowledge contradictions

`memo contradict` analyzes the durable corpus and stores relations locally.
Supersession remains reversible: Markdown is the source of truth and the SQLite
projection can be rebuilt with `memo reindex --rebuild`.

## Operational conflicts

Agents can open a conflict in the Memo operation journal, optionally freezing
writes for its topic:

```bash
memo operational conflict open \
  deployment \
  "Production state is disputed" \
  --freeze

memo operational state
memo operational conflict resolve \
  <conflict-id> \
  "Verified against the production deployment" \
  --actor agent:operator
```

The journal is append-only and hash chained. `memo operational verify` checks
its integrity. A resolution appends a new event; it never rewrites history.

The MCP tools are `memo_conflict_open`, `memo_conflict_resolve`,
`memo_operational_state`, and `memo_journal_verify`. Resolution remains a
local-human authority boundary: `memo_conflict_resolve` is read-only and
returns the exact CLI action required instead of mutating the conflict.

## Write policy

Every save resolves a Memo-native write policy before committing. An active
matching freeze refuses the write with a Memo domain error. Decisions are
recorded with trace and actor context in the operation ledger, making the
result explainable without relying on a sibling service.

Legacy provenance and environment names are accepted only by
`memo migrate-independence` so existing vaults can be normalized. They do not
activate external runtime behavior.
