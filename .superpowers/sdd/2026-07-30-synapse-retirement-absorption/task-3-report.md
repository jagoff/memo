# Task 3 report — Memo-native Synapse parity gate

## Delivered

- Added an offline parity runner that consumes only `CapabilityManifest` plus
  redacted `ParityFixture` values; it never imports or executes Synapse.
- The runner canonicalizes source IDs, compares answer versus abstention state
  and provenance, records per-fixture latency, reports p50/p95, and converts
  every mismatch into an explicit `gap_id` / `blocked` result.
- Absorption is fail-closed unless the signed manifest records a mapped
  Memo-native route, at least one observed usage receipt, and a named Memo
  target.  Where an admitted SLO baseline exists, a fixture exceeding its p95
  tolerance is also blocked.
- Added deterministic parity cases for a Memo-native federated route, an
  unmapped operation, canonicalized provenance, abstention, and absent usage
  evidence.  The fixture corpus remains redacted (query/status/source IDs only).
- Added `EvidencePack` regressions proving legacy provenance is exposed as
  Memo-native provenance and an empty result is an explicit abstention.

## Disposition

No reproducible admitted chat delta required a change to `ask_ops.py`,
`evidence_ops.py`, or `server_core_search.py`.  The existing Memo `search` and
`evidence_pack` contracts satisfy the admitted fixture paths; the new runner
leaves any future unmapped route as a signed blocker rather than creating an
undocumented fallback API.

## Verification

- `uv run --no-sync pytest tests/tools/test_synapse_parity.py tests/test_memory_ask.py tests/test_memory_evidence_pack.py -v` — 26 passed.
- `uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force` — completed; best configuration remained `D hyb/0.40/ctx` (precision@5 0.757, noise@5 0.0).
- `uv run --no-sync ruff check src/memo/memory/ask_ops.py src/memo/memory/evidence_ops.py src/memo/server_core_search.py tools/memflow_absorption tests` — passed.
- `uv run --no-sync mypy src/memo` — passed, 463 source files.

## Concerns

None.  Pre-existing changes to unrelated SDD ledgers/reports were preserved.

## Correction round 1

- The facade now resolves `memo_unified_briefing` through Memo-native briefing
  helpers (or a supplied Memo facade implementation) without importing
  Synapse.  It also evaluates native conflict, session, and health surfaces,
  preserving their `answered`, `insufficient_evidence`, `conflicted`, and
  `error` statuses plus native source/provenance IDs.
- Routes are no longer flattened: the runner evaluates each signed closed
  predicate against fixture parameters, maps only the selected route's
  parameters, applies its defaults, and blocks when no route matches.
- Before a fixture timer starts, the runner now requires a real
  `VerificationRoster` and calls `verify_capability_manifest`.  Missing roster,
  unfrozen/blocking authority, a digest mismatch, or an invalid signature raise
  `ParityManifestError`; no parity report is emitted for an unverifiable
  manifest.

Correction verification: focused parity plus ask/evidence regressions 34
passed; Ruff passed; Mypy passed for 464 source files.

## Correction round 2

- `memo_ask` parity now derives its status from Memo's native result before
  inspecting source IDs: explicit errors remain `error`; an explicit
  `abstained="disputed"` result remains `conflicted`; other abstentions remain
  `insufficient_evidence`.  Sources are still retained as provenance and never
  upgrade either state to `answered`.
- The real-Memory unified-briefing fallback now returns full, structured native
  `source_ids` alongside its Markdown and lines.  This keeps briefing
  provenance available to `_source_ids` even when no injected
  `unified_briefing` facade method exists.

Correction verification: 37 focused parity/ask/evidence tests passed; Ruff
passed; Mypy passed for 464 source files.
