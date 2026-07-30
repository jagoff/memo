# Task 1 brief — safe snapshots, capability manifest, and inventory

BASE: `90a7144e04366e36201e7d366d928151b5c80d17`

Normative source:
`docs/superpowers/plans/2026-07-29-memflow-absorption/03-cutover-readiness.md`,
Task 1 and global constraints.

Status: implementation starting. No acceptance may be claimed without an
independent specification/durability/quality PASS.

## Scope

- Add the private `tools.memflow_absorption` package with strict schemas,
  attempt-root safety, immutable read-only snapshot receipts, signed
  capability-manifest construction, consumer inventory, Synapse retirement
  manifest, control-record verification, and dry-run-first operator commands.
- Add sanitized fixtures and the five normative tooling test modules.
- Reuse Plan 01 canonical JSON, Ed25519 signer/verifier, verification roster,
  and descriptor-relative atomic I/O. Add only the cutover signature domains
  required by this task.

## Closed safety boundaries

- No live MCP calls, LaunchAgent mutations, service restarts, configuration
  replacement, quiesce, activation epoch, or deletion.
- Apply writes only under an exact
  `.../memo/cutover/<attempt-id>/` root with a matching sentinel.
- Reject broad roots, repository roots, home, symlink components, unresolved
  variables, and any destination outside the validated attempt root.
- Snapshot only explicit regular files with `O_NOFOLLOW`; write receipt and
  payload immutably, fsync both, and publish read-only.
- Usage exclusions require exact signed event/attempt IDs. Client names,
  topics, and suffix heuristics never exclude evidence.
- `frozen=True` requires exact inclusive 90-day bounds, complete and fresh
  signed proof from both configured machines, no coverage gaps, no unknown or
  ambiguous traffic, exactly one disposition per source operation, complete
  route/parity/deletion proofs, and qualifying SLO baselines.
- Operator commands are dry-run by default and are not added to Memo's public
  CLI/MCP surfaces.

## Required evidence

1. Real RED from the five missing tooling modules.
2. Focused tooling tests, Ruff, strict mypy, and proportional non-slow suite.
3. Frozen Plan 01 v1 source hashes unchanged.
4. Exact technical commit and review range.
5. Explicit statement that Memflow and every live service remained untouched.
