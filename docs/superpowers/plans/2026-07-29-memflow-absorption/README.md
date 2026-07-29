# Memflow Absorption Program

**Approved design:** `docs/superpowers/specs/2026-07-28-native-memflow-absorption-design.md`

This program retires Memflow by moving only its proven, live-used behavior into
Memo as native functionality. It is split into five implementation plans because
the approved design spans independently reviewable subsystems. No plan creates a
compatibility package or public `flow_*` alias.

## Plan Set

1. [`01-operational-ledger-v2.md`](01-operational-ledger-v2.md) evolves Memo's
   existing operational authority into a compactable, signed v2 journal with
   transactional derived views and durable idempotency.
2. [`02-live-coordination-runtime.md`](02-live-coordination-runtime.md) builds
   channels, delivery/ACK, presence, continuity, terminal delivery, continuous
   operational sync, and the Memo operational daemon on ledger v2.
3. [`03-cutover-readiness.md`](03-cutover-readiness.md) freezes the capability
   and consumer manifests, adds Memflow fencing and drain, isolates Synapse, and
   stages Memo-only configurations without changing the active authority.
4. [`04-active-state-migration.md`](04-active-state-migration.md) consumes the
   frozen 90-day capability manifest, migrates missing durable knowledge plus
   active operational state, and produces reproducible rehearsal evidence.
5. [`05-atomic-cutover-retirement.md`](05-atomic-cutover-retirement.md) builds
   the signed CAS controller, rehearses rollback, performs the activation epoch,
   verifies Memo-only operation, and removes Memflow.

## Dependency Graph

```text
01 ledger v2 ───────> 02 live runtime ───────┐
       │                                      │
       └──────────────────────> 04 migration ─┼──> 05 cutover/retirement
03 readiness/Synapse ─────────────────────────┘
```

Plan 03 Task 4 may begin in parallel with Plan 01. Plan 03 Tasks 1–3 consume
Plan 01's permanent signing/record contracts. Its signed frozen capability
manifest is the mandatory admission gate for Plan 02; Plan 02 otherwise
requires Plan 01.
Plan 03's Memo-only Synapse activation consumes Plan 02 and therefore finishes
after it. Plan 04 requires the v2 ledger and admitted runtime contracts from
Plans 01–02 plus the completed manifests/readiness work from Plan 03. Plan 05
cannot begin until all prior plans have passed their acceptance gates on every
configured Mac.

## Specification Coverage

| Approved design requirement | Owning plan |
| --- | --- |
| Native Memo authority, v1 preservation, signed v2 records/anchors/checkpoints/views, epoch fencing, sessions, durable promotion | Plan 01 |
| Coordination, delivery/ACK, presence, continuity, terminal bridge, live sync, compaction, daemon, `memo_*` APIs | Plan 02 |
| 90-day live-use gate, Memflow fence/drain, Synapse isolation, consumer inventory and staged configs | Plan 03 |
| Existing Memo state + missing durable knowledge + active Memflow state, source proofs, idempotent rehearsal, rollback bundle | Plan 04 |
| Signed CAS state machine, epoch fencing, client restart, rollback, reboot verification, retirement and no data archive | Plan 05 |
| Test/parity/CI gates and explicit commits | Every plan |

## Execution Rules

- Use `superpowers:using-git-worktrees` at execution time. The current Memo
  checkout is dirty and behind its remote; implementation must start from a
  clean, current worktree without disturbing existing user changes.
- Use `superpowers:subagent-driven-development` for one fresh worker per task,
  with specification review followed by code-quality review.
- Treat `/Users/fer/repos/memo`, `/Users/fer/repos/memflow`, and the Synapse
  repository as separate commit authorities. Never sweep changes across repos.
- Stage explicit paths only. Never use `git add -A`, `git commit -a`, destructive
  reset, or broad cleanup.
- Follow TDD inside every task: failing focused test, minimal implementation,
  focused pass, relevant suite, explicit commit.
- Do not start Phase 0 of the operational cutover merely because code exists.
  Plan 05 owns the only authorization path to quiesce, activate, or delete.
- Before the committed activation epoch, failures restore Memflow globally.
  After the epoch, repairs move forward in Memo; Memflow never becomes a
  fallback.
- Destructive cleanup is permitted only after the global verification gate and
  an exact target check on every Mac.

## Program Definition of Done

The program is complete only when all five plans pass, both configured Macs run
the same Memo version/runtime/epoch, live Memo-to-Memo coordination works, a
logout/login or reboot scan finds no executable Memflow route, the Memflow data
remote is removed, and only the source-code remote remains archived read-only.
