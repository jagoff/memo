# Synapse Retirement and Selective Memo Absorption

**Date:** 2026-07-30  
**Status:** Design approved in brainstorming; written specification pending user review  
**Scope:** Retire Synapse completely while preserving only its proven, useful behavior as native Memo functionality.

## Executive decision

Memo becomes the only installed product and the only runtime authority. The
existing Memflow absorption program remains the coordinated cutover mechanism,
but its Synapse branch changes from “keep Synapse in Memo-only mode” to
“inventory, selectively absorb, and retire Synapse.”

Synapse is not ported as a package, compatibility layer, sidecar, gateway, or
new namespace. Its value is evaluated per source operation using signed live-use
evidence. Each operation receives exactly one disposition:

- **Memo native:** Memo already owns equivalent behavior; prove parity and use
  the existing surface.
- **Absorb:** implement a genuinely useful Synapse behavior as a native Memo
  domain, with a Memo owner and a bounded public interface.
- **Internal:** keep only the minimum implementation detail required by an
  admitted operation; expose no Synapse API.
- **Delete:** remove unused, redundant, or product-specific behavior.

The final activation epoch is Memo-only. After it, no client, daemon,
LaunchAgent, configuration, import, executable path, fallback, or data namespace
may depend on Synapse.

## Context and evidence

The existing design at
`docs/superpowers/specs/2026-07-28-native-memflow-absorption-design.md`
retired Memflow but left Synapse as a possible Memo-only consumer. Plan 03
Task 05 (“Replace Synapse's Memflow contract with a Memo backend registry”) has
not started. This specification supersedes that target.

Plan 03 Task 04 isolated Synapse from the Memflow runtime in Synapse worktree
`feat/memflow-absorption-runtime`, technical commit `933445fd`. That work is a
temporary safety prerequisite only; it does not authorize keeping Synapse.

The local Synapse installation currently has:

- 159 Python source files and approximately 54,481 source lines;
- a canonical MCP catalog of 31 tools across six namespaces;
- ten `com.synapse.*` LaunchAgents, including jobs that actually perform Memo
  ingest, nightly maintenance, digest, and recall work;
- active dashboard and watcher processes launched from the Memflow virtual
  environment;
- approximately 808 MB under `~/.synapse`, including about 633 MB of runtime
  history;
- 9,827 local ledger records, of which 7,921 are `runtime_loop`, 1,043 are
  `agent_event`, and 644 are `chat_backend_query`;
- 1,046 chat traces and 950 pipeline traces, with the latest chat backend
  query on 2026-06-16;
- nine explicit chat feedback records in the ledger.

The current live-use evidence is directional. The final manifest must collect
the inclusive 90-day window from both configured Macs, exclude tests, smoke,
benchmarks, and self-auditing traffic, and include daemon activity and live
consumer configuration.

## Goals

1. Make Memo the sole product, MCP entrypoint, runtime, and operational
   authority.
2. Preserve useful Synapse behavior without preserving Synapse as a product.
3. Avoid copying capabilities already native to Memo.
4. Preserve explicit feedback and reproducible evaluation evidence where it
   improves Memo; do not import Synapse's operational history wholesale.
5. Reuse the existing two-Mac fencing, staging, rollback, and activation-epoch
   machinery from the Memflow absorption program.
6. Prove that every live Synapse route is migrated or intentionally retired.

## Non-goals

- Porting Synapse's 54k-line implementation or its 31-tool catalog.
- Keeping a `synapse_*` compatibility namespace, wrapper, gateway, or
  deprecation alias.
- Treating Synapse's ledger, traces, runtime snapshots, or caches as durable
  memory truth.
- Preserving federation between two memory backends after Memflow retirement.
- Importing expired runtime history, old presence, acknowledged delivery, or
  completed operational records.
- Making a final capability decision from source size, test count, or historical
  documentation alone.

## Capability admission and manifest

The operator-only inventory tool takes immutable snapshots of the pinned
Synapse commit, source/tests/goldens, configured clients, LaunchAgents,
runtime paths, and state roots. It joins those snapshots with signed usage
receipts from both Macs over the exact 90-day window.

The manifest is canonical and signed. Each operation records:

- source tool, CLI verb, daemon route, or configuration route;
- consumers, machines, and evidence receipt IDs;
- observed calls and daemon events after exclusions;
- transitive dependencies;
- disposition;
- Memo target or explicit deletion proof;
- parameter, result, error, and retryability mapping;
- parity fixtures and SLO baseline;
- source snapshot and artifact digests.

Missing coverage, unknown operations, ambiguous predicates, or an operation
without a complete mapping blocks the manifest. A `delete` disposition requires
complete zero-use evidence and a negative deletion proof. A capability with
uncertain classification remains admitted until direct evidence resolves it.

## Preliminary capability map

The signed manifest controls the final result. This table is the starting
hypothesis, not a substitute for the manifest.

| Synapse capability | Preliminary disposition | Memo target or proof |
| --- | --- | --- |
| Boot packet, federated query, route explanation, explicit remember/action | Memo native plus a thin composition only where missing | `memo_unified_briefing`, `memo_context`, `memo_search`, `memo_ask`, write policy, and `EvidencePack` |
| Provenance graph and evidence normalization | Memo native or absorb only missing edges/format | Memo provenance, graph, citations, and evidence-pack contracts |
| `RealityConflict` list/workbench/lifecycle | Memo native or absorb lifecycle/provenance gaps | Operational conflict records, contradict/temporal analysis, and review APIs |
| Attention, reconciliation, lifecycle recommendations | Memo native | Operational state, briefing, outcomes, contradiction and lifecycle surfaces |
| Chat query rewrite, multi-query, rerank, source deduplication, entity overview, follow-up context, feedback | Absorb only measured gaps | `memo_chat_ask`, retrieval/rerank, contextual/session surfaces, regression/eval corpus |
| Replay, contracts, deterministic evals, policy simulation | Internal or absorb selected fixtures | Memo replay, health, eval, and quality gates |
| Runtime snapshots, trust receipts, adoption/observability reports | Internal or Memo native | Memo journal signing, runtime health, doctor, usefulness/adoption reporting |
| Dashboard and local review ledger | Memo native or delete duplicate views | Memo health/dashboard/TUI and operational journal |
| Vault, WhatsApp, nightly, watcher, digest, dream jobs | Rehome active jobs; delete Synapse wrappers | Memo import, maintenance, ingest daemon, briefing, and synthesis jobs |
| Memflow/Memo federation and backend registry | Delete | No second backend exists after cutover |
| Synapse MCP/CLI aliases and gateway | Delete | Clients point directly to Memo before the activation epoch |

## Data migration policy

Synapse is not a source of durable truth. Data migration is therefore bounded
to four classes:

1. **Explicit feedback:** the nine recorded chat feedback items are imported
   into Memo's feedback/signals only when their IDs are not already present.
2. **Evaluation evidence:** high-signal chat/pipeline traces may be reduced to
   redacted regression fixtures, labels, latency/SLO aggregates, and source
   IDs. Full traces are not imported as memories.
3. **Active configuration:** live consumer routes, job schedules, and required
   paths are transformed into Memo-owned configuration and verified before the
   old routes are disabled.
4. **Audit receipt:** counts, hashes, source commit, manifest digest, and
   pass/fail evidence are retained in Memo's operational audit record.

The following are explicitly excluded: the runtime history directory, full
ledger/log files, cold ledger, caches, per-tick snapshots, stale adoption
records, and Synapse trust anchors. Memo reissues its own signing/trust
material. Active operational state belongs to the existing Memflow→Memo
migration policy, not to a second Synapse import path.

## Target architecture and flow

```text
Synapse snapshot + two-Mac usage evidence
                 |
                 v
       signed operation manifest
                 |
                 v
   Memo-native / absorb / internal / delete
                 |
                 v
     temporary Memo staging namespace
                 |
       parity, replay, client checks
                 |
                 v
        one coordinated activation epoch
                 |
                 v
             Memo-only
```

The transition has five phases:

### Phase 0 — inventory and freeze

Build the immutable Synapse snapshot and signed capability/consumer manifests.
Capture all client registrations, LaunchAgents, paths, ports, processes, state
roots, active jobs, and configured Macs. No production behavior changes.

### Phase 1 — native mapping and staging

Implement only manifest-approved Memo deltas. Rehome active jobs to Memo-owned
entrypoints. Import bounded feedback/eval/configuration inputs into a
disposable staging namespace. Run deterministic replay and parity checks.

### Phase 2 — consumer switch preparation

Prepare Memo MCP/CLI/hooks and require every client to close and reconnect.
Validate dashboard/ingest/digest/watcher replacements and ensure no route uses
the Memflow virtual environment or Synapse executable.

### Phase 3 — coordinated cutover

Reuse the existing signed two-Mac controller: publish the lock, drain writers,
stop old daemons, apply the final delta idempotently, start Memo in nonpublic
staging, test cross-Mac delivery/health/recovery, and commit one activation
epoch. A missing peer or mismatched digest aborts before activation.

### Phase 4 — retirement and independence proof

Disable and remove Synapse LaunchAgents, MCP registrations, gateway entries,
wrappers, state roots, and runtime paths. Remove the temporary importer and
legacy readers. Reboot or re-login each Mac, run a negative independence scan,
and perform a Memo-only cross-machine smoke test. Archive the Synapse source
repository read-only for provenance, then remove installed runtime/code only
after the audit receipt is complete.

## Error handling and safety

- Incomplete evidence, an unknown operation, or an ambiguous mapping is a
  manifest blocker.
- Staging writes are idempotent and transactional; a partial import cannot be
  promoted.
- A failed rehearsal or peer vote leaves the old services installed and active.
- A stale MCP connection must be closed/restarted; editing a config file alone
  is not considered a cutover.
- After the activation epoch, Synapse startup and writes fail closed; no
  retryable fallback route is permitted.
- Any resurrected process, loaded LaunchAgent, path reference, import, or
  namespace discovered by the independence scan blocks final cleanup.

## Verification matrix

The implementation plan must include at least:

- canonical manifest reproducibility, signature, and immutable snapshot tests;
- operation-level route, parameter, result, error, retry, and provenance tests;
- duplicate feedback/eval import, replay, rollback, and crash-recovery tests;
- Memo-vs-Synapse retrieval/chat regression and SLO comparisons for admitted
  chat deltas;
- consumer migration tests for Codex, Claude Code, gateway, Devin, hooks,
  shell, MCP, and LaunchAgents;
- two-Mac offline/mismatch/rollback tests;
- no-Synapse negative scans over source, installed runtime, configuration,
  processes, ports, and loaded jobs;
- reboot/logout-login independence smoke tests;
- full Memo quality gates plus focused runtime, operational, migration, and
  install suites.

## Acceptance criteria

The design is successfully implemented only when:

1. The signed 90-day manifest is complete and accepted on both Macs.
2. Every admitted Synapse operation has a Memo owner, mapping, fixture, and
   parity/SLO evidence.
3. Every active Synapse consumer is migrated or explicitly retired.
4. Approved feedback/eval/configuration data is imported exactly once.
5. Memo-only staging passes cross-Mac health, delivery, recovery, and client
   reconnect checks.
6. One activation epoch fences Synapse and Memflow with no mixed mode.
7. No Synapse process, daemon, LaunchAgent, executable path, package, import,
   MCP registration, wrapper, state root, or fallback remains after reboot.
8. The temporary importer and legacy readers are removed before the final Memo
   release is tagged.
9. Memo retains the audit receipt but not discarded Synapse operational
   payloads.

## Relationship to existing work

- Existing Memflow absorption Plans 01–04 remain the foundation for Memo's
  operational authority, active-state migration, fencing, drain, and runtime
  isolation.
- Existing Plan 03 Task 04 remains a prerequisite for safe disentanglement but
  is not a product-retention decision.
- Existing Plan 03 Task 05 is replaced by this design's inventory, native
  mapping, consumer migration, and Synapse retirement work.
- Existing Plan 05's atomic cutover and final independence gates remain the
  execution mechanism.

This document defines the design only. A separate `writing-plans` pass will
split it into implementation tasks after written-spec review.
