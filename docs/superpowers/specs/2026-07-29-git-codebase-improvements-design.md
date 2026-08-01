# Git Codebase Improvements for Memo

**Date:** 2026-07-29

**Status:** Approved design; written specification awaiting final user review

**Scope:** Transfer only the highest-value engineering invariants from the
`git/git` codebase into Memo when they produce a demonstrated improvement over
Memo's current baseline.

## Executive Decision

Memo will not imitate Git as a product and will not pursue Git feature parity.
It will selectively adopt six engineering patterns that Git demonstrates at
production scale:

1. atomic mutation transactions and quarantine;
2. immutable content-addressed revisions, refs, reflogs, verification, and
   recovery;
3. explicit repository/request context and a plumbing/porcelain boundary;
4. retrieval planning with backend filter pushdown;
5. structured tracing plus performance, fuzz, and fault-injection discipline;
6. incremental maintenance governed by measured need.

Every transfer is conditional. It is admitted only when Memo has a reproduced
gap, the smallest viable change improves a primary metric or invariant, all
guardrails stay within a predeclared budget, and rollback is safe. A Git idea
that merely looks elegant, adds conceptual parity, or fails to beat Memo's
baseline is rejected.

The objective is improvement, not accumulation.

## Improvement-Only Rule

This rule controls the complete program and overrides the breadth of the
research backlog.

Each proposed transfer must have an admission record containing:

- the exact current Memo behavior and evidence;
- a falsifiable improvement hypothesis;
- the smallest implementation slice capable of testing it;
- one primary improvement gate;
- correctness, quality, latency, memory, storage, compatibility, and
  maintainability guardrails as applicable;
- a shadow or differential comparison against the existing path;
- a rollback route that does not require destructive downgrade; and
- a deletion or consolidation plan for any old mechanism it supersedes.

The outcome is exactly one of:

- **Admit:** the slice improves Memo and passes every guardrail.
- **Defer:** the hypothesis is plausible but current scale or evidence does not
  justify the complexity.
- **Reject:** the slice does not improve Memo, duplicates an existing
  capability, or violates a guardrail.

No behavior is enabled by default merely because its implementation exists.
No public surface is added before the underlying invariant is proven. Failed
experiments are removed or left disabled with an explicit rejection receipt;
they do not become permanent optional complexity.

## Evidence Snapshot

### Git source snapshot

The analysis used the official publish-only `git/git` mirror at commit
[`13c7afec212fc97ce257d15601659314c6673d6c`](https://github.com/git/git/tree/13c7afec212fc97ce257d15601659314c6673d6c),
dated 2026-07-27. The inspected shallow history covered 300 commits.

At that snapshot the repository contained approximately:

- 4,828 tracked files;
- 130 built-in commands;
- 2,533 files under `t/`;
- 980 files under `Documentation/`;
- 75 unit-test source files;
- 80 performance-test scripts; and
- dedicated fuzzing, portability, compatibility, and technical-design
  surfaces.

The important lesson is not repository size. It is how Git maintains a small
set of hard invariants while allowing storage, transport, indexing, and command
implementations to evolve independently.

### Recent code evidence

Recent Git changes reinforce the patterns selected here:

- [`55f961a`](https://github.com/git/git/commit/55f961a55) refactors
  `receive-pack` to stage incoming objects through generic ODB transactions
  instead of managing a temporary object directory directly. This separates
  the transaction guarantee from one storage backend.
- [`5183689`](https://github.com/git/git/commit/518368999) adds filters to
  object enumeration so storage backends can optimize candidate selection
  rather than forcing callers to over-enumerate and post-filter.
- [`0c20b08`](https://github.com/git/git/commit/0c20b0863) continues removing
  implicit `the_repository` use by passing repository context through refs and
  worktree APIs.
- [`ebc9fb5`](https://github.com/git/git/commit/ebc9fb587) memoizes shared graph
  traversal and uses generation information to avoid repeated reachability
  work.
- [`7d64c1e`](https://github.com/git/git/commit/7d64c1e1f) removes quadratic
  insertion behavior from a status path, illustrating the repository's
  explicit complexity discipline.

### Stable Git mechanisms

The selected transfers are also grounded in Git's documented architecture:

- the
  [core tutorial](https://github.com/git/git/blob/master/Documentation/gitcore-tutorial.adoc)
  separates low-level object/ref operations from user-facing commands;
- the
  [reftable design](https://github.com/git/git/blob/master/Documentation/technical/reftable.adoc)
  uses immutable sorted tables, atomic ref transactions, tombstones, update
  indices, consistent reader snapshots, and independent compaction;
- [`git fsck`](https://github.com/git/git/blob/master/Documentation/git-fsck.adoc)
  distinguishes object validity, connectivity, reachability, and recovery;
- [Trace2](https://github.com/git/git/blob/master/Documentation/technical/api-trace2.adoc)
  models nested regions, processes, threads, events, and performance data;
- [`git maintenance`](https://github.com/git/git/blob/master/Documentation/git-maintenance.adoc)
  runs tasks on different schedules and only when their cost is justified;
- the
  [commit-graph design](https://github.com/git/git/blob/master/Documentation/technical/commit-graph.adoc)
  treats reachability acceleration as rebuildable derived data; and
- the
  [partial-clone design](https://github.com/git/git/blob/master/Documentation/technical/partial-clone.adoc)
  explicitly distinguishes promised absence from corruption.

Partial clone is evidence for a future experiment, not part of the initial
implementation program.

## Current Memo Baseline

This design is grounded in Memo at commit
`25e56dfee7df71fe0e555d2ca621ededde758970`. The source tree contains 461 Python
files and 136,515 lines under `src/memo`; the test tree contains 537 Python
files and 111,996 lines.

Memo already has substantial relevant machinery:

- Markdown is current-state authority and hand-edited Markdown wins on
  reindex.
- `Memory` is the high-level application facade and `VecStore` is the storage
  entry point.
- individual write paths already use atomic file replacement and targeted
  rollback.
- `ContentAddressedArtifactStore` already provides SHA-256 identity, immutable
  artifact publication, exact size/hash verification, and atomic replacement
  for repository-intelligence artifacts.
- `HistoryStore` records append-only save/update/delete events in a separate
  database.
- `VersionStore` retains full bodies for versioning and rollback.
- `time_machine` reconstructs historical state from current rows plus reverse
  history replay.
- `search_with_trace` already returns a structured list of retrieval stages,
  while `memo.trace` propagates one process-local trace ID.
- vector search already pushes several filters into SQL before kNN selection.
- `memo maintain`, `memo dream`, the maintenance daemon, idle maintenance, and
  the sleep cycle already perform different forms of maintenance.
- Git sync, federation signatures, portable backups, path guards, secret
  scanning, soft delete, tombstones, and derived-index rebuilds already exist.

The program must extend or consolidate this machinery. It must not build a
second CAS, ledger, daemon, identity model, trace identity, sync protocol, or
maintenance authority.

### Reproduced gaps

The inspection found concrete gaps worth testing:

1. `FederationManager.import_bundle` verifies a bundle up front but saves its
   memories one at a time, accumulates errors, and can leave a partially
   imported bundle.
2. save/update/delete rollback is operation-specific; there is no shared
   compare-and-swap contract or atomic multi-record mutation rail.
3. `HistoryStore` explicitly swallows audit failures and does not retain full
   bodies for save/delete events.
4. `VersionStore`, `HistoryStore`, and `time_machine` overlap without one exact
   historical authority. `time_machine` documents unavailable deleted bodies
   and current-embedding historical search.
5. `Memory.gc()` checks store/file orphans and stale syntheses, but it is not a
   full hash, schema, connectivity, history, graph, ledger, and derived-state
   verifier.
6. filter handling is partially pushed down and partially repeated across
   vector, BM25, exact, fuzzy, fact, graph, and post-fusion paths. In
   particular, common date/tag checks still have post-generation seams.
7. trace propagation, search-stage explanations, and slow-call logging exist,
   but there is no single nested cross-process event model for CLI, MCP,
   daemons, storage, and retrieval.
8. maintenance behavior is spread across several schedulers and commands
   without one task registry describing locks, cost, `is-needed`, dependencies,
   budgets, and recovery.

These are hypotheses for improvement slices, not blanket authorization to
replace working code.

## Goals

1. Prevent accepted writes, imports, and sync operations from leaving
   logically partial visible state.
2. Detect concurrent lost updates with explicit expected/current revision
   errors.
3. Make historical state exact and verifiable when the immutable-revision gate
   proves worthwhile.
4. Make corruption, missing data, stale projections, and recoverable orphans
   distinguishable.
5. Make repository, project, principal, actor, transaction, and trace context
   explicit at domain boundaries.
6. Ensure every retrieval leg applies the same eligibility semantics before
   candidate truncation when its backend supports doing so.
7. Measure end-to-end behavior before and after every behavioral change.
8. Replace fragmented maintenance scheduling with one native task registry
   only if doing so reduces complexity and operational cost.
9. Preserve Markdown portability, offline-first operation, existing CLI/MCP
   contracts, and Memo's current trust and Memflow operational architecture.
10. Produce net improvements: admitted work must improve an invariant or metric
    without hiding unacceptable regressions elsewhere.

## Non-Goals

- Reimplementing Git's packfile format, branch model, wire protocol, clone,
  fetch, push, checkout, or index.
- Treating every memory edit as a source-control workflow.
- Making staging mandatory for normal trusted capture.
- Replacing current Markdown with an opaque object database.
- Treating a content hash as an authenticity signature.
- Introducing a second operational ledger, principal model, coordinator,
  daemon, or delivery/ACK system alongside the approved Memflow absorption.
- Adding arbitrary executable hooks.
- Adding hidden replacement views equivalent to `git replace`.
- Moving current Markdown out of the local machine.
- Implementing cold tiers, sparse views, shared object pools, generation
  numbers, bitmaps, or immutable index stacks before measured scale requires
  them.
- Guaranteeing atomic observation of several independent Markdown files to an
  external editor. Memo can provide batch atomicity to Memo readers through a
  commit marker and read barrier; filesystem-wide external atomicity would
  require a different vault layout.
- Keeping a failed experiment as permanent optional surface.

## Priority Model

Priority is based on user impact, not resemblance to Git:

- **P0:** prevents corruption, loss, partial publication, or incorrect
  concurrent writes.
- **P1:** improves historical exactness, retrieval correctness, or the ability
  to prove and diagnose behavior.
- **P2:** reduces recurring operational cost or complexity.
- **P3:** prepares for scale that has not yet been observed.

Expected-value rank and implementation order are different. A thin explicit
context and tracing slice must be established before changing high-value write
behavior, even though transactions rank above those foundations by user
impact.

## Essential Transfer Set

| Rank | Priority | Git invariant | Memo improvement candidate | Admission gate |
| --- | --- | --- | --- | --- |
| 1 | P0 | ODB/ref transactions and receive quarantine | shared CAS mutation rail; atomic import promotion | fault-injected and concurrent workloads show no accepted partial state or lost update, within frozen write budgets |
| 2 | P0/P1 | immutable objects, refs, reflogs, `fsck` | exact revisions, head movement, verification, recovery | exact as-of/restore and corruption detection improve over the three current history paths without harming current Markdown |
| 3 | P0 foundation | explicit repository APIs; plumbing/porcelain | explicit operation context and stable core primitives | removes ambient/scattered context hazards and reduces or holds complexity with zero behavior regression |
| 4 | P1 | backend-aware filtered object enumeration | one search plan with capability-aware pushdown | adversarial eligibility tests reproduce and then eliminate candidate crowd-out or semantic divergence |
| 5 | P1 | Trace2, `t/perf`, fuzzing, complexity tests | causal spans and stable quality/performance/fault gates | materially improves diagnosis and comparison while meeting disabled/sampled overhead and privacy budgets |
| 6 | P2 | need-driven maintenance and incremental derived data | one task registry using the native coordinator | replaces fragmented scheduling, prevents incompatible overlap, and reduces measured maintenance cost or operator complexity |

The table is the implementation ceiling for the first program. Supporting
ideas do not become implementation tasks without a separate admission review.

## Target Architecture

```text
                   CLI / MCP / daemon / hooks
                              |
                       porcelain adapters
                              |
                          MemoContext
          vault · project · principal · actor · trace · txn
                              |
                       stable plumbing API
          +-------------------+--------------------+
          |                   |                    |
     mutation rail       revision/ref store     search planner
   prepare/validate/CAS   immutable history     pushed/residual filters
          |                   |                    |
          +--------- current Markdown authority --+
                              |
                   rebuildable projections
             vector · BM25 · graph · facts · signals
                              |
                  fsck / Trace2 / maintenance
```

CLI and MCP remain wiring layers. `Memory` remains the supported application
facade. The design introduces small internal contracts underneath or alongside
the facade; callers must not import mixins or storage implementations directly.

## Explicit Context and Plumbing Boundary

### Context contract

A minimal immutable operation context carries only values that must remain
consistent across a domain operation:

- vault identity and resolved paths through the existing `Config`;
- project or namespace scope;
- owner principal and actor identity;
- trace ID and request ID;
- transaction ID when a mutation exists;
- configuration snapshot/version relevant to the operation; and
- an injectable clock only where deterministic recovery/tests require it.

The context does not replace `Config`, duplicate `ActorIdentity`, or become a
service locator. Large dependencies such as `Memory`, stores, embedders, and
models are not fields on the context.

### Admission characterization

Before introducing a new `MemoContext` type, the implementation plan must
inventory:

- ambient environment/config reads inside domain logic;
- repeated trace/actor/project parameters;
- cross-vault or cross-principal failure possibilities;
- signatures that would grow or shrink; and
- hot-path allocation and latency impact.

If explicit parameters plus existing `Config` already provide a clearer
contract, no new wrapper is introduced. The improvement is explicit context,
not a mandatory class name.

### Plumbing primitives

The desired internal seams are conceptual:

- prepare and validate a mutation;
- compare and update a logical head;
- store and verify an immutable revision;
- recover or abort an interrupted mutation;
- validate connectivity and projections;
- plan and execute a filtered search; and
- run one declared maintenance task.

Porcelain adapters translate Click, MCP, HTTP, hook, and daemon inputs into
these primitives. They do not own transaction, revision, filter, or recovery
semantics.

## Mutation Transaction and Quarantine

### Mutation contract

A mutation plan contains:

- a unique transaction ID;
- one or more logical memory IDs;
- expected current revisions when applicable;
- proposed canonical Markdown states or tombstones;
- derived projection intents;
- actor, principal, provenance, and policy decision;
- validation results; and
- an idempotency key for inbound or retryable operations.

The state machine is:

```text
new → prepared → validated → committing → committed → projecting → finalized
          \           \             \
           +-----------+-------------+→ aborted
```

`committed` is the visibility boundary. Before it, recovery aborts and removes
temporary state. At or after it, recovery completes idempotently. A transaction
never transitions from committed back to aborted.

### Publication

For one memory, Markdown publication continues to use atomic file replacement.
For a multi-memory batch:

1. validate every item and expected revision;
2. write immutable revisions and temporary Markdown files;
3. persist and harden a recovery manifest;
4. enter the committing state;
5. publish current Markdown files;
6. atomically publish the commit marker;
7. update heads/reflogs and derived projections; and
8. finalize the manifest and receipt.

Memo readers honor the commit marker/read barrier and never expose a partially
committed batch. The barrier is cross-process: while a transaction is
`committing`, readers either wait for its durable outcome or serve the
pre-transaction revisions recorded in the manifest. They do not read a mixture
of replaced files. External filesystem readers may still observe a short
mixed-file window; the CLI and documentation state this limitation rather than
claiming filesystem atomicity that the layout cannot provide.

### Compare-and-swap

Update, delete, restore, merge, and promotion may provide an expected revision.
Mismatch returns a typed conflict containing logical ID, expected revision,
actual revision, and a safe retry/re-read instruction. It never silently
chooses last writer wins.

### Quarantine

Federation, JSON/CSV imports, sync ingestion, and future streaming connectors
stage untrusted input in an isolated, bounded path. Before promotion Memo
checks:

- archive and path safety;
- schema and size limits;
- signature/trust policy where applicable;
- secret policy;
- canonical IDs and hashes;
- duplicate and idempotency keys;
- internal references and operation-ledger validity; and
- the mutation plan's complete CAS preconditions.

Promotion uses the shared mutation rail. Rejection leaves no record in current
Markdown or searchable projections. Quarantine cleanup is an explicit
maintenance task with retention and size budgets.

The durable-memory commit and the signed operational ledger are not forced
into a fictitious filesystem/SQLite distributed transaction. When an import
also carries operational events, the durable commit records an idempotent
promotion outbox intent using the operational contracts approved by the
Memflow absorption design. Recovery completes ledger application and the final
receipt. It never rolls back or rewrites an already accepted signed ledger
event.

### Reuse requirement

The implementation must reuse the existing authority lock, atomic-write
helpers, actor/write-policy contracts, operational receipts, and backup/path
guards where their contracts are sufficient. A new transaction abstraction
must consolidate per-operation rollback code rather than wrap and duplicate
it.

## Immutable Revisions, Refs, and Verification

### Authority model

Current Markdown remains the human-readable authority for the current memory
state. Immutable revisions are an additive historical authority. Losing the
revision archive must never make the current corpus unreadable.

The durable revision archive lives under a reserved vault-internal path so it
participates in sync, backup, restore, secret scanning, and path validation. It
is not placed where normal Markdown discovery or reindex can ingest it as a
current memory.

The exact directory and extension are frozen in the implementation plan only
after backup/sync compatibility tests, but these properties are mandatory:

- content-addressed sharding;
- immutable atomic publication;
- no symlink traversal;
- portable relative references;
- bounded manifest parsing;
- explicit inclusion in portable backup and restore; and
- current-vault readability when the archive is absent.

### Revision object

A versioned canonical envelope contains:

- schema version;
- logical memory ID;
- canonical current-state digest;
- complete reconstructable record state;
- zero, one, or two parent revision IDs;
- actor and provenance;
- transaction and trace IDs;
- creation time; and
- reason/kind such as save, update, delete, restore, merge, or external edit.

The revision ID is SHA-256 over the exact canonical envelope. The state digest
is separate, so identical record state can be recognized even when history,
actor, or time differs.

Hashes establish identity and integrity, not authorization or authenticity.
Where a signature is required, the revision uses Memo's approved principal and
ledger identity contracts rather than inventing another key system.

### Reuse of existing CAS

`ContentAddressedArtifactStore` is the starting implementation candidate, not a
guaranteed fit. A characterization must test concurrent same-object writes,
crash points between payload/manifest publication, portable paths, durability,
and backup behavior.

If it passes after a small generalization, Memo reuses it. If revision
requirements differ, common digest/atomic-write/verification primitives are
extracted once; a second unrelated CAS implementation is forbidden.

### Logical head and reflog

The current Markdown records or deterministically exposes its active revision
ID. Head changes use compare-and-swap. An append-only reflog records old/new
revision, transaction, actor, reason, and time.

Rollback creates a new revision whose state matches an earlier revision. It
does not delete or rewrite history. A resolved merge may have two parents.

### Historical convergence

Migration is staged:

1. characterize `HistoryStore`, `VersionStore`, and `time_machine` behavior;
2. shadow-write revisions without changing reads;
3. backfill current state and available historical state;
4. run differential as-of, diff, restore, update, delete, and recovery tests;
5. switch exact historical reads only after parity plus the new exactness
   cases pass; and
6. retire overlapping writes only after rollback no longer depends on them.

Legacy history is never fabricated. Missing bodies remain explicitly unknown
until a real source supplies them.

### `memo fsck`

The verifier has three levels:

- `--connectivity-only`: current Markdown IDs, revision heads, required files,
  store rows, and referenced objects.
- `--full`: hashes, canonical schemas, reflogs, history connectivity, graph and
  fact references, embedding dimensions, ledger invariants, and derived
  revision correspondence.
- `--strict`: full checks plus deprecated formats, policy violations, and
  migration debt.

The default is read-only. Repairs require an explicit subcommand or flag and
produce a receipt. Recoverable but unreachable objects are copied or linked
into `lost-found`; evidence is never silently deleted.

`Memory.gc()` remains a conservative cleanup surface until the verifier proves
replacement coverage. `fsck` and GC are not synonyms: verification discovers
and classifies; GC removes only policy-expired unreachable data.

## Retrieval Planning and Filter Pushdown

### Existing baseline

Memo already pushes date, excluded tags, validity, and related predicates into
parts of vector SQL, and it applies validity to several non-vector candidate
legs. It also has final date/tag filters after candidate generation.

The first task is therefore not a new planner. It is an adversarial
characterization matrix across exact, BM25, fuzzy, vector, fact, graph, hybrid,
HyPE, cached, and federated candidates.

### Admission gate

The planner is admitted only if tests demonstrate at least one of:

- an eligible result is crowded out because an unsupported leg truncates
  before a common filter;
- two modes disagree on eligibility for the same filter;
- a filter is implemented repeatedly with observable drift; or
- pushing the predicate down produces a statistically sound latency or
  candidate-volume improvement without recall loss.

If no material gap is reproduced, the existing implementation remains.

### Planner contract

When admitted, the design uses:

- `SearchFilter`: one typed predicate tree or normalized conjunction for type,
  tags, date, valid time, project/namespace, trust, visibility, verification,
  and deletion state;
- `BackendCapabilities`: predicates and boolean forms a candidate source can
  execute exactly;
- `SearchPlan`: pushed predicates, residual predicates, per-leg limits,
  fusion/rerank budget, and trace explanation; and
- `EligibilityOracle`: one reference implementation used by differential and
  property tests.

Supported predicates run before backend limits. Residual predicates run before
cross-leg fusion whenever possible, never only after the final `top-k`.
`search_with_trace` explains which predicates were pushed, residual, or
unsupported.

The planner does not change ranking policy by itself. Candidate-generation
changes and ranking changes receive separate evaluation receipts.

## Structured Tracing and Evaluation Discipline

### Trace model

Memo keeps the native trace ID and extends it with versioned structured events:

- process/request start and finish;
- child process and daemon ancestry;
- nested region enter/leave;
- transaction state transitions;
- storage/file/SQLite operations;
- candidate generation by leg;
- filter pushdown and residual counts;
- fusion, rerank, graph, fact, cache, and model regions;
- maintenance task decisions; and
- typed errors and recovery outcomes.

Events contain IDs, names, timing, counts, status, and bounded metadata. Memory
bodies, raw prompts, embeddings, secrets, credentials, and unapproved
configuration are excluded.

### Overhead and privacy

Phase 0 freezes overhead budgets on representative cold and warm paths.
Disabled tracing must remain statistically indistinguishable from the baseline
within the benchmark's noise floor. Sampled tracing receives an explicit
latency and allocation budget before implementation.

If cross-process propagation or event capture violates privacy or hot-path
budgets, the relevant span is rejected or sampled more narrowly; observability
does not automatically outrank the recall-hook budget.

### Test discipline transferred from Git

The program adds or strengthens:

- state-machine/model tests for transactions and refs;
- deterministic kill-points at every durable transition;
- multi-process and multi-thread CAS tests;
- property tests for filter semantics, canonicalization, and merge/ref
  invariants;
- fuzz targets for manifests, bundles, revision envelopes, and verifier input;
- complexity tests for large candidate sets and history traversals;
- stable performance cases for save/update/import/search/as-of/fsck/maintenance;
- differential tests against the existing implementation; and
- platform coverage on Linux and macOS.

Retrieval receipts record dataset, corpus revision, configuration, model,
hardware, code commit, and cold/warm state. Quality and performance gates are
separate so a latency win cannot hide recall loss and a recall win cannot hide
an unacceptable resource regression.

## Need-Driven Maintenance

### Admission condition

Memo already has several valuable maintenance paths. A unified registry is
admitted only if it replaces duplicated scheduling/locking/decision code and
improves at least one of:

- incompatible-task overlap;
- unnecessary task executions;
- recovery after interruption;
- operator understanding;
- maintenance wall time or resource use; or
- the amount of scheduling code and configuration.

It must not become another daemon or a wrapper that leaves every old scheduler
active underneath.

### Task descriptor

Each registered task declares:

- stable name and schema version;
- dependencies and incompatible tasks;
- required locks and scope;
- `is-needed` evidence and threshold;
- estimated/observed cost;
- hourly, daily, weekly, idle, or manual eligibility;
- time, CPU, memory, I/O, and model budgets;
- checkpoint/restart behavior;
- verification after completion; and
- dry-run explanation.

Candidate tasks include reindex drift, WAL checkpointing, SQLite maintenance,
`fsck-lite`, revision/ref cleanup, artifact cleanup, graph refresh, embedding
repair, and selected semantic dream passes.

The native Memo coordinator approved by the Memflow absorption design executes
the registry. Existing CLI commands may remain as porcelain aliases for direct
task execution, but task semantics and locking live in one place.

### Incremental data structures

Git's reftable stacks, commit-graph chains, MIDX, generations, and bitmaps are
not initial implementation requirements. They become separate P3 proposals
only when Trace2 and maintenance receipts show that simple storage and indexes
violate an agreed SLA.

## End-to-End Data Flow

```text
capture / update / delete / import / sync
                    |
                    v
         policy + explicit operation context
                    |
            trusted direct input?
              /              \
            yes               no
             |                 |
             |            quarantine
             |                 |
             +------ validation + CAS
                            |
                    mutation manifest
                            |
               immutable revision objects
                            |
                  current Markdown publish
                            |
                 atomic batch commit marker
                            |
                    heads + reflog
                            |
                rebuildable projections
             vector / BM25 / graph / facts
                            |
                     search planner
              pushed filters → residuals
                            |
                   fusion → rerank
                            |
                         result

        Trace2 spans and fsck identities cross every stage.
```

## Error Handling and Recovery

| Failure | Required behavior |
| --- | --- |
| policy, schema, secret, size, hash, or reference validation fails | abort before commit; current corpus and projections unchanged |
| expected revision differs | typed CAS conflict; include expected/actual; no implicit overwrite |
| crash before commit marker | recovery restores every already-replaced Markdown preimage, then aborts and removes prepared temporary state |
| crash after commit marker | recovery completes heads, reflog, projections, and receipt idempotently |
| one derived projection fails | current Markdown remains valid; mark repair debt; stale projection cannot surface an inactive revision |
| revision hash or schema is invalid | fail closed for that object; preserve evidence; report through `fsck` and `lost-found` |
| history backfill lacks a deleted body | preserve explicit unknown state; never fabricate content |
| external Markdown edit changes current state | detect as an external revision; apply policy or staging; hand-edited current Markdown remains authoritative |
| trace export fails | operation continues unless the trace is a required audit receipt; record bounded local failure telemetry |
| maintenance task exceeds budget | stop at a safe checkpoint; retain resumable state; do not cascade into later tasks |
| quarantine promotion partially reaches storage | transaction recovery finishes or aborts according to the durable commit boundary |

Domain failures use typed `MemoError` subclasses. Recovery actions are
idempotent and produce receipts with transaction and trace IDs.

## Rollout

### Phase 0 — Characterize and freeze gates

- Freeze Memo and test snapshots.
- Build failure, concurrency, history, filter, tracing, and maintenance
  baselines.
- Define explicit improvement and guardrail budgets before behavioral code.
- Inventory context propagation and plumbing boundaries.
- Instrument only the minimum spans needed to compare later phases.
- Produce admission records for all six transfers.

This phase may reject or narrow any later phase.

### Phase 1 — Integrity foundation

- Introduce the minimum explicit context/plumbing seams needed by writes.
- Characterize and consolidate per-operation rollback helpers.
- Add mutation manifests and compare-and-swap.
- Add quarantine and all-or-nothing promotion for one importer first.
- Add `fsck --connectivity-only`.
- Run fault, concurrency, migration, and write-performance gates.

The first importer is selected by highest reproduced partial-state risk, not by
implementation convenience. Other importers migrate only after the first slice
proves an improvement.

### Phase 2 — Exact revision experiment

- Characterize the existing CAS for reuse.
- Shadow-write immutable revisions.
- Add exact head/ref checks and a reflog.
- Add full verifier coverage.
- Backfill and compare history/version/time-machine results.
- Switch one historical read path only after exactness and resource gates pass.

If storage growth, write cost, portability, or operational complexity fails its
budget, the revision experiment remains off and the useful verifier work may
ship independently.

### Phase 3 — Retrieval experiment

- Build the cross-backend eligibility matrix.
- Reproduce a concrete pushdown/crowd-out or semantic-parity problem.
- Introduce the smallest planner slice for that problem.
- Measure candidate counts, recall, ranking, latency, and memory.
- Expand predicate/backend coverage only while each slice improves Memo.

No reproduced problem means no new planner.

### Phase 4 — Trace and maintenance convergence

- Expand tracing only where it closes a demonstrated diagnostic gap.
- Define the maintenance task descriptor.
- Migrate one overlapping scheduler/task decision.
- Prove lock, budget, recovery, and complexity improvement.
- Migrate remaining tasks incrementally and retire replaced scheduler logic.

### Default-on policy

Behavioral features begin default-off through Memo's registered flag system.
They become default-on only after:

- all primary and guardrail gates pass;
- shadow/differential receipts show no unresolved mismatch;
- migration and rollback are tested on copied real-world vault layouts;
- portable backup and restore include every new durable artifact;
- strict doctor and `fsck` are clean;
- Linux and macOS gates pass; and
- the old path has a documented retirement or ongoing necessity.

Disabling a new path never deletes its data. Old stores are not removed until
the new path has passed the agreed stability window and rollback no longer
depends on them.

## Verification Matrix

### Correctness

- save/update/delete/restore/merge state-machine transitions;
- exact CAS conflicts across threads and processes;
- batch read-barrier behavior;
- quarantine rejection and idempotent promotion;
- canonical revision and state hashes;
- reflog order and rollback-as-new-revision;
- exact as-of bodies across update/delete/restore;
- `fsck` classification and `lost-found`;
- filter eligibility parity across every candidate source; and
- maintenance lock/dependency decisions.

### Fault injection

- before and after every manifest write, fsync, atomic replace, commit marker,
  head update, reflog append, projection update, receipt, and cleanup;
- truncated, duplicated, reordered, oversized, and malformed manifests;
- corrupt revision payloads and manifests;
- stale writers and concurrent expected-revision updates;
- interrupted backup, restore, backfill, fsck, and maintenance; and
- unavailable daemon/model/index components.

### Security

- symlink, traversal, archive bomb, unsafe URL, and path-boundary cases;
- secret scanning before quarantine promotion and support export;
- principal/actor/visibility preservation;
- signatures verified by existing trust contracts;
- unauthorized hash knowledge does not grant object access; and
- traces contain no bodies, prompts, embeddings, credentials, or secrets.

### Compatibility

- legacy vaults without revision IDs;
- hand-edited Markdown;
- current Git sync and conflict markers;
- backup/restore with and without historical objects;
- CLI, MCP, HTTP, hooks, and daemons from the same isolated runtime;
- no module-level MLX/MLX-LM load; and
- stable current API behavior when every new flag is off.

### Quality and performance

- existing `eval/regression_labels.json` recall gate;
- capability-bucket evaluation for candidate-generation changes;
- cold/warm p50/p95 search and write latency;
- import throughput and peak memory;
- revision storage amplification;
- `fsck` connectivity/full runtime;
- maintenance work avoided versus performed;
- disabled and sampled trace overhead; and
- algorithmic scaling tests that catch quadratic behavior.

CI order remains Ruff, mypy, and pytest. Slow/MLX and platform smokes stay
separate as required by the repository.

## Supporting Ideas, Not Initial Tasks

These ideas remain useful but subordinate. Each requires a separate
improvement admission record after the essential program:

1. streaming, checkpointed bulk import built on the proven mutation rail;
2. a redacted `memo diagnose` bundle built on proven Trace2 and `fsck`;
3. trust-aware staging for low-confidence captures only;
4. structured three-way memory merge and exact-fingerprint resolution reuse;
5. `memo blame` and revision notes for provenance and review; and
6. `memo bisect` for reproducible retrieval/configuration regressions.

They are not implied implementation scope.

## Deferred P3 Experiments

The following remain design ideas until measured thresholds justify them:

- promisor-style cold storage for historical revisions or artifacts;
- immutable split index segments and logarithmic compaction;
- generation numbers and reachability bitmaps;
- shared immutable object pools; and
- sparse project or namespace views.

Current Markdown must remain local even if a future historical cold tier is
admitted.

## Explicit Rejections

The initial analysis rejects these Git concepts for Memo:

- branches as competing knowledge truths;
- mandatory index/staging for ordinary capture;
- Git packfiles or delta chains as Memo's base storage;
- Git's wire protocol;
- clone/fetch/push semantics as the user model;
- hidden object replacement;
- arbitrary executable hooks;
- content hashes presented as signatures;
- another sync, ledger, coordinator, or daemon architecture; and
- optimizing for hypothetical repository scale before benchmarks demonstrate
  the need.

## Risks and Controls

### Overengineering

The largest risk is importing Git's complexity without Git's scale. The
admission record, minimal slice, and reject/defer outcome are mandatory
controls.

### Dual historical authorities

Shadow-write migration temporarily increases overlap. Reads stay on the old
path until parity; writes to overlapping stores are retired as soon as the new
path is stable. The program must not leave four permanent history mechanisms.

### Write amplification

Manifests, revisions, reflogs, fsync, and projections can increase latency and
storage. Phase 0 freezes budgets; integrity slices can trade a bounded amount
of performance only when they eliminate a reproduced correctness failure.

### Manual-edit portability

Revision metadata must not make current Markdown unreadable or uneditable.
External edits are incorporated as real revisions rather than rejected.

### Partial abstraction

A new context, planner, transaction rail, or task registry that merely wraps
old duplicated logic fails the maintainability gate. Admission requires
consolidation or deletion.

### Scope collision with Memflow absorption

Operational identity, signing, coordination, presence, delivery/ACK, and the
native coordinator come from the approved Memflow absorption design. This
program consumes those contracts and never forks them.

## Definition of Success

The program succeeds when:

1. every shipped transfer has a before/after admission receipt;
2. no shipped transfer exists solely for Git parity;
3. accepted batch/import mutations cannot expose logically partial state to
   Memo readers;
4. concurrent stale writers receive deterministic CAS conflicts;
5. exact revisions and historical convergence ship only if they beat the
   current history/version/time-machine baseline;
6. `fsck` detects and classifies seeded corruption without destructive default
   behavior;
7. retrieval changes fix a reproduced eligibility or efficiency problem and
   pass quality gates;
8. tracing improves diagnosis within privacy and overhead budgets;
9. maintenance convergence removes duplicated scheduling or avoids measured
   work; and
10. rejected or deferred Git ideas add no permanent runtime surface.

## Planning Boundary

This specification authorizes design documentation only. It does not authorize
implementation.

After user review, the implementation plan must:

- start with Phase 0 characterization;
- split each transfer into an independent admit/defer/reject decision;
- include exact files, tests, commands, baselines, thresholds, and rollback;
- preserve unrelated work in the shared repository;
- avoid bundling the six transfers into one large migration; and
- stop a transfer when its improvement gate fails.

The first implementation milestone is evidence and contracts, not a feature
count.
