# Native Memflow Absorption and Retirement Design

**Date:** 2026-07-28
**Status:** Approved design; written specification awaiting final user review
**Scope:** Absorb the live-used, proven Memflow capabilities into Memo as native
functionality, perform one coordinated cutover across every configured Mac, and
retire Memflow as an active product.

## Executive Decision

Memo becomes the only installed product and the only authority for durable
memory, live coordination, operational state, identity, synchronization,
runtime health, configuration, and user-facing interfaces.

The integration is a native absorption, not a package rename or an internal
sidecar:

- no Memflow package, import, process, daemon, binary, data directory, or
  runtime dependency remains;
- no public `flow_*` alias, wrapper, or deprecation window is introduced;
- capabilities admitted by the live-use gate become native Memo domains and
  `memo_*` interfaces;
- durable behavior already implemented by Memo is reused and parity-tested
  rather than copied;
- unused or experimental Memflow behavior is deleted even when tests exist;
- the cutover is coordinated across all configured Macs and never permits a
  mixed Memo/Memflow operating mode; and
- only missing durable knowledge and currently active operational state are
  migrated.

This decision supersedes the earlier boundary in which Memo owned durable
memory while Memflow remained a separate live coordination bus.

## Context and Evidence

Memo already owns the durable system: the memory facade, write policy,
identity, sessions and handoffs, contradiction handling, consolidation,
history, attention, runtime health, and Git synchronization. Memflow currently
retains the live bus: continuity packets, channels, delivery and ACK state,
presence, heartbeat, terminal injection, active workspaces, tasks, and its own
daemon and synchronization loop.

The deployed Memflow runtime has already disabled most duplicate durable work:

- `MEMFLOW_MEMO_BRIDGE=1`
- `MEMFLOW_DURABLE_ROUTER=memo`
- `MEMFLOW_DURABLE_READER=memo`
- `MEMFLOW_CAPTURE=off`
- `MEMFLOW_DURABLE_MAINTENANCE=off`

That means the remaining product boundary is primarily operational rather than
semantic.

Preliminary local telemetry for the last 90 days shows 4,046 Memflow tool
calls. `flow_continuity` dominates with 3,739 calls, followed by live handoff
and delivery behavior and duplicate durable routes such as write and capture.
Daemon activity is not fully represented by MCP tool-call counts, so the final
scope decision cannot rely on direct tool telemetry alone.

The replicated local data also shows real activity from both configured Macs:
253 handoff events split 196/57 and 80,291 presence events split 73,380/6,911.
This proves that presence, heartbeat, and cross-machine delivery are live even
though direct MCP calls undercount them. It does not prove that both copies are
fresh or complete. The direct freeze from both machines remains mandatory.

Usage auditing is itself instrumented and can increment the counters it reads.
The manifest builder must exclude its own audit calls, known `test-wire`
traffic, smoke sessions, and `*-eval` topics. When traffic cannot be classified
reliably, the capability remains admitted until direct per-machine evidence
resolves it.

Memo already has operational state. `Memory` creates an `OperationalStore` and
`WritePolicyEngine`; a per-device JSONL `OperationLedger` stores a hash chain,
and a derived JSON snapshot contains focus, handoffs, attention, conflicts, and
outcomes. That state participates in write-policy and feedback decisions and
cannot be overwritten or compacted under generic Memflow lifecycle rules.

The deployed environment also has external consumers beyond Memo and Memflow.
In particular, Synapse services currently execute from Memflow's virtual
environment and reference its backend. Codex, Claude Code, Gemini, Devin,
OpenCode, shell startup, and LaunchAgents contain live Memflow routes. Complete
retirement therefore includes consumer and runtime disentanglement, not only
tool migration.

The current checkout is not cutover-ready. Memo does not yet implement the
operational coordinator, request fence, activation epoch, live operational sync,
terminal bridge, or presence domains described here, and Memflow cannot yet
prove a global write drain. These are required implementation gates, not
assumptions about existing behavior.

The implementation must freeze a combined 90-day usage manifest from both
configured Macs before changing production behavior. The current observations
are directional evidence, not a substitute for that manifest.

## Goals

1. Make Memo the only product a user or agent installs, configures, invokes,
   observes, upgrades, or supports.
2. Preserve every proven capability that has real live use during the
   90-day admission window.
3. Preserve active coordination state without importing expired operational
   history.
4. Preserve valid durable knowledge that exists only in Memflow, deduplicating
   it against Memo before import.
5. Preserve Memo's existing focus, handoff, attention, conflict, outcome, and
   write-policy state while evolving its operational schema.
6. Retain or improve delivery, ACK, continuity, presence, recovery, and
   cross-machine behavior.
7. Make the cutover logically atomic and fail closed across all configured
   Macs.
8. Use selected Memflow tests as a parity oracle while deleting tests that only
   preserve eliminated surfaces.
9. Leave no runtime route back to Memflow after the final activation epoch.

## Non-Goals

- Preserving the complete Memflow API or tool catalog.
- Importing expired presence, closed channels, acknowledged deliveries,
  completed operational history, or terminated sessions.
- Keeping a compatibility package, compatibility namespace, hidden sidecar, or
  alternate data store.
- Reimplementing Memo capabilities merely because Memflow has another version
  of them.
- Maintaining a permanent copy of `.memflow` or the Memflow operational-data
  remote.
- Requiring the durable vault and operational ledger to use the same internal
  replication representation. They share one product authority, identity,
  configuration, and observability, but may use fit-for-purpose transports.
- Treating Memo's existing operational journal as disposable derived state.
- Deleting or breaking an external product merely because it currently shares
  Memflow's runtime. Live consumers must be migrated or explicitly retired
  first.

## Capability Admission Policy

### Ninety-day live-use gate

A Memflow capability is admitted when at least one of these conditions is true
within the 90 days immediately preceding the frozen inventory:

1. A real client invoked it on either configured Mac.
2. The daemon produced or consumed real operational events through it.
3. It is a transitive dependency required to preserve an admitted behavior.
4. It is needed to read and migrate active state at cutover.

Tests, smoke scripts, manual capability probes, and synthetic benchmark traffic
do not count as live use. Autonomic daemon behavior does count even when it has
no direct MCP invocation.

Every capability is assigned exactly one disposition:

- **Native Memo:** Memo already owns the behavior; prove parity and route all
  callers to the existing implementation.
- **Absorb:** implement the behavior as a native Memo domain.
- **Internal dependency:** retain only the minimum internal behavior required
  by an admitted capability; expose no unused public API.
- **Delete:** remove the capability and its dedicated tests, documentation, and
  configuration.

The frozen manifest records the source tool or daemon path, telemetry evidence,
transitive dependants, selected disposition, Memo replacement, parity tests,
and deletion proof. Scope changes after the freeze require an explicit manifest
amendment, not an opportunistic port.

### Preliminary capability map

The final manifest controls scope, but current evidence establishes this
starting map:

| Memflow behavior | Target disposition |
| --- | --- |
| Durable write, capture, lookup, query, index, identity, maintenance | Use native Memo behavior; do not port duplicate storage or cognition |
| Continuity packet | Absorb as a Memo composer over durable and operational state |
| Handoffs, channels, messages | Absorb into native Memo coordination |
| Tasks | Absorb only if the frozen manifest classifies the low-volume calls as real; otherwise delete |
| Delivery, retries, ACK, status, terminal cursor | Absorb into native Memo delivery |
| Presence, heartbeat, active workspace, conflicts | Absorb into native Memo presence |
| Session checkpoint and recovery needed by continuity | Absorb the live subset; use Memo durable sessions where applicable |
| Terminal delivery and daemon loop | Absorb into the Memo runtime coordinator |
| Git sync and peer notification | Extend Memo's sync authority with the dedicated operational transport |
| Watchdog, signals, and observability | Keep as an internal dependency of the Memo runtime |
| Semantic extract, awareness, homeostasis, autopilot, consolidation | Use native Memo cognition or delete duplicates; separate active-workspace emission into presence |
| MCP, CLI, install, doctor, config | Use and extend native Memo surfaces |
| Unused TUI, web, fallback, experimental, or duplicate durable surfaces | Delete |

## Target Architecture

```text
                     one Memo distribution

       CLI / MCP / hooks / terminal / background clients
                              |
                       memo_* interfaces
                              |
                 Memo operational application layer
          +-------------------+--------------------+
          |                   |                    |
     coordination          delivery            presence
    channels/tasks      retries/ACK/cursor    leases/heartbeat
          |                   |                    |
          +-------------------+--------------------+
                              |
                 immutable operational event set
                              |
                     derived local views
                              |
          +-------------------+--------------------+
          |                                        |
  continuity composer                     runtime coordinator
          |                                        |
          +------------- Memo Memory --------------+
              durable vault and promotion outbox
```

### Product and process model

Memo may continue to have separate technical entry points for its CLI, MCP
server, and daemon, but they are shipped by one package, use the same release
identity, resolve the same configuration, and must come from the same isolated
runtime. `memo doctor --strict-runtime` is a mandatory preflight check.

The Memflow daemon is replaced by a Memo-owned operational coordinator. It is
not a renamed Memflow process and does not import Memflow modules.

### State ownership

Memo keeps two explicit state classes:

- **Durable memory:** Markdown remains the source of truth. Existing Memo
  indexing, trust, history, contradiction, and synchronization rules continue
  to apply.
- **Operational state:** Memo's per-device hash-chained journal remains the
  authority and evolves to an operational-event v2 contract. Derived local
  views provide transactions and fast queries but are rebuildable from
  verified journal segments and anchors.

The v2 journal uses a dedicated Memo-managed operational remote and local
checkout under Memo's configured state directory. It does not synchronize
SQLite database files, does not reuse `.memflow`, and does not share mutable
files between device writers. Memo's sync authority owns both the durable and
operational transports even though their repositories and cadence differ.

### Single-authority rule

Every fact has one owner:

- live delivery and liveness facts belong to the operational ledger;
- durable facts and decisions belong to the Memo vault; and
- promotion from operational to durable state is explicit, journaled,
  idempotent, and linked in both directions.

No write path writes the same semantic fact independently to both systems.

### Compatibility with Memo's current operational system

The absorption evolves existing Memo primitives rather than introducing a
parallel ledger:

- `OperationLedger` v1 events remain byte-identical, are verified, and are
  sealed into the v2 genesis anchor without changing their historical meaning.
- The current derived JSON operational snapshot is replaced by a versioned
  local view store only after replay-parity tests prove focus, handoff,
  attention, conflict, outcome, and write-policy behavior.
- `FederationManager` chain verification and import become the basis for
  rehearsal and peer ingestion, extended for incremental segments and
  compaction anchors.
- `Memory.operational`, existing reducers, atomic I/O, write policy, identity,
  briefing, health, and Git-sync locking are reused through explicit adapters.
- No second operational authority may be created during the transition.

The target v2 derived view store is SQLite so multi-view transitions and the
promotion outbox share local transactions. This is a schema migration from the
current JSON snapshot, not an already-existing Memo contract. The implementation
plan must include a replay comparison and rollback fixture before switching the
view implementation.

## Native Memo Domains

### Coordination

Coordination owns channels, messages, handoffs, and live tasks. A handoff is a
typed message with addressing, topic, evidence references, optional ACK
expectation, optional topic termination, and optional expiry.

Task scope is intentionally narrow: create, inspect, complete, cancel when
required by live evidence, and relate a task to its source message or handoff.
No broader project-management product is introduced.

### Delivery

Delivery owns recipient routing, terminal injection, retry policy,
deduplication, ACK processing, delivery status, and per-consumer channel
cursors.

The core state machine is:

```text
pending -> delivered -> acknowledged
    |           |
    +--------> expired
    +--------> failed
```

Retries never create a new logical message. Duplicate delivery or ACK events
collapse on event identity and idempotency key. Cursor movement is monotonic per
consumer and channel.

### Presence

Presence uses renewable leases with an explicit expiry. A lease identifies
actor, device, project, topic, intent, and optionally touched files. Heartbeat
renews liveness without manufacturing a synthetic handoff.

Active-workspace conflicts are derived from nonexpired leases. Expired presence
is not retained as durable history.

### Continuity

Continuity is a read composer, not a second memory system. It combines:

- Memo's durable briefing and relevant memories;
- active handoffs, tasks, deliveries, and conflicts;
- the latest recoverable session checkpoint; and
- runtime and peer-health information needed to resume work safely.

The composer must have a strict size budget, deterministic section ordering,
source identifiers, and graceful degradation when the operational coordinator
is unavailable.

### Runtime coordinator

The Memo daemon owns heartbeat scheduling, replicated-event ingestion,
terminal delivery, retries, view rebuilding, expiry, compaction, health,
cutover barriers, and operational sync-loop status.

This is one new operational daemon built on `daemon_common`, with one
LaunchAgent and one exclusive operational-writer lease. It does not become a
second supervisor for Memo's recall, ingest, maintenance, embed, or idle
daemons. Those existing daemons remain separate Memo processes and submit
operational work through the same journal contract rather than writing the
derived views directly.

Startup verifies package version, runtime digest, state schema, peer roster,
device identity, control-plane epoch, and writer exclusivity. Shutdown first
fences new work, then drains event append, sync, terminal delivery, and outbox
queues, fsyncs the local head, publishes health, and releases the writer lease.

Background failures are isolated by domain and surfaced through Memo health
reporting. They must not silently start an alternate Memflow path.

## Operational Data Model

### Event envelope

Every operational event uses a versioned envelope with at least:

| Field | Purpose |
| --- | --- |
| `event_id` | Globally unique, stable event identity |
| `event_type` | Versioned domain transition |
| `schema_version` | Payload migration and validation |
| `principal_id` / `actor_id` / `target_id` | Authenticated user, acting identity, and recipient |
| `project` / `workspace` | Scope and conflict derivation |
| `origin_device` / `origin_sequence` | Replication ordering and gap detection |
| `logical_clock` | Causal ordering without trusting wall-clock timestamps |
| `created_at` / `expires_at` | Lifecycle and TTL |
| `idempotency_key` | Retry collapse |
| `caused_by` | Source message, handoff, task, or promotion |
| `subject_uri` / `content_hash` | Compatibility with Memo identity and deduplication |
| `previous_hash` / `event_hash` | Per-device chain verification |
| `source_proof` | Optional original schema, event ID, and hashes for migrated state |
| `payload` | Validated domain data |

Unknown schema versions are quarantined and reported; they are never partially
applied.

The migration does not rewrite v1 bytes and pretend their hashes still validate.
It verifies and seals every v1 device chain as a genesis anchor containing the
original final sequence/head, reducer version, and reduced-state root. The v2
ledger starts a new epoch linked to that anchor. Any v2 state-seed event carries
the v1 actor, subject URI, content hash, device, sequence, previous hash, and
event hash under `source_proof`; it derives new scope only from verified context
and never invents a target or expiry when none existed.

The v2 decoder rejects unsupported schema versions before reducer dispatch.
Unknown operations are quarantined rather than silently ignored.

Per-origin sequence and hash chain determine integrity. Causal references and a
hybrid logical clock determine cross-origin ordering. Domain reducers use
monotonic joins rather than timestamp-only last-write-wins: an ACK cannot
regress to delivered, a completed task cannot return to open without an
explicit reopen event, and a cursor cannot move backwards. Where a scalar field
truly requires a deterministic concurrent winner, logical clock and event ID
form the tie-breaker.

### Derived views

The local operational store maintains rebuildable views for:

- messages and channel membership;
- delivery attempts, terminal cursor, and acknowledgements;
- handoff lifecycle;
- task state;
- presence leases and active-workspace claims;
- session checkpoints;
- peer heartbeat and synchronization health;
- idempotency records and bounded tombstones; and
- durable-promotion outbox state.

View updates caused by one event are committed in one SQLite transaction.
Replaying the same event is a no-op.

The first v2 release must dual-reduce isolated fixtures through the current JSON
snapshot reducer and the new view reducer, then compare all v1 fields. This is a
test/rehearsal mechanism only, not a production dual-write path.

### Retention and compaction

Operational state is bounded:

- expired presence and heartbeat events are removed after their deduplication
  horizon;
- acknowledged or terminal deliveries retain only a bounded tombstone long
  enough to reject late duplicates;
- completed tasks and closed channels leave the operational active set after
  their configured safety horizon;
- terminated sessions are removed after any approved durable promotion; and
- compaction cannot remove an event still referenced by an active view.

The horizons are derived from maximum retry and replication delays and are
validated by tests. Operational history is not promoted merely to preserve
diagnostics.

Compaction cannot truncate the current hash chain naively. Each compacted
per-device segment produces an authenticated anchor containing device, ledger
epoch, base and final sequence, prior and final event hashes, active-state
Merkle root, reducer version, and creation time. Incremental import accepts a
segment only when it extends a known verified head or a trusted anchor. Forks,
gaps, invalid signatures, and anchor regressions are rejected.

Memo's existing focus, attention, conflict, outcome, and write-policy events
have domain-specific retention. Outcomes and conflicts referenced by trust,
feedback, or policy cannot be removed by generic channel/task compaction.

### Durable promotion

Durable promotion crosses Markdown and SQLite boundaries through a
transactional-outbox protocol:

1. The operational transaction records a promotion intent with an idempotency
   key and source event IDs.
2. Memo validates the durable write through its existing write policy.
3. The durable record is created or corroborated.
4. The outbox records the resulting memory ID.
5. Retries reuse the same key and cannot create a second memory.

Failure before step 3 leaves a retryable intent. Failure after step 3 is
reconciled by querying the idempotency key. A rejected durable write leaves the
operational fact intact until its normal lifecycle ends.

This requires a real Memo durable-idempotency contract; current topic
deduplication alone is insufficient. The durable write API must accept a stable
operation key, persist a hashed `promotion/<operation-key>` identity plus source
event provenance, and provide lookup by that key. Reconciliation compares the
stored payload/content hash before reusing the result. A key collision with
different content is quarantined as an integrity error.

## Operational Synchronization

Memo owns a dedicated operational Git remote separate from the durable corpus
remote. It reuses Memo's Git locking, retry, and recovery primitives but has a
continuous low-latency loop appropriate for live delivery.

The repository layout is single-writer by origin:

```text
events/<device>/<ledger-epoch>/<segment>.jsonl
anchors/<device>/<ledger-epoch>/<sequence>.json
heads/<device>.json
control/<cutover-attempt>/<device-vote>.json
```

Only a device may append its event segments or advance its head. Peers ingest
immutable segments and never rewrite another origin. Segments are published
only after local fsync and chain verification. A head declares the complete
sequence watermark, segment hashes, anchor, runtime version, and device
signature.

The sync loop polls continuously and supports an authenticated peer wakeup for
lower latency. Its healthy visibility and recovery SLOs must equal or improve
the frozen Memflow baseline. The capability manifest records that baseline
before implementation. Missing sequences trigger targeted segment recovery;
they do not permit reducers to skip ahead.

Remote credentials authenticate repository access. Device signing keys
authenticate event heads, anchors, cutover manifests, and votes. A checksum
alone is never accepted as authorship. Key enrollment, rotation, and peer
removal are Memo configuration operations and are included in the peer roster.

The operational sync lock and runtime coordinator share one ownership protocol
so two local processes cannot publish the same origin sequence. Sync failure
does not redirect to Memflow.

## Public Interface Contract

The final product exposes only Memo-owned names. There is no `flow_*` alias,
environment flag, binary, server registration, or compatibility wrapper.

The admitted semantic families are:

- continuity;
- channel and message operations;
- handoff create, consume, and status;
- delivery ACK and status;
- task create, complete, and status;
- presence announce, active peers, and conflicts;
- session checkpoint and recovery where required by live use; and
- runtime and synchronization health.

Precise CLI and MCP names follow existing Memo domain conventions:

- CLI implementation belongs in `cli_<domain>.py` and is wired by `cli.py`;
- MCP tools belong in `server_<domain>.py` and are wired by `build_server()`;
- the core API is exposed through `memo.memory.Memory`; and
- all public tools use the canonical Memo identity and the common error
  contract defined below.

The implementation plan must produce an explicit old-behavior-to-new-interface
mapping from the frozen manifest. It may consolidate multiple Memflow tools into
one Memo contract when behavior and call-site migration remain unambiguous.

### Identity and authorization

Memo identity is normalized into:

- `principal_id`: the authenticated owner;
- `actor_id`: the agent or human acting for that principal;
- `device_id`: the enrolled signing and origin identity;
- `session_id`: the live client session; and
- `terminal_id`: an optional user-owned delivery target.

Public tools do not accept an arbitrary actor as proof of identity. The runtime
derives principal and device from the authenticated connection and checks
whether the requested actor and target are authorized. Legacy Memflow IDs map
deterministically during migration; aliases remain only in the cutover audit,
not as a public identity namespace.

### Error contract

Memo does not currently have one universal error envelope, so this work
introduces `memo.error.v1` with:

- stable `code`;
- human-readable `message`;
- `retryable`;
- `trace_id`;
- optional `event_id`;
- safe structured `details`; and
- current runtime and activation epoch when relevant.

Domain exceptions map to this contract at CLI, MCP, and daemon boundaries. CLI
may render it for humans, but structured callers receive the same semantics.
Maintenance fencing, stale epoch, schema quarantine, identity rejection,
sequence gap, expiry, and delivery failure each have distinct codes.

### Session authority

The operational journal becomes the authority for live
`active -> recoverable -> terminated` session lifecycle. Existing JSON
checkpoints under Memo state become a derived hot-resume cache. Reusable session
patterns or summaries belong in durable Memo memory. Any redundant SQLite
session authority is migrated and retired after parity.

Existing public `mem_session_*` tools are renamed to coherent `memo_session_*`
contracts during the atomic client update; no new alias is added. The migration
manifest maps every existing session representation and proves that no
recoverable session is lost.

## Core Runtime Flows

### Send, deliver, and acknowledge

1. A client submits a message or handoff with an idempotency key.
2. Coordination appends the event and commits its local views.
3. Operational sync replicates the immutable event.
4. The target coordinator validates identity, scope, schema, and expiry.
5. Delivery injects the message once and advances the target cursor.
6. An ACK references the source event and is replicated idempotently.
7. The origin view moves to acknowledged.

A target that is temporarily offline receives the event after reconnection if
it has not expired. An expired event is never injected merely because it
arrived late.

Terminal delivery uses a controlled Memo bridge rather than arbitrary raw TTY
writes. It verifies that the target terminal belongs to the authenticated user,
strips unsafe control sequences, enforces a size limit, includes the event ID,
and suppresses an event already present in the terminal-delivery registry.
Notification and text injection are distinct authorized modes. The receiver
bridge records the event ID before presentation and returns a delivery receipt,
closing the crash window that would otherwise make raw terminal side effects
impossible to deduplicate reliably.

### Presence and conflict detection

1. The client publishes or renews a lease.
2. Peers derive active presence from unexpired leases.
3. Overlapping file or workspace claims are surfaced as conflicts.
4. Missing renewal expires the lease without a durable write.

Clock-drift limits are part of cutover preflight. Events outside the accepted
drift envelope are quarantined until corrected.

### Continuity

1. Memo loads the project-scoped durable briefing.
2. It loads active operational views for the same identity and project.
3. It ranks recoverable checkpoints and unresolved coordination items.
4. It emits one bounded packet with provenance and explicit omissions.

If operational state is unavailable, continuity returns durable context plus a
visible degraded-state warning; it never falls back to Memflow.

## Migration Input Policy

The one-time migration has three independent inputs:

### Existing Memo operational state

Memo's verified journal and heads are the base, not an import to be overwritten.
The v1 chains are sealed as the v2 genesis. Focus, handoffs, attention,
conflicts, outcomes, and policy-relevant state are seeded from the verified v1
reduction before any Memflow event is considered.

The merge preserves each Memo v1 origin head in its genesis anchor and assigns
translated Memflow events to a distinct migration origin in the v2 ledger
epoch. Stable source IDs and content hashes collapse semantic duplicates;
conflicts remain explicit. Memflow state cannot overwrite Memo focus, trust,
conflict, or outcome state merely because its timestamp is newer.

### Missing durable knowledge

Valid facts, decisions, preferences, and procedures that exist in Memflow but
not in Memo are normalized, deduplicated, and written through Memo's normal
write policy. Corroborated duplicates update evidence instead of creating new
records. Operational logs are not treated as durable knowledge by default.

### Active operational state

Only the cutover-active set is imported:

- pending or delivered-but-unacknowledged messages and handoffs;
- ACK state required to prevent redelivery;
- open tasks;
- active channels and required consumer cursors;
- recoverable session checkpoints;
- nonexpired presence and heartbeat leases; and
- runtime metadata required to resume safe delivery.

Expired leases, acknowledged deliveries beyond their required tombstone,
closed channels, completed tasks, terminated sessions, and old heartbeat or
delivery history are excluded.

The importer is operator-only migration tooling. It is never installed as a
public Memo command and is removed before the final Memo release is tagged.

## Live Consumer Migration Inventory

The cutover manifest must resolve every currently observed route:

| Consumer | Current Memflow coupling | Required target |
| --- | --- | --- |
| Codex | MCP registration and startup hook | Memo MCP plus Memo continuity hook |
| Claude Code | HTTP MCP, settings hooks, handoff and continuity scripts | Memo MCP and Memo-owned hooks |
| Gemini | MCP and startup hook | Memo MCP and Memo-owned startup |
| Devin | Permission, startup hook, HTTP MCP | Memo permission and endpoint |
| OpenCode | Skills loaded from the Memflow checkout | Memo-owned skills only |
| Shells | `MEMFLOW_*`, PATH entry, shim, startup scripts | Memo environment or removal |
| launchd | Memflow KeepAlive daemon and backups | Memo operational LaunchAgent |
| Synapse | MCP/dashboard/watcher runtime and backend from Memflow venv | Synapse-owned runtime with Memo backend or explicit retirement |

The implementation inventory records exact files, connection owners, PIDs,
ports, loaded jobs, and session restart requirements per Mac. Historical docs
may name Memflow; no executable or active configuration may do so after
retirement.

## Atomic Multi-Mac Cutover

### Atomicity definition

Atomicity is logical and externally observable: clients can observe either the
old Memflow service or the new Memo service, never both. A bounded maintenance
pause is allowed between them.

The configured peer roster is frozen into the cutover manifest. Every peer in
that roster must participate; an offline or mismatched peer aborts the attempt.

Every old and new request is fenced. A pre-cutover Memflow release must reject
writes while an active cutover lock exists and refuse all startup and writes
after a committed retirement epoch. Memo rejects operational requests before
commit or when their epoch is stale. Editing configuration files is not treated
as fencing: established MCP connections and cached tool catalogs must be closed
and reconnected against the committed Memo epoch.

### Control plane and state machine

An operator-owned cutover controller advances one signed attempt through:

```text
PREPARING -> READY -> QUIESCED -> STAGED -> COMMITTED -> VERIFIED
      \          \          \          \
       +----------+----------+----------+-> ABORTED
```

The controller uses an authenticated Git control ref with compare-and-swap
updates. Each attempt has a random unique ID, designated coordinator, lease,
roster, and monotonic state. Votes expire and are valid only for that attempt.
A stale `READY` or `ACTIVATION_READY` can never authorize a later attempt.

If the coordinator fails before `QUIESCED`, lease expiry aborts without changing
the active service. At or after `QUIESCED`, peers fail closed and remain fenced
until a replacement controller acquires the control ref by CAS and publishes a
signed resume or abort decision. They never auto-unfence on a timeout.

### Cutover manifest

The signed manifest contains:

- unique attempt ID, state, coordinator, lease, and vote expiry;
- peer roster and canonical identities;
- target Memo commit, package version, and runtime digest;
- capability-manifest hash;
- per-peer Memflow origin-sequence vector, local root, and remote Git ref;
- existing Memo journal heads and anchor roots;
- shared cutover timestamp and clock-drift result;
- durable-import and active-state counts;
- rehearsal result and parity-suite result;
- readiness votes bound to every preceding field;
- activation epoch; and
- final verification evidence.

Device signatures, not checksums alone, authenticate the manifest and votes.

### Required pre-cutover disentanglement

Phase 0 cannot begin until:

- Synapse MCP, dashboard, and watcher run from a Synapse-owned isolated runtime
  and either use Memo as their backend or have an approved retirement plan;
- no Synapse LaunchAgent or config resolves a binary from Memflow's checkout or
  virtual environment;
- Memo implements request-level fence and epoch validation;
- the operational daemon supports nonpublic staging and observable drain;
- Memflow's write paths, autonomous loops, and KeepAlive startup all honor the
  same signed fence;
- every registered client has a tested close/restart/reconnect procedure; and
- the cutover controller and rollback runbook have passed a disposable
  two-machine rehearsal.

### Phase 0: prepare

On every configured Mac:

1. Install the exact candidate Memo version with the new operational runtime
   disabled.
2. Verify identity, peer connectivity, disk, permissions, clock drift,
   synchronization, runtime digest, config digest, daemon build, peer roster,
   control epoch, and an extended `memo doctor --strict-runtime`.
3. Inventory live processes, ports, connections, sessions, LaunchAgents,
   hooks, MCP registrations, shell routes, locks, Git refs, and Memflow
   consumers.
4. Synchronize Memflow and freeze a common origin-sequence vector and remote
   ref.
5. Preserve a rollback bundle containing the exact Memflow commit/runtime,
   plist and load state, client configurations, `.memflow` Git state and refs,
   cursors, caches, origin heads, and active session roster.
6. Translate existing Memo operational state into an isolated v2 rehearsal.
7. Deduplicate missing durable knowledge into that rehearsal.
8. Import the active Memflow operational set into the temporary ledger.
9. Run migration invariants and selected parity tests.
10. Emit `READY` bound to the runtime, config, journal, snapshot, and manifest
    hashes.

Any missing vote, differing hash, failed check, or peer disconnect aborts the
attempt while Memflow remains installed and active everywhere.

### Phase 1: quiesce

After unanimous `READY`:

1. Publish a cutover lock.
2. Disable Memflow's LaunchAgent/KeepAlive relaunch capability without deleting
   the rollback definition.
3. Fence every Memflow MCP, CLI, hook, shim, and autonomous-loop write with a
   deterministic retryable maintenance error.
4. Close or restart clients so no established connection retains an unfenced
   tool catalog.
5. Drain requests, event appends, terminal delivery, ACK, cursors, Git locks,
   local commits, pushes, and autonomous loops.
6. Prove zero in-flight writers, no locks, no unpushed commits, remote
   convergence, and matching final origin-sequence vectors.
7. Capture and hash the final delta and canonical active-state root.
8. Stop every Memflow daemon and verify zero PID, listener, connection, worker
   thread, or writable handle.

No Memo operational interface is public in this phase.

### Phase 2: stage activation

1. Import each final delta idempotently.
2. Recheck active-set counts, identities, TTLs, cursors, and hashes.
3. Start each Memo daemon in a nonpublic, disposable staging namespace.
4. Test peer connectivity, cross-Mac delivery, ACK, presence, continuity,
   terminal injection, and recovery.
5. Emit `ACTIVATION_READY`.

Any failure before unanimous `ACTIVATION_READY` stops staged Memo daemons and
discards all staging events. The controller restores every rollback-bundle
artifact, verifies its hashes and Git refs, then reactivates Memflow everywhere.
Staging tests can never mutate the production ledger, cursors, or remote.

### Phase 3: commit

After unanimous `ACTIVATION_READY`, the coordinator publishes one activation
epoch by compare-and-swap. Client configuration, MCP registration, hook,
terminal route, LaunchAgent, and instruction changes have already been staged;
they become usable only when the local signed epoch marker matches the committed
control ref. Clients reconnect and submit the epoch on every request.

Memflow observes the committed retirement epoch and permanently rejects startup
or writes even while its binary and rollback bundle still exist during
verification. Memo rejects any request carrying an earlier epoch.

Once the epoch is published, rollback to Memflow is forbidden. Subsequent
defects are repaired forward in Memo so the system cannot split into old and
new authorities.

### Phase 4: verify and retire

The global verification gate checks:

- active-set identity and hash parity;
- no duplicate delivery and no lost ACK;
- open-task and cursor parity;
- nonexpired presence and heartbeat;
- recoverable session and continuity output;
- healthy daemons and operational synchronization;
- a single Memo version and activation epoch on every peer; and
- absence of any live Memflow consumer, connection, relaunch route, or shared
  external runtime.

Only after the gate passes may retirement cleanup begin.

## Error Handling and Observability

- Invalid events are quarantined with event ID, schema version, origin, and
  reason; valid domains continue operating.
- Sequence gaps pause cursor advancement for the affected origin and trigger
  resynchronization.
- Duplicate events and ACKs are recorded as deduplication outcomes, not errors.
- Expired events produce a terminal expiry state and are never delivered.
- Terminal-injection failure remains retryable within the event TTL.
- Operational-view corruption triggers a rebuild from the active event set.
- Durable-promotion failures remain in the outbox with actionable status.
- Peer or runtime mismatch is a hard cutover-preflight failure.
- Barrier failures include the exact peer and check that prevented progress.
- Coordinator failure after quiesce remains fenced until a signed CAS takeover
  chooses resume or abort.
- Stale or missing epochs are rejected at every request boundary.
- After activation, no exception handler invokes a Memflow fallback.

Memo health reporting exposes at least peer liveness, last sync, origin-sequence
gaps, delivery backlog, oldest pending delivery, retry count, ACK latency,
expired-event count, deduplication count, outbox failures, view-rebuild status,
runtime identity, writer-lease owner, fence state, cutover attempt/state,
connection counts, drain counters, and activation epoch.

## Test Strategy

### Traceability

Every admitted behavior links:

```text
90-day evidence
    -> capability-manifest row
    -> Memflow source contract/test
    -> Memo replacement contract/test
    -> cutover verification
```

Tests for deleted behavior are not ported merely to keep the old suite green.

### Preliminary Memflow parity oracles

These are candidate source contracts, not evidence that the suite passes at
design time:

| Capability | Candidate Memflow tests |
| --- | --- |
| Durable routing/replacement | `test_durable_routing.py`, `test_durable_reader.py`, `test_memo_write.py`, `test_capture_gate.py`, `test_lookup.py` |
| Continuity and checkpoints | `test_continuity.py`, `test_kernel.py`, `test_integration_cross_feature.py`, `test_session_snapshot.py` |
| Channels and handoffs | `test_channel_supersede.py`, `test_handoff_evidence.py`, `test_peer_notify.py`, `test_peers.py` |
| Delivery and terminal cursor | `test_delivery.py`, `test_delivery_idempotency.py`, `test_tty_escalation.py`, `test_daemon.py`, `test_lifecycle.py` |
| Tasks | `test_tasks.py`, admitted only if real usage is confirmed |
| Presence and heartbeat | `test_presence.py`, `test_heartbeat.py`, `test_fast_lane.py` |
| Terminal delivery | `test_terminal.py`, `test_authoring_tty.py`, `test_tty_escalation.py` |
| Operational sync | `test_peer_sync.py`, `test_sync_conflict.py`, `test_sync_lock.py`, `test_git_sync_rebase_recovery.py`, `test_remote_breaker.py` |
| Runtime health | `test_health_watchdog.py`, `test_health_verdict.py`, `test_event_store_health.py`, `test_runtime_observatory.py` |
| Product surface/retirement | `test_mcp_server.py`, `test_mcp_tools.py`, `test_installer.py`, `test_daemon_launchd.py`, `test_cli_ops_doctor.py`, `test_architecture_boundaries.py` |

The planning phase must first run and record the baseline of every selected
oracle. A currently failing source test cannot serve as proof without an
explicitly reviewed contract interpretation.

### Contract tests

Contract coverage includes:

- delivery and ACK state transitions;
- TTL and late-arrival behavior;
- idempotent send, import, promotion, and replay;
- cursor monotonicity;
- v1 genesis anchoring, source-proof preservation, and reducer parity;
- logical-clock and concurrent monotonic joins;
- anchor-based prefix compaction and incremental import;
- presence renewal and expiry;
- workspace-conflict derivation;
- task transitions;
- session recovery;
- unknown schema quarantine;
- durable operation-key lookup and collision handling;
- bounded continuity composition; and
- compaction with active references and tombstones.

### Integration tests

Integration coverage exercises:

- CLI, MCP, hooks, and Memory-facade paths;
- daemon startup and shutdown;
- replicated-event ingestion;
- terminal delivery and retry;
- operational-view rebuilding;
- durable promotion through Memo write policy;
- runtime isolation and extended strict-runtime checks;
- authenticated peer heads, anchors, wakeups, and cutover votes;
- request fencing and stale-epoch rejection;
- controlled terminal ownership, escaping, deduplication, and receipt;
- Memo operational-state replay parity;
- session-authority migration; and
- degraded continuity with no Memflow fallback.

### Migration and distributed tests

Migration tests use synthetic fixtures and sanitized real-state shapes for:

- empty state;
- missing durable knowledge plus duplicates;
- pending and partially delivered handoffs;
- duplicate final deltas;
- active and expired leases;
- open and completed tasks;
- recoverable and terminated sessions; and
- repeated importer execution.

Two-Mac tests inject disconnects, duplicated events, clock drift, daemon crashes,
stale versions, sequence gaps, and failure at every cutover barrier. They must
prove global rollback before the activation epoch and no mixed authority after
it.

Cutover rehearsals also cover controller loss before and after quiesce, expired
votes, CAS conflict, an established old MCP connection, LaunchAgent KeepAlive
relaunch, unpushed Git state, a mutating staging test, rollback-bundle
restoration, and logout/login or reboot. Synapse isolation is tested before the
Memflow virtual environment can be removed.

### Verification order

The implementation follows repository CI order:

1. `ruff`
2. `mypy`
3. focused contract, runtime, hook, migration, and distributed tests
4. full non-slow `pytest`
5. separate slow suite and macOS runtime smoke where applicable
6. live cutover rehearsal on the configured Macs

## Acceptance Criteria

The integration project is complete only when all of the following are true:

- The combined 90-day capability manifest is frozen and reviewed.
- Every admitted capability has a native Memo owner and parity evidence.
- The complete Memo suite and 100% of selected parity contracts pass.
- Memo v1 journal heads and policy-relevant operational state replay identically
  under the v2 reducer before Memflow state is merged.
- The active operational set matches by count, identity, lifecycle, and hash.
- No pending delivery, required ACK, open task, active cursor, nonexpired lease,
  or recoverable session is lost.
- Failure injection proves signed fencing, CAS-controlled global rollback before
  the epoch, and no mixed operation after it.
- All configured Macs report the same final Memo version, runtime, and epoch.
- Memo-to-Memo delivery, ACK, presence, heartbeat, continuity, terminal
  injection, session recovery, and synchronization work without Memflow code or
  data.
- Synapse and every other live consumer has its own valid runtime and no
  executable dependency on Memflow.
- No active process, LaunchAgent, package, binary, MCP registration, hook,
  environment flag, config entry, instruction, data directory, or caller
  references Memflow or `flow_*`.
- A logout/login or reboot scan reports zero Memflow PID, listener, connection,
  relaunch job, hook, binary, environment route, MCP registration, or executable
  reference.
- The operator-only importer and its temporary snapshots have been removed.
- The Memflow operational-data remote has been deleted after final verification.
- The Memflow code remote is read-only and contains no operational data.
- Memo owns the sole documentation and support surface.

## Retirement Procedure

After the global verification gate:

1. Confirm Synapse and every other shared-runtime consumer is already migrated
   or explicitly retired.
2. Uninstall `memflow` and `memflow-mcp` on every Mac.
3. Remove the Memflow daemon, LaunchAgent definition, disabled override, and
   plist backups.
4. Remove Memflow MCP registrations, hooks, permissions, skills, environment
   variables, shell shims, client instructions, sync-loop jobs, and terminal
   routes from Codex, Claude Code, Gemini, Devin, OpenCode, and shell startup.
5. Delete Memflow logs, caches, terminal cursors, local `.memflow` directories,
   rehearsal ledgers, rollback bundles, snapshots, and cutover tooling.
6. Delete the Memflow operational-data remote after verifying its active state
   and unique durable knowledge are represented in Memo.
7. Tag the Memflow code repository at its final pre-retirement commit and
   archive that remote read-only for source provenance only.
8. Remove local Memflow code checkouts and virtual environments after the
   archived tag and external-runtime independence are verified.
9. Remove the one-time importer and any legacy schema reader before tagging the
   final Memo release.
10. Logout/login or reboot every peer, then run a negative independence scan and
    Memo-only cross-machine smoke test.

The audit manifest retained by Memo contains hashes, counts, versions, epochs,
invariants, and pass/fail evidence. It does not retain discarded operational
payloads or act as a Memflow data archive.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Porting the product instead of the used behavior | Frozen live-use manifest and one disposition per capability |
| Losing continuity while deleting duplicate storage | Build continuity as a composer over Memo durable and operational state |
| Breaking Memo's existing operational truth | Explicit v1-to-v2 translation, preserved heads, dual-reducer rehearsal, domain-specific retention |
| Split-brain across Macs | Request fencing, signed manifests/votes, CAS control ref, frozen roster, nonpublic staging, one monotonic epoch |
| Duplicate delivery after replay or compaction | Stable event IDs, idempotency keys, monotonic cursors, bounded tombstones |
| Breaking the hash chain during compaction | Authenticated per-device anchors with sequence and Merkle roots |
| TTL errors caused by clock skew | Preflight drift gate, explicit `expires_at`, quarantine outside the drift envelope |
| Cross-store partial durable promotion | Transactional outbox plus idempotent reconciliation |
| Hidden fallback keeps Memflow alive | Post-cutover scans, failure tests, and a hard ban on Memflow fallback paths |
| Existing connections bypass edited config | Fence every request and require client close/reconnect at the barrier |
| LaunchAgent resurrects Memflow | Disable KeepAlive before stop and verify after reboot |
| Shared Memflow venv breaks Synapse | Move Synapse to its own runtime/backend before Phase 0 |
| Importing operational history as memory | Separate durable-dedup and active-state policies |
| Divergent CLI and MCP runtimes | Same release identity and mandatory strict-runtime preflight |
| Permanent migration debris | Operator-only importer, explicit cleanup gate, no operational-data archive |

## Final State

The user installs and operates Memo. Agents consult and write Memo. Live
handoffs, delivery, ACK, presence, tasks, continuity, terminal delivery, and
recovery are Memo capabilities. Durable cognition remains Memo's responsibility.

Memflow exists only as an archived source-code record of the retired
implementation. It has no running service, installed package, public namespace,
configuration, operational data, compatibility contract, or independent product
identity.
