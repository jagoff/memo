# Native Memflow Absorption and Retirement Design

**Date:** 2026-07-28
**Status:** Approved
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
5. Retain or improve delivery, ACK, continuity, presence, recovery, and
   cross-machine behavior.
6. Make the cutover logically atomic and fail closed across all configured
   Macs.
7. Use selected Memflow tests as a parity oracle while deleting tests that only
   preserve eliminated surfaces.
8. Leave no runtime route back to Memflow after the final activation epoch.

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
| Handoffs, channels, messages, tasks | Absorb into native Memo coordination |
| Delivery, retries, ACK, status, terminal cursor | Absorb into native Memo delivery |
| Presence, heartbeat, active workspace, conflicts | Absorb into native Memo presence |
| Session checkpoint and recovery needed by continuity | Absorb the live subset; use Memo durable sessions where applicable |
| Terminal delivery and daemon loop | Absorb into the Memo runtime coordinator |
| Synchronization and install/runtime health | Use or extend Memo's native runtime and sync authority |
| Awareness compaction or other helpers | Keep internally only when the manifest proves a live transitive dependency |
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
- **Operational state:** a bounded immutable event set is the source of truth
  for active coordination. Local SQLite views provide transactions and fast
  queries but are rebuildable from the active event set.

The operational event set may use a dedicated replicated subtree or transport
inside Memo's synchronization system. It must not synchronize SQLite database
files directly and must not reuse `.memflow`.

### Single-authority rule

Every fact has one owner:

- live delivery and liveness facts belong to the operational ledger;
- durable facts and decisions belong to the Memo vault; and
- promotion from operational to durable state is explicit, journaled,
  idempotent, and linked in both directions.

No write path writes the same semantic fact independently to both systems.

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
| `actor_id` / `target_id` | Canonical Memo identities |
| `project` / `workspace` | Scope and conflict derivation |
| `origin_device` / `origin_sequence` | Replication ordering and gap detection |
| `created_at` / `expires_at` | Lifecycle and TTL |
| `idempotency_key` | Retry collapse |
| `caused_by` | Source message, handoff, task, or promotion |
| `payload` | Validated domain data |

Unknown schema versions are quarantined and reported; they are never partially
applied.

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
- all public tools use canonical Memo identity and error envelopes.

The implementation plan must produce an explicit old-behavior-to-new-interface
mapping from the frozen manifest. It may consolidate multiple Memflow tools into
one Memo contract when behavior and call-site migration remain unambiguous.

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

The one-time migration has two independent inputs:

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

## Atomic Multi-Mac Cutover

### Atomicity definition

Atomicity is logical and externally observable: clients can observe either the
old Memflow service or the new Memo service, never both. A bounded maintenance
pause is allowed between them.

The configured peer roster is frozen into the cutover manifest. Every peer in
that roster must participate; an offline or mismatched peer aborts the attempt.

### Cutover manifest

The signed or checksummed manifest contains:

- peer roster and canonical identities;
- target Memo commit and package version;
- exact isolated runtime identity;
- capability-manifest hash;
- per-peer Memflow active-snapshot hash;
- shared cutover timestamp and clock-drift result;
- durable-import and active-state counts;
- rehearsal result and parity-suite result;
- readiness votes;
- activation epoch; and
- final verification evidence.

### Phase 0: prepare

On every configured Mac:

1. Install the exact candidate Memo version with the new operational runtime
   disabled.
2. Verify identity, peer connectivity, disk, permissions, clock drift,
   synchronization, and `memo doctor --strict-runtime`.
3. Synchronize Memflow and freeze a common snapshot boundary.
4. Deduplicate missing durable knowledge into an isolated Memo rehearsal.
5. Import the active operational set into a temporary ledger.
6. Run migration invariants and selected parity tests.
7. Emit `READY` bound to the version, snapshot, and manifest hashes.

Any missing vote, differing hash, failed check, or peer disconnect aborts the
attempt while Memflow remains installed and active everywhere.

### Phase 1: quiesce

After unanimous `READY`:

1. Publish a cutover lock.
2. Reject new Memflow writes with a deterministic retryable maintenance error.
3. Drain in-flight deliveries and synchronization.
4. Capture and hash the final delta.
5. Stop every Memflow daemon.

No Memo operational interface is public in this phase.

### Phase 2: stage activation

1. Import each final delta idempotently.
2. Recheck active-set counts, identities, TTLs, cursors, and hashes.
3. Start each Memo daemon in nonpublic staging mode.
4. Test peer connectivity, cross-Mac delivery, ACK, presence, continuity,
   terminal injection, and recovery.
5. Emit `ACTIVATION_READY`.

Any failure before unanimous `ACTIVATION_READY` stops staged Memo daemons and
reactivates Memflow everywhere from the untouched original state.

### Phase 3: commit

After unanimous `ACTIVATION_READY`, the coordinator publishes one activation
epoch. Every client configuration, MCP registration, hook, terminal route,
LaunchAgent, and instruction set switches to `memo_*` for that epoch.

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
- absence of any live Memflow consumer.

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
- After activation, no exception handler invokes a Memflow fallback.

Memo health reporting exposes at least peer liveness, last sync, origin-sequence
gaps, delivery backlog, oldest pending delivery, retry count, ACK latency,
expired-event count, deduplication count, outbox failures, view-rebuild status,
runtime identity, and activation epoch.

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

### Contract tests

Contract coverage includes:

- delivery and ACK state transitions;
- TTL and late-arrival behavior;
- idempotent send, import, promotion, and replay;
- cursor monotonicity;
- presence renewal and expiry;
- workspace-conflict derivation;
- task transitions;
- session recovery;
- unknown schema quarantine;
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
- runtime isolation and strict-runtime checks; and
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

### Verification order

The implementation follows repository CI order:

1. `ruff`
2. `mypy`
3. focused contract, runtime, hook, migration, and distributed tests
4. full non-slow `pytest`
5. separate slow suite and macOS runtime smoke where applicable
6. live cutover rehearsal on the configured Macs

## Acceptance Criteria

The design is complete only when all of the following are true:

- The combined 90-day capability manifest is frozen and reviewed.
- Every admitted capability has a native Memo owner and parity evidence.
- The complete Memo suite and 100% of selected parity contracts pass.
- The active operational set matches by count, identity, lifecycle, and hash.
- No pending delivery, required ACK, open task, active cursor, nonexpired lease,
  or recoverable session is lost.
- Failure injection proves global rollback before the epoch and forbids mixed
  operation after it.
- All configured Macs report the same final Memo version, runtime, and epoch.
- Memo-to-Memo delivery, ACK, presence, heartbeat, continuity, terminal
  injection, session recovery, and synchronization work without Memflow code or
  data.
- No active process, LaunchAgent, package, binary, MCP registration, hook,
  environment flag, config entry, instruction, data directory, or caller
  references Memflow or `flow_*`.
- The operator-only importer and its temporary snapshots have been removed.
- The Memflow operational-data remote has been deleted after final verification.
- The Memflow code remote is read-only and contains no operational data.
- Memo owns the sole documentation and support surface.

## Retirement Procedure

After the global verification gate:

1. Uninstall `memflow` and `memflow-mcp` on every Mac.
2. Remove the Memflow daemon and LaunchAgent definitions.
3. Remove Memflow MCP registrations, hooks, environment variables, client
   instructions, sync-loop jobs, and terminal routes.
4. Delete local `.memflow` directories, rehearsal ledgers, snapshots, and
   cutover tooling.
5. Delete the Memflow operational-data remote after verifying its active state
   and unique durable knowledge are represented in Memo.
6. Tag the Memflow code repository at its final pre-retirement commit and
   archive that remote read-only for source provenance only.
7. Remove local Memflow code checkouts after the archived tag is verified.
8. Remove the one-time importer and any legacy schema reader before tagging the
   final Memo release.
9. Run a post-cleanup scan and Memo-only smoke test on every peer.

The audit manifest retained by Memo contains hashes, counts, versions, epochs,
invariants, and pass/fail evidence. It does not retain discarded operational
payloads or act as a Memflow data archive.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Porting the product instead of the used behavior | Frozen live-use manifest and one disposition per capability |
| Losing continuity while deleting duplicate storage | Build continuity as a composer over Memo durable and operational state |
| Split-brain across Macs | Frozen peer roster, unanimous barriers, staged nonpublic startup, one activation epoch |
| Duplicate delivery after replay or compaction | Stable event IDs, idempotency keys, monotonic cursors, bounded tombstones |
| TTL errors caused by clock skew | Preflight drift gate, explicit `expires_at`, quarantine outside the drift envelope |
| Cross-store partial durable promotion | Transactional outbox plus idempotent reconciliation |
| Hidden fallback keeps Memflow alive | Post-cutover scans, failure tests, and a hard ban on Memflow fallback paths |
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
