# Memo Native Live Coordination Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Memflow's proven live coordination behavior as native Memo
domains on operational ledger v2, with one Memo daemon and only `memo_*`
interfaces.

**Architecture:** Domain services append immutable v2 commands and query
transactional views. A dedicated signed Git transport replicates per-origin
segments, while one operational daemon owns sync, delivery, heartbeat, expiry,
and health loops. Continuity composes durable Memo briefing with active
operational state; it never reads Memflow.

**Tech Stack:** Python 3.13+, Memo operational ledger v2, stdlib SQLite/Git
subprocess/socket APIs, FastMCP, Click, pytest, mypy, ruff. No runtime import
from Memflow and no new third-party dependency.

## Global Constraints

- Plan 01 and its acceptance gate are prerequisites. Before Task 1, require a
  Plan 03 `CapabilityManifest` with `frozen=True`, zero blockers, complete
  signed evidence from both Macs, and a pinned manifest hash. Every fixture,
  event, API, CLI, and daemon loop in this plan must map to an admitted row;
  omitted source operations require an explicit signed `delete` disposition.
- Source Memflow files and tests are parity oracles only; Memo never imports the
  `memflow` package at runtime.
- No `flow_*` alias, wrapper, fallback, environment flag, or compatibility
  command is added.
- Task support is created only if the frozen manifest admits its exact source
  operations; complete synthetic/zero-use evidence removes the task
  event/API/test surface before Task 1 begins.
- Presence TTL defaults to 60 seconds, minimum 5, maximum 3,600.
- Operational sync polls every 3 seconds by default; presence TTL defaults to
  60 seconds and heartbeat to 15 seconds. Each lease derives its renewal
  interval as `min(configured_heartbeat_seconds, max(1.0, ttl_seconds / 3))`;
  the daemon never substitutes a global TTL.
- Continuity output is deterministic and capped at 12,000 characters.
- Terminal payloads are UTF-8 and at most 16,384 bytes.
- SQLite never synchronizes. Only immutable journal segments, anchors, heads,
  signed votes, and control records cross machines.
- The new operational daemon does not supervise or replace recall, ingest,
  maintenance, embed, or idle daemons.
- Tests never access real TTYs, remotes, LaunchAgents, state directories, or
  user configuration.
- Follow TDD and commit each task with explicit paths.

---

## File Structure

### Create

- `src/memo/operational_coordination.py` — channels, messages, handoffs, tasks.
- `src/memo/operational_delivery.py` — delivery/ACK lifecycle, retry, cursors.
- `src/memo/operational_presence.py` — leases, heartbeat, workspace conflicts.
- `src/memo/operational_continuity.py` — bounded deterministic composer.
- `src/memo/operational_session_loop.py` — admitted periodic checkpoint and
  recoverable-session freshness loop using Plan 01 session methods.
- `src/memo/terminal_bridge.py` — user-owned terminal registration and
  receipt-before-present.
- `src/memo/git_transport.py` — reusable Git lock/publish/recovery primitives.
- `src/memo/operational_sync.py` — dedicated remote publisher/ingester.
- `src/memo/operational_peer.py` — roster, signatures, host pinning, wakeup.
- `src/memo/operational_compaction.py` — retention, tombstones, signed prefix
  anchors, and safe segment removal.
- `src/memo/operational_control.py` — writer lease and daemon coordination
  around Plan 01's request-level `EpochFence`.
- `src/memo/operational_health.py` — health snapshot and counters.
- `src/memo/operational_daemon.py` — operational loop coordinator.
- `src/memo/memory/operational_ops.py` — core `Memory` API.
- `src/memo/server_coordination.py`
- `src/memo/server_delivery.py`
- `src/memo/server_presence.py`
- `src/memo/server_continuity.py`
- `src/memo/server_operational_runtime.py`
- `src/memo/cli_coordination.py`
- `src/memo/cli_presence.py`
- `src/memo/cli_operational_runtime.py`
- `src/memo/cli_peer.py`
- `tests/fixtures/memflow_parity/` — sanitized contract fixtures only.

### Modify

- `src/memo/operational.py` — compose the v2 journal/views with domain services.
- `src/memo/memory/facade.py` — include `_OperationalOpsMixin`.
- `src/memo/session.py` — use operational sessions and keep JSON as cache.
- `src/memo/server_idle_capture.py` — checkpoint through `Memory`.
- `src/memo/briefing.py`
- `src/memo/server_core_search.py`
- `src/memo/cli_session.py`
- `src/memo/sync_git.py` — extract shared primitives; durable sync behavior stays
  unchanged.
- `src/memo/daemon_common.py` — reuse paths/shutdown helpers.
- `src/memo/config.py` — operational remote, roster, signing keys, socket.
- `src/memo/flags_misc.py` — runtime cadences and bounded limits.
- `src/memo/errors.py`
- `src/memo/server.py`
- `src/memo/cli.py`
- `src/memo/cli_doctor.py`
- `src/memo/runtime/install.py`
- `src/memo/runtime/report.py`
- `pyproject.toml` — `memo-operational-daemon` entry point.

## Shared Interfaces

All domain views are immutable:

```python
@dataclass(frozen=True)
class MessageView:
    message_id: str
    event_id: str
    channel: str
    body: str
    actor_id: str
    target_ids: tuple[str, ...]
    topic: str
    expects_ack: bool
    expires_at: str | None


@dataclass(frozen=True)
class ChannelView:
    channel: str
    topic: str
    status: Literal["open", "terminated"]
    superseded_by_message_id: str | None
    terminated_at: str | None


@dataclass(frozen=True)
class HandoffView:
    id: str
    message_id: str
    project: str
    status: Literal["open", "consumed", "expired"]


@dataclass(frozen=True)
class TaskView:
    id: str
    project: str
    title: str
    status: Literal["open", "completed", "cancelled", "expired"]
    assignee_id: str | None
    result: str | None
    expires_at: str | None
    caused_by: str | None


@dataclass(frozen=True)
class DeliveryView:
    id: str
    message_id: str
    target_id: str
    state: Literal[
        "pending", "reserved", "presented", "acknowledged",
        "known_failed", "uncertain", "expired",
    ]
    terminal_id: str | None
    attempt_count: int
    next_attempt_at: str | None
    deadline_at: str | None
    last_error_code: str | None
    ack_actor_id: str | None
    ack_event_id: str | None


@dataclass(frozen=True)
class RetryPolicy:
    initial_delay_seconds: float = 1.0
    multiplier: float = 2.0
    max_delay_seconds: float = 60.0
    max_attempts: int = 8


@dataclass(frozen=True)
class CursorView:
    consumer_id: str
    channel: str
    logical_clock: str
    event_id: str


@dataclass(frozen=True)
class PresenceLease:
    id: str
    actor_id: str
    device_id: str
    project: str
    workspace: str
    topic: str
    intent: str
    files: tuple[str, ...]
    ttl_seconds: int
    expires_at: str


@dataclass(frozen=True)
class WorkspaceConflict:
    project: str
    file: str
    lease_ids: tuple[str, ...]


@dataclass(frozen=True)
class ContinuitySource:
    kind: str
    id: str
    title: str


@dataclass(frozen=True)
class ContinuityPacket:
    text: str
    sources: tuple[ContinuitySource, ...]
    omissions: tuple[str, ...]
    durable_available: bool
    operational_available: bool
    fallbacks: tuple[str, ...]


@dataclass(frozen=True)
class SyncResult:
    published_events: int
    ingested_events: int
    duplicates: int
    gaps: Mapping[str, int]
    pending: int


@dataclass(frozen=True)
class PeerIdentity:
    device_id: str
    host_id: str
    key_id: str
    algorithm: Literal["ed25519"]
    fingerprint: str
    public_key: str
    roles: tuple[str, ...]
    enrolled_at: str
    activation_roster_version: int
    activation_positions: Mapping[str, int]
    revoked_at: str | None
    revocation_positions: Mapping[str, int] | None


@dataclass(frozen=True)
class SignedRecoveryAuthorization:
    proposal_hash: str
    operator_key_ids: tuple[str, ...]
    quorum: int
    signatures: tuple[SignatureEnvelope, ...]
    expires_at: str


@dataclass(frozen=True)
class PeerRoster:
    version: int
    previous_hash: str | None
    proposal_hash: str
    peers: tuple[PeerIdentity, ...]
    recovery_keys: tuple[PublicKeyRecord, ...]
    recovery_quorum: int
    votes: tuple[SignedRosterVote, ...]
    recovery_authorization: SignedRecoveryAuthorization | None


@dataclass(frozen=True)
class RosterProposal:
    operation: Literal["enroll", "rotate", "revoke"]
    previous_roster_hash: str
    next_version: int
    peer_record_hash: str
    activation_positions: Mapping[str, int]
    nonce: str
    expires_at: str


@dataclass(frozen=True)
class SignedRosterAuthorization:
    proposal: RosterProposal
    proposal_hash: str
    votes: tuple[SignedRosterVote, ...]
    recovery_authorization: SignedRecoveryAuthorization | None


@dataclass(frozen=True)
class PeerHealth:
    device_id: str
    last_verified_head_at: str | None
    last_verified_ingest_at: str | None
    last_verified_publish_at: str | None
    last_verified_wakeup_at: str | None
    stale_after: str
    verdict: Literal["healthy", "stale", "unhealthy"]


@dataclass(frozen=True)
class OperationalHealthSnapshot:
    schema: Literal["memo.operational_health.v1"]
    verdict: Literal["healthy", "degraded", "unhealthy"]
    memo_version: str
    runtime_digest: str
    authority_epoch: int
    control_oid: str
    roster_version: int
    writer_lease_owner: str
    writer_lease_expires_at: str
    local_heads: Mapping[str, int]
    remote_heads: Mapping[str, int]
    peers: tuple[PeerHealth, ...]
    gaps: Mapping[str, int]
    oldest_pending_at: str | None
    pending_deliveries: int
    retry_count: int
    deduplicated_count: int
    expired_count: int
    outbox_pending: int
    outbox_failures: int
    ack_latency_p95_ms: float | None
    view_lag_events: int
    rebuild_status: str
    listener_probe: str
    active_connections: int
    drain_counters: Mapping[str, int]
    checked_at: str
```

Exact method signatures:

- `CoordinationService.send_message(*, identity: PrincipalIdentity, channel:
  str, body: str, target_ids: tuple[str, ...], topic: str, evidence_uris: tuple[str,
  ...], expects_ack: bool, expires_at: str | None, idempotency_key: str,
  caused_by: str | None = None) -> MessageView`
- `CoordinationService.supersede_message(*, identity: PrincipalIdentity,
  message_id: str, body: str, idempotency_key: str) -> MessageView`
- `CoordinationService.terminate_topic(*, identity: PrincipalIdentity, channel:
  str, topic: str, idempotency_key: str) -> ChannelView`
- `CoordinationService.unread_count(*, identity: PrincipalIdentity, channel:
  str) -> int`
- `CoordinationService.create_handoff(*, identity: PrincipalIdentity, project:
  str, summary: str, target_id: str | None, topic: str, evidence_uris:
  tuple[str, ...], expects_ack: bool, expires_at: str | None, idempotency_key:
  str) -> HandoffView`
- `CoordinationService.create_task(*, identity: PrincipalIdentity, project: str,
  title: str, assignee_id: str | None, expires_at: str | None, caused_by: str |
  None, idempotency_key: str) -> TaskView`
- `CoordinationService.consume_handoff(*, identity: PrincipalIdentity,
  handoff_id: str, idempotency_key: str) -> HandoffView`
- `CoordinationService.complete_task(*, identity: PrincipalIdentity, task_id:
  str, result: str, idempotency_key: str) -> TaskView`
- `CoordinationService.assign_task(*, identity: PrincipalIdentity, task_id:
  str, assignee_id: str, idempotency_key: str) -> TaskView`
- `CoordinationService.cancel_task(*, identity: PrincipalIdentity, task_id:
  str, idempotency_key: str) -> TaskView`
- `CoordinationService.expire_tasks(*, now: datetime) -> int`
- `CoordinationService.handoff_status(handoff_id: str) -> HandoffView`
- `CoordinationService.task_status(task_id: str) -> TaskView`
- `CoordinationService.messages(*, channel: str, after_event_id: str | None =
  None, limit: int = 100) -> list[MessageView]`
- `DeliveryService.transition(delivery_id: str, event: Literal["reserved",
  "presented", "known_failed", "uncertain", "expired"], detail: str = "") ->
  DeliveryView`
- `DeliveryService.acknowledge(*, identity: PrincipalIdentity, message_id: str,
  idempotency_key: str) -> DeliveryView`
- `DeliveryService.advance_cursor(*, identity: PrincipalIdentity, channel: str,
  logical_clock: str, event_id: str, idempotency_key: str) -> CursorView`
- `DeliveryService.expire_due(*, now: datetime) -> int`
- `DeliveryService.retry_policy() -> RetryPolicy`
- `PresenceService.announce(*, identity: PrincipalIdentity, project: str,
  workspace: str, topic: str, intent: str, files: tuple[str, ...],
  ttl_seconds: int, idempotency_key: str) -> PresenceLease`
- `PresenceService.active(*, project: str, now: datetime) ->
  list[PresenceLease]`
- `PresenceService.renew(*, identity: PrincipalIdentity, lease_id: str,
  ttl_seconds: int, idempotency_key: str) -> PresenceLease`
- `PresenceService.conflicts(*, project: str, files: tuple[str, ...], now:
  datetime) -> list[WorkspaceConflict]`
- `ContinuityComposer.compose(*, query: str = "", cwd: str | None = None,
  max_chars: int = 12_000) -> ContinuityPacket`
- `OperationalSync.publish() -> SyncResult`
- `OperationalSync.ingest() -> SyncResult`
- `OperationalSync.recover_gap(*, device_id: str, expected_sequence: int) ->
  RecoveryResult`
- `OperationalSync.status() -> OperationalSyncStatus`
- `PeerRosterStore.enroll(*, peer: PeerIdentity, authorization:
  SignedRosterAuthorization) -> PeerRoster`
- `PeerRosterStore.rotate_key(*, device_id: str, new_key: PublicKeyRecord,
  old_key_proof: SignedKeyProof, new_key_proof: SignedKeyProof,
  authorization: SignedRosterAuthorization) -> PeerRoster`
- `PeerRosterStore.revoke(*, device_id: str, authorization:
  SignedRosterAuthorization) -> PeerRoster`
- `PeerRosterStore.current() -> PeerRoster`
- `PeerRosterStore.verify(record: OperationalEventV2 | ChainAnchor | SignedHead
  | SignedWakeup, *, at_roster_version: int) -> PeerIdentity`

`RecoveryResult` contains `device_id`, `requested_sequence`, `recovered_events`,
and `remaining_gap`. `OperationalSyncStatus` contains the local/remote heads,
last publish/ingest timestamps, pending count, gaps, and health. These fields
are used unchanged by daemon health and public status.

The signed operation map is executable scope. For every admitted row, the
corresponding method above, fully qualified event, reducer transition, MCP/CLI
surface, and parity test are mandatory. The conditional set covers broadcast
and multiple recipients, message supersede, topic termination, unread cursor,
handoff status, task status/assignee/result/cancel/expiry, active workspace,
peer notification, fast lane, and health. An operation can be absent from the
implementation only when Plan 03 contains a signed `disposition="delete"` row
for that exact source operation before Task 2 begins. Task 10 asserts that the
exported method/MCP/CLI set equals the admitted signed rows exactly: neither
missing surfaces nor unmanifested extras are allowed.

`SignedRosterAuthorization` aggregates the signatures required by operation
over a canonical `RosterProposal` hash, previous roster hash, next version, and
expiry. Bootstrap enrollment is the only one-peer case. Rotation requires
proof from both the old and new keys. Revocation takes effect at the roster's
activation sequence; records
signed by the revoked key after that sequence fail closed, while historical
records remain verifiable against their recorded roster version.

`RosterProposal` contains no signatures or votes. Its canonical
`proposal_hash` is computed first, so every `SignedRosterVote` signs the same
immutable hash plus voter device/key ID and Ed25519 `SignatureEnvelope`;
neither the final roster hash nor the aggregate is signed recursively.
`SignedKeyProof` uses the same envelope and binds old/new fingerprints. Roster
publication is CAS against the previous blob hash.
Enrollment and rotation require exactly one unexpired valid vote from every
currently active peer. Revocation requires every active peer except the target
plus a `SignedRecoveryAuthorization` from the separately enrolled
operator/recovery quorum; the target's vote is optional and never blocks
removal of a lost or compromised peer. Recovery keys cannot sign operational
events or heads. The final roster hash is computed only after proposal, votes,
and recovery authorization validate. Bootstrap enrollment is the sole
exception.
Removing a peer stops it blocking compaction only after every remaining peer
has ingested the removal roster and advanced past all recorded revocation
positions; historical key records remain in the roster chain forever.

### Task 1: Freeze selected Memflow parity fixtures

**Files:**
- Create: `tests/fixtures/memflow_parity/README.md`
- Create: `tests/fixtures/memflow_parity/*.json`
- Create: `tests/fixtures/memflow_parity/operation-map.json`
- Create: `tests/fixtures/memflow_parity/slo-baseline.json`
- Create: `tests/test_memflow_parity_fixtures.py`

**Interfaces:**
- Consumes: the signed Plan 03 `OperationMappingRow` and `SloBaseline` sets,
  their manifest-bound digests, selected Memflow source tests, and sanitized
  JSON shapes.
- Produces: immutable fixtures with source operation/test/commit, target
  predicate-dispatched Memo route(s), parameter/default/result/error mapping,
  atomic grouping, expected output, SHA-256, and
  hard SLO thresholds.

- [ ] **Step 1: Write the fixture-schema test**

```python
def test_every_parity_fixture_has_provenance_and_digest():
    for path in PARITY_ROOT.glob("*.json"):
        raw = json.loads(path.read_text())
        assert raw["source_commit"]
        assert raw["source_test"].startswith("tests/")
        assert raw["source_operation"]
        assert raw["routes"] or raw["disposition"] == "delete"
        for route in raw["routes"]:
            assert route["memo_methods"]
            assert route["transform_id"]
            assert route["fixture_sha256"]
        assert raw["capability"] in {
            "coordination", "delivery", "presence", "session",
            "continuity", "terminal", "sync", "health",
        }
        assert sha256_payload(raw["expected"]) == raw["expected_sha256"]
```

- [ ] **Step 2: Run selected source tests in the Memflow worktree**

```bash
cd /Users/fer/repos/memflow
uv run --no-sync pytest \
  tests/test_channel_supersede.py tests/test_handoff_evidence.py \
  tests/test_delivery.py tests/test_delivery_idempotency.py \
  tests/test_presence.py tests/test_heartbeat.py \
  tests/test_session_snapshot.py tests/test_continuity.py \
  tests/test_terminal.py tests/test_peer_sync.py \
  tests/test_health_watchdog.py tests/test_health_verdict.py \
  tests/test_kernel.py tests/test_integration_cross_feature.py \
  tests/test_peer_notify.py tests/test_peers.py tests/test_tasks.py \
  tests/test_fast_lane.py tests/test_authoring_tty.py \
  tests/test_sync_conflict.py tests/test_sync_lock.py \
  tests/test_git_sync_rebase_recovery.py tests/test_remote_breaker.py \
  tests/test_event_store_health.py tests/test_runtime_observatory.py -v
```

Expected: record the exact pass/fail baseline. Do not convert a failing source
test into an oracle without adding a reviewed interpretation to the fixture
README.

- [ ] **Step 3: Add only sanitized, stable fixture shapes**

Each JSON fixture must contain:

```json
{
  "schema": "memo.memflow_parity.v1",
  "capability": "delivery",
  "source_operation": "flow_delivery_ack",
  "source_commit": "5426e8e5fce83d8ccf98fa7ecba3bcc634531ae2",
  "source_test": "tests/test_delivery_idempotency.py::test_duplicate_ack",
  "disposition": "absorb",
  "routes": [{
    "route_id": "ack-default",
    "predicate": {"mode": {"eq": "ack"}},
    "memo_methods": ["Memory.delivery_ack"],
    "memo_mcp": ["memo_delivery_ack"],
    "memo_cli": ["memo delivery ack"],
    "parameter_mapping": {"message_id": "message_id"},
    "defaults": {},
    "result_mapping": {"state": "state", "ack_count": "ack_count"},
    "error_mapping": {"not_found": "not_found"},
    "transform_id": "delivery-ack-v1",
    "fixture_sha256": [
      "64911fb8f0185e83a56a1674a8d01bb51682cffc2f7c23fb51d756b0d8307079"
    ],
    "atomic_group": null
  }],
  "input": {"message_id": "m1", "acks": ["a1", "a1"]},
  "expected": {"state": "acknowledged", "ack_count": 1},
  "expected_sha256": "64911fb8f0185e83a56a1674a8d01bb51682cffc2f7c23fb51d756b0d8307079"
}
```

The shown commit is the audited Memflow baseline. If execution intentionally
rebases the source oracle, update every fixture to the reviewed source commit
and recompute the digest from canonical JSON. Never include user text, absolute
paths, hostnames, or live event IDs.

Verify and copy `operation-map.json` and `slo-baseline.json` byte-for-byte from
the canonical artifacts already signed into the Plan 03 manifest; Plan 02
never originates or amends them. Fail when any
admitted source operation lacks a target method, MCP/CLI disposition, parameter
mapping, defaults, result/error mapping, deterministic transform, complete
route predicates, fixture digests, and parity test. Composite routes with an
`atomic_group` must prove all effects in one transaction. This includes
broadcast, supersede,
topic termination, unread cursor, handoff/task status, task
expiry/result/assignee, heartbeat/active-workspace, peer notification, fast
lane, and health surfaces. An intentionally omitted operation must carry the
manifest's signed `delete` row.

`slo-baseline.json` records immutable source commit, workload ID, machine
class, sample count, p50/p95/p99/max visibility and recovery milliseconds,
error rate, and frozen tolerances. Plan 02 assertions require at least 100
samples per workload, zero data loss/duplicates, Memo p95/p99 no more than the
frozen tolerance, and recovery within the exact recorded threshold; “record
for later” is not a gate.

- [ ] **Step 4: Run fixture tests and ruff**

```bash
cd /Users/fer/repos/memo
uv run --no-sync pytest tests/test_memflow_parity_fixtures.py -v
uv run --no-sync ruff check tests/test_memflow_parity_fixtures.py
```

Expected: all fixtures validate.

- [ ] **Step 5: Commit parity fixtures**

```bash
git add tests/fixtures/memflow_parity tests/test_memflow_parity_fixtures.py
git commit -m "test: freeze memflow operational parity fixtures"
```

### Task 2: Implement native coordination, handoffs, and tasks

**Files:**
- Create: `src/memo/operational_coordination.py`
- Create: `tests/test_operational_coordination.py`
- Modify: `src/memo/operational.py`

**Interfaces:**
- Consumes: Plan 01 `OperationalStore.commit`, v2 views.
- Produces: `CoordinationService`; event types
  `coord.channel.opened.v1`, `coord.message.sent.v1`,
  `coord.message.superseded.v1`, `coord.topic.terminated.v1`,
  `coord.handoff.created.v1`, `coord.handoff.consumed.v1`,
  `coord.task.created.v1`, `coord.task.assigned.v1`,
  `coord.task.completed.v1`, `coord.task.cancelled.v1`, and
  `coord.task.expired.v1`, restricted to the signed admitted set.

- [ ] **Step 1: Write failing idempotency/lifecycle tests**

```python
def test_send_and_handoff_are_idempotent(service, identity):
    kwargs = dict(
        identity=identity, channel="handoff", body="resume here",
        target_ids=("agent-b", "agent-c"), topic="absorption",
        evidence_uris=("commit:abc",),
        expects_ack=True, expires_at=None, idempotency_key="send-1",
    )
    first = service.send_message(**kwargs)
    second = service.send_message(**kwargs)
    assert second.message_id == first.message_id
    assert service.messages(channel="handoff") == [first]


def test_consumed_handoff_cannot_regress(service, handoff, identity):
    consumed = service.consume_handoff(
        identity=identity, handoff_id=handoff.id, idempotency_key="consume-1"
    )
    assert consumed.status == "consumed"
    assert service.consume_handoff(
        identity=identity, handoff_id=handoff.id, idempotency_key="consume-1"
    ) == consumed
```

Also cover target authorization, evidence URI preservation, expiry, concurrent
ordering, task complete idempotency, and invisible expired messages. For every
admitted mapping row, cover broadcast/multiple recipients, supersede,
terminated topics, unread count/cursor, handoff status, and the full
task assignee/result/cancel/expiry/status lifecycle. If the signed row says
`delete`, assert the method, event, MCP tool, and CLI command are all absent.

- [ ] **Step 2: Run the test and confirm the service is missing**

```bash
uv run --no-sync pytest tests/test_operational_coordination.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement command builders and reducers**

Every mutation builds one `OperationalCommand`; reducers are monotonic and
reject consumed→open or completed→open without an explicit reopen event.
Messages use `subject_uri="memo://coord/message/<event-id>"`; handoffs and tasks
reference their source message via `caused_by`.

- [ ] **Step 4: Run focused and parity tests**

```bash
uv run --no-sync pytest \
  tests/test_operational_coordination.py \
  tests/test_operational_memory.py \
  tests/test_memflow_parity_fixtures.py -v
uv run --no-sync mypy src/memo/operational_coordination.py
uv run --no-sync ruff check \
  src/memo/operational_coordination.py src/memo/operational.py \
  tests/test_operational_coordination.py
```

- [ ] **Step 5: Commit coordination**

```bash
git add \
  src/memo/operational_coordination.py src/memo/operational.py \
  tests/test_operational_coordination.py
git commit -m "feat: add native Memo coordination"
```

### Task 3: Implement delivery, ACK, retries, and cursors

**Files:**
- Create: `src/memo/operational_delivery.py`
- Create: `tests/test_operational_delivery.py`

**Interfaces:**
- Consumes: coordination message IDs and v2 command store.
- Produces: `DeliveryService`; registry events
  `delivery.reserved.v1`, `delivery.presented.v1`,
  `delivery.acknowledged.v1`, `delivery.known_failed.v1`,
  `delivery.uncertain.v1`, `delivery.expired.v1`, and
  `delivery.cursor.advanced.v1`.
- `DeliveryService.reserve_due(*, now: datetime, limit: int = 100) ->
  list[DeliveryView]`
- `DeliveryService.reconcile_uncertain(*, delivery_id: str,
  presenter_receipt: PresenterReceipt | None, idempotency_key: str) ->
  DeliveryView`

- [ ] **Step 1: Write failing state-machine tests**

```python
@pytest.mark.parametrize(
    ("start", "event", "expected"),
    [
        ("pending", "reserved", "reserved"),
        ("reserved", "presented", "presented"),
        ("presented", "acknowledged", "acknowledged"),
        ("pending", "expired", "expired"),
        ("reserved", "known_failed", "known_failed"),
        ("reserved", "uncertain", "uncertain"),
    ],
)
def test_delivery_transitions(service, delivery_factory, start, event, expected):
    delivery = delivery_factory(state=start)
    assert service.transition(delivery.id, event).state == expected


def test_cursor_and_ack_never_regress(service, delivered, identity):
    ack = service.acknowledge(
        identity=identity, message_id=delivered.message_id,
        idempotency_key="ack-1",
    )
    assert ack.state == "acknowledged"
    with pytest.raises(OperationalError):
        service.transition(delivered.id, "presented")
    first = service.advance_cursor(
        identity=identity, channel="handoff",
        logical_clock="10-0-device-b", event_id="e10",
        idempotency_key="cursor-10",
    )
    with pytest.raises(OperationalError):
        service.advance_cursor(
            identity=identity, channel="handoff",
            logical_clock="9-0-device-b", event_id="e9",
            idempotency_key="cursor-9",
        )
    assert first.event_id == "e10"
```

Cover atomic message-to-per-recipient-pending creation, ACK-before-presentation
join, ACK actor/target authorization, duplicate ACK, exponential bounded retry
retaining logical message ID, deadline/TTL, late expiry, concurrent cursor join,
known-failed retry, uncertain reconciliation without blind retry, and tombstone
redelivery suppression. Cursor/unread tests derive `consumer_id` from the
authenticated principal, require idempotency on advancement, and reject a
spoofed consumer in service, MCP, and CLI adapters.

- [ ] **Step 2: Run failing delivery tests**

```bash
uv run --no-sync pytest tests/test_operational_delivery.py -v
```

- [ ] **Step 3: Implement the monotonic reducer**

Use a transition table, not nested conditionals:

```python
_ALLOWED = {
    "pending": {"reserved", "expired"},
    "reserved": {"presented", "known_failed", "uncertain", "expired"},
    "presented": {"acknowledged", "expired"},
    "known_failed": {"reserved", "expired"},
    "uncertain": {"presented", "known_failed", "expired"},
    "acknowledged": set(),
    "expired": set(),
}
```

Duplicate events with the same idempotency key return the stored view. An ACK
causally referencing a presented message joins to `acknowledged` even if its
replica has not reduced `presented` yet; only the target identity (or an
explicitly authorized delegate) may ACK, and it cannot acknowledge an unknown
message. `coord.message.sent.v1` contains canonical ordered recipient IDs. Its
single reducer transaction creates the message plus one pending delivery per
recipient with deterministic ID
`sha256(message_event_id + "\\0" + target_id)`; there is no separate pending
event or batch-append requirement. Import, replay, and normal send use this
same rule. Backoff, attempt count, next-attempt time, deadline, last error, and
ACK identity are event-derived. The default retry delay for attempt `n` is
`min(1.0 * 2 ** (n - 1), 60.0)` seconds with at most 8 presentation attempts;
config may only supply a validated `RetryPolicy` with positive finite values
and `max_attempts <= 32`. `expires_at` sets `deadline_at`; when it is absent,
`deadline_at` is `None` but max attempts still terminates the delivery as
`known_failed`. A nonterminal delivery with no deadline is never compacted;
after acknowledgement, expiry, or terminal known failure, the normal signed
tombstone horizon governs compaction. Automatic retry is permitted only from
`known_failed`; an
`uncertain` side effect requires a signed presenter receipt or explicit
negative reconciliation.

- [ ] **Step 4: Run focused gates**

```bash
uv run --no-sync pytest \
  tests/test_operational_delivery.py \
  tests/test_operational_coordination.py -v
uv run --no-sync mypy src/memo/operational_delivery.py
uv run --no-sync ruff check \
  src/memo/operational_delivery.py tests/test_operational_delivery.py
```

- [ ] **Step 5: Commit delivery**

```bash
git add src/memo/operational_delivery.py tests/test_operational_delivery.py
git commit -m "feat: add delivery and acknowledgement state"
```

### Task 4: Implement presence, heartbeat, and workspace conflicts

**Files:**
- Create: `src/memo/operational_presence.py`
- Create: `tests/test_operational_presence.py`

**Interfaces:**
- Consumes: v2 commands and normalized project/workspace paths.
- Produces: `PresenceService`; lease/heartbeat events and derived conflicts.

- [ ] **Step 1: Write failing TTL/conflict tests**

```python
def test_announce_clamps_ttl_and_renew_keeps_lease(service, identity, clock):
    lease = service.announce(
        identity=identity, project="memo", workspace="/work/memo",
        topic="ledger", intent="editing", files=("src/memo/a.py",),
        ttl_seconds=1, idempotency_key="presence-1",
    )
    assert lease.ttl_seconds == 5
    renewed = service.renew(
        identity=identity, lease_id=lease.id,
        ttl_seconds=60, idempotency_key="presence-2",
    )
    assert renewed.id == lease.id


def test_conflicts_ignore_self_and_expired_leases(service, two_identities, clock):
    a, b = two_identities
    announce(service, a, files=("src/memo/a.py",), ttl=60)
    announce(service, b, files=("src/memo/a.py",), ttl=60)
    assert len(service.conflicts(
        project="memo", files=("src/memo/a.py",), now=clock.now()
    )) == 1
    clock.advance(seconds=61)
    assert service.active(project="memo", now=clock.now()) == []
```

Also test heartbeat renews without creating a handoff, minimum-TTL renewal at
the per-lease interval, max TTL 3,600, portable workspace identity
(`repo_remote_hash + relative_worktree`), path normalization, clock-drift
quarantine/recovery, and reduction of an 80k-event-shaped fixture to only
active leases.

- [ ] **Step 2: Run the tests and confirm missing service**

```bash
uv run --no-sync pytest tests/test_operational_presence.py -v
```

- [ ] **Step 3: Implement leases and derived conflicts**

Expiration is each renewal's signed event time plus clamped TTL. Receivers
quarantine origins outside configured drift and recover only after a fresh
valid renewal. Heartbeat is a lease-renew event, never a handoff, and its
interval is computed per lease as
`min(configured_heartbeat_seconds, max(1.0, ttl_seconds / 3))`. Normalize files
relative to the portable canonical workspace before overlap comparison.

- [ ] **Step 4: Run focused gates**

```bash
uv run --no-sync pytest \
  tests/test_operational_presence.py tests/test_operational_coordination.py -v
uv run --no-sync mypy src/memo/operational_presence.py
uv run --no-sync ruff check \
  src/memo/operational_presence.py tests/test_operational_presence.py
```

- [ ] **Step 5: Commit presence**

```bash
git add src/memo/operational_presence.py tests/test_operational_presence.py
git commit -m "feat: add presence leases and conflicts"
```

### Task 5: Extend sessions and build continuity composition

**Files:**
- Create: `src/memo/operational_continuity.py`
- Create: `tests/test_operational_continuity.py`
- Modify: `src/memo/session.py`
- Modify: `src/memo/server_idle_capture.py`
- Modify: `src/memo/server_session_patterns.py`
- Modify: `src/memo/briefing.py`
- Modify: `src/memo/server_core_search.py`
- Modify: `src/memo/cli_session.py`
- Create: `src/memo/operational_session_loop.py`
- Create: `tests/test_operational_session_loop.py`

**Interfaces:**
- Consumes: Plan 01 sessions, `memo_unified_briefing`, active coordination,
  delivery, presence, and runtime health.
- Produces: `ContinuityComposer.compose -> ContinuityPacket`.
- Produces, only when admitted by the manifest:
  `OperationalSessionLoop.run_once(now: datetime) -> SessionLoopReport`.
  It calls the Plan 01 `Memory` session signatures; it defines no second
  checkpoint/recover API.

- [ ] **Step 1: Write failing deterministic/bounded tests**

```python
def test_continuity_order_cap_and_provenance(composer):
    packet = composer.compose(query="resume", cwd="/work/memo", max_chars=500)
    assert packet.text.index("Durable briefing") < packet.text.index("Handoffs")
    assert packet.text.index("Handoffs") < packet.text.index("Checkpoint")
    assert len(packet.text) <= 500
    assert packet.omissions
    assert all(source.id for source in packet.sources)


def test_continuity_degrades_without_operational_views(composer, break_views):
    packet = composer.compose(cwd="/work/memo")
    assert "operational state unavailable" in packet.text.lower()
    assert packet.durable_available is True
    assert "memflow" not in packet.fallbacks
```

Also test latest recoverable selection, terminated exclusion, derived JSON
cache rebuild, stable section order, 12,000 hard cap, periodic checkpoint
freshness/restart/latest selection, and no Memflow import.

- [ ] **Step 2: Run failing continuity/session tests**

```bash
uv run --no-sync pytest \
  tests/test_operational_continuity.py \
  tests/test_operational_session_loop.py \
  tests/test_operational_sessions_v2.py \
  tests/test_session.py tests/test_briefing_unified.py -v
```

- [ ] **Step 3: Implement composition and route checkpoint writers**

Compose in this exact order: durable briefing; unresolved handoffs; pending
deliveries and conflicts; latest recoverable checkpoint; runtime/peer health;
omissions. Allocate the character budget per section, trim at record
boundaries, and always reserve space for omissions. `server_idle_capture` and
CLI session code call `Memory` session methods rather than writing JSON
authority directly. `server_session_patterns.py` only registers the Plan 01
`memo_session_*` tools. If the manifest admits Memflow's snapshot loop,
`OperationalSessionLoop` checkpoints changed live sessions at the admitted
cadence and marks only the newest valid checkpoint recoverable.

- [ ] **Step 4: Run focused gates**

```bash
uv run --no-sync pytest \
  tests/test_operational_continuity.py \
  tests/test_operational_session_loop.py \
  tests/test_operational_sessions_v2.py tests/test_session.py \
  tests/test_continuity.py tests/test_briefing_unified.py -v
uv run --no-sync mypy \
  src/memo/operational_continuity.py src/memo/operational_session_loop.py \
  src/memo/briefing.py
uv run --no-sync ruff check \
  src/memo/operational_continuity.py src/memo/session.py \
  src/memo/server_idle_capture.py src/memo/briefing.py \
  src/memo/server_core_search.py src/memo/server_session_patterns.py \
  src/memo/cli_session.py src/memo/operational_session_loop.py \
  tests/test_operational_continuity.py tests/test_operational_session_loop.py
```

- [ ] **Step 5: Commit continuity**

```bash
git add \
  src/memo/operational_continuity.py src/memo/session.py \
  src/memo/server_idle_capture.py src/memo/briefing.py \
  src/memo/server_core_search.py src/memo/server_session_patterns.py \
  src/memo/cli_session.py src/memo/operational_session_loop.py \
  tests/test_operational_continuity.py tests/test_operational_session_loop.py
git commit -m "feat: compose native Memo continuity"
```

### Task 6: Add a controlled terminal bridge

**Files:**
- Create: `src/memo/terminal_bridge.py`
- Create: `tests/test_terminal_bridge.py`

**Interfaces:**
- Produces:
  - `TerminalBridge.register(registration: TerminalRegistration, *, peer_uid:
    int) -> TerminalRegistration`
  - `TerminalBridge.present(request: TerminalPresentRequest, *, peer_uid: int)
    -> PresenterReceipt`
  - `TerminalBridge.reconcile(*, terminal_id: str, event_id: str,
    observation: Literal["presented", "not_presented"]) -> PresenterReceipt`
  - local socket mode `0600`, terminal registry, and receipt table.

`TerminalRegistration` contains terminal ID, authenticated principal/session,
UID, capabilities (`notify` and/or `inject`), issued/expiry times, nonce, and
signature. `TerminalPresentRequest` contains event/message/delivery IDs,
terminal ID, mode, sanitized payload hash, deadline, idempotency key, and
principal. `PresenterReceipt` contains those IDs plus
`reserved|presented|known_failed|uncertain`, attempt, presenter timestamp,
error code, and receipt hash.

- [ ] **Step 1: Write failing security/idempotency tests**

```python
def test_bridge_rejects_foreign_uid_and_sanitizes_payload(bridge, presenter):
    target = bridge.register(registration("term-1"), peer_uid=os.getuid())
    with pytest.raises(OperationalError):
        bridge.present(request(target.id, "hello"), peer_uid=os.getuid() + 1)
    receipt = bridge.present(
        request(target.id, "\x1b]0;bad\x07safe\n", event_id="e1"),
        peer_uid=os.getuid(),
    )
    assert presenter.calls == ["safe\n"]
    assert receipt.event_id == "e1"


def test_duplicate_reserves_before_side_effect(bridge, presenter):
    req = request("term-1", "handoff", event_id="e1")
    bridge.present(req, peer_uid=os.getuid())
    bridge.present(req, peer_uid=os.getuid())
    assert presenter.call_count == 1
```

Cover unregistered target, unauthorized mode, 16,384-byte limit, C0/DEL
stripping, socket permissions, expired registration/re-registration, principal
and session binding, notify/inject separation, known-failure retry inside TTL,
and uncertain crash state that is reported but not automatically re-presented.

- [ ] **Step 2: Run the test and confirm missing bridge**

```bash
uv run --no-sync pytest tests/test_terminal_bridge.py -v
```

- [ ] **Step 3: Implement registration, sanitization, reserve, present, receipt**

Public APIs accept `terminal_id`, never `/dev/tty*`. Verify peer UID from the
Unix socket, authorize `notify` separately from `inject`, sanitize before
hashing, reserve `(terminal_id, event_id)` transactionally, call an injected
presenter, then store the receipt. A crash after reservation is `uncertain`;
automatic retry is forbidden until `reconcile` supplies a definite negative
observation. A presenter-returned known failure may retry within the request
deadline using the same logical event and a higher attempt.

- [ ] **Step 4: Run terminal gates**

```bash
uv run --no-sync pytest tests/test_terminal_bridge.py -v
uv run --no-sync mypy src/memo/terminal_bridge.py
uv run --no-sync ruff check \
  src/memo/terminal_bridge.py tests/test_terminal_bridge.py
```

- [ ] **Step 5: Commit terminal bridge**

```bash
git add src/memo/terminal_bridge.py tests/test_terminal_bridge.py
git commit -m "feat: add controlled terminal delivery bridge"
```

### Task 7: Add signed operational Git sync

**Files:**
- Create: `src/memo/git_transport.py`
- Create: `src/memo/operational_peer.py`
- Create: `src/memo/operational_sync.py`
- Create: `tests/test_git_transport.py`
- Create: `tests/test_operational_peer.py`
- Create: `tests/test_cli_peer.py`
- Create: `tests/test_operational_sync.py`
- Modify: `src/memo/sync_git.py`
- Modify: `src/memo/config.py`
- Modify: `src/memo/cli.py`
- Modify: `src/memo/cli_doctor.py`

**Interfaces:**
- Consumes: ledger bundles and anchors.
- Produces: `OperationalSync`, `PeerRoster`, signed heads/anchors/wakeups, and
  `memo peer enroll|rotate|revoke|status` with dry-run, CAS, and signed
  receipts.

- [ ] **Step 1: Write failing two-clone and signature tests**

```python
def test_single_writer_publish_and_incremental_ingest(two_clones, peer_roster):
    a, b = two_clones
    a.ledger.append(
        command("coord.message.sent.v1"), context=a.system_commit_context()
    )
    assert a.sync.publish().published_events == 1
    assert b.sync.ingest().ingested_events == 1
    assert b.views.state() == a.views.state()


def test_gap_blocks_reduce_until_recovered(two_clones, peer_roster):
    a, b = two_clones
    publish_sequences(a, [1, 2, 3])
    remove_remote_segment_for_sequence(a.remote, 2)
    report = b.sync.ingest()
    assert report.gaps == {"device-a": 2}
    assert b.views.cursor("device-a") == 1
```

Also test origin-only writes, immutable segments, invalid head signature,
anchor regression, fork, local lock, offline pending, invalid wakeup, host-key
mismatch, unchanged durable `sync_git` behavior, bootstrap enrollment,
unanimous enrollment, dual-proof key rotation, revocation activation,
post-revocation rejection, expired authorization, roster rollback, and
historical verification against the recorded roster version. Prove proposal
hashing excludes votes/final roster hash, and that a lost target can be revoked
only with every remaining peer vote plus the independent recovery quorum.
The CLI tests also cover dry-run purity, stale CAS rejection, expired or
incomplete votes, dual-key rotation proof, receipt verification, and strict
doctor failure when the installed roster/key history differs from the
published chain.

- [ ] **Step 2: Run failing sync tests**

```bash
uv run --no-sync pytest \
  tests/test_git_transport.py \
  tests/test_operational_peer.py tests/test_cli_peer.py \
  tests/test_operational_sync.py \
  tests/test_sync_git.py -v
```

- [ ] **Step 3: Extract Git primitives and implement dedicated transport**

Use this remote layout:

```text
events/<device>/<ledger-epoch>/<segment>.jsonl
anchors/<device>/<ledger-epoch>/<sequence>.json
checkpoints/<device>/<ledger-epoch>/<sequence>.json
heads/<device>.json
control/peers/roster-v<version>.json
control/<attempt>/<device-vote>.json
```

Only the enrolled origin key may advance its head. Publish fsynced immutable
segments first and the signed head last. Ingest verifies device enrollment,
signature, anchor, segment hashes, continuity, and watermark before importing.
Extract lock/rebase/retry helpers from `sync_git.py` without changing durable
sync semantics. Roster updates are append-only, hash-linked, monotonically
versioned, and published with compare-and-swap against the previous roster
blob. A peer must ingest and validate the complete roster chain before it
accepts a head or segment that names a newer roster version.
The first `PeerRoster` adopts Plan 01's persisted
`VerificationRoster.bootstrap` record with the same version, hash, and key;
there is no unsigned or second trust-root bootstrap.

`cli_peer.py` exposes only the four lifecycle commands. Mutation commands
default to a printable dry-run plan and require `--apply`,
`--expected-roster-hash`, an exact signed authorization file, and an
idempotency key. Successful apply prints and stores a signed
`RosterChangeReceipt` bound to old/new hashes, version, operation, actor,
timestamp, and published Git OID. `memo peer status` and strict doctor verify
the full roster/key history, local Keychain key availability, publication
OID, revocation activation positions, and receipt chain.

- [ ] **Step 4: Run sync gates**

```bash
uv run --no-sync pytest \
  tests/test_git_transport.py tests/test_operational_peer.py \
  tests/test_operational_sync.py tests/test_cli_peer.py \
  tests/test_sync_git.py -v
uv run --no-sync mypy \
  src/memo/git_transport.py src/memo/operational_peer.py \
  src/memo/operational_sync.py src/memo/cli_peer.py
uv run --no-sync ruff check \
  src/memo/git_transport.py src/memo/operational_peer.py \
  src/memo/operational_sync.py src/memo/sync_git.py src/memo/cli_peer.py \
  src/memo/cli.py src/memo/cli_doctor.py \
  tests/test_git_transport.py tests/test_operational_peer.py \
  tests/test_operational_sync.py tests/test_cli_peer.py
```

- [ ] **Step 5: Commit operational sync**

```bash
git add \
  src/memo/git_transport.py src/memo/operational_peer.py \
  src/memo/operational_sync.py src/memo/sync_git.py src/memo/config.py \
  src/memo/cli_peer.py src/memo/cli.py src/memo/cli_doctor.py \
  tests/test_git_transport.py tests/test_operational_peer.py \
  tests/test_operational_sync.py tests/test_cli_peer.py tests/test_sync_git.py
git commit -m "feat: add signed operational sync"
```

### Task 8: Implement bounded retention and signed prefix compaction

**Files:**
- Create: `src/memo/operational_compaction.py`
- Create: `tests/test_operational_compaction.py`
- Modify: `src/memo/operation_ledger_v2.py`
- Modify: `src/memo/operation_views.py`

**Interfaces:**
- Consumes: peer watermarks, active views, tombstone horizons, signed anchors.
- Produces: `CompactionPlanner.plan(now: datetime) -> CompactionPlan` and
  `OperationLedgerV2.compact_prefix(plan: CompactionPlan) ->
  CompactionReport`.

- [ ] **Step 1: Write failing retention/anchor tests**

```python
def test_compaction_keeps_active_refs_and_required_tombstones(compactor, state):
    state.add_acknowledged_delivery(sequence=10, tombstone_until="2026-08-01")
    state.add_active_handoff(sequence=11)
    plan = compactor.plan(now=datetime.fromisoformat("2026-07-29T00:00:00+00:00"))
    assert 10 not in plan.removable_sequences
    assert 11 not in plan.removable_sequences


def test_compacted_anchor_allows_incremental_peer_import(compactor, peer):
    plan = compactor.plan_after_all_peers_ack(sequence=100)
    report = compactor.apply(plan)
    assert report.anchor.final_sequence == 100
    assert report.anchor.state_sha256 == compactor.views.root_hash()
    assert sha256(report.checkpoint) == report.anchor.checkpoint_sha256
    imported = peer.ledger.import_bundles([report.remaining_bundle])
    assert imported.final_position.sequence >= 100
```

Also test expired presence removal, conflict/outcome policy references,
unacknowledged delivery retention, lagging peer blocker, invalid signature,
anchor regression, crash before/after anchor publication, and deterministic
rebuild from the exact retained checkpoint bytes plus remaining events. Add a
hard failure proving a state hash without its checkpoint can never authorize
segment deletion.

- [ ] **Step 2: Run failing compaction tests**

```bash
uv run --no-sync pytest \
  tests/test_operational_compaction.py tests/test_operation_ledger_v2.py -v
```

- [ ] **Step 3: Implement plan-first compaction**

The planner calculates a per-origin removable prefix only when every enrolled
peer watermark is beyond it, no active view references it, and domain retention
has elapsed. It retains bounded tombstones through maximum retry/replication
horizon and never applies generic cleanup to policy-referenced conflicts or
outcomes.

`compact_prefix` first writes/fsyncs canonical `StateCheckpoint` bytes
containing the complete reducer state at the prefix boundary, then writes a
signed anchor containing base/final sequence, prior/final event hashes,
checkpoint SHA-256/size, state root, reducer version, ledger epoch, timestamp,
and previous anchor hash. It rebuilds a fresh SQLite view only from that
checkpoint plus remaining events and proves the root. Sync publishes checkpoint
then anchor before any head that references them. Only after every enrolled
peer verifies those exact bytes may the origin atomically remove the proven
prefix segments; otherwise it retains them.

- [ ] **Step 4: Run compaction, ledger, and view gates**

```bash
uv run --no-sync pytest \
  tests/test_operational_compaction.py tests/test_operation_ledger_v2.py \
  tests/test_operation_views_v2.py tests/test_operational_delivery.py \
  tests/test_operational_presence.py -v
uv run --no-sync mypy \
  src/memo/operational_compaction.py src/memo/operation_ledger_v2.py
uv run --no-sync ruff check \
  src/memo/operational_compaction.py src/memo/operation_ledger_v2.py \
  src/memo/operation_views.py tests/test_operational_compaction.py
```

- [ ] **Step 5: Commit bounded compaction**

```bash
git add \
  src/memo/operational_compaction.py src/memo/operation_ledger_v2.py \
  src/memo/operation_views.py tests/test_operational_compaction.py
git commit -m "feat: compact operational history through signed anchors"
```

### Task 9: Add writer control, daemon lifecycle, and health

**Files:**
- Create: `src/memo/operational_control.py`
- Create: `src/memo/operational_health.py`
- Create: `src/memo/operational_daemon.py`
- Create: `tests/test_operational_control.py`
- Create: `tests/test_operational_health.py`
- Create: `tests/test_operational_daemon.py`
- Create: `tests/test_runtime_operational_install.py`
- Modify: `src/memo/daemon_common.py`
- Modify: `src/memo/runtime/install.py`
- Modify: `src/memo/runtime/report.py`
- Modify: `src/memo/flags_misc.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: Plan 01 `EpochFence`.
- Produces: `WriterLease`, `OperationalHealthSnapshot`,
  `memo-operational-daemon`.
- `OperationalHealth.check(*, now: datetime) -> OperationalHealthSnapshot`
- `OperationalWatchdog.run_once(*, now: datetime) -> WatchdogReport`

- [ ] **Step 1: Write failing lifecycle/fence tests**

```python
def test_only_one_daemon_holds_writer_lease(runtime):
    first = runtime.start()
    with pytest.raises(OperationalError, match="writer"):
        runtime.start_second()
    first.stop()


def test_shutdown_fences_drains_fsyncs_then_releases(runtime):
    runtime.start()
    runtime.enqueue_delivery("d1")
    report = runtime.shutdown(timeout=5)
    assert report.order == ["fence", "drain", "fsync", "health", "release"]
    assert report.pending == 0


def test_stale_epoch_is_rejected(fence):
    fence.activate(epoch=7)
    with pytest.raises(OperationalError, match="epoch"):
        fence.authorize(request_epoch=6, mutating=True)


def test_silent_peer_becomes_unhealthy_without_pending_traffic(runtime, clock):
    runtime.record_verified_peer_activity("device-b", at=clock.now())
    clock.advance(seconds=runtime.peer_unhealthy_seconds + 1)
    snapshot = runtime.health.check(now=clock.now())
    assert snapshot.verdict == "unhealthy"
    assert snapshot.peers[0].device_id == "device-b"
    assert snapshot.peers[0].verdict == "unhealthy"
```

Cover runtime/config/roster mismatch, domain failure isolation, bounded
shutdown failure, poll/heartbeat flags, LaunchAgent identity, and health fields
for the complete shared snapshot. Add wedged-listener self-probe, oldest-pending
threshold, outbox/rebuild failure, gap persistence, signal handling, one
bounded restart remediation, cooldown, and escalation without restart loops.

- [ ] **Step 2: Run failing runtime tests**

```bash
uv run --no-sync pytest \
  tests/test_operational_control.py tests/test_operational_health.py \
  tests/test_operational_daemon.py tests/test_runtime_operational_install.py -v
```

- [ ] **Step 3: Implement one operational coordinator**

Register `memo-operational-daemon = "memo.operational_daemon:main"`. Reuse
`daemon_common.serve_until_shutdown()`. Startup verifies package/runtime digest,
schema, roster, keys, epoch, and writer lease. The loop schedules ingest,
publish, terminal delivery, retries, presence/heartbeat, expiry, outbox, and
health independently, plus the admitted `OperationalSessionLoop`. Health
performs a real local authenticated status round-trip rather than checking only
the PID/port. Thresholds are exact config values: listener probe failure is
immediately unhealthy; a sync gap or pending age over two poll intervals is
degraded; five intervals, invalid epoch/roster/key, rebuild failure, or repeated
outbox failure is unhealthy. The watchdog emits a health event, attempts at
most one graceful restart per cooldown, then leaves the runtime fenced and
surfaces escalation. Shutdown follows the tested five-stage order.

Peer liveness is evaluated even when there are no messages, gaps, or pending
publishes. For every enrolled non-revoked peer, health records the most recent
verified signed head, ingest, publish, and wakeup timestamps. The freshest of
those must be newer than `peer_stale_seconds` for healthy; exceeding that is
degraded, and exceeding `peer_unhealthy_seconds` is unhealthy. Missing initial
activity after the enrollment grace period is unhealthy. These exact
thresholds are required config values and strict doctor verifies them.

- [ ] **Step 4: Run runtime and install gates**

```bash
uv run --no-sync pytest \
  tests/test_operational_control.py tests/test_operational_health.py \
  tests/test_operational_daemon.py tests/test_runtime_operational_install.py \
  tests/test_runtime_isolation.py -v
uv run --no-sync mypy \
  src/memo/operational_control.py src/memo/operational_health.py \
  src/memo/operational_daemon.py
uv run --no-sync ruff check \
  src/memo/operational_control.py src/memo/operational_health.py \
  src/memo/operational_daemon.py tests/test_operational_control.py \
  tests/test_operational_health.py tests/test_operational_daemon.py
```

- [ ] **Step 5: Commit coordinator**

```bash
git add \
  src/memo/operational_control.py src/memo/operational_health.py \
  src/memo/operational_daemon.py src/memo/daemon_common.py \
  src/memo/runtime/install.py src/memo/runtime/report.py \
  src/memo/flags_misc.py pyproject.toml \
  tests/test_operational_control.py tests/test_operational_health.py \
  tests/test_operational_daemon.py tests/test_runtime_operational_install.py
git commit -m "feat: add Memo operational coordinator"
```

### Task 10: Expose only native Memo APIs

**Files:**
- Create: `src/memo/memory/operational_ops.py`
- Create: `src/memo/server_coordination.py`
- Create: `src/memo/server_delivery.py`
- Create: `src/memo/server_presence.py`
- Create: `src/memo/server_continuity.py`
- Create: `src/memo/server_operational_runtime.py`
- Create: `src/memo/cli_coordination.py`
- Create: `src/memo/cli_presence.py`
- Create: `src/memo/cli_operational_runtime.py`
- Create: `tests/test_server_coordination.py`
- Create: `tests/test_server_delivery.py`
- Create: `tests/test_server_presence.py`
- Create: `tests/test_server_continuity.py`
- Create: `tests/test_server_operational_runtime.py`
- Create: `tests/test_cli_operational_runtime.py`
- Modify: `src/memo/memory/facade.py`
- Modify: `src/memo/server.py`
- Modify: `src/memo/cli.py`
- Modify: `src/memo/cli_doctor.py`

**Interfaces:**
- Produces the exact signed admitted surface set. Representative admitted
  names include
  `memo_message_send`, `memo_channel_messages`,
  `memo_handoff_create`, `memo_handoff_consume`, `memo_task_create`,
  `memo_task_complete`, `memo_delivery_ack`, `memo_delivery_status`,
  `memo_presence_announce`, `memo_presence_active`,
  `memo_presence_conflicts`, `memo_session_checkpoint`,
  `memo_session_recover`, `memo_continuity`,
  `memo_operational_sync_status`, and `memo_operational_health`.
  Conditional broadcast, supersede, topic termination, unread, handoff/task
  status, task assignee/result/cancel/expiry, active-workspace, fast-lane, and
  peer-notification surfaces are registered iff their operation-map row is
  `absorb`; signed `delete` rows assert their absence.

- [ ] **Step 1: Write failing surface and identity tests**

```python
def test_operational_tools_derive_identity_and_require_epoch(mcp_client):
    out = mcp_client.call(
        "memo_message_send",
        {"channel": "handoff", "body": "resume", "idempotency_key": "m1"},
        headers={"Memo-Epoch": "7"},
    )
    assert out["actor_id"] == mcp_client.authenticated_actor
    with pytest.raises(McpError) as exc:
        mcp_client.call(
            "memo_message_send",
            {"channel": "handoff", "body": "resume", "actor_id": "spoof"},
        )
    assert exc.value.data["code"] in {"identity_rejected", "stale_epoch"}


def test_no_flow_tools_or_aliases(server):
    names = {tool.name for tool in server.list_tools()}
    assert not {name for name in names if name.startswith("flow_")}


def test_public_surfaces_equal_signed_operation_map(server, cli_runner, manifest):
    expected_mcp = manifest.admitted_mcp_names()
    expected_cli = manifest.admitted_cli_paths()
    assert {tool.name for tool in server.list_operational_tools()} == expected_mcp
    assert enumerate_operational_cli_paths(cli_runner) == expected_cli
```

Also verify correct read/write annotations, identical CLI/MCP error codes,
required idempotency keys, profile registration, and health/continuity
read-only behavior.

- [ ] **Step 2: Run failing server/CLI tests**

```bash
uv run --no-sync pytest \
  tests/test_server_coordination.py tests/test_server_delivery.py \
  tests/test_server_presence.py tests/test_server_continuity.py \
  tests/test_server_operational_runtime.py \
  tests/test_cli_operational_runtime.py -v
```

- [ ] **Step 3: Implement mixin and domain registration**

Each `server_<domain>.py` exposes `register(server, memory)`. `server.py`
installs epoch/error middleware before `McpWriteCoordinator`, then registers
the five modules. `Memory` composes `_OperationalOpsMixin`; CLI wiring remains
in `cli.py` and implementations in the new domain modules. Extend strict doctor
to verify exact package/runtime digest, operational daemon, roster, config, and
epoch. Registration is generated from and verified against the frozen signed
operation map; no method is inferred from source naming.

- [ ] **Step 4: Run surface and doctor gates**

```bash
uv run --no-sync pytest \
  tests/test_server_coordination.py tests/test_server_delivery.py \
  tests/test_server_presence.py tests/test_server_continuity.py \
  tests/test_server_operational_runtime.py \
  tests/test_cli_operational_runtime.py \
  tests/test_cli_mcp_surface_smoke.py tests/test_surface_profiles.py \
  tests/test_runtime_isolation.py -v
uv run --no-sync mypy src/memo
uv run --no-sync ruff check \
  src/memo/memory/operational_ops.py src/memo/server_coordination.py \
  src/memo/server_delivery.py src/memo/server_presence.py \
  src/memo/server_continuity.py src/memo/server_operational_runtime.py \
  src/memo/cli_coordination.py src/memo/cli_presence.py \
  src/memo/cli_operational_runtime.py
```

- [ ] **Step 5: Commit Memo interfaces**

```bash
git add \
  src/memo/memory/operational_ops.py src/memo/memory/facade.py \
  src/memo/server_coordination.py src/memo/server_delivery.py \
  src/memo/server_presence.py src/memo/server_continuity.py \
  src/memo/server_operational_runtime.py src/memo/server.py \
  src/memo/cli_coordination.py src/memo/cli_presence.py \
  src/memo/cli_operational_runtime.py src/memo/cli.py src/memo/cli_doctor.py \
  tests/test_server_coordination.py tests/test_server_delivery.py \
  tests/test_server_presence.py tests/test_server_continuity.py \
  tests/test_server_operational_runtime.py \
  tests/test_cli_operational_runtime.py
git commit -m "feat: expose native Memo coordination"
```

### Task 11: Prove the distributed runtime end to end

**Files:**
- Create: `tests/test_operational_runtime_integration.py`
- Modify: `tests/conftest.py` only for isolated two-peer fixtures.

**Interfaces:**
- Consumes: all Plan 02 services.
- Produces: end-to-end proof and measured latency baseline.

- [ ] **Step 1: Write the failing two-peer integration**

```python
@pytest.mark.slow
def test_send_sync_present_ack_round_trip(two_peer_runtime):
    a, b = two_peer_runtime
    sent = a.memory.message_send(
        channel="handoff", body="resume", target_ids=(b.actor_id,),
        idempotency_key="roundtrip-1",
    )
    a.sync.publish()
    b.sync.ingest()
    receipt = b.daemon.deliver_once()
    assert receipt.event_id == sent.event_id
    b.memory.delivery_ack(
        message_id=sent.message_id, idempotency_key="roundtrip-ack"
    )
    b.sync.publish()
    a.sync.ingest()
    assert a.memory.delivery_status(sent.message_id).state == "acknowledged"
```

Add duplicate segment, disconnect/reconnect, sequence gap/recovery, crash
before/after terminal presentation, daemon restart, view rebuild, expiry, and
continuity degraded-state scenarios.

- [ ] **Step 2: Run the new slow test and confirm missing fixture/wiring**

```bash
uv run --no-sync pytest \
  tests/test_operational_runtime_integration.py -v
```

- [ ] **Step 3: Implement hermetic two-peer fixtures**

Each peer gets a separate temp state root, operational remote clone, signing
key, roster entry, Unix terminal socket, injected fake presenter, and daemon
clock. No fixture may read a real config, environment route, TTY, or Git remote.
Record round-trip and recovery latency into the test report for later
comparison with the frozen Memflow baseline. The test loads
`slo-baseline.json` and makes hard assertions for sample count, p50/p95/p99/max,
visibility, recovery, error rate, and zero loss/duplicates.

- [ ] **Step 4: Run final Plan 02 gates**

```bash
uv run --no-sync ruff check src/memo tests
uv run --no-sync mypy src/memo
uv run --no-sync pytest \
  tests/test_operational_coordination.py \
  tests/test_operational_delivery.py \
  tests/test_operational_presence.py \
  tests/test_operational_continuity.py \
  tests/test_terminal_bridge.py \
  tests/test_operational_sync.py \
  tests/test_operational_daemon.py \
  tests/test_server_coordination.py \
  tests/test_server_delivery.py \
  tests/test_server_presence.py \
  tests/test_server_continuity.py \
  tests/test_server_operational_runtime.py -v
uv run --no-sync pytest -m "not slow" -n auto --timeout=120
uv run --no-sync pytest -m "slow" --timeout=300 -v
```

Expected: all pass; every measured SLO stays within the exact frozen tolerance.

- [ ] **Step 5: Commit distributed proof**

```bash
git add tests/test_operational_runtime_integration.py tests/conftest.py
git commit -m "test: prove distributed Memo coordination"
```

## Plan Acceptance Gate

- The selected Memflow contract tests have a recorded baseline and sanitized
  parity fixtures.
- Every signed admitted source operation has exactly one Memo method/MCP/CLI
  mapping and parity oracle; every omitted operation has a signed `delete`
  disposition. SLO assertions pass against the frozen workload thresholds.
- Memo-to-Memo send, sync, delivery, terminal presentation, ACK, presence,
  recovery, and continuity work without any Memflow import or data path.
- Reducers are monotonic and retries do not duplicate messages, ACKs, cursors,
  leases, checkpoints, or terminal presentation.
- Operational sync uses enrolled signatures, per-origin ownership, gap recovery,
  and a dedicated remote.
- One operational daemon owns the writer lease and drains cleanly.
- Public surfaces contain only `memo_*` names and derive identity from
  authenticated context.
- Full focused, non-slow, slow, mypy, and ruff gates pass.
- No cutover, consumer mutation, Synapse migration, or Memflow deletion occurs
  in this plan.
