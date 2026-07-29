# Memflow Active-State Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an idempotent, fully rehearsed migration that preserves
Memo's existing operational truth, imports valid durable knowledge missing from
Memo, and translates only Memflow state active at the frozen cutover boundary.

**Architecture:** Operator-only scanners read immutable Memo/Memflow snapshots
and create a deterministic `MigrationPlan`. Applying a plan writes only to an
isolated staging root through Memo's v2 commands and normal durable write
policy. A signed rollback bundle preserves the exact pre-epoch runtime/config
state, while verification compares source proofs, active roots, and parity
contracts without retaining expired operational payloads.

**Tech Stack:** Python 3.13+, Plans 01–03 contracts, Memo write policy and v2
ledger, stdlib JSON/hash/tar/git subprocess, pytest, mypy, ruff. No installed
public importer.

## Global Constraints

- Plans 01, 02, and 03 must pass before this plan starts.
- Inputs are immutable snapshots tied to one attempt ID, capability-manifest
  hash, origin-sequence vector, and remote ref.
- Existing Memo operational v2 state is the base. Memflow never overwrites Memo
  focus, trust, conflict, outcome, or write-policy state by recency.
- Durable import includes only valid knowledge not already represented in Memo.
- Operational import includes only pending/unacknowledged delivery state, open
  tasks, active channel cursors, recoverable sessions, nonexpired leases, and
  runtime metadata needed for safe resumption.
- TTL is evaluated against the frozen fence timestamp, not execution time.
- Expired presence, acknowledged history beyond required tombstones, closed
  channels, completed tasks, terminated sessions, and old heartbeat/delivery
  events are excluded.
- Every translated item carries stable `source_system`, `source_event_id`,
  `source_hash`, and a distinct migration origin.
- Re-running the same plan is a verifiable no-op. Reusing a source ID with
  different content is an integrity failure.
- All apply/rehearsal work occurs under a validated staging root with fake TTY,
  disposable remote, and cloned vault.
- Rollback bundles are mode `0600`, never uploaded, and usable only before the
  committed activation epoch.
- No production process, service, config, remote, vault, or `.memflow` path is
  mutated by this plan.
- This plan implements and rehearses scanners with a signed `FinalFenceProof`
  created only inside immutable/disposable fixtures. It never labels a live
  pre-quiesce snapshot final. Plan 05 is the sole production producer and must
  rerun scan/plan/apply against its post-zero-drain proof.

---

## File Structure

### Create in Memo

- `tools/memflow_absorption/importer.py`
- `tools/memflow_absorption/source_memo.py`
- `tools/memflow_absorption/source_memflow.py`
- `tools/memflow_absorption/durable_import.py`
- `tools/memflow_absorption/active_import.py`
- `tools/memflow_absorption/rollback_bundle.py`
- `tools/memflow_absorption/verify.py`
- `tools/memflow_absorption/rehearsal.py`
- `tests/tools/test_absorption_source_memo.py`
- `tests/tools/test_absorption_source_memflow.py`
- `tests/tools/test_absorption_durable_import.py`
- `tests/tools/test_absorption_active_import.py`
- `tests/tools/test_absorption_rollback_bundle.py`
- `tests/tools/test_absorption_verify.py`
- `tests/tools/test_absorption_rehearsal.py`

### Extend

- `tools/memflow_absorption/schemas.py`
- `tools/memflow_absorption/safety.py`
- `tools/memflow_absorption/snapshot.py`
- `tools/memflow_absorption/__main__.py`
- `tests/fixtures/memflow_absorption/`

## Shared Interfaces

`OriginVector` is an immutable mapping from origin device ID to final included
sequence. `MemoSourceProof` contains the verified v1 genesis anchors keyed by
origin, v2 origin positions, operational state root, durable-vault manifest
hash, and policy-relevant IDs. `DurableSourceProof` contains only allowlisted
durable schemas/files, candidate rows, evidence/supersession state, source
event IDs, and file/row hashes. `MemflowSourceProof` is derived from one signed
Plan 03 `FinalFenceProof` and contains its frozen timestamp, origin vector,
immutable remote commit OID, source manifest hash, active items, exclusions by
reason, and per-origin source hashes.

`ActiveStateItem.kind` is closed to `channel`, `message`, `handoff`,
`delivery_pending`, `delivery_delivered_unacked`, `ack_tombstone`, `cursor`,
`task`, `session`, `presence_lease`, `heartbeat_lease`, and
`runtime_metadata`. Every item inside the final signed vector must map to a
registry event/view or have a signed proof that it is non-live; unknown or
unparseable in-vector state is a blocker, never a quarantine-only exclusion.

`MigrationPlan` contains `attempt_id`, source proof hashes, capability-manifest
hash, the exact signed Plan 01 `MigrationOrigin` plus its SHA-256, target Memo
commit/runtime/schema, frozen time, canonical durable rows, canonical
operational commands, expected counts/root, and plan SHA-256.
`ImportReceipt` contains plan hash, target root, inserted/replayed/rejected
counts, `migration_origin_sha256`, durable result IDs, operational event IDs,
final positions/root, and receipt signature.

Exact function signatures:

- `scan_memo_source(snapshot: Path) -> MemoSourceProof`
- `scan_memflow_durable(snapshot: Path, *, fence: FinalFenceProof) ->
  DurableSourceProof`
- `scan_memflow_active(snapshot: Path, *, fence: FinalFenceProof) ->
  MemflowSourceProof`
- `plan_import(memo: MemoSourceProof, durable: DurableSourceProof,
  memflow: MemflowSourceProof,
  capability_manifest: CapabilityManifest) -> MigrationPlan`
- `apply_to(plan: MigrationPlan, staging_root: Path) -> ImportReceipt`
- `install_inactive_generation(receipt: ImportReceipt, attempt_root: Path,
  device_id: str) -> InactiveGenerationReceipt`
- `verify_import(plan: MigrationPlan, staging_root: Path) ->
  MigrationVerificationReport`

`MigrationPlan` is canonical JSON and includes source manifest hashes, frozen
time, origin vector, target Memo commit/runtime/schema, durable writes,
operational commands, excluded-item counts by reason, and expected roots.

### Task 1: Scan and prove the three migration inputs

**Files:**
- Create: `tools/memflow_absorption/source_memo.py`
- Create: `tools/memflow_absorption/source_memflow.py`
- Extend: `tools/memflow_absorption/schemas.py`
- Create: `tests/tools/test_absorption_source_memo.py`
- Create: `tests/tools/test_absorption_source_memflow.py`

**Interfaces:**
- Produces: `MemoSourceProof`, `MemflowSourceProof`, `OriginVector`,
  `DurableSourceProof`, `ActiveStateItem`.

- [ ] **Step 1: Write failing source-proof tests**

```python
def test_memo_scan_preserves_v2_base_and_v1_genesis(memo_snapshot):
    proof = scan_memo_source(memo_snapshot)
    assert proof.v1_genesis_anchors
    assert proof.v2_positions == {"device-a": 42}
    assert proof.state_domains >= {
        "focus", "handoffs", "attention", "conflicts", "outcomes", "sessions"
    }
    assert proof.verify().ok is True


def test_memflow_scan_is_bound_to_final_fence_proof(memflow_snapshot, final_fence):
    proof = scan_memflow_active(
        memflow_snapshot,
        fence=final_fence,
    )
    assert {item.kind for item in proof.active_items} == {
        "channel", "message", "handoff", "delivery_pending",
        "delivery_delivered_unacked", "ack_tombstone", "cursor", "task",
        "session", "presence_lease", "heartbeat_lease", "runtime_metadata",
    }
    assert proof.excluded["expired_presence"] == 2
    assert max(proof.sequences["device-a"]) <= 100
```

Also test chain/remote-OID mismatch, missing origin, event beyond vector,
unknown schema blocker, durable candidate schema/supersession/evidence,
source file changed after snapshot, and no reliance on caller time/vector or
current wall clock.

- [ ] **Step 2: Run tests and confirm scanners are missing**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_source_memo.py \
  tests/tools/test_absorption_source_memflow.py -v
```

- [ ] **Step 3: Implement read-only scanners**

Memo scanning calls Plan 01 verification and captures genesis anchor, positions,
state root, durable vault manifest, and policy-relevant domain IDs. Memflow
scanning first verifies the signed `FinalFenceProof`, then reads frozen
channel/event/session/delivery/presence and allowlisted durable files directly
through snapshot paths. It derives time, vector, immutable remote OID, and
snapshot hashes only from that proof, then emits a typed active item, durable
candidate, or proven exclusion. Do not import Memflow modules or call its MCP
server.

Use stable active IDs:

```python
def migration_key(item: ActiveStateItem) -> str:
    return (
        f"memflow-active/{item.origin_device}/"
        f"{item.source_sequence}/{item.source_event_id}"
    )
```

- [ ] **Step 4: Run scanner gates**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_source_memo.py \
  tests/tools/test_absorption_source_memflow.py \
  tests/tools/test_absorption_snapshot.py -v
uv run --no-sync mypy tools/memflow_absorption
uv run --no-sync ruff check \
  tools/memflow_absorption/source_memo.py \
  tools/memflow_absorption/source_memflow.py \
  tests/tools/test_absorption_source_memo.py \
  tests/tools/test_absorption_source_memflow.py
```

- [ ] **Step 5: Commit source proofs**

```bash
git add \
  tools/memflow_absorption/source_memo.py \
  tools/memflow_absorption/source_memflow.py \
  tools/memflow_absorption/schemas.py \
  tests/tools/test_absorption_source_memo.py \
  tests/tools/test_absorption_source_memflow.py
git commit -m "feat: prove Memflow migration inputs"
```

### Task 2: Plan missing durable-knowledge import through Memo policy

**Files:**
- Create: `tools/memflow_absorption/durable_import.py`
- Create: `tests/tools/test_absorption_durable_import.py`

**Interfaces:**
- Consumes: `MemoSourceProof`, `DurableSourceProof`, native
  `Memory.write_policy.preflight`, and normal `Memory.save_operation`.
- Produces: `DurableImportPlan` with `create`, `corroborate`, `reject`, and
  `already_present` rows.
- `preview_memo_write(memory: Memory, row: DurableCandidate) ->
  WritePolicyDecision` calls the existing `WritePolicyEngine.preflight` with
  exact `title`, `content`, `tags`, `extra`, `actor`,
  `allow_conflict_override=False`, and empty `override_reason`; it never calls
  `enforce` or writes. No new `Memory.preview_write` API is invented.

- [ ] **Step 1: Write failing dedup/policy tests**

```python
def test_durable_plan_corroborates_duplicate_and_creates_only_missing(
    memo_source, durable_source_proof, memory_stub
):
    plan = plan_durable_import(
        memo_source, durable_source_proof, memory=memory_stub
    )
    assert plan.counts == {
        "create": 1, "corroborate": 1, "reject": 1, "already_present": 1
    }
    assert all(row.operation_key.startswith("memflow-durable/") for row in plan.rows)


def test_rejected_operational_log_is_not_promoted(memory_stub):
    candidate = durable_candidate(type="log", body="heartbeat delivered")
    plan = plan_durable_import(
        empty_memo(), durable_proof([candidate]), memory=memory_stub
    )
    assert plan.rows[0].disposition == "reject"
    assert plan.rows[0].reason == "not_durable_knowledge"
```

Cover exact content duplicate, semantic/topic duplicate, superseded/forgotten
candidate, invalid evidence, conflicting fact, repeated plan, and stable write
request/timestamp.

- [ ] **Step 2: Run durable-plan tests**

```bash
uv run --no-sync pytest tests/tools/test_absorption_durable_import.py -v
```

- [ ] **Step 3: Implement deterministic policy planning**

Normalize each valid fact/decision/preference/procedure, compute source and
content hashes, query Memo by durable identity/topic, then call the existing
side-effect-free write preview. Freeze the exact save request:

```python
@dataclass(frozen=True)
class DurableWritePlanRow:
    operation_key: str
    disposition: Literal["create", "corroborate", "reject", "already_present"]
    source_event_ids: tuple[str, ...]
    topic_key: str
    content_sha256: str
    save_kwargs: Mapping[str, object]
    reason: str
```

Applying later must pass the same request through normal Memo write policy and
durable idempotency; the planner never writes.

- [ ] **Step 4: Run durable policy gates**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_durable_import.py \
  tests/test_operational_memory.py tests/test_write_freeze_gc.py \
  tests/test_operational_idempotency.py \
  tests/test_durable_outbox.py -v
uv run --no-sync mypy tools/memflow_absorption/durable_import.py
uv run --no-sync ruff check \
  tools/memflow_absorption/durable_import.py \
  tests/tools/test_absorption_durable_import.py
```

- [ ] **Step 5: Commit durable planning**

```bash
git add \
  tools/memflow_absorption/durable_import.py \
  tests/tools/test_absorption_durable_import.py
git commit -m "feat: plan missing durable knowledge import"
```

### Task 3: Build deterministic active-state translation and staging apply

**Files:**
- Create: `tools/memflow_absorption/active_import.py`
- Create: `tools/memflow_absorption/importer.py`
- Create: `tests/tools/test_absorption_active_import.py`
- Extend: `tools/memflow_absorption/__main__.py`

**Interfaces:**
- Consumes: source proofs, capability manifest, Plan 01/02 v2 APIs.
- Produces: `MigrationPlan`, `ImportReceipt`, `apply_to`.

- [ ] **Step 1: Write failing active translation/idempotency tests**

```python
def test_active_plan_translates_only_admitted_live_state(source_proofs, manifest):
    plan = plan_import(*source_proofs, capability_manifest=manifest)
    assert {c.event_type for c in plan.operational_commands} >= {
        "coord.channel.opened.v1",
        "coord.message.sent.v1",
        "coord.handoff.created.v1",
        "delivery.presented.v1",
        "delivery.acknowledged.v1",
        "delivery.cursor.advanced.v1",
        "coord.task.created.v1",
        "session.recoverable.v1",
        "presence.lease.announced.v1",
        "presence.lease.renewed.v1",
        "runtime.metadata.recorded.v1",
    }
    assert "presence.lease.expired.v1" not in {
        c.event_type for c in plan.operational_commands
    }


def test_apply_is_noop_on_repeat_and_uses_frozen_ttl(plan, staging_root):
    first = apply_to(plan, staging_root)
    second = apply_to(plan, staging_root)
    assert first.inserted > 0
    assert second.inserted == 0
    assert second.replayed == first.inserted
    assert current_clock_changes_do_not_change_root(plan, staging_root)
```

Cover duplicate source ID/different hash, target identity mapping, ACK
tombstone, cursor monotonicity, session recoverability, active lease remaining
TTL, channel/message/pending-delivery coverage, runtime metadata, Memo base
precedence, unknown-item blocking, capability disposition enforcement, and
attempt IDs containing `/`, `\\`, `..`, Unicode confusables, or traversal.

- [ ] **Step 2: Run importer tests and observe missing implementation**

```bash
uv run --no-sync pytest tests/tools/test_absorption_active_import.py -v
```

- [ ] **Step 3: Implement plan/apply with a distinct migration origin**

Canonicalize and hash `MigrationPlan`. Translate each item into an
`OperationalCommand` whose source proof contains original origin, sequence,
schema, actor, subject URI, content/event hashes, and event ID. Create a signed,
roster-enrolled Plan 01 `MigrationOrigin` for
`device_id=f"migration-{safe_attempt_id(plan.attempt_id)}"`; the canonical
`safe_attempt_id` permits only lowercase ASCII letters, digits, and hyphens,
rejects separators, dot segments, traversal, empty values, and values over 64
bytes. Pass it only through `CommitContext`, never by setting an undeclared
command field. Preserve the
source origin only in `SourceProof`. Import event names and payload validators
from `memo.operational_event_types`; evaluate `expires_at` from the signed final
fence timestamp.

`apply_to` verifies the staging sentinel, plan hash, target Memo commit/runtime,
empty or matching previous receipt, then:

1. opens a cloned Memo vault and v2 staging ledger;
2. verifies/catches up the existing Memo base;
3. applies frozen durable writes;
4. commits operational commands in canonical order;
5. rebuilds views; and
6. writes/fsyncs an `ImportReceipt`.

`apply_to` is staging-only and never installs a peer generation. After
`verify_import` succeeds, the orchestrator performs this explicit, separately
receipted loop:

```python
for peer in verified_peers:
    install_inactive_generation(
        receipt,
        attempt_root=peer.attempt_root,
        device_id=peer.device_id,
    )
```

Each inactive receipt binds generation ID, device ID, plan/import hashes,
ledger/vault roots, runtime/config digests, fsync proof, and target relative
path. No active Memo pointer changes in this plan. Plan 05 may activate only
these exact signed generation digests, after applying and re-verifying the
post-drain final delta.

- [ ] **Step 4: Run active importer gates**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_active_import.py \
  tests/test_operational_coordination.py \
  tests/test_operational_delivery.py \
  tests/test_operational_presence.py \
  tests/test_operational_sessions_v2.py -v
uv run --no-sync mypy \
  tools/memflow_absorption/active_import.py \
  tools/memflow_absorption/importer.py
uv run --no-sync ruff check \
  tools/memflow_absorption/active_import.py \
  tools/memflow_absorption/importer.py \
  tests/tools/test_absorption_active_import.py
```

- [ ] **Step 5: Commit importer**

```bash
git add \
  tools/memflow_absorption/active_import.py \
  tools/memflow_absorption/importer.py \
  tools/memflow_absorption/__main__.py \
  tests/tools/test_absorption_active_import.py
git commit -m "feat: import active Memflow state idempotently"
```

### Task 4: Create and prove a pre-epoch rollback bundle

**Files:**
- Create: `tools/memflow_absorption/rollback_bundle.py`
- Create: `tests/tools/test_absorption_rollback_bundle.py`

**Interfaces:**
- Produces: `create_rollback_bundle`, `verify_rollback_bundle`,
  `restore_rollback_bundle` restricted to noncommitted attempts.
- `restore_rollback_bundle(bundle: EncryptedRollbackBundle, target:
  ValidatedRestoreTarget, *, control: VerifiedControlRecord,
  expected_control_oid: str) -> RollbackRestoreReceipt` re-fetches and verifies
  the exact control OID immediately before its first mutation.

- [ ] **Step 1: Write failing bundle safety/round-trip tests**

```python
def test_bundle_is_private_signed_and_complete(attempt_fixture):
    receipt = create_rollback_bundle(attempt_fixture)
    assert stat.S_IMODE(receipt.path.stat().st_mode) == 0o600
    assert receipt.encrypted is True
    verified = verify_rollback_bundle(
        receipt.path, roster=attempt_fixture.roster,
        recipient=attempt_fixture.device_recipient,
    )
    assert verified.components >= {
        "runtime", "plists", "launchd_state", "consumer_configs",
        "memflow_git_bundle", "local_state", "cursors", "roster", "epochs"
    }


def test_restore_refuses_after_committed_epoch(bundle, target, control_ref):
    committed = control_ref.install(state=CutoverState.EPOCH_COMMITTED)
    with pytest.raises(SafetyError, match="committed"):
        restore_rollback_bundle(
            bundle, target, control=verify(committed),
            expected_control_oid=committed.oid,
        )
```

Also test traversal member rejection, symlink, wrong signature, wrong attempt,
wrong manifest hash, corrupted component, mode preservation, Git refs, and
round-trip into a temporary target only. Also test stale/forged local state,
control OID changing between verification and mutation, wrong recipient,
missing Keychain key, cross-recipient unwrap refusal, and private wrapping
key/bundle destruction after VERIFIED.

- [ ] **Step 2: Run rollback tests**

```bash
uv run --no-sync pytest tests/tools/test_absorption_rollback_bundle.py -v
```

- [ ] **Step 3: Implement exact bundle contents and restore guard**

The bundle manifest records exact Memo/Memflow/Synapse commits, runtime
digests/wheels, plist bytes and launchd loaded/disabled state, consumer config
bytes/modes, a Memflow `git bundle` of required refs, required unversioned state,
cursors/caches, origin vector, remote ref, roster, fence marker, and epochs.
Archive paths are relative allowlisted names; extraction never trusts archive
paths. Omit secrets that can be reacquired from Keychain. Encrypt content with
AES-256-GCM using a random per-attempt data key. Generate an ephemeral X25519
wrapping keypair per attempt, keep its private key as a non-exportable Keychain
secret, and derive a distinct key-encryption key for each enrolled recipient
with X25519 plus HKDF-SHA256 over attempt/device/manifest context. Wrap the data
key for that recipient with AES-GCM and a unique nonce. Ed25519 remains signing
only and never performs encryption. Sign the manifest plus ciphertext and
wrapped-key digests with Plan 01 contracts. Never upload it. The signed
VERIFIED transition destroys the data key, ephemeral private wrapping key,
every wrapped key, and bundle, fsyncs the containing directory, and records
only hashes/destruction receipts.

- [ ] **Step 4: Run safety and bundle gates**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_rollback_bundle.py \
  tests/tools/test_absorption_safety.py -v
uv run --no-sync mypy tools/memflow_absorption/rollback_bundle.py
uv run --no-sync ruff check \
  tools/memflow_absorption/rollback_bundle.py \
  tests/tools/test_absorption_rollback_bundle.py
```

- [ ] **Step 5: Commit rollback tooling**

```bash
git add \
  tools/memflow_absorption/rollback_bundle.py \
  tests/tools/test_absorption_rollback_bundle.py
git commit -m "feat: build verified pre-epoch rollback bundles"
```

### Task 5: Verify and rehearse migration across two isolated peers

**Files:**
- Create: `tools/memflow_absorption/verify.py`
- Create: `tools/memflow_absorption/rehearsal.py`
- Create: `tests/tools/test_absorption_verify.py`
- Create: `tests/tools/test_absorption_rehearsal.py`
- Extend: `tools/memflow_absorption/__main__.py`

**Interfaces:**
- Produces: signed `MigrationVerificationReport`,
  `RehearsalReport`, and `ACTIVATION_READY` candidate evidence.

- [ ] **Step 1: Write failing parity/rehearsal tests**

```python
def test_verify_matches_active_identity_lifecycle_and_root(plan, staged_peer):
    report = verify_import(plan, staged_peer.root)
    assert report.equal is True
    assert report.counts["pending_handoffs"] == plan.expected_counts["pending_handoffs"]
    assert report.duplicates == 0
    assert report.active_root == plan.expected_active_root


def test_two_peer_rehearsal_survives_duplicate_gap_and_restart(rehearsal):
    report = rehearsal.run(
        faults=("duplicate_segment", "sequence_gap", "daemon_restart")
    )
    assert report.peer_roots["device-a"] == report.peer_roots["device-b"]
    assert report.lost_acks == 0
    assert report.duplicate_deliveries == 0
    assert report.production_paths_touched == ()
```

Cover durable write parity, source-proof coverage, excluded counts, TTL,
terminal fake presenter, sessions, locks, disconnect, view rebuild, crash, and
repeat rehearsal.

- [ ] **Step 2: Run verification/rehearsal tests**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_verify.py \
  tests/tools/test_absorption_rehearsal.py -v
```

- [ ] **Step 3: Implement signed verification and hermetic rehearsal**

Verification compares every planned source item to a durable record,
operational event/view, or explicit exclusion reason; checks active counts,
identities, lifecycle, origin positions, roots, duplicates, and policy results;
then signs the report. Rehearsal creates two independent temp state roots,
disposable operational remote, cloned vaults, fake terminals, isolated sockets,
and injected clocks. It must prove none resolve the production Memo/Memflow
state, port 18766, TTYs, or user config.

- [ ] **Step 4: Run final migration plan gates**

```bash
uv run --no-sync ruff check \
  tools/memflow_absorption tests/tools/test_absorption_*.py
uv run --no-sync mypy tools/memflow_absorption
uv run --no-sync pytest tests/tools/test_absorption_*.py -v
uv run --no-sync pytest \
  tests/test_operation_migration_v2.py \
  tests/test_operational_runtime_integration.py -v
```

Expected: two peer roots match, repeat apply inserts zero, no production path is
touched, and every active source item is accounted for.

- [ ] **Step 5: Commit migration proof**

```bash
git add \
  tools/memflow_absorption/verify.py \
  tools/memflow_absorption/rehearsal.py \
  tools/memflow_absorption/__main__.py \
  tests/tools/test_absorption_verify.py \
  tests/tools/test_absorption_rehearsal.py
git commit -m "test: prove Memflow active-state migration"
```

## Plan Acceptance Gate

- All three inputs—Memo operational truth, missing durable knowledge, and
  Memflow active state—have verified immutable source proofs.
- Durable import creates/corroborates only valid missing knowledge through Memo
  write policy.
- Every active Memflow item maps to one v2 record/view or an explicit exclusion;
  expired/closed history is absent, and unknown/unparseable in-vector state
  blocks readiness.
- Repeated plan/apply/verify operations are no-ops with identical roots.
- Rollback bundle is complete, private, signed, restorable in a temp target, and
  encrypted, unusable after a committed epoch, and scheduled for key/bundle
  destruction after VERIFIED.
- Both Macs hold fsynced inactive Memo generations with identical signed
  ledger/vault/config/runtime digests; no active pointer has changed.
- Two isolated peers converge under duplicate, gap, disconnect, crash, restart,
  and rebuild injection.
- No production state, config, TTY, service, or remote is read as a write target
  or modified.
