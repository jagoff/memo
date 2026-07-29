# Memo Operational Ledger v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve Memo's existing hash-chained operational journal into a signed,
compactable v2 authority with transactional SQLite views, durable idempotency,
and one canonical live-session model, without changing any v1 bytes.

**Architecture:** The current v1 reader is frozen as
`LegacyOperationLedger`. A new per-origin v2 ledger starts from a verified
genesis anchor, while SQLite stores rebuildable views and an outbox. A facade
preserves current `OperationalStore` behavior until replay parity passes, then
switches authority without production dual-write.

**Tech Stack:** Python 3.13+, stdlib `sqlite3`, JSONL, SHA-256, existing Memo
atomic I/O and locking, pytest, mypy, ruff. No new third-party dependency.

## Global Constraints

- `ActorIdentity`, `MemoEvent`, and their canonical v1 encoding are immutable.
- Existing `<state_dir>/journal/` and `operational-state.json` remain
  byte-identical until v2 activation succeeds.
- Markdown remains the durable-memory source of truth; SQLite is derived
  operational state only.
- There is no production v1/v2 dual-write. Dual-reduce is allowed only in
  isolated tests and rehearsal.
- The ledger is authoritative; `operational.db` must be rebuildable from
  verified anchors and v2 events.
- Tests use `tmp_cfg` or explicit temporary paths and never touch the real
  vault, journal, session state, or default state directory.
- Behavioral flags are registered through Memo's flag registry, never read with
  raw `os.environ.get`.
- Preserve Python `>=3.13`, deferred MLX imports, and the existing
  `Memory.operational` public attribute.
- Follow CI order: `ruff -> mypy -> focused pytest -> non-slow pytest`.
- Stage only explicit paths and commit after every task.
- This plan excludes channels, delivery, presence, live sync, daemon work,
  Memflow import, and multi-Mac cutover.

---

## File Structure

### Create

- `src/memo/operational_event.py` — v2 identities, commands, events, source
  proofs, anchors, bundles, receipts, and canonical hashes.
- `src/memo/operational_event_types.py` — one versioned event-name/payload
  registry consumed by reducers, public APIs, parity fixtures, and migrations.
- `src/memo/operational_signing.py` — domain-separated Ed25519 signing and
  verification over canonical operational records.
- `src/memo/operational_key_store.py` — Keychain-backed private-key provider
  plus an in-memory test provider; private keys never enter journal/config.
- `src/memo/operational_roster.py` — signed bootstrap verification roster and
  immutable public-key history.
- `src/memo/operational_epoch.py` — durable authority epoch/control marker and
  request-level fence used by every mutation path.
- `src/memo/operation_ledger_v1.py` — frozen v1 reader/validator exported as
  `LegacyOperationLedger`.
- `src/memo/operation_ledger_v2.py` — v2 append, verification, bundles,
  quarantine, anchors, and positions.
- `src/memo/operation_view_schema.py` — `operational.db` DDL and schema version.
- `src/memo/operation_views.py` — transactional reducers, catch-up, rebuild, and
  canonical state digest.
- `src/memo/operation_migration.py` — deterministic v1 verification, genesis,
  seed-event generation, parity, and activation stamp.
- `src/memo/durable_outbox.py` — exactly-once durable promotion reconciler.
- `src/memo/operational_sessions.py` — canonical operational session lifecycle
  and adapters for local artifacts.
- `src/memo/error_contract.py` — `memo.error.v1` serialization.

### Modify

- `src/memo/operation_ledger.py` — compatibility facade; switches to v2 only at
  the final task.
- `src/memo/operational.py` — route current methods through command + view
  services while preserving return shapes.
- `src/memo/config.py` — add `operational_root` and `operational_db`.
- `src/memo/identity.py` — derive `PrincipalIdentity` from current machine,
  session, and client context.
- `src/memo/errors.py` — typed operational exceptions.
- `src/memo/memory/facade.py` — instantiate the v2 components behind
  `self.operational`.
- `src/memo/memory/outcome_feedback_ops.py` — enqueue stable durable promotions.
- `src/memo/memory/write_ops.py` — durable operation-key reconciliation.
- `src/memo/cli_operational.py` — require promotion idempotency keys.
- `src/memo/federation.py` — exchange verified anchors and origin bundles.
- `src/memo/definitive.py` — verify migration stamp, anchors, and view catch-up.
- `src/memo/session.py` — make JSON session data a derived local cache.
- `src/memo/server_core_history.py` — read canonical sessions.
- `src/memo/server_session_patterns.py` — remove the competing session
  authority and expose `memo_session_*`.
- `src/memo/server_operational.py` — require idempotency keys for mutations.
- `src/memo/server_annotations.py` — serialize typed operational failures.

### Tests

- Create:
  - `tests/test_operation_ledger_v1_compat.py`
  - `tests/test_operational_event_v2.py`
  - `tests/test_operational_event_types.py`
  - `tests/test_operational_signing.py`
  - `tests/test_operational_key_store.py`
  - `tests/test_operational_roster.py`
  - `tests/test_operational_epoch.py`
  - `tests/test_operation_ledger_v2.py`
  - `tests/test_operation_views_v2.py`
  - `tests/test_operation_migration_v2.py`
  - `tests/test_operational_idempotency.py`
  - `tests/test_durable_outbox.py`
  - `tests/test_write_ops_operation_identity.py`
  - `tests/test_operational_sessions_v2.py`
  - `tests/test_operational_errors.py`
- Extend:
  - `tests/test_operational_memory.py`
  - `tests/test_definitive_memory.py`
  - `tests/test_identity.py`
  - `tests/test_session.py`
  - `tests/test_session_patterns.py`
  - `tests/test_server_core_history.py`
  - `tests/test_briefing_unified.py`
  - `tests/test_surface_profiles.py`
  - `tests/test_cli_mcp_surface_smoke.py`

## Shared Interfaces

The plan uses these exact names across tasks:

```python
@dataclass(frozen=True)
class PrincipalIdentity:
    principal_id: str
    actor_id: str
    kind: str
    device_id: str
    session_id: str
    source_client: str
    signature: str = ""
    key_id: str = ""


@dataclass(frozen=True)
class SourceProof:
    source_system: str
    source_event_id: str
    source_schema: str
    source_origin: str
    source_sequence: int
    source_previous_hash: str
    source_event_hash: str
    source_content_hash: str
    source_actor: Mapping[str, object]
    source_subject_uri: str


@dataclass(frozen=True)
class MigrationOrigin:
    schema: Literal["memo.operational_migration_origin.v1"]
    attempt_id: str
    migration_device_id: str
    source_manifest_sha256: str
    capability_manifest_sha256: str
    attestor_device_id: str
    attestor_key_id: str
    roster_version: int
    issued_at: str
    expires_at: str
    signature: str


@dataclass(frozen=True)
class EpochMarkerAuthorization:
    schema: Literal["memo.operational_epoch_authorization.v1"]
    attempt_id: str
    device_id: str
    epoch: int
    control_oid: str
    artifact_digests: Mapping[str, str]
    roster_version: int
    key_id: str
    signature: SignatureEnvelope


@dataclass(frozen=True)
class ChainAnchor:
    schema: Literal["memo.operational_anchor.v1"]
    anchor_id: str
    origin_device: str
    ledger_epoch: int
    reducer_version: int
    kind: Literal["empty", "memo_v1", "compaction"]
    base_sequence: int
    base_event_hash: str
    final_sequence: int
    final_event_hash: str
    previous_anchor_hash: str
    source_manifest_sha256: str
    state_sha256: str
    checkpoint_id: str
    checkpoint_sha256: str
    checkpoint_size: int
    created_at: str
    anchor_hash: str
    roster_version: int
    signer_role: Literal["origin", "migration_attestor"]
    attested_origin: str
    key_id: str
    signature: str


@dataclass(frozen=True)
class OperationalCommand:
    event_type: str
    actor: PrincipalIdentity
    target_id: str | None
    project: str
    workspace: str
    expires_at: str | None
    visibility: str
    idempotency_key: str
    caused_by: tuple[str, ...]
    subject_uri: str
    trace_id: str
    payload: Mapping[str, object]
    source_proof: SourceProof | None = None


@dataclass(frozen=True)
class OperationalEventV2:
    schema: Literal["memo.operational_event.v2"]
    schema_version: Literal[2]
    event_id: str
    event_type: str
    actor: PrincipalIdentity
    target_id: str | None
    project: str
    workspace: str
    origin_device: str
    origin_sequence: int
    logical_clock: str
    authority_epoch: int
    control_oid: str
    created_at: str
    expires_at: str | None
    visibility: str
    idempotency_key: str
    caused_by: tuple[str, ...]
    subject_uri: str
    trace_id: str
    payload: Mapping[str, object]
    content_hash: str
    previous_hash: str
    event_hash: str
    source_proof: SourceProof | None
    roster_version: int
    key_id: str
    signature: str


@dataclass(frozen=True)
class CommandResult:
    event: OperationalEventV2
    replayed: bool
    result: Mapping[str, object]


@dataclass(frozen=True)
class OriginPosition:
    origin_device: str
    sequence: int
    event_hash: str
    anchor_hash: str


@dataclass(frozen=True)
class OriginBundle:
    anchor: ChainAnchor
    checkpoint: bytes
    events: tuple[OperationalEventV2, ...]
    head_sequence: int
    head_hash: str


@dataclass(frozen=True)
class StateCheckpoint:
    schema: Literal["memo.operational_checkpoint.v1"]
    checkpoint_id: str
    reducer_version: int
    origin_device: str
    through_sequence: int
    through_event_hash: str
    state_bytes: bytes
    state_sha256: str
    created_at: str


@dataclass(frozen=True)
class SessionCheckpoint:
    session_id: str
    principal_id: str
    project: str
    workspace: str
    status: Literal["active", "recoverable", "terminated"]
    branch: str | None
    head: str | None
    summary: str
    checkpointed_at: str
    source_event_id: str


@dataclass(frozen=True)
class MigrationPreparedStamp:
    schema: Literal["memo.operational_migration_prepared.v1"]
    source_manifest_sha256: str
    target_generation_sha256: str
    parity_report_sha256: str
    attestor_key_id: str
    signature: str


@dataclass(frozen=True)
class LedgerImportReport:
    manifest_sha256: str
    origins_seen: tuple[str, ...]
    events_inserted: int
    events_replayed: int
    quarantined: tuple[str, ...]
    final_positions: tuple[OriginPosition, ...]


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    checked_origins: tuple[str, ...]
    checked_events: int
    state_sha256: str
    errors: tuple[str, ...]
```

The retained checkpoint bytes are authority material: an anchor without the
matching bytes cannot authorize deletion or rebuild.

`CommitContext` contains authenticated `PrincipalIdentity`, `authority_epoch`,
fresh verified `control_oid`, local origin device, and optional signed
`MigrationOrigin`. External adapters must call
`EpochFence.context(identity, *, request_epoch: int, request_control_oid: str)
-> CommitContext`; the fence re-reads the durable marker and rejects omitted,
stale, future, or mismatched caller values. It never fills authority fields
from local state on behalf of a client. Only internal daemon/migration code
holding an unforgeable, process-local `SystemCapability` may call
`EpochFence.system_context(identity, *, capability: SystemCapability) ->
CommitContext`. `MigrationOrigin` is bound to one attempt, enrolled migration
attestor key, manifest hash, and `SourceProof`; ordinary callers cannot set an
origin override.
Its signed bytes use domain
`memo.operational.migration_origin.v1` and canonical JSON with `signature=""`.
The attestor key must be enrolled with only the `migration_attestor` role at
the recorded roster version; its source/capability manifest hashes, safe
migration device ID, attempt, and validity window are immutable. The exact
canonical `MigrationOrigin` and its SHA-256 are fields of `MigrationPlan`,
`ImportReceipt`, and every imported event's `CommitContext`; verification
recomputes all three before admission.

`EpochFence.activate(*, authorization: EpochMarkerAuthorization,
observed_artifact_digests: Mapping[str, str]) -> None` verifies the complete
canonical authorization under domain
`memo.operational_epoch_authorization.v1`, verifies its enrolled device key and
roster version, requires exact equality with freshly hashed local artifacts,
then atomically/fsyncs a monotonic marker containing epoch, control OID,
artifact digests, and authorization SHA-256. `EpochFence.bootstrap(...)` uses
the same record at epoch 0 with bootstrap-roster and empty-anchor digests.
`EpochFence.verify(context: CommitContext) -> None` re-reads it under the
transaction lock. Missing context, stale/future epoch, wrong control OID, bad
signature, marker rollback, or marker loss after first activation fails closed.
Fresh local installs start at signed epoch 0.

`VerificationRoster.bootstrap(*, device_id: str, key: PublicKeyRecord,
root: Path) ->
VerificationRoster` creates the sole fresh-install trust root: a canonical,
signed, monotonic one-peer roster at version 1. The bootstrap signature is
self-verifiable against `key`, permits only the local origin and migration
attestor roles explicitly present on that key, and is atomically persisted
before the signed epoch-0 marker is created. Plan 02 adopts this exact roster;
it does not invent a second bootstrap path.

Permanent signing contracts used by every later plan:

- `OperationalSigner.sign(*, domain: str, payload: bytes, key_id: str) ->
  SignatureEnvelope`
- `OperationalVerifier.verify(*, domain: str, payload: bytes, envelope:
  SignatureEnvelope, roster: VerificationRoster) -> None`
- `DeviceKeyStore.generate(*, device_id: str) -> PublicKeyRecord`
- `DeviceKeyStore.sign(*, key_id: str, payload: bytes) -> bytes`
- `DeviceKeyStore.destroy(*, key_id: str) -> None`
- `VerificationRoster.bootstrap(*, device_id: str, key: PublicKeyRecord,
  root: Path) ->
  VerificationRoster`

`SignatureEnvelope` contains `algorithm="ed25519"`, `key_id`, `roster_version`,
and base64url `signature`. `PublicKeyRecord` contains device ID, key ID,
SHA-256 fingerprint, raw public key, roles, enrollment sequence, and optional
revocation sequence. Production private keys are non-exportable application
secrets in macOS Keychain; tests use the in-memory provider.

`OperationalStore.commit(command: OperationalCommand, *, context:
CommitContext) -> CommandResult` is the only v2 mutation entry. It rejects a
missing/stale epoch, mismatched control OID, unauthorized origin override, or
principal mismatch before touching the journal. Its
lock/catch-up/idempotency/append/apply order is fixed:

1. acquire `operational-transactions`;
2. re-read and validate the epoch/control fence;
3. catch views up to the ledger;
4. resolve `(project, idempotency_key)`;
5. return the stored result when request hashes match;
6. raise `idempotency_conflict` when they differ;
7. append and fsync one event;
8. apply it and store the result in one SQLite transaction; and
9. return `CommandResult`.

### Task 1: Freeze v1 and define pure v2 contracts

**Files:**
- Create: `src/memo/operation_ledger_v1.py`
- Create: `src/memo/operational_event.py`
- Create: `src/memo/operational_event_types.py`
- Create: `src/memo/operational_signing.py`
- Create: `src/memo/operational_key_store.py`
- Create: `src/memo/operational_roster.py`
- Create: `src/memo/operational_epoch.py`
- Create: `src/memo/error_contract.py`
- Modify: `src/memo/identity.py`
- Modify: `src/memo/errors.py`
- Test: `tests/test_operation_ledger_v1_compat.py`
- Test: `tests/test_operational_event_v2.py`
- Test: `tests/test_operational_event_types.py`
- Test: `tests/test_operational_signing.py`
- Test: `tests/test_operational_key_store.py`
- Test: `tests/test_operational_roster.py`
- Test: `tests/test_operational_epoch.py`
- Test: `tests/test_operational_errors.py`

**Interfaces:**
- Consumes: current `MemoEvent`, `ActorIdentity`, canonical JSON, and v1 journal
  validation from `operation_ledger.py`.
- Produces: all shared dataclasses above, `LegacyOperationLedger`,
  the shared event registry,
  `OperationalSigner`, `OperationalVerifier`, `DeviceKeyStore`,
  `VerificationRoster`,
  `EpochFence`,
  `OperationalError`, `OperationalErrorCode`, and `MemoErrorEnvelope`.

- [ ] **Step 1: Write failing contract and v1-compatibility tests**

```python
def test_v1_reader_preserves_bytes_and_head(tmp_path, legacy_journal_bytes):
    root = tmp_path / "journal" / "events" / "device-a"
    root.mkdir(parents=True)
    path = root / "2026-07-29.jsonl"
    path.write_bytes(legacy_journal_bytes)
    before = path.read_bytes()
    ledger = LegacyOperationLedger(tmp_path, device_id="device-a")
    report = ledger.verify()
    assert report["ok"] is True
    assert ledger.head_hashes()["device-a"]
    assert path.read_bytes() == before


def test_v2_event_hash_is_canonical_and_tamper_evident(event_factory):
    first = event_factory(payload={"b": 2, "a": 1})
    same = event_factory(payload={"a": 1, "b": 2})
    assert first.event_hash == same.event_hash
    changed = replace(first, payload={"a": 9, "b": 2})
    assert canonical_event_hash(changed) != first.event_hash


def test_signature_is_domain_separated_and_roster_bound(signer, verifier, roster):
    envelope = signer.sign(
        domain="memo.operational.event.v2", payload=b'{"event_id":"e1"}',
        key_id=roster.local_key_id,
    )
    verifier.verify(
        domain="memo.operational.event.v2", payload=b'{"event_id":"e1"}',
        envelope=envelope, roster=roster,
    )
    with pytest.raises(SignatureError):
        verifier.verify(
            domain="memo.cutover.vote.v1", payload=b'{"event_id":"e1"}',
            envelope=envelope, roster=roster,
        )


def test_fresh_bootstrap_persists_one_peer_roster_before_epoch_zero(
    tmp_path, local_public_key
):
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=local_public_key, root=tmp_path
    )
    assert roster.version == 1
    assert roster.peers == ("device-a",)
    assert verify_bootstrap(roster, local_public_key)
    assert (tmp_path / "verification-roster.json").exists()
    assert not (tmp_path / "authority-epoch.json").exists()


def test_external_context_requires_explicit_epoch_and_control(epoch_fence, identity):
    with pytest.raises(TypeError):
        epoch_fence.context(identity)
    with pytest.raises(AuthorityEpochError):
        epoch_fence.context(
            identity, request_epoch=4, request_control_oid="stale-control"
        )


def test_error_envelope_is_stable():
    err = OperationalError(
        OperationalErrorCode.SEQUENCE_GAP,
        "origin device-a expected 4",
        retryable=True,
        details={"expected": 4, "actual": 6},
    )
    out = MemoErrorEnvelope.from_error(err, runtime_version="4.4.6", epoch=0)
    assert out.schema == "memo.error.v1"
    assert out.code == "sequence_gap"
    assert out.retryable is True
```

- [ ] **Step 2: Run the tests and confirm the missing contracts fail**

Run:

```bash
uv run --no-sync pytest \
  tests/test_operation_ledger_v1_compat.py \
  tests/test_operational_event_v2.py \
  tests/test_operational_event_types.py \
  tests/test_operational_signing.py \
  tests/test_operational_key_store.py \
  tests/test_operational_roster.py \
  tests/test_operational_epoch.py \
  tests/test_operational_errors.py -v
```

Expected: import failures for the new frozen-reader, event, signing, key-store,
roster, epoch, and error-contract modules.

- [ ] **Step 3: Freeze v1 and implement canonical v2 value objects**

Copy/freeze the current v1 implementation without semantic edits into
`operation_ledger_v1.py`, name the copy `LegacyOperationLedger`, and keep the
active `operation_ledger.py` implementation/imports unchanged until Task 7. Keep its
decoder, canonical serialization, head repair, and verification byte-for-byte
equivalent. In `operational_event.py`, implement:

```python
def canonical_event_hash(event: OperationalEventV2) -> str:
    body = asdict(event)
    body["event_hash"] = ""
    body["signature"] = ""
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_event(event: OperationalEventV2) -> None:
    if event.schema != "memo.operational_event.v2" or event.schema_version != 2:
        raise OperationalError(
            OperationalErrorCode.UNKNOWN_SCHEMA,
            f"unsupported operational schema: {event.schema}/{event.schema_version}",
            retryable=False,
        )
    if event.origin_sequence < 1 or not event.idempotency_key:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "sequence must be positive and idempotency_key must be non-empty",
            retryable=False,
        )
    if canonical_event_hash(event) != event.event_hash:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "event hash mismatch",
            retryable=False,
        )
```

Use Ed25519 from Memo's existing `cryptography` dependency. The bytes signed
are exactly `b"memo-signature-v1\\0" + domain.encode("ascii") + b"\\0" +
payload`; payload is canonical UTF-8 JSON with sorted keys, compact separators,
no NaN, and the signature field blank. Reject unknown algorithms/domains,
duplicate key IDs/fingerprints, mismatched device roles, roster regression,
revoked keys at or after their activation sequence. Signature verification for
operational events and anchors is stateless and therefore allows an identical
record to be checked repeatedly; ledger import owns duplicate handling and
accepts an identical event as a no-op while rejecting a same-position fork.
Replay rejection applies only to monotonic control records and votes. Keychain
access is isolated behind `DeviceKeyStore`; no PEM/private
seed is written to disk or included in diagnostics.

Anchor authorization is kind-specific. `empty` and `compaction` anchors require
`signer_role="origin"` and a key authorized for `origin_device`.
`memo_v1` anchors preserve the legacy source in `origin_device` and
`attested_origin`, but are signed by an enrolled
`signer_role="migration_attestor"` key; they never pretend that historical
origins signed new material. Their checkpoint is the canonical empty reducer
state immediately before the first seed event. Deterministic seed events then
reduce the legacy state exactly once. A pre-populated legacy-state checkpoint
plus seed events is invalid.

`operational_event_types.py` owns every fully qualified event constant and a
payload validator. The initial closed registry includes all current Memo seed
events and later native domains: focus, handoff, attention, conflict, outcome,
session, coordination, delivery, cursor, presence, terminal, health, roster,
compaction, and durable promotion. Unknown names fail validation. Plans 02–05
must import these constants; no plan may spell a competing short name.

Implement `PrincipalIdentity.from_current(identity, source_client)` without
changing `ActorIdentity`. Add the exact error codes from the design:
`invalid_event`, `unknown_schema`, `sequence_gap`, `anchor_conflict`,
`idempotency_conflict`, `signature_invalid`, `key_revoked`, `expired`,
`not_found`, and `storage_unavailable`.

- [ ] **Step 4: Run focused tests, mypy, and ruff**

```bash
uv run --no-sync pytest \
  tests/test_operation_ledger_v1_compat.py \
  tests/test_operational_event_v2.py \
  tests/test_operational_errors.py \
  tests/test_identity.py -v
uv run --no-sync mypy \
  src/memo/operational_event.py src/memo/operational_event_types.py \
  src/memo/operational_signing.py \
  src/memo/operational_key_store.py src/memo/operational_roster.py \
  src/memo/operational_epoch.py \
  src/memo/error_contract.py
uv run --no-sync ruff check \
  src/memo/operation_ledger_v1.py src/memo/operational_event.py \
  src/memo/operational_event_types.py \
  src/memo/operational_signing.py src/memo/operational_key_store.py \
  src/memo/operational_roster.py src/memo/operational_epoch.py \
  src/memo/error_contract.py src/memo/identity.py src/memo/errors.py \
  tests/test_operation_ledger_v1_compat.py \
  tests/test_operational_event_v2.py tests/test_operational_event_types.py \
  tests/test_operational_signing.py \
  tests/test_operational_key_store.py tests/test_operational_roster.py \
  tests/test_operational_epoch.py \
  tests/test_operational_errors.py
```

Expected: all pass; `git diff -- src/memo/contracts.py` is empty.

- [ ] **Step 5: Commit the frozen reader and contracts**

```bash
git add \
  src/memo/operation_ledger_v1.py src/memo/operational_event.py \
  src/memo/operational_event_types.py \
  src/memo/operational_signing.py src/memo/operational_key_store.py \
  src/memo/operational_roster.py src/memo/operational_epoch.py \
  src/memo/error_contract.py src/memo/identity.py src/memo/errors.py \
  tests/test_operation_ledger_v1_compat.py \
  tests/test_operational_event_v2.py tests/test_operational_event_types.py \
  tests/test_operational_signing.py \
  tests/test_operational_key_store.py tests/test_operational_roster.py \
  tests/test_operational_epoch.py \
  tests/test_operational_errors.py
git commit -m "feat: define operational ledger v2 contracts"
```

### Task 2: Implement local v2 anchors, append, verification, and bundles

**Files:**
- Create: `src/memo/operation_ledger_v2.py`
- Test: `tests/test_operation_ledger_v2.py`

**Interfaces:**
- Consumes: `ChainAnchor`, `OperationalCommand`, `OperationalEventV2`,
  `LegacyOperationLedger`, existing atomic write/lock helpers.
- Produces: `OperationLedgerV2.ensure_anchor`, `append`, `iter_events`,
  `positions`, `verify`, `export_bundles`, `validate_import_bundles`,
  `import_bundles`, and `quarantine`.

- [ ] **Step 1: Write failing ledger tests**

```python
def test_genesis_anchor_attests_verified_v1_head(
    tmp_path, legacy_ledger, migration_attestor, empty_checkpoint
):
    ledger = OperationLedgerV2(tmp_path, device_id="device-a", clock=frozen_clock)
    anchor = ledger.ensure_anchor(
        ChainAnchor.from_v1(
            legacy_ledger,
            source_head_hash=legacy_ledger.head_hashes()["device-a"],
            migration_attestor=migration_attestor,
            checkpoint=empty_checkpoint,
        )
    )
    assert anchor.kind == "memo_v1"
    assert anchor.source_manifest_sha256 == legacy_ledger.manifest_sha256()
    assert anchor.attested_origin == "device-a"
    assert anchor.signer_role == "migration_attestor"
    assert ledger.position("device-a").sequence == anchor.base_sequence


def test_append_fsyncs_and_advances_exactly_once(tmp_path, command, commit_context):
    ledger = OperationLedgerV2(tmp_path, device_id="device-a", clock=frozen_clock)
    ledger.ensure_anchor()
    first = ledger.append(command, context=commit_context)
    assert first.origin_sequence == 1
    assert first.previous_hash == ledger.anchor("device-a").base_event_hash
    assert ledger.verify().ok is True


def test_import_rejects_gap_and_quarantines_unknown_schema(
    tmp_path, signed_bundle_factory
):
    ledger = OperationLedgerV2(tmp_path, device_id="device-b", clock=frozen_clock)
    gap = signed_bundle_factory(origin="device-a", sequences=[1, 3])
    with pytest.raises(OperationalError, match="sequence"):
        ledger.import_bundles([gap])
    assert list(ledger.quarantine_dir.iterdir())
```

Also cover empty anchors, head repair after append-before-head crash, tampering,
forks, repeated bundle import, anchor regression, symlink rejection, and a
source-proof event linked to the sealed v1 head.

- [ ] **Step 2: Run the tests and confirm `OperationLedgerV2` is missing**

```bash
uv run --no-sync pytest tests/test_operation_ledger_v2.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement the v2 ledger**

Use this filesystem contract:

```text
<operational_root>/journal/
  anchors/<origin>.json
  checkpoints/<origin>/<anchor-id>.json
  events/<origin>/<YYYY-MM-DD>.jsonl
  heads/<origin>.json
  quarantine/<timestamp>-<sha256>.json
```

Implement `append()` under the existing file lock:

```python
def append(
    self, command: OperationalCommand, *, context: CommitContext
) -> OperationalEventV2:
    with self._lock:
        self.epoch_fence.verify(context)
        anchor = self.ensure_anchor()
        origin = context.migration_origin.device_id if context.migration_origin else self.device_id
        position = self._load_position(origin, anchor)
        event = self._build_event(
            command,
            context=context,
            sequence=position.sequence + 1,
            previous_hash=position.event_hash,
        )
        event = self.signer.sign_event(event)
        validate_event(event)
        append_jsonl_fsync(self._segment_path(event), event_to_dict(event))
        self._write_head_atomic(event)
        return event
```

Bundle validation must verify checkpoint bytes/hash/reducer version, anchor
hash/signature, every event signature/hash,
contiguous origin sequence, previous-hash continuity, final head, and enrolled
origin before writing. Validate all bundles first; only then import. A repeated
identical bundle returns an `ImportReport` with zero inserted events.

- [ ] **Step 4: Run focused verification**

```bash
uv run --no-sync pytest \
  tests/test_operation_ledger_v2.py \
  tests/test_operation_ledger_v1_compat.py -v
uv run --no-sync mypy src/memo/operation_ledger_v2.py
uv run --no-sync ruff check \
  src/memo/operation_ledger_v2.py tests/test_operation_ledger_v2.py
```

Expected: all pass.

- [ ] **Step 5: Commit the v2 ledger**

```bash
git add src/memo/operation_ledger_v2.py tests/test_operation_ledger_v2.py
git commit -m "feat: add anchored operational ledger v2"
```

### Task 3: Add transactional SQLite views and idempotent commit

**Files:**
- Create: `src/memo/operation_view_schema.py`
- Create: `src/memo/operation_views.py`
- Modify: `src/memo/operational.py`
- Modify: `src/memo/config.py`
- Test: `tests/test_operation_views_v2.py`
- Test: `tests/test_operational_idempotency.py`
- Extend: `tests/test_operational_memory.py`

**Interfaces:**
- Consumes: `OperationLedgerV2`, `OperationalEventV2`, existing v1 reducer
  behavior.
- Produces: `OperationalViewStore.apply_events`, `catch_up`, `rebuild`,
  `state`; `OperationalStore.commit`.

- [ ] **Step 1: Write failing view and crash-window tests**

```python
def test_apply_is_one_transaction_and_replay_is_noop(view_store, event):
    first = view_store.apply_events([event])
    second = view_store.apply_events([event])
    assert first.applied == 1
    assert second.applied == 0
    assert second.duplicates == 1
    assert view_store.state()["focus"]


def test_commit_recovers_append_before_view_crash(
    store, command, commit_context, monkeypatch
):
    monkeypatch.setattr(store.views, "apply_events", Mock(side_effect=OSError("boom")))
    with pytest.raises(OSError):
        store.commit(command, context=commit_context)
    event_count = len(list(store.ledger.validated_events()))
    monkeypatch.undo()
    replay = store.commit(command, context=commit_context)
    assert len(list(store.ledger.validated_events())) == event_count
    assert replay.replayed is True


def test_idempotency_key_with_different_request_is_rejected(
    store, command, commit_context
):
    store.commit(command, context=commit_context)
    changed = replace(command, payload={"different": True})
    with pytest.raises(OperationalError) as exc:
        store.commit(changed, context=commit_context)
    assert exc.value.code == OperationalErrorCode.IDEMPOTENCY_CONFLICT
```

Cover rollback on reducer failure, quarantine of unknown operations, unique
`(origin_device, origin_sequence)`, rebuild root equality, and exact v1 state
shape for focus, handoffs, attention, conflicts, and outcomes. Add bypass tests
showing stale or missing epochs fail before append through MCP, CLI, daemon,
`Memory.operational`, and a direct internal `OperationalStore.commit` call.

- [ ] **Step 2: Run the focused tests and observe missing schema/store**

```bash
uv run --no-sync pytest \
  tests/test_operation_views_v2.py \
  tests/test_operational_idempotency.py \
  tests/test_operational_memory.py -v
```

Expected: import or constructor failures.

- [ ] **Step 3: Implement schema, reducers, and commit protocol**

`operation_view_schema.py` must create:

```sql
CREATE TABLE applied_events (
  event_id TEXT PRIMARY KEY,
  origin_device TEXT NOT NULL,
  origin_sequence INTEGER NOT NULL,
  event_hash TEXT NOT NULL,
  applied_at TEXT NOT NULL,
  UNIQUE(origin_device, origin_sequence)
);
CREATE TABLE idempotency (
  scope TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  event_id TEXT NOT NULL,
  result_json TEXT NOT NULL,
  PRIMARY KEY(scope, idempotency_key)
);
```

Add `view_meta`, `origin_cursors`, `focus`, `handoffs`, `attention`,
`conflicts`, `outcomes`, `sessions`, `session_local_artifacts`,
`durable_outbox`, and `quarantined_events`. Use `BEGIN IMMEDIATE` for one event
application. `durable_outbox` is derived only from
`durable.promotion.requested.v1`,
`durable.promotion.retry_scheduled.v1`,
`durable.promotion.completed.v1`, and
`durable.promotion.rejected.v1` events; rebuilding the view reconstructs every
pending intent, attempt count, deterministic `retry_at`, and reconciled memory
ID. `OperationalStore.commit()` follows the nine-step protocol in
Shared Interfaces and preserves the current convenience methods as adapters
that build `OperationalCommand`. External CLI/MCP/library adapters must receive
and pass authenticated identity plus explicit request epoch/control OID into
`EpochFence.context`; they cannot default or synthesize them. Only the
operational daemon and migration worker may request a context through
`system_context` with their process-local `SystemCapability`.

- [ ] **Step 4: Run focused and legacy operational tests**

```bash
uv run --no-sync pytest \
  tests/test_operation_views_v2.py \
  tests/test_operational_idempotency.py \
  tests/test_operational_memory.py \
  tests/test_write_freeze_gc.py -v
uv run --no-sync mypy \
  src/memo/operation_view_schema.py src/memo/operation_views.py \
  src/memo/operational.py
uv run --no-sync ruff check \
  src/memo/operation_view_schema.py src/memo/operation_views.py \
  src/memo/operational.py src/memo/config.py \
  tests/test_operation_views_v2.py tests/test_operational_idempotency.py
```

Expected: all pass; `test_write_freeze_gc.py` requires no production change.

- [ ] **Step 5: Commit the derived view store**

```bash
git add \
  src/memo/operation_view_schema.py src/memo/operation_views.py \
  src/memo/operational.py src/memo/config.py \
  tests/test_operation_views_v2.py tests/test_operational_idempotency.py \
  tests/test_operational_memory.py
git commit -m "feat: add transactional operational views"
```

### Task 4: Build deterministic v1 genesis migration and parity gate

**Files:**
- Create: `src/memo/operation_migration.py`
- Modify: `src/memo/definitive.py`
- Test: `tests/test_operation_migration_v2.py`
- Extend: `tests/test_definitive_memory.py`

**Interfaces:**
- Consumes: `LegacyOperationLedger`, current v1 reducer,
  `OperationLedgerV2`, `OperationalViewStore`.
- Produces: `plan_v1_migration`, `apply_v1_migration`,
  `verify_v1_parity`, `MigrationPreparedStamp`. The production activation
  stamp is deliberately owned by Task 7.

- [ ] **Step 1: Write failing migration/parity tests**

```python
def test_v1_migration_is_deterministic_and_idempotent(tmp_path, v1_fixture):
    first = migrate_v1(v1_fixture, tmp_path / "v2")
    second = migrate_v1(v1_fixture, tmp_path / "v2")
    assert first.manifest_sha256 == second.manifest_sha256
    assert second.events_inserted == 0
    assert first.v1_state_sha256 == first.v2_state_sha256


def test_corrupt_v1_aborts_before_any_v2_write(tmp_path, corrupt_v1_fixture):
    target = tmp_path / "v2"
    with pytest.raises(OperationalError):
        migrate_v1(corrupt_v1_fixture, target)
    assert not target.exists()


def test_changed_source_after_plan_is_hard_failure(tmp_path, v1_fixture):
    plan = plan_v1_migration(v1_fixture)
    mutate_legacy_journal(v1_fixture)
    with pytest.raises(OperationalError, match="manifest"):
        apply_v1_migration(plan, tmp_path / "v2")
```

Also assert deterministic seed IDs
`memo-v1/<manifest>/<domain>/<id>`, preserved source proofs, exact domain
parity, and no activation stamp before parity.

- [ ] **Step 2: Run the tests and confirm migration functions are absent**

```bash
uv run --no-sync pytest \
  tests/test_operation_migration_v2.py tests/test_definitive_memory.py -v
```

Expected: import failures.

- [ ] **Step 3: Implement plan/apply/verify without touching v1**

Implement this fixed sequence:

```python
def migrate_v1(source: Path, target: Path) -> MigrationReport:
    plan = plan_v1_migration(source)
    staging = target.with_name(f"{target.name}.staging-{plan.manifest_sha256[:12]}")
    report = apply_v1_migration(plan, staging)
    parity = verify_v1_parity(plan, staging)
    if not parity.equal:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            f"v1/v2 state mismatch: {parity.diff}",
            retryable=False,
        )
    install_prepared_generation_atomic(staging, target, plan.prepared_stamp())
    return report.with_parity(parity)
```

Verify all v1 chains, compute canonical manifest and state root, seal one
genesis anchor per origin with the enrolled migration attestor, checkpoint the
empty pre-seed reducer state, emit deterministic v2 seed events, rebuild SQLite
only from v2, compare canonical state excluding head metadata, then atomically
write `migration-v1.json`. Never use `operational-state.json` as input; use it
only as an additional parity oracle. `migration-v1.json` proves prepared
migration and parity but cannot select v2 for production.

- [ ] **Step 4: Run migration, definitive, and legacy tests**

```bash
uv run --no-sync pytest \
  tests/test_operation_migration_v2.py \
  tests/test_operation_ledger_v1_compat.py \
  tests/test_operation_ledger_v2.py \
  tests/test_operation_views_v2.py \
  tests/test_definitive_memory.py -v
uv run --no-sync mypy src/memo/operation_migration.py src/memo/definitive.py
uv run --no-sync ruff check \
  src/memo/operation_migration.py src/memo/definitive.py \
  tests/test_operation_migration_v2.py tests/test_definitive_memory.py
```

Expected: all pass.

- [ ] **Step 5: Commit migration and parity**

```bash
git add \
  src/memo/operation_migration.py src/memo/definitive.py \
  tests/test_operation_migration_v2.py tests/test_definitive_memory.py
git commit -m "feat: migrate operational v1 through verified genesis"
```

### Task 5: Add exactly-once durable promotion outbox

**Files:**
- Create: `src/memo/durable_outbox.py`
- Modify: `src/memo/memory/write_ops.py`
- Modify: `src/memo/memory/outcome_feedback_ops.py`
- Modify: `src/memo/cli_operational.py`
- Modify: `src/memo/server_operational.py`
- Test: `tests/test_durable_outbox.py`
- Test: `tests/test_write_ops_operation_identity.py`
- Extend: `tests/test_definitive_memory.py`
- Extend: `tests/test_cli_mcp_surface_smoke.py`

**Interfaces:**
- Consumes: the event-derived `durable_outbox` view, `Memory.save_operation`,
  and source event IDs.
- Produces: `DurableOutboxWorker.run_once(limit: int = 100)`,
  durable operation identity `promotion/<sha256(key)>`.
- Changes: `_OutcomeFeedbackOpsMixin.promote_learning(memory_ids: list[str],
  *, title: str, kind: str = "procedure", content: str | None = None,
  reason: str = "outcome-backed promotion", actor_id: str = "memo",
  idempotency_key: str) -> MemoryRecord`. CLI and MCP callers must supply the
  key. The method commits `durable.promotion.requested.v1`, synchronously
  reconciles that intent, and returns the existing/new `MemoryRecord`; the
  daemon only recovers interrupted intents.
- `Memory.save_operation(*, operation_key: str, request_hash: str,
  save_kwargs: Mapping[str, object]) -> MemoryRecord`
- `Memory.find_by_operation_key(operation_key: str, request_hash: str) ->
  MemoryRecord | None`
- `OperationalViewStore.pending_outbox(*, limit: int) ->
  list[FrozenPromotionIntent]`
- `OperationalViewStore.outbox_report() -> OutboxRunReport`

`FrozenPromotionIntent` contains `id`, `idempotency_key`, `operation_key`,
`request_hash`, immutable `save_kwargs`, ordered `source_event_ids`,
`created_at`, and `attempts`. `OutboxRunReport` contains `examined`, `completed`,
`retried`, `quarantined`, and `pending`.
`attempts` and retry timing are reducer output, never mutable worker metrics:
each failed transient attempt commits
`durable.promotion.retry_scheduled.v1` with the intent ID, failure class,
attempt number, and deterministic `retry_at`. Requested, retry-scheduled,
completed, and rejected events are the complete outbox authority.

- [ ] **Step 1: Write failing outbox crash/reconciliation tests**

```python
def test_outbox_reuses_memory_after_crash_post_save(outbox, memory, intent):
    memory.save_side_effect = SaveThenRaise("memory-123")
    with pytest.raises(RuntimeError):
        outbox.run_once()
    memory.save_side_effect = None
    report = outbox.run_once()
    assert report.completed == 1
    record = memory.find_by_operation_key(
        intent.operation_key, intent.request_hash
    )
    assert record is not None
    assert record.id == "memory-123"


def test_promotion_key_collision_with_new_payload_is_quarantined(outbox, intent):
    outbox.enqueue(intent)
    changed_kwargs = {**intent.save_kwargs, "body": "different"}
    outbox.enqueue(
        replace(
            intent,
            save_kwargs=changed_kwargs,
            request_hash=canonical_save_request_hash(changed_kwargs),
        )
    )
    report = outbox.run_once()
    assert report.quarantined == 1
```

Cover crash before save, rejected write policy, stable frozen timestamps,
bounded batch, source-event provenance, retry, and no duplicate record.

- [ ] **Step 2: Run the tests and confirm the worker is missing**

```bash
uv run --no-sync pytest tests/test_durable_outbox.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement stable intent identity and reconciliation**

```python
def promotion_operation_key(idempotency_key: str) -> str:
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"promotion/{digest}"


class DurableOutboxWorker:
    def run_once(self, *, limit: int = 100) -> OutboxRunReport:
        intents = self.store.pending_outbox(limit=limit)
        for intent in intents:
            try:
                saved = self.memory.save_operation(
                    operation_key=intent.operation_key,
                    request_hash=intent.request_hash,
                    save_kwargs=intent.save_kwargs,
                )
            except IdentityConflictError as exc:
                self.operational.commit(
                    intent.rejected_command(str(exc)),
                    context=self.context_factory(),
                )
                continue
            except Exception as exc:
                self.operational.commit(
                    intent.retry_scheduled_command(str(exc)),
                    context=self.context_factory(),
                )
                raise
            self.operational.commit(
                intent.completed_command(saved.id),
                context=self.context_factory(),
            )
        return self.store.outbox_report()
```

Extend `promote_learning` to require an `idempotency_key`, freeze the complete
save request once, commit the requested event, reconcile it, and preserve the
current `MemoryRecord` return shape. `save_operation` holds the existing
authority write lock, stores `operation_key` plus `request_hash` in Markdown
frontmatter, repairs an interrupted Markdown/index mapping by an exact
allowlisted frontmatter scan, returns the same record for the same hash, and
raises `IdentityConflictError` for a different hash. Topic identity remains a
secondary convenience, not the exactly-once proof. Explicitly test the crash
where Markdown exists but the vector row is pending and a full SQLite rebuild
between requested/completed events.

- [ ] **Step 4: Run focused outcome/outbox tests**

```bash
uv run --no-sync pytest \
  tests/test_durable_outbox.py \
  tests/test_write_ops_operation_identity.py \
  tests/test_operational_idempotency.py \
  tests/test_definitive_memory.py \
  tests/test_cli_mcp_surface_smoke.py -v
uv run --no-sync mypy \
  src/memo/durable_outbox.py src/memo/memory/write_ops.py \
  src/memo/memory/outcome_feedback_ops.py
uv run --no-sync ruff check \
  src/memo/durable_outbox.py src/memo/memory/write_ops.py \
  src/memo/memory/outcome_feedback_ops.py src/memo/cli_operational.py \
  src/memo/server_operational.py tests/test_durable_outbox.py \
  tests/test_write_ops_operation_identity.py
```

Expected: all pass.

- [ ] **Step 5: Commit durable promotion**

```bash
git add \
  src/memo/durable_outbox.py src/memo/memory/write_ops.py \
  src/memo/memory/outcome_feedback_ops.py src/memo/cli_operational.py \
  src/memo/server_operational.py tests/test_durable_outbox.py \
  tests/test_write_ops_operation_identity.py tests/test_definitive_memory.py \
  tests/test_cli_mcp_surface_smoke.py
git commit -m "feat: reconcile durable promotions exactly once"
```

### Task 6: Make ledger v2 the canonical session authority

**Files:**
- Create: `src/memo/operational_sessions.py`
- Modify: `src/memo/session.py`
- Modify: `src/memo/server_core_history.py`
- Modify: `src/memo/server_session_patterns.py`
- Test: `tests/test_operational_sessions_v2.py`
- Extend: `tests/test_session.py`
- Extend: `tests/test_session_patterns.py`
- Extend: `tests/test_server_core_history.py`

**Interfaces:**
- Consumes: `OperationalStore.commit`, current JSON session sidecar, current
  SQLite session-pattern rows.
- Produces: `OperationalSessionService.checkpoint`, `mark_recoverable`,
  `terminate`, `latest_recoverable`; public `memo_session_*` tools.

- [ ] **Step 1: Write failing lifecycle and merge tests**

```python
def test_session_lifecycle_is_monotonic(session_service, identity):
    cp = session_service.checkpoint(
        identity=identity,
        session_id="s1",
        project="memo",
        workspace="/work",
        summary="working",
        branch="main",
        head="abc",
        idempotency_key="cp-1",
    )
    done = session_service.terminate(session_id="s1", idempotency_key="done-1")
    assert cp.status == "active"
    assert done.status == "terminated"
    with pytest.raises(OperationalError):
        session_service.checkpoint(
            identity=identity,
            session_id="s1",
            project="memo",
            workspace="/work",
            summary="regress",
            branch="main",
            head="abc",
            idempotency_key="cp-2",
        )


def test_merge_keeps_portable_state_and_local_artifacts_separate(migrator):
    merged = migrator.merge_legacy(
        json_checkpoint={"id": "s1", "cwd": "/work", "prompt_path": "/private/p"},
        sqlite_row={"id": "s1", "project": "memo", "status": "recoverable"},
    )
    assert merged.checkpoint.project == "memo"
    assert "prompt_path" not in asdict(merged.checkpoint)
    assert merged.local_artifacts["prompt_path"] == "/private/p"
```

Cover incompatible project/cwd rejection, idempotent checkpoint, terminated
not recoverable, JSON cache rebuild, and public-name replacement without alias.

- [ ] **Step 2: Run session tests and observe missing service**

```bash
uv run --no-sync pytest \
  tests/test_operational_sessions_v2.py \
  tests/test_session.py \
  tests/test_session_patterns.py \
  tests/test_server_core_history.py -v
```

Expected: import failures for `operational_sessions`.

- [ ] **Step 3: Implement one authority and derived adapters**

Use events `session.checkpointed.v1`, `session.recoverable.v1`, and
`session.terminated.v1`.
Portable checkpoint data is identity, project, workspace, status, branch/head,
summary, timestamp, and source event. Transcript paths and prompt trails remain
in `session_local_artifacts` and never federate. Convert `session.py` writes to
`OperationalSessionService`; keep JSON output only as an atomic derived cache.
Rename `mem_session_*` registrations to `memo_session_*` in
`server_session_patterns.py` and update all first-party callers in the same
commit.

- [ ] **Step 4: Run session, history, briefing, and surface tests**

```bash
uv run --no-sync pytest \
  tests/test_operational_sessions_v2.py \
  tests/test_session.py tests/test_session_patterns.py \
  tests/test_server_core_history.py tests/test_briefing_unified.py \
  tests/test_surface_profiles.py tests/test_cli_mcp_surface_smoke.py -v
uv run --no-sync mypy \
  src/memo/operational_sessions.py src/memo/session.py \
  src/memo/server_session_patterns.py
uv run --no-sync ruff check \
  src/memo/operational_sessions.py src/memo/session.py \
  src/memo/server_core_history.py src/memo/server_session_patterns.py \
  tests/test_operational_sessions_v2.py
```

Expected: all pass and `rg -n 'mem_session_' src tests` returns no active
first-party registration or caller.

- [ ] **Step 5: Commit canonical sessions**

```bash
git add \
  src/memo/operational_sessions.py src/memo/session.py \
  src/memo/server_core_history.py src/memo/server_session_patterns.py \
  tests/test_operational_sessions_v2.py tests/test_session.py \
  tests/test_session_patterns.py tests/test_server_core_history.py \
  tests/test_briefing_unified.py tests/test_surface_profiles.py \
  tests/test_cli_mcp_surface_smoke.py
git commit -m "feat: make operational ledger the session authority"
```

### Task 7: Switch the Memo operational facade after full parity

**Files:**
- Modify: `src/memo/operation_ledger.py`
- Modify: `src/memo/memory/facade.py`
- Modify: `src/memo/federation.py`
- Modify: `src/memo/server_operational.py`
- Modify: `src/memo/server_annotations.py`
- Modify: `src/memo/definitive.py`
- Extend: `tests/test_operational_memory.py`
- Extend: `tests/test_definitive_memory.py`
- Extend: `tests/test_cli_mcp_surface_smoke.py`

**Interfaces:**
- Consumes: every interface produced by Tasks 1–6.
- Produces: v2-backed `Memory.operational`, verified bundle federation, typed
  operational MCP errors; keeps `LegacyOperationLedger` migration-only.

- [ ] **Step 1: Write failing facade/activation tests**

```python
def test_memory_uses_v2_only_after_valid_activation_stamp(tmp_cfg, v1_fixture):
    mem = Memory(tmp_cfg)
    assert mem.operational.backend_version == 1
    migrate_v1(v1_fixture, tmp_cfg.operational_root)
    activate_v2_after_full_parity(tmp_cfg)
    mem = Memory(tmp_cfg)
    assert mem.operational.backend_version == 2
    assert mem.operational.state() == expected_v1_state(v1_fixture)


def test_fresh_install_without_v1_starts_empty_v2(tmp_cfg):
    mem = Memory(tmp_cfg)
    assert mem.operational.backend_version == 2
    assert mem.operational.state() == empty_operational_state()
    assert mem.operational.verification_roster.version == 1
    assert mem.operational.epoch_fence.read().epoch == 0


def test_invalid_stamp_fails_closed_without_writing_v1(tmp_cfg, v1_fixture):
    migrate_v1(v1_fixture, tmp_cfg.operational_root)
    corrupt_activation_stamp(tmp_cfg.operational_root)
    with pytest.raises(OperationalError):
        Memory(tmp_cfg)
    assert legacy_bytes(tmp_cfg) == v1_fixture.original_bytes
```

Cover federation bundle validation before import, MCP idempotency requirement,
typed error envelope, definitive anchor/view checks, and unchanged successful
response shapes.

- [ ] **Step 2: Run facade/federation tests and observe legacy authority**

```bash
uv run --no-sync pytest \
  tests/test_operational_memory.py \
  tests/test_definitive_memory.py \
  tests/test_cli_mcp_surface_smoke.py -v
```

Expected: new activation assertions fail.

- [ ] **Step 3: Wire v2 through the facade and federation**

Make `operation_ledger.py` a narrow backend selector:

```python
def open_operational_backend(
    cfg: Config,
) -> LegacyOperationalBackend | V2OperationalBackend:
    source = inspect_operational_install(cfg)
    if source.kind == "fresh":
        return V2OperationalBackend.create_empty(cfg)
    if source.kind == "legacy_only":
        return LegacyOperationalBackend.from_existing(cfg)
    if source.kind != "activated_v2":
        raise OperationalError.corrupt_or_partial_install(source.details)
    stamp = load_activation_stamp(cfg.operational_root)
    verify_activation_stamp(stamp, cfg.operational_root)
    return V2OperationalBackend.open_activated(cfg)
```

`LegacyOperationalBackend` is an explicit adapter over the current unmodified
`OperationLedger` plus current operational reducer/query facade. It exposes the
same public methods and `backend_version=1` but never constructs v2
`OperationalViewStore`, `OperationalStore`, roster, or epoch components.
`V2OperationalBackend` owns exactly one v2 ledger/view/store stack and exposes
`backend_version=2`. Both implement the closed `OperationalBackend` protocol;
callers never receive a raw ledger union.

Task 7 writes `operational-v2-activated.json` only after Tasks 1–6 pass. Its
signed digest binds the prepared-migration stamp, every genesis anchor,
reducer/schema versions, canonical session/parity root, event-registry hash,
and Memo runtime version. Presence of v1 plus missing, partial, or corrupt v2
artifacts is never treated as fresh and fails closed.
`V2OperationalBackend.create_empty` first generates the local key,
persists/verifies the signed
one-peer `VerificationRoster.bootstrap`, creates the empty anchor, then writes
the signed epoch-0 marker. A crash between those steps is a partial install and
fails closed on reopen; no component silently recreates trust state.

`Memory` constructs one selected `OperationalBackend`. Only the v2 backend
constructs a view store and `OperationalStore`; the legacy adapter keeps the
existing v1 path intact until migration activation. Federation uses the
selected backend's protocol and validates all v2 anchors/bundles before import
and catches views up afterward. MCP mutations require non-empty idempotency
keys; `server_annotations` maps only typed operational exceptions to
`memo.error.v1`.

- [ ] **Step 4: Run complete ledger gates**

```bash
uv run --no-sync ruff check \
  src/memo/operation_ledger*.py src/memo/operational*.py \
  src/memo/operation_view_schema.py src/memo/operation_views.py \
  src/memo/operation_migration.py src/memo/durable_outbox.py \
  src/memo/error_contract.py src/memo/federation.py src/memo/definitive.py \
  src/memo/memory/facade.py src/memo/memory/outcome_feedback_ops.py \
  tests/test_operation_*.py tests/test_operational_*.py
uv run --no-sync mypy src/memo
uv run --no-sync pytest \
  tests/test_operation_ledger_v1_compat.py \
  tests/test_operational_event_v2.py \
  tests/test_operation_ledger_v2.py \
  tests/test_operation_views_v2.py \
  tests/test_operation_migration_v2.py \
  tests/test_operational_idempotency.py \
  tests/test_durable_outbox.py \
  tests/test_operational_sessions_v2.py \
  tests/test_operational_memory.py \
  tests/test_definitive_memory.py \
  tests/test_write_freeze_gc.py -v
uv run --no-sync pytest -m "not slow" -n auto --timeout=120
```

Expected: all pass. No v1 file changes during the suite.

- [ ] **Step 5: Commit the activated v2 foundation**

```bash
git add \
  src/memo/operation_ledger.py src/memo/memory/facade.py \
  src/memo/federation.py src/memo/server_operational.py \
  src/memo/server_annotations.py src/memo/definitive.py \
  tests/test_operational_memory.py tests/test_definitive_memory.py \
  tests/test_cli_mcp_surface_smoke.py
git commit -m "feat: activate verified operational ledger v2"
```

## Plan Acceptance Gate

- Every v1 journal fixture is byte-identical before and after migration.
- Corrupt, forked, gapped, or changed v1 sources abort before v2 activation.
- Repeated migration and command retries create no duplicate event or memory.
- SQLite rebuild matches the current focus, handoff, attention, conflict, and
  outcome state root.
- Session JSON and the old table are no longer competing authorities.
- `Memory.operational` keeps its public attribute and successful return shapes.
- `LegacyOperationLedger` is reachable only from migration/audit code.
- Focused suites, mypy, ruff, and the full non-slow suite pass.
- No Memflow code, live sync, daemon, or cutover behavior is introduced by this
  plan.
