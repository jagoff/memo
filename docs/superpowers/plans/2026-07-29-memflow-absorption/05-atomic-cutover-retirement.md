# Atomic Memo Activation and Memflow Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Coordinate one signed, observable activation epoch across every
configured Mac, verify Memo-only operation after restart, and irreversibly
retire Memflow without mixed authority or a compatibility fallback.

**Architecture:** An operator-only controller advances a signed cutover attempt
through a monotonic state machine stored in a Git control ref updated by
compare-and-swap. Before commit, every peer is fenced and rollback-capable.
After the epoch CAS, Memflow permanently rejects startup and all repairs move
forward in Memo. Verification and cleanup are separately gated.

**Tech Stack:** Python 3.13+, Plans 01–04, signed operational Git control ref,
Memo/Memflow fence APIs, launchd, structured config staging, pytest, mypy,
ruff. Temporary controller/import/cleanup tools are removed before the final
Memo release tag.

## Global Constraints

- Plans 01–04 and all their acceptance gates are prerequisites.
- The exact peer roster is frozen; any offline, mismatched, or failed peer
  aborts before quiesce.
- There is no dual-write. Clients observe Memflow, a bounded maintenance pause,
  then Memo.
- Every request is fenced; config edits alone do not establish authority.
- Established clients must close/reconnect and submit the committed epoch.
- The control ref is
  `refs/heads/memo-cutover-control` and only the designated controller writes
  it using `--force-with-lease` against the expected OID.
- Manifests, records, votes, heads, anchors, and epoch markers require enrolled
  device signatures. A checksum is not authentication.
- A vote is bound to attempt ID, roster, target commit/runtime/config digests,
  capability manifest, migration report, origin heads/root, and expiry.
- Before `EPOCH_COMMITTED`, a signed ABORT restores every peer globally. At or
  after `EPOCH_COMMITTED`, rollback to Memflow is forbidden.
- No mutating tool accepts a caller-supplied state enum as authority. It accepts
  a freshly fetched signature-verified `VerifiedControlRecord` plus exact OID
  and re-reads that OID immediately before mutation.
- ABORT is two-phase: ABORTING restores exact preimages and starts Memflow
  health-only behind the write fence on every peer; unanimous signed
  restored/health-only receipts permit CAS to ABORTED. Only after ABORTED does
  Memflow start publicly, and unanimous signed active-smoke receipts permit
  client release.
- Coordinator failure before quiesce aborts on lease expiry. At/after quiesce,
  peers stay fenced until a CAS-authorized replacement publishes resume/abort.
- Staging uses a disposable state root, remote, socket, vault clone, clock, and
  terminal presenter; it cannot access production paths.
- No deletion occurs before unanimous `VERIFIED` after real logout/login or
  reboot on every Mac.
- Clients remain closed from QUIESCING until unanimous ACTIVATED (or until
  unanimous healthy ABORTED); none reconnects to an intermediate authority.
- Cleanup targets are exact resolved paths/remote identities from the signed
  inventory. No glob, `$HOME`, `~`, symlink, repository root, or broad recursive
  delete is permitted.
- The Memflow source-code remote is archived read-only. The Memflow operational
  data remote is deleted and not archived.
- No public `flow_*` alias, fallback, sidecar, or mixed mode survives.

---

## File Structure

### Create in Memo

- `tools/memflow_absorption/control.py`
- `tools/memflow_absorption/controller.py`
- `tools/memflow_absorption/peer_agent.py`
- `tools/memflow_absorption/staging.py`
- `tools/memflow_absorption/client_switch.py`
- `tools/memflow_absorption/cleanup_scan.py`
- `tools/memflow_absorption/retire.py`
- `tests/tools/test_absorption_control.py`
- `tests/tools/test_absorption_controller.py`
- `tests/tools/test_absorption_peer_agent.py`
- `tests/tools/test_absorption_staging.py`
- `tests/tools/test_absorption_client_switch.py`
- `tests/tools/test_absorption_cleanup_scan.py`
- `tests/tools/test_absorption_retire.py`
- `docs/runbooks/memflow-atomic-retirement.md`

### Modify in Memo

- `tools/memflow_absorption/__main__.py`
- `tools/memflow_absorption/schemas.py`
- `tools/memflow_absorption/safety.py`
- `tools/memflow_absorption/verify.py`
- `src/memo/cli_doctor.py`
- `src/memo/definitive.py`
- `tests/test_definitive_memory.py`
- `tests/test_cli_doctor.py`

### Final Synapse cleanup

- Delete after verification:
  - `src/synapse/memflow_backend.py`
  - Memflow-only routes/options/adapters identified by Plan 03 inventory
  - `tests/test_memflow_backend.py`
  - `tests/test_memflow_http.py`
  - `tests/test_memflow_inprocess.py`
- Modify Synapse doctor/dashboard/CLI tests to assert Memo-only independence.

## Shared Interfaces

`EpochMarkerAuthorization`, `CutoverState`, `VerifiedControlRecord`, and
canonical signature verification are imported from Plans 01/03. This plan
never redefines the epoch schema, state enum, or cryptography.

The only forward path is `PREPARING → READY → QUIESCING → QUIESCED → STAGED →
ACTIVATION_READY → EPOCH_COMMITTED → ACTIVATED → VERIFIED → RETIRED`.
`ABORTING → ABORTED` may branch only before EPOCH_COMMITTED; it requires the
two-phase unanimous receipts below.

Each peer creates this authorization after its `ActivationVote`.
The activation vote first binds all target digests. The controller then builds
the immutable candidate EPOCH_COMMITTED Git commit after ACTIVATION_READY; no
`ActivationVote` contains or signs that future OID. It publishes the candidate
create-only to the immutable proposal ref. Each peer fetches the candidate,
recomputes its Git OID from canonical commit bytes, verifies its parent equals
the current control OID, and verifies the exact next state, sequence, epoch,
votes, and target hashes before signing Plan 01's canonical
`EpochMarkerAuthorization`. Its `artifact_digests` must contain exactly
`memo_generation`, `memo_config_set`, and `memflow_retired_marker`. The peer
publishes the authorization on its receipt ref. Because authorizations are not
embedded in the candidate commit, there is no hash/signature cycle. The
controller verifies one from every enrolled peer before pushing that exact
candidate to control by CAS. ACTIVATED later records their hashes through
`ActivatedPeerReceipt`. This complete authorization, never a detached envelope,
is the only input accepted by that peer's `EpochFence.activate`.

Exact control signatures:

- `ControlRef.read() -> tuple[str | None, CutoverRecord | None]`
- `ControlRef.compare_and_swap(*, expected_oid: str | None, next_record:
  CutoverRecord) -> str`
- `ProposalRef.publish_once(*, attempt_id: str, epoch: int,
  candidate_commit: bytes, expected_parent_oid: str) -> str`
- `ProposalRef.fetch_and_verify(*, attempt_id: str, epoch: int,
  expected_parent_oid: str) -> tuple[str, CutoverRecord]`
- `CutoverController.prepare(manifest: CutoverManifest) -> CutoverRecord`
- `CutoverController.record_ready(vote: ReadinessVote) -> CutoverRecord`
- `CutoverController.quiesce() -> CutoverRecord`
- `CutoverController.record_quiesced(report: DrainReport) -> CutoverRecord`
- `CutoverController.stage(report: RehearsalReport) -> CutoverRecord`
- `CutoverController.record_activation_ready(vote: ActivationVote) ->
  CutoverRecord`
- `CutoverController.commit_epoch() -> CutoverRecord`
- `CutoverController.record_activated(receipt: ActivatedPeerReceipt) ->
  CutoverRecord`
- `CutoverController.verify(report: GlobalVerificationReport) ->
  CutoverRecord`
- `CutoverController.retire(report: RetirementReport) -> CutoverRecord`
- `CutoverController.begin_abort(control: VerifiedControlRecord) ->
  CutoverRecord`
- `CutoverController.record_restored(vote: RollbackRestoredVote) ->
  CutoverRecord`
- `CutoverController.complete_abort() -> CutoverRecord`
- `CutoverController.record_abort_health(receipt: RollbackHealthyReceipt) ->
  CutoverRecord`
- `CutoverController.record_abort_active_smoke(receipt:
  RollbackActiveSmokeReceipt) -> CutoverRecord`
- `CutoverController.release_clients() -> ClientReleaseReceipt`
- `CutoverController.resume(takeover: SignedTakeover) -> CutoverRecord`

### Task 1: Extend signed records with votes, monotonic state, and CAS control

**Files:**
- Create: `tools/memflow_absorption/control.py`
- Create: `tests/tools/test_absorption_control.py`
- Extend: `tools/memflow_absorption/control_record.py`
- Extend: `tools/memflow_absorption/schemas.py`

**Interfaces:**
- Consumes: Plan 01 Ed25519/Keychain/domain-separation and Plan 03 verified
  record/state schemas.
- Produces: signed `CutoverRecord`, votes, takeover, `ControlRef`.

- [ ] **Step 1: Write failing signature/state/CAS tests**

```python
def test_vote_is_bound_to_attempt_and_all_required_hashes(keys):
    vote = ready_vote(attempt_id="a1", runtime_digest="r1", state_root="s1")
    signed = sign_record(vote, keys.device_a)
    verify_record(signed, roster=keys.roster)
    with pytest.raises(SignatureError):
        verify_record(
            replace(signed, payload=replace(vote, state_root="changed")),
            roster=keys.roster,
        )


def test_state_machine_rejects_skip_and_expired_vote(machine, expired_vote):
    with pytest.raises(ControlError):
        machine.transition(CutoverState.EPOCH_COMMITTED)
    with pytest.raises(ControlError):
        machine.record_ready(expired_vote)


def test_control_ref_uses_compare_and_swap(control_ref):
    first = control_ref.compare_and_swap(
        expected_oid=None, next_record=record(CutoverState.PREPARING)
    )
    with pytest.raises(ControlConflict):
        control_ref.compare_and_swap(
            expected_oid=None, next_record=record(CutoverState.READY)
        )
    second = control_ref.compare_and_swap(
        expected_oid=first, next_record=record(CutoverState.READY)
    )
    assert second != first
```

Cover unauthorized signer, wrong roster, reused vote, expired lease, stale
attempt, two-controller race, invalid backward transition, ABORT rules, and
takeover after quiesce.

- [ ] **Step 2: Run tests and confirm modules are missing**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_control_record.py \
  tests/tools/test_absorption_control.py -v
```

- [ ] **Step 3: Implement canonical signed records and CAS**

Reuse the exact Plan 01 canonical JSON, domain-separated Ed25519 envelope,
Keychain-backed private keys, and versioned roster verification. A record
contains attempt ID, state, controller, lease expiry, previous control OID,
monotonic sequence, roster/version, target digests,
manifest/migration/generation/root hashes, votes, nonces, epoch, and timestamp.
Reject duplicate key IDs/fingerprints, unauthorized roles, revoked keys,
replayed nonces/sequences, and signatures from the wrong domain.

`ControlRef.compare_and_swap` writes a temporary Git commit containing only the
record and runs this exact argument vector:

```python
lease_value = expected_oid or ""
args = [
    "git",
    "push",
    self.remote,
    f"{commit_oid}:refs/heads/memo-cutover-control",
    "--force-with-lease="
    f"refs/heads/memo-cutover-control:{lease_value}",
]
subprocess.run(args, cwd=self.checkout, check=True, text=True)
```

Use an absent-ref lease form for the first record. Treat any rejection as a
control conflict; never retry with a broadened force.

- [ ] **Step 4: Run control gates**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_control_record.py \
  tests/tools/test_absorption_control.py -v
uv run --no-sync mypy \
  tools/memflow_absorption/control_record.py \
  tools/memflow_absorption/control.py
uv run --no-sync ruff check \
  tools/memflow_absorption/control_record.py \
  tools/memflow_absorption/control.py \
  tests/tools/test_absorption_control_record.py \
  tests/tools/test_absorption_control.py
```

- [ ] **Step 5: Commit signed control plane**

```bash
git add \
  tools/memflow_absorption/control.py \
  tools/memflow_absorption/control_record.py \
  tools/memflow_absorption/schemas.py \
  tests/tools/test_absorption_control_record.py \
  tests/tools/test_absorption_control.py
git commit -m "feat: add signed cutover control plane"
```

### Task 2: Implement the fail-closed controller and disposable staging

**Files:**
- Create: `tools/memflow_absorption/controller.py`
- Create: `tools/memflow_absorption/peer_agent.py`
- Create: `tools/memflow_absorption/staging.py`
- Create: `tests/tools/test_absorption_controller.py`
- Create: `tests/tools/test_absorption_peer_agent.py`
- Create: `tests/tools/test_absorption_staging.py`
- Extend: `tools/memflow_absorption/__main__.py`

**Interfaces:**
- Produces: `CutoverController`; operator commands `prepare`, `ready`,
  `quiesce`, `stage`, `activation-ready`, `commit`, `verify`, `abort`, `resume`.
  It also produces read-only `approved-attempt --print-path`, which succeeds
  only when exactly one signed, unexpired attempt has passed Plans 03–04.
- `PeerTransport.submit(request: PeerRequest) -> str`
- `PeerTransport.wait_receipt(*, request_id: str, deadline: datetime) ->
  PeerReceipt`
- `PeerAgent.execute(request: PeerRequest) -> PeerReceipt`

- [ ] **Step 1: Write failing controller/recovery tests**

```python
def test_commit_requires_unanimous_fresh_activation_votes(controller, two_peers):
    controller.prepare(manifest(two_peers))
    controller.record_ready(ready_vote(two_peers[0]))
    controller.record_ready(ready_vote(two_peers[1]))
    controller.quiesce()
    controller.record_quiesced(clean_drain(two_peers))
    controller.stage(clean_rehearsal(two_peers))
    controller.record_activation_ready(activation_vote(two_peers[0]))
    with pytest.raises(ControlError, match="unanimous"):
        controller.commit_epoch()


def test_controller_loss_after_quiesce_remains_fenced(controller, clock):
    controller.advance_to(CutoverState.QUIESCED)
    clock.advance(past_lease=True)
    assert controller.peer_action() == "remain_fenced"
    takeover = signed_takeover(expected_oid=controller.control_oid)
    controller.resume(takeover)
    assert controller.state == CutoverState.QUIESCED


def test_abort_is_two_phase_and_clients_stay_closed(controller, two_peers):
    controller.advance_to(CutoverState.QUIESCED)
    controller.begin_abort(controller.verified_record())
    for peer in two_peers:
        controller.record_restored(restored_vote(peer, memflow_fenced=True))
    for peer in two_peers:
        controller.record_abort_health(
            peer.start_memflow_abort_healthcheck_and_sign_receipt()
        )
    aborted = controller.complete_abort()
    assert aborted.state == CutoverState.ABORTED
    assert controller.clients_released is False
    for peer in two_peers:
        peer.install_active_marker(aborted)
        peer.start_memflow_public()
        controller.record_abort_active_smoke(peer.run_active_smoke_and_sign())
    assert controller.release_clients().released is True
```

Also test pre-quiesce lease abort, wrong report hash, state-root mismatch, abort
restoration requirement, stale control record, request nonce replay, peer
timeout, wrong-device receipt, health receipt before restored vote, ABORTED
before unanimous health, client release before unanimous active smoke,
request/receipt ref ownership, cross-device write rejection, monotonic ref
sequence, immutable proposal create-only behavior, peer-side commit OID/parent/
payload recomputation, proposal/control OID identity, stale force-with-lease
rejection, commit-once, and post-commit abort rejection.

- [ ] **Step 2: Run controller/staging tests**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_controller.py \
  tests/tools/test_absorption_peer_agent.py \
  tests/tools/test_absorption_staging.py -v
```

- [ ] **Step 3: Implement orchestration with explicit peer adapters**

The controller never shells into a host. `PeerTransport` publishes signed,
immutable requests to the dedicated control remote and waits for signed
receipts; a temporary local `PeerAgent` on each enrolled Mac polls only its
device queue. `PeerRequest` binds schema, attempt ID, fresh control OID,
monotonic request sequence, nonce, target device, one allowlisted action,
payload hash, issued time, deadline, and controller signature. `PeerReceipt`
binds the request hash/nonce, device/key, control OID observed immediately
before action, status, before/after digests, timestamps, stdout digest (not
payload), and signature. Wrong device/OID/nonce, expiry, transport error, or
timeout fails closed and leaves the peer fenced.

Transport ownership is fixed and enforced by both Git credentials and
signature roles:

```text
refs/heads/memo-cutover-requests/<attempt-id>/<device-id>
refs/heads/memo-cutover-receipts/<attempt-id>/<device-id>
refs/heads/memo-cutover-proposals/<attempt-id>/epoch-<epoch>
refs/heads/memo-cutover-control
```

Only the controller role may advance request refs and the control ref. A device
role may advance only its own receipt ref; it can never write requests,
another device's receipts, or control. Each request/receipt commit adds one
immutable file named `<zero-padded-sequence>-<record-sha256>.json`, requires the
next monotonic sequence, and advances its ref with
`--force-with-lease=<ref>:<expected-oid>`. The first write uses the absent-ref
lease form. Rejection is final for that action; no broadened force or shared
branch fallback exists.
Only the controller may create the epoch proposal ref, exactly once with the
absent-ref lease form; it is immutable thereafter. Peers have read-only access
and sign only after fetching its full commit/parent and validating the
candidate `CutoverRecord`. The final control CAS must push that identical
proposal OID, not rebuild an equivalent commit.

The action enum is exactly `STATUS`, `INSTALL_FENCE`, `DRAIN`,
`SNAPSHOT_FINAL`, `IMPORT_FINAL`, `INSTALL_GENERATION`,
`ACTIVATE_PEER_BUNDLE`, `START_MEMO`, `START_MEMFLOW_FENCED`, `VERIFY`,
`RESTORE_PREIMAGE`, `INSTALL_SYNAPSE_RUNTIME`, `SWITCH_SYNAPSE_RUNTIME`,
`RESTART_SYNAPSE_EXACT`, and `RETIRE_EXACT_TARGETS`. Each enrolled peer adapter
returns signed status for Memo/Memflow runtime, fence, drain, snapshot, import,
stage, and verification. Staging launches:

```python
args = [
    "memo-operational-daemon",
    "--mode", "staging",
    "--state-dir", str(attempt.root / "staging" / peer.device_id),
    "--remote", str(attempt.disposable_remote),
    "--epoch", str(attempt.candidate_epoch),
    "--public", "off",
]
```

Require a cloned vault, temp operational remote, private loopback endpoint,
fake terminal presenter, injected clock, and exact `production_paths_touched=()`
before accepting `ACTIVATION_READY`.

Pre-epoch abort keeps the control state ABORTING while every peer restores its
preimage and returns `RollbackRestoredVote`. Only then may
`START_MEMFLOW_FENCED` invoke Plan 03's private `abort_healthcheck`; it proves
read-only health without a listener, loops, writer, or mutations and returns
`RollbackHealthyReceipt`. `complete_abort()` requires unanimous restored and
healthy receipts before its CAS to ABORTED. Each peer may then install the
newer signed ACTIVE marker, start public Memflow, and return a
`RollbackActiveSmokeReceipt` covering listener, read/write round trip, sync,
and runtime/config hashes. Clients remain closed until all active-smoke
receipts match the ABORTED OID and `release_clients()` records its own signed
receipt.

- [ ] **Step 4: Run controller/failure gates**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_controller.py \
  tests/tools/test_absorption_peer_agent.py \
  tests/tools/test_absorption_staging.py \
  tests/tools/test_absorption_rehearsal.py \
  tests/tools/test_absorption_rollback_bundle.py -v
uv run --no-sync mypy \
  tools/memflow_absorption/controller.py \
  tools/memflow_absorption/peer_agent.py \
  tools/memflow_absorption/staging.py
uv run --no-sync ruff check \
  tools/memflow_absorption/controller.py \
  tools/memflow_absorption/peer_agent.py \
  tools/memflow_absorption/staging.py \
  tests/tools/test_absorption_controller.py \
  tests/tools/test_absorption_peer_agent.py \
  tests/tools/test_absorption_staging.py
```

- [ ] **Step 5: Commit controller/staging**

```bash
git add \
  tools/memflow_absorption/controller.py \
  tools/memflow_absorption/peer_agent.py \
  tools/memflow_absorption/staging.py \
  tools/memflow_absorption/__main__.py \
  tests/tools/test_absorption_controller.py \
  tests/tools/test_absorption_peer_agent.py \
  tests/tools/test_absorption_staging.py
git commit -m "feat: coordinate fail-closed Memo activation"
```

### Task 3: Build atomic client switching and pre-epoch rollback proof

**Files:**
- Create: `tools/memflow_absorption/client_switch.py`
- Create: `tests/tools/test_absorption_client_switch.py`
- Extend: `tests/tools/test_absorption_controller.py`
- Create: `docs/runbooks/memflow-atomic-retirement.md`

**Interfaces:**
- Consumes: Plan 03 staged configs and Plan 04 rollback bundle.
- Produces: `ActivationSwitchPlan.apply/verify`,
  `RollbackRestorePlan.prepare/verify`, and exact operator runbook.
- `ActivationSwitchPlan.apply(*, control: VerifiedControlRecord,
  expected_control_oid: str, epoch_authorization: EpochMarkerAuthorization) ->
  ActivatedPeerReceipt` re-fetches/verifies that OID immediately before
  starting/resuming the journaled peer activation transaction.

- [ ] **Step 1: Write failing config-switch/rollback tests**

```python
def test_switch_requires_committed_epoch_and_matching_live_digest(switch_plan):
    with pytest.raises(SafetyError):
        switch_plan.apply(
            control=verified_record(CutoverState.ACTIVATION_READY),
            expected_control_oid="oid-ready",
            epoch_authorization=switch_plan.authorization_for("oid-ready"),
        )
    mutate_live_file(switch_plan.targets[0])
    with pytest.raises(SafetyError, match="digest"):
        switch_plan.apply(
            control=verified_record(CutoverState.EPOCH_COMMITTED),
            expected_control_oid="oid-committed",
            epoch_authorization=switch_plan.authorization_for("oid-committed"),
        )


def test_switch_installs_memo_epoch_marker_bound_to_receipt(switch_plan):
    receipt = switch_plan.apply(
        control=verified_record(CutoverState.EPOCH_COMMITTED, epoch=7),
        expected_control_oid="oid-committed",
        epoch_authorization=switch_plan.authorization_for("oid-committed"),
    )
    marker = switch_plan.epoch_fence.read()
    assert marker.epoch == 7
    assert marker.control_oid == "oid-committed"
    assert receipt.memo_epoch_marker_sha256 == sha256_file(marker.path)


def test_pre_epoch_abort_restores_every_component(controller, rollback_bundle):
    controller.advance_to(CutoverState.QUIESCED)
    controller.begin_abort(controller.verified_record())
    report = rollback_bundle.restore(control=controller.verified_record())
    vote = controller.record_restored(restored_vote(report))
    assert vote.restored == {
        "runtime", "plists", "launchd_state", "consumer_configs",
        "memflow_git_refs", "local_state", "cursors", "roster", "epochs"
    }
    assert report.post_restore_hashes_match is True
```

Cover file mode/owner preservation, fsync+rename, generation-pointer digest,
already-switched idempotency, forged/stale enum rejection, control OID changing
between check/mutation, established connection blocker, clients remaining
closed through CAS, restart acknowledgement only after activation, and
post-commit restore rejection. Also cover absent/corrupt/stale Memo epoch
marker, mismatch between marker and ActivatedPeerReceipt, and failure after
every individual target replacement. On restart, a valid transaction journal
must resume forward to the exact postimage before any runtime/client starts;
wrong pre/post digest or missing journal fails closed.

- [ ] **Step 2: Run client switch/rollback tests**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_client_switch.py \
  tests/tools/test_absorption_controller.py \
  tests/tools/test_absorption_rollback_bundle.py -v
```

- [ ] **Step 3: Implement exact staged replacement and runbook**

`ACTIVATE_PEER_BUNDLE` is one idempotent, recoverable transaction, not several
independent switch actions. It first verifies every original/staged
digest/mode, parent directory, the freshly fetched signature-verified
EPOCH_COMMITTED OID, and the peer's `EpochMarkerAuthorization`. It writes and
fsyncs `activation-transaction.json` with exact ordered targets, pre/post
digests, and state `PREPARED`. For each target it writes a sibling temp file,
fsyncs, preserves mode, renames, fsyncs the directory, and atomically records
that target's completion bitmap. The fixed order is: permanent Memflow RETIRED
marker, configs, Memo generation pointer, then
`EpochFence.activate(authorization=epoch_authorization,
observed_artifact_digests=fresh_local_activation_digests())`. The fence
reconstructs/verifies the full signed payload, exact control OID and all three
local digests, then persists its authorization hash. It finally records/fsyncs
`COMMITTED` and emits the receipt.

There is no claim that multiple filesystem renames are physically atomic.
Atomic authority comes from the single control CAS plus fail-closed recovery:
from EPOCH_COMMITTED onward Memflow is RETIRED, all clients remain closed, and
Memo/Synapse startup refuses while the peer transaction is absent, PREPARED,
incomplete, corrupt, or mismatched. A restarted peer agent can only resume the
signed postimage forward; it cannot roll back. Thus a crash may leave staged
files at mixed physical versions, but no active process can observe them as an
authority. `ActivatedPeerReceipt` binds the committed transaction hash,
completion bitmap, Memo epoch-marker SHA-256, generation pointer, config-set
digest, RETIRED marker hash, and observed control OID. The switch report
requires each registered client/shell to remain closed until all peers have
returned matching signed receipts; an established connection to port 18766 is
a blocker.

The runbook must spell out this exact sequence:

1. unanimous PREPARING/READY;
2. disable Memflow KeepAlive without deleting rollback plist;
3. close clients, publish QUIESCING, and keep clients stopped;
4. signed zero-writer drain on both Macs;
5. signed post-drain final snapshot, origin vector/root, delta import, and
   fsynced inactive production generation on both Macs;
6. stop Memflow and prove zero PID/listener/connection/handle;
7. disposable staging and unanimous ACTIVATION_READY;
8. publish the post-ACTIVATION_READY candidate commit once on its immutable
   proposal ref, have every peer recompute/validate it and sign an
   `EpochMarkerAuthorization`, pre-stage the permanent RETIRED and Memo epoch
   markers bound to that OID, then push that identical commit by the single CAS;
9. run/resume the journaled `ACTIVATE_PEER_BUNDLE` to its signed committed
   postimage on each peer while all runtimes and clients remain closed;
10. start Memo operational and isolated Synapse, collect signed ACTIVATED
    receipts, and CAS to ACTIVATED;
11. reconnect clients only after unanimous ACTIVATED; repair forward only.

- [ ] **Step 4: Run switch tests and markdown checks**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_client_switch.py \
  tests/tools/test_absorption_controller.py -v
uv run --no-sync mypy tools/memflow_absorption/client_switch.py
uv run --no-sync ruff check \
  tools/memflow_absorption/client_switch.py \
  tests/tools/test_absorption_client_switch.py
git diff --check -- docs/runbooks/memflow-atomic-retirement.md
```

- [ ] **Step 5: Commit switch/runbook**

```bash
git add \
  tools/memflow_absorption/client_switch.py \
  tests/tools/test_absorption_client_switch.py \
  tests/tools/test_absorption_controller.py \
  docs/runbooks/memflow-atomic-retirement.md
git commit -m "feat: stage atomic Memo client activation"
```

### Task 4: Implement global verification and permanent independence audit

**Files:**
- Create: `tools/memflow_absorption/cleanup_scan.py`
- Create: `tests/tools/test_absorption_cleanup_scan.py`
- Modify: `tools/memflow_absorption/verify.py`
- Modify: `src/memo/cli_doctor.py`
- Modify: `src/memo/definitive.py`
- Extend: `tests/test_cli_doctor.py`
- Extend: `tests/test_definitive_memory.py`

**Interfaces:**
- Produces: signed `GlobalVerificationReport`,
  `memo doctor --strict-runtime --operational --attempt-file PATH --json`, and
  permanent `independence_audit`.

- [ ] **Step 1: Write failing verification/negative-scan tests**

```python
def test_verification_requires_both_peers_same_runtime_root_and_epoch(reports):
    reports["device-b"] = replace(reports["device-b"], epoch=8)
    out = verify_global(reports, expected_epoch=7)
    assert out.verified is False
    assert out.blockers == ("device-b:epoch",)


def test_independence_scan_finds_runtime_not_historical_docs(scan_root):
    add_executable_config(scan_root, "hook.sh", "exec /tmp/memflow")
    add_historical_doc(scan_root, "history.md", "Memflow was retired")
    out = independence_audit(scan_root)
    assert "hook.sh" in out.executable_references
    assert "history.md" not in out.executable_references
```

Cover PIDs, listeners, connections, launchd loaded/disabled definitions,
handles, binaries, shims, env/PATH, MCP registrations, hooks, Synapse
executable paths, origin roots, ACK/cursor/lease/session parity, writer lease,
and post-reboot evidence.

- [ ] **Step 2: Run verification/doctor tests**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_cleanup_scan.py \
  tests/tools/test_absorption_verify.py \
  tests/test_cli_doctor.py tests/test_definitive_memory.py -v
```

- [ ] **Step 3: Implement signed verification and permanent scan logic**

`GlobalVerificationReport` includes control OID/epoch, runtime/config digests,
active root and origin heads, pending ACK/cursors/leases/sessions, duplicate/loss
counts, writer lease, daemon/sync health, client connections, restart/reboot
evidence, the exact `ActivatedPeerReceipt` set, and per-peer signatures.
`VERIFIED` requires control state ACTIVATED and unanimous exact values.

Move reusable negative-scan logic into `definitive.independence_audit`; the
temporary cleanup tool only adds attempt inventory and deletion eligibility.
Historical docs and archived source are allowed; executable/config/runtime
references are not.

- [ ] **Step 4: Run full pre-retirement verification gates**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_*.py \
  tests/test_cli_doctor.py tests/test_definitive_memory.py -v
uv run --no-sync mypy tools/memflow_absorption src/memo
uv run --no-sync ruff check \
  tools/memflow_absorption src/memo/cli_doctor.py src/memo/definitive.py \
  tests/tools/test_absorption_cleanup_scan.py \
  tests/test_cli_doctor.py tests/test_definitive_memory.py
```

- [ ] **Step 5: Commit verification**

```bash
git add \
  tools/memflow_absorption/cleanup_scan.py \
  tools/memflow_absorption/verify.py \
  src/memo/cli_doctor.py src/memo/definitive.py \
  tests/tools/test_absorption_cleanup_scan.py \
  tests/tools/test_absorption_verify.py \
  tests/test_cli_doctor.py tests/test_definitive_memory.py
git commit -m "feat: verify Memo independence globally"
```

### Task 5: Execute the rehearsed activation epoch

**Files:**
- Modify: signed attempt artifacts only under the validated attempt root.
- No source change is expected in this task.

**Interfaces:**
- Consumes: exact runbook, target releases, manifests, rollback bundle,
  controller, staged configs.
- Produces: committed epoch and unanimous `VERIFIED`.

- [ ] **Step 1: Create a fresh attempt and run PREPARING checks**

Run from the clean Memo execution worktree:

```bash
cutover_attempt_file="$(
  uv run --no-sync python -m tools.memflow_absorption \
    approved-attempt --print-path
)"
test -f "$cutover_attempt_file"
uv run --no-sync python -m tools.memflow_absorption \
  prepare --attempt-file "$cutover_attempt_file" --dry-run
```

`approved-attempt` must return exactly one signed, unexpired attempt whose
manifests, roster, target commit/runtime, migration plan, and rollback receipt
passed Plans 03–04. Review the emitted paths and blockers. Then repeat the same
command with `--apply` only when the dry-run reports both Macs READY and the
attempt-root sentinel exists.

- [ ] **Step 2: Quiesce and prove drain, retaining rollback**

```bash
cutover_attempt_file="$(
  uv run --no-sync python -m tools.memflow_absorption \
    approved-attempt --print-path
)"
uv run --no-sync python -m tools.memflow_absorption \
  quiesce --attempt-file "$cutover_attempt_file" --apply
```

Expected signed report: both peers QUIESCED, zero writers/locks/unpushed commits,
matching origin vectors/remote ref, Memflow KeepAlive disabled, every client
closed and still stopped, and complete encrypted rollback bundle. Any mismatch
runs the two-phase signed ABORTING → ABORTED path; no client is released until
both fenced Memflow instances have returned health-only receipts, ABORTED is
committed, both public instances have passed signed active smoke, and the
client-release receipt is committed.

- [ ] **Step 3: Import final delta and obtain ACTIVATION_READY**

```bash
cutover_attempt_file="$(
  uv run --no-sync python -m tools.memflow_absorption \
    approved-attempt --print-path
)"
uv run --no-sync python -m tools.memflow_absorption \
  stage --attempt-file "$cutover_attempt_file" --apply
```

The controller first issues `SNAPSHOT_FINAL` only after both zero-drain reports.
It creates one signed `FinalFenceProof` bound to the immutable post-drain remote
OID and every origin head, builds a new final `MigrationPlan`, applies only that
delta, verifies roots, and installs/fsyncs the exact inactive generation on each
Mac. Each fresh activation vote binds the FinalFenceProof, final plan/import
receipt, generation/vault/ledger roots, configs, runtime, manifest, roster, and
candidate epoch. Only the later, separate `EpochMarkerAuthorization` binds the
immutable EPOCH_COMMITTED proposal OID.

Expected: identical final inactive generation roots, zero loss/duplicates,
fake terminal only, and fresh unanimous activation votes. Any item beyond the
signed vector, unknown schema, or hash mismatch blocks ACTIVATION_READY. If
blocked, invoke the controller's two-phase abort with the encrypted rollback
receipt.

- [ ] **Step 4: Commit once, switch clients, and verify after reboot**

```bash
cutover_attempt_file="$(
  uv run --no-sync python -m tools.memflow_absorption \
    approved-attempt --print-path
)"
uv run --no-sync python -m tools.memflow_absorption \
  commit --attempt-file "$cutover_attempt_file" --apply
```

Before CAS, every peer has fsynced a permanent Memflow RETIRED marker bound to
the candidate control commit OID and a signed Memo epoch marker bound to the
same epoch/OID, but has activated neither. The `commit` command performs the
single CAS, runs/resumes each peer's fail-closed `ACTIVATE_PEER_BUNDLE`
transaction to its committed postimage, starts Memo operational/Synapse only
after that peer transaction is COMMITTED, and collects signed
`ActivatedPeerReceipt`. It advances to
ACTIVATED only after both receipts match. After successful CAS, do not run
abort; clients reconnect only after unanimous ACTIVATED. Then run:

```bash
memo doctor --strict-runtime --operational \
  --attempt-file "$cutover_attempt_file" --json
```

Perform real logout/login or reboot on each Mac one at a time, wait for Memo
health, rerun doctor and the Memo-to-Memo send/delivery/ACK/presence/continuity
smoke, then submit both signed verification reports to controller `verify`.
Memflow startup validates the committed control ref before any listener/loop
and permanently refuses because the RETIRED marker is monotonic.

- [ ] **Step 5: Record the operational result**

Store only signed manifests/reports/hashes under the attempt reports directory.
Do not commit secrets, rollback contents, user payloads, `.memflow`, or generated
config files to Git. Immediately after unanimous VERIFIED, destroy every local
encrypted rollback bundle and wrapped data key, fsync the directories, and add
only signed destruction receipts to the report. Verify the attempt root now
contains only the explicit allowlist of signed manifests, hashes, reports, and
destruction receipts. The controller state must be `VERIFIED` before Task 6.

### Task 6: Retire Memflow and remove temporary migration machinery

**Files:**
- Create: `tools/memflow_absorption/retire.py`
- Create: `tests/tools/test_absorption_retire.py`
- Delete temporary tools/tests listed below after successful live retirement.
- Delete final Synapse Memflow backend/routes/tests.

**Interfaces:**
- Produces: signed `RetirementReport`, final Memo-only code and runtime.
- Consumes: Plan 03 signed `SynapseRetirementManifest`; no ad hoc Synapse
  deletion target is permitted.
- `RetirementPlanner.plan(*, control: VerifiedControlRecord,
  expected_control_oid: str, inventory: SignedInventory) -> RetirementPlan`
- `RetirementExecutor.apply(plan: RetirementPlan, *, control:
  VerifiedControlRecord, expected_control_oid: str) -> RetirementReport`
  re-fetches/verifies the OID and every target/remote identity immediately
  before mutation.
- `RemoteProvider.compare_and_delete(*, provider: str, account_id: str,
  repository_id: str, expected_head: str) -> RemoteDeletionReceipt`
- `RemoteProvider.make_read_only(*, provider: str, account_id: str,
  repository_id: str, expected_head: str) -> RemoteArchiveReceipt`

- [ ] **Step 1: Write failing exact-target cleanup tests**

```python
def test_retire_requires_verified_state_and_exact_inventory(retirer):
    with pytest.raises(SafetyError):
        retirer.plan(
            control=verified_record(CutoverState.EPOCH_COMMITTED),
            expected_control_oid="oid-committed",
            inventory=signed_inventory(),
        )
    report = retirer.plan(
        control=verified_record(CutoverState.VERIFIED),
        expected_control_oid="oid-verified",
        inventory=signed_inventory(),
    )
    assert report.targets == signed_inventory_targets()
    assert not report.contains_broad_or_symlink_target


def test_data_remote_deleted_but_code_remote_archived(remote_stub, retirer):
    report = retirer.apply(
        exact_retirement_plan(remote_stub),
        control=verified_record(CutoverState.VERIFIED),
        expected_control_oid="oid-verified",
    )
    assert remote_stub["memflow-data"].exists is False
    assert remote_stub["memflow-code"].archived is True
    assert remote_stub["memflow-code"].read_only is True
```

Cover wrong remote identity, process/listener still live, Synapse reference,
missing final tag, rollback data still required, idempotent rerun, and permanent
independence audit. Also cover stale/forged control state, control OID change,
provider/repository ID mismatch, expected-HEAD mismatch, residual rollback
bundle/key, source-history secret/payload detection, Synapse commit drift,
unlisted Memflow symbols/tests/goldens, and a residual active reference after
the manifest-driven deletion. Also cover Synapse build/tag mismatch, one peer
remaining on the old runtime pointer, installed-runtime residual references,
wrong LaunchAgent digest, and missing or forged `SynapseActivatedReceipt`.

- [ ] **Step 2: Implement retirement planner/executor**

The signed inventory names each remote by provider, account, immutable
repository ID, URL, purpose, and expected HEAD. The executor first performs a
full-history path/content/secret scan of the exact Memflow source commit and
blocks archival if any `.memflow`, credentials, generated config, operational
payload, or data-remote object exists. It tags the exact commit, makes that
repository read-only, and verifies the provider state. Deleting the operational
data remote requires repository-ID plus expected-HEAD compare-and-delete; URL
or display name alone is insufficient.

It then removes only signed-inventory targets:
package/binaries, LaunchAgent and backups, MCP registrations, hooks, permissions,
skills, shell env/shims, logs/caches/cursors, `.memflow`, checkout/venv, and the
operational-data remote. Each target is re-resolved immediately before action;
any drift aborts the remaining cleanup. No operational payload is copied into
the audit report.

In Synapse, verify the current commit and full reference-scan digest against
the signed `SynapseRetirementManifest`, then delete exactly its listed
Memflow backend, `/memflow` routes, target options, symbols, tests, and goldens.
Run a full negative source/test scan plus Memo-only suites and commit in
Synapse. Any unlisted or surviving active reference aborts retirement and
requires a newly signed Plan 03 manifest.

The cleaned source commit is not sufficient evidence. Build a reproducible,
versioned Synapse runtime from that exact commit, record its wheel/runtime and
negative-scan digests, tag the cleaned Synapse release, and install the same
inactive runtime generation on both Macs through Plan 03's Synapse runtime
installer. Each peer action atomically advances the signed Synapse runtime
pointer, restarts only the exact inventoried Synapse LaunchAgents, and returns
a signed `SynapseActivatedReceipt` bound to commit, tag, wheel/runtime digest,
pointer digest, LaunchAgent config digests, and post-start negative scan. The
retirement control/report cannot complete until every configured Mac reports
the same Synapse digest and no installed runtime contains a Memflow
file/symbol/route. These three peer actions are authorized only from VERIFIED
and are included in the signed retirement plan.

- [ ] **Step 3: Run retirement tests before live apply**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_retire.py \
  tests/tools/test_absorption_cleanup_scan.py \
  tests/test_definitive_memory.py -v
uv run --no-sync mypy tools/memflow_absorption/retire.py
uv run --no-sync ruff check \
  tools/memflow_absorption/retire.py \
  tests/tools/test_absorption_retire.py
```

Then run `retire --dry-run` with exact attempt/control/report hashes. Apply only
when it lists no unexpected target and control state is unanimously VERIFIED.

- [ ] **Step 4: Remove temporary cutover tooling before final Memo tag**

After the signed retirement report and a second reboot independence scan, move
the generic permanent independence logic into `memo doctor`/`definitive`, then
delete the complete temporary package, all dedicated tests, and migration
fixtures:

```text
tools/memflow_absorption/__init__.py
tools/memflow_absorption/__main__.py
tools/memflow_absorption/schemas.py
tools/memflow_absorption/safety.py
tools/memflow_absorption/snapshot.py
tools/memflow_absorption/manifest.py
tools/memflow_absorption/inventory.py
tools/memflow_absorption/control_record.py
tools/memflow_absorption/config_stage.py
tools/memflow_absorption/source_memo.py
tools/memflow_absorption/source_memflow.py
tools/memflow_absorption/durable_import.py
tools/memflow_absorption/active_import.py
tools/memflow_absorption/importer.py
tools/memflow_absorption/rollback_bundle.py
tools/memflow_absorption/verify.py
tools/memflow_absorption/rehearsal.py
tools/memflow_absorption/control.py
tools/memflow_absorption/controller.py
tools/memflow_absorption/peer_agent.py
tools/memflow_absorption/staging.py
tools/memflow_absorption/client_switch.py
tools/memflow_absorption/cleanup_scan.py
tools/memflow_absorption/retire.py
tests/tools/test_absorption_safety.py
tests/tools/test_absorption_snapshot.py
tests/tools/test_absorption_manifest.py
tests/tools/test_absorption_inventory.py
tests/tools/test_absorption_control_record.py
tests/tools/test_absorption_config_stage.py
tests/tools/test_absorption_source_memo.py
tests/tools/test_absorption_source_memflow.py
tests/tools/test_absorption_durable_import.py
tests/tools/test_absorption_active_import.py
tests/tools/test_absorption_rollback_bundle.py
tests/tools/test_absorption_verify.py
tests/tools/test_absorption_rehearsal.py
tests/tools/test_absorption_control.py
tests/tools/test_absorption_controller.py
tests/tools/test_absorption_peer_agent.py
tests/tools/test_absorption_staging.py
tests/tools/test_absorption_client_switch.py
tests/tools/test_absorption_cleanup_scan.py
tests/tools/test_absorption_retire.py
```

No importer, source scanner, durable/active import, verifier, rehearsal,
rollback, controller, peer agent, staging, switch, retirement, manifest, or
cutover fixture remains. Fixture files are enumerated individually by the
signed inventory and checked against the directory contents before deletion;
an unexpected file blocks cleanup. Permanent runtime verification remains only
in `memo doctor` and `definitive`.

- [ ] **Step 5: Run final suites, commit cleanup, and tag Memo**

```bash
uv run --no-sync ruff check src/memo tests
uv run --no-sync mypy src/memo
uv run --no-sync pytest -m "not slow" -n auto --timeout=120
uv run --no-sync pytest -m "slow" --timeout=300 -v
memo doctor --strict-runtime --operational --json
```

Expected independence scan: zero Memflow PID, listener, connection, LaunchAgent,
hook, binary, environment route, MCP registration, Synapse executable
reference, local state, or operational-data remote. Historical design/runbook
references and the read-only source archive are allowed.

Commit exact removed/retained source paths, synchronize Memo release metadata,
tag the final Memo release, deploy that exact tag and the exact cleaned Synapse
tag to both Macs, and verify the same Memo runtime digest/epoch and Synapse
runtime digest again. Then reboot both Macs again from the deployed final tags
and rerun independence audit (including installed Synapse runtime trees),
epoch/root parity, daemon health, Synapse doctor/dashboard health, and
Memo-to-Memo send/delivery/ACK/presence/continuity smoke before declaring the
program complete.

## Plan Acceptance Gate

- The control state advanced by signed CAS with fresh unanimous votes.
- Every peer was fenced, drained, staged, and activated under one attempt.
- Before commit, injected failures restored the complete rollback bundle.
- After commit, stale Memo requests and all Memflow startup/writes were rejected.
- Both Macs passed Memo-to-Memo behavior and strict doctor after real restart.
- Both Macs run the same cleaned, tagged Synapse runtime digest, and Synapse
  plus every client runs without a Memflow runtime/backend/route.
- The Memflow package, daemon, binaries, configs, hooks, shims, state, checkout,
  venv, logs, caches, and operational-data remote are absent.
- The Memflow code remote is tagged, read-only, and contains no operational data.
- Temporary importer/controller/rollback/retirement code is absent from the
  final Memo release.
- Only signed hashes/counts/invariants/reports remain; no discarded operational
  payload archive exists.
