# Memflow Cutover Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing environment safely quiesceable and fully
inventoried by adding Memflow request fencing/drain, isolating Synapse, freezing
the live-use manifest, and generating staged Memo-only consumer configurations
without changing the active runtime.

**Architecture:** Operator-only tools in the Memo repository read immutable
snapshots and produce signed/checksummed manifests under a per-attempt state
root. A temporary Memflow cutover gate fences every write boundary and measures
drain. Synapse moves to its own runtime and backend registry. Consumer files are
parsed and rendered into staging, never edited in place by this plan.

**Tech Stack:** Python 3.13+, Memo and Memflow existing CLI/FastMCP/launchd
patterns, Synapse package, stdlib JSON/TOML/plist/path/process inspection,
pytest, mypy, ruff. No service mutation from tests.

## Global Constraints

- This plan is reversible and does not quiesce production, publish an activation
  epoch, replace live user configuration, stop services, or delete anything.
- Operator tools are dry-run by default and are not installed as Memo public
  commands.
- Attempt artifacts live only under
  `~/.local/state/memo/cutover/<attempt-id>/`.
- Every apply-capable command requires an explicit attempt ID, exact manifest
  SHA-256, sentinel file, and resolved path under the attempt root.
- Reject `/`, a home directory, repository root, symlink target, unresolved
  environment variable, or path outside the attempt root.
- Usage baselines come from read-only snapshots, never live MCP calls, because
  Memflow instruments its own usage queries.
- The admission window is exactly `[frozen_at - 90 days, frozen_at]` in UTC on
  both Macs. A manifest cannot become frozen while either Mac has a coverage
  gap, an unfresh source receipt, or ambiguous traffic without classified
  evidence.
- Exclude audit, `test-wire`, smoke, synthetic benchmark, and `*-eval` traffic.
- Ambiguous traffic is provisionally admitted during analysis, but Plan 05 may
  consume only a signed frozen manifest where every row is resolved. Complete
  zero-use evidence yields `delete`, never speculative absorption.
- Memflow `ACTIVE` behavior remains unchanged until a signed `QUIESCING` or
  `RETIRED` marker is explicitly installed by Plan 05.
- Synapse must have its own runtime before its backend changes. Do not create a
  `NullMemflowBackend`.
- Consumer transformers preserve unknown fields and file modes; no `sed` or
  regex rewrite of TOML/JSON/plist.
- Plan 02 must pass before activating Synapse `mode="memo"`; registry and tests
  may be built earlier with a fake `OperationalBackend`.
- Stage explicit paths and commit separately in Memo, Memflow, and Synapse.

---

## File Structure

### Memo repository: create

- `tools/memflow_absorption/__init__.py`
- `tools/memflow_absorption/__main__.py`
- `tools/memflow_absorption/schemas.py`
- `tools/memflow_absorption/safety.py`
- `tools/memflow_absorption/snapshot.py`
- `tools/memflow_absorption/manifest.py`
- `tools/memflow_absorption/inventory.py`
- `tools/memflow_absorption/control_record.py`
- `tools/memflow_absorption/config_stage.py`
- `tests/tools/test_absorption_safety.py`
- `tests/tools/test_absorption_snapshot.py`
- `tests/tools/test_absorption_manifest.py`
- `tests/tools/test_absorption_inventory.py`
- `tests/tools/test_absorption_control_record.py`
- `tests/tools/test_absorption_config_stage.py`
- `tests/fixtures/memflow_absorption/`

### Memflow repository: create/modify

- Create: `memflow/cutover_fence.py`
- Create: `tests/test_cutover_fence.py`
- Create: `tests/test_cutover_drain.py`
- Modify write boundaries:
  - `memflow/core/channels.py`
  - `memflow/core/events.py`
  - `memflow/capture.py`
  - `memflow/presence.py`
  - `memflow/delivery.py`
  - `memflow/session_snapshot.py`
  - `memflow/usage.py`
  - `memflow/mcp/specs.py`
  - `memflow/mcp/handlers.py`
  - `memflow/mcp/loops.py`
  - `memflow/mcp_server.py`
  - `memflow/core/git_sync.py`
  - `memflow/daemon_launchd.py`
  - `memflow/cli/__init__.py`
  - `memflow/cli/parser.py`
- Modify daemon loops:
  - `memflow/daemon_loops/sync_loop.py`
  - `memflow/daemon_loops/terminal_delivery_loop.py`
  - `memflow/daemon_loops/session_snapshot_loop.py`
  - `memflow/daemon_loops/chat_notify_loop.py`
  - `memflow/daemon_loops/consolidation_loop.py`
  - `memflow/daemon_loops/health_watchdog.py`

### Synapse repository: create/modify

- Modify: `pyproject.toml`
- Create: `src/synapse/runtime_install.py`
- Create: `scripts/install-runtime.py`
- Create: `launchd/com.synapse.dashboard.plist.template`
- Create: `launchd/com.synapse.watcher.plist.template`
- Create: `launchd/com.synapse.runtime-loop.plist.template`
- Create: `src/synapse/backend_registry.py`
- Modify: `src/synapse/memo_backend/__init__.py` — preserve current exports and
  add only `MemoOperationalBackend`.
- Create: `src/synapse/memo_backend/_operational.py`
- Modify:
  - `src/synapse/federator/base.py`
  - `src/synapse/federator/federation.py`
  - `src/synapse/federator/routing.py`
  - `src/synapse/federator/actions.py`
  - `src/synapse/federator/insights.py`
  - `src/synapse/federator/chat.py`
  - `src/synapse/federator/ops.py`
  - `src/synapse/cli/_deps.py`
  - `src/synapse/cli/parser.py`
  - `src/synapse/mcp_server.py`
  - `src/synapse/runtime.py`
  - `src/synapse/trinity_doctor.py`
  - `src/synapse/dashboard/_server.py`
  - `src/synapse/dashboard/_payloads.py`
  - `src/synapse/dashboard/_backend_health.py`
  - `src/synapse/dashboard/_html.py`

## Shared Interfaces

`CutoverState` is defined once in `tools/memflow_absorption/schemas.py` with
`PREPARING`, `READY`, `QUIESCING`, `QUIESCED`, `STAGED`,
`ACTIVATION_READY`, `EPOCH_COMMITTED`, `ACTIVATED`, `VERIFIED`, `ABORTING`,
`ABORTED`, and `RETIRED`. Plans 04–05 import it and never redefine it.

`CutoverMode` is a `StrEnum` with `ACTIVE="active"`,
`QUIESCING="quiescing"`, and `RETIRED="retired"`.

`FenceMarker` contains `schema`, `attempt_id`, `mode`, `epoch`,
`expected_commit`, `runtime_digest`, `device_id`, `key_id`, `issued_at`,
`expires_at`, `control_oid`, `control_sequence`, `previous_control_oid`, and
`signature`. `VerifiedControlRecord` contains the freshly fetched control OID,
canonical signed payload, state, sequence, previous OID, roster version,
verification timestamp, and signer identity. `FinalFenceProof` contains the
attempt/control OID, zero-drain timestamp, immutable Memflow remote commit OID,
per-origin signed head/sequence/hash, source snapshot hashes, and signature.
`DrainSnapshot` contains immutable per-domain counts for requests, event
append, delivery, ACK, cursor, sync, Git push, autonomous loops, and writable
handles, plus `inflight_total`, `clean`, and the last fsync timestamp.

Exact fence signatures:

- `FenceGate.install(marker: FenceMarker, control: VerifiedControlRecord) ->
  None`
- `FenceGate.authorize(kind: Literal["read", "write", "startup",
  "abort_healthcheck"]) -> None`
- `FenceGate.admit_mutation(domain: str) -> ContextManager[None]`
- `FenceGate.drain_snapshot() -> DrainSnapshot`
- `verify_control_record(*, expected_oid: str, roster: VerificationRoster) ->
  VerifiedControlRecord`

`QUIESCING` rejects startup and newly admitted mutations with retryable
`memflow.cutover.maintenance`; already tracked work may finish. `RETIRED`
rejects startup and mutation with non-retryable `memflow.cutover.retired`.
`abort_healthcheck` is a private, non-listening startup path authorized only by
a freshly verified `ABORTING` control record for the same attempt. It may open
the read-only store, verify chains/config/runtime, and return a signed
`RollbackHealthyReceipt`; it cannot bind a public listener, launch autonomous
loops, acquire a writer, or admit mutations. Public `startup` remains rejected
through QUIESCING and ABORTING.
Mode validation and in-flight increment happen under the same process/file lock
inside `admit_mutation`, removing the authorize/track race. The marker is an
fsynced atomic file at `<memflow_state>/cutover/fence.json`; installing the
first non-ACTIVE marker also creates an fsynced `fence-seen` sentinel. From
then on missing, corrupt, or expired state refuses startup and mutation. Expiry
never restores ACTIVE. RETIRED is monotonic and permanent; QUIESCING changes
only with a newer signature-verified control OID/sequence explicitly installed
by the controller. Returning to ACTIVE is accepted only from a newer verified
ABORTED record and only if RETIRED has never been observed.

Exact Synapse signatures:

- `BackendRegistry.from_config(*, mode: Literal["memo"], client: MemoClient |
  None = None) -> BackendRegistry`
- `BackendRegistry.durable() -> DurableBackend`
- `BackendRegistry.operational() -> OperationalBackend`

### Task 1: Build safe snapshot, capability-manifest, and inventory tooling

**Files:**
- Create all Memo
  `tools/memflow_absorption/{schemas,safety,snapshot,manifest,inventory,control_record}.py`
- Create `tools/memflow_absorption/{__init__,__main__}.py`
- Create the five corresponding test modules and sanitized fixtures.

**Interfaces:**
- Produces:
  - `assert_safe_attempt_root(path, attempt_id) -> Path`
  - `create_readonly_snapshot(source, target) -> SnapshotReceipt`
  - `build_capability_manifest(memo_snapshot: Path, memflow_snapshot: Path,
    usage_snapshot: Path, audit_exclusions: AuditExclusions) ->
    CapabilityManifest`
  - `build_consumer_inventory(roots: Sequence[Path], process_snapshot:
    ProcessSnapshot, launchd_snapshot: LaunchdSnapshot) -> ConsumerInventory`
  - `build_synapse_retirement_manifest(snapshot: Path) ->
    SynapseRetirementManifest`
  - `verify_control_record(*, expected_oid: str, roster:
    VerificationRoster) -> VerifiedControlRecord`

- [ ] **Step 1: Write failing safety and manifest tests**

```python
@pytest.mark.parametrize("bad", [Path("/"), Path.home(), Path("/Users/fer/repos/memo")])
def test_attempt_root_rejects_broad_targets(bad):
    with pytest.raises(SafetyError):
        assert_safe_attempt_root(bad, "attempt-123")


def test_manifest_excludes_audit_and_blocks_freeze_on_ambiguous_usage(snapshot_fixture):
    out = build_capability_manifest(
        memo_snapshot=snapshot_fixture.memo,
        memflow_snapshot=snapshot_fixture.memflow,
        usage_snapshot=snapshot_fixture.usage,
        audit_exclusions=signed_exclusions(
            event_ids=("evt-audit-1", "evt-eval-4"),
            attempt_ids=("audit-1",),
            window=snapshot_fixture.window,
        ),
    )
    assert out.by_name("continuity").disposition == "absorb"
    assert out.by_name("tasks").disposition == "absorb"
    assert out.by_name("audit-query") is None
    assert out.frozen is False
    assert out.blockers == ("tasks:ambiguous-traffic",)
```

Also test symlink rejection, sentinel requirement, immutable snapshot receipt,
stable canonical JSON, self-instrumented usage exclusion, second-Mac freshness
flag, exact inclusive 90-day boundaries, complete zero-use deletion, refusal to
sign with blockers, exclusion signature/window/provenance validation, unknown
traffic blocking, and one disposition per capability. Client names and topic
suffixes may flag rows for review but never exclude them without an exact
signed event/attempt ID.

- [ ] **Step 2: Run tests and confirm modules are missing**

```bash
uv run --no-sync pytest \
  tests/tools/test_absorption_safety.py \
  tests/tools/test_absorption_snapshot.py \
  tests/tools/test_absorption_manifest.py \
  tests/tools/test_absorption_inventory.py \
  tests/tools/test_absorption_control_record.py -v
```

- [ ] **Step 3: Implement strict schemas and read-only snapshot inputs**

The capability row fields are exactly:

```python
@dataclass(frozen=True)
class OperationRoute:
    route_id: str
    predicate: Mapping[str, object]
    memo_methods: tuple[str, ...]
    memo_mcp: tuple[str, ...]
    memo_cli: tuple[str, ...]
    parameter_mapping: Mapping[str, str]
    defaults: Mapping[str, object]
    result_mapping: Mapping[str, str]
    error_mapping: Mapping[str, str]
    transform_id: str
    fixture_sha256: tuple[str, ...]
    atomic_group: str | None


@dataclass(frozen=True)
class OperationMappingRow:
    source_operation: str
    source_commit: str
    source_tests: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    capability: str
    disposition: Literal["memo_native", "absorb", "internal", "delete"]
    routes: tuple[OperationRoute, ...]
    parity_tests: tuple[str, ...]
    deletion_proof: tuple[str, ...]


@dataclass(frozen=True)
class SloBaseline:
    baseline_id: str
    source_commit: str
    workload_id: str
    machine_class: str
    window_started_at: str
    window_ended_at: str
    sample_count: int
    visibility_p50_ms: float
    visibility_p95_ms: float
    visibility_p99_ms: float
    visibility_max_ms: float
    recovery_max_ms: float
    error_rate: float
    data_loss_count: int
    duplicate_count: int
    tolerance_ratio: float


@dataclass(frozen=True)
class CapabilityRow:
    name: str
    sources: tuple[str, ...]
    consumers: tuple[str, ...]
    window_started_at: str
    window_ended_at: str
    observed_calls: int
    observed_daemon_events: int
    machines: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    exclusion_counts: Mapping[str, int]
    evidence_complete: bool
    source_operations: tuple[str, ...]
    operation_mappings: tuple[OperationMappingRow, ...]
    slo_baseline_ids: tuple[str, ...]
    dependencies: tuple[str, ...]
    disposition: Literal["memo_native", "absorb", "internal", "delete"]
    memo_target: str
    parity_tests: tuple[str, ...]
    deletion_proof: tuple[str, ...]


@dataclass(frozen=True)
class SynapseRetirementManifest:
    schema: Literal["memo.synapse_retirement.v1"]
    source_commit: str
    files: tuple[str, ...]
    symbols: tuple[str, ...]
    tests: tuple[str, ...]
    goldens: tuple[str, ...]
    active_reference_sha256: str
    signer_key_id: str
    signature: str
```

`CapabilityManifest` additionally contains `frozen_at`, `window_started_at`,
`window_ended_at`, both machine IDs, immutable source-receipt hashes,
canonical `operation_mappings`, canonical `slo_baselines`, their SHA-256
digests, `blockers`, `frozen`, signer/key ID, and signature. Disposition is
resolved per source operation and summarized at capability level.
Canonicalization sorts capability/operation names and evidence IDs but never
discards duplicate telemetry counts. `frozen=True` requires zero blockers,
complete evidence on both machines, one mapping/deletion proof per source
operation, and at least one SLO baseline for every admitted latency-sensitive
operation.

Routes make multimode and composite source operations explicit. Predicates are
closed canonical match expressions over source arguments; all predicates for a
row must be disjoint and collectively cover its signed fixtures. A route may
target several Memo methods/surfaces. When `atomic_group` is non-null, all
listed method effects commit in one `OperationalStore` transaction and expose
one idempotent result; partial application is invalid. `transform_id` names an
allowlisted, versioned deterministic transformer whose code digest is included
in the capability manifest. Defaults, error mapping, and fixture digests are
authority data, not prose.

The manifest builder, not Plan 02, originates and signs the canonical
`operation-map.json` and `slo-baseline.json` bytes. It derives source
operations from the pinned Memflow commit and exact source tests, joins them
to raw signed usage evidence from both configured Macs over the inclusive
window `[frozen_at - 90 days, frozen_at]`, and rejects missing hours, clock
coverage gaps, unsigned evidence, unknown operations, or mixed commits. Every
operation has exactly one disposition. `delete` requires complete zero-use
evidence plus exact deletion proof; every other row requires target parameter
and result/error mappings, complete route predicates, fixture digests, and
parity tests. SLO rows use the same signed workload
receipts, require at least 100 samples, and freeze visibility distribution,
recovery maximum, error rate, and zero data-loss/duplicate counts. Their
canonical byte digests are fields of `CapabilityManifest`; Plan 02 only copies
and verifies those bytes.

The same immutable Synapse snapshot produces a signed
`SynapseRetirementManifest`. It enumerates every Memflow-specific file, symbol,
test, and golden at the pinned Synapse commit (initially including
`src/synapse/memflow_backend.py`) and binds the digest of a full source/test
reference scan. Plan 05 may delete only listed rows and must finish with a
negative scan proving that no active Memflow reference remains; any newly
discovered reference blocks readiness and requires a re-signed manifest.

Manifest, inventory, and control-proof signatures use the permanent
`memo.operational_signing` contracts from Plan 01. This task defines the
cutover record schemas and verification; Plan 05 later adds CAS transitions
and votes. Each Mac's usage proof binds device/key ID, query version, exact
window, snapshot commit OID, raw event-set hash, exclusion-set hash, and
signature.

Snapshots copy only explicitly enumerated files with `O_NOFOLLOW`, preserve
mode/timestamp in the receipt, fsync files and directory, then mark them
read-only. `__main__` exposes `snapshot`, `manifest`, and `inventory`; all write
only under a validated attempt root and default to `--dry-run`.

- [ ] **Step 4: Run focused tooling tests**

```bash
uv run --no-sync pytest tests/tools/test_absorption_*.py -v
uv run --no-sync mypy tools/memflow_absorption
uv run --no-sync ruff check \
  tools/memflow_absorption tests/tools/test_absorption_*.py
```

- [ ] **Step 5: Commit operator tooling**

```bash
git add \
  tools/memflow_absorption/__init__.py \
  tools/memflow_absorption/__main__.py \
  tools/memflow_absorption/schemas.py \
  tools/memflow_absorption/safety.py \
  tools/memflow_absorption/snapshot.py \
  tools/memflow_absorption/manifest.py \
  tools/memflow_absorption/inventory.py \
  tools/memflow_absorption/control_record.py \
  tests/tools/test_absorption_safety.py \
  tests/tools/test_absorption_snapshot.py \
  tests/tools/test_absorption_manifest.py \
  tests/tools/test_absorption_inventory.py \
  tests/tools/test_absorption_control_record.py \
  tests/fixtures/memflow_absorption
git commit -m "feat: freeze Memflow absorption manifests safely"
```

### Task 2: Add Memflow request fencing at every mutation boundary

**Files:**
- Create: `/Users/fer/repos/memflow/memflow/cutover_fence.py`
- Create: `/Users/fer/repos/memflow/tests/test_cutover_fence.py`
- Modify the Memflow write-boundary files listed above.

**Interfaces:**
- Produces: `CutoverMode`, `FenceMarker`,
  `FenceGate.admit_mutation/drain_snapshot`.

- [ ] **Step 1: Write failing mode and in-flight tests**

```python
def test_quiescing_rejects_new_write_but_finishes_admitted_work(gate):
    with gate.admit_mutation("delivery"):
        gate.install(
            marker(mode="quiescing", attempt_id="a1"),
            control=verified_control(state="QUIESCING", attempt_id="a1"),
        )
        gate.authorize("read")
        with pytest.raises(CutoverError) as exc:
            with gate.admit_mutation("delivery"):
                pass
        assert exc.value.code == "memflow.cutover.maintenance"
        assert exc.value.retryable is True
    assert gate.drain_snapshot().inflight_total == 0


def test_retired_rejects_startup_nonretryably(gate):
    gate.install(
        marker(mode="retired", epoch=7),
        control=verified_control(state="RETIRED", epoch=7),
    )
    with pytest.raises(CutoverError) as exc:
        gate.authorize("startup")
    assert exc.value.code == "memflow.cutover.retired"
    assert exc.value.retryable is False
```

Test the authorize/admit race with deterministic barriers, invalid signature,
wrong attempt, marker/control state or OID mismatch, stale epoch, missing
commit/runtime binding, nested admission,
exception cleanup, restart with a valid marker, missing/corrupt marker after
`fence-seen`, expired QUIESCING fail-closed, RETIRED monotonicity, and ACTIVE
parity before any fence has ever been installed. Include cross-repository
golden vectors proving a marker signed by Plan 01 canonical bytes verifies in
Memflow without importing Memo at runtime. Also prove that
`abort_healthcheck` requires ABORTING, never binds/listens/loops/writes, and
returns a receipt bound to the verified control OID.

- [ ] **Step 2: Run the new test in Memflow**

```bash
cd /Users/fer/repos/memflow
uv run --no-sync pytest tests/test_cutover_fence.py -v
```

- [ ] **Step 3: Implement the gate and wrap every real writer**

`FenceMarker` includes `attempt_id`, mode, epoch, expected Memflow commit,
runtime digest, device ID, key ID, issued/expiry timestamps, and signature.
Verify it before changing mode. Wrap MCP mutation handlers before execution and
wrap channels/events/capture/presence/delivery/session/usage/Git append code
with the single `admit_mutation(domain)` context. Its lock validates the durable
marker and increments the domain/in-flight counters atomically before releasing
admission. Read-only status remains available in QUIESCING and RETIRED.

- [ ] **Step 4: Run Memflow mutation parity**

```bash
uv run --no-sync pytest \
  tests/test_cutover_fence.py tests/test_capture_gate.py \
  tests/test_mcp_server.py tests/test_mcp_tools.py \
  tests/test_delivery.py tests/test_delivery_idempotency.py \
  tests/test_presence.py tests/test_session_snapshot.py \
  tests/test_usage.py -v
uv run --no-sync mypy memflow
uv run --no-sync ruff check memflow tests/test_cutover_fence.py
```

- [ ] **Step 5: Commit fencing in the Memflow repository**

```bash
git add \
  memflow/cutover_fence.py memflow/core/channels.py memflow/core/events.py \
  memflow/capture.py memflow/presence.py memflow/delivery.py \
  memflow/session_snapshot.py memflow/usage.py memflow/mcp/specs.py \
  memflow/mcp/handlers.py memflow/mcp/loops.py memflow/mcp_server.py \
  memflow/core/git_sync.py tests/test_cutover_fence.py
git commit -m "feat: fence Memflow for atomic retirement"
```

### Task 3: Add observable drain and startup refusal

**Files:**
- Modify: Memflow daemon loops, CLI, parser, launchd, and `cutover_fence.py`.
- Create: `/Users/fer/repos/memflow/tests/test_cutover_drain.py`
- Extend: Memflow daemon/sync tests.

**Interfaces:**
- Produces:
  - `memflow cutover status --json`
  - `memflow cutover drain --control-oid OID --timeout SECONDS`
  - `DrainSnapshot` counters.

- [ ] **Step 1: Write failing complete-drain tests**

```python
def test_drain_requires_every_writer_domain_clean(runtime):
    runtime.gate.install(
        marker(mode="quiescing"),
        control=verified_control(state="QUIESCING"),
    )
    runtime.counters.set("git_push", 1)
    report = runtime.drain(timeout=0.1)
    assert report.clean is False
    assert report.pending == {"git_push": 1}
    runtime.counters.set("git_push", 0)
    assert runtime.drain(timeout=1).clean is True


def test_keepalive_start_refuses_retired_marker(runtime):
    runtime.gate.install(
        marker(mode="retired", epoch=7),
        control=verified_control(state="RETIRED", epoch=7),
    )
    result = runtime.launchd_start()
    assert result.exit_code != 0
    assert runtime.listener_open is False
```

Counters cover requests, event append, delivery, ACK, cursor, sync, push,
autonomous loops, and writable handles. Also test fsync-before-clean, timeout,
stuck loop, and five-second legacy join no longer claiming clean.

- [ ] **Step 2: Run drain/daemon tests**

```bash
cd /Users/fer/repos/memflow
uv run --no-sync pytest \
  tests/test_cutover_drain.py tests/test_mcp_loops.py \
  tests/test_daemon.py tests/test_daemon_launchd.py \
  tests/test_git_sync_rebase_recovery.py -v
```

- [ ] **Step 3: Instrument loops and add temporary CLI**

Every loop enters a named tracker around a mutation-capable iteration and
checks the gate before scheduling new work. `drain` never changes mode: it
re-fetches and verifies the supplied control OID, requires the durable marker
already be QUIESCING for the same attempt/OID, then waits for every counter plus
Git locks/unpushed commits/writable handles to reach zero, fsyncs
journal/cursors, and returns canonical JSON. Only the Plan 05 controller may
install a transition marker. Launchd startup calls `authorize("startup")`
before binding a port or creating threads.

- [ ] **Step 4: Run Memflow runtime suite**

```bash
uv run --no-sync pytest \
  tests/test_cutover_drain.py tests/test_mcp_loops.py \
  tests/test_daemon.py tests/test_daemon_launchd.py \
  tests/test_delivery.py tests/test_session_snapshot.py \
  tests/test_git_sync_rebase_recovery.py -v
uv run --no-sync mypy memflow
uv run --no-sync ruff check memflow tests/test_cutover_drain.py
```

- [ ] **Step 5: Commit drain**

```bash
git add \
  memflow/cutover_fence.py memflow/daemon_loops \
  memflow/daemon_launchd.py memflow/cli/__init__.py memflow/cli/parser.py \
  tests/test_cutover_drain.py tests/test_mcp_loops.py \
  tests/test_daemon.py tests/test_daemon_launchd.py
git commit -m "feat: expose verifiable Memflow drain"
```

### Task 4: Isolate the Synapse runtime from Memflow

**Files:**
- Modify/create Synapse runtime installer, pyproject, templates, and tests.
- Create:
  - `/Users/fer/repos/synapse/tests/test_runtime_isolation.py`

**Interfaces:**
- Produces: commit-versioned Synapse runtime and LaunchAgent templates with no
  Memflow interpreter, root, binary, or environment dependency.

- [ ] **Step 1: Write failing runtime-template tests**

```python
def test_runtime_and_plists_have_no_memflow_path(built_runtime, rendered_plists):
    forbidden = (
        "/Users/fer/repos/memflow/.venv",
        "/Users/fer/.memflow/bin",
        "SYNAPSE_MEMFLOW_BIN",
        "SYNAPSE_MEMFLOW_ROOT",
    )
    assert built_runtime.python.exists()
    for path in rendered_plists:
        text = path.read_text()
        assert not any(item in text for item in forbidden)
        assert str(built_runtime.python) in text
```

Also test runtime digest, install idempotency, old runtime left untouched in
dry-run, template modes, and atomic current-version pointer.

- [ ] **Step 2: Run Synapse isolation tests**

```bash
cd /Users/fer/repos/synapse
uv run --no-sync pytest tests/test_runtime_isolation.py -v
```

- [ ] **Step 3: Implement an isolated, versioned installer**

`scripts/install-runtime.py` builds under a Synapse-owned state root named by
commit and dependency-lock digest, verifies the installed version, renders the
three templates, fsyncs, and atomically updates a `current` pointer. Default is
dry-run; `--apply` requires expected commit/digest. It does not load or unload
LaunchAgents in this task.

- [ ] **Step 4: Run Synapse runtime/package gates**

```bash
uv run --no-sync pytest tests/test_runtime_isolation.py tests/test_runtime_policy.py -v
uv run --no-sync mypy src/synapse
uv run --no-sync ruff check \
  src/synapse/runtime_install.py scripts/install-runtime.py \
  tests/test_runtime_isolation.py
```

- [ ] **Step 5: Commit in Synapse**

```bash
git add \
  pyproject.toml src/synapse/runtime_install.py scripts/install-runtime.py \
  launchd/com.synapse.dashboard.plist.template \
  launchd/com.synapse.watcher.plist.template \
  launchd/com.synapse.runtime-loop.plist.template \
  tests/test_runtime_isolation.py
git commit -m "feat: isolate Synapse runtime"
```

### Task 5: Replace Synapse's Memflow contract with a Memo backend registry

**Files:**
- Create: Synapse backend registry, Memo backend, and tests.
- Modify: Synapse federator, CLI, MCP, runtime, doctor, and dashboard files
  listed in File Structure.

**Interfaces:**
- Consumes: Plan 02 Memo `memo_*` operational APIs.
- Produces: `BackendRegistry.from_config(mode="memo")`.

- [ ] **Step 1: Write failing Memo-only tests**

```python
def test_memo_only_registry_has_no_memflow_backend(fake_memo_client):
    registry = BackendRegistry.from_config(mode="memo", client=fake_memo_client)
    assert registry.durable().name == "memo"
    assert registry.operational().name == "memo"
    assert "memflow" not in registry.backend_names()


def test_dashboard_and_mcp_payloads_have_no_memflow_route(app):
    payload = app.dashboard_payload()
    assert "memflow" not in json.dumps(payload).lower()
    assert "/memflow" not in app.routes()
```

Also cover routing of handoff, delivery, presence, continuity, health, errors,
CLI choices, doctor, and no `NullMemflowBackend`.

- [ ] **Step 2: Run Synapse backend tests**

```bash
cd /Users/fer/repos/synapse
uv run --no-sync pytest \
  tests/test_memo_only_mode.py tests/test_dashboard_no_memflow.py \
  tests/test_backends.py tests/test_federator.py tests/test_mcp_server.py -v
```

- [ ] **Step 3: Implement registry and Memo adapter**

The Memo operational adapter calls only the Plan 02 contracts and maps
`memo.error.v1` without changing retryability. Replace backend branching across
federator, CLI, MCP, runtime, doctor, and dashboard with the registry. Keep the
current Memflow backend only as a source parity oracle until Plan 05 deletes it;
it is not selectable in `mode="memo"`.

- [ ] **Step 4: Run Synapse full focused gates**

```bash
uv run --no-sync pytest \
  tests/test_memo_only_mode.py tests/test_runtime_isolation.py \
  tests/test_dashboard_no_memflow.py tests/test_backends.py \
  tests/test_federator.py tests/test_runtime_policy.py \
  tests/test_trinity_doctor.py tests/test_dashboard.py \
  tests/test_mcp_server.py tests/test_cli.py -v
uv run --no-sync mypy src/synapse
uv run --no-sync ruff check src/synapse tests
```

- [ ] **Step 5: Commit Memo-only Synapse**

```bash
git add \
  src/synapse/backend_registry.py \
  src/synapse/memo_backend/__init__.py \
  src/synapse/memo_backend/_operational.py \
  src/synapse/federator/base.py \
  src/synapse/federator/federation.py \
  src/synapse/federator/routing.py \
  src/synapse/federator/actions.py \
  src/synapse/federator/insights.py \
  src/synapse/federator/chat.py src/synapse/federator/ops.py \
  src/synapse/cli/_deps.py src/synapse/cli/parser.py \
  src/synapse/mcp_server.py src/synapse/runtime.py \
  src/synapse/trinity_doctor.py src/synapse/dashboard/_server.py \
  src/synapse/dashboard/_payloads.py \
  src/synapse/dashboard/_backend_health.py \
  src/synapse/dashboard/_html.py tests/test_memo_only_mode.py \
  tests/test_dashboard_no_memflow.py tests/test_backends.py \
  tests/test_federator.py tests/test_runtime_policy.py \
  tests/test_trinity_doctor.py tests/test_dashboard.py \
  tests/test_mcp_server.py tests/test_cli.py
git commit -m "feat: route Synapse through Memo only"
```

### Task 6: Stage every consumer configuration and readiness report

**Files:**
- Create: `tools/memflow_absorption/config_stage.py`
- Create: `tests/tools/test_absorption_config_stage.py`
- Modify: `tools/memflow_absorption/inventory.py`
- Modify: `src/memo/cli_doctor.py`
- Extend: `tests/test_cli_doctor.py`

**Interfaces:**
- Consumes: exact consumer inventory and Plan 02 Memo endpoints.
- Produces: staged replacements plus `memo doctor --strict-runtime
  --operational-readiness --json`.

- [ ] **Step 1: Write failing round-trip/staging tests**

```python
def test_stage_preserves_unknown_fields_and_never_touches_live_file(
    tmp_path, codex_config
):
    live = tmp_path / "config.toml"
    live.write_text(codex_config + '\nunknown = "keep"\n')
    before = live.read_bytes()
    staged = stage_consumer_config(
        live, attempt_root=tmp_path / "attempt-a1",
        mapping=memo_only_mapping(epoch=7),
    )
    assert live.read_bytes() == before
    assert 'unknown = "keep"' in staged.read_text()
    assert "memflow" not in staged.read_text().lower()


def test_readiness_fails_for_established_memflow_connection(report):
    report.connections = [connection(remote_port=18766, state="ESTABLISHED")]
    assert report.ready is False
    assert "connection:18766" in report.blockers
```

Cover JSON/TOML/plist/shell-hook formats, mode preservation, parse failure,
unknown fields, connection/process/listener/LaunchAgent detection, and missing
restart procedure.

- [ ] **Step 2: Run staging/doctor tests**

```bash
cd /Users/fer/repos/memo
uv run --no-sync pytest \
  tests/tools/test_absorption_config_stage.py \
  tests/tools/test_absorption_inventory.py \
  tests/test_cli_doctor.py -v
```

- [ ] **Step 3: Implement exact target adapters**

Support these inventory targets without changing them:

```text
~/.codex/config.toml
~/.codex/hooks.json
~/.claude.json
~/.claude/settings.json
~/.claude/hooks/memflow-handoff.sh
~/.claude/hooks/memflow-continuity.sh
~/.gemini/settings.json
~/.config/devin/config.json
~/.config/opencode/opencode.json
~/.zshenv ~/.zshrc ~/.zprofile ~/.bashrc ~/.bash_profile
~/Library/LaunchAgents/com.fer.memflow.mcp.plist
~/Library/LaunchAgents/com.synapse.dashboard.plist
~/Library/LaunchAgents/com.synapse.watcher.plist
~/Library/LaunchAgents/com.synapse.runtime-loop.plist
~/Library/LaunchAgents/com.synapse.memo-nightly.plist
~/Library/LaunchAgents/com.synapse.dream-synthesis.plist
~/Library/LaunchAgents/com.synapse.morning-digest.plist
~/Library/LaunchAgents/com.synapse.vault-ingest.plist
~/Library/LaunchAgents/com.synapse.whatsapp-ingest.plist
~/Library/LaunchAgents/com.synapse.memo-recall-daemon.plist
```

Parse structured files, render staged files under
`<attempt>/configs/<relative-target>`, fsync them, preserve original digest and
mode in the inventory, and emit a required client close/restart action. The
read-only inventory resolves each target separately and signs absolute path,
device, inode, SHA-256, mode, owner, and adapter; staging and later cleanup
accept only those immutable rows, never a glob. Doctor checks exact Memo
version/runtime/config/daemon/roster/epoch plus Memflow processes, port 18766,
connections, hooks, shims, and Synapse executable paths.
Every listed LaunchAgent must have one signed inventory row with an explicit
`replace` or `delete` disposition before READY; missing, extra, or unresolved
labels block the manifest.

- [ ] **Step 4: Run final readiness gates in all three repos**

```bash
cd /Users/fer/repos/memo
uv run --no-sync pytest tests/tools/test_absorption_*.py tests/test_cli_doctor.py -v
uv run --no-sync mypy tools/memflow_absorption src/memo
uv run --no-sync ruff check tools/memflow_absorption tests/tools src/memo/cli_doctor.py

cd /Users/fer/repos/memflow
uv run --no-sync pytest tests/test_cutover_fence.py tests/test_cutover_drain.py -v

cd /Users/fer/repos/synapse
uv run --no-sync pytest \
  tests/test_runtime_isolation.py tests/test_memo_only_mode.py \
  tests/test_dashboard_no_memflow.py -v
```

- [ ] **Step 5: Commit Memo readiness staging**

```bash
cd /Users/fer/repos/memo
git add \
  tools/memflow_absorption/config_stage.py \
  tools/memflow_absorption/inventory.py \
  src/memo/cli_doctor.py tests/tools/test_absorption_config_stage.py \
  tests/tools/test_absorption_inventory.py tests/test_cli_doctor.py
git commit -m "feat: stage Memo-only consumer cutover"
```

## Plan Acceptance Gate

- Capability and consumer manifests are reproducible from immutable snapshots
  and include a complete exact 90-day window from both Macs; missing or
  ambiguous evidence is a blocker, and complete zero-use capabilities are
  classified `delete`.
- The frozen manifest is signature-verified, has zero blockers, binds exact
  source operation mappings/SLO baselines, and is the mandatory Plan 02 input.
- Memflow ACTIVE behavior retains source-test parity.
- QUIESCING fences every mutation and reports a complete zero-writer drain.
- RETIRED refuses startup before binding a listener or creating loops.
- Synapse runs from its own versioned runtime and `mode="memo"` exposes no
  Memflow backend or dashboard route.
- Every live consumer has a staged Memo-only replacement and explicit restart
  procedure; no live file is changed by this plan.
- Strict readiness reports current connections, PIDs, ports, jobs, hooks,
  shims, versions, config digests, roster, and epoch.
- No service is stopped, started, unloaded, reconfigured, or deleted.
