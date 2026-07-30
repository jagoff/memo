# Synapse Retirement and Selective Memo Absorption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire Synapse completely while preserving only its proven, useful behavior as native Memo functionality during the existing coordinated Memflow→Memo cutover.

**Architecture:** Extend the existing signed Memflow absorption inventory with a
Synapse operation manifest and a bounded Synapse data receipt. Map admitted
operations to existing Memo domains first, add only evidence-backed deltas,
stage all imports and consumer replacements in a disposable namespace, then
reuse the existing two-Mac fence/activation epoch to switch every client and
remove Synapse permanently.

**Tech Stack:** Python 3.11+, dataclasses, canonical JSON, Ed25519 operational
signing, SQLite/Markdown Memo storage, Click CLI, FastMCP, pytest, Ruff, mypy,
macOS launchd, and Git snapshots.

## Global Constraints

- Work from the existing Memo worktree `/Users/fer/repos/memo/.worktrees/memflow-absorption` and the isolated Synapse worktree `/Users/fer/repos/synapse/.worktrees/memflow-absorption-runtime`; do not modify unrelated dirty files.
- Markdown remains Memo's source of truth; all operational writes use Memo's operational ledger and write policy.
- No production service, LaunchAgent, client configuration, or state directory changes occur before the signed activation attempt reaches its cutover phase.
- The final release contains no Synapse package, import, binary, MCP/CLI namespace, compatibility alias, fallback, LaunchAgent, configuration, or state root.
- Usage evidence covers the inclusive 90-day window on both configured Macs; tests, smoke, benchmarks, and Synapse `runtime_loop` self-audit traffic are excluded from capability admission.
- Every admitted operation has one disposition, one Memo owner or deletion proof, parameter/result/error mappings, fixtures, and an SLO baseline.
- Temporary importer and legacy readers are operator-only and are removed before the final Memo release is tagged.
- CI order remains `ruff -> mypy -> pytest`; changes to runtime, hooks, retrieval, installer, migrations, or MLX paths also preserve macOS smoke gates.

---

### Task 1: Freeze and sign the Synapse operation manifest

**Files:**
- Modify: `tools/memflow_absorption/schemas.py` (`SynapseRetirementManifest`, `CapabilityManifest` support records)
- Modify: `tools/memflow_absorption/inventory.py` (`build_synapse_retirement_manifest`, source operation discovery)
- Modify: `tools/memflow_absorption/manifest.py` (`build_capability_manifest`, validation)
- Modify: `tools/memflow_absorption/__main__.py` (manifest/inventory CLI arguments)
- Create: `tools/memflow_absorption/synapse_catalog.py` (canonical Synapse operation extraction and exclusions)
- Test: `tests/tools/test_absorption_inventory.py`
- Test: `tests/tools/test_absorption_manifest.py`
- Create: `tests/tools/test_synapse_catalog.py`

**Interfaces:**
- Consumes: an immutable Synapse snapshot containing `source.json`, `src/`,
  `tests/`, `eval/`, `launchd/`, `pyproject.toml`, and the two-Mac signed
  usage proofs already defined by `UsageProof`.
- Produces: `discover_synapse_operations(snapshot: Path) -> tuple[SynapseOperation, ...]`,
  `build_synapse_capability_manifest(snapshot: Path, usage_proofs: tuple[Path, ...], exclusions: tuple[Path, ...], ...) -> CapabilityManifest`,
  and a signed `memo.synapse_retirement.v2` manifest that binds source
  operations, consumers, dispositions, routes, fixtures, and SLOs.

`SynapseOperation` is the canonical row type:
`SynapseOperation(source_operation: str, source_files: tuple[str, ...],
source_symbols: tuple[str, ...], consumers: tuple[str, ...],
daemon_routes: tuple[str, ...], exclusion_reason: str | None,
fixture_paths: tuple[str, ...])`.

- [ ] **Step 1: Write the failing catalog tests**

```python
def test_catalog_includes_canonical_mcp_and_live_daemon_operations(snapshot):
    rows = discover_synapse_operations(snapshot)
    names = {row.source_operation for row in rows}
    assert "synapse.federate.query" in names
    assert "synapse.chat.ask" in names
    assert "synapse.runtime.loop" in names
    assert "synapse.watcher.event" in names

def test_catalog_excludes_runtime_self_audit_from_admission(snapshot):
    rows = discover_synapse_operations(snapshot)
    runtime = next(row for row in rows if row.source_operation == "synapse.runtime.loop")
    assert runtime.exclusion_reason == "self_audit"
```

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run: `uv run --no-sync pytest tests/tools/test_synapse_catalog.py tests/tools/test_absorption_manifest.py -q`

Expected: FAIL because `SynapseOperation` and the Synapse manifest builder do
not exist.

- [ ] **Step 3: Implement canonical operation discovery**

Create `SynapseOperation` with fields `source_operation`, `source_files`,
`source_symbols`, `consumers`, `daemon_routes`, `exclusion_reason`, and
`fixture_paths`. Parse the canonical MCP list from the pinned
`src/synapse/mcp_catalog.py`, CLI verbs from `src/synapse/cli/parser.py`, and
daemon routes from `runtime.py`, `watcher.py`, `morning_digest.py`,
`whatsapp_live.py`, and `vault_archive.py`. Sort all rows and reject symlinks,
noncanonical `source.json`, and source commits that are not 40-character OIDs.

- [ ] **Step 4: Extend the signed manifest builder**

Use the existing `CapabilityManifest`/`OperationMappingRow` authority. Require
one mapping per discovered operation, bind every route to fixture SHA-256
digests, reject overlapping predicates, require `memo_native`/`absorb`/`internal`
targets for non-deleted rows, and require complete zero-use evidence plus
`deletion_proof` for `delete` rows. Keep the existing Ed25519 domain separation
and canonical JSON bytes.

- [ ] **Step 5: Add CLI inspection without production writes**

Extend `tools.memflow_absorption` with `synapse-catalog` and
`synapse-manifest` subcommands. Default to dry-run, write only under an
explicit attempt root, and print the manifest digest, operation counts,
dispositions, blockers, source commit, and both usage-proof IDs.

- [ ] **Step 6: Run focused verification**

Run:

```bash
uv run --no-sync pytest tests/tools/test_synapse_catalog.py tests/tools/test_absorption_inventory.py tests/tools/test_absorption_manifest.py -v
uv run --no-sync ruff check tools/memflow_absorption tests/tools
uv run --no-sync mypy tools/memflow_absorption
git diff --check
```

Expected: all focused tests pass and a fixture manifest verifies after a byte
mutation fails closed.

- [ ] **Step 7: Commit the manifest authority**

```bash
git add tools/memflow_absorption/schemas.py tools/memflow_absorption/inventory.py tools/memflow_absorption/manifest.py tools/memflow_absorption/__main__.py tools/memflow_absorption/synapse_catalog.py tests/tools/test_absorption_inventory.py tests/tools/test_absorption_manifest.py tests/tools/test_synapse_catalog.py
git commit -m "feat: freeze signed Synapse capability manifest"
```

### Task 2: Import bounded Synapse feedback and evaluation evidence

**Files:**
- Create: `tools/memflow_absorption/synapse_data.py` (read-only extraction and receipts)
- Modify: `tools/memflow_absorption/schemas.py` (`SynapseDataReceipt`)
- Modify: `tools/memflow_absorption/__main__.py` (`synapse-data` command)
- Modify: `src/memo/server_feedback.py` only if an idempotent import helper is missing
- Modify: `src/memo/memory/rerank_ops.py` only if the existing feedback API cannot preserve source IDs
- Test: `tests/tools/test_synapse_data.py`
- Modify: `tests/test_source_feedback.py`

**Interfaces:**
- Consumes: Synapse `state/ledger.jsonl`, `state/observability/chat-traces.jsonl`,
  `state/observability/chat_pipeline_trace.jsonl`, and `state/eval/`, all read
  through descriptor-safe, no-follow-symlink readers.
- Defines: `FeedbackImport(feedback_id: str, source_id: str, query: str, rating: str, answer: str = "")`,
  `EvalFixture(fixture_id: str, query: str, source_ids: tuple[str, ...], content_sha256: str, answer: str = "")`,
  `SynapseDataBundle(feedback: tuple[FeedbackImport, ...], eval_fixtures: tuple[EvalFixture, ...], input_sha256: str)`,
  and `SynapseDataReceipt(attempt_id: str, input_sha256: str, feedback_imported: int, feedback_skipped: int, eval_fixture_count: int, event_ids: tuple[str, ...], status: Literal["applied", "reused"])`.
- Produces: `extract_synapse_feedback(state_dir: Path, seen_ids: set[str]) -> tuple[FeedbackImport, ...]`,
  `extract_synapse_eval_fixtures(state_dir: Path) -> tuple[EvalFixture, ...]`,
  `apply_synapse_data(memory: Memory, data: SynapseDataBundle, *, attempt_id: str) -> SynapseDataReceipt`,
  and a canonical receipt with counts, skipped IDs, input hashes, and output
  operational event IDs.

- [ ] **Step 1: Write extraction and redaction tests**

```python
def test_feedback_extraction_is_idempotent_and_does_not_copy_answers(state_dir):
    bundle = extract_synapse_feedback(state_dir, seen_ids={"already-seen"})
    assert {item.feedback_id for item in bundle} == {"new-feedback"}
    assert all(item.answer == "" for item in bundle)

def test_eval_extraction_keeps_fixture_metadata_only(state_dir):
    fixture = extract_synapse_eval_fixtures(state_dir)[0]
    assert fixture.source_ids
    assert fixture.query
    assert fixture.answer == ""
    assert fixture.content_sha256
```

- [ ] **Step 2: Run tests to verify the expected failure**

Run: `uv run --no-sync pytest tests/tools/test_synapse_data.py -q`

Expected: FAIL because the bounded bundle types and extractors do not exist.

- [ ] **Step 3: Implement descriptor-safe extraction**

Parse only explicit feedback records (`action == "chat_feedback"`) and
high-signal eval fixtures. Never promote `runtime_loop`, raw chat answers,
full ledger entries, runtime snapshots, caches, cold-ledger files, or logs.
Normalize IDs and timestamps, redact answer text and arbitrary metadata, and
reject malformed JSON or symlinked paths. Duplicate IDs are skipped and counted.

- [ ] **Step 4: Implement idempotent Memo application**

Use `Memory.feedback_record(source_id, query_text, rating)` for ranking signals
and the existing Memo write policy/operational ledger for the receipt. Use a
stable operation key derived from `attempt_id` and source feedback ID. Do not
create a memory record for a trace or feedback event. Import eval fixtures into
the operator-only staging corpus, not the user memory vault.

- [ ] **Step 5: Verify receipt integrity and rollback**

Require the same input digest and attempt ID on replay, return the original
receipt without a second write, and reject a changed input bundle. Add a test
that a failed fixture import leaves no partial feedback or receipt event.

- [ ] **Step 6: Run focused verification and commit**

```bash
uv run --no-sync pytest tests/tools/test_synapse_data.py tests/test_source_feedback.py -v
uv run --no-sync ruff check tools/memflow_absorption src/memo tests/tools tests/test_source_feedback.py
uv run --no-sync mypy tools/memflow_absorption src/memo
git add tools/memflow_absorption/synapse_data.py tools/memflow_absorption/schemas.py tools/memflow_absorption/__main__.py tests/tools/test_synapse_data.py src/memo/server_feedback.py src/memo/memory/rerank_ops.py tests/test_source_feedback.py
git commit -m "feat: import bounded Synapse feedback evidence"
```

### Task 3: Prove Memo-native parity and absorb only measured chat deltas

**Files:**
- Create: `tools/memflow_absorption/synapse_parity.py` (adapter comparison runner)
- Create: `tests/tools/test_synapse_parity.py`
- Modify: `src/memo/memory/ask_ops.py` only for a failing admitted chat delta
- Modify: `src/memo/memory/evidence_ops.py` only for a failing provenance/abstention delta
- Modify: `src/memo/server_core_search.py` only for a failing Memo-facing contract
- Modify: `tests/test_memory_ask.py` and relevant retrieval regression tests
- Create: `tests/test_memory_evidence_pack.py` for new provenance/abstention cases
- Copy approved fixtures into: `tests/fixtures/synapse_retirement/`

**Interfaces:**
- Consumes: the signed operation manifest from Task 1 and the redacted eval
  fixtures from Task 2.
- Defines: `ParityFixture(fixture_id: str, source_operation: str, query: str, expected_status: str, expected_source_ids: tuple[str, ...])` and `ParityReport(status: Literal["pass", "blocked"], rows: tuple[ParityRow, ...], gap_ids: tuple[str, ...], p50_ms: float, p95_ms: float)`.
- Produces: `run_synapse_parity(manifest: CapabilityManifest, memo: Memory, fixtures: Sequence[ParityFixture]) -> ParityReport`,
  with per-operation status, latency, citation/provenance comparison,
  abstention comparison, and explicit `gap_ids`.

`ParityRow(fixture_id: str, status: str, latency_ms: float,
memo_source_ids: tuple[str, ...], provenance_ok: bool, error: str | None)` is
the per-fixture result type. Test fixtures are built by a deterministic
`fixture(name: str) -> ParityFixture` helper in the parity test module.

- [ ] **Step 1: Write parity tests for the already-native surfaces**

```python
def test_native_surface_maps_to_memo_without_synapse_namespace(manifest):
    row = manifest.by_name("federated_query")
    assert row is not None
    assert row.disposition == "memo_native"
    assert row.operation_mappings[0].routes[0].memo_mcp == ("memo_ask",)

def test_parity_report_blocks_unmapped_admitted_operation(manifest, memory):
    report = run_synapse_parity(manifest, memory, [fixture("unmapped")])
    assert report.status == "blocked"
    assert report.gap_ids == ("unmapped",)
```

- [ ] **Step 2: Run the expected failing tests**

Run: `uv run --no-sync pytest tests/tools/test_synapse_parity.py -q`

Expected: FAIL because the parity runner and report types are absent.

- [ ] **Step 3: Implement the comparison runner**

Call Memo's existing `memo_unified_briefing`, `Memory.ask`, `Memory.search`,
`Memory.evidence_pack`, conflict, session, and health APIs through the facade.
Compare canonicalized source IDs/provenance, answer status, conflict
surfacing, and bounded latency. Never import Synapse at runtime; the Synapse
snapshot is an offline oracle only.

- [ ] **Step 4: Apply the disposition rule to chat deltas**

Run the runner against every admitted chat fixture. A delta is absorbable only
when it has at least one real usage receipt, a reproducible failing fixture,
and a named Memo target. If no gap remains, record `delete`/`memo_native` in
the signed manifest and commit the parity harness without adding a new API.
If a gap remains, add the smallest change in `ask_ops.py` or `evidence_ops.py`,
then add a regression test and update the route/fixture digest in the manifest.

- [ ] **Step 5: Run retrieval and quality gates**

```bash
uv run --no-sync pytest tests/tools/test_synapse_parity.py tests/test_memory_ask.py tests/test_memory_evidence_pack.py -v
uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force
uv run --no-sync ruff check src/memo/memory/ask_ops.py src/memo/memory/evidence_ops.py src/memo/server_core_search.py tools/memflow_absorption tests
uv run --no-sync mypy src/memo
```

Expected: no admitted fixture regresses provenance, abstention, or recall;
any unresolved gap remains a signed blocker rather than an undocumented
fallback.

- [ ] **Step 6: Commit the parity gate**

```bash
git add tools/memflow_absorption/synapse_parity.py tests/tools/test_synapse_parity.py tests/fixtures/synapse_retirement src/memo/memory/ask_ops.py src/memo/memory/evidence_ops.py src/memo/server_core_search.py tests/test_memory_ask.py tests/test_memory_evidence_pack.py
git commit -m "test: prove Memo parity for Synapse capabilities"
```

### Task 4: Stage Memo-owned replacements for every Synapse consumer

**Files:**
- Create: `tools/memflow_absorption/consumer_migration.py`
- Modify: `tools/memflow_absorption/inventory.py` and `tools/memflow_absorption/schemas.py`
- Modify: `src/memo/runtime/daemon.py`, `src/memo/runtime/install.py`, `src/memo/watcher.py`, `src/memo/cli_import.py`, `src/memo/cli_dream.py`, and `src/memo/cli_ingest_daemon.py` only where an active Synapse job lacks a Memo-owned entrypoint
- Modify: `tests/test_runtime_isolation.py`, `tests/test_watcher.py`, `tests/test_whatsapp_ingest.py`, and new `tests/tools/test_consumer_migration.py`
- Modify external configuration through an operator-only staged renderer; do not edit production files in tests

**Interfaces:**
- Consumes: signed consumer inventory, capability manifest, and the existing
  Memo runtime install/report APIs.
- Produces: `build_consumer_replacement_plan(inventory: ConsumerInventory, manifest: CapabilityManifest) -> ConsumerReplacementPlan`,
  `render_memo_launch_agents(plan: ConsumerReplacementPlan, root: Path) -> tuple[Path, ...]`,
  and `verify_no_synapse_runtime_reference(root: Path, plan: ConsumerReplacementPlan) -> None`.

`ConsumerReplacement(old_label: str, new_label: str, command: tuple[str, ...],
owner: str, restart_required: bool, config_sha256: str,
rollback_action: str)` is one replacement row. `ConsumerReplacementPlan` stores
`rows: tuple[ConsumerReplacement, ...]` and `digest: str`, and exposes
`by_old_label(label: str) -> ConsumerReplacement`.

- [ ] **Step 1: Write replacement-plan tests**

```python
def test_replacement_plan_maps_active_jobs_to_memo_owned_commands(inventory, manifest):
    plan = build_consumer_replacement_plan(inventory, manifest)
    assert plan.by_old_label("com.synapse.whatsapp-ingest").command[:2] == ("memo", "import")
    assert plan.by_old_label("com.synapse.watcher").command[:2] == ("memo", "watch")

def test_rendered_plists_contain_no_synapse_or_memflow_paths(plan, tmp_path):
    paths = render_memo_launch_agents(plan, tmp_path)
    assert all("synapse" not in path.read_text().lower() for path in paths)
    assert all("memflow" not in path.read_text().lower() for path in paths)
```

- [ ] **Step 2: Run the expected failing tests**

Run: `uv run --no-sync pytest tests/tools/test_consumer_migration.py -q`

Expected: FAIL because the replacement plan and renderer do not exist.

- [ ] **Step 3: Implement the staged replacement planner**

Map the current labels as follows: `com.synapse.whatsapp-ingest` to
`memo import whatsapp`, `com.synapse.watcher` to the Memo watcher,
`com.synapse.memo-recall-daemon` to the existing Memo recall daemon, and
nightly/digest/dream/vault jobs to their existing Memo CLI commands. For the
dashboard, gateway, and MCP routes, emit a required client close/reconnect
action rather than inventing a dashboard compatibility service. Every plan row
records old label, new command, owner, restart requirement, config digest, and
rollback path.

- [ ] **Step 4: Add Memo runtime gaps only when the planner proves one**

Keep `src/memo/runtime/daemon.py` and `src/memo/watcher.py` as the owners of
launchd rendering. Add a focused command only if an active inventory row has
no existing Memo entrypoint; otherwise mark the row `memo_native` and do not
add a wrapper. Use stable isolated Memo runtime paths and preserve
`MEMO_NONINTERACTIVE=1` for hook/LaunchAgent commands.

- [ ] **Step 5: Run staged configuration checks**

```bash
uv run --no-sync pytest tests/tools/test_consumer_migration.py tests/test_runtime_isolation.py tests/test_watcher.py tests/test_whatsapp_ingest.py -v
uv run --no-sync ruff check tools/memflow_absorption src/memo/runtime src/memo/watcher.py src/memo/cli_import.py src/memo/cli_dream.py src/memo/cli_ingest_daemon.py tests
uv run --no-sync mypy tools/memflow_absorption src/memo/runtime src/memo
```

Expected: rendered staged plists are linted, point only to Memo-owned runtime
and commands, and contain no Synapse/Memflow path.

- [ ] **Step 6: Commit the consumer replacement stage**

```bash
git add tools/memflow_absorption/consumer_migration.py tools/memflow_absorption/inventory.py tools/memflow_absorption/schemas.py tests/tools/test_consumer_migration.py src/memo/runtime/daemon.py src/memo/runtime/install.py src/memo/watcher.py src/memo/cli_import.py src/memo/cli_dream.py src/memo/cli_ingest_daemon.py tests/test_runtime_isolation.py tests/test_watcher.py tests/test_whatsapp_ingest.py
git commit -m "feat: stage Memo-owned Synapse consumer replacements"
```

### Task 5: Extend the coordinated cutover with Synapse fencing

**Files:**
- Modify: `tools/memflow_absorption/schemas.py` (`CutoverControlRecord`, final verification fields)
- Modify: `tools/memflow_absorption/control_record.py` (monotonic Synapse retirement state)
- Modify: `tools/memflow_absorption/safety.py` (preflight and final independence checks)
- Modify: `tools/memflow_absorption/__main__.py` (`synapse-preflight`, `synapse-verify`)
- Test: `tests/tools/test_absorption_control_record.py`
- Test: `tests/tools/test_absorption_safety.py`
- Create: `tests/tools/test_synapse_cutover.py`

**Interfaces:**
- Consumes: signed capability manifest, consumer replacement plan, active-state
  migration receipt, and the existing CAS control record.
- Produces: `prepare_synapse_retirement(control: CutoverControlRecord, manifest: CapabilityManifest, consumer_plan: ConsumerReplacementPlan) -> CutoverControlRecord`,
  `verify_synapse_retired(control: VerifiedControlRecord, inventory: ConsumerInventory, manifest: SynapseRetirementManifest) -> IndependenceReceipt`,
  and a fail-closed `synapse.cutover.retired` refusal before any listener or
  worker starts.

Define `CutoverSafetyError(RuntimeError)`,
`commit_synapse_activation(control: CutoverControlRecord, epoch: int) -> VerifiedControlRecord`,
and `validate_synapse_request(control: VerifiedControlRecord, epoch: int) -> None`.

- [ ] **Step 1: Write the failing fence and epoch tests**

```python
def test_retired_synapse_refuses_startup_before_listener(control):
    with pytest.raises(CutoverSafetyError, match="synapse.cutover.retired"):
        prepare_synapse_retirement(control, manifest, consumer_plan)

def test_stale_synapse_epoch_cannot_write_after_memo_commit(control):
    committed = commit_synapse_activation(control, epoch=8)
    with pytest.raises(CutoverSafetyError, match="stale activation epoch"):
        validate_synapse_request(committed, epoch=7)
```

- [ ] **Step 2: Run the expected failing tests**

Run: `uv run --no-sync pytest tests/tools/test_synapse_cutover.py -q`

Expected: FAIL because Synapse-specific retirement state and negative checks do
not exist.

- [ ] **Step 3: Add monotonic Synapse retirement state**

Extend the existing control record with the signed Synapse manifest digest,
consumer-plan digest, retirement epoch, and final independence receipt. Permit
only `PREPARING -> READY -> QUIESCED -> STAGED -> COMMITTED -> VERIFIED`, with
`ABORTED` as the sole failure branch. Reject stale attempts, changed digests,
missing peer votes, or a second activation epoch.

- [ ] **Step 4: Add preflight and fail-closed verification**

Preflight must verify every Synapse process, port, LaunchAgent, MCP/gateway
route, shell/config path, and state root is represented by the signed plan.
Final verification must prove zero active references after stop and after
reboot. Read-only status remains available; startup, writes, and fallback
routes return `synapse.cutover.retired`.

- [ ] **Step 5: Run multi-peer safety checks**

```bash
uv run --no-sync pytest tests/tools/test_absorption_control_record.py tests/tools/test_absorption_safety.py tests/tools/test_synapse_cutover.py -v
uv run --no-sync ruff check tools/memflow_absorption tests/tools
uv run --no-sync mypy tools/memflow_absorption
```

Expected: offline peer, digest mismatch, stale epoch, resurrected process, and
loaded LaunchAgent all fail closed before Memo activation.

- [ ] **Step 6: Commit the Synapse cutover gate**

```bash
git add tools/memflow_absorption/schemas.py tools/memflow_absorption/control_record.py tools/memflow_absorption/safety.py tools/memflow_absorption/__main__.py tests/tools/test_absorption_control_record.py tests/tools/test_absorption_safety.py tests/tools/test_synapse_cutover.py
git commit -m "feat: fence Synapse retirement at the activation epoch"
```

### Task 6: Execute retirement cleanup and prove permanent independence

**Files:**
- Modify: `tools/memflow_absorption/inventory.py` (full negative scan)
- Modify: `tools/memflow_absorption/safety.py` (post-reboot proof and cleanup receipt)
- Modify: `tools/memflow_absorption/__main__.py` (`retirement-audit`)
- Create: `tests/tools/test_retirement_audit.py`
- Modify: existing Memo install/runtime documentation and release manifests only after the audit passes
- Modify/remove: Synapse source/config/tests listed by the signed `SynapseRetirementManifest`; no unlisted deletion is permitted

**Interfaces:**
- Consumes: verified activation control record, final Synapse retirement
  manifest, consumer replacement receipt, bounded data receipt, and process/
  launchd snapshots captured after reboot.
- Produces: `build_independence_receipt(...) -> IndependenceReceipt` and a
  `retirement-audit` report whose negative scan covers source, runtime,
  configuration, processes, ports, LaunchAgents, MCP registrations, wrappers,
  state roots, and package metadata.

- [ ] **Step 1: Write negative-scan tests**

```python
def test_retirement_audit_rejects_unlisted_reference(tmp_path, manifest):
    (tmp_path / "config.toml").write_text("command = 'synapse-mcp'", encoding="utf-8")
    with pytest.raises(InventoryError, match="unlisted active reference"):
        build_independence_receipt((tmp_path,), manifest=manifest)

def test_retirement_audit_accepts_clean_tree(tmp_path, manifest):
    receipt = build_independence_receipt((tmp_path,), manifest=manifest)
    assert receipt.status == "verified"
```

- [ ] **Step 2: Run the expected failing tests**

Run: `uv run --no-sync pytest tests/tools/test_retirement_audit.py -q`

Expected: FAIL because the final independence receipt does not exist.

- [ ] **Step 3: Implement the exact negative scan**

Reuse `_safe_files` and the signed manifest's source/test/golden lists. Scan
the installed Memo/Synapse roots, client configs, `~/Library/LaunchAgents`,
process/port snapshots, MCP gateway config, shell startup files, and package
metadata. Reject every active `synapse`, `SYNAPSE_*`, or Memflow-runtime
reference not explicitly represented as archived provenance. Do not follow
symlinks or delete anything during the scan.

- [ ] **Step 4: Execute the staged cleanup only after VERIFIED**

Disable KeepAlive, close/reconnect clients, stop Synapse services, remove the
staged Synapse LaunchAgents/configs/state only from paths recorded in the
signed plan, remove the temporary importer and legacy readers, and archive the
Synapse source commit read-only. The cleanup command must require the exact
verified control-record digest and refuse broad/unresolved paths.

- [ ] **Step 5: Reboot and run Memo-only smoke**

Run the post-reboot audit, `memo doctor --strict-runtime`, Memo MCP tool-list
smoke, cross-Mac handoff/delivery/ACK/presence/continuity smoke, and the full
negative scan. Record hashes/counts/epochs in Memo's audit receipt without
retaining discarded Synapse payloads.

- [ ] **Step 6: Run final verification and commit cleanup tooling/docs**

```bash
uv run --no-sync pytest tests/tools/test_retirement_audit.py tests/tools/test_absorption_*.py -v
uv run --no-sync ruff check tools/memflow_absorption src/memo
uv run --no-sync mypy tools/memflow_absorption src/memo
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing
git diff --check
git add tools/memflow_absorption src/memo/runtime src/memo/watcher.py src/memo/cli_import.py src/memo/cli_dream.py src/memo/cli_ingest_daemon.py docs/install-new-mac.md docs/reference.md pyproject.toml .claude-plugin/plugin.json server.json CHANGELOG.md
git commit -m "feat: complete Synapse retirement independence audit"
```

Expected: the Memo-only smoke passes after reboot and the final scan reports no
active Synapse or Memflow route.

## Plan self-review

- **Spec coverage:** inventory/signing is Task 1; bounded data migration is
  Task 2; Memo-native parity and chat deltas are Task 3; consumers/jobs are
  Task 4; atomic fencing is Task 5; cleanup and permanent independence are
  Task 6.
- **Completeness scan:** no unfinished markers or unspecified edge-case steps
  are present; conditional chat work is governed by a concrete parity
  report and signed disposition.
- **Type consistency:** Task 1 produces `CapabilityManifest`; Task 2 produces
  `SynapseDataReceipt`; Task 4 consumes `ConsumerInventory` and the manifest;
  Task 5 consumes those receipts and produces the verified control record;
  Task 6 consumes the verified record and produces `IndependenceReceipt`.
- **Scope:** each task has one independently testable authority: manifest,
  data import, parity, consumer staging, cutover fence, or cleanup audit.
