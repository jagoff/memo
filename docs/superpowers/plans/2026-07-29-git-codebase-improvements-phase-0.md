# Git Codebase Improvements Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible Admit/Defer/Reject evidence for the six approved Git-derived transfers without adding product behavior or permanent runtime abstractions.

**Architecture:** Phase 0 uses the existing pytest/eval surfaces plus one versioned Markdown receipt. Reproduced product gaps become regression tests owned by a later subsystem plan; candidates without a concrete gap remain documentation-only and add no runtime code. The clean baseline and all thresholds are frozen before any behavioral implementation.

**Tech Stack:** Python 3.11+, pytest, pytest-xdist, Ruff 0.15.22, mypy, SQLite, Markdown receipts, Git.

## Global Constraints

- The objective is improvement, not Git feature parity or feature accumulation.
- Every candidate ends in exactly one of `Admit`, `Defer`, or `Reject`.
- `Admit` requires a reproduced Memo gap, a falsifiable smallest slice, a primary gate, guardrails, and non-destructive rollback.
- `Defer` and `Reject` add no product runtime surface.
- Current Markdown remains authoritative for current memory state; SQLite projections remain rebuildable.
- New behavioral paths, if later admitted, begin default-off through `src/memo/flags.py`.
- No module-level MLX or MLX-LM import is permitted.
- Retrieval changes must pass `memo eval recall` in addition to focused tests.
- CI order is Ruff, mypy, then pytest.
- Work only on branch `feat/git-codebase-improvements` in worktree `.worktrees/git-codebase-improvements`.
- Preserve unrelated changes in every other checkout and branch.

---

### Task 1: Freeze the clean baseline and budgets

**Files:**
- Create: `docs/eval/2026-07-29-git-improvements-phase-0.md`
- Reference: `docs/superpowers/specs/2026-07-29-git-codebase-improvements-design.md`

**Interfaces:**
- Consumes: approved six-transfer design and clean worktree at commit `0c3224776b74e2115a21b41ca09434212dfefb69`.
- Produces: a receipt section named `Baseline and frozen gates` that later tasks append to.

- [ ] **Step 1: Create the receipt with the exact baseline**

```markdown
# Git-derived improvements: Phase 0 admission receipt

Date: 2026-07-29
Base commit: `0c3224776b74e2115a21b41ca09434212dfefb69`
Design: `docs/superpowers/specs/2026-07-29-git-codebase-improvements-design.md`

## Baseline and frozen gates

- Environment: editable install with `dev` and `http` extras.
- Focused HTTP baseline: `13 passed in 3.82s`.
- Non-slow baseline: `6065 passed, 18 skipped in 41.38s`.
- Correctness: no existing passing test may regress.
- Retrieval quality: no expected ID may be lost from the committed recall labels.
- Retrieval latency: warm p50 may not regress by more than 5%; any result within the
  benchmark noise band is treated as no improvement.
- Write latency: warm p50 may not regress by more than 10% for a correctness fix.
- Storage: a narrowed history fix may add at most one compressed/current body snapshot
  per destructive event; a content-addressed revision experiment must publish its own
  measured amplification before admission.
- Maintainability: an admitted abstraction must consolidate or delete duplicated logic;
  a wrapper that leaves all previous mechanisms intact fails.
- Rollback: disabling or reverting a new path must not delete current Markdown or require
  a destructive downgrade.
```

- [ ] **Step 2: Verify the baseline commands remain green**

Run:

```bash
uv run --no-sync pytest tests/test_cli_http.py tests/test_server_http.py -q
uv run --no-sync pytest -m 'not slow' -n auto --timeout=120 -q
```

Expected: `13 passed`; then `6065 passed, 18 skipped`.

- [ ] **Step 3: Commit the frozen baseline**

```bash
git add -f docs/eval/2026-07-29-git-improvements-phase-0.md
git commit -m "docs: freeze Git improvement admission gates"
```

### Task 2: Characterize federation partial-state risk

**Files:**
- Modify: `tests/test_definitive_memory.py`
- Modify: `docs/eval/2026-07-29-git-improvements-phase-0.md`

**Interfaces:**
- Consumes: `FederationManager.import_bundle(...) -> dict[str, Any]` and the existing `_target_config` test helper.
- Produces: regression test `test_federation_import_rolls_back_current_state_when_a_later_save_fails`.

- [ ] **Step 1: Add a strict expected-failure characterization**

Append this test beside the existing federation import tests:

```python
@pytest.mark.xfail(
    strict=True,
    reason="phase-0: federation currently leaves earlier records visible after a later save fails",
)
def test_federation_import_rolls_back_current_state_when_a_later_save_fails(
    mem_with_stub,
    tmp_cfg,
    tmp_path,
    monkeypatch,
) -> None:
    key = b"0123456789abcdef0123456789abcdef"
    owner = "owner-fer"
    for title in ("first", "second"):
        mem_with_stub.save(
            content=f"{title} shared body",
            title=title,
            extra={
                "visibility": "shared",
                "owner_principal": owner,
                "principals": ["bob"],
            },
            auto_project=False,
        )
    bundle_path = tmp_path / "atomicity.memo-federation.json"
    mem_with_stub.federation.export_bundle(
        bundle_path,
        principal="bob",
        owner_principal=owner,
        signing_key=key,
    )

    target = Memory(_target_config(tmp_cfg, tmp_path))
    original_save = target.save
    calls = 0

    def fail_second_save(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoError("seeded second-save failure")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(target, "save", fail_second_save)
    try:
        result = target.federation.import_bundle(
            bundle_path,
            principal="bob",
            signing_key=key,
        )
        assert result["failed"] == 1
        assert target.store.count() == 0
        assert target.list(limit=10) == []
        assert result["journal"]["imported"] == 0
    finally:
        target.close()
```

- [ ] **Step 2: Run the characterization**

Run:

```bash
uv run --no-sync pytest tests/test_definitive_memory.py::test_federation_import_rolls_back_current_state_when_a_later_save_fails -rxX -q
```

Expected: one strict XFAIL because the first imported memory remains in the current store.

- [ ] **Step 3: Record the decision**

Append:

```markdown
## 1. Atomic mutation and quarantine

Decision: **Admit, narrowed**

Observed gap: a seeded failure on the second save leaves the first federation record in
current Markdown/searchable state, while operational journal import still runs.

Smallest admitted slice: make one federation bundle import all-or-nothing in final current
state, keep operational events unapplied on memory failure, and emit an honest rollback
receipt. The full cross-process read-barrier transaction and general importer framework are
deferred until this slice proves value.

Primary gate: after a seeded failure at every imported item boundary, zero bundle records
are current/searchable and zero bundle operational events are imported.

Guardrails: successful and idempotent imports keep their current API; rollback failure is
typed and fail-closed; no second transaction framework or daemon is introduced.

Rollback: revert the federation-only rail; current Markdown remains readable and no schema
downgrade is required.
```

- [ ] **Step 4: Commit the characterization**

```bash
git add tests/test_definitive_memory.py
git add -f docs/eval/2026-07-29-git-improvements-phase-0.md
git commit -m "test: reproduce partial federation imports"
```

### Task 3: Characterize historical exactness and existing CAS overlap

**Files:**
- Modify: `docs/eval/2026-07-29-git-improvements-phase-0.md`
- Reference: `tests/test_time_machine.py`
- Reference: `src/memo/history.py`
- Reference: `src/memo/versioning.py`
- Reference: `src/memo/artifact_store.py`

**Interfaces:**
- Consumes: existing exactness assertion `test_snapshot_between_save_and_delete_includes_record`.
- Produces: a narrowed admission decision; no new runtime type.

- [ ] **Step 1: Run the exactness characterization**

Run:

```bash
uv run --no-sync pytest tests/test_time_machine.py::test_snapshot_between_save_and_delete_includes_record -q
```

Expected: PASS while explicitly confirming `body_unavailable is True`.

- [ ] **Step 2: Inventory overlapping stores**

Run:

```bash
rg -n "class (HistoryStore|VersionStore|ContentAddressedArtifactStore)|body_unavailable|delete_versions" \
  src/memo/history.py src/memo/versioning.py src/memo/time_machine.py \
  src/memo/artifact_store.py src/memo/memory/delete_ops.py
uv run --no-sync pytest \
  tests/test_memory_write.py::test_concurrent_topic_key_save_keeps_markdown_fts_and_vector_coherent \
  tests/test_memory_write.py::test_two_memory_instances_do_not_lose_concurrent_appends \
  tests/test_store.py::test_concurrent_writes_and_reads_do_not_collide \
  -q
```

Expected: three independent history/CAS mechanisms and a destructive-event body gap; the
existing write-lock/optimistic-retry concurrency baseline passes.

- [ ] **Step 3: Record the decision**

Append:

```markdown
## 2. Immutable revisions, refs, verifier, and recovery

Decision: **Admit only the exact-delete-history slice; Defer the revision archive**

Observed gap: time-machine reconstruction before a later delete returns the record but marks
its body unavailable. Memo already has `HistoryStore`, `VersionStore`, and
`ContentAddressedArtifactStore`; adding a fourth historical authority before convergence
would violate the maintainability gate.

Smallest admitted slice: preserve the canonical pre-delete body and tags in the existing
append-only delete event and teach `time_machine.reconstruct` to consume that snapshot.

Primary gate: reconstruction immediately before a later delete returns the exact body and
tags, while old history rows without snapshots remain explicitly unavailable.

Guardrails: current Markdown authority is unchanged; no new database or public command;
legacy rows remain readable; hard-delete behavior and portable backup remain compatible.

Deferred slice: immutable revision envelopes, heads, reflog, `fsck`, lost-found, and
historical-store retirement require a separate shadow-write/storage-amplification plan.

Rollback: readers ignore the additive delete-event fields; old and new history databases
remain readable.
```

- [ ] **Step 4: Commit the exactness decision**

```bash
git add -f docs/eval/2026-07-29-git-improvements-phase-0.md
git commit -m "docs: narrow immutable history admission"
```

### Task 4: Characterize retrieval filter crowd-out

**Files:**
- Modify: `tests/test_search_date_filters.py`
- Modify: `docs/eval/2026-07-29-git-improvements-phase-0.md`

**Interfaces:**
- Consumes: `Memory.search(..., mode="bm25", date_from=..., exclude_tags=...)`.
- Produces: regression tests `test_bm25_date_filter_does_not_spend_limit_on_ineligible_hits` and `test_bm25_tag_filter_does_not_spend_limit_on_ineligible_hits`.

- [ ] **Step 1: Add strict expected-failure characterizations**

```python
@pytest.mark.xfail(
    strict=True,
    reason="phase-0: BM25 applies date filtering after the backend limit",
)
def test_bm25_date_filter_does_not_spend_limit_on_ineligible_hits(mock_memory):
    old = [
        mock_memory.save(
            content="crowdneedle crowdneedle crowdneedle",
            title=f"Old {i}",
            auto_project=False,
        )
        for i in range(25)
    ]
    eligible = mock_memory.save(
        content="crowdneedle",
        title="Eligible",
        auto_project=False,
    )
    with mock_memory.store._conn:
        mock_memory.store._conn.executemany(
            "UPDATE meta SET updated = ? WHERE id = ?",
            [("2026-01-01T00:00:00+00:00", record.id) for record in old],
        )
        mock_memory.store._conn.execute(
            "UPDATE meta SET updated = ? WHERE id = ?",
            ("2026-07-01T00:00:00+00:00", eligible.id),
        )

    hits = mock_memory.search(
        "crowdneedle",
        mode="bm25",
        date_from="2026-06-01",
        limit=1,
    )

    assert [hit.id for hit in hits] == [eligible.id]


@pytest.mark.xfail(
    strict=True,
    reason="phase-0: BM25 applies excluded-tag filtering after the backend limit",
)
def test_bm25_tag_filter_does_not_spend_limit_on_ineligible_hits(mock_memory):
    for i in range(25):
        mock_memory.save(
            content="tagcrowd tagcrowd tagcrowd",
            title=f"Blocked {i}",
            tags=["blocked"],
            auto_project=False,
        )
    eligible = mock_memory.save(
        content="tagcrowd",
        title="Eligible",
        auto_project=False,
    )

    hits = mock_memory.search(
        "tagcrowd",
        mode="bm25",
        exclude_tags={"blocked"},
        limit=1,
    )

    assert [hit.id for hit in hits] == [eligible.id]
```

- [ ] **Step 2: Run the crowd-out characterizations**

Run:

```bash
uv run --no-sync pytest \
  tests/test_search_date_filters.py::test_bm25_date_filter_does_not_spend_limit_on_ineligible_hits \
  tests/test_search_date_filters.py::test_bm25_tag_filter_does_not_spend_limit_on_ineligible_hits \
  -rxX -q
```

Expected: two strict XFAIL results caused by post-limit filtering.

- [ ] **Step 3: Record the decision**

Append:

```markdown
## 4. Retrieval planning and backend filter pushdown

Decision: **Admit, narrowed**

Observed gap: BM25/exact/fuzzy candidate generation spends its limit before applying common
date and excluded-tag filters. Eligible lower-ranked records can therefore be crowded out,
while vector search already widens/pushes the same predicates.

Smallest admitted slice: extend existing BM25/fuzzy store methods with the two common
predicates and over-fetch only when a backend cannot execute them before its own limit.
Do not add `SearchPlan`, `SearchFilter`, or a planner class.

Primary gate: all modes return the same eligible record in the adversarial matrix at
`limit=1`; candidate count remains bounded.

Guardrails: committed recall labels lose no expected ID; ranking is unchanged when filters
are absent; warm p50 is within 5% of baseline; no new flag is needed for a correctness fix.

Rollback: revert the extra store parameters; no data migration exists.
```

- [ ] **Step 4: Commit the retrieval characterization**

```bash
git add tests/test_search_date_filters.py
git add -f docs/eval/2026-07-29-git-improvements-phase-0.md
git commit -m "test: reproduce BM25 filter crowd-out"
```

### Task 5: Inventory context, tracing, and maintenance without adding wrappers

**Files:**
- Modify: `docs/eval/2026-07-29-git-improvements-phase-0.md`
- Reference: `src/memo/dev_audit.py`
- Reference: `src/memo/trace.py`
- Reference: `src/memo/memory/search_ops.py`
- Reference: `src/memo/cli_maintain.py`
- Reference: `src/memo/cli_dream.py`
- Reference: `src/memo/maint_server.py`
- Reference: `src/memo/session.py`

**Interfaces:**
- Consumes: existing AST environment-read audit, native trace propagation, search trace, and maintenance entry points.
- Produces: three explicit no-runtime decisions.

- [ ] **Step 1: Run the explicit-context audit**

Run:

```bash
uv run --no-sync pytest tests/test_dev_audit.py -q
rg -n "os\\.environ\\.get\\(\"MEMO_|ambient_trace\\(|ActorIdentity\\(|current_project" \
  src/memo -g '*.py'
```

Expected: the AST ratchet passes; raw environment reads remain classified at boundary files.

- [ ] **Step 2: Run trace and maintenance inventories**

Run:

```bash
uv run --no-sync pytest tests/test_trace_id_env.py tests/test_memory_search.py tests/test_maintain.py -q
rg -n "@cli\\.command|def .*maint|sleep_cycle|idle.*maint|maint-daemon|dream" \
  src/memo/cli_maintain.py src/memo/cli_dream.py src/memo/maint_server.py \
  src/memo/session.py src/memo/runtime.py
```

Expected: existing trace/search-trace tests pass; multiple maintenance surfaces are listed,
but no measured overlap or wasted-work receipt is available.

- [ ] **Step 3: Record the three decisions**

Append:

```markdown
## 3. Explicit operation context and plumbing/porcelain

Decision: **Defer**

Evidence: `Config`, explicit `ActorIdentity`, native trace scope, project tags, and the
raw-environment AST ratchet already constrain the important boundaries. No cross-vault,
cross-principal, or signature-growth failure was reproduced.

Re-entry gate: a concrete context-propagation bug or a measured reduction in repeated
parameters/ambient reads must be shown before introducing a context type.

## 5. Structured tracing plus perf/fuzz/fault discipline

Decision: **Defer runtime tracing; Admit test discipline inside admitted slices**

Evidence: Memo already propagates a native trace ID and `search_with_trace` exposes
candidate stages. No diagnosis incident was reproduced that requires a second event system,
and no sampled-trace overhead budget has been measured.

Re-entry gate: a fault that cannot be localized with existing receipts/traces plus a
disabled/sampled overhead benchmark. Admitted transaction, history, and retrieval slices
must still add deterministic fault and complexity tests.

## 6. Need-driven incremental maintenance

Decision: **Defer**

Evidence: maintenance entry points are fragmented, but Phase 0 has no receipt proving
incompatible overlap, avoidable work, or a lock/recovery failure. A registry now would be a
wrapper around live schedulers and fail the consolidation gate.

Re-entry gate: two existing paths must be shown scheduling the same or incompatible work,
with before/after code and execution counts that a single task descriptor can reduce.
```

- [ ] **Step 4: Commit the inventories**

```bash
git add -f docs/eval/2026-07-29-git-improvements-phase-0.md
git commit -m "docs: classify context trace and maintenance transfers"
```

### Task 6: Close Phase 0 and hand off only admitted slices

**Files:**
- Modify: `docs/eval/2026-07-29-git-improvements-phase-0.md`

**Interfaces:**
- Consumes: all six admission sections.
- Produces: exact implementation-plan list for three narrowed slices.

- [ ] **Step 1: Append the final decision table**

```markdown
## Final Phase 0 decisions

| Transfer | Decision | Next artifact |
| --- | --- | --- |
| Atomic mutation/quarantine | Admit, narrowed | federation final-state rollback plan |
| Immutable revisions/refs/fsck | Admit exact-delete snapshot; defer archive | exact historical delete plan |
| Explicit context/plumbing | Defer | none |
| Retrieval planner/pushdown | Admit, narrowed | BM25/fuzzy common-filter plan |
| Structured Trace2/perf/fuzz | Defer runtime; use test discipline | none |
| Need-driven maintenance | Defer | none |

No public command, flag, database, daemon, context wrapper, planner class, revision archive,
or maintenance registry was added in Phase 0.
```

- [ ] **Step 2: Run Phase 0 quality gates**

Run:

```bash
uv run --no-sync ruff check src/memo tests
uv run --no-sync mypy src/memo
uv run --no-sync pytest \
  tests/test_definitive_memory.py \
  tests/test_search_date_filters.py \
  tests/test_time_machine.py \
  tests/test_dev_audit.py \
  tests/test_trace_id_env.py \
  tests/test_maintain.py \
  -q
```

Expected: Ruff and mypy pass; pytest passes with exactly three strict XFAIL
characterizations and no unexpected XPASS.

- [ ] **Step 3: Commit the closed receipt**

```bash
git add -f docs/eval/2026-07-29-git-improvements-phase-0.md
git commit -m "docs: close Git improvement phase zero"
```

- [ ] **Step 4: Create separate plans for admitted slices**

Create:

```text
docs/superpowers/plans/2026-07-29-federation-final-state-rollback.md
docs/superpowers/plans/2026-07-29-exact-delete-history.md
docs/superpowers/plans/2026-07-29-search-filter-pushdown.md
```

Each plan must remove the matching `xfail`, implement only its narrowed slice, repeat its
primary and guardrail gates, and define a standalone revert boundary.
