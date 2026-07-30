# Federation Final-State Rollback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure a failed federation bundle import leaves no partial bundle state current/searchable and imports no operational events.

**Architecture:** Snapshot only records that the deterministic federation topic/exact identity can mutate, apply memories sequentially, and keep a compact rollback journal of created IDs and preimages. On any memory failure, restore revised records and purge newly created records before returning; the operational ledger seam runs only after all durable memories succeed. This is final-state rollback for one importer, not a general transaction or cross-process read barrier.

**Tech Stack:** Python dataclasses, existing Memory write/delete rails, pytest fault injection.

## Global Constraints

- Do not claim filesystem-wide or cross-process atomic visibility.
- Do not add a daemon, database, public command, flag, or general transaction framework.
- Successful and idempotent federation response fields remain compatible.
- Rollback uses existing write/delete policy and fails closed with `FederationError` if restoration fails.
- Operational events are never imported after any memory item fails.
- Current Markdown is the authority; rollback must restore exact title/type/tags/body/extra for revised records.
- CI order is Ruff, mypy, then pytest.

---

### Task 1: Turn the Phase 0 failure into a passing final-state contract

**Files:**
- Modify: `src/memo/federation.py`
- Modify: `tests/test_definitive_memory.py`

**Interfaces:**
- Produces private `_ImportPreimage` and `_AppliedImport` dataclasses.
- Produces private `FederationManager._rollback_import(...) -> None`.

- [ ] **Step 1: Remove the strict `xfail` marker**

Run the existing regression and confirm it FAILS because one record remains.

- [ ] **Step 2: Add rollback journal types**

```python
@dataclass(frozen=True)
class _ImportPreimage:
    id: str
    title: str
    type: str
    tags: tuple[str, ...]
    body: str
    extra: dict[str, Any]


@dataclass(frozen=True)
class _AppliedImport:
    id: str
    preimage: _ImportPreimage | None
```

- [ ] **Step 3: Capture deterministic preimages**

Before each save, sanitize the item with `sanitize_memory_input`, normalize tags,
truncate content to `cfg.max_content_chars`, derive `identity_keys(...,
auto_project=False)`, and query topic identity first, then exact identity. Resolve
the single candidate through `memory.get`; ambiguous candidates are left to the
existing save conflict checks and cannot mutate.

- [ ] **Step 4: Implement reverse rollback**

For each applied change in reverse:

```python
if change.preimage is None:
    if self.memory.delete(change.id):
        self.memory.store.hard_delete(change.id)
else:
    before = change.preimage
    restored = self.memory.update(
        before.id,
        title=before.title,
        type_=before.type,
        tags=list(before.tags),
        content=before.body,
        extra=dict(before.extra),
        actor=ActorIdentity(actor_id="memo-federation-rollback", actor_kind="system"),
    )
    if restored is None:
        raise FederationError(f"federation rollback target disappeared: {before.id[:12]}")
```

Collect rollback errors and raise one typed `FederationError` naming affected IDs.

- [ ] **Step 5: Stop at first memory failure and gate the journal**

On a caught item error:

- append the existing error string;
- rollback all previously applied changes;
- do not process later items;
- set `journal` to `{"devices": 0, "imported": 0, "unchanged": 0}`;
- return the existing response envelope with `imported=0`, `revised=0`, and
  `failed=1`.

Move `ledger.import_events(operations)` below the successful-memory condition.

- [ ] **Step 6: Run the original and fault test**

```bash
uv run --no-sync pytest --basetemp=/tmp/memo-git-improvements.yVRZ6x \
  tests/test_definitive_memory.py::test_federation_import_rolls_back_current_state_when_a_later_save_fails \
  tests/test_definitive_memory.py::test_federation_enforces_acl_signature_and_idempotent_import -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/memo/federation.py tests/test_definitive_memory.py
git commit -m "fix: rollback failed federation imports"
```

### Task 2: Cover revised records, every boundary, and rollback failure

**Files:**
- Modify: `tests/test_definitive_memory.py`

- [ ] **Step 1: Parameterize failure boundaries**

Create a three-record bundle and fail calls 1, 2, and 3. Assert after each:

```python
assert target.store.count() == 0
assert target.list(limit=10) == []
assert result["imported"] == 0
assert result["revised"] == 0
assert result["journal"]["imported"] == 0
```

- [ ] **Step 2: Add revised-record restoration**

Import version 1 successfully, export version 2 with the same source record/topic,
inject a later-item failure, and assert the target's original body/title/tags/extra
are exact after rollback.

- [ ] **Step 3: Add fail-closed rollback test**

Monkeypatch `target.delete` or `target.update` to fail during rollback and assert
`FederationError` contains `rollback` and the affected ID prefix.

- [ ] **Step 4: Run federation tests**

```bash
uv run --no-sync pytest --basetemp=/tmp/memo-git-improvements.yVRZ6x \
  tests/test_definitive_memory.py tests/test_operational_memory.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_definitive_memory.py
git commit -m "test: fault inject federation rollback boundaries"
```

### Task 3: Verify and record the narrowed integrity improvement

**Files:**
- Modify: `docs/eval/2026-07-29-git-improvements-phase-0.md`

- [ ] **Step 1: Run quality gates**

```bash
uv run --no-sync ruff check src/memo/federation.py tests/test_definitive_memory.py
uv run --no-sync mypy src/memo
uv run --no-sync pytest --basetemp=/tmp/memo-git-improvements.yVRZ6x \
  tests/test_definitive_memory.py tests/test_operational_memory.py \
  tests/test_write_ops_failure.py -q
```

- [ ] **Step 2: Record limitations honestly**

Append:

```markdown
Implementation result: **Admitted as final-state rollback for federation only.** A failed
bundle restores/purges its applied memory changes and never imports operational events.
This does not claim a cross-process read barrier; the general mutation transaction remains
deferred.
```

- [ ] **Step 3: Commit**

```bash
git add -f docs/eval/2026-07-29-git-improvements-phase-0.md
git commit -m "docs: record federation rollback improvement"
```
