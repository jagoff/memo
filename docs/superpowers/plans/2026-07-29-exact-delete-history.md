# Exact Delete History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make time-machine snapshots immediately before a later delete return the exact deleted body and tags while keeping legacy history rows readable.

**Architecture:** Extend the existing `events.delta_json` envelope for delete events with a private `_snapshot` object. The delete path captures canonical body/tags before removing Markdown; `time_machine.reconstruct` uses the snapshot when present and retains `body_unavailable=True` for legacy rows.

**Tech Stack:** Python, SQLite, JSON, pytest, Ruff, mypy.

## Global Constraints

- Do not add a database, revision archive, public command, flag, or runtime dependency.
- Current Markdown remains authoritative for current state.
- The history sidecar remains best-effort and must never make an already-completed delete fail.
- Legacy delete rows with `delta_json IS NULL` remain readable and explicitly unavailable.
- The additive payload must be JSON serializable and preserve Unicode.
- CI order is Ruff, mypy, then pytest.

---

### Task 1: Store an additive pre-delete snapshot

**Files:**
- Modify: `src/memo/history.py:194`
- Modify: `src/memo/memory/delete_ops.py:225`
- Modify: `tests/test_history_store.py`

**Interfaces:**
- Consumes: canonical body read once from Markdown by `_delete_locked`.
- Produces: `HistoryStore.log_delete(..., tags: list[str] | None = None, body: str | None = None) -> None`.

- [ ] **Step 1: Write the failing store test**

```python
def test_log_delete_preserves_optional_snapshot(tmp_path: Path) -> None:
    history = HistoryStore(tmp_path / "history.db", device_id="local")
    try:
        history.log_delete(
            ts="2026-01-01T00:00:00Z",
            record_id="r1",
            title="Deleted",
            type_="decision",
            tags=["one", "dos"],
            body="cuerpo exacto",
        )
        event = history.list_recent(record_id="r1")[0]
        assert event["delta"] == {
            "_snapshot": {"tags": ["one", "dos"], "body": "cuerpo exacto"}
        }
    finally:
        history.close()
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
uv run --no-sync pytest --basetemp=/tmp/memo-git-improvements.yVRZ6x \
  tests/test_history_store.py::test_log_delete_preserves_optional_snapshot -q
```

Expected: FAIL because `log_delete` does not accept `tags` or `body`.

- [ ] **Step 3: Implement the additive envelope and wire the delete path**

Change the signature and insert:

```python
def log_delete(
    self,
    *,
    ts: str,
    record_id: str,
    title: str,
    type_: str,
    tags: list[str] | None = None,
    body: str | None = None,
) -> None:
    snapshot = None
    if tags is not None or body is not None:
        snapshot = json.dumps(
            {"_snapshot": {"tags": list(tags or []), "body": body}},
            ensure_ascii=False,
        )
    try:
        with self._tx() as cx:
            cx.execute(
                "INSERT INTO events "
                "(ts, op, record_id, title, type, delta_json, device_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, "delete", record_id, title, type_, snapshot, self.device_id),
            )
    except (sqlite3.Error, TypeError, ValueError) as exc:
        self._error_count += 1
        _log.warning("history log_delete failed (id=%s): %s", record_id[:8], exc)
```

In `_delete_locked`, assign
`canonical_body = self._read_body(str(r["path"]))` before write-policy
preflight, pass that same value to preflight, and later pass
`tags=list(r.get("tags") or ())` plus `body=canonical_body` to `log_delete`.
Do not use the rebuildable FTS body as the historical authority.

- [ ] **Step 4: Run store and delete regression tests**

Run:

```bash
uv run --no-sync pytest --basetemp=/tmp/memo-git-improvements.yVRZ6x \
  tests/test_history_store.py tests/test_write_ops_failure.py tests/test_memory_history.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memo/history.py src/memo/memory/delete_ops.py tests/test_history_store.py
git commit -m "feat: preserve exact pre-delete history"
```

### Task 2: Reconstruct exact deleted bodies and keep legacy fallback

**Files:**
- Modify: `src/memo/time_machine.py:258`
- Modify: `tests/test_time_machine.py:71`

**Interfaces:**
- Consumes: delete event `delta["_snapshot"]`.
- Produces: exact `SnapshotRecord.body`, `tags`, and `body_unavailable` semantics.

- [ ] **Step 1: Change the existing characterization into the required behavior**

Replace the final assertion in
`test_snapshot_between_save_and_delete_includes_record` with:

```python
    restored = snap.records[rec.id]
    assert restored.body == "bye"
    assert restored.tags == []
    assert restored.body_unavailable is False
```

Add a legacy compatibility test:

```python
def test_snapshot_legacy_delete_without_snapshot_stays_unavailable(mem: Memory) -> None:
    rec = mem.save(content="legacy body", title="Legacy", type_="note")
    middle = _now()
    time.sleep(0.01)
    mem.history.log_delete(
        ts=_now().isoformat(),
        record_id=rec.id,
        title=rec.title,
        type_=rec.type,
    )
    rec_path = mem.cfg.memory_dir / rec.path
    rec_path.unlink()
    mem.store.delete(rec.id)

    restored = reconstruct(mem, as_of=middle).records[rec.id]

    assert restored.body is None
    assert restored.body_unavailable is True
```

- [ ] **Step 2: Verify the exactness test fails**

Run:

```bash
uv run --no-sync pytest --basetemp=/tmp/memo-git-improvements.yVRZ6x \
  tests/test_time_machine.py::test_snapshot_between_save_and_delete_includes_record \
  tests/test_time_machine.py::test_snapshot_legacy_delete_without_snapshot_stays_unavailable -q
```

Expected: exactness FAILS; legacy fallback PASSES.

- [ ] **Step 3: Consume the optional snapshot**

In the delete branch:

```python
            delta = ev.get("delta")
            snapshot = delta.get("_snapshot") if isinstance(delta, dict) else None
            snapshot = snapshot if isinstance(snapshot, dict) else {}
            body = snapshot.get("body")
            has_body = isinstance(body, str)
            raw_tags = snapshot.get("tags")
            tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
            snap[rid] = SnapshotRecord(
                id=rid,
                title=ev.get("title") or "(deleted)",
                type=ev.get("type") or "note",
                tags=tags,
                created=None,
                updated=None,
                body=body if has_body else None,
                body_unavailable=not has_body,
                _deleted_after=ts,
            )
```

- [ ] **Step 4: Run historical tests**

Run:

```bash
uv run --no-sync pytest --basetemp=/tmp/memo-git-improvements.yVRZ6x \
  tests/test_time_machine.py tests/test_history_store.py tests/test_sync.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memo/time_machine.py tests/test_time_machine.py
git commit -m "feat: reconstruct deleted memory bodies exactly"
```

### Task 3: Verify the narrowed slice

**Files:**
- Modify: `docs/eval/2026-07-29-git-improvements-phase-0.md`

**Interfaces:**
- Consumes: exact-history implementation.
- Produces: post-implementation result in the admission receipt.

- [ ] **Step 1: Run quality gates**

```bash
uv run --no-sync ruff check src/memo/history.py src/memo/time_machine.py \
  src/memo/memory/delete_ops.py tests/test_history_store.py tests/test_time_machine.py
uv run --no-sync mypy src/memo
uv run --no-sync pytest --basetemp=/tmp/memo-git-improvements.yVRZ6x \
  tests/test_time_machine.py tests/test_history_store.py tests/test_sync.py \
  tests/test_write_ops_failure.py tests/test_memory_history.py -q
```

Expected: all commands pass.

- [ ] **Step 2: Record the result**

Append to section 2:

```markdown
Implementation result: **Admitted and shipped on the feature branch.** New delete events
preserve exact body/tags in the existing history envelope; legacy events remain explicitly
unavailable. No new store, command, flag, or authority was added.
```

- [ ] **Step 3: Commit**

```bash
git add -f docs/eval/2026-07-29-git-improvements-phase-0.md
git commit -m "docs: record exact history improvement"
```
