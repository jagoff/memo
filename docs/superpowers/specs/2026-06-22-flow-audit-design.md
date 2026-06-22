# Live flow audit — save / recovery / idle / github sync

**Date:** 2026-06-22
**Type:** one-time live end-to-end audit (no new code). Both Macs
(HOST local + USER@REDACTED-IP remote), both on memo 1.0.1.

## Goal

Confirm the four critical flows actually work end-to-end on the real installs,
not just in unit tests. Output: a pass/fail table per flow × machine + any bugs.

## Safety

Write/destructive flows run in an **isolated** `MEMO_DATA_DIR`/`MEMO_STATE_DIR`
(temp dir) → zero impact on the real corpus. Only the sync round-trip touches
the real corpus/GitHub, via a marked `AUDIT-PROBE-<ts>` memoria that is deleted
+ sync-propagated at the end.

## Checks

### 1. SAVE (markdown is the source of truth)
- `memo save` → `.md` on disk with `id`/`type` frontmatter → present in index
  (`get`/`search`) → semantic recall finds it.
- Failure path: index failure → `.md` still written + `embed_pending` stamp →
  `memo reindex` recovers it.

### 2. RECOVERY (the 1.0.1 guards on the real binary)
- Guard 1 live: populated index + empty data_dir → `reindex --rebuild` refuses.
- Guard 2 live: `sync bootstrap` over a broken clone refuses.
- Restore: `sync bootstrap` from GitHub into a temp dest rebuilds.

### 3. IDLE auto-save
- idle-daemon alive (pid) or auto-starts per machine.
- Evidence of real recent capture (log + watermark), no fake-turn injection.

### 4. GITHUB SYNC (real cross-mac round-trip)
- local `save AUDIT-PROBE-<ts>` → `sync once` (push) → remote `sync pull` →
  probe appears (`get`/`search`).
- Cleanup: `delete` probe → sync → gone on **both** (delete-propagation).

## Out of scope
No new code, no repeatable harness (separate decision). Report only.
