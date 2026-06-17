# Design: memo two-tier sync (local vs git) + identity + durable capture

**Date:** 2026-06-17
**Goal:** maximize how much durable knowledge is available to every session — across
multiple Macs AND multiple concurrent sessions on the same Mac — in the least time,
without losing info in long sessions.

## Problem (verified in code)

Two distinct bottlenecks, conflated today:

1. **Capture** — durable insight is extracted from the live session into `.md` only on
   `Stop` (`memo capture-stop`) and on context-overflow snapshots. A long session without
   a Stop leaves insight only in the live transcript (not in memo). `memory_save` writes
   `.md` immediately (disk-durable), so explicit saves are safe; ambient insight is the risk.
2. **Propagation** — captured `.md` reaches other sessions/machines via git (`memo-sync`
   repo → GitHub), push on Stop, pull on SessionStart.

Gaps found:
- `device_id` exists+persisted (`state_dir/.device_id`, random uuid) but is NOT wired into
  the git-sync path and is NOT aligned with memflow's machine identity (trinity disagrees).
- `SyncCoordinator` mentioned in CLAUDE.md is NOT actually wired in `sync*.py`.
- No machine-level lock/leader → concurrent same-machine sessions racing `sync push`.
- `sync_push` does NOT pull-rebase before push → concurrent-machine push rejected → stranded.
- git push/pull fires per-session unconditionally — same-machine sessions do redundant git
  even though they already share the local sqlite store.

## Two explicit models

- **LOCAL (intra-machine):** all sessions on a Mac share one `data_dir` + `memvec.db`
  (WAL, thread-local conns). Capture → visible to sibling sessions on their next recall.
  NO git. Zero-config, always on.
- **REMOTE (inter-machine):** the git `memo-sync` remote is the ONLY cross-machine channel,
  owned by ONE coordinator per machine.

## Sections

### 1. Identity layer — `src/memo/identity.py` (new)
One unit answering "who am I?", stable + shared:
- `machine_id`: reuse `consciousness_contracts.identity` (trinity agrees) → fallback to
  `cfg.device_id` (already persisted) + `hostname` for a human label. Shape `{id, hostname, label}`.
- `session_id`: the client-supplied id already flowing into recall.log, now first-class.
- `terminal`: TTY (`os.ttyname`) when available — so memflow can address "terminal X" in
  future (memo only EXPOSES it; addressing is memflow's job — YAGNI here).
- Used for: `history.device_id = machine_id` (fix "unknown"), git commit attribution
  (`sync: <machine> · <session>`), optional provenance stamp in memoria `extra`, and the
  coordinator's owner identity.

### 2. LOCAL tier explicit
- Document + test the invariant: intra-machine visibility is via the shared index, never git.
- `sync_tier(cfg)` helper → `"local"` (always) vs `"remote"` (git remote configured). Collapses
  the today-implicit, scattered decision into one place.
- A session never needs to push for a sibling on the same Mac to see its writes.

### 3. Machine-level git coordinator — single owner
All triggers (per-prompt hook, maint-daemon tick, Stop) call ONE `sync_once(cfg, store, mem)`
guarded by a machine file lock:
- **Lock** (`flock` on `state_dir/.sync.lock`): lock holder does git; others skip (their changes
  are already in the shared store → the leader carries them). Kills same-machine races.
- **Debounce** (`.sync_auto_ts`, already built): coalesce rapid saves into one push.
- **pull-rebase-before-push**: `sync_once` = `fetch → rebase --autostash → reindex → push`.
  Remote-advanced no longer rejects. Real `.md` conflict → abort + `pending` marker (manual).
- **Commit attributed** to identity.
- Triggers become thin: hook + Stop just call `sync_once()` async; maint-daemon adds a periodic
  pull tick calling the same `sync_once()`. No new daemon (reuse existing — option C).

### 4. Durable capture cadence (in scope)
- Periodic incremental ambient capture: debounced trigger (every `MEMO_CAPTURE_INTERVAL_S` or
  N turns, async) runs capture over NEW turns since a watermark (last-captured turn) → bounded,
  no reprocessing. Writes `.md` → picked up by local index + coordinator.
- Reuses `capture-stop` logic; the watermark makes it incremental.

### 5. Recovery
- After pull → reindex → in local index. Session uses it via recall hook (every prompt, fresh
  read), El Briefing (SessionStart), `memory_search/ask`. Mid-session pull → long session
  converges without restart.
- Fix: invalidate the query-cache (LRU keyed by query text) when reindex brings changes, else a
  repeated query could serve stale results.
- Crash: `.md` already on disk → next SessionStart pull + briefing recovers.

### 6. Errors + testing
- Errors: lock held → skip (not error); push reject → rebase; real `.md` conflict → `pending` +
  manual; offline/auth → `pending` + retry; non-clone → local tier only (no error); reindex fail
  → log + retry next tick.
- Tests (2 clones + embed stub, no MLX): identity stable + contracts-aligned; `sync_tier`
  local/remote; `sync_once` pull-rebase-before-push with advanced remote; lock contention
  (2 procs → 1 does git); incremental capture respects watermark; cache invalidated after pull.

## Reuses existing (uncommitted) work
`sync status`, `sync auto` (evolves into the lock-guarded `sync_once`), `_pending_marker` +
stranded-commit retry, `memo doctor` sync line, dashboard sync chip, `MEMO_SYNC_*` flags.

## Out of scope (YAGNI)
- memflow addressing "terminal X" to assign tasks (memo only EXPOSES identity).
- Real-time cross-machine push (sub-second); debounce/interval is intentional.
