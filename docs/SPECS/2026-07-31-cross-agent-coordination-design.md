# Cross-Agent Coordination — Design

**Date:** 2026-07-31 · **Status:** approved (Fer) · **Decision:** full-LLM coordination.
Revokes the "no auto-action without human review" principle for the coordination
domain only (recorded in memo as a decision memory).

## Problem

memo is cross-machine aware but not cross-agent aware *in flight*. Real collisions
from 2026-07-31, all detected only after the fact:

1. Two agents fixed the same README banner in parallel (PR #140 vs #143).
2. One agent realigned the uv-tool runtime and broke another session's
   `com.memo.chat` daemon (crashloop under KeepAlive).
3. PR #144 merged to master mid-way through another agent's serialized
   dependabot merge queue, forcing extra update-branch rounds.

The corpus-level contradiction radar runs nightly over saved memories; nothing
correlates the *operational state of live sessions*.

## Goal

When two or more active agents touch the same resource (file, branch, daemon,
PR, topic), memo must (a) notice within minutes, (b) decide with the local 4B
helper LLM what each agent should do, and (c) deliver those directives into each
agent's next turn automatically.

## Non-goals (v1)

- No blocking/locking of agent actions (directives are injected instructions,
  not enforcement).
- No cross-machine live coordination (rides on existing git sync cadence).
- No 30B model use — 4B helper only (OOM constraint on this hardware).

## Architecture

Three pieces, following existing repo patterns:

### 1. `CoordinationStore` — sidecar sqlite (pattern: `proactive/store.py`)

`~/.local/share/memo/coordination.db`, WAL, busy_timeout. One table:

```sql
CREATE TABLE collisions (
  id            TEXT PRIMARY KEY,       -- hash(session_a, session_b, resource)
  session_a     TEXT NOT NULL,
  session_b     TEXT NOT NULL,
  resource      TEXT NOT NULL,          -- file path / branch / daemon label / topic
  kind          TEXT NOT NULL,          -- file | branch | daemon | pr | topic
  severity      TEXT NOT NULL,          -- info | warn | block
  rationale     TEXT,
  directive_a   TEXT NOT NULL,          -- instruction injected into session_a
  directive_b   TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'open',  -- open | delivered | resolved | stale
  created_at    TEXT NOT NULL,
  delivered_a   TEXT,                   -- ts of delivery to each side
  delivered_b   TEXT
);
```

Dedup by primary key: a recurring collision refreshes `created_at` only if
previously `resolved`/`stale`; open/delivered rows are not re-judged.

### 2. Scanner + LLM judge — `src/memo/coordination.py`

`scan_collisions(mem, cfg, *, now) -> ScanResult`:

1. **Gather** active sessions: `session.list_sessions()` filtered to
   `updated within ACTIVE_WINDOW` (default 6h), joined with
   `OperationalStore` focus items and each session's last `N=20` capture
   memories (`files_read`/`files_modified` stamps, titles, tags).
2. **Candidate pairs** (pure code, no LLM): for each pair of active sessions,
   overlap on (a) normalized file paths, (b) branch names found in capture
   metadata, (c) `com.*` launchd labels mentioned in titles/bodies,
   (d) title-token Jaccard over recent memories above `TOPIC_JACCARD=0.35`.
3. **Judge** each candidate with the 4B helper via `MLXChat` +
   `chat_with_timeout` (pattern: `temporal._classify_pair`): prompt carries both
   sessions' recent activity summaries and the overlapping resource; model
   returns strict JSON `{collision, severity, rationale, directive_a,
   directive_b}`. Non-JSON or timeout ⇒ candidate skipped (fail-open, logged).
4. **Persist** confirmed collisions to `CoordinationStore`.

### 3. Delivery — recall hook + trigger

- **Recall hook** (`cli_recall_hook.py`, at the `systemMessage` assembly): read
  `pending_directives(session_id)` — a single indexed sqlite read, well inside
  the 5s hook budget — append a `<memo-coordination>` block listing each open
  directive for this session, then stamp `delivered_<side>`. When both sides are
  delivered, status becomes `delivered`.
- **Trigger**: `run_watcher` loop gains a time-based check — if
  `now - last_scan >= COORD_SCAN_INTERVAL` (default 300s) it runs
  `scan_collisions` in a daemon thread (never blocks the reindex debounce).
  Manual: `memo coordinate scan`.

### CLI — `memo coordinate` group (pattern: `cli_contradict.py`)

- `scan` — run one scan now, print confirmed collisions.
- `status` — list open/delivered collisions.
- `resolve <id>` — mark resolved (either agent or the user).

### Config knobs (env, all optional)

- `MEMO_COORD_ENABLED` (default `1`)
- `MEMO_COORD_SCAN_INTERVAL` (default `300` s)
- `MEMO_COORD_ACTIVE_WINDOW` (default `21600` s)

## Failure modes

- LLM timeout/garbage → skip candidate, keep scanning (fail-open; a missed
  collision is the status quo ante, never worse).
- GPU lock contention: scan uses the shared helper-model path; if model load
  is locked, scan skips this round (recorded: cold-loads during daemon testing
  caused GPU flock timeouts).
- Session id absent in hook payload → no directives, no error.

## Test plan (TDD)

- Store: dedup, delivery stamping, status transitions.
- Candidate detection: file overlap, daemon label, branch, topic Jaccard —
  fixtures reproducing the three 2026-07-31 collisions.
- Judge: mocked `MLXChat` returning JSON / garbage / timeout.
- Hook injection: directive block appears once, marked delivered, second call
  injects nothing.
- CLI smoke: scan/status/resolve.
- Gates: quality-gate complexity budget, file <800 lines, coverage ≥80% on the
  new module.
