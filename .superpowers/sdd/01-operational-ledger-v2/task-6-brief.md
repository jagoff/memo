# Task 6 — canonical sessions over operational ledger v2

BASE: `6b68a260`

Status: brief frozen; production edits and RED execution have not started.
Tasks 2–5 remain technically green but independently unaccepted. v1 remains
the production authority, v2 remains dormant, and Memflow remains live.

## Owned paths

Production:

- create `src/memo/operational_sessions.py`
- modify `src/memo/operational_event_types.py`
- modify `src/memo/operation_views.py`
- modify `src/memo/session.py`
- modify `src/memo/server_core_history.py`
- modify `src/memo/server_session_patterns.py`

Tests:

- create `tests/test_operational_sessions_v2.py`
- extend `tests/test_operational_event_types.py`
- extend `tests/test_operation_views_v2.py`
- extend `tests/test_session.py`
- extend `tests/test_session_patterns.py`
- extend `tests/test_server_core_history.py`
- extend `tests/test_briefing_unified.py` only if canonical-session rendering
  requires it
- extend `tests/test_surface_profiles.py` and
  `tests/test_cli_mcp_surface_smoke.py` only for public-name assertions

Do not edit the frozen v1 ledger, v1 operational facade, migration format,
activation selector, federation, Memflow runtime, live state, or production
configuration.

## Authority and activation boundary

- The three v2 events below are the sole portable session authority.
  `operational.db`, legacy JSON sidecars, and the legacy memory-store
  `sessions` table are derived/local inputs, never competing authorities.
- `OperationalSessionService` is explicitly constructed against a v2
  `OperationalStore`, `OperationalViewStore`, and authenticated
  `PrincipalIdentity -> CommitContext` factory.
- Task 6 does not activate v2. `memo.session` accepts an explicitly installed
  runtime binding and delegates canonical mutations/reads to it. Without that
  binding, the frozen v1-compatible JSON behavior remains active until Task 7.
- Task 7 must install the binding only after verified v2 activation and remove
  it on facade close. No implicit construction, environment flag, dual-write,
  or automatic fallback from a failed v2 commit is allowed.

## Closed event vocabulary and payloads

The preliminary `session.status_changed` event is replaced before activation:

- `memo.operational.session.checkpointed.v1`
- `memo.operational.session.recoverable.v1`
- `memo.operational.session.terminated.v1`

Checkpointed payload:

```text
session_id, principal_id, project, workspace, status="active",
branch, head, summary, checkpointed_at, source_event_id
```

Recoverable payload:

```text
session_id, recoverable_at, reason
```

Terminated payload:

```text
session_id, terminated_at, summary
```

Payload fields are exact. IDs, project, workspace, principal, source event,
and timestamps are non-empty; branch, head, summary, and recoverable reason
may be empty strings. Timestamps are timezone-aware canonical UTC ISO-8601.

Command idempotency keys are caller-provided, normalized non-empty strings.
Commands use `target_id=session_id`,
`subject_uri=memo://session/<session_id>`, and the authenticated principal as
actor. No local path or prompt/transcript content may enter a command.

## Portable session model and monotonic reducer

`OperationalSession` contains:

```text
session_id, principal_id, project, workspace, status,
branch, head, summary, checkpointed_at, source_event_id,
recoverable_at, terminated_at, recoverable_reason, updated_event_id
```

Allowed transitions:

- absent → checkpointed/`active`
- `active` → checkpointed/`active`
- `active` → `recoverable`
- `active` or `recoverable` → `terminated`

Forbidden transitions:

- status events for an absent session
- `recoverable` → checkpointed/`active`
- every mutation after `terminated`
- duplicate terminal/recoverable transitions under a new event ID
- changing `principal_id`, `project`, or `workspace` for an existing session

Operational command idempotency handles exact replay. Reducer violations fail
closed with typed `OperationalError`; no row is overwritten.

`OperationalSessionService` exposes:

```text
checkpoint(identity, session_id, project, workspace, summary, branch, head,
           source_event_id, checkpointed_at, idempotency_key)
mark_recoverable(identity, session_id, reason, recoverable_at,
                 idempotency_key)
terminate(identity, session_id, summary, terminated_at, idempotency_key)
get(session_id)
latest_recoverable(project=None, workspace=None)
```

The injected clock is used only when an explicit timestamp is omitted. The
service returns the canonical row materialized after commit, including on
idempotent replay.

## Local artifacts and derived cache

- Portable: identity, project, workspace, lifecycle status, branch/head,
  summary, timestamps, and source event ID.
- Local-only: transcript/prompt paths, prompt trail, last user/assistant text,
  modified-file paths, running-summary internals, turn counters, recall/recap
  stamps, autosave/reflection timestamps, and any unknown legacy sidecar key.
- `session_local_artifacts` is local state and is excluded from operational
  state hashes and federation. A ledger/view rebuild preserves it.
- Local artifact writes are one SQLite transaction keyed by session ID and
  replace the prior local set deterministically. Values must be canonical JSON.
- The JSON sidecar remains an atomic derived cache containing the portable row
  plus local artifacts. Rebuilding a cache from the canonical row and local
  table must reproduce the same portable/local separation and must never emit
  a ledger event.

`LegacySessionMigrator.merge_legacy(json_checkpoint, sqlite_row)`:

- accepts `session_id` or legacy `id`;
- maps legacy `completed` to `terminated`;
- rejects incompatible session IDs, projects, or resolved workspaces;
- uses the newer source timestamp deterministically;
- produces one portable checkpoint plus detached local artifacts;
- derives `source_event_id` from canonical source bytes when absent;
- never writes.

## Public API migration

- Rename only the session lifecycle registrations
  `mem_session_start`/`mem_session_end` to
  `memo_session_start`/`memo_session_end`.
- No aliases remain. First-party source and tests contain no active
  `mem_session_` registration or call.
- Existing `memo_session_list`/`memo_session_get` read the installed canonical
  session runtime when present and use the legacy cache only before Task 7.
- Other established `mem_*` session-pattern tools are outside this naming
  change.

## RED-first contracts

1. New service import is absent.
2. Exact validators accept all three payloads and reject missing/unknown
   fields, local artifacts, invalid timestamps, empty required strings, and
   non-active checkpoint status.
3. Reducer lifecycle is monotonic and identity/project/workspace are immutable.
4. Same command/idempotency key replays exactly; changed request conflicts.
5. Terminated sessions cannot checkpoint or become recoverable.
6. `latest_recoverable` excludes active/terminated rows and orders
   deterministically by recoverable/checkpoint time and session ID.
7. Legacy merge separates every local artifact and rejects incompatible
   project/workspace/session identity.
8. Local artifacts never appear in event bytes, state hashes, or exported
   canonical session rows.
9. View rebuild preserves local artifacts while reproducing portable state.
10. Derived JSON cache rebuild is atomic and does not emit an event.
11. `memo_session_start`/`memo_session_end` exist; no lifecycle alias remains.
12. Default v1 session behavior remains compatible while the v2 runtime is
    uninstalled.
13. Frozen v1 operational sources remain byte-identical and no activation
    marker is created.

## Gates

- Focused session service, event/view, sidecar, history, briefing, and surface
  tests.
- Task 1–6 cumulative operational/definitive regression.
- Ruff and mypy over every touched path.
- Full non-slow suite.
- Frozen-v1 diff and activation-marker audit.
- Explicit-path technical commit and clean tracked worktree.
- Implementation report and review package.
- Independent specification/durability review and PASS before acceptance.
