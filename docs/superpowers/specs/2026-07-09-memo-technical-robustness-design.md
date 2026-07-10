# memo Technical Robustness Roadmap

Date: 2026-07-09
Status: design approved for implementation planning

## Summary

memo's previous improvement sprint closed the known verification-signal issues and
left `master` green. The next valuable track is technical robustness: reduce the
chance that sync, runtime, update, HTTP API, daemon, or recovery paths surprise
users when memo runs on real machines.

This design chooses a reliability-first roadmap with a minimum security lane. It
does not redesign memo's retrieval model or expand product scope. It strengthens
the operational contracts around the parts that can affect user data, local
runtime health, or exposed interfaces.

## Context

The prior deep-improvement roadmap already covered verification hygiene, eval
profile clarity, error policy, flag discipline, and resource cleanup. Recent work
also pushed a resource cleanup fix and proved it with strict warning-as-error
tests, full non-slow tests, ruff, mypy, coverage, and the pre-push recall gate.

The remaining robustness risks are concentrated in operational boundaries:

- operations that can affect Markdown memories or sqlite indexes
- runtime install/update and daemon lifecycle paths
- HTTP API exposure and authentication defaults
- diagnostic quality in `memo doctor`
- recovery from interrupted or partially failed operations

memo also has prior durable decisions that matter here:

- The HTTP API has been identified as a security-sensitive surface when
  authentication is absent.
- Long-running LLM/entity work has needed explicit timeouts to avoid blocking a
  pipeline indefinitely.
- Restricted clients benefit from smaller MCP profiles, so operational clarity
  should preserve token economy and avoid broadening default tool surfaces.

## Goals

- Prevent silent or ambiguous failures in data-affecting operations.
- Make install, update, daemon, and runtime failures diagnosable and recoverable.
- Require authentication for the HTTP API by default.
- Make `memo doctor` a reliable contract for broken-state detection and next
  actions.
- Prove each robustness fix with focused empirical tests and full-suite gates.

## Non-Goals

- No retrieval/ranking redesign unless a robustness bug directly requires it.
- No large refactor of CLI or MCP surfaces.
- No general security audit beyond the minimum high-risk HTTP/sync/import
  surfaces listed here.
- No automatic destructive repair unless a safe pattern already exists and is
  covered by tests.

## Approach Options Considered

### Option A: Reliability Core

Focus on data safety, runtime/update/install, daemon lifecycle, HTTP API auth,
doctor, and recovery. This is the chosen approach because it targets the highest
impact operational failures without expanding product scope.

### Option B: Security And Privacy Hardening

Focus on secrets, redaction, transcript imports, retention, sync safety, and HTTP
API exposure. This is valuable, but too broad for the next single implementation
plan. The roadmap keeps HTTP API auth and basic secret-sensitive preflights as a
minimum security lane.

### Option C: Operational UX

Focus on clearer `doctor`, install, update, runtime, and MCP-profile messages.
This reduces support burden, but it would not fully cover data safety or recovery
from partial failures. The roadmap includes this as a later phase after P0/P1
risks are under test.

## Scope

### P0: Data Safety

Data-affecting operations include sync, reindex, migrate, import/export, delete,
rollback, and any path that can rewrite Markdown memory files or sqlite state.

Requirements:

- Run preflight checks before destructive or difficult-to-recover actions.
- Refuse unsafe paths outside the configured memory vault or state directory.
- Detect interrupted git/rebase/conflict states before sync mutates local files.
- Emit a receipt or structured result for completed, skipped, and failed actions.
- Ensure the next `memo doctor` can identify known broken states and explain a
  safe next step.

### P1: Runtime Reliability

Runtime reliability covers install, update, isolated runtime checks, daemon
lifecycle, subprocess cleanup, timeouts, lock contention, and warm/cold fallback
behavior.

Requirements:

- Use explicit timeouts for subprocess or service calls that can block a user
  command.
- Close owned resources deterministically.
- Keep MLX and other heavy imports deferred on CLI startup paths.
- Distinguish healthy, degraded, cold, and unreachable daemon states.
- Make partial update/install states visible through `doctor`.

### P2: Security Minimum

The HTTP API is useful for non-MCP clients, but it exposes memo operations over a
network interface. Authentication must be default-on.

Requirements:

- `memo http-api` must reject unauthenticated requests by default.
- Auth bypass, if supported, must be explicit and clearly labeled for local
  development only.
- Binding to non-loopback hosts must require an explicit acknowledgement or token
  configuration.
- Import/sync paths should avoid logging obvious secrets in receipts or error
  messages.

### P3: Doctor As Contract

`memo doctor` should be the user's first recovery tool.

Requirements:

- Report broken states with cause, impact, and a safe next action.
- Keep read-only modes read-only.
- Avoid false failures for supported dev-mode versus isolated-runtime
  differences.
- Cover runtime, MCP config, sqlite/FTS/vec indexes, daemon, sync, and HTTP API
  auth readiness.

## Architecture

The roadmap should add small contracts around existing modules rather than create
a new central robustness subsystem.

- `src/memo/runtime/`: owns install/update/runtime/daemon lifecycle checks,
  timeouts, cleanup, and partial-state diagnostics.
- Sync and storage modules: own vault containment, preflights, receipts, and
  recovery checks for memory and sqlite state.
- `server_http` / `cli_http`: own auth defaults, token validation, host binding
  guards, and HTTP-specific error responses.
- Doctor modules: own read-only diagnostics and actionable output. Doctor may
  call read-only helpers from runtime, sync, storage, and HTTP modules.
- Tests: own contract verification for broken states, timeouts, auth failures,
  warning-as-error cleanup, and recovery.

The implementation should prefer explicit result objects or `MemoError`
subclasses for expected failures. Broad best-effort swallowing is acceptable only
in hook/daemon paths where blocking user work would be worse, and those paths
must leave structured evidence.

## Operation Contract

Every robust operation should follow the same lifecycle:

1. **Preflight** validates the state required to proceed safely.
2. **Action** performs the operation with timeout, locking, and cleanup where
   needed.
3. **Receipt** records what happened, including skip and failure reasons.
4. **Recovery** lets `memo doctor` identify a known partial state and present a
   safe next action.

User-facing CLI routes should raise or render `MemoError` with actionable
messages. Daemon and hook routes should avoid blocking work but must log enough
evidence to debug the failure. Destructive write paths must not silently swallow
failures after partial mutation.

## Phases

### Phase 1: Executable Audit

Create an audit that turns major robustness risks into either a passing check or
a reproducible failure. Each P0/P1 finding must have a focused test, command, or
fixture that demonstrates the issue.

Deliverables:

- HTTP API auth/bind exposure audit.
- Sync, reindex, migrate, and import/export safety audit.
- Runtime/update/daemon timeout and partial-state audit.
- `doctor` broken-state coverage audit.
- Prioritized issue list with P0/P1/P2 labels.

Exit criteria:

- No P0/P1 item is only described in prose; each has a reproduction or test.
- The audit can be rerun locally without touching the real user vault.

### Phase 2: Close P0/P1 Risks

Fix data safety and runtime risks first.

Deliverables:

- HTTP API default authentication and binding guards.
- Destructive-operation preflights for unsafe paths and known broken git/sqlite
  states.
- Runtime/update/daemon timeout and cleanup fixes.
- Focused tests proving each fix.

Exit criteria:

- P0/P1 issues from Phase 1 are fixed or explicitly deferred with a documented
  reason and guardrail.
- Focused robustness tests pass with warnings treated as errors where relevant.

### Phase 3: Doctor And Messages

Convert known broken states into actionable diagnostics.

Deliverables:

- Doctor checks for HTTP auth readiness, sync conflict states, runtime mismatch,
  daemon unreachable/degraded states, sqlite/FTS/vec health, and MCP config.
- CLI messages that name the cause, impact, and safe next command.
- Tests for JSON and text output where stable output matters.

Exit criteria:

- A user can run `memo doctor` after the main induced failures and see the next
  safe action.
- `--json` remains machine-readable and `--check` remains read-only.

### Phase 4: Empirical Proof

Run and document the verification matrix.

Required checks:

- Focused tests for data safety, runtime reliability, HTTP auth, and doctor.
- Strict warning-as-error tests for changed resource paths.
- `uv run --no-sync ruff check src/ tests/`
- `uv run --no-sync mypy src/memo`
- `uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing`
- Isolated runtime doctor: `/Users/fer/.local/bin/memo doctor --strict-runtime`
- HTTP auth smoke with token missing, invalid, and valid.
- Sync/reindex smoke in a temporary repo/vault.
- Pre-push recall gate.

Exit criteria:

- All required checks pass or any skipped check has an explicit environment
  reason.
- A final report lists commands, results, and any residual risk.

## Testing Strategy

Use focused tests for each robustness contract and broaden only when the touched
surface requires it.

Test categories:

- Data safety: temp vaults, temp git repos, interrupted states, unsafe paths,
  corrupt sqlite fixtures, import/export path containment.
- Runtime: fake subprocess hangs, partial install/update markers, daemon socket
  stale/unreachable/degraded states, cleanup under exceptions.
- HTTP API: default auth required, invalid token rejected, valid token accepted,
  non-loopback binding guard.
- Doctor: text and JSON checks for each induced broken state.
- Resource hygiene: warning-as-error tests for sqlite connections, subprocess
  pipes, sockets, and temp files touched by the implementation.

Tests must not touch the real vault or default state directory.

## Error Handling Policy

- Expected domain failures use `MemoError` subclasses or structured result
  objects.
- User-visible commands should include a concise cause and an actionable next
  step.
- Hook and daemon paths can degrade, but they must log structured evidence.
- Destructive paths must either complete, roll back, or leave an explicit receipt
  for recovery.
- Security failures should be explicit denials, not permissive fallbacks.

## Documentation

Update documentation only where it helps users recover or configure safely:

- HTTP API auth and binding behavior.
- `doctor` recovery meanings and safe commands.
- Sync/reindex/migrate safety guarantees.
- Runtime install/update troubleshooting.

Avoid documenting implementation internals unless they are part of the public
operational contract.

## Design Decisions

The roadmap uses these defaults unless the executable audit proves a better
local pattern already exists:

- HTTP API auth uses a bearer token supplied by `MEMO_HTTP_API_TOKEN` or a local
  state-dir token file created by an explicit setup command. The token must not be
  printed in normal logs.
- The HTTP API has no implicit no-auth mode. A development bypass, if kept, must
  require an explicit CLI flag such as `--allow-no-auth` and must be rejected for
  non-loopback binds.
- Receipts should reuse existing history/log locations when they already model
  the operation. Add new structured receipt files only when no existing sink can
  represent the recovery data cleanly.
- `doctor --fix` should be conservative. The first implementation may diagnose
  more states than it repairs; automatic fixes are limited to idempotent config or
  stale-runtime cleanup operations with focused tests.

## Completion Criteria

The roadmap is complete when:

- No P0/P1 robustness finding remains without a fix, test, or explicit guarded
  deferral.
- HTTP API cannot be exposed without authentication by accident.
- Data-affecting operations have preflight, receipt, and recovery coverage.
- Runtime/update/daemon failures are bounded by timeouts and visible in doctor.
- `memo doctor` detects the main induced broken states and reports safe next
  actions.
- The final empirical report includes reproducible commands and passing results.
