# memo Deep Improvement Roadmap

Date: 2026-07-09
Status: design approved for implementation planning

## Summary

memo is operationally healthy, but its internal surface has grown large enough that
future improvements need a stronger reliability and evaluation base before changing
retrieval, capture, or ranking behavior.

This design defines a phased improvement roadmap. It deliberately starts with
verification hygiene, error policy, and eval profile clarity, then moves into memory
quality and product surface work. The goal is not to rewrite memo; it is to make
each future improvement easier to trust, measure, and ship.

Prior durable work already identified several of the same themes: CI/CD, stable
core boundaries, test coverage, import laziness, error handling, and release
packaging were captured in the earlier memo improvement evaluation `0b488a60`.
Some parts have since improved, but the underlying direction remains valid.

## Baseline Evidence

The review used local repo state, recent commits, memo's own diagnostics, and the
full test suite.

Current repository state:

- `/Users/fer/repos/memo` was clean and aligned with `origin/master`.
- Recent work heavily touched memory quality, context packs, eval, and empirical
  verification.
- `memo-sync` had one new memory file during the session; that is normal
  background capture state, not a code repo issue.

Operational health:

- `memo health` reported 5,982 memories, 949 archived, 693.5MB corpus, vec dims
  2560/2560, FTS ready, 10,565 health rows, 812 low-confidence rows, 466 high-ROI
  rows, 24 feedback signals, and no warnings.
- `/Users/fer/.local/bin/memo doctor --strict-runtime` passed with the isolated
  uv-tool runtime.
- `uv run --no-sync memo doctor --strict-runtime` correctly reported project
  `.venv` mode. That is a development invocation mismatch, not a runtime install
  failure, but it is confusing enough to document in the roadmap.

Memory utility signals:

- `memo usefulness` sampled 437 consults across `synapse`, `claude-code`, and
  `codex`.
- Recall-hook hit rate was 98.7%, strong-hit rate 97.4%.
- `grounded_rate` was 0.371, but `referenced_rate` was 0.009. This means surfaced
  memories often help answers, but very few are later fetched directly. The metric
  is a lower bound, but it still suggests a product/eval gap worth studying.
- `memo gaps` had one unresolved topic.

Corpus quality:

- `memo lint --limit 10` reported:
  - `legacy_extra`: 0
  - `few_tags`: 2,635
  - `body_skinny`: 242
  - `untitled`: 0

Latency:

- `memo stats` showed recall daemon p50 655ms, p95 6,803ms, p99 8,152ms.
- Subprocess recall p50 was 8,896ms, p95 10,668ms, p99 11,572ms.
- The daemon path is the right production path, but fallback/subprocess latency is
  high enough to keep visible in health and docs.

Codebase scale:

- `src/memo` is about 86k Python lines.
- There are 113 top-level `cli*.py` and `server*.py` modules: 74 CLI modules and
  39 server modules.
- Large modules include:
  - `src/memo/cli_dream.py`: 1,373 lines
  - `src/memo/flags_misc.py`: 1,264 lines
  - `src/memo/dream_tune.py`: 1,240 lines
  - `src/memo/recall_logic.py`: 1,208 lines
  - `src/memo/capture_core.py`: 1,180 lines
  - `src/memo/eval_recall.py`: 1,033 lines
  - `src/memo/store/queries.py`: 1,023 lines
  - `src/memo/memory/write_ops.py`: 1,007 lines
  - `src/memo/memory/ask_ops.py`: 951 lines

Debt counts:

- 516 `except Exception` sites across 142 source files.
- 79 silent `pass` sites across 55 source files.
- 17 raw `os.environ.get("MEMO_*")` sites across 9 source files.
- 7 TODO/FIXME/HACK sites across 4 source files.

Some raw `MEMO_*` reads are intentional: store/schema and memory/facade include
comments explaining bootstrap, circular import, or tri-state behavior. The issue
is not that every instance is wrong; the issue is that the repo lacks a small
policy and audit test that distinguishes intentional bootstrap reads from
accidental flag drift.

Testing:

- Full non-slow suite passed: 3,534 passed, 29 skipped, 6 warnings.
- Coverage was 73.35%, above the current 68% floor.
- The suite emitted `ResourceWarning: unclosed database` warnings from sqlite
  connections. These are not failing today, but they add noise to the verification
  signal and should be fixed or explicitly allowlisted.

Low-coverage risk areas include:

- `src/memo/semantic_relations.py`: 0%
- `src/memo/runtime/daemon.py`: 20%
- `src/memo/memory/secret_ops.py`: 23%
- `src/memo/cli_contradict.py`: 24%
- `src/memo/cli_transcripts.py`: 24%
- `src/memo/synapse_backend.py`: 24%
- `src/memo/server_session_patterns.py`: 33%
- `src/memo/server_idle_capture.py`: 36%
- `src/memo/llm.py`: 37%
- `src/memo/runtime/update.py`: 44%

Eval and gates:

- `memo eval recall` default grid is now short enough for practical runs.
- The local pre-push hook runs A/B plus E-I as a vec-only subset:
  `--config A --config B --config E --config F --config G --config H --config I`.
- That pre-push gate passed in the latest push, running 238 searches with visible
  progress.
- The pre-push subset is useful, but it is currently encoded as a local hook list
  rather than a named eval profile. That makes future changes easy to
  misunderstand.

## Problem Priorities

### P0: Verification Signal Noise

Warnings in a green full suite are easy to ignore until they hide real resource
leaks. The sqlite `ResourceWarning` class should be addressed first because it
directly affects confidence in tests and cleanup behavior.

### P1: Error Handling Policy Drift

The count of broad exception handlers is high. Many are probably correct in hook,
daemon, dashboard, and best-effort diagnostic paths. The gap is that the intent is
not consistently machine-checkable or documented.

The target is not zero broad exceptions. The target is an explicit policy:

- hook hot path: never block user work, but log structured debug evidence
- user-visible CLI: convert expected domain failures to `MemoError`
- maintenance/daemon best effort: capture receipt/log context
- destructive write paths: no silent swallow without rollback or explicit receipt

### P2: Flag Access Discipline

AGENTS.md says behavioral `MEMO_*` flags should be registered/accessed through
the flags system, while storage/model config belongs in config. The repo has
intentional exceptions, but the exception pattern itself needs a documented helper
or audit rule.

### P3: Eval Profile Ambiguity

There are now several distinct eval roles:

- quick smoke
- default practical regression run
- pre-push vec subset
- full/tuning matrix
- named expensive probes such as HyDE

These need names, docs, and tests so changing one does not silently change the
others.

### P4: Memory Quality And Corpus Utility

The corpus is large and healthy, but the lint and usefulness metrics point to
quality work:

- thousands of low-tag memories
- skinny bodies
- low direct referenced rate
- one known gap
- strong hook hit-rate that may hide over-injection or weak post-use tracking

This should be improved only after P0-P3 make measurement cleaner.

### P5: Surface Area And Operability

memo exposes a large CLI/MCP surface. The stable core is documented, but users and
maintainers still pay cognitive cost when commands are numerous and advanced
surfaces have uneven coverage.

This track should improve:

- stable/advanced/experimental labeling
- doctor/sync/install messaging in dev vs isolated runtime modes
- release and push workflow checks
- MCP profile/tool-surface clarity

### P6: Security, PII, And Secret Handling

Security is not the first sprint because the current review did not find a live
secret leak. It remains a roadmap track because memo ingests personal memory,
WhatsApp, transcripts, and agent context. The repo already contains privacy and
secret-scan plans, and memory `7fa5c724` notes that `memo invalidate` can weaken
groups of memories after major changes.

The security track should focus on:

- secret scanning before sync/push
- `<private>` and redaction behavior
- safer defaults for imported chat/transcript corpora
- clear invalidation/retention workflows after sensitive changes

## Roadmap

### Phase 1: Clean Verification Signals

Goal: a green verification run should be easy to trust.

Deliverables:

- Reproduce sqlite `ResourceWarning` under targeted warning-as-error tests.
- Fix connection lifecycle or fixtures responsible for unclosed sqlite handles.
- Introduce an exception-handling policy document or module-level annotation
  pattern.
- Classify top broad-exception sites by layer and intent.
- Audit raw `MEMO_*` reads and mark each as registered, bootstrap-intentional, or
  migratable.
- Define warning policy for tests: unexpected warnings fail; explicit known
  warnings are allowlisted with a reason.

Exit criteria:

- Full non-slow test suite passes.
- Targeted warning-as-error tests pass for the sqlite lifecycle paths touched.
- `ruff` and `mypy src/memo` pass.
- `memo doctor --strict-runtime` passes with the isolated binary.

### Phase 2: Make Eval Profiles Explicit

Goal: every eval command and gate should have an explicit cost/coverage contract.

Deliverables:

- Define named eval profiles:
  - `quick`: fastest smoke
  - `default`: practical local regression
  - `pre-push`: vec-only gate used by local hook
  - `matrix`: expensive tuning comparison
  - `expensive`: opt-in LLM or hybrid probes
- Move the pre-push config list into a tested helper or documented hook template.
- Ensure baseline files include enough metadata to know which profile seeded them.
- Update CLI help/docs to explain which profile to run for which kind of change.

Exit criteria:

- Focused tests cover profile selection and pre-push profile membership.
- Existing pre-push gate still passes.
- `memo eval recall --quick` and the default human run remain visible and bounded.

### Phase 3: Improve Memory Quality And Usefulness

Goal: fewer useless surfaced memories and stronger durable corpus quality.

Deliverables:

- Build a corpus cleanup plan from `memo lint`: tags, skinny bodies, gaps, and
  duplicate/low-quality clusters.
- Add before/after tracking for `few_tags`, `body_skinny`, `grounded_rate`,
  `referenced_rate`, `tokens saved`, and recall eval metrics.
- Investigate why direct `referenced_rate` is low despite strong grounded rate.
- Improve capture/tagging or recall presentation only through measured,
  default-safe changes.

Exit criteria:

- Documented baseline and post-change metrics.
- No drop in recall gate precision/noise.
- Corpus lint categories trend down without deleting or weakening useful memory
  blindly.

### Phase 4: Rationalize Product And Ops Surface

Goal: make memo easier to operate and maintain.

Deliverables:

- Audit `memo --help` groups and mark core/advanced/experimental surfaces more
  consistently.
- Prioritize low-coverage but user-facing CLI/MCP modules.
- Improve dev-vs-install doctor messaging so project `.venv` warnings are less
  confusing during development.
- Review sync/release docs and automate checks where the user already expects
  commit + push to master after validated work.

Exit criteria:

- CLI help and docs explain stable vs experimental surfaces.
- Doctor output gives actionable next steps for dev mode and installed mode.
- Release/push checklist is shorter and less dependent on tribal knowledge.

## First Sprint

The first sprint should cover Phase 1 and the smallest useful slice of Phase 2.

### 1. SQLite Resource Hygiene

Reproduce the sqlite `ResourceWarning` with warning-as-error in the tests that
currently emit it. Identify whether the leak comes from fixtures, `Memory`,
`VecStore`, MCP server setup, or runtime installer tests.

Preferred fixes:

- explicit `close()` in test fixtures
- context-manager cleanup for store/memory objects where feasible
- deterministic teardown around MCP/server helpers

Avoid suppressing the warning globally unless the source is a third-party object
outside memo's lifecycle control.

### 2. Exception Policy Audit

Create a narrow policy and apply it to the highest-risk modules first. Do not try
to fix all 516 broad exception handlers in one pass.

Initial target modules:

- `src/memo/recall_logic.py`
- `src/memo/memory/write_ops.py`
- `src/memo/cli_recall_hook.py`
- `src/memo/store/queries.py` if sqlite lifecycle work touches it

For each edited site, choose one:

- keep broad catch but add structured debug/log/receipt context
- narrow to expected exception classes
- convert to a `MemoError` subclass for normal user-visible failures
- preserve best-effort swallow with a short reason

### 3. Flag Access Audit

Classify all raw `MEMO_*` reads:

- `accepted_bootstrap`: config, store, or tri-state cases where flags cannot
  express the needed behavior
- `migrate_to_flags`: normal behavioral flags
- `migrate_to_config`: storage/model config
- `needs_helper`: cases requiring a tri-state or raw-value helper

Add a small test or lint-like check that fails on new unclassified raw `MEMO_*`
reads in app code.

### 4. Eval Profile Naming

Add a named source of truth for the pre-push vec subset, or document it beside
`eval_recall` so it does not drift from default/tuning configs.

The first sprint does not need a full new command if a helper plus tests and docs
solve the ambiguity.

### 5. Audit Report Baseline

Record the baseline metrics in the implementation plan/report so future work can
prove movement:

- full suite result and warning count
- coverage total and key low-coverage modules
- broad exception counts
- raw `MEMO_*` count
- `memo lint` categories
- `memo usefulness` grounded/referenced rates
- recall daemon/subprocess latency

## Non-Goals For The First Sprint

The first sprint must not:

- rewrite retrieval ranking
- flip HyDE, MMR, graph, or capture defaults
- bulk-edit thousands of memories
- delete or weaken memory records without an explicit cleanup plan
- restructure the whole CLI/MCP surface
- chase coverage percentage with low-value tests
- eliminate every broad exception handler

## Risks

Overcorrecting best-effort behavior:

Hook and daemon paths intentionally avoid blocking the user. The policy work must
not turn non-critical telemetry failures into user-visible breakages.

Coverage theater:

Raising coverage without targeting high-risk paths would improve the number while
leaving real risk untouched. Prioritize lifecycle, storage, runtime, and user
visible commands.

Eval gate cost:

The pre-push gate is valuable but already expensive enough to be noticed. Profile
naming should reduce confusion without making every push slower.

Corpus cleanup damage:

Tags and skinny bodies are quality signals, not proof that a memory is useless.
Cleanup must be reversible or conservative.

Dev/install confusion:

`uv run memo doctor --strict-runtime` reports `.venv` mode by design. The docs and
doctor output should explain this without weakening the actual isolated-runtime
contract.

## Validation Contract

Every implementation plan derived from this spec should include:

- `uv run --no-sync ruff check src/ tests/`
- `uv run --no-sync mypy src/memo`
- `uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing`
- targeted warning-as-error tests for resource lifecycle changes
- `/Users/fer/.local/bin/memo doctor --strict-runtime`
- `memo eval recall --quick` for non-ranking work
- the pre-push gate for any pushed code

Ranking, recall-hook, search, capture, or corpus mutation work must additionally
run the appropriate recall/token eval gate and document before/after metrics.

## User Review Gate

After this design is committed, the next step is not implementation. The user
should review this spec and approve it or request edits. Once approved, the next
skill is `superpowers:writing-plans`, producing a task-by-task implementation
plan for the first sprint only.
