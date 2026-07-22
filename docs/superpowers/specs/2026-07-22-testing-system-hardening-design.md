# Testing System Hardening Design

**Date:** 2026-07-22
**Status:** Approved
**Scope:** Resource hygiene, deterministic CI gates, branch and diff coverage,
stateful vector-store testing, stability testing, and scoped mutation testing.

## Objective

Turn memo's broad test suite into a stricter testing system that detects
resource leaks, order dependencies, untested branches, weak assertions, and
state-machine failures without making every pull request depend on expensive or
probabilistic jobs.

The implementation builds on the scoring, replay, vector-database, and
housekeeping contracts added on 2026-07-22. It preserves the existing Linux
Python 3.13/3.14, int8, contract, macOS/MLX, and nightly slow-test lanes.

## Chosen Operating Model

The selected approach is balanced enforcement:

- fast, deterministic checks block pull requests;
- randomized repetition and mutation testing run on schedules and on manual
  dispatch;
- no automatic retry plugin may turn a red test into a green check;
- a failing scheduled job remains visible and reproducible through its seed or
  mutation identifier.

Two alternatives were rejected. Running every randomized and mutation check on
every pull request would add avoidable latency and probabilistic failures.
Making the new deterministic checks informational would not prevent regressions
and would fail the hardening objective.

## Constraints

- Tests must never access the real vault, state directory, daemon socket, or
  model configuration.
- Markdown remains source of truth and SQLite remains rebuildable derived state.
- Real MLX tests stay behind `requires_mlx`; new database tests use small,
  deterministic vectors and no model downloads.
- Existing user changes in the dirty worktree must be preserved. Commits created
  for this work stage only files owned by this design.
- Production behavior changes follow RED/GREEN/REFACTOR. CI-only configuration
  changes are validated with contract tests that inspect the workflows.
- Standard PR CI keeps the order `ruff -> mypy -> quality gate -> pytest`.

## Tooling and Dependency Boundaries

Add compatible bounded dependencies and regenerate `uv.lock`:

- `hypothesis>=6.158,<7` and `diff-cover>=9,<10` join the `dev` extra because
  they participate in blocking PR validation;
- `pytest-randomly>=4.1,<5` and `pytest-repeat>=0.9.4,<1` live in a new
  `test-stability` extra, so installing ordinary development dependencies does
  not silently randomize every local pytest invocation;
- `mutmut>=3,<4` lives in a new `test-mutation` extra.

Hypothesis supports Python 3.10–3.14 and shrinks failures to minimal examples.
pytest-randomly supports Python 3.10–3.14, works with xdist, and prints a
replayable seed. pytest-repeat provides explicit `--count` and
`--repeat-scope` controls. diff-cover consumes coverage.py XML and compares it
to the PR base. mutmut 3 supports Python 3.14, scoped source paths, pytest test
selection, and covered-line filtering.

Generated `.hypothesis/`, coverage XML, mutation work directories, and local
reports are ignored by git. CI uploads reports explicitly.

Authoritative tool references:

- [Hypothesis package and supported Python versions](https://pypi.org/project/hypothesis/)
- [pytest-randomly behavior and replayable seeds](https://pypi.org/project/pytest-randomly/)
- [pytest-repeat counts and repeat scopes](https://pypi.org/project/pytest-repeat/)
- [diff-cover XML, comparison, and fail-under contract](https://pypi.org/project/diff-cover/)
- [coverage.py branch measurement](https://coverage.readthedocs.io/en/latest/branch.html)
- [mutmut configuration and covered-line filtering](https://pypi.org/project/mutmut/)

## Workstream 1: Restore a Clean Baseline

### Broad-exception classification

The proactive urgent-message block in `cli_recall_hook.recall_hook` is a
deliberate hook-hot-path fail-open boundary. Add its stable lexical identifier
to `BROAD_EXCEPTION_ALLOWED` with a local reason comment. Do not narrow it to a
partial exception set: proactive storage, rendering, and time parsing are
optional work and must never block recall.

The focused `test_broad_exception_policy_targets_are_classified` test must fail
before the classification and pass afterward.

### Resource ownership

Replace direct, unclosed `Memory`, `VecStore`, and `ProactiveStore` ownership in
the failing resource-sensitive tests with context managers or yield fixtures.
Introduce shared factory fixtures only where they reduce repeated cleanup code;
every factory registers a finalizer immediately after constructing the resource.

Do not weaken `test_http_auth` by ignoring all `ResourceWarning` instances. It
continues to assert that FastMCP leaves no stream resources open, while unrelated
memo-owned SQLite objects are closed by their owning tests.

### Opt-in leak detector

Add a pytest `--resource-hygiene` option. When enabled, an autouse fixture:

1. runs `gc.collect()` before the test to drain older objects;
2. captures `ResourceWarning` during the test and teardown;
3. runs `gc.collect()` after teardown;
4. fails with the warning messages for unclosed SQLite connections, memory
   streams, sockets, or files.

The fixture is inert during normal pytest runs. Tests that own native resources
receive the `resource_hygiene` marker. A blocking serial CI step runs that marker
with `-n 0 --resource-hygiene`, preventing xdist garbage collection from
misattributing another test's resource warning.

## Workstream 2: Coverage Gates

### Branch-aware aggregate gate

Enable `branch = true` under `[tool.coverage.run]`. Run the complete non-slow
suite to establish the real branch-aware baseline. The implementation must add
focused tests until the combined branch-aware percentage is at least 74%, then
set `fail_under` to the lower whole-number percentage of that measured baseline,
with an allowed one-point buffer and an absolute minimum of 74%. For example, a
75.8% result produces a 74% gate; a 77.1% result produces a 76% gate.

This replaces the stale 72% floor. Existing tests that assert the configured
quality floor are updated to verify branch measurement and the selected ratchet.
Coverage effort prioritizes stable critical paths such as runtime startup,
recall hooks, backup/snapshot, storage, and signal queries. Experimental CLI and
TUI modules do not receive tests solely to inflate the aggregate number.

### Changed-lines gate

The Linux coverage job also writes `coverage.xml`. On pull requests, CI fetches
the base branch and runs:

```text
diff-cover coverage.xml --compare-branch=origin/master --fail-under=90
```

The command shows uncovered changed lines and fails below 90%. Pushes to master
skip the diff-only step because there is no open PR diff, while the aggregate
coverage gate still applies. A workflow contract test verifies checkout depth,
XML generation, base comparison, and the 90% threshold.

## Workstream 3: Stateful VecStore Contracts

Add `tests/test_vector_store_state_machine.py` using Hypothesis'
`RuleBasedStateMachine` against the real SQLite + sqlite-vec implementation.
Each machine owns a temporary database and an in-memory reference model.

The generated rules cover:

- inserting new records with deterministic four-dimensional vectors;
- replacing metadata, body text, type, tags, and vector for an existing ID;
- soft deletion and restoration;
- permanent deletion of active or tombstoned records;
- closing and reopening the database;
- clearing the derived memory index;
- vector, BM25, type, exact-tag, and deleted-record observations.

After every rule, invariants compare the database with the model:

- active count and `get()` visibility match;
- every active ID has exactly one metadata row and one vector row;
- tombstones appear only in the deleted listing;
- hard-deleted IDs have no metadata, vector, FTS, or attached signal rows;
- replacement never leaves old searchable text;
- reopen preserves every committed model state;
- query result IDs are a subset of eligible active IDs and deterministic
  nearest-neighbour fixtures retain the expected order.

The PR profile uses 25 examples with 30 state-machine steps and no wall-clock
deadline. A scheduled profile may raise those limits through
`HYPOTHESIS_PROFILE=ci_extended`. Hypothesis' reported seed/example is retained
in CI output for reproduction.

## Workstream 4: Test Taxonomy

Register three strict markers:

- `db_contract`: real SQLite/sqlite-vec behavioral contracts;
- `resource_hygiene`: tests that own native resources and run in the leak lane;
- `concurrency`: thread, process, socket, locking, or WAL interleaving tests.

Apply them to the new suites and the existing focused concurrency/resource
files. Do not move the repository's roughly 470 flat test modules; markers give
CI selection and ownership without a high-churn import/path migration.

## Workstream 5: Scheduled Stability Workflow

Add `.github/workflows/test-stability.yml` with nightly and manual triggers,
Python 3.13, frozen uv installation, and a hard job timeout.

It has two deterministic-reproduction phases:

1. run the complete non-slow suite serially with pytest-randomly and a seed
   derived from `github.run_id`;
2. run `concurrency or resource_hygiene` tests ten times with
   `--repeat-scope=session`, serial execution, and stop-on-first-failure.

The job prints the replay command and uploads JUnit XML. It never uses
`pytest-rerunfailures` or equivalent masking behavior.

## Workstream 6: Scheduled Mutation Workflow

Add `.github/workflows/mutation-tests.yml` with weekly and manual triggers,
Python 3.13, frozen dependencies, and a 30-minute outer timeout.

Configure mutmut in `pyproject.toml` to mutate only covered lines in a bounded
initial surface representing all three requested domains:

- storage: `src/memo/store/vec_base.py` and SQLite snapshot behavior;
- retrieval: `src/memo/memory/search_scoring_ops.py`;
- housekeeping: `src/memo/session.py` and `src/memo/sqlite_snapshot.py`.

Test selection is restricted to the focused scoring, vector/database,
housekeeping, session, snapshot, and store contract files. The workflow runs the
baseline tests before mutation, runs `mutmut`, writes a textual result summary,
and uploads the result plus the mutation work directory even on failure.
Surviving mutants fail the scheduled job; they are never silently accepted into
a baseline.

## CI Integration

Pull-request CI gains only deterministic work:

1. existing format, Ruff, mypy, and progressive quality gates;
2. serial resource-hygiene marker lane;
3. normal non-slow suite including the bounded Hypothesis state machine;
4. branch-aware aggregate coverage;
5. 90% changed-lines coverage.

The Python 3.13/3.14 matrix remains. The int8 lane continues to exclude only
`float32_precision`. macOS continues to exercise real MLX and installer paths.
The nightly slow suite is unchanged.

Workflow contract tests assert dependency extras, markers, schedules, hard
timeouts, no rerun plugin, random seed visibility, repeat count, mutation scope,
artifact upload, branch coverage, and diff coverage.

## Error Handling and Reproducibility

- A leak failure reports the owning test and captured resource warnings.
- A randomized failure reports an exact `--randomly-seed` replay command.
- A Hypothesis failure reports the minimized operation sequence and reproduction
  blob/example.
- A diff-coverage failure reports exact uncovered changed lines.
- A surviving mutant reports its module/function identifier and is retained in
  the uploaded results.
- Scheduled jobs use hard outer timeouts; pytest tests retain the existing
  per-test timeout.
- Tool caches and reports never enter release artifacts or the source tree.

## Acceptance Criteria

- The current broad-exception audit passes with a documented fail-open reason.
- The resource-hygiene lane passes serially without SQLite, stream, socket, or
  file `ResourceWarning` instances.
- `test_http_auth` passes both alone and adjacent to database/resource tests.
- Branch coverage is enabled and the aggregate gate is at least 74%, strictly
  above the previous 72% floor.
- Pull requests fail below 90% changed-lines coverage.
- The VecStore state machine completes its PR profile and detects a deliberately
  injected model/database divergence during RED verification.
- Registered markers pass `--strict-markers` and select non-empty suites.
- The stability workflow exposes a replayable seed and repeats the focused
  concurrency/resource suite ten times without automatic reruns.
- The mutation workflow covers storage, retrieval, and housekeeping, retains
  results, and fails when mutants survive.
- `uv.lock`, Ruff formatting/lint, mypy, progressive quality gate, focused tests,
  workflow contracts, the complete non-slow suite, and coverage gates all pass.

## Out of Scope

- Moving all test files into a new unit/integration/e2e directory hierarchy.
- Making probabilistic or mutation jobs required pull-request checks.
- Raising every experimental module to an arbitrary per-file coverage target.
- Running real MLX models in Linux database/property tests.
- Masking flaky tests with retries.
- Publishing artifacts, pushing to a remote, or modifying unrelated user work.
