# Engram trust invariants — P1 proof

Date: 2026-07-22

Implementation base: `a6bfb94c`; final implementation commit recorded below

Host: Apple arm64, Darwin 27.0.0, Python 3.13.9

## Implemented contract

- Mandatory known-pattern masking and `<private>` stripping before any normal
  memory persistence or derived indexing; entropy masking remains opt-in.
- Canonical topic identity `(namespace, topic_key)` and exact identity
  `(namespace, type, normalized_title, normalized_content_hash)`.
- `_global`, `_unscoped`, and `project:<slug>` coexist without overwrite;
  incompatible project tags fail before mutation.
- Save resolves to `created`, `corroborated`, or `revised`; responses include
  truthful `index_pending` state without changing list/get response shape.
- Schema v5 adds rebuildable identity fields, non-destructive migration,
  conflict-aware topic uniqueness, and atomic block/re-enable on rebuild.
- Update checks prospective identity before mutation. Reindex derives from
  sanitized Markdown views while preserving Markdown and user-signal tables.
- Update and topic-revision embeddings are prepared optimistically outside the
  authority lock, then accepted only when the exact embed input still matches
  under lock. Derived chunk emission also runs after the lock.
- `memo doctor --db` is read-only and reports counts without excerpts, matched
  values, or secret-derived hashes.

## Scorecard

| Gate | Result | Evidence |
|---|---|---|
| Storage-boundary privacy | PASS | Direct save/update, capture, ingest, server, reindex, history/receipt/log canary suites |
| Namespace matrix | PASS | Project A/B, global, and unscoped produce distinct canonical IDs |
| Exact corroboration | PASS | Thread and real-process tests converge to one file with support `N-1` |
| Topic revision | PASS | Same ID/path/created, prior version snapshot, current body/index coherent |
| Failure injection | PASS | Corroboration transaction rolls back; revision store failure restores Markdown/index; receipt failure cannot falsify a committed save |
| Historical corpus handling | PASS | v4 migration is non-destructive; ambiguous rows stay readable; constraint blocks/re-enables |
| Lock/model isolation | PASS | Direct update and topic revision probes observe lock depth `0` for every embed call |
| Intent journal | NOT NEEDED | Existing authority lock, atomic Markdown replacement, text reservation, transaction rollback, and pending marker cover observed failures |

## 100-save result

Same host and configuration as P0:

| Metric | P0 | P1 | Change |
|---|---:|---:|---:|
| p50 | 4.670 ms | 1.476 ms | -68.4% |
| p95 | 5.511 ms | 1.684 ms | -69.4% |
| distinct IDs | 100 | 1 | -99 |
| Markdown files | 100 | 1 | -99 |
| embedding calls | 100 | 1 | -99 |
| canonical support | 1 | 99 | correct `N-1` |

Exact identity resolution occurs before the optional semantic near-duplicate
probe, so corroboration avoids redundant embeddings.

## Verification log

Final implementation commit: `84bf9a41`.

Component gates:

- `pytest tests/test_engram_trust_invariants.py -q` — 18 passed, zero
  xfails/skips.
- Privacy/capture/ingest/server/doctor focused suite — 191 passed.
- `pytest tests/test_memory_reindex.py -q` — 42 passed.
- Store/reindex/support focused suite — 98 passed.
- Write/failure/support focused suite — 74 passed.
- Final write/failure/invariant regression after lock isolation — 61 passed.

Final CI-parity order:

- `env -u FORCE_COLOR uv run --no-sync ruff check src/ tests/` — PASS.
- `env -u FORCE_COLOR uv run --no-sync mypy src/memo` — PASS, 420 source
  files.
- `env -u FORCE_COLOR uv run --no-sync pytest -m "not slow" -n auto
  --timeout=120 --cov=memo --cov-report=term` — 5,239 passed, 29 skipped,
  4 warnings; coverage 75.97% (required 74%).
- `env -u FORCE_COLOR uv run --no-sync pytest -m "slow" --timeout=300 -v`
  — 7 passed, 4 skipped (MLX, real save/search, external-edit reindex, reranker,
  and trust-state fixtures passed).

The host exported both `FORCE_COLOR=1` and `NO_COLOR=1`; `FORCE_COLOR` was
removed for verification so Click/Rich test output remained deterministic.
No product setting was changed.

## Retrieval regression

Command:

```bash
env -u FORCE_COLOR uv run --no-sync memo eval recall \
  --labels eval/regression_labels.json --k 5 --force
```

41 prompts, four configurations, 164 searches:

| Config | precision@5 | noise@5 | R@5 | nDCG | MRR | p50 |
|---|---:|---:|---:|---:|---:|---:|
| A vec/0.60/keep | 0.762 | 0.0 | 0.667 | 0.499 | 0.444 | 203.2 ms |
| B vec/0.72/excl | 0.730 | 0.0 | 0.667 | 0.499 | 0.444 | 198.9 ms |
| C hyb/0.40/excl | 0.730 | 0.0 | 0.333 | 0.229 | 0.194 | 241.0 ms |
| D hyb/0.40/ctx | 0.757 | 0.0 | 0.556 | 0.474 | 0.444 | 244.7 ms |

Baseline A remains the winner; the evaluator recommended no knob change. Graph
noise and hub noise were `0.0` for every configuration.

## Compatibility and remaining diagnostics

- Fresh/clean v5 fixtures report `identity_constraint=enabled`.
- Historical collision fixtures remain readable and report `blocked`; removing
  the conflicting derived identities and rebuilding re-enables the constraint.
- Legacy v4 fixtures report `unavailable` with upgrade/rebuild guidance.
- No production vault was mutated or auto-repaired during verification.

No relation judgment, lifecycle convergence, MCP write queue, installer setup,
or review scheduling was added in this phase.
