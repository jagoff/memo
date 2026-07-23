# Engram Learnings — P2–P5 Completion Scorecard

**Date:** 2026-07-22

**Implementation commit:** `56ac6140`

**Scope:** canonical relations, explicit lifecycle, MCP write coordination,
first-class agent setup, graduation evidence, and legacy convergence.

## Outcome

The remaining Engram-inspired program is implemented end to end. The main
index now owns relation judgments and review evidence; truth validity, review
timing, and verification are distinct; mutating MCP calls can use bounded
process-local backpressure; Codex and Claude Code share one declarative setup
path; and legacy contradiction storage is import-only rather than a second
writable truth.

## Phase scorecard

| Phase | Delivered | Proof |
|---|---|---|
| P2 relations | deterministic pair identity; pending/judged/orphaned lifecycle; max-three namespace-safe post-save candidates; no service LLM; idempotent/conflict-safe judgment; supersede rollback; search/ask/briefing annotations; legacy import | relation convergence/workflow tests; 12-case fixed corpus |
| P3 lifecycle | `review_after`; 90/180/365-day policy; evidence ledger; due/mark CLI+MCP; explicit invalidate/supersede; due-only VERIFIED→STALE; no age-driven invalidation/unverification | lifecycle policy, idempotency, reindex persistence, as-of, and verification tests |
| MCP writes | single-worker bounded FIFO; retryable saturation; queued cancellation; started completion; read bypass; safe typed failures; metrics; clean lifespan shutdown | 32-job FIFO/load test plus cancellation/saturation/error tests |
| P4 adoption | declarative Codex/Claude Code adapters; detect/dry-run/apply receipts; atomic managed instructions; backup/idempotency; compensating removal; exact partial remediation; doctor verification and isolated save/search | setup, installer, mandate, runtime-isolation, and doctor tests |
| P5 convergence | canonical contradiction adapter; one-way deterministic migration; fixed relation eval; graduation registry; obsolete age flags removed; surface/docs/releases reconciled | migration parity, full suites, relation and recall gates |

## Measured gates

### Relation policy corpus

- Cases: 12
- Expected/predicted candidates: 5 / 5
- Recall: **1.0000**
- Precision: **1.0000**
- Noise: **0.0000**
- Fingerprint: `2da3bd3a9fcda701`

### Save-path latency characterization

Deterministic 32-dimension local embedder, 40-record seed corpus, 30 measured
decision saves per arm:

| Arm | p50 | p95 |
|---|---:|---:|
| relation candidates OFF | 5.319 ms | 5.724 ms |
| relation candidates ON | 13.702 ms | 14.520 ms |

The measured p50 increment was **8.384 ms**. At the initial 2026-07-22
completion point, candidate generation and annotations remained report-only
pending explicit product approval. On 2026-07-23 the user approved dogfooding
all three graduated capabilities. Both relation flags are now default-on, have
independent opt-outs, and no longer appear in the dark-flag or graduation
registries.

### MCP coordinator characterization

The 32-job FIFO run completed 32/32 with zero failures, rejections, or pending
jobs. Mean queue wait was 19.995 ms and maximum wait was 39.833 ms for synthetic
1 ms mutations. Following explicit approval on 2026-07-23, capacity 32 became
the default. Capacity `0` remains the immediate rollback control.

### Retrieval regression

On 41 labeled prompts at K=5, baseline config A retained precision **0.762**,
noise **0.000**, recall **0.667**, nDCG **0.499**, and MRR **0.444**. The eval
recommended no ranking change.

## Verification receipt

- `ruff check src/ tests/`: pass
- `mypy src/memo`: pass (431 source files)
- non-slow suite: 5,284 passed, 29 skipped, 4 warnings; 76.06% coverage
- slow/MLX suite: 7 passed, 4 skipped
- relation gate: pass
- recall regression eval: pass/no change recommended
- focused hooks, recall server, runtime isolation, setup/installer, migration,
  history/as-of, session patterns, redaction, relation, and lifecycle checks:
  pass

## Compatibility and rollback

- Markdown remains the content source of truth; schema v6/v7 additions are
  additive and `memo reindex --rebuild` preserves relation/review signals.
- Existing contradiction commands project canonical rows. A historical
  `contradictions.db` is read only and imported idempotently by migration key.
- Existing install commands remain compatibility surfaces. Registry ownership
  is used for first-class agent metadata; broader adapters stay in their old
  paths until their parity window is measured.
- Relation candidate/annotation and coordinator defaults were activated only
  after explicit product approval. Each retains a zero/false rollback control,
  while correctness invariants and canonical storage remain always on.
