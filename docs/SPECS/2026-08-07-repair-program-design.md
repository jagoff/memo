# memo repair program — six phases

Date: 2026-08-07
Status: design approved, not yet planned
Scope: whole-product re-evaluation of memo three days after
[2026-08-04-memo-audit-design.md](2026-08-04-memo-audit-design.md), plus a
sequenced repair program covering retrieval quality, latency, instrumentation,
and surface reduction. This document does not authorize implementation; each
phase needs its own plan.

---

## Why a second audit three days later

The 2026-08-04 audit produced seven findings. P1 shipped (#192). P0 was
re-measured and re-diagnosed (#197) but not fixed. P2, P3, P4, P5 and P6 were
not started. Meanwhile eight commits landed, including a search-pipeline
refactor (#210) and a graph-compaction feature (#211, open).

Re-measuring on 2026-08-07 against the live install answers a narrower
question than the first audit did: **did anything the audit found get better,
and did the work that shipped in between make anything worse?**

The answer to the second half is yes, in two places, which is why this is a
program and not a punch list.

Every claim below is a measurement taken on 2026-08-07 against the production
install (`uv tool` runtime, corpus at `/Users/fer/repos/memo-sync/memorias`),
or a test/CI run on commit `a2706251`.

---

## What is healthy

Do not destabilize these.

| Check | Result |
|---|---|
| `ruff check src/ tests/` | clean |
| `ruff format --check` | 1124 files already formatted |
| `mypy src/memo` | no issues, 513 source files |
| Test suite | 2280 passed, 1 failed (see Phase 1), 181s |
| Daemons | all six `com.memo.*` up (`chat`, `vault-ingest`, `nightly`, `recall-daemon`, `dream`, `watch`) |
| `memo doctor` | every probe green except a known 2-unpushed-commit sync notice |
| codegraph | index ok, nodes=23950 edges=64249, fresh <24h |
| Recall hook | hit_rate 0.956, top_composite_score_rate 0.944 |
| Flag wiring | 515 declared, 512 referenced outside `flags*.py` |

The flag-wiring number matters: the P5 finding in the previous audit was
sometimes read as "498 flags are dead code". They are not. Three are dead
(listed in Phase 5); the rest are consumed. The problem is that they are
unexercised, not unreferenced.

---

## Measured regressions since 2026-08-04

### The control query got worse, not better

Query `por que se deprecó synapse`, the same control used in the previous
audit:

| | 2026-08-04 | 2026-08-07 |
|---|---|---|
| Rank of the answer-bearing record | 5 | **8** |
| Its score | 0.348 | 0.494 |
| Top-1 score | 1.411 | **3.456** |
| Composition of top 7 | mixed tiers | **7/7 reference chunks** |

The gap between the top hit and the answer widened from 4.1× to 7.0×. The
mechanism was diagnosed on 2026-08-05 (`retrieval_boost`, a multiplicative
curatorial term capped at 12× applied downstream of a fused score bounded at
1.0). Nothing has been changed in that path since, so this is drift in the
candidate pool amplifying an unfixed defect, not a new one.

### Retrieval quality per consumer degraded

`memo usefulness`, 227 consults sampled:

| Consumer | hit% (08-04 → 08-07) | composite>0.85% (08-04 → 08-07) |
|---|---|---|
| claude-code | 83 → **81** | 82 → **75** |
| codex | 45 → 48 | 28 → **19** |
| mcp-prompt | — | **40 / 0** (new, worst on record) |

`grnd%` is populated for `claude-code` only. Every other consumer shows `—`,
meaning outcome recording is wired on one path.

### Latency: one surface improved, one got worse

| Surface | 08-04 | 08-07 | Budget |
|---|--:|--:|--:|
| `memo search` (warm) | 26s | 11.5s | 2s |
| `memo ask` | 42s | **76s** | undefined |

`memo ask` nearly doubled. Neither surface routes through `recall.sock`; the
recall daemon holds a hot 4B MLX embedder and serves only the hook.

---

## New defects found

### A single document can occupy the entire result window

Query `Comandos disponibles CLI por función` returns ten results, all ten of
which are chunks of the same file:

```
6.895  …/Comandos Disponibles.md#chunk-4
6.775  …/Comandos Disponibles.md#chunk-10
6.712  …/Comandos Disponibles.md#chunk-5
6.601  …/Comandos Disponibles.md#chunk-1
6.591  …/Comandos Disponibles.md#chunk-2
6.508  …/Comandos Disponibles.md#chunk-3
6.484  …/Comandos Disponibles.md#chunk-17
6.406  …/Comandos Disponibles.md#chunk-15
6.317  …/Comandos Disponibles.md#chunk-7
6.253  …/Comandos Disponibles.md#chunk-11
```

The fix exists, has tests, and is off: `MEMO_SEARCH_CHUNK_PARENT`, whose flag
description reads *"Off by default — eval-gated before any flip."* The gate has
never been run for it. This is the single clearest instance of the pattern this
audit is really about: **memo builds faster than it enables.**

### `HEAD` of the open PR is red

`tests/test_dev_audit.py::test_broad_exception_policy_targets_are_classified`
fails on `a2706251`, verified in a clean worktree so it is not working-tree
contamination:

```
recall_logic.py:1223:_graph_compact_clusters:1   unclassified
recall_logic:8                                    unclassified
```

The graph-compaction commit added two broad `except Exception` sites to
`recall_logic.py` — one of the four files the repository's own broad-exception
policy test guards. CI caught it correctly: PR #211 shows `test (3.13)`,
`test (3.14)` and `test-int8` failing. The gate is wired; the fix was never
finished.

---

## Unwired surface

| Surface | Built | Reachable by default | Note |
|---|--:|--:|---|
| Bool flags | 212 | 82 on | 130 default-off |
| …of those, `search` + `recall` | 62 | 31 on | 31 features shipped and never enabled |
| MCP tools | 164 | 41 (`agent` profile) | 123 need `MEMO_MCP_PROFILE=full`, which `doctor` itself advises against at ~18.1k tokens/connection |
| CLI commands | 144 | 144 | no reconciliation against the 164 tools |

The 31 default-off search/recall features include HyDE, `quality_rerank`,
`rrf_adaptive`, `rerank_adaptive_pool`, `ask_multi_round`, `context_pack`,
`contradict_penalty`, negative recall (four flags), `precision_gate`,
`confidence_gate`, `graph_compact` and `chunk_parent`.

CLI/MCP asymmetry is real and unmeasured: `memo_review_due` exists as an MCP
tool; `memo review-due` does not exist as a CLI command. Nobody knows how many
more gaps there are because the matrix has never been generated.

## Housekeeping debt

| Item | Measurement |
|---|--:|
| Source files over the repo's own 800-line limit | 39 |
| …over 1200 lines | 13 |
| Largest | `recall_logic.py` at 2177 |
| `except Exception` sites in `src/` | 718 |
| `contextlib.suppress(Exception)` sites | 126 |
| Files guarded by the `dev_audit` broad-exception ratchet | 4 |
| `server_*.py` modules | 50 |
| Tools migrated to the declarative `mcp_tools.py` registry | 10 of 164 (6%) |

`mcp_tools.py` documents itself as "the home for *new* tools going forward and
a migration target for the old ones". The migration is at 6% and has no
recorded finish line.

---

## The program

Six phases. Order is by dependency, not by severity: Phase 0 exists because
Phase 2 is unverifiable without it, and Phase 5's central verdict
("is there evidence for this flag?") is unanswerable without Phase 0 either.

### Phase 0 — Make the eval gate measure code

Every change in Phase 2 is a ranking change, and the only arbiter is
`memo eval recall --labels eval/regression_labels.json`. Today that gate
measures corpus drift as well as code: a push can be blocked by variation in
the store with no diff to blame. A gate that cannot separate the two cannot
approve a ranking change.

1. Reseed the baseline from a clean worktree at `master`.
2. Record the corpus fingerprint the baseline was measured on, inside the
   baseline itself, so a later failure can state whether the corpus moved.
3. Report `Δcode` and `Δcorpus` as separate quantities: a same-corpus,
   two-revision comparison (`--against <ref>`) is the check a ranking change
   has to clear; the saved-baseline gate keeps guarding pushes but now names
   which of the two deltas it is looking at.

> An earlier draft of this section said "version the baseline in the
> repository". That is not available: the baseline is deliberately
> machine-local (`cli_eval.py:38` — "the gate runs against THIS machine's live
> index, so the baseline can't be a committed repo file"), and committing one
> machine's numbers would make the gate meaningless everywhere else. Recording
> the corpus fingerprint plus the two-revision comparison reaches the same
> goal — attribution — without contradicting that constraint.

**Verify**

- The same commit evaluated twice yields the same verdict.
- A deliberately introduced ranking regression fails the gate.
- A corpus-only change does not fail it.

### Phase 1 — Close the red

Classify the two remaining broad-exception sites in `recall_logic.py`: either
narrow the caught type or add them to `BROAD_EXCEPTION_ALLOWED` with a written
justification. CI needs no work — it already fails correctly.

**Verify**

- `pytest tests/test_dev_audit.py` green.
- PR #211's three failing checks green.

### Phase 2 — Retrieval

Three sub-fixes. Each runs the Phase 0 gate before and after.

**2a — Stop `retrieval_boost` from double-counting.** The 2026-08-05 diagnosis
offered three candidates. Take the second, not the first: apply the boost only
where the filename is a human curatorial act (ingested vault files), not to
memo-authored records whose title is derived from the same body text that
already matched. Capping the multiplier treats the magnitude; the cause is that
the signal is counted twice for auto-titled records.

**2b — Enable `MEMO_SEARCH_CHUNK_PARENT`.** Run the gate, then flip it. One
prerequisite: `_map_chunks_to_parents` calls `self.get(parent_id)` once per
chunk, an N+1 over the wide pool. Batch the parent fetch before enabling, or
the fix trades a ranking defect for a latency one.

**2c — Decide the `reference` tier's place in explicit search.** The recall
hook SQL-excludes the tier; `search` returns it freely, which is why seven of
the top ten on the control query are reference chunks. Choose one: a separate
pool, a `--scope reference` flag, or a rank floor. This is open question #1
from the 2026-08-04 audit, still unanswered, and it constrains 2a and 2b —
decide it first.

**Verify**

- The control query returns the answer-bearing record in the top 3.
- No query returns more than two chunks sharing a parent.
- `recall@5` at or above the Phase 0 baseline.

### Phase 3 — Latency

Independent of Phase 2; they touch transport and scoring respectively.

1. Route `memo search` and `memo ask` through `recall.sock` with a subprocess
   fallback, the same pattern the hook already uses.
2. Declare a latency budget per command and warn when a run exceeds it.
3. Emit per-pass progress for `maintain`, `dream` and `reindex`.

**Verify**

- `memo search` p50 under 2s warm.
- `memo ask` reports retrieval time and generation time separately, with
  retrieval under 2s. A single wall-clock number for `ask` hides which half is
  slow and must not be the acceptance criterion.
- `memo maintain` prints per-pass progress.

### Phase 4 — Instrumentation and the CLI/MCP surface

1. Generate the CLI↔MCP matrix. Resolve every gap explicitly as *add CLI*,
   *add tool*, or *intentional, documented*. Start with `review-due`.
2. Wire outcome and grounding recording for all consumers, not only
   `claude-code`.
3. Add a per-consumer quality floor that raises a maintenance nudge when hit
   rate falls below it over a rolling window. This depends on step 2: the
   nudge cannot fire on consumers whose outcomes are not recorded.

**Verify**

- The matrix is committed and every gap carries a disposition.
- `grnd%` is populated for at least three consumers.
- A synthetic per-consumer regression fires the nudge.

### Phase 5 — Reduction

1. Classify all 212 bool flags as **core-tuned**, **advanced**, or
   **internal**. Delete the three that are declared and never read:
   `MEMO_CRUSHER_CACHE_TTL_DAYS`, `MEMO_CRUSHER_ROWS_KEEP_RATIO`,
   `MEMO_STATUSLINE_ACTIVITY`.
2. Give each of the 31 default-off search/recall features one verdict: **flip**
   (the gate shows a gain), **delete** (no gain and no user), or **freeze** as
   an internal test seam. Phase 0 is what makes "the gate shows a gain"
   a statement with content.
3. Decide the fate of the 123 MCP tools outside the `agent` profile: promote,
   demote, or delete. A tool nobody can afford to load is not a feature.
4. Split the 13 source files over 1200 lines.
5. Finish the `mcp_tools.py` migration or abandon it in writing. A migration at
   6% with no finish line is worse than either outcome.

**Verify**

- Every flag carries a classification; the core-tuned set is documented in the
  reference manual.
- No source file exceeds 1200 lines.
- Test-suite runtime does not regress.

---

## Execution order

```
Phase 0  eval gate            unblocks Phase 2 and Phase 5's verdict
Phase 1  close the red        cheap, unblocks clean merges
Phase 2  retrieval            the lever; 2c decides before 2a and 2b
Phase 3  latency              independent of Phase 2, can run in parallel
Phase 4  instrumentation      Phase 4.3 depends on 4.2
Phase 5  reduction            consumes Phase 0's evidence; last
```

Phases 1 and 0 are independent and can proceed in either order. Phase 3 can
run in parallel with Phase 2. Phase 5 must be last, because its central
question — does this flag earn its existence — is answered by the harness
Phase 0 repairs.

## Where to cut if the program has to shrink

Cut from the end. Phase 2 holds nearly all the measurable value; Phase 5 has
the worst effort-to-payoff ratio in the program, and its payoff is
maintainability, which is not felt for months. Phases 0 and 1 are cheap enough
that cutting them saves nothing and costs the ability to verify everything
downstream.

## Open questions for the implementation plans

1. **2c mechanism** — namespace, tier flag, or `memo search --scope reference`?
   This affects the sync contract and the `memo chat` retrieval path, which
   legitimately wants the reference corpus. (Carried unanswered from
   2026-08-04.)
2. **2a blast radius** — how many records in the live store are
   "memo-authored, auto-titled" versus "ingested vault file"? The fix's value
   depends on that ratio; measure it before implementing.
3. **Phase 3 `ask` budget** — what is an acceptable generation time for a local
   30B MLX model? The retrieval budget is defensible at 2s; the generation
   budget needs a number derived from measurement, not preference.
4. **Phase 4 consumer scope** — `mcp-prompt` at 40% hit / 0% composite is worse
   than codex and newer. Diagnose whether it is a distinct defect or the same
   one before committing to a per-consumer floor.
5. **Phase 5 deletion authority** — deleting a default-off flag removes a
   documented knob. Does that require a major version bump under memo's
   compatibility policy?

## Non-goals

- No new features. Every item is repair, enablement, or reduction.
- No refactor of the graph, dream, or federation subsystems; the 2026-08-04
  audit found them populated and functioning, and nothing measured on
  2026-08-07 contradicts that.
- No deletion of ingested `reference` content.
- No change to the recall hook's 5s budget or its SQL exclusion of the
  reference tier — both are working and load-bearing.
