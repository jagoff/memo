Status: partially shipped — P1 (write-policy freeze) shipped in #192. P0 (search ranking dilution) was diagnosed down to the `retrieval_boost` multiplier (documented in this file's own "P0 diagnosis result" section) but the fix was never designed or implemented on master. P2 (daemon-backed CLI latency), P3 (codex consumer gap), P4 (unified score scale), P5 (flag classification), P6 (multi-session synthesis) were not started. Independently corroborated by an unmerged 2026-08-07 re-audit (`docs/SPECS/2026-08-07-repair-program-design.md` on local branch `docs/repair-program`, not yet a PR, not one of the 21 files in scope here): "P1 shipped; P0 was diagnosed and not fixed; P2-P6 were not started."

# memo audit — 7 prioritized findings

Date: 2026-08-04
Status: design approved, not yet planned
Scope: whole-product evaluation of memo's main features, with a ranked
improvement program. This document does not authorize implementation; each
finding needs its own plan.

---

## Why this audit

memo reached 148k LOC of source and 123k LOC of tests across 497 modules in
roughly three months, exposing 162 MCP tools (41 in the default `agent`
profile), 144 CLI commands, and 498 `MEMO_*` flags. External validation is
near zero: 12 stars, 5 forks, 0 open issues. Build velocity has outrun
validation velocity, so the question this audit answers is not "what could be
added" but "what does the existing surface actually deliver, measured on the
live store."

Every finding below is backed by a measurement taken on 2026-08-04 against the
production install (`uv tool` runtime, corpus at
`/Users/fer/repos/memo-sync/memorias`), not by reading documentation.

### Evaluation weights

Three lenses, weighted:

| Lens | Weight | Rationale |
|---|--:|---|
| Does memo serve the maintainer's daily loop | 50% | The only lens with measured signal (`memo usefulness`, `memo roi`, recall metrics). |
| Is memo maintainable at this size | 30% | 497 modules and 498 flags in three months is the structural risk. |
| Does memo attract external users | 20% | 0 issues means adoption cannot lead prioritization yet. |

---

## What the audit confirmed is working

These are load-bearing and should not be destabilized by the fixes below.

| Feature | Measurement |
|---|---|
| Recall hook + `com.memo.recall-daemon` | 1,884 hook fires, 96% hit rate, p50 829 ms / p95 1,735 ms / p99 1,930 ms over 7 days. |
| `memo ask` grounding | Returns a correct, cited answer and surfaces `⚠ Disputed evidence` with contesting IDs. 1,860 contradiction pairs on record. |
| Knowledge graph | 12,506 entities, 58,979 entity edges, 69,409 projection edges, 45,494 memberships, 6,770 code links, 5,280 semantic relations. Populated, not aspirational. |
| Dream pipeline | 160 `synthesis` memories produced; ledger and checkpoints current. |
| Value instrumentation | `memo usefulness` / `memo roi` measure whether a recalled memory was *used*, not just shown. Rare in this product category and the reason this audit could be quantitative at all. |
| CI | 14 workflows including mutation tests, test stability, contract-stub compatibility, and codegraph-affected shadow. Suite green at 7m28s. |

---

## Corpus baseline

| Metric | Value |
|---|--:|
| Live memories | 11,407 |
| Tombstoned | 858 |
| Namespaces | 62 |
| Chunk records | 361 |
| `reference` tier | 5,494 (48% of live) |
| Never accessed (`access_count = 0`) | 5,457 (48% of live) |
| …of which `reference` | 4,286 (78% of the tier) |
| Accessed 20+ times | 3,900 |
| Markdown files on disk | 9,246 |

Retrieval quality, live: hook hit rate 96%, `grounded_rate` 0.489,
**`referenced_rate` 0.011** (18 of 1,709 surfaced memories were later fetched).
Per consumer: claude-code 83% hit / 82% composite>0.85; codex 45% / 28%.

Retrieval quality, benchmark (`docs/eval/capability-baseline-and-levers.md`,
LongMemEval oracle, stratified n=60): micro recall@5 0.746, weakest bucket
`multi_session_synthesis` at 0.493.

---

> Finding numbers are identifiers, not execution order. The findings are
> presented in execution order; see [Execution order](#execution-order).

## P1 — Write policy frozen by a synthetic conflict

**Severity: critical. Cost: low. Do this first.**

A live `memo maintain` run emitted:

```
synthesize: save failed: Memo write policy blocked conflict
  conflict-9a4e7272767009fa: write frozen by native conflict: test conflict.
  Resolve the conflict or retry with an explicit override reason.
consolidation: merge-proposal LLM timeout
```

A conflict whose payload reads `test conflict` is freezing writes in the
production store, and the synthesis pass silently lost its output. The
`consolidation` pass separately timed out. **The command exited 0.**

Follow-up measurement (2026-08-04, after the spec was written): three abandoned
QA conflicts share the topic `test_conflict`, all with `freeze_write: true`.
Because `_conflict_matches_query` substring-matched the write's tokens against
the conflict topic, **every durable write whose topic contained the token
`test` was refused**. Confirmed through `WritePolicyEngine.preflight` against
the live store. A fourth QA artifact (`zzz_mcp_qa_probe_conflict`) matches the
same broken rule but has `freeze_write: false` and never refused anything —
measure blast radius through `preflight`, not through the matcher, which
ignores the freeze flag.

### Fix

1. Resolve or purge `conflict-9a4e7272767009fa` and audit for other synthetic
   conflicts in the durable conflict ledger.
2. Make the write coordinator distinguish test-origin conflicts from durable
   ones so a fixture can never freeze production writes.
3. `memo maintain` must surface per-pass failure: a failed save or a pass
   timeout is an error line and a non-zero exit, not a log line under a
   success banner.

### Verification

- `memo maintain` completes with no `save failed` and no silent pass timeout.
- A deliberately failing pass produces a non-zero exit code.
- Regression test: a synthetic conflict cannot block a durable write.

---

## P0 — Explicit `memo search` ranks the direct answer below topical neighbours

**Severity: high. Cost: unknown until diagnosed. Rewritten 2026-08-05.**

> This section replaces an earlier version built on corpus statistics rather
> than measured behaviour. Three of its four causal claims did not survive
> checking; the record of what was wrong is kept below, because the failure mode
> is more useful than the conclusion was.

### What is actually measured

Control query `por que se deprecó synapse`, run through `memo search`:

```
1.411  bug        Boost de título en synapse es plano
1.395  note       Fix de MyPy en Synapse
1.152  reference  Synapse — cerebro neutral operativo funcional § Fuentes
0.461  note       Decisión y estado final de la deprecación de Synapse
0.348  fact       synapse deprecado completo        <- the direct answer, rank 5
```

The record that answers the question ranks **fifth**, at a quarter of the top
score. What outranks it is mostly *durable* material that merely shares the
entity "synapse": a bug about title boosting, a mypy fix. This is a ranking
problem on the explicit-search surface, and the reference chunk at rank 3 is a
participant in it, not its cause.

Ambient recall is unaffected — see the correction above.

### What is NOT the problem (each checked, each disproven)

| Claim in the first draft | Status |
|---|---|
| Reference dilutes ambient recall | **False.** `MEMO_RECALL_EXCLUDE_REFERENCE=True`; 1,000 grounding-log lines contain zero reference records. |
| `referenced_rate=0.011` shows recall injects the wrong things | **False.** It counts explicit re-fetch via `record_click`, which recall makes unnecessary, across differently-bounded logs. |
| Dead-weight archival is broken (0 of 5,457) | **False.** `dead_weight()` targets memories surfaced ≥8 times and never grounded (`MEMO_OUTCOME_DEAD_MIN_SURFACED=8`, enabled). Zero means no recall noise was found — a healthy result. The 5,457 never-accessed records are reference, which recall never surfaces, so they are not candidates by construction. |

### What remains true

- **The search ranking above.** Reproducible, on the surface a human types into.
- **48% of the corpus (5,457 records) has never been accessed**, 78% of the
  reference tier. This is a real *cost* — index size (memvec.db at 1.3 GB),
  `maintain` wall clock, embedding and graph processing — with no demonstrated
  harm to recall quality. Treat it as hygiene, not as a retrieval defect.

### Next step: diagnose before designing

Do not write a fix yet. The open question is why entity-overlap outranks
answer-bearing content in `search` but apparently not in the recall hook, which
uses a different path (`recall_logic._recall_logic` versus `Memory.search`).
Establish that first:

1. Run the control query through both paths with score components exposed
   (`memo search --explain` / `MEMO_RECALL_DEBUG=1`) and diff the contributing
   terms.
2. Check whether the durable records that outrank the answer carry a title-boost
   or recency term that the answer lacks — note that one of them is literally a
   memory titled "Boost de título en synapse es plano".
3. Only then decide whether the fix is ranking, tiering, or both, and gate it on
   `memo eval recall`, which is load-bearing for a ranking change in a way it was
   not for P1.

### Process note

The first draft of this section measured *corpus composition* and inferred
*system behaviour* from it. The measurements were right; the inferences were
wrong three times out of four. Any future finding here states which of the two
it is, and a causal claim about retrieval is checked against `grounding.log` or a
live query before it is written down.

## P2 — Human-facing surfaces do not use the daemon

**Severity: high. Cost: medium.**

The recall daemon keeps a 4B MLX embedder hot and serves the hook in under a
second. Nothing a human types goes through it.

| Path | Latency |
|---|--:|
| Recall hook (daemon) | p50 829 ms |
| Recall subprocess | p50 2,264 ms, **p99 8,808 ms** — over the documented 5 s hook budget |
| `memo search` (warm) | **26 s** |
| `memo search` (cold) | **72 s** |
| `memo ask` | 42 s |
| `memo maintain` | **~10 min**, no progress output |

### Fix

1. Route CLI retrieval surfaces through `recall.sock` with a subprocess
   fallback, the same way the hook does.
2. Declare a latency budget per command and emit a warning when a run exceeds
   it, so regressions are visible instead of endured.
3. Long passes (`maintain`, `dream`, `reindex`) report progress and elapsed
   time per pass.

### Verification

- `memo search` p50 under 2 s warm.
- Recall subprocess p99 back inside the 5 s budget, or the subprocess path is
  removed from the hook entirely.
- `memo maintain` prints per-pass progress.

---

## P4 — Score scales are not comparable across surfaces

**Severity: medium. Cost: low. Blocks P0 and P3.**

`memo search` returns 3.499 and 0.564. `memo ask` returns sources scored 0.016
through 0.035. These are different quantities printed under the same column
name. No threshold expressed in one is meaningful in the other, which is why
"composite > 0.85" in `memo usefulness` cannot be reasoned about alongside a
search score.

### Fix

Define one normalized, documented scale. Re-express existing thresholds
(`MEMO_RECALL_MIN_SIM`, `MEMO_CHAT_RELEVANCE_FLOOR`, the composite gate,
abstention floors) in it. Keep raw scores available behind a debug flag.

### Verification

- Every user-visible score comes from the documented scale.
- The reference manual states the scale and what each threshold means in it.

---

## P3 — Per-consumer quality gap with no alert

**Severity: medium. Cost: low to diagnose, unknown to fix.**

| Consumer | Consults | Hit % | Composite > 0.85 |
|---|--:|--:|--:|
| claude-code | 124 | 83% | 82% |
| **codex** | **80** | **45%** | **28%** |
| mcp-test-client | 14 | 79% | 21% |

Same store, same embedder, half the hit rate. Nothing alerts on this; it was
only visible because `memo usefulness` was run manually during the audit.

### Fix

1. Diagnose first: capture the actual query text, parameters, and injected
   budget on the codex path versus the claude-code path and diff them. Do not
   propose a fix before the diff exists.
2. Add a per-consumer quality threshold that raises a maintenance nudge when a
   consumer's hit rate falls below a floor over a rolling window.

### Verification

- A written diagnosis of the codex gap with the measured difference.
- codex hit rate above 70%, or a documented reason it structurally cannot be.
- The nudge fires on a synthetic regression.

---

## P5 — 498 flags, roughly 38 ever exercised

**Severity: medium. Cost: high. Long-horizon payoff.**

498 `MEMO_*` flags are declared in `flags*.py` and 494 are consumed elsewhere
in the source — so this is not dead code. It is unexercised configurability:
the maintainer's own environment sets about 38 across shell, `settings.json`,
and launchd plists, and `memo config flags` shows the `active` column empty for
the defaults inspected. Roughly 92% of the knobs have exactly one user: their
own default value. `docs/reference.md` documents 98.

Each flag is a branch the test suite must cover and a degree of freedom nobody
exercises.

### Fix

1. Classify every flag as **core-tuned** (documented, supported, expected to be
   changed), **advanced** (documented, changeable at your own risk), or
   **internal** (test seam, not a public knob).
2. Freeze the internal set behind a non-public prefix or collapse it into
   constants where no test or environment exercises it.
3. Document the core-tuned set completely; `memo config flags` remains the live
   registry for the rest.

### Verification

- Every flag carries a classification.
- The core-tuned set is fully documented in the reference manual.
- Test-suite runtime does not regress after collapsing internal seams.

---

## P6 — Weak benchmark bucket: multi-session synthesis

**Severity: medium. Cost: high. Deliberately deferred.**

`multi_session_synthesis` scores recall@5 0.493 — half the cross-session
evidence is not in the top 5. `docs/eval/capability-baseline-and-levers.md`
already establishes that this is a retrieval *feature* gap (query decomposition
or multi-vector retrieval), not a knob, and that MMR and recency decay are
measured-negative levers.

**Recommendation: do not start this until P0 ships.** Part of the 0.493 is
plausibly candidate-pool dilution rather than missing decomposition, and P0
changes the pool. Re-measure the bucket after P0 and re-scope from the new
number.

### Verification

- The per-bucket bench is re-run after P0 and the result recorded here before
  any decomposition work is planned.

---

## Execution order

```
P1  write-policy conflict          cheap, and writes are broken now
P0  reference dilution + archival  biggest lever, depends on nothing
P2  daemon-backed CLI latency      independent, high daily value
P4  unified score scale            unblocks thresholds in P0 and P3
P3  codex consumer gap             diagnose first, then decide
P5  flag classification            long-horizon maintainability
P6  multi-session synthesis        re-measure after P0, then re-scope
```

P1 and P2 are independent and can proceed in parallel. P0 should land before
P6 is scoped. P4 should land before P0's cross-tier scoring rule is finalized.

## Open questions for the implementation plans

1. **P0 pool separation mechanism** — namespace, tier flag, or a dedicated
   `memo search --scope reference`? This affects the sync contract and the
   `memo chat` retrieval path, which legitimately wants the reference corpus.
2. **P0 archival destination** — archive to a subdirectory (existing
   `lifecycle.py` behavior) or tombstone? Archived content must remain
   reachable by `memo ask` if it is to stay useful to chat.
3. **P0 archival age threshold** — the `age > N days` term is unset. Pick N
   from the access-recency distribution rather than by intuition, and state the
   measurement in the plan.
4. **P3 scope** — is the codex gap worth fixing at all, given codex is a
   secondary consumer? The diagnosis is cheap; the fix may not be.

## Non-goals

- No new features. Every item above is repair or reduction.
- No deletion of ingested reference content.
- No refactor of the graph, dream, or federation subsystems; the audit found
  them populated and functioning.

---

## P0 diagnosis result (2026-08-05)

The diagnosis step above was run. The mechanism is identified.

### The cross-encoder gets it right; a downstream multiplier overrides it

Scoring the five candidates of the control query directly against
`MLXReranker.score`:

| final rank | final score | cross-encoder |
|---|--:|--:|
| 1 · bug "Boost de título en synapse es plano" | 1.411 | **0.013** |
| 2 · note "Fix de MyPy en Synapse" | 1.395 | 0.173 |
| 3 · reference chunk | 1.152 | 0.281 |
| 4 · note "Decisión y estado final de la deprecación" | 0.461 | **0.990** |
| 5 · fact "synapse deprecado completo" | 0.348 | **0.719** |

The reranker — the most expensive component in the pipeline — identifies both
answer-bearing records (0.990, 0.719) and rejects the irrelevant one (0.013).
The delivered order is close to its inverse.

### Which post-rerank stage does it

Stages after `rerank` are `entity_boost`, `verification_decay`,
`retrieval_boost`. Measured on these candidates:

- **`entity_boost` — no-op.** `extract_entities("por que se deprecó synapse")`
  returns `[]`, so `_apply_entity_boost` returns early.
- **`verification_decay` — uniform.** All five records are `unverified`, giving
  the same ×0.8. Order-preserving by construction.
- **`retrieval_boost` — the only stage that varies.** ×4.2 for four of the five,
  ×3.0 for the fourth.

`boost_for` (`retrieval_boost.py:126`) is a **multiplicative** curatorial boost
**capped at 12×**, composed from filename overlap (up to ×4.0), title ≥50% match
(×1.5), heading (×1.25) and tag match (×1.4). It is applied to a fused score
bounded at 1.0 by `alpha * rerank + (1 - alpha) * rrf_bonus` (α = 0.7). A term
that can reach 12× sitting downstream of a term bounded at 1.0 can reorder
anything.

### Why it misfires here

memo derives titles and filenames from memory content, so "the title mentions
synapse" is nearly universal among records about synapse. The boost therefore
fires at essentially the same magnitude for the record the cross-encoder scored
0.990 and the one it scored 0.013 — it encodes *entity mention*, not *answer
relevance*, and it is the largest multiplicative term in the pipeline.

The boost's own docstring states the intent: "a note whose metadata is the
answer wins decisively over body-text-only matches". That intent is sound for an
ingested vault of hand-named files, where a filename is a human curatorial act.
It does not hold for auto-titled durable memories, where the filename is derived
from the same text the body already matched — so the signal is double-counted
rather than curatorial.

### Fix direction (not yet designed)

The candidates, in increasing order of invasiveness:

1. **Bound the boost relative to the reranker** — cap the post-rerank multiplier
   so it can reorder within a relevance band but not across one.
2. **Make the boost curatorial again** — apply it only where the filename is a
   human artefact (ingested vault files), not to memo-authored records whose
   title is derived from their own body.
3. **Move the boost upstream of the reranker**, into candidate generation, so the
   cross-encoder has the last word on ordering.

All three are ranking changes, so `memo eval recall --labels
eval/regression_labels.json` is load-bearing and must be run before and after —
and, unlike P1, a regression here is meaningful rather than corpus drift.

Not started. This section records the diagnosis only.
