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

## P0 — Reference dilution and dead weight that never gets archived

**Severity: high. Cost: medium. Highest impact across all three lenses.**

Nearly half the corpus is ingested reference material — Obsidian vault notes,
WhatsApp exports, PDF chunks — and 78% of that tier has never been accessed
once. It is not inert: it competes in the same score space as durable agent
memory and wins.

Control query, run live:

```
memo search "por que se deprecó synapse"

  3.499  reference  "Dominios - Synapse, Memflow y Memo — 2. Synapse…"   ← vault chunk
  0.564  fact       "synapse deprecado completo"                        ← correct answer
```

The ingested chunk outranks the durable fact by roughly 6×.

Meanwhile the outcome loop reported `roi_score re-derived for 149 memories,
0 dead-weight archived` — the archival machinery exists and fired against
5,457 never-accessed candidates without archiving any of them.

### Correction (measured 2026-08-05, before planning P0)

Two claims in the paragraphs above are wrong. Both were written from corpus
statistics without checking what recall actually does, and measuring closed the
gap.

**1. Reference does not dilute ambient recall — it is already excluded.**
`MEMO_RECALL_EXCLUDE_REFERENCE` defaults to `True` and is effectively `True` on
this machine; `recall_logic.py:2112` drops `REFERENCE_TYPES` from the candidate
pool. Checking 1,000 lines of `grounding.log` against `meta`, the types recall
actually injects are:

```
389 note · 243 decision · 149 fact · 105 bug · 93 procedure
8 preference · 4 failure_pattern · 4 feedback · 2 synthesis · 0 reference
```

Zero reference records. The 3.499-versus-0.564 inversion is real but happens in
**explicit `memo search`**, which does not apply the exclusion — a CLI-surface
problem, not an ambient-recall one.

**2. `referenced_rate = 0.011` is not a failure signal.** It counts surfaced
memories that later appear in `usage.log`, which is written only by
`contextual.record_click` — reached from `memo get`, `memo contextual
record-click`, and the matching MCP tool. Recall injects the memory *body*, so
an agent has no reason to re-fetch by id; the metric measures a behavior the
design makes unnecessary. It is also bounded differently on each side:
`usage.log` is capped at 500 entries (244 today) while the denominator comes
from 1,709 surfaced records. The honest usefulness signal is `grounded_rate`
(0.489), which is outcome-based.

**What survives.** The dead weight is real and so is the search inversion; only
the ambient-recall framing and the `referenced_rate` inference were wrong. P0 is
re-scoped accordingly below.

### Fix (re-scoped after the correction)

1. **Apply the tier split to explicit `memo search`.** Recall already excludes
   reference; `search` does not, which is why an ingested vault chunk outranks
   the durable fact on the surface a human actually types into. Either exclude
   reference by default with an opt-in scope argument, or rank the tiers
   separately so they never compete on one incommensurable score.
2. **Make dead-weight archival actually fire.** The machinery reported
   `0 dead-weight archived` against 5,457 never-accessed candidates on two
   separate live runs. Define the criterion explicitly (`access_count = 0` ∧
   age > N ∧ not pinned), report candidates-matched alongside archived, and
   fail loudly when a non-empty candidate set archives nothing.
3. **Drop the cross-tier normalization sub-item** into P4, where the scale
   problem belongs. It was only ever needed because the tiers share a ranking;
   fix 1 removes that premise.

### Verification

- The control query (`por que se deprecó synapse`) ranks the `fact` above the
  vault chunk in `memo search`.
- A `maintain` run reports a non-zero archived count against a non-empty
  candidate set, and `memo stats` shows the never-accessed population shrinking.
- `memo eval recall --labels eval/regression_labels.json` precision does not
  drop and noise does not rise — this touches ranking, so the gate is load-bearing
  here in a way it was not for P1.
- **Not** `referenced_rate`: see the correction above.

### Explicitly out of scope

Deleting ingested content. The vault/WhatsApp corpus has value for `memo ask`
and `memo chat`; the problem is that it defaults into ambient recall, not that
it exists.

---

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
