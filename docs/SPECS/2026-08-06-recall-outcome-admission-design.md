Status: partially shipped — the adjacent prerequisite this design references as already "in flight" (`src/memo/recall_admission.py`, filtering harness-envelope prompts out of retrieval) shipped in #209. This design's own proposal — `src/memo/recall_utility.py`, a utility-based demote multiplier inside `rank_hits`, `MEMO_RECALL_UTILITY_ENABLED` — was never implemented; grep finds no `recall_utility` anywhere on master or on any local/remote branch.

# Recall admission by outcome — design

**Date:** 2026-08-06
**Status:** proposed
**Scope:** `src/memo/recall_utility.py` (new), `src/memo/recall_logic.py` (`rank_hits`), `src/memo/cli_dream_passes.py`, `src/memo/dream_flags.py`, `src/memo/flags_recall.py`

## Problem

The recall hook fires with a 96% hit rate and 94% top-composite rate. Those
measure that retrieval *found something confident*. They do not measure that the
injected block was worth its tokens.

Measured on this install (`memo usefulness`, 305 consults sampled;
`memo stats`):

| Metric | Value |
|---|---|
| tokens injected (bounded recent sample) | 246,585 |
| unique memories surfaced | 207 |
| `referenced_rate` — surfaced then later fetched | **0.011** (18 / 1702) |
| `grounded_rate` — surfaced and used in the answer | 0.501 (280 / 559) |
| recall hooks fired | 1884 |

Roughly half the injected block is ballast, paid on every prompt.

`src/memo/recall_admission.py` (in flight on `fix/bounded-mcp-payloads`) closes
the cheap half of this: 40% of 1500 consecutive hook fires were harness
envelopes (`<task-notification`, `<system-reminder`, …), each paying a full MLX
embed, a vec search and an injected block for a prompt no human wrote. That
decides **which prompts deserve retrieval**.

This spec decides **which memories deserve a slot**. Today that is settled by
composite score alone. Nothing in the ranking knows that a given memory has been
shown thirty times and used zero.

## What already exists (and changes the shape of this work)

`rank_hits` (`recall_logic.py:1214`) is already extracted and pure, and the eval
harness already ranks through it (`eval_recall.py:802` — *"Used by both
`_recall_logic` and the eval harness so they cannot diverge"*). `eval_ab.py`
provides a `flag_overrides` A/B seam over the same function.

CLAUDE.md still describes this extraction as deferred and ranking work as
therefore unmeasurable. That note is stale. There is no preparatory phase: the
change lands inside `rank_hits` and is measurable on day one.

`capture_weights.py` is the precedent for the mechanism proposed here — it
already joins `grounding.log` rows to memory type nightly and writes a weights
file consumed at capture time. This spec applies the same shape to recall
ranking, per memory instead of per type.

## Goals

1. Fewer injected tokens per consult at equal or better retrieval quality.
2. `referenced_rate` and `grounded_rate` up.
3. No memory becomes permanently invisible as a side effect.
4. Off by default, A/B'd nightly, self-reverting on regression.

## Non-goals

- Deleting or archiving low-utility memories. This is a ranking change only;
  `memo maintain` owns lifecycle.
- Changing retrieval (vec/BM25/fusion) or the reranker.
- Touching the `min_sim` floor, which the tuner owns.

## Design

### 1. Utility prior, computed nightly

New dream pass writes `state_dir/recall_utility.json`:

```json
{"schema": "memo.recall_utility.v1", "computed": "2026-08-06T03:00:12Z",
 "prior": 0.501,
 "memories": {"a1b2c3d4": {"surfaced": 31, "grounded": 0, "u": 0.031}}}
```

Source is `grounding.log` — the `(session_id, turn, recall_id)` recall→use
ledger — read via `dashboard_logs.read_grounding_log`.

Smoothing is Beta-Bernoulli with the corpus-wide grounded rate as the prior
mean, so a memory with one hit does not score 1.0 and, critically, a memory with
**no** history lands exactly on the prior:

```
u = (grounded + α·p) / (surfaced + α)     α = 8, p = corpus grounded_rate
```

`u == p` for an unobserved memory is the property the rest of the design leans
on.

### 2. Applied as a demote multiplier inside `rank_hits`

A new stage, after the existing boosts and before the `min_sim` gate:

```
score *= clip(1 + δ·2·(u − p), 1 − δ, 1 + δ)        δ = MEMO_RECALL_UTILITY_STRENGTH (0.15)
```

- Unobserved memory → `u = p` → multiplier exactly `1.0`. Nothing changes for
  anything the ledger has never seen.
- Never a hard drop. A demoted memory that still outranks the field is still
  injected.

Placing this in `rank_hits` and nowhere else is what keeps the daemon path, the
subprocess path, and the eval harness from diverging — the same reason the
function was extracted.

`RankKnobs` gains `utility: Mapping[str, float] | None`. The file is loaded once
per process and cached; ~10k entries of JSON is single-digit milliseconds, well
inside the 5s hook budget. Missing or malformed file → `None` → the stage is
skipped entirely.

### 3. Exploration slot

The cold-start trap is real and must be designed against, not noted: a memory
never surfaced is never grounded, so it is never surfaced. A pure exploit policy
freezes the corpus.

The last slot of the block is reserved for the highest-scoring candidate with
`surfaced < MEMO_RECALL_UTILITY_MIN_OBSERVATIONS` (default 3), **among
candidates that already passed the `min_sim` gate**. It is a reordering
preference within admitted candidates, never an injection of something retrieval
rejected. If no cold candidate qualifies, the slot goes to the normal ranking.

### 4. Variable K

Instead of a fixed `top_k`, drop the tail below a relative floor:

```
keep while score_i >= MEMO_RECALL_MARGINAL_FLOOR × score_0      (default 0.6)
```

Bounded by the existing `top_k` — K can only shrink, never grow. On a prompt
with one strong match and four weak ones, four slots of ballast disappear.

## Flags

All default OFF. Registered in `flags_recall.py`:

- `MEMO_RECALL_UTILITY_ENABLED`
- `MEMO_RECALL_UTILITY_STRENGTH` (0.15)
- `MEMO_RECALL_UTILITY_MIN_OBSERVATIONS` (3)
- `MEMO_RECALL_MARGINAL_FLOOR` (0.6)

`MEMO_RECALL_UTILITY_ENABLED` declares a `GateSpec` in `dream_flags.GATES` with
gate kind `recall`, A/B'd nightly through the eval `flag_overrides` seam.
`test_dream_flags.py` enforces this — a dark flag without a declared gate cannot
merge. Winners graduate to ON via the tuned overlay after
`MEMO_FLAG_GRADUATION_WIN_NIGHTS`; a regression auto-reverts.

## Testing

- Unit: smoothing arithmetic; unobserved memory → multiplier exactly 1.0;
  multiplier clipped at ±δ; missing/corrupt file → stage skipped, ranking
  byte-identical; exploration slot never admits a gate-rejected candidate;
  variable-K never grows K.
- Regression: `memo eval recall --labels eval/regression_labels.json --k 5`
  with the flag on vs off. precision@5, noise@5 and avoid@k must not regress.
- A/B: the nightly `dream_flags` gate is the acceptance mechanism, not a
  one-shot local measurement.

## Success criteria

Measured over a comparable consult sample, flag on vs off:

- injected tokens per consult **down**
- `grounded_rate` **up**
- `referenced_rate` **up**
- precision@5 / noise@5 / avoid@k on the curated label set **not worse**

Quality guard first: a token reduction that costs precision is a failure, and
the gate rejects it automatically.

## Risks

- **Feedback loop.** Addressed by the exploration slot and by demote-not-drop.
  Worth re-measuring after a month of nightly runs: if the set of memories with
  `surfaced == 0` grows monotonically, the exploration budget is too small.
- **`grounding.log` is thin for a young corpus.** The prior handles it — with no
  observations the whole stage is a no-op, so the feature is inert rather than
  wrong on a fresh install.
- **The eval gate measures corpus state, not the diff.** Recorded from the
  2026-08-06 sweep: draining the contradiction backlog fired `negative_capture`
  32 times, the new anti-memories displaced an `expect_avoid_id` from the top-5,
  and avoid@k fell 1.0 → 0.5 with no ranking code involved. When the gate blocks
  a push, confirm the cause is this change before reaching for `--no-verify`.
- **Ledger keyed on 8-char id prefixes.** Collision probability is negligible at
  10k memories but the join should be prefix-safe, matching `grounding.py`'s
  existing `match_cited` behaviour rather than inventing a second rule.
