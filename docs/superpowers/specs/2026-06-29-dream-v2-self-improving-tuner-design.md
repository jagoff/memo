# Dream v2 — Substrate + Self-Improving Tuner (Phase 0 + Phase 1)

- **Date:** 2026-06-29
- **Status:** Approved design — ready for implementation plan
- **Scope:** `memo dream` evolves from nightly janitor (clean/dedupe/synthesize/prune) into a
  measured self-improving system. This spec covers **Phase 0** (shared substrate) and
  **Phase 1** (the self-tuning retrieval optimizer). Phases 2 (episodic→semantic
  consolidation) and 3 (anticipatory memory) reuse this substrate and get their own
  specs later.

## Motivation

Today `memo dream` mutates the corpus (prune, compress, synthesize) but **never measures
whether retrieval got better or worse**. Ranking parameters (`MEMO_RECALL_*` boosts,
pool sizes, thresholds) are hand-tuned constants. memo already has every ingredient to
close the loop and never uses them together:

- **Ground-truth-by-use signal** — `grounding.log` records, per turn, which recalled
  memory was actually *used* in the answer (`used_score`, `grounding_used()`), joinable
  to the prompt via `_prompt_for_turn`. `eval_recall.py:472-511` already reads
  `grounding.log` and pairs it with the turn's prompt.
- **A measurable gate** — `eval_recall.py` computes precision@K / noise@K over the live
  index (retrieval-only, no MLX, ~0.5s/prompt) with `evaluate()`, `gate_metrics()`,
  `check_gate()`, and a persisted baseline.
- **Tunable params** — `flags_recall.py`: `MEMO_RECALL_PROJECT_BOOST`,
  `MEMO_RECALL_GLOBAL_BOOST`, `MEMO_RECALL_MIN_SIM`, `MEMO_RECALL_RERANK_INPUT_K`,
  `MEMO_RECALL_STALENESS_DAYS`, etc.

The gamechanger: **memory that measurably learns to retrieve better every night, from its
own real usage, with zero human labeling** — gated, reversible, opt-in.

## Non-goals (YAGNI / honesty)

- **No code/ingest auto-changes.** Structural ingest/ranking findings are *reported* to
  the human, never auto-applied. Night-time autonomy is limited to **parameters**.
- **No re-litigation of settled decisions.** `MEMO_RECALL_MODE` (vec) is excluded from
  the search space — hybrid was measured over 130 labels and **rejected** (vec wins
  prec@5 0.095 vs 0.062 *and* latency 28ms vs 9.6s). Hard-pinned with a code comment +
  memory reference.
- **No LLM in the tuning loop.** Evaluation is retrieval-only → fast, deterministic,
  CI-friendly.
- **No Bayesian/RL optimizer.** Coordinate descent is sufficient at this corpus size
  (~126 durable memories).

## Architecture

Five small units, each one responsibility, each independently testable.

### Phase 0 — substrate

#### 1. `dream_labels.py` — self-supervised label miner
- **Does:** mines `(prompt → used-memory)` pairs from real usage and emits eval labels.
- **Input:** `grounding.log` × `recall.log` (reuse `read_grounding_log` and
  `_prompt_for_turn`; reuse the `grounding_used(row)` production decision so the miner's
  notion of "used" matches the detector).
- **Per qualifying turn** (memory genuinely used — `grounding_used()` true, weighted by
  `used_score`): emit
  `{"prompt": <turn prompt>, "expect_ids": [<recall_id>], "relevant": true,
    "source": "grounding", "weight": <used_score>, "session_id", "turn"}`.
- **Output:** appends to `state_dir/eval/auto_labels.jsonl` (schema
  `memo.eval_recall.labels.v1`). **Machine-local, never committed**, always tagged
  `source: "grounding"` so it is never confused with the curated
  `eval/regression_labels.json`.
- **Hygiene:** dedup by `(prompt, expect_id)`; cap total; drop turns whose prompt is
  empty/too short; cap per-prompt expect_ids.
- **How to use:** `mine_grounding_labels(state_dir, *, limit, min_used_score) -> list[Label]`
  plus a writer that merges into `auto_labels.jsonl` idempotently.
- **Depends on:** `eval_grounding.grounding_used`, `eval_recall.read_grounding_log`,
  `eval_recall._prompt_for_turn`, `cfg.state_dir`.

#### 2. `dream_gate.py` — measurement + auto-rollback harness
- **Does:** measures retrieval quality on a label set against the live index and decides
  accept/reject/rollback.
- **API:**
  - `measure(labels, *, k) -> {"prec_at_k": float, "noise_at_k": float, "n": int}`
    (thin wrapper over `eval_recall.evaluate` + `gate_metrics`; retrieval-only, no MLX).
  - `load_baseline(state_dir) / save_baseline(state_dir, metrics)` →
    `state_dir/eval/dream_baseline.json` (per-machine).
  - `accepts(before, after) -> bool` via `eval_recall.check_gate` semantics: precision
    must not drop and noise must not rise (within tol).
- **Depends on:** `eval_recall` (evaluate/gate_metrics/check_gate), live `Memory`/index.

#### 3. `dream_receipt` (extend existing receipt) + morning briefing
- **Does:** records a structured artifact of the night and surfaces it to the human.
- **Receipt fields added:** `mined_labels`, `tuner: {before, after, params_before,
  params_after, applied|rejected|rolled_back}`, and **`errors` populated — not
  swallowed** (Phase 0 replaces the silent `except Exception: pass` in the tuner/miner
  paths with error capture into the receipt).
- **Briefing:** SessionStart "El Briefing" gains a one-line dream summary:
  `"anoche · tuner: prec@5 0.20→0.24, params {project_boost 0.25→0.30} · 1 rollback"`.
- **Depends on:** existing `cli_dream` receipt dict, the SessionStart briefing surface.

### Phase 1 — self-improving tuner

#### 4. `dream_tuner.py` — the optimizer pass
- **Does:** searches the bounded ranking-param space for a config that improves the
  union eval, then applies it only if it passes the curated gate.
- **Runs:** as a new dream phase, **after** orientation, **before** the destructive
  passes (prune/evict/compress) so tuning sees the pre-mutation corpus.
- **Input labels:** `auto_labels.jsonl` ∪ committed `regression_labels.json`. Curated
  labels carry higher weight.
- **Search space (bounded, continuous):**
  | Param | Default | Range | Max Δ/night |
  |---|---|---|---|
  | `MEMO_RECALL_PROJECT_BOOST` | 0.25 | 0.0–0.5 | 0.05 |
  | `MEMO_RECALL_GLOBAL_BOOST` | 0.10 | 0.0–0.3 | 0.05 |
  | `MEMO_RECALL_MIN_SIM` | (current) | bounded | small |
  | `MEMO_RECALL_RERANK_INPUT_K` | 10 | 6–20 | 2 |
  | `MEMO_RECALL_STALENESS_DAYS` | (current) | bounded | small |
  Exact ranges finalized in the plan; `MEMO_RECALL_MODE` is **not** in the set.
- **Method:** coordinate descent — one param at a time, try ±step, keep the change if it
  improves the union measure, bounded to ≤K total evaluations per night. Deterministic
  (no RNG), cheap.
- **Output:** a candidate param dict (does not write anything itself; apply is gated).

#### 5. `tuned_params.json` overlay + flag-accessor hook
- **Does:** the apply mechanism the recall path reads.
- **File:** `state_dir/tuned_params.json` =
  `{"<param>": <value>, ..., "_meta": {"set_by": "dream", "ts": ..., "prev": {...},
   "baseline_prec": ..., "baseline_noise": ...}}`.
- **Precedence (critical):** **explicit env var > overlay > built-in default.** The tuner
  never overrides a human-set `MEMO_RECALL_*` env var. Integration point: the typed
  `flag_float/flag_int` accessors (or a single overlay-aware resolver they call).
- **Reversibility:** `_meta.prev` holds the previous values; `memo dream tune --rollback`
  restores them; deleting the file = pure defaults.

## Data flow (nightly, inside `memo dream run`)

```
grounding.log + recall.log
        │  dream_labels.mine_grounding_labels
        ▼
auto_labels.jsonl ─────┐
committed labels ──────┤  (union; committed weighted higher)
        ▼              │
dream_gate.measure(union) ──► metrics_before
        ▼
dream_tuner: coordinate descent over bounded params
        │   each candidate → dream_gate.measure(union)
        ▼
best candidate → dream_gate.measure(COMMITTED set)   ← the trusted gate
        ▼
dream_gate.accepts(baseline, after_on_committed)?
   ├─ yes → write tuned_params.json (auto-apply) + save_baseline + receipt(applied)
   └─ no  → discard, no write + receipt(rejected)
        ▼
next session's recall reads tuned_params.json (live)
        ▼
[a later night] measure live config vs baseline → regressed?
        └─ yes → rollback overlay to _meta.prev + receipt(rolled_back)
```

### Two safety layers (why auto-apply is acceptable)
1. **Committed-set veto.** A candidate must beat the baseline on the **human-curated**
   `regression_labels.json` before it is applied — even if the auto-labels love it. This
   neutralizes the survivorship bias of the grounding signal (you can only ground a
   memory that was surfaced; the right-but-never-surfaced memory produces no positive
   label). Auto-labels *propose*; curated labels *veto*.
2. **Continuous re-measurement.** Every night re-measures the live config against the
   baseline; if reality disagrees, **auto-rollback**.

## Guardrails

- **Committed-set veto** (above).
- **Env override wins** — explicit `MEMO_RECALL_*` is never tuned.
- **Settled-decision pin** — `MEMO_RECALL_MODE` excluded; comment cites the vec-vs-hybrid
  rejection memory.
- **Bounded ranges + max Δ/night** — no param leaves its sane range or jumps hard.
- **Full reversibility** — `--rollback`, `_meta.prev`, delete-file-to-default.
- **Flag-gated OFF by default** — `MEMO_DREAM_TUNE_ENABLED=0`. Opt-in.
- **Errors surfaced** — tuner/miner failures land in `receipt.errors`, never silent.

## Error handling

- Miner: a malformed grounding/recall row is skipped with a counted warning (not a silent
  pass); the miner never raises into the dream pipeline.
- Gate: a measurement failure (index unavailable, label parse error) aborts the tuning
  phase cleanly — dream continues its other passes; receipt records `tuner: skipped`.
- Tuner: any exception → no overlay write, receipt records the error. Default-safe: when
  in doubt, **do not change live params**.
- Overlay: a corrupt `tuned_params.json` is ignored (fall back to defaults) and flagged in
  the next receipt; recall never crashes on a bad overlay.

## Configuration (new flags)

- `MEMO_DREAM_TUNE_ENABLED` (bool, default `0`) — master switch for Phase 1.
- `MEMO_DREAM_TUNE_MAX_EVALS` (int) — cap on per-night evaluations (cost ceiling).
- `MEMO_DREAM_TUNE_K` (int, default 5) — the K for prec@K/noise@K during tuning.
- Phase 0 miner: `MEMO_DREAM_MINE_MIN_USED_SCORE` (float), `MEMO_DREAM_MINE_LIMIT` (int).
All via `flags_<group>.py` + typed accessors; covered by `memo config validate`.

## CLI surface

- `memo dream tune --dry-run` — run the tuner, print before/after + candidate, write
  nothing.
- `memo dream tune --rollback` — restore `_meta.prev`.
- `memo dream tune --status` — show current overlay, baseline, last decision.
- The full `memo dream run` invokes the tuner phase when `MEMO_DREAM_TUNE_ENABLED=1`.

## Testing (all MLX-free → CI-friendly)

- **Unit — miner:** synthetic grounding/recall rows → expected labels; dedup, cap,
  empty-prompt drop, weight = used_score.
- **Unit — gate:** `accepts()` accept/reject/rollback decisions over synthetic
  before/after metrics; baseline load/save round-trip.
- **Unit — overlay precedence:** env > overlay > default (pin `MEMO_EMBEDDER_DIMS` and use
  isolated `Config` per conftest).
- **Integration — tuner:** over a tiny isolated index, deliberately detune one param;
  tuner recovers it and gates correctly. A candidate that helps auto-labels but hurts the
  curated set is **rejected** (committed-set veto proven).
- **Integration — rollback:** simulate a next-night regression → overlay rolls back to
  `_meta.prev`.
- **Regression:** existing `memo eval recall --gate` stays green; the tuner never lowers
  the committed baseline.
- Isolation per `tests/conftest.py` (`tmp_cfg`, `MEMO_NONINTERACTIVE=1`,
  `MEMO_DATA_DIR`/`MEMO_STATE_DIR`); never touch the real vault.

## Open questions for the plan

- Exact numeric ranges/steps per param (table above is the starting point).
- Whether the morning-briefing line ships in Phase 0+1 or waits for Phase 3's briefing
  work (leaning: ship a minimal line now — it is the trust mechanism for auto-apply).
- Coordinate-descent ordering / step schedule (single pass vs two passes).
