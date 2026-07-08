# Token-Economy Measurement Gate — Design

**Date:** 2026-07-08
**Status:** Approved design, pending implementation plan
**Topic:** `memo eval tokens` — a measurement gate that proves, per lever, whether a token-economy lever saves tokens without degrading recall.

## Problem

memo shipped a "token economy" across Wave 1+2 (merged at HEAD `94f3b94`/`310fcb5`/`39b2669`, unreleased at v2.12.17). It consists of four levers plus a reporting CLI. Exploration on 2026-07-08 found the edifice is almost entirely unwired scaffolding whose savings have never been measured:

| Lever | Apply point | Wiring reality |
|---|---|---|
| **L1 SmartCrusher** (drop low-relevance JSON rows on ingest) | `capture_core.py:1102` `maybe_crush_json_capture` | **Not wired** — only tests call it. Scorer is a placeholder `0.5` for every row, so "top-K" is just position order. Flag `MEMO_CRUSHER_ENABLED`, default OFF. |
| **L2 stream-compress** (drop LLM preamble tokens) | `stream_compress.py:16` `compress_token_stream` | **Not wired into memo** — operates on the LLM response token stream; no memo consumer. Flag `MEMO_STREAM_COMPRESS`, default OFF. |
| **L3 prefix-align** (reorder memories by SHA256 for KV-cache prefix stability) | `prefix_optimizer.py:78` `optimize_recall_prefix` | **Not wired into the hook path.** Benefit is KV-cache latency, not token count. Flag `MEMO_PREFIX_CACHE_ALIGN`, default OFF. |
| **L4 verbosity steer** (append a "skip preamble" steering block to recall output) | `cli_recall_hook.py:47` `maybe_inject_verbosity_steering`, applied at `cli_recall_hook.py:545` | **Wired only in the subprocess fallback**, not the primary daemon path (`recall_logic.py:_recall_logic`). It *appends* text — it adds tokens to the context to try to shrink the model's reply. Flag `MEMO_RECALL_VERBOSITY_LEVEL` (0–3), default 0. |

The reporting CLI `cli_token_savings.py` (`memo token-savings`) reports a **hardcoded** `compact_savings_pct = 65` and a `4 chars/token` heuristic — a fabricated number, not a measurement. The "savings gate" scripts (`scripts/wave1_token_baseline.py`, `scripts/wave2_token_baseline.py`) only count tokens over a hand-supplied `prompts.json`; there is no committed corpus, no before/after A/B, no gate, and they have never been run.

**Net:** three of four levers do not run in the live pipeline, the fourth adds tokens, and every savings claim is an estimate with no quality guard preventing a lever from "saving" tokens by dropping the right memory.

## Goal

A measurement gate — a structural clone of memo's existing `memo eval recall --gate` retrieval-regression discipline — that proves, per lever, over a committed corpus:

> `Δtokens ≤ −threshold` **AND** `Δquality within ε`

The data then decides which levers get wired ON and which get pruned. The gate's measured output replaces the hardcoded estimate in `memo token-savings`.

This is the same principle CLAUDE.md already mandates for retrieval ("every failed search → a system change, measured; never per-query"), applied to the token economy, which currently has no equivalent gate.

## Core principle: measure the transformation, not the wiring

Each lever declares a triple `(plane, apply_fn, quality_guard)`. The gate measures the lever's `apply_fn` against the corpus at **unit level** — it does **not** require the lever to be wired into the live path first. This is deliberate: 3/4 levers are dead code, and we refuse to wire dead code just to measure it. Measurement decides wiring; wiring winners is a per-lever follow-up.

## Measurement planes

A single "tokens saved" number cannot span the levers — each acts at a different plane. v1 covers the two planes memo controls directly and that require no LLM in the loop:

- **P1 — Recall-output size.** Tokens in the injected recall block (`render_recall_context` output, `recall_logic.py:103`).
  - **Quality guard:** precision@K over `eval/regression_labels.json`, reusing `eval_recall` (the labels already carry `expect_ids`, so the guard is free).
  - **Levers here:** L4 (adds tokens), and the *real* shrinkers already present — `MEMO_RECALL_FORMAT=compact`, `MEMO_RECALL_TOKEN_BUDGET`, `body_chars`.
- **P2 — Capture size.** Tokens written to disk on ingest.
  - **Quality guard:** "answer survived" — after crushing, the retained content must still retrieve the labeled must-keep row.
  - **Lever here:** L1 crusher.

**Deferred to v2** (require an LLM or timing loop; premature until levers are wired):
- **P3 — model-output tokens.** Where L2 and L4's *actual payoff* live. Needs an LLM in the loop (non-determinism, latency budget) → its own spec.
- **P4 — latency / KV-cache hit.** Where L3 lives. Needs timing/cache instrumentation → its own spec.

### The L4 paradox (an intended outcome, not a bug)

On P1, L4 always shows a **cost** because it appends a steering block. Its payoff is on P3, which v1 does not measure. The gate therefore honestly surfaces L4's value as *unproven* rather than assumed — exactly the correction this design exists to make.

## Corpus

- **P1:** reuse `eval/regression_labels.json` (schema `memo.eval_recall.labels.v1`) — no new corpus needed.
- **P2:** new committed `eval/token_corpus/capture_*.json` — a small set of realistic tool-output JSON arrays (search results, log dumps) each ≥10 rows, each labeled with a must-keep row index so the "answer survived" guard is checkable. Since L1 only fires on JSON arrays ≥10 rows, a hand-authored fixture set of that exact input class is the right scope; harvesting live captures is unnecessary.

## Harness

- **`src/memo/eval_tokens.py`** — structurally cloned from `eval_recall.py`:
  - `run(mem, cfg, corpus, levers) -> list[Row]` — for each lever: baseline tokens (lever OFF) vs treatment tokens (lever's `apply_fn` applied), plus the plane's quality metric OFF vs ON.
  - `gate_metrics(rows) -> dict` and `check_gate(rows, baseline, tol) -> GateResult` — same shape as `eval_recall.gate_metrics`/`check_gate` (`eval_recall.py:741`/`747`). A lever fails the gate if it does not reduce tokens by ≥ threshold, or if its quality metric regresses beyond ε.
- **`memo eval tokens`** — new subcommand in `cli_eval.py`, mirroring the recall gate CLI:
  - Baseline stored at `state_dir/eval/token_baseline.json` (parallels `recall_baseline.json` at `cli_eval.py:37`).
  - `--update-baseline` seeds it; `--gate` compares and `sys.exit(0 if passed else 1)` (parallels `cli_eval.py:222`); `--force` bypasses cache.
  - Prints a per-lever table: `Δtokens`, `Δquality`, `PASS/FAIL`.
- **Token counting:** the single `token_meter._CHARS_PER_TOKEN = 4` heuristic (`token_meter.py:147`). For a *delta* gate, consistency matters more than absolute accuracy; a real tokenizer can be swapped later behind the same call site without changing the gate contract.

## Honesty fix to `memo token-savings`

Rewrite `cli_token_savings.py`:
- Drop the hardcoded `compact_savings_pct = 65` and the estimate arithmetic.
- Report measured per-lever deltas sourced from the last gate run (or the durable ledger).
- A lever that has never passed the gate reports `0 tokens saved`, not an estimate.

## Disposition of losers

The gate produces a verdict per lever. Levers that fail or are structurally unmeasurable in v1 are marked deprecated in their flag docstrings. L1 specifically: it either earns a real relevance scorer that passes the P2 "answer survived" guard, or it is pruned. The gate decides — not vibes.

## Non-goals (v1)

- No LLM in the loop (P3/P4 deferred).
- No wiring of levers into the live path in this cut. Measurement first; wiring each winner is a separate follow-up spec.
- Machine-local and opt-in, exactly like `memo eval recall --gate` (runs against the live index / committed corpus, not GitHub CI).

## Testing

- **Unit:** `check_gate` pass/fail logic (token-delta AND quality guard), baseline seed/compare, corpus loaders for both planes.
- **Fixtures:**
  - A synthetic "good" P1 lever that shrinks the recall block without dropping any `expect_ids` → **PASS**.
  - L4 measured on P1 → shows a token **cost** (regression on the token axis), demonstrating the paradox is captured.
  - A P2 crush that drops the labeled must-keep row → **FAIL** on the quality guard.
- Reuse `eval_recall` test patterns and the `tmp_cfg` isolation fixture (`tests/conftest.py`).

## Key file anchors (from 2026-07-08 code map)

- Recall gate to mirror: `src/memo/eval_recall.py:741` `gate_metrics`, `:747` `check_gate`; `src/memo/cli_eval.py:37` `_baseline_path`, `:192` `--update-baseline`, `:222` exit code.
- Labels + precision: `eval_recall.py:185` `load_labels`, `:387` `_is_relevant`, `:446` `_run_config_inner`.
- Levers: `flags_capture.py:45` (L1), `flags_recall.py:537` (L2), `:543` (L3), `:528` (L4); apply points `capture_core.py:1102`, `stream_compress.py:16`, `prefix_optimizer.py:78`, `cli_recall_hook.py:47`.
- Metering + ledger: `token_meter.py:147` `_CHARS_PER_TOKEN`; `token_ledger.py:44` `ledger_path`, `:80` `read_ledger`, `:160` `roll_up`.
- Recall assembly chokepoint: `recall_logic.py:103` `render_recall_context`.
- Reporting CLI to fix: `cli_token_savings.py`.
</content>
</invoke>
