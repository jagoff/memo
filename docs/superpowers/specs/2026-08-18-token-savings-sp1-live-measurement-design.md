# SP1 — Token Savings: Live-Distribution Measurement (Replay on Real Emissions)

Date: 2026-08-18
Status: Approved design
Branch: feat/emission-ledger

## Context

memo's token-savings surface today measures every lever at **unit level** over a
fixed committed corpus (`memo eval tokens` → `eval/token_baseline.json`), and the
CI gate only guards levers that passed there. That measurement is honest but
distribution-blind: the corpus is 46 prompts + 2 crush cases, while real recall
traffic differs in block sizes, hit counts, session turns, and capture row
shapes. A lever can lose the unit gate yet win on real traffic (observed:
`verbosity_steer_L2` fails the unit gate because the steering directive costs
more than it saves on tiny synthetic blocks) — or win the unit gate and lose in
the field.

This is the first of four sub-projects from the token-savings review of nine
external token-optimization projects (caveman, ccusage, token-optimizer, snip,
headroom, mcp-server-code-execution-mode, jcodemunch-mcp, entroly,
awesome-llm-token-optimization). Ideas were evaluated for license compatibility
(BSL-1.1 / PolyForm Noncommercial / GPLv3 / dual-use repos are **idea-only**,
never copied; memo implements natively) and against memo's existing surface,
which already covers: capture crusher (L1), recall verbosity steering (L4),
minimal MCP tool profile, emitted ledger, repo intelligence, token meter/ledger,
and the measurement gate.

## Goal

Measure every flag-gated token lever on memo's **real emitted traffic** without
touching the runtime hot path, so wiring decisions are made on data: replay the
emitted recall ledger, re-render each emission OFF vs ON per lever, and report
token deltas + quality (id survival) per lever with sample counts.

## Non-Goals (SP1)

- No changes to runtime behavior, hook hot path, or the recall daemon.
- No CI gate on the live distribution (it drifts; gates stay on the fixed corpus).
- No change to lever defaults, unless a lever wins live with large margin — and
  then only with user approval after seeing the numbers.
- No absorb (MEMO_SAVE_ABSORB) — storage plane, not context plane.
- No provider-side token attribution from agent session logs (SP2).

## Section 1 — Lever audit

`src/memo/lever_catalog.py` — a data module, single source of truth for the
audit table, the replay engine, and the CLI. Each entry:

```python
@dataclass
class LeverSpec:
    name: str                    # unique, kebab-case
    flag: str                    # MEMO_* env flag that gates the lever
    plane: str                   # "recall_output" | "capture"
    measurer: str                # registered measurer id (P1 render | P2 crush)
    gateable: bool               # False => listed with reason, never measured
    reason: str | None           # why not gateable, if gateable is False
```

Audit inventory (verified against `flags_recall.py` / `flags_capture.py`):

| Lever | Flag | Plane | Gateable |
|---|---|---|---|
| recall_format_compact | `MEMO_RECALL_FORMAT` | recall_output | yes |
| verbosity_steer_L1..L4 | `MEMO_RECALL_VERBOSITY_LEVEL` | recall_output | yes |
| footer_short / footer_none | `MEMO_RECALL_FOOTER_STYLE` | recall_output | yes |
| feedback_hint_off | `MEMO_RECALL_FEEDBACK_HINT` | recall_output | yes |
| directive_turn1_only | `MEMO_RECALL_DIRECTIVE_ONCE` | recall_output | yes |
| body_chars | `MEMO_RECALL_BODY_CHARS` | recall_output | yes (quality-sensitive) |
| crusher_L1 | `MEMO_CRUSHER_ENABLED` | capture | yes |
| save_absorb | `MEMO_SAVE_ABSORB` | capture (indirect) | no — storage plane |

Measurers: P1 = render-based (recall_output), P2 = crush-based (capture).
A lever is gateable iff flag-gated AND measurable by deterministic render/crush
without an LLM on the path.

The catalog is the **superset**: the existing fixed-corpus gate
(`eval_tokens.RECALL_LEVERS`, today only format + verbosity L2) keeps its own
list untouched — the catalog drives the live engine and the audit CLI only.

CLI: `memo lever-audit [--json]` prints the table. New lever = new catalog row.

## Section 2 — Replay engine

`src/memo/eval_tokens_live.py`:

**Source**: `state_dir/emitted/<session>.json` (emitted ledger entries:
`memory_id`, emitted `text`, `turn`, `ref` id set, `src` channel).

**Sampling** (anti-bias, bounded cost):
- Window: `--since <days>` (default 14)
- Cap: `--max-samples` (default 200), deterministic via `--seed` (default 42)
- Dedup by `emitted_hash` — a block re-emitted across sessions counts once
- Stratified by `src` (recall-hook / ask / search); report separates channels

**Hit reconstruction**: fetch each `memory_id` (title + body) from the vec
store. Missing ids → sample dropped, counted under `skipped:stale` (never
counted against quality).

**Re-render**: `render_recall_context(hits, [], turn=t, body_chars, token_budget)`
under `env_pins` per lever — the same deterministic, LLM-free function the
existing gate uses. OFF = baseline render; ON = lever active (verbosity levers
additionally apply `maybe_inject_verbosity_steering`). body_chars/token_budget
are the current flag values — the lever is measured against today's real
render, documented assumption.

**Quality**:
- P1 precision = fraction of `ref` ids whose 8-char prefix survives into the
  re-rendered block (reuses `surviving_ids`).
- P2 crusher: sample recent real JSON capture rows from the store; crush with
  `MEMO_CRUSHER_ENABLED=1`; must-keep row survival via `_row_survived`.

**Output**: existing `LeverRow` schema plus `n_muestras`, date range, and
`skipped` counts by category. Reuses `aggregate_recall` / `aggregate_capture` /
`gate_metrics` — no duplicated math.

## Section 3 — CLI + report

- `memo lever-audit [--json]` — catalog table
- `memo eval tokens --live [--since N] [--max-samples N] [--seed N] [--json]`
  — runs replay, prints per-lever `saved%`, `Δquality`, `n`, `skipped`
- `memo eval tokens --live --save-baseline` — writes
  `state_dir/eval/token_live_baseline.json` (informational snapshot)
- `memo token-savings --live` — extra section in the existing report: levers
  measured on live distribution + sample count + snapshot date; honest
  "no live baseline yet" message when absent (same pattern as today)

Deliberate non-gate: live distribution drifts, so it never gates CI. The fixed
corpus gate (`token_baseline.json`) is untouched.

`--live` is read-only on the index (no search, no `access_count` inflation).

## Section 4 — Error handling

- Missing/corrupt ledger files → skipped with count, never crash
- Orphaned memory ids → `skipped:stale`
- No samples in window → honest empty report with hint (mirror existing
  `token-savings` empty-state)
- Unrecognized lever flag in catalog → validation error at audit time (tests)
- Replay failures are contained per-sample; a broken sample never fails the run

## Section 5 — Testing

- Unit (catalog): unique names, flag exists in flag registry, measurer resolves
- Unit (sampling): seed determinism, hash dedup, src stratification,
  stale-skip accounting
- Unit (replay): synthetic ledger fixture (2-3 sessions, mixed src, duplicates,
  orphan ids) against a Memory stub (no MLX) → stable expected LeverRows
- CLI: CliRunner with isolated `MEMO_DATA_DIR`/`MEMO_STATE_DIR`; `--json`
  parseable; `--save-baseline` writes file; `token-savings --live` empty-state
- CI: runs in default suite (`not slow and not conformance`); live run is
  manual, like the existing gate
- License note: native implementation; no code lifted from BSL-1.1 /
  PolyForm-NC / GPLv3 / dual-use sources

## Section 6 — Wiring (final step, data-driven, optional)

If a lever passes live with margin (`saved_frac >= 0.10`, `quality_delta >= 0`,
`n >= 50`), propose flipping its default flag + CHANGELOG entry + re-measure.
No defaults change inside SP1 without user approval of the numbers.

## Out of scope (later sub-projects)

- SP2: token-sink audit from agent session logs (ccusage/caveman-learn style)
- SP3: structure maps / symbol retrieval on existing codegraph
- SP4: cache-safe injection hardening + tool-output crush extension
