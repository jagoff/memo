# memo — Trust + Adoption Doctor

**Date:** 2026-07-08
**Status:** approved design, pending written-spec review
**Scope:** integrate Wave 2 confidence/lifecycle signals with `memo usefulness` so memo can diagnose whether agents are consulting memory, attributing consults, grounding retrieved memories, and relying on trustworthy records.

## Context

memo already has two mostly separate signal families:

1. **Adoption telemetry** — `memo usefulness` reads consult logs, consumer attribution, hit rates, grounded rates, and silent expected consumers.
2. **Corpus trust telemetry** — Wave 2 confidence/lifecycle work adds or uses `support_count`, confidence, invalidation stamps, supersedence, and mutability signals.

The gap is not another ranking knob. The gap is that a user cannot run one command and answer: "Is memo being used by my agents, and are they using the right memories?"

This spec builds that command as a read-only diagnostic layer. It does not mutate memories, change ranking, call MLX, or touch the recall-hook hot path.

## Goals

- Show whether expected clients are consulting memo and passing `source=...`.
- Show whether consults are resulting in grounded/useful recall.
- Surface trust problems in the memories that are being used.
- Surface starvation in trust signals, especially sparse `support_count`.
- Produce concrete actions the user or agent can take next.
- Keep the output machine-readable enough for dashboards or CI.

## Non-goals

- No automatic fixes in the first version.
- No ranking changes.
- No recall-hook changes.
- No LLM calls.
- No new persistence schema unless a small read helper is required for existing fields.
- No attempt to complete all remaining Wave 2 C tasks inside this feature.

## Architecture

Add a read-only module:

```text
src/memo/usefulness_doctor.py
```

It reads existing sources:

- `dashboard_metrics.consult_breakdown()`
- `dashboard_metrics.recall_health()`
- `dashboard_logs.read_*()` helpers where detailed rows are needed
- `VecStore` or existing signal dump helpers for `memory_health`
- memory metadata for tags and `extra` values such as `_invalidated`, `invalidated_reason`, `superseded_by`, and confidence-related fields

It returns a structured report:

```python
{
    "verdict": "healthy" | "degraded" | "silent" | "untrusted" | "unknown",
    "adoption": [DiagnosticItem, ...],
    "trust": [DiagnosticItem, ...],
    "actions": [ActionItem, ...],
    "summary": {...},
}
```

`DiagnosticItem` should be plain dictionaries rather than a public class unless the implementation already has a local pattern for small dataclasses in CLI reports. JSON stability matters more than internal object shape.

Add CLI surface:

```bash
memo usefulness doctor
memo usefulness doctor --json
memo usefulness doctor --limit 1000
```

If Click command nesting makes this awkward with the current `usefulness` command, use:

```bash
memo usefulness --doctor
memo usefulness --doctor --json
```

The preferred UX is the subcommand, but implementation should follow the existing CLI structure with minimal churn.

## Checks

### 1. Silent Consumers

Detect expected consumers with zero consults.

Evidence:
- `consult_breakdown()["silent"]`

Action:
- Configure the missing client to call memo.
- Ensure MCP tools pass `source="<client>"`.

Severity:
- `warning` if at least one expected consumer is active.
- `critical` if all expected consumers are silent.

### 2. Anonymous Or Unattributed Consults

Detect consult rows that do not carry a useful `source`, or that collapse into a generic/unknown consumer.

Evidence:
- consult log rows and `consumer_label()`

Action:
- Add `source="<client>"` to memo read tool calls.

Severity:
- `warning`, because recall can still work but `memo usefulness` cannot prove client adoption.

### 3. Consults Without Grounding

Detect consumers with consult volume but weak grounding.

Evidence:
- `recall_health()`
- `consult_breakdown()` grounded fields where available

Action:
- Run `memo debug-recall <recent prompt>` when available.
- Inspect low-score/noisy memories.
- Treat this as a retrieval-quality problem, not an adoption problem.

Severity:
- `warning` when consults exist but grounded rate is absent or low.

### 4. Trusted Memories Not Used

Detect high-trust records that rarely or never appear in grounded recall.

Evidence:
- `memory_health.support_count`
- confidence/roi score
- grounding log memory ids

Action:
- Check project/global scope.
- Check duplicate or stale versions.
- Consider targeted retrieval eval labels if the memories should have appeared.

Severity:
- `info` by default. Escalate to `warning` only for records above a high threshold, such as `support_count >= 3`, that are never grounded in the sampled window.

### 5. Untrusted Memories Being Used

Detect low-trust records that do get grounded.

Evidence:
- grounding log memory ids
- `memory_health.confidence`
- tags/extra containing `_invalidated`, `invalidated_reason`, `superseded_by`, or equivalent existing fields

Action:
- Update or delete stale records.
- Run `memo invalidate --undo` only if the invalidation was wrong.
- Run contradiction triage if the memory is superseded but still used.

Severity:
- `warning` for low confidence.
- `critical` for invalidated or superseded memories grounded recently.

### 6. Lifecycle Signal Starvation

Detect whether confidence/lifecycle signals are not accumulating.

Evidence:
- count of `memory_health` rows
- count and ratio with `support_count > 0`
- optional presence of `support_count` column

Action:
- Verify corroboration bump sites.
- Verify signal export/import preserves `support_count`.
- Keep `MEMO_SUPPORT_CONFIDENCE_LIFT` separate from this check; the counter itself should be visible even if ranking lift is disabled.

Severity:
- `info` for small corpora.
- `warning` when corpus is large and `support_count` is effectively all zero.

## Output

Human output should be compact and action-oriented:

```text
memo trust + adoption doctor

verdict: degraded

adoption
  - codex: healthy, 42 consults, source attributed
  - memflow: silent, 0 consults
    action: configure memflow to call memo tools with source="memflow"

trust
  - support_count signal is sparse: 1/6231 memories have support_count > 0
    action: verify corroboration bump sites and sync-signal export
  - 3 invalidated memories were grounded recently
    action: update those memories or run contradiction triage
```

JSON output must keep stable top-level keys:

- `verdict`
- `adoption`
- `trust`
- `actions`
- `summary`

Individual items should include:

- `id`
- `severity`
- `status`
- `message`
- `evidence`
- `action`

## Error Handling

- Missing logs: return `verdict="silent"` with an action to run a session or check hooks.
- Missing DB: return `verdict="unknown"` with an action to run `memo doctor`.
- Missing `support_count` column: report `schema_missing` and suggest the existing migration/reindex path.
- Malformed log rows: skip bad rows, count them in `summary["malformed_rows"]`.
- Any reader failure should degrade the relevant section only; the command should still render partial results.

## Testing

Add focused tests:

- `tests/test_usefulness_doctor.py`
- CLI smoke in either that file or `tests/test_usefulness.py`

Coverage:

- silent and healthy consumers from synthetic consult logs
- unattributed consult rows
- consults with low/absent grounding
- sqlite temp store with `memory_health.support_count`
- support-count starvation
- low-confidence or invalidated grounded memory
- missing logs and missing DB degrade cleanly
- `--json` preserves stable top-level keys

All tests must use isolated `Config` or `CliRunner` env values:

- `MEMO_NONINTERACTIVE=1`
- `MEMO_DATA_DIR`
- `MEMO_STATE_DIR`

No test may touch the real vault or default state dir.

## Verification

Focused checks:

```bash
uv run --no-sync pytest tests/test_usefulness_doctor.py tests/test_usefulness.py tests/test_dashboard.py -v
uv run --no-sync ruff check src/memo/usefulness_doctor.py src/memo/cli_usefulness.py tests/test_usefulness_doctor.py
uv run --no-sync mypy src/memo
```

If implementation only reads retrieval logs and store signals, `memo eval recall` is not required. If a later revision touches ranking or recall-hook behavior, run:

```bash
uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force
```

## Success Criteria

- A user can run one command and see whether memo is consulted, attributed, grounded, and trustworthy.
- The command identifies silent clients and missing `source` attribution.
- The command identifies sparse lifecycle signals such as nearly empty `support_count`.
- The command identifies grounded memories that are invalidated, superseded, or low confidence.
- The first version is read-only and cannot make recall slower.
