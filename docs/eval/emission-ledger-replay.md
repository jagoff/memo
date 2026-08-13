# Emission ledger — replay harness and the promotion decision

Companion to `docs/SPECS/2026-08-10-emission-ledger-design.md` (design +
success criteria) and `scripts/eval_emission_ledger.py` (the harness this
document explains). Read this before trusting — or distrusting — a run of
that script.

## What the harness does

`scripts/eval_emission_ledger.py` replays a real Claude Code transcript's
`memo_search` / `memo_ask` / `memo_evidence_pack` calls twice against the
**same live corpus** — once with `MEMO_EMITTED_LEDGER=0`, once with `=1` —
and diffs the token cost of what each participating tool call actually
returned. It does not touch a test fixture or a synthetic corpus: it points
at `~/.claude/projects/-Users-fer/<session>.jsonl` and drives the real
`Memory` facade against whatever this machine's real index holds right now.

```bash
uv run --no-sync python scripts/eval_emission_ledger.py \
  ~/.claude/projects/-Users-fer/<session>.jsonl
```

Because the replay runs against the live corpus rather than a frozen
snapshot, re-running it later against the same transcript can produce
slightly different numbers if the corpus has changed underneath it
(memories added, updated, or archived since the transcript was recorded).
That is expected and is the same caveat that applies to `memo eval recall`.

### Tool scope: three, not five

Per the Task 5 finding already recorded in the design spec, `memo_context`
and `memo_unified_briefing` never got wired to the ledger at all: neither
has a hit list this mechanism can suppress (`memo_context`'s structured
`hits` carry no body text, and the body text it does return lives inside a
packed prose string; `memo_unified_briefing` returns one compacted markdown
blob with no per-hit list). The harness's participating-tool set —
`memo_search`, `memo_ask`, `memo_evidence_pack` — matches
`MEMO_EMITTED_LEDGER_TOOLS`'s actual default exactly. This lowers the
achievable ceiling versus a hypothetical five-tool version: a transcript
whose repetition lives entirely in `memo_context`/`memo_unified_briefing`
calls contributes nothing to the measured saving, correctly, because the
feature itself does nothing for those calls either.

## Criterion 1 — the denominator, and why it is not whole-session tokens

The design spec's own words: *"The denominator is the total
`mcp_budget.est_tokens` of everything memo put into that session's window
with `MEMO_EMITTED_LEDGER=0` — recall-hook injections plus participating
tool results. Not whole-session tokens; memo is not responsible for those."*

The harness computes that sum from two sources:

1. **Recall-hook injections**, read directly off the transcript's own
   recorded hook output (`attachment.type == "hook_additional_context"`
   entries), not re-simulated. This is deliberate, not a shortcut: the
   recall hook (`cli_recall_hook.py`) only *writes* to the ledger — after
   rendering, it records what it injected so a later MCP call can digest it
   — it never *reads* the ledger to shrink its own output. So the hook's
   token cost is bit-for-bit identical whether `MEMO_EMITTED_LEDGER` is 0 or
   1, and the number already sitting in the transcript **is** that cost;
   re-running the hook against a corpus that has since drifted would only
   introduce noise a direct read avoids.
2. **Participating tool results**, replayed for real: each `memo_search` /
   `memo_ask` / `memo_evidence_pack` call in the transcript, reissued
   against the live `Memory` facade with the tool wrapper's own defaults and
   post-processing (body truncation, `apply_ledger`) reproduced inline, then
   sized with `mcp_budget.est_tokens` over the same kind of payload the tool
   would actually return. This mirrors the "whole serialized row" basis
   Task 8 established for `apply_ledger`'s own counters (see below), not
   just the body/snippet field alone.

`reduction = (baseline_total - treated_total) / baseline_total`, where
`baseline_total = hook_tokens + tool_tokens_off` and `treated_total =
hook_tokens + tool_tokens_on`. Because `hook_tokens` is the same constant on
both sides, the whole saving comes from the tool side — which means a
session that never repeats an MCP tool call, or relies purely on the recall
hook, will correctly show 0% by construction. That is not a harness bug; it
is the feature doing nothing because there was nothing to suppress.

### Measured result

Replayed against a real session
(`e00b57f5-8745-4462-a8dd-fbb60a6616b9.jsonl`, a manual QA sweep of memo's
MCP surface that happens to call `memo_search` on the identical query
`"memo release flow"` four times across the session — exactly the pattern
this feature targets). The harness runs against the **live** corpus, and
this machine runs it while other real sessions (recall daemon, chat
service, other Claude Code sessions) keep writing to that same corpus, so
four consecutive runs of the exact same transcript were kept as a
reproducibility check rather than reporting a single cherry-picked number:

| Run | Conditions | Reduction | Digests served |
| --- | --- | --- | --- |
| 1 | Clean, quiet system, first invocation | **31.4%** | 8 |
| 2 | Heavy resource contention (see below) | **21.8%** | 14* |
| 3 | Discarded — see below | n/a | n/a |
| 4 | Clean, quiet system, after the fix below | **36.6%** | 9 |

\* Run 2's `digests served` / `net_saved_est` are also affected by the
counters bug described below; its `reduction` figure is not, and is the
number in the table.

```
# Run 4 (the cleanest run, after the counters fix):
participating calls:    7  (memo_ask, memo_search)
recall-hook tokens:     661   (identical off/on)
tool tokens, flag off:  9561
tool tokens, flag on:   5818
TOTAL, flag off:        10222
TOTAL, flag on:         6479
reduction:              36.6%   (criterion 1: >= 25%)
digests served:         9
net_saved_est:          3543 tokens (this session's counters)
```

**Two things surfaced while producing these numbers, both real, neither
hidden:**

1. **A counters-accumulation bug in this harness, found and fixed.**
   `emitted_ledger.reset()` deliberately clears only the `.jsonl` ledger
   entries, never the `.counters.json` sidecar — correct for a real
   session (counters must survive a mid-session PreCompact reset), wrong
   for this harness, which reuses the same `eval-off` / `eval-on` session
   ids across repeated invocations. Without also clearing the counters
   file, `digests_served` / `net_saved_est` silently accumulated across
   runs (8 → 11 → 14 in the raw counter, not the true per-run count). Fixed
   by having the harness clear both files at the start of each pass (see
   `_reset_session` in the script) rather than relying on the production
   `reset()` semantics, which were never designed for this. This bug only
   ever affected the `digests served` / `net_saved_est` *display* lines —
   the `reduction` percentage is computed independently, from the
   `.jsonl`-backed partition each pass actually returns, and was never
   contaminated by it.
2. **Run 3 was invalidated by operator error, not a harness bug.** While a
   background run was still mid-flight, its `eval-on.jsonl` ledger file was
   deleted externally (cleaning up what looked like stale state from an
   earlier attempt, without checking whether the process using it was still
   running). That deletion landed between two of the run's own tool calls,
   erasing the entries the second call needed to digest against the first
   — deflating that run's `reduction` artificially. Its output is not
   trustworthy and is excluded from the table above.

**Why run 2 is lower.** It executed while two full `pytest tests/` suites
were running concurrently on the same machine (a self-inflicted duplicate —
see the task report), competing for CPU and MLX/GPU resources for several
minutes. memo's cross-encoder reranker has its own wall-clock budget and
falls back to raw RRF order when it elapses (`flags_search.py`:
"under GPU contention that loop is what turns a 6s search into minutes");
under contention, a slower or abandoned rerank can change which hits land
inside a `limit`-bounded result, which changes what there is to digest. The
run also took long enough (~7 minutes, vs ~1-2 minutes uncontended) to
widen the window for the live corpus to genuinely change underneath it.
Once the duplicate process was killed and the counters bug fixed, run 4 —
under normal, uncontended conditions — landed back above 31%.

**Criterion 1: PASSES, with margin, under normal operating conditions**
(two independent clean runs: 31.4% and 36.6%, both comfortably over the 25%
floor). It is sensitive to system load in a way worth naming plainly: a
severely contended machine can pull the measured saving below the
threshold on a run-by-run basis, though the mechanism (reranker fallback
changing result sets under GPU contention, not the ledger itself) is a
property of memo's search path generally, not something this feature
introduces. No harness parameter was tuned toward a favorable number — the
counters fix was a correctness fix (found via a suspicious `digests_served`
reading, not by chasing this number), and every run's output above is
reported, not just the best one.

The script itself never prints a bare `PROMOTE`, even when criterion 1
passes: promotion needs criterion 2 too, and criterion 2 is structurally
outside what a replay can supply (see below) — see the script's own
`VERDICT: KEEP AT MEMO_EMITTED_LEDGER=0` line for why a criterion-1 pass
alone is not read as "ship it."

`net_saved_est` (from `emitted_ledger.stats`, surfaced via
`memo_cache_stats`'s `emit_ledger` block) is a useful cross-check but not
the primary number: Task 8's review found it can read negative on a corpus
of very short bodies purely from the digest stub's own fixed overhead
(`hint` string + `{id, title, ref}` JSON), even when the feature is a clear
win on realistic bodies — see `task-8-report.md`'s "Concerns for Task 10"
section. `net_saved_est` in run 4 (+3543) is positive and consistent with
that run's 36.6% reduction.

## Criterion 2 — `memo_get_after_digest` < 20% of `digests_served`

**This cannot be measured by a replay, and the harness does not pretend
otherwise.** The recovery counter only increments when a model, holding a
digest pointer (`{id, title, ref}`), *decides* to call `memo_get(id)` to
recover the full body. A transcript replay has no model in the loop making
that decision — it deterministically reissues the recorded tool calls, and
none of those calls is a `memo_get` triggered by a digest the replay itself
produced (the replay's digests didn't exist yet when the original session
ran, so the original session couldn't have reacted to them). Any recovery
rate a replay reports is an artifact of the harness's own determinism, not
evidence about the design.

The run above shows `memo_get after digest: 0` for exactly that reason —
not because recovery would actually be zero in practice, but because
nothing in the replay could ever produce a nonzero value here. Treat this
line as structurally unmeasurable, not as a pass.

**What would actually measure it:** a live dogfooding period with
`MEMO_EMITTED_LEDGER=1` set for real sessions, then reading
`memo_cache_stats`'s `emit_ledger.memo_get_after_digest` /
`emit_ledger.digests_served` ratio after enough sessions have accumulated
digests and (potentially) recovered from them. That is the only path to a
real number for criterion 2 — this repo does not yet have that data, and
this task does not fabricate it.

## Criterion 3 — recall-hook p95 latency delta < 20ms

Already measured in Task 6, not re-measured here (the dispatch for this
task explicitly says not to unless there's reason to distrust it — there
isn't). Two paths were measured:

- **Warm-daemon path** (`com.memo.recall-daemon`, what this machine actually
  serves recall-hook traffic on): OFF p50=397.0ms p95=515.9ms; ON
  p50=396.4ms p95=413.3ms → **delta p50 = -0.56ms, delta p95 = -102.64ms**.
  Every percentile is *negative* — the ledger write is noise against a
  ~400ms daemon round trip.
- **Subprocess-fallback path** (no warm daemon; cold CLI invocation),
  N=60/arm with real hits confirmed: OFF p50=175.17ms p95=178.35ms; ON
  p50=181.12ms p95=191.18ms → **delta p50 ~5.95ms, delta p95 ~12.83ms**,
  comfortably under the 20ms ceiling.

**Criterion 3: PASSES on both paths** (source: `task-6-report.md`).

## Criterion 4 — the recall eval gate is not regressed

The pre-push gate (`memo eval recall --labels eval/regression_labels.json
--gate`) measures **corpus drift, not code**: it compares (current code,
current corpus) against a saved baseline captured from (old code, old
corpus), so a real-corpus change between baseline and gate run can fail the
gate for reasons that have nothing to do with any particular diff. Two
checks were run to isolate that:

1. **Same command, two isolated checkouts.** `git worktree add --detach
   /tmp/.../wt-master origin/master` (a detached checkout, never a
   `git checkout` in the shared tree), then `memo eval recall --labels
   eval/regression_labels.json --k 5 --gate` in both the master worktree and
   this branch. **Both fail identically**:

   ```
   ✗ recall gate: FAIL — gated config 'H synth/0.05' not evaluated this run
   (ran: A vec/0.60/keep, B vec/0.72/excl, C hyb/0.40/excl, D hyb/0.40/ctx);
   the baseline pins a config this run did not measure — select it, or
   refresh the baseline with --update-baseline
   ```

   This is not even corpus drift — it's a saved baseline pinned to a config
   (`H synth/0.05`) that the current default config selection no longer
   runs. Since master itself fails the exact same way with none of this
   branch's code, the failure predates and is independent of this diff.

2. **The stronger check: `--against origin/master`.** This flag runs the
   SAME live corpus through both this branch's code and master's code in
   one uncached pass, so the corpus term cancels and any remaining delta is
   attributable to the diff — the intended replacement for the
   confounded-by-corpus-drift saved-baseline gate:

   ```
   ✓ vs origin/master: PASS — prec@k 0.724 vs ref 0.724, noise@k 0.000 vs
   ref 0.000 (same corpus, both runs uncached)
   ```

   Precision and noise are bit-identical between this branch and master on
   the same 43-prompt labeled set.

**Criterion 4: PASSES.** The pre-existing `--gate` failure is a stale
baseline/config mismatch reproduced identically on stock `origin/master`;
the corpus-cancelling `--against` comparison shows zero recall delta from
this diff.

## Verdict

| # | Criterion | Result | Measured |
| - | --- | --- | --- |
| 1 | >=25% fewer emitted tokens on a replayed transcript | **PASS under normal load** | 31.4%, 36.6% clean; 21.8% under self-inflicted contention (see above) |
| 2 | `memo_get_after_digest` < 20% of `digests_served` | **UNMEASURABLE by replay** | needs live dogfooding |
| 3 | Recall-hook p95 delta < 20ms | **PASS** | -102.64ms (warm daemon), +12.83ms (subprocess fallback) |
| 4 | Recall eval gate not regressed | **PASS** | 0.724 vs 0.724 prec@k, 0.000 vs 0.000 noise@k (`--against origin/master`) |

Criterion 1 passes with margin under normal conditions (measured on two
independent clean runs, not assumed) but is measurably sensitive to system
load, as shown above. Criterion 3 is a clean pass. Criterion 4 is a clean
pass once corpus drift is isolated out. Criterion 2
is the honest gap: nothing in this repo can produce that number without a
live session where a model actually decides whether to call `memo_get` on a
digest. Per the design spec ("If 1 or 2 fail, the feature stays at
`MEMO_EMITTED_LEDGER=0`... that is the decision, not a deferral"), an
unmeasured criterion is not the same as a failed one — but it is also not a
pass. The recorded decision (see `docs/SPECS/2026-08-10-emission-ledger-design.md`'s
"Measured result" section) reflects that: ship the code as already planned
(`MEMO_EMITTED_LEDGER` stays `0` by default), and let a live dogfooding
period supply the number criterion 2 needs before anyone flips the default.
