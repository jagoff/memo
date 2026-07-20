# memo — measured benchmark

Every number on this page comes from real command output — no extrapolation, no
synthetic multipliers. Anything memo cannot yet measure is listed as **not yet
measured** in [Limitations](#limitations). Where a metric is an estimate rather
than a direct measurement (the tokens-saved ledger), the estimation constants
are disclosed inline.

**Measured 2026-07-20** unless noted otherwise, with `memo 3.7.0`, on the
author's live corpus:

| Environment | Value |
|---|---|
| Corpus | 4,933 memories (single-user, machine-local, grown from real daily agent use) |
| Embedder | `Qwen3-Embedding-4B-4bit-DWQ` (2560-dim, `quality` profile) |
| LLM | `Qwen3-30B-A3B-Instruct-2507-4bit-DWQ` |
| Hardware | Apple Silicon Mac (MLX backend) |

This is a **live, single-user corpus** — the numbers describe how memo performs
on a real installation, not a standardized multi-repo benchmark. Run the
[Reproduce](#reproduce) commands to get the same report over *your* corpus.

---

## Retrieval quality — curated regression set

`memo eval recall` runs a committed label set
([`eval/regression_labels.json`](../eval/regression_labels.json), schema
`memo.eval_recall.labels.v1`) against the live index. The set has grown one
incident at a time: **37 prompts** — 34 answerable, 6 of which pin exact memory
ids that must surface, plus noise labels (tags and path fragments that must
**not** crowd the top-5). Retrieval-only, no LLM in the loop.

Output of `memo eval recall --labels eval/regression_labels.json --k 5 --force`
(4 configs × 37 prompts):

| Config | precision@5 | noise@5 | recall@5 | nDCG@5 | MRR | p50 latency |
|---|--:|--:|--:|--:|--:|--:|
| A — vec, min_sim 0.60, keep archived | 0.794 | 0.000 | 0.333 | 0.250 | 0.222 | 58.6 ms |
| B — vec, min_sim 0.72, exclude archived | 0.747 | 0.000 | 0.333 | 0.250 | 0.222 | 59.2 ms |
| C — hybrid, min_sim 0.40, exclude archived | 0.794 | 0.000 | 0.500 | 0.314 | 0.256 | 80.7 ms |
| D — hybrid, min_sim 0.40, + session context | 0.829 | 0.000 | 0.500 | 0.314 | 0.256 | 178.7 ms |

How to read this:

- **precision@5** — fraction of top-5 slots filled by labeled-relevant memories,
  over the 34 answerable prompts.
- **noise@5** — fraction of top-5 slots occupied by labeled noise (garbled OCR
  chunks, archived/stale notes), over all 37 prompts. Zero across every config.
- **recall@5 / nDCG@5 / MRR** — computed only over the 6 id-pinned prompts, so
  they are small-sample and structurally capped: a prompt with a single pinned
  answer can fill at most 1 of its 5 slots.

memo's development rule is that every failed real-world search adds a labeled
prompt to this set, and every retrieval change must hold precision/noise across
**all** of them — `memo eval recall --gate` exits non-zero on regression and is
wireable into a pre-commit hook.

## Public benchmark — LongMemEval (retrieval-only)

Measured **2026-07-10** with `memo eval bench` on a **stratified 60-question
subset** of `longmemeval_oracle` (10–20 per question type), ingested into an
isolated store — not the live corpus. Same 4B embedder, reranker on, k=5,
retrieval-only. Full write-up, lever sweeps, and measured-negative results in
[docs/eval/capability-baseline-and-levers.md](eval/capability-baseline-and-levers.md).

| Bucket | n | recall@5 | nDCG@5 | MRR |
|---|--:|--:|--:|--:|
| preference_understanding | 10 | 0.867 | 0.711 | 0.737 |
| single_session_grounding | 20 | 0.825 | 0.668 | 0.633 |
| temporal_state_tracking | 10 | 0.750 | 0.736 | 0.920 |
| knowledge_update_conflict | 10 | 0.717 | 0.552 | 0.573 |
| multi_session_synthesis | 10 | 0.493 | 0.450 | 0.617 |
| **weighted micro** | 60 | **0.746** | | |

The weak buckets are documented, not hidden: `multi_session_synthesis` (a
single query vector misses half the cross-session evidence) and
`knowledge_update_conflict` (the right memory is retrieved but sometimes ranked
below the stale one). The linked note also records levers that were measured
and **rejected** (MMR diversity, recency decay, contradiction-penalty-on-raw-turns)
because the numbers got worse.

## Live recall telemetry

From `memo stats` and `memo usefulness` over the same live corpus (last 7 days
of production use, 2026-07-20):

| Metric | Value |
|---|---|
| Recall hooks fired (7 days) | 1,226 |
| Hit rate | 99% |
| Strong hits (score > 0.7) | 97% |
| Latency, warm daemon (n=129) | p50 616 ms · p95 7,610 ms |
| Latency, cold subprocess fallback (n=31) | p50 8,935 ms |
| Consults sampled, by consumer | claude-code 95 · synapse 58 · codex 32 · claude 1 (hit rate 97–100% each) |
| grounded_rate | 0.371 — 157/423 surfaced memories actually used in the answer (outcome-based, not just shown) |
| reask_avoided | 19/32 grounded recalls the user did not have to ask again |

## Token economy

Two different instruments, reported separately because they say different
things:

**1. The usage ledger** (`memo tokens`) counts real events — grounded memory
uses and cross-agent consults — and converts them to tokens with disclosed
constants (350 tokens per grounded use, 200 per consult;
`MEMO_ROI_TOKENS_PER_GROUNDED` / `MEMO_ROI_TOKENS_PER_CONSULT`). The events are
measured; the per-event token values are configured estimates.

| Period | Tokens saved (estimated) | Events |
|---|--:|---|
| All-time | 1,209,450 | 3,259 grounded uses + 344 consults |
| 2026-07 (month) | 885,700 | 2,334 grounded + 344 consults |

By agent, all-time: claude-code 1.14M · synapse 36.8k · codex 29.4k · others < 2k.

**2. The direct transcript measurement** (same `memo tokens` output, 54
measured sessions): grounded sessions spent **4,825** tool-loop tokens/turn vs
**4,779** for ungrounded sessions — a delta of **−46 tokens/turn**, i.e. **no
measured tool-loop saving** in this sample. The honest summary: the ledger
estimates savings from answer-grounding events; the transcript proxy does not
(yet) show a reduction in tool-loop spend.

**3. Per-lever eval** (`memo eval tokens` — renders recall output OFF vs ON
under each token-economy lever; a lever passes iff it cuts ≥5% tokens without
dropping quality):

| Lever | Tokens saved | Quality delta | Verdict |
|---|--:|--:|---|
| crusher_L1 (capture crusher) | +44.4% | 0.00 | PASS |
| recall_format_compact | +0.0% | 0.00 | FAIL |
| verbosity_steer_L2 | −12.1% | 0.00 | FAIL |

Failed levers are reported as failed — they stay default-off.

## Reproduce

All commands are retrieval-only and read-only against your index (the eval
`--force` re-evaluates; it never mutates the corpus). Run them sequentially on
your own install:

```bash
memo --version
memo stats                                                      # corpus size + live recall telemetry
memo tokens                                                     # tokens-saved ledger (add --json for raw data)
memo usefulness                                                 # per-agent consult attribution + grounded rate
memo eval recall --labels eval/regression_labels.json --k 5 --force
memo eval tokens --labels eval/regression_labels.json --corpus eval/token_corpus.json --k 5
memo eval bench run --dataset longmemeval_oracle --retrieval-only --k 5   # public benchmark, isolated store (downloads data)
memo eval bench report
```

One caveat on the bench: the oracle question file is grouped by type, so
`--max-samples N` alone gives a monochromatic subset — the 2026-07-10 numbers
above used a stratified 60-question subset fed via `--file` (details in the
[capability note](eval/capability-baseline-and-levers.md)).

The committed label set targets the author's corpus; on your install,
`memo eval recall` builds meaning from your own labels — seed them from real
usage with the dream tuner's harvest, or start your own
`regression_labels.json` from the first search that disappoints you.

## Limitations

Read these before quoting any number above:

- **Single-user, live corpus.** This is one real installation (4,933 memories),
  not a standardized multi-repo benchmark. Results are corpus-specific and not
  directly comparable across installs — that's why the Reproduce section
  exists.
- **Retrieval only.** Precision/noise/recall measure whether the right memory
  surfaces, not whether a downstream answer synthesized from it is correct.
  Synthesis quality is evaluated elsewhere (in the synapse layer) and is out of
  scope here.
- **precision@5 is structurally capped for single-answer prompts.** A prompt
  whose label pins exactly one expected memory can fill at most 1 of 5 slots —
  a 0.2 ceiling for that prompt. The 0.75–0.83 aggregate is possible because
  most prompts are term-labeled (multiple memories can be relevant); the
  id-pinned subset is what caps recall@5/nDCG/MRR at their lower values.
- **Abstention is not yet measured.** The recall block is injected labeled as
  authoritative, and nothing here measures whether memo correctly stays silent
  when there is no evidence (refuse-when-empty). The abstention metric is
  wired (`longmemeval_oracle` ships 30 `*_abs` questions) but the end-to-end
  QA run is blocked on local GPU memory — see
  [the capability note](eval/capability-baseline-and-levers.md#abstention).
- **Tokens saved is an estimated ledger, not billed tokens.** Events are real;
  the 350/200 tokens-per-event constants are estimates. The direct transcript
  proxy currently shows no tool-loop saving (−46 tok/turn) — reported above,
  not hidden.
- **LongMemEval samples are small** (10–20 questions per bucket, oracle
  variant, isolated store) — directional, not leaderboard-grade.
- **Latency tails.** The warm-daemon p50 (616 ms) is the normal path; the p95
  (7.6 s) and the cold-subprocess fallback reflect cold model loads and
  contention, and are part of why the recall hook keeps a hard time budget.
