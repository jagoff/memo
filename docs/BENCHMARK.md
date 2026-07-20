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
on a real installation, not a standardized multi-repo benchmark. It also means
every number is a **point-in-time snapshot**: a same-day re-run drifts most
counts by ~1% and some derived rates by more (see the
[Challenger review](#challenger-review-2026-07-20), C10). Run the
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
  over the 34 answerable prompts (empty slots count as misses). **Read it
  narrowly**: for the 28 term-labeled prompts, "relevant" means the hit mentions
  any of a 30-term global vocabulary (memo/recall/hook/daemon/embedder/…) in its
  title, tags, path, or first 200 body chars — and ~83% of this corpus already
  passes that test (random 500-file sample), so on this corpus precision@5
  mostly measures *noise-free slot-fill above the similarity floor*, not topical
  precision. That is also why the stricter-floor config B scores lower — more
  empty slots, not worse ranking (its relevant-per-filled-slot ratio, 0.80, is
  identical to A's).
- **noise@5** — fraction of top-5 slots occupied by labeled noise (garbled OCR
  chunks, archived/stale notes), over all 37 prompts. Only config A's zero is a
  measurement: the "exclude archived" post-filter in configs B/C/D drops hits
  matching the same noise labels that noise@5 counts, so their zeros hold by
  construction.
- **recall@5 / nDCG@5 / MRR** — computed only over the 6 id-pinned prompts
  (covering 5 distinct target memories — two prompts are ES/EN paraphrases of
  one target), so they are small-sample. They are **not** structurally capped —
  each can reach 1.0 even on a single-answer prompt. The low values are real
  misses: the pinned memory surfaces in top-5 for 2 of 6 prompts under the vec
  configs and 3 of 6 under hybrid.

Two scope notes. The A–D grid is `memo eval recall`'s fixed default grid
(`default_configs()`), not a per-run selection — but none of the four is this
install's production operating point (the live recall hook resolves
`min_sim 0.8835` from the tuned overlay, mode `vec`), and non-pinned ranking
knobs (`mmr_lambda 0.7`, `project_boost 0.2`) inherit the live overlay during
the eval. Config D's gain over C also partly reflects that its fixed
session-context string shares exact vocabulary (`lambda`, `deploy`,
`dev-publiccloudinfrastructure`) with the relevance labels.

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

From `memo stats` and `memo usefulness` over the same live corpus (2026-07-20).
**Windows vary by instrument** and are labeled per row: the hook-quality
numbers come from the recall-hook log tail (capped at 2,000 rows — here
spanning 2026-06-19 → 2026-07-20, roughly a month), the consult and grounding
rows from the last ≤500 log entries of their logs, and only the latency block
is a true 7-day window.

| Metric | Value |
|---|---|
| Recall hooks fired (log tail, ~31 days) | 1,226 |
| Hit rate | 99% |
| Strong hits (score > 0.7) | 97% |
| Latency, warm daemon, last 7 days (n=129) | p50 616 ms · p95 7,610 ms |
| Latency, cold subprocess fallback, last 7 days (n=31) | p50 8,935 ms |
| Consults, by consumer (last ≤500 consult-log rows) | claude-code 95 · synapse 58 · codex 32 · claude 1 (hit rate 97–100% each) |
| grounded_rate (log-tail sample) | 0.371 — 157/423 surfaced memories actually used in the answer (outcome-based, not just shown) |
| referenced_rate (log-tail sample) | 0.01 — 13/1,251 surfaced memories later explicitly fetched; the stricter lower bound on "used" |
| reask_avoided (log-tail sample) | 19/32 grounded recalls the user did not have to ask again |

## Token economy

Two different instruments, reported separately because they say different
things:

**1. The usage ledger** (`memo tokens`) counts real events — grounded memory
uses and cross-agent consults — and converts them to tokens with disclosed
constants (350 tokens per grounded use, 200 per consult;
`MEMO_ROI_TOKENS_PER_GROUNDED` / `MEMO_ROI_TOKENS_PER_CONSULT`). The events are
measured; the per-event token values are configured estimates. Per-agent
consult counting only began 2026-07-08, so "all-time" and "2026-07" consults
are the same 344 — the consult instrument is ~12 days old.

| Period | Tokens saved (estimated) | Events |
|---|--:|---|
| All-time | 1,209,450 | 3,259 grounded uses + 344 consults |
| 2026-07 (month) | 885,700 | 2,334 grounded + 344 consults |

By agent, all-time: claude-code 1.14M · synapse 36.8k · codex 29.4k · others < 2k.

**2. The direct transcript measurement** (same `memo tokens` output, 54
measured sessions): at publication time grounded sessions spent **4,825**
tool-loop tokens/turn vs **4,779** for ungrounded — Δ −46, an apparent cost. A
same-day re-run of the same instrument over the same 54 sessions gave grounded
**4,921** vs ungrounded **5,167** — Δ +246, an apparent saving. The proxy is an
observational comparison (sessions that happened to ground vs not, re-rolled as
sessions grow) and swings by hundreds of tokens/turn within a day, so it
currently supports **no conclusion in either direction** about tool-loop
savings. The honest summary: the ledger estimates savings from answer-grounding
events; the transcript proxy is too noisy to confirm or deny them.

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

All commands are retrieval-only and never modify your memories. Two
side-effects to know about: every eval search logs access rows — the same
read-tracking any search performs, which feeds LFU/promotion signals — and the
eval caches its results under `state_dir/eval/` (`--force` re-evaluates instead
of reading that cache; it never mutates the corpus). Run them sequentially on
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
- **precision@5 is a weak label on this corpus.** Term-labeled relevance uses a
  30-term global vocabulary that ~83% of this corpus already matches, so the
  0.75–0.83 aggregate sits at the vocabulary's base rate and mostly tracks
  noise-free slot-fill (config B's lower score is empty slots from its stricter
  floor, not worse ranking). The id-pinned subset is the sharp measurement, and
  it shows real misses: the pinned memory surfaces in top-5 for only 2/6 (vec)
  to 3/6 (hybrid) prompts. Single-answer prompts do cap *per-prompt precision*
  at 0.2, but recall@5/nDCG/MRR carry no such cap — their low values are
  misses, not metric artifacts.
- **The label set is self-authored.** Prompts, relevance terms, and noise
  labels were written by the same developer whose corpus and search failures
  they encode; the relevance vocabulary is the corpus's own dominant
  vocabulary. There is no external annotator or inter-rater check.
- **Numbers are point-in-time.** A same-day re-run drifted every live-corpus
  count (4,933 → 4,944 memories; hook fires 1,226 → 1,228; grounded_rate
  0.371 → 0.365; config C precision 0.794 → 0.788; eval p50 latencies moved up
  to 2× with warm state). The pinned-subset ranked metrics and noise@5
  reproduced exactly. Quote numbers together with their date.
- **Abstention is not yet measured.** The recall block is injected labeled as
  authoritative, and nothing here measures whether memo correctly stays silent
  when there is no evidence (refuse-when-empty). The abstention metric is
  wired (`longmemeval_oracle` ships 30 `*_abs` questions) but the end-to-end
  QA run is blocked on local GPU memory — see
  [the capability note](eval/capability-baseline-and-levers.md#abstention).
- **Tokens saved is an estimated ledger, not billed tokens.** Events are real;
  the 350/200 tokens-per-event constants are estimates. The direct transcript
  proxy is sign-unstable (−46 → +246 tok/turn between two same-day runs) —
  inconclusive in either direction, reported above, not hidden.
- **LongMemEval samples are small** (10–20 questions per bucket, oracle
  variant, isolated store) — directional, not leaderboard-grade.
- **Latency tails.** The warm-daemon p50 (616 ms) is the normal path; the p95
  (7.6 s) and the cold-subprocess fallback reflect cold model loads and
  contention, and are part of why the recall hook keeps a hard time budget.

---

## Challenger review (2026-07-20)

The day this page was published, an **independent adversarial pass** re-ran the
light commands (`memo stats`, `memo tokens`, `memo usefulness`,
`memo eval recall`) against the same live install and audited every
computation claim against the source (`src/memo/eval_recall.py`,
`src/memo/token_meter.py`, `src/memo/token_ledger.py`,
`src/memo/dashboard_metrics.py`). Findings are published verbatim below, with
what was done about each: **corrected** (the text above was changed),
**defended** (the claim holds as written), or **accepted limitation** (added
to Limitations). Nothing was removed from the page in response to a finding.

**C1 · CRITICAL — precision@5's relevance predicate barely discriminates on
this corpus.** For the 28 term-labeled prompts, a hit counts "relevant" if its
title/tags/path/first-200-body-chars contain any of one global 30-term
vocabulary (`memo`, `recall`, `hook`, `daemon`, `synapse`, `embedding`, …).
Measured base rate: ~83% of a 500-file random corpus sample passes that test
(81.6% even excluding the path field). The published 0.747–0.829 sits at or
below that base rate; the relevant-per-filled-slot ratio is 0.79–0.83 in every
config; and the entire A→B drop (0.794→0.747) is explained by slot-fill
(0.994→0.929) at an identical relevant/filled ratio (0.799 vs 0.804). As
measured here, precision@5 ≈ "noise-free fill rate above the floor", not
topical precision. → **Corrected**: the "How to read this" bullet and the
Limitations bullet now say exactly this; the id-pinned subset is flagged as
the only sharp measurement.

**C2 · HIGH — noise@5 = 0.000 is guaranteed by construction for 3 of 4
configs.** The harness implements "exclude archived" as
`ranked = [h for h in ranked if not _is_noise(h, labels)]` — the *same*
predicate noise@5 counts — and this label set has no per-prompt `avoid_ids`,
so configs B/C/D cannot register noise at all. Only config A's zero is a
measurement. → **Corrected** in the noise@5 bullet.

**C3 · HIGH — "structurally capped" misattributed the low ranked metrics.**
The original text claimed recall@5/nDCG/MRR are capped because a single-answer
prompt "can fill at most 1 of its 5 slots". That caps *precision* (0.2/prompt),
not recall/nDCG/MRR — each of those reaches 1.0 on a single-answer prompt. The
low values are genuine misses (pinned memory absent from top-5 in 4/6 vec, 3/6
hybrid prompts). → **Corrected** in both the metric notes and Limitations.

**C4 · HIGH — the direct transcript delta flipped sign the same day.**
Published: Δ −46 tok/turn ("no measured saving"). Re-run ~3 h later, same
instrument, same 54 sessions: grounded 4,920.86 vs ungrounded 5,166.89 —
Δ +246 (an apparent *saving*). The proxy is an observational, confounded,
rolling comparison; single-day swings exceed the published effect by 5×.
→ **Corrected**: the section now reports both runs and draws no conclusion in
either direction.

**C5 · HIGH — "(7 days)" mislabeled the telemetry window.** Hook fires / hit
rate / strong hits come from `recall_health()`, which reads the recall-hook
log tail (cap 2,000 rows); the live log spans 2026-06-19 → 2026-07-20 (~31
days, 1,354 rows). The consult breakdown is the last ≤500 consult-log rows,
also not day-windowed. Only the latency block is a true 7-day window
(`summarize(days=7)`). → **Corrected**: the table now labels each row's real
window.

**C6 · MEDIUM — none of the published configs is the production operating
point.** The live recall hook resolves `MEMO_RECALL_MIN_SIM = 0.8835` from the
tuned overlay (mode `vec` by default) — stricter than any of A–D's pinned
floors (0.40–0.72). Non-pinned knobs (`mmr_lambda 0.7`, `project_boost 0.2`)
inherit the live overlay during the eval. → **Corrected** (scope note added).
Partially **defended**: A–D is the tool's fixed default grid — comparable
across installs and not cherry-picked per run; the worst config (B) is
published too.

**C7 · MEDIUM — "read-only" was overstated.** Every eval search goes through
`Memory.search`, which unconditionally records access rows
(`_record_access`, feeding LFU/promotion signals), and `--force` writes the
results cache under `state_dir/eval/`. Health metrics filter out eval-probe
sessions, but the access-table writes remain — 148 searches per run.
→ **Corrected** in Reproduce ("never modify your memories" + side-effects
disclosed).

**C8 · MEDIUM — three irreconcilable consult denominators.** The telemetry row
(claude-code 95 · synapse 58 · codex 32) is a ≤500-row log sample; the ledger's
"344 consults" is a monotonic all-time counter; the ledger's by-agent line
implies synapse 184 / codex 147 consults all-time. The page never explained
why 58 ≠ 184. Additionally, all-time consults equal 2026-07 consults because
per-agent consult attribution only began 2026-07-08. → **Corrected**: windows
labeled per row; instrument age disclosed in the ledger paragraph.

**C9 · MEDIUM — selective omission of an unflattering companion metric.** The
same `memo usefulness` output that supplied grounded_rate and reask_avoided
also reports `referenced_rate = 0.01` (13/1,251 surfaced memories later
explicitly fetched — the stricter lower bound on "used"), which the page
omitted. → **Corrected**: referenced_rate added to the telemetry table. The
omission of the eval table's `assoc@5` column (0.0) is **defended**: this
label set contains no `expect_associative_ids`, so the column is vacuous here,
not a hidden failure.

**C10 · LOW — exact numbers do not reproduce within hours on a live corpus.**
Same-day re-run: corpus 4,944 vs 4,933; hook fires 1,228 vs 1,226; consult
sample 188 vs 186; grounded_rate 0.365 (156/427) vs 0.371 (157/423);
reask_avoided 22/35 vs 19/32; config C precision 0.788 vs 0.794 (one slot);
eval p50 latencies A 65.0 vs 58.6 ms and D 89.8 vs 178.7 ms (warm-state
dependent). Configs A/B/D precision, all noise@5, and all pinned-subset
ranked metrics reproduced exactly. → **Accepted limitation** ("Numbers are
point-in-time" added to Limitations; snapshot note added to the intro).

**C11 · LOW — config D is partially seeded with label vocabulary.** Its fixed
session-context string ("deploy de lambdas en el entorno
dev-PublicCloudInfrastructure") shares three exact terms with
`relevant_terms` and matches the topic of one id-pinned prompt, so part of
D's precision gain over C is the query being seeded with the vocabulary the
metric rewards. → **Corrected** (disclosed in the scope note).

**C12 · INFO — what was re-verified and what was not.** Re-verified: ledger
math (3,259×350 + 344×200 = 1,209,450 ✓; month 885,700 ✓; by-agent totals ✓),
ROI constants are the shipped defaults (350/200 in `flags_misc.py` ✓), the
token-lever names and the ≥5% gate exist in code (`MIN_SAVING_FRAC = 0.05` ✓),
the LongMemEval table matches
[the capability note](eval/capability-baseline-and-levers.md) verbatim and its
weighted micro recomputes to 0.746 ✓, the label set contains exactly 37/34/6
prompts as claimed ✓, and the label file grew across 8 commits
(2026-06-08 → 2026-07-11) ✓. Not re-run in this pass: `memo eval tokens` (the
+44.4% / +0.0% / −12.1% lever figures) and `memo eval bench` — both stand as
published single-run numbers.

The framing claims that survived attack unchanged: every number on the page
does come from real command output; negative results (failed levers, weak
LongMemEval buckets, the noisy transcript proxy) were already published before
this review; and the noise@5 = 0 result for config A — the only config where
it is falsifiable — is real.
