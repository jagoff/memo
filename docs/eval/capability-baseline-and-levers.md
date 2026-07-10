# Capability baseline + retrieval-lever measurements (LongMemEval oracle)

Measured 2026-07-10 with `memo eval bench` (the capability-taxonomy rollup added
in v2.12.19). Embedder `Qwen3-Embedding-4B-4bit-DWQ` (2560-dim), reranker on,
retrieval-only, `k=5`. Corpus = a **stratified** 60-question subset of
`longmemeval_oracle` (10 per LongMemEval `question_type`) — the oracle set is
grouped by type (first 133 are all `temporal-reasoning`), so `--max-samples N`
alone is monochromatic; feed a stratified subset via `--file`.

This note is the recall-faithful evidence for the boost knobs that memo
previously could not measure (the long-standing "eval recall-faithful for
boosts" gap): the per-bucket bench now scores `mmr_lambda` / decay / etc.
directly.

## Per-bucket baseline (no boosts)

| bucket | n | recall@5 | ndcg@5 | mrr |
|---|--:|--:|--:|--:|
| preference_understanding | 10 | 0.867 | 0.711 | 0.737 |
| single_session_grounding | 20 | 0.825 | 0.668 | 0.633 |
| temporal_state_tracking | 10 | 0.750 | 0.736 | 0.920 |
| knowledge_update_conflict | 10 | 0.717 | 0.552 | 0.573 |
| multi_session_synthesis | 10 | 0.493 | 0.450 | 0.617 |
| **weighted micro** | 60 | **0.746** | | |

Weak buckets: **multi_session_synthesis** (recall 0.49 — half the cross-session
evidence isn't even in top-5) and **knowledge_update_conflict** (recall OK but
ndcg/mrr low — the right doc is retrieved yet ranked below the stale one).
`temporal_state_tracking` is NOT weak (best mrr 0.92).

## Lever sweep — what does NOT help (measured-negative)

### MMR diversity (`MEMO_RECALL_MMR_LAMBDA`)

| bucket | baseline | λ=0.3 | λ=0.5 | λ=0.7 |
|---|--:|--:|--:|--:|
| knowledge_update_conflict | 0.717 | 0.367 | 0.417 | 0.567 |
| multi_session_synthesis | 0.493 | 0.212 | 0.312 | 0.398 |
| preference_understanding | 0.867 | 0.550 | 0.717 | 0.800 |
| single_session_grounding | 0.825 | 0.725 | 0.725 | 0.825 |
| temporal_state_tracking | 0.750 | 0.500 | 0.550 | 0.700 |

MMR recovers monotonically toward the baseline as λ→1 (pure relevance) but
**never exceeds it** on any bucket. Diversity is the wrong objective for
single-evidence retrieval: LongMemEval wants THE specific supporting turn, not a
varied set. Keep `MEMO_RECALL_MMR_LAMBDA=0` (the default) for recall of factual
evidence. MMR remains defensible only where the user wants breadth, not a single
answer.

### Recency decay (`MEMO_SEARCH_DECAY_HALFLIFE` / a hypothetical recency boost)

Not applicable to this benchmark and **not a real lever for the
knowledge-update weakness**. `_apply_decay` computes `days_since_updated` from
*now*; LongMemEval conversations are dated ~2023, so every memory is many
half-lives old and the decay factor is near-uniform — it cannot separate a
"fixed the fence in March" fact from its "bought cows in April" update. A new
`recency_boost` knob was considered and rejected: it would duplicate the
existing decay and hit the same old-date wall.

## The real fixes (features, not knobs)

- **multi_session_synthesis (recall gap):** the missing evidence is not in the
  candidate pool a single query vector surfaces. The fix is a retrieval-side
  feature — query decomposition or multi-vector retrieval that issues sub-queries
  per referenced session and unions the pools — measured on this same per-bucket
  bench. Out of scope for a knob; needs its own design + gate.
- **knowledge_update_conflict (mis-ranking):** memo's actual conflict handling is
  the **contradiction penalty** (`MEMO_CONTRADICT_PENALTY_ENABLED`), which
  penalizes the older side of a *detected* contradiction/evolution pair. The
  isolated bench store never runs `memo contradict scan`, so the raw bucket
  understates production. **Implemented + measured** as `memo eval bench run
  --contradict-scan` (runs the scanner per store with the small `cfg.helper_model`
  — off the 30B OOM path — then enables the penalty during scoring):

  | knowledge_update (10 Q) | baseline | +contradict-scan |
  |---|--:|--:|
  | recall@5 | 0.717 | 0.383 |
  | ndcg@5 | 0.552 | 0.343 |
  | mrr | 0.573 | 0.495 |

  **Measured-NEGATIVE.** The scan detected 367 contradiction/evolution pairs
  across just 10 samples (835 examined). Root cause: the bench ingests every
  conversation *turn* as its own memory, and the classifier over-fires on
  ordinary dialogue evolution ("I prefer X" → later "now Y"), so the penalty
  demotes the gold turn along with the noise. The contradiction penalty is
  calibrated for **curated, deduplicated durable facts** (decisions/facts), not
  raw turn-granular dialogue — so it cannot be faithfully measured on this
  ingestion model. The `--contradict-scan` flag ships (default OFF, unit-tested)
  as the instrument; a faithful knowledge-update measurement needs either
  fact-level ingestion (consolidate turns into durable facts before scoring) or a
  much stricter classifier gate (contradiction-only, confidence ≥ 0.9). Deferred
  with this evidence.

## Abstention

The abstention metric (`abstention_summary`: correct-abstentions / hallucinations
/ abstention-accuracy / hallucination-rate) is wired and unit-tested. It needs a
QA run (generation + judge), not retrieval-only. `longmemeval_oracle` carries 30
`*_abs` variants (answer = "the information is not enough…"), so no giant
`longmemeval_s` download is required.

**Local GPU wall (measured 2026-07-10 on mac-black, 36GB).** Two attempts, both
failed:

1. Default judge (`live.llm_model` = the 30B) → **Metal OOM** hard exception
   (`Insufficient Memory … kIOGPUCommandBufferCallbackErrorOutOfMemory`): the 30B
   loaded twice (generation + judge) plus the 4B embedder plus the always-on
   recall daemon's resident 4B exceeded Metal memory.
2. Small judge (`MEMO_BENCH_JUDGE_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit`,
   30B stays the answerer) → ran further without a Metal exception but was
   **OS-killed** part-way (memory pressure).

Conclusion: the production 30B QA path is not runnable on this 36GB Mac under
load. Measure abstention on a higher-RAM machine (mac-work) or offload
generation+judging to an API. Reproduction:

```
MEMO_BENCH_JUDGE=mlx MEMO_BENCH_JUDGE_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit \
  memo eval bench run --dataset longmemeval_oracle --file <30 *_abs questions> --k 5 --json
```

or `MEMO_BENCH_JUDGE=api` with a small `MEMO_LLM_MODEL`. The metric itself is
unit-tested (`tests/test_eval_bench_taxonomy.py::…abstention…`); what is blocked
is only the live end-to-end number on this hardware.

## BEAM — why it is not wired as a bench dataset

Researched against the canonical benchmark (`github.com/mohammadtavakoli78/BEAM`,
HF `Mohammadta/BEAM`, arXiv 2510.27246). BEAM does **not** fit memo's
evidence-labeled retrieval harness:

1. **No gold-evidence / gold-session labels.** BEAM grades every answer by
   LLM-judge against a per-question `rubric`; there is no evidence-turn or
   gold-session id. memo's `score_retrieval` needs `evidence_turn_ids` to compute
   recall@k/ndcg — so BEAM can produce **no retrieval metrics at all**, only
   rubric-scored QA.
2. **Rubric-judged, not gold-string-judged.** memo's judge compares an answer to
   a gold string; BEAM needs per-category rubric judges (BLEU/ROUGE/semantic-F1/
   Kendall-tau + LLM rubric). Different judge contract.
3. **Parquet + Python-repr, not JSON.** Data ships as HF Parquet (no `.json`
   blob to fetch); `probing_questions` is a stringified Python dict
   (`ast.literal_eval`, not `json.loads`); `chat` is a flat message list whose
   session boundaries are derived (`question_type == "main_question"`).
4. **QA-only ⇒ same 30B OOM wall** as abstention on this hardware.

A faithful BEAM adapter is therefore a **separate QA-rubric evaluation mode**
(new parser + a rubric judge + a `datasets`/parquet dependency), not the small
JSON parser the LongMemEval/LoCoMo path uses. Category → bucket mapping for when
it is built: `information_extraction`→single_session_grounding;
`preference_following`→preference_understanding;
`multi_session_reasoning`,`summarization`→multi_session_synthesis;
`temporal_reasoning`,`event_ordering`→temporal_state_tracking;
`knowledge_update`,`contradiction_resolution`→knowledge_update_conflict;
`abstention`,`instruction_following`→abstention_constraint.
