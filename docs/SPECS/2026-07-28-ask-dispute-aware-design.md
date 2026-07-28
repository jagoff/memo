# Dispute-aware ask — design

**Date:** 2026-07-28
**Status:** approved (brainstorm), pending implementation plan
**Origin:** MCP gap analysis (public-product criterion). Verified gap: memo's
trust machinery (contradict_store, quality buckets, dossier, penalties) never
reaches the default `memo_ask` path or its grounding gate, so ask can ground an
answer in a fact that recall itself would flag as disputed — violating the
documented "respect freshness" contract.

## Prior art (acknowledged — this design wires it, it does not rebuild it)

- **Recall dossier:** `cli_recall_hook.py:736-748` builds `disputed_by` with one
  batched `contradict_store.pairs_for_ids(ids, "open") + pairs_for_ids(ids,
  "competing")`, fail-open to `{}`; rendered by `recall_logic.py:122-129`
  ("⚔ disputed by [id8]"). This is the exact lookup pattern reused here.
- **Context-pack branch of ask** already demotes/labels stale-or-conflicting
  hits via `quality.py` buckets (`context_pack.py:234-253`, `quality_bucket` on
  sources) — untouched by this design.
- **Ranking-side mechanisms** (`MEMO_CONTRADICT_PENALTY_ENABLED`,
  `MEMO_DECLARE_DISPUTES`, `MEMO_HIT_DOSSIER`, `MEMO_QUALITY_RERANK`) and the
  Trust & Belief-Revision program
  (`docs/superpowers/plans/2026-07-10-trust-belief-revision-p0-p1.md`) stay
  as-is; graduating those dark flags is the dream tuner's job, not this spec's.
- **Grounding judge** (`grounding_judge.score_grounding`, gate
  `MEMO_GROUNDING_ASK_MIN` default 0.0) is unchanged and orthogonal: it scores
  answer↔source entailment. The dispute gate below is **deterministic** — set
  logic over citations — so it costs zero extra LLM calls and cannot fail open
  the way the judge can.

## Verified blind spots this design closes

1. Default (snippet-branch) ask builds no dispute information
   (`ask_ops.py:536-585`).
2. The grounding/abstention gate (`ask_ops.py:951-962`, stream `:1063-1075`)
   never consults `contradict_store`.
3. No abstention distinguishes "nothing found" from "found, but contested".
4. Verbatim short-circuit (`ask_ops.py:919-929`, impl `:818-849`) bypasses the
   LLM and every gate — a disputed top hit is dumped verbatim, unannotated.

## Design

One flag: **`MEMO_ASK_DISPUTES`** (bool, **default ON**, `opt_out=True`,
registered in `flags_search.py` next to `MEMO_ASK_MULTI_ROUND`, group
`search`). Default-on is a deliberate product decision (annotation is additive;
abstention below is narrow and high-precision). Because it is default-on it is
not a dark flag — no `dream_flags.GATES` entry required. Opt-out restores
today's behavior exactly.

### 1. Dispute lookup (in `_build_ask_context`)

After the sensitive-memory filter (`ask_ops.py:534`), when final memory hits
are settled: one batched `pairs_for_ids(ids, "open") + pairs_for_ids(ids,
"competing")` → `disputed_by: dict[str, list[str]]` (both directions, like the
recall hook). Fail-open to `{}` on any exception. Repo/synthesis-expansion
rows are never disputed (no pairs). Zero cost when the corpus has no pairs.

### 2. Source annotation

- Snippet header line (`ask_ops.py:564-567`) gains a segment when disputed:
  `  |  ⚔ disputed-by: [e5f6a7b8]` — same glyph as the recall dossier.
- Each affected source dict gains `"disputed_by": ["<full-id>", ...]`.
  **On copies only:** the session RAG cache shares source dicts read-only
  (`ask_ops.py:406-413`, put at `:767-775`) — never mutate a cached dict;
  rebuild the row (`{**d, "disputed_by": [...]}`) in the per-request list.
- Consumers read the new key with `.get()` (test stubs build minimal dicts).

### 3. Prompt steering

When `disputed_by` is non-empty, append one instruction to the system prompt
**at the call sites** (`ask_ops.py:936-942` ask, `:1044-1050` ask_stream), not
inside the replaceable prompt file — so it survives `resolve_prompt("ask", …)`
user overrides: "Snippets marked ⚔ disputed-by are contested by another
memory. Present contested facts as contested and cite the disputing id."
This extends the existing CONFLICTING SNIPPETS rule (`prompts.py:157-160`).

### 4. Deterministic dispute gate (the abstention)

At the existing gate seam, after the LLM answer and before the
`MEMO_GROUNDING_ASK_MIN` check:

- `D` = source ids with entries in `disputed_by`.
- `C` = `match_cited(cited_ids(answer), source_ids)` — reusing
  `grounding.py:101/:112`.
- **Contested abstention** when the answer rests only on disputed evidence:
  `C ⊆ D` and `C` non-empty, or `C` empty and **all** memory sources are in
  `D`. Replace the answer with the contested message (below) and set
  `"disputed"` (see §6).
- **Deterministic caveat** when partially disputed: `C ∩ D` non-empty but not
  all — if the answer text does not already reference the disputing id, append
  `\n\n⚠ Disputed evidence: [a1b2c3d4] is contested by [e5f6a7b8] (open).`
  (one line per disputed cited id).
- Contested message (constant, not the generic fallback):
  `"I couldn't find an undisputed answer: [<id8>] and [<id8>] record
  conflicting facts. Resolve with `memo contradict` or ask about one side
  explicitly."` — it MUST keep a `journey_check.py:374-384` abstain marker
  ("couldn't find" does).
- `ask_stream`: same logic on the accumulated answer at the done event —
  tokens already streamed are not retracted (same accepted limitation as the
  existing grounding gate, `ask_ops.py:1063-1075`).

### 5. Verbatim short-circuit fix

`_verbatim_short_circuit` is skipped when the would-be top hit's id is in the
dispute lookup (falls through to the LLM + gate path). Requires running the
lookup before the short-circuit check (`ask_ops.py:919-929`) — the lookup is
already computed inside `_build_ask_context`, threaded out via the source
dicts.

### 6. Return contract (additive only — established norm: `synthesizer`,
`notification` are already added top-level by `server_core_search.py:425-428`)

- Per-source: `"disputed_by": [ids]` (absent when clean).
- Top-level: `"disputed": {"<id>": ["<other-id>", ...]}` present only when
  non-empty; on contested abstention additionally `"abstained": "disputed"`.
- `chat_ask` / `chat_ask_stream` inherit via `self.ask(...)`
  (`chat_ask_ops.py:51/:176`); `_chat_citations` untouched.
- MCP `memo_ask`, CLI `--json`, as-of variants: dict pass-through — safe.

## Error handling

- Dispute lookup: fail-open to `{}` (matches `dev_audit.py:88-89` contract) —
  ask never breaks because contradictions.db is absent/corrupt.
- Gate logic: pure set operations; wrapped so any unexpected exception leaves
  the answer untouched (log at debug).
- Flag off: no lookup, no annotation, no gate — byte-identical legacy output.

## Testing (stub pattern: `tests/test_ask_strict.py` — stubbed chat +
`_build_ask_context`)

- Unit: lookup fail-open; annotation on copies (cached dict unmutated);
  contested abstention when `C ⊆ D`; caveat append when partial and LLM
  didn't cite the dispute; no caveat when the LLM already flagged it;
  `C` empty + all-disputed → contested; opt-out flag → legacy byte-identical;
  verbatim skip when top hit disputed; stream done-event swap +
  `abstained: "disputed"`.
- Contested message asserted to match the journey abstain markers
  (`test_journey`-style substring check).
- Integration (`@pytest.mark.requires_mlx`, precedent
  `tests/test_trust_states_fixture.py`): seed an open pair, `ask` about the
  contested fact → contested abstention naming both ids.
- Eval: `eval/regression_labels.json` untouched (corpus is dispute-free by
  design, `test_trust_states_fixture.py:1-16`); end-to-end behavior gates via
  the MLX fixture test.

## Decision record

| Decision | Choice | Alternative rejected |
|---|---|---|
| Guarantee mechanism | Deterministic citation-set gate | Extra grounding-judge LLM call (cost, fail-open, non-determinism) |
| Default | ON (single opt-out flag) | Dark flag + graduation (annotation is additive; abstention is narrow) |
| Detection scope | `open` + `competing` pairs only | Also resolved losers (`kept_newer`/`kept_older` archive the loser — self-cleaning) |
| Prompt injection | Conditional call-site suffix | Editing the user-replaceable prompt (lost on override) |
| Ranking changes | None (annotation/abstention only) | Enabling contradict penalty in ask (belongs to the trust program's measured rollout) |
