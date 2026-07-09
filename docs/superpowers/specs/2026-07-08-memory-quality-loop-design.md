# Memory Quality Loop Design

## Goal

Improve memo's memory quality where it hurts most today: retrieval can rank
obsolete or noisy memories above better ones, and agents often receive loose
hits instead of composed, actionable context.

The design introduces a default-off Memory Quality Loop with two phases:

1. Quality-aware ranking and context packs for explicit retrieval surfaces.
2. Opt-in compaction that supersedes or archives redundant memories with
   receipts and undo.

This work must preserve memo's source-of-truth model: Markdown memories remain
authoritative, sqlite indexes are rebuildable, and destructive maintenance must
be reversible.

## Non-Goals

- Do not replace the existing hybrid/vector/BM25 candidate retrieval path.
- Do not enable new ranking or compaction behavior by default.
- Do not delete Markdown memories.
- Do not compact memories across project or scope boundaries.
- Do not compact encrypted secrets or sensitive memories.
- Do not put context-pack construction on the ambient recall hot path until it
  has passed latency and quality gates.

## Architecture

The Memory Quality Loop has three bounded components.

### Quality Signals

Quality Signals compute explainable per-memory features after candidate
retrieval. They read existing metadata and sidecars, then return a small
structured record for each candidate.

Initial signals:

- `staleness`: memory is invalidated, superseded, contradicted, or older than a
  stronger scoped fact.
- `confidence`: confidence score, verification state, and source quality.
- `usage`: recent useful retrievals, feedback, citations, and multi-session
  reinforcement.
- `specificity`: project/entity/scope match and whether the memory is more
  specific than a competing generic fact.
- `redundancy`: near-duplicate or cluster membership.
- `synthesis_support`: whether the memory supports or is supported by a
  synthesis/profile/canonical memory.

Missing optional inputs must degrade gracefully. A candidate with no feedback or
contradiction state keeps its base score and records a trace reason such as
`quality_signal_missing`.

### Quality Reranker

The reranker takes candidates from the existing retrieval path and computes:

```text
final_score = base_score + quality_boosts - quality_penalties
```

Boosts:

- Confirmed across sessions.
- Recently used with positive feedback.
- Specific to the requested project, entity, or time range.
- Cited by a current synthesis/profile/canonical memory.

Penalties:

- Invalidated or superseded.
- Resolved contradiction where this memory lost.
- Lower confidence than a competing scoped memory.
- Generic when a more specific candidate is available.
- Near-duplicate of a stronger canonical candidate.

The reranker must not silently hide useful historical context. Penalized stale
or conflicting memories may still be returned to the Context Pack Builder as
`stale_or_conflicting` when they explain how a decision changed.

### Context Pack Builder

The Context Pack Builder converts a ranked candidate list into a compact,
agent-facing package:

```json
{
  "summary": "The current decision appears to be X; Y is historical because Z.",
  "current_facts": [],
  "supporting_context": [],
  "stale_or_conflicting": [],
  "omissions": "+N candidates omitted by budget"
}
```

Responsibilities:

- Separate current facts from stale or conflicting evidence.
- Keep enough supporting context to make the current facts actionable.
- Explain why high-similarity candidates were demoted when that matters.
- Preserve citation ids so downstream answers can cite sources.
- Enforce a token/character budget deterministically.

Budget trimming order:

1. Trim `supporting_context`.
2. Trim `stale_or_conflicting`.
3. Trim lower-value `current_facts`.
4. Preserve `summary` and at least the top current fact whenever possible.

Initial integration points:

- `memo ask` behind `MEMO_CONTEXT_PACK`.
- A new explicit MCP/CLI surface for context packs behind the same flag.
- Retrieval trace output for debugging quality decisions.

Ambient recall and session briefing can use context packs only after focused
latency and recall-quality evaluation proves the hot path remains healthy.

## Compaction Pass

The compaction pass is an opt-in maintenance operation named
`quality_compact`. It can run under `memo maintain` and later under `memo dream`
once proven safe.

Modes:

- `preview`: identify candidate clusters, propose a canonical memory, list
  affected source ids, and explain reasons.
- `apply`: write the canonical memory, mark sources as `superseded_by` or move
  them to archive, and emit a receipt that supports undo.

Candidate clusters:

- Near-duplicates.
- Multiple small memories about the same stable decision or preference.
- Historical memories replaced by a newer current decision.
- Clusters where a synthesis already captures the durable fact and sources add
  recall noise.

Safety rules:

- Never delete Markdown.
- Never compact across different scopes or projects.
- Never compact encrypted secrets or sensitive memories.
- Never compact if the canonical body cannot cite every source id.
- Never apply if receipt creation or undo registration fails.
- Default to `preview` unless the user explicitly asks for `apply`.

The canonical memory must include provenance for source ids. Source memories
must point back to the canonical id through metadata or body front matter so
reindex can reconstruct the relationship.

## Flags

All behavior ships default-off:

- `MEMO_QUALITY_RERANK`
- `MEMO_CONTEXT_PACK`
- `MEMO_QUALITY_COMPACT`

Flags must be registered and accessed through `src/memo/flags.py`; app code
must not read `MEMO_*` values with raw environment access.

## Error Handling

- If optional quality data is unavailable, fall back to base ranking and add a
  trace reason.
- If contradiction state is ambiguous, keep the memory visible but classify it
  as `stale_or_conflicting`.
- If scope or project agreement cannot be proven, compaction skips the cluster.
- If receipt or undo setup fails, compaction aborts before mutating memories.
- If context-pack construction exceeds budget, trim deterministically using the
  budget order above.
- Domain failures should use `memo.errors.MemoError` subclasses rather than
  bare exceptions.

## Testing And Evaluation

Unit tests:

- Quality signal extraction with missing optional sidecars.
- Reranking order for invalidated, superseded, contradicted, verified, generic,
  specific, duplicate, and canonical candidates.
- Context pack classification and deterministic budget trimming.
- Compaction candidate selection, preview output, apply behavior, and undo.

Integration tests:

- `memo ask` with `MEMO_CONTEXT_PACK=0` preserves existing behavior.
- `memo ask` with `MEMO_CONTEXT_PACK=1` returns composed context with citations.
- MCP context-pack tool respects source attribution and compact output budgets.
- `memo maintain quality-compact --preview` is read-only.
- `memo maintain quality-compact --apply` writes receipts and can be undone.

Evaluation gates:

- `uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force`
- New or extended metrics:
  - `stale@k`: stale or superseded hits in the top k.
  - `canonical_hit@k`: canonical/current memory appears when redundant sources exist.
  - `pack_answerability`: context pack contains enough current facts to answer.
  - `compaction_safety`: preview/apply never crosses project/scope boundaries.

The first implementation should target explicit retrieval surfaces only. The
ambient recall hot path remains unchanged until the evals show quality gains
without latency regressions.

## Rollout

1. Add signal extraction and tracing behind `MEMO_QUALITY_RERANK`.
2. Enable reranking for explicit search/ask paths only and run regression evals.
3. Add Context Pack Builder for `memo ask` and an explicit MCP/CLI surface.
4. Add `quality_compact --preview`.
5. Add `quality_compact --apply` with receipts and undo.
6. Consider ambient recall or briefing integration only after metrics improve.

## Open Decisions Resolved

- Ranking/context packs happen before automatic compaction.
- Compaction is opt-in, receipt-backed, and reversible.
- Obsolete memories are demoted or classified, not silently hidden.
- The design favors systemic quality signals over patching individual queries.
