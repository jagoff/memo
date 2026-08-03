# Design: graph-aware source compaction for chat/ask

**Status:** proposed
**Date:** 2026-08-03
**Scope:** `memo_ask` / `memo_chat_ask` / chat UI synthesis (`src/memo/chat/pipeline.py`) only. Recall-hook and `memo_context` compaction are explicitly out of scope — candidate follow-up specs, not part of this change.

## Problem

memo's chat pipeline feeds up to `MEMO_CHAT_SYNTH_HEAD` (default 8) distinct sources to answer synthesis, each with a full snippet/body. When several of those sources are *different memories about the same underlying entity/topic* — not chunks of the same document (that's already handled by `MEMO_CHAT_FULLDOC`/`dominant_doc_group`, which reassembles same-document chunks) — the synthesis prompt pays for redundant content multiple times: the same fact restated across 2-3 near-duplicate sources, each burning its full token cost.

memo already measures the token cost this injection causes downstream (`token_meter.py` joins Claude Code transcript usage with memo's own `context_cost.log`), but nothing today uses the entity graph to shrink that cost. This design closes that gap for the chat surface specifically.

## Prior art — what NOT to repeat

Two graph-and-retrieval experiments are already closed and must not be relitigated by this design:

- **Graph-as-ranking-signal** (`MEMO_RECALL_GRAPH_PROXIMITY`, memory `memo-graph-program-0-4`): measured, rejected. Proximity re-rank does not improve precision on memo's corpus (too sparse/noisy at ~1900 memories).
- **Graph-as-retrieval-source** (`MEMO_GRAPH_RETRIEVAL_ENABLED`/`MEMO_GRAPH_EXPANSION_ENABLED`, memory `memo-graph-injection-negative`): measured, rejected. Injecting graph-discovered candidates into the vec/hybrid result set does not raise answer recovery — vec's ranking ceiling isn't the bottleneck.

This design is deliberately a **different** application: it never changes *which* memories get retrieved or how they're ranked. It only changes how compactly an *already-decided* source list is rendered before synthesis. That sidesteps both closed experiments' failure mode (which was about the wrong things going into the ranked pool or the wrong things floating to the top).

The one landmine that *does* carry over: **raw (unweighted) entity overlap is unsafe.** The `memo-graph-injection-negative` writeup found that ubiquitous entities ("memo", "synapse") dominate raw overlap scoring and produce wrong associations, and that the fix was IDF weighting. That mechanism (`MEMO_RECALL_GRAPH_PROXIMITY` / `graph_proximity.graph_boost_factory`) has since been removed from the codebase entirely — retrieval no longer uses `graph_proximity.py` (its current docstring: "Retrieval serving no longer uses this module"). What survives, unused, are the two primitives it was built on: `GraphStore.total_indexed_memories()` and `GraphStore.entity_doc_freqs()` (`src/memo/graph.py:669,675`), still tested (`tests/test_graph_store.py`), just orphaned. This design revives them with a freshly-written IDF helper — `idf(df, total) = max(0.0, log(total / df))` when `df > 0`, else `0.0` — not by importing now-dead code.

## Design

### Module

New pure-ish module `src/memo/chat/graph_compact.py`, matching the existing `chat/fulldoc.py` / `chat/dedup.py` pattern:

```python
def compact_by_entity_overlap(
    sources: list[dict[str, Any]],
    memory: Any,
    *,
    min_idf_overlap: float,
    min_group_size: int = 2,
) -> list[dict[str, Any]]:
    ...
```

### Data flow

1. For each source dict in the pipeline's `fused` list, fetch its canonical entity set via `memory.graph.memory_entities(source["id"])` (existing primitive, already used in `contextual.py`).
2. One batched call to `memory.graph.entity_doc_freqs(all_entity_names)` + `memory.graph.total_indexed_memories()` — one query for the whole source list, not per-pair.
3. Single greedy pass, sources sorted by relevance descending (mirrors `recall_logic.collapse_near_dups`'s structure): walk sources in that order, keeping a list of group representatives seen so far. For each source, compare its entity set against each existing representative's; join the *first* representative where `Σ(IDF of shared entities) / union-weight ≥ min_idf_overlap`, else start a new singleton group with this source as its own representative.
4. For each group with size ≥ `min_group_size`: keep the highest-relevance member's full dict; attach a `related_ids: list[tuple[id, title]]` field listing the dropped members; drop the rest from the returned list entirely.
5. Return the compacted list. Everything downstream — `filter_by_relevance`, the `[:cfg.synth_head]` slice, prompt assembly — is unchanged. Dropping redundant members before the `synth_head` slice also means a non-redundant source that would otherwise have been pushed out of the top-8 now gets a slot.

### Prompt rendering

Wherever the synthesis prompt currently renders one source's snippet, also render `related_ids` as a short trailing note (e.g. `(+2 related: 90a3aa5e, a78346ac)`), reusing the existing `[id]` short-citation convention (`recall_logic.CITE_INSTRUCTION`) so the LLM can still reference a collapsed source by id if the representative's content isn't enough — the information isn't lost, just not paid for by default.

### Config

Follows the existing chat-config convention: **env-only, read directly in `src/memo/chat/config.py`, not registered in `flags.py`'s markdown-config/tuned-overlay chain** (per CLAUDE.md, this is deliberate for the 9 existing `MEMO_CHAT_*` knobs).

New fields on `ChatConfig`:
- `graph_compact: bool` — env `MEMO_CHAT_GRAPH_COMPACT`, default `False`.
- `graph_compact_min_idf: float` — env `MEMO_CHAT_GRAPH_COMPACT_MIN_IDF`, default `0.5`. Conservative starting point (favors under-collapsing over merging unrelated sources); the implementation plan should treat this as a value to validate against `memo eval chat`, not a final number.

Both names must be added to the exclusion list in `src/memo/flags.py` (currently lines ~296-304) alongside the other 9 `MEMO_CHAT_*` vars, so `memo config validate` doesn't flag them as typos.

### Error handling

Fail-open, matching the pattern used throughout `recall_logic.py` and the rest of the chat pipeline: any exception during entity lookup, IDF lookup, or an empty/uninitialized graph is caught broadly, logged at debug level, and the function returns `sources` unchanged. This can never block or degrade a chat response — worst case it's a silent no-op.

### Testing

Unit tests for `compact_by_entity_overlap` against a fake `memory.graph` stub (no DB, no MLX) — same style as `tests/test_graph_bridges.py`:
- Two sources sharing only a ubiquitous entity (high raw overlap, low IDF) → **not** collapsed.
- Two sources sharing a rare entity (low raw overlap, high IDF) → collapsed.
- Empty graph / no entities on either source → no-op passthrough.
- A "group" of size 1 → never collapses (`min_group_size` floor).
- Exception from `memory.graph.*` → returns input unchanged (fail-open).

Gate before enabling by default anywhere: run `memo eval chat` before/after with the flag on, on a real corpus. Must not regress pass-rate or latency. Measure the actual token delta via the existing `token_meter`/`eval_tokens` harness — this is what closes the loop back to the original ask ("save users tokens") with a real number instead of an assumption.

## Success criteria

- `memo eval chat` pass-rate unchanged (or improved) with `MEMO_CHAT_GRAPH_COMPACT=1` vs. off, on the existing regression corpus.
- Measurable reduction in injected/answer tokens per chat turn where compaction fires (via `token_meter`), on turns where retrieval actually surfaces same-entity-cluster redundancy.
- Zero behavior change when the flag is off (default) — this ships inert.
- Zero new dependency on `memo.flags`'s markdown-config chain (stays consistent with existing `MEMO_CHAT_*` knobs).

## Out of scope (future specs)

- **Recall-hook compaction.** Same primitive could apply to `collapse_near_dups`'s output, but the hook's 5s budget (CLAUDE.md) makes any addition there higher-risk for a much smaller per-turn ceiling (already only 2-5 hits, 400 chars each). Worth a dedicated spec once the chat version is measured and proven safe.
- **`memo_context` compaction.** The MCP tool's `budget_chars`/omissions packing is a natural second home for this, but has no dedicated eval harness today (unlike `memo eval chat`) — needs its own measurement story first.
- **Preferring an existing `type=synthesis` memory over the mechanical representative**, when the dream `run_synthesize_communities` pass has already produced one covering the same cluster (denser, human/LLM-curated prose vs. a raw memory body). Deliberately deferred — YAGNI for v1; the mechanical IDF-overlap representative is enough to validate the mechanism before layering synthesis-preference on top.

## Dependencies

Reads `memory.graph.memory_entities`/`entity_doc_freqs`/`total_indexed_memories` — all live, unconditionally available (`GraphStore` is constructed unconditionally in `Memory.__init__`, not gated behind a flag), so this has no hard dependency on the graph being populated — an empty graph just means the function never fires (fail-open). Compaction quality does improve with fresher canonical entities/edges, which is what the now-merged `dream-hyde-graph-hygiene` activation (HyDE-tune, entity-canon, edge-verify — see `docs/superpowers/specs/2026-08-03-dream-hyde-graph-hygiene-design.md`) provides — a soft, not hard, dependency.
