# Design: graph-aware source compaction for `memo_context_pack`

**Status:** proposed
**Date:** 2026-08-04
**Scope:** `memo_context_pack` (`src/memo/server_context_pack.py` / `src/memo/context_pack.py`) only. `memo_ask`/`memo_chat_ask`/chat UI synthesis already got this in PR #186 (`src/memo/chat/graph_compact.py`, v4.9.0) — this spec is the follow-up that design's "Out of scope" section flagged as a candidate: "The MCP tool's `budget_chars`/omissions packing is a natural second home for this, but has no dedicated eval harness today ... needs its own measurement story first." Recall-hook compaction remains out of scope here too, unchanged from the original design.

## Problem

`memo_context_pack` retrieves up to `k` (default 5) memories via `memory.search(...)` and composes them into a `ContextPack` (current/supporting/stale-or-conflicting buckets + a text summary), which callers render via `.to_prompt()`. When several retrieved hits are *different memories about the same underlying entity* — not chunks of the same document, and not literal near-duplicates — each occupies its own bucket slot and pays its full `snippet_chars` cost, exactly the redundancy problem `MEMO_CHAT_GRAPH_COMPACT` (PR #186) already solved for chat/ask synthesis. `memo_context_pack` has zero graph awareness today (verified: no hits for "graph" anywhere in `server_context_pack.py` or `context_pack.py`).

Today the pack only shrinks by *truncating snippet text* (`_trim_to_budget` in `context_pack.py`) once the composed prompt exceeds `budget_chars` — it never drops a whole redundant hit before that point. A collapsed duplicate keeps its bucket slot (crowding out a would-be `k`-th distinct hit) and gets truncated character-by-character instead of removed whole.

## Prior art — what NOT to repeat

Same two closed graph experiments the chat design already ruled out, and for the same reason (this is post-rank compaction, not a ranking or retrieval-source change):

- **Graph-as-ranking-signal** (`MEMO_RECALL_GRAPH_PROXIMITY`, memory `memo-graph-program-0-4`): measured, rejected — proximity re-rank doesn't improve precision on memo's corpus.
- **Graph-as-retrieval-source** (`MEMO_GRAPH_RETRIEVAL_ENABLED`/`MEMO_GRAPH_EXPANSION_ENABLED`, memory `memo-graph-injection-negative`): measured, rejected — injecting graph-discovered candidates into the result set doesn't raise recovery.

Same landmine carries over unchanged: **raw (unweighted) entity overlap is unsafe** — ubiquitous entities dominate and produce wrong groupings; the fix is IDF weighting, exactly as chat's design already established. This spec does not re-derive that mechanism; it reuses it (see Design below).

**Verified before designing (per this spec's explicit instruction) that no existing mechanism already does this dedup for `memo_context_pack`:**

- `build_context_row`/`_snippet` (`context_pack.py`) build one row per hit unconditionally — no grouping, no dedup key, no title/path collapsing (unlike chat's `dedup.collapse_near_duplicates`, which is chat-only and handles *same-document* `(§N/M)` chunk siblings, a different problem this spec also does not touch).
- `_trim_to_budget` only removes/truncates rows to satisfy `budget_chars`, working strictly right-to-left through already-built buckets (supporting → stale → current), and never merges or groups by entity/content similarity.
- `quality_rerank=True` (passed to `memory.search(...)` at the `memo_context_pack` call site, gated by `MEMO_QUALITY_RERANK`) demotes/boosts *individual* hits by verification/contradiction/supersession signals — it changes score ordering, not which distinct memories co-occur in the result set. It has no notion of "these N hits are about the same thing."

Conclusion: the gap is real and clean. No existing redundant mechanism is being duplicated by this design.

## Design

### Module: thin wrapper, not a duplicated algorithm

New module `src/memo/context_compact.py` with one function:

```python
def compact_hits_by_entity_overlap(
    hits: list[Any],
    memory: Any,
    *,
    min_idf_overlap: float,
    min_group_size: int = 2,
) -> list[Any]:
    ...
```

**Adaptation decision:** `hits` here are `memory.search(...)` results — `MemoryRecord`-shaped *objects* with attributes (`.id`, `.title`, `.score`, ...), not the `dict` sources chat's pipeline already produces by the time `compact_by_entity_overlap` runs. Rather than duplicating chat's IDF/overlap/grouping algorithm for objects, this module adapts each hit to the minimal dict shape `compact_by_entity_overlap` (`src/memo/chat/graph_compact.py`) already expects — `{"id", "title", "score"}` — plus a synthetic `_hit_index` key, calls the existing (already-tested) function unchanged, then uses `_hit_index` to map surviving dict rows back to the original hit *objects*, filtering the original `hits` list in its original order (not the order the delegate returns, since order there is re-sorted by score and this module wants to preserve `memory.search`'s original rank order intact — hit 0 stays the pack's "current" candidate exactly as before, per `build_context_pack`'s `index == 0` rule).

This was checked against the alternative (a parallel object-shaped reimplementation of the grouping/IDF logic) and rejected: the wrapper is ~20 lines, reuses the already-unit-tested overlap/IDF math verbatim (no algorithm drift between the chat and context_pack compaction paths), and the only real adaptation work — object-to-dict shape conversion and order-preserving id mapping — is exactly the "three similar lines" the task called out as preferable to a premature shared abstraction. `MemoryRecord` is also a dataclass with fields not safe to repurpose for scratch state (e.g. stuffing a citation pointer into `.extra` would leak into serialization paths `extra` is used for elsewhere), which further favors doing the bookkeeping in the adapter dict rather than mutating hit objects.

**Why `_hit_index` instead of `id`-based reconciliation:** hit ids are assumed unique in practice, but nothing guarantees it structurally, and `_hit_index` sidesteps that assumption entirely — it's carried through both of `compact_by_entity_overlap`'s output paths (`dict(sources[group[0]])` copies it onto the survivor; `sources[i]` for below-`min_group_size` groups passes it through unchanged), so reconciliation never depends on `id` equality.

**No `related_ids` citation surfaced in the pack.** Chat's design renders `related_ids` into the synthesis prompt (`(+2 related: ...)`) so a collapsed source's identity isn't fully lost to the LLM. `memo_context_pack` has no equivalent single downstream LLM-prompt-rendering step — its consumers read structured JSON (`current_facts`/`supporting_context`/`stale_or_conflicting`) built by `build_context_row`, which `context_pack.py` is deliberately left untouched by this change (see Data flow below). Surfacing which ids got absorbed would require either mutating hit objects (rejected above) or extending `build_context_pack`'s signature/row schema, both larger changes than this spec's stated goal ("a collapsed duplicate never occupies a bucket slot at all"). Listed under Out of scope as a candidate follow-up.

### Data flow

1. `server_context_pack.py`'s `memo_context_pack` tool calls `memory.search(...)` as it does today, producing `hits: list[Any]`.
2. **New step**, gated by `MEMO_CONTEXT_GRAPH_COMPACT`: `hits = compact_hits_by_entity_overlap(hits, memory, min_idf_overlap=flag_float("MEMO_CONTEXT_GRAPH_COMPACT_MIN_IDF"))`, inserted immediately before the existing `pack = build_context_pack(question, hits, snippet_chars=snippet_chars)` call — i.e. before hits are bucketed, exactly mirroring where chat's compaction runs (right before `synth_head` slicing, before any downstream section-building/truncation).
3. `build_context_pack`, `build_context_row`, `_trim_to_budget` — all of `context_pack.py` — are **unchanged**. A collapsed hit is simply absent from the `hits` list `build_context_pack` receives, so it never occupies a current/supporting/stale slot and is never a candidate for `_trim_to_budget`'s truncation pass. A non-redundant hit that would otherwise have been pushed out by `k`'s cap now has a better chance of a slot (same effect chat's design notes for `synth_head`).

### Config

Unlike chat's `MEMO_CHAT_*` family (env-only, deliberately outside `flags.py`'s markdown-config/tuned-overlay chain per existing precedent), `memo_context_pack`'s one existing flag (`MEMO_CONTEXT_PACK`) is a normal registered flag in `src/memo/flags_search.py`, read via `flag_bool`/`flag_float`. This feature follows that precedent — it is part of the `flags.py` registry, not the chat-config pattern, since `context_pack` is not part of the `MEMO_CHAT_*` family and there is no reason to special-case it.

New `FlagSpec`s in `src/memo/flags_search.py`, alongside `MEMO_CONTEXT_PACK`:

- `MEMO_CONTEXT_GRAPH_COMPACT` — bool, default `False`. Enables graph-aware entity-overlap compaction of `memo_context_pack` hits before bucketing.
- `MEMO_CONTEXT_GRAPH_COMPACT_MIN_IDF` — float, default `0.5` (matches chat's `MEMO_CHAT_GRAPH_COMPACT_MIN_IDF` default; same conservative starting point — favors under-collapsing over merging unrelated hits), `min_val=0.0`.

Both flow through `memo config validate`'s normal typo/unknown-var detection automatically (no exclusion-list entry needed, unlike the `MEMO_CHAT_*` vars — this is the benefit of using the standard registry instead of the chat env-only convention).

### Error handling

Fail-open, identical contract to chat's: any exception during entity lookup, IDF lookup, or hit-to-dict adaptation is caught inside `compact_hits_by_entity_overlap` and the function returns the input `hits` list unchanged. This can never raise into `memo_context_pack` and can never block the tool response — worst case, a silent no-op (the pack builds exactly as it does today). `compact_by_entity_overlap` (the delegate) already fail-opens internally for the graph/IDF lookup itself; the wrapper additionally guards its own adaptation step (dict construction, id remapping) with the same broad catch, since the fail-open contract in this spec covers the whole function, not just the delegated call.

### Testing

Unit tests for `compact_hits_by_entity_overlap` against a fake `memory.graph` stub (no DB, no MLX) — mirrors `tests/test_chat_graph_compact.py`'s five cases, adapted for hit *objects* instead of dicts:

- Two hits sharing only a ubiquitous entity (high raw overlap, low IDF) → not collapsed.
- Two hits sharing a rare entity (low raw overlap, high IDF) → collapsed (lower-scoring hit dropped, order preserved).
- Empty graph / no entities on either hit → no-op passthrough.
- A group below `min_group_size` → never collapses.
- Exception from `memory.graph.*` → returns input unchanged (fail-open).

Integration-level test at the `server_context_pack.py`/`context_pack.py` wiring point, mirroring `tests/test_chat_pipeline.py`'s `test_graph_compact_collapses_related_sources_when_enabled`/`test_graph_compact_noop_when_disabled`: using the existing `mem_with_stub` fixture (a real `Memory` with a real `GraphStore`, per `tests/test_context_pack_surface.py`'s established pattern), monkeypatch `.search()` to return two same-entity hits and `.graph.memory_entities`/`.entity_doc_freqs`/`.total_indexed_memories` to simulate a shared rare entity, then assert the resulting pack has one fewer distinct hit across its buckets with `MEMO_CONTEXT_GRAPH_COMPACT=1`, and both hits present with the flag off (default).

No eval-harness-gated regression run is part of this spec — see Out of scope.

## Success criteria

- Zero behavior change when `MEMO_CONTEXT_GRAPH_COMPACT` is off (default) — ships inert, matching chat's precedent.
- When on, `memo_context_pack` output for two-or-more hits sharing a rare (low document-frequency) entity collapses them into one bucket slot; hits sharing only a ubiquitous entity are left uncollapsed.
- No change to `context_pack.py`'s bucketing/truncation behavior for hits that were never subject to compaction (i.e., the change is additive/subtractive on the `hits` list only, never a change to `build_context_pack`'s internal logic).
- Zero new dependency on `flags.py`'s chat-config exclusion list — this stays a normal registered flag pair.

## Out of scope (future follow-ups)

- **Eval-harness-gated graduation to default-on.** Unlike chat (`memo eval chat`), there is no dedicated regression corpus for `memo_context_pack` today, and building one is explicit scope creep for this change (per the task's own "No existing eval harness" instruction). This ships default-off with unit + integration test coverage only. Flipping the default requires its own measurement story — a `memo_context_pack`-specific regression corpus (or a demonstrated proxy via the existing `eval/regression_labels.json` retrieval gate) plus a token-savings measurement, the same shape of gate chat's Task 4 defined — before any human graduates the flag. This spec deliberately does not build that gate.
- **Surfacing `related_ids` in the pack's JSON or `.to_prompt()` output.** Chat cites collapsed sources by id in its synthesis prompt; `memo_context_pack` currently drops them silently. A follow-up could extend `build_context_pack`'s row schema (or the `omissions` free-text field, which already carries a similar "+N sensitive memories omitted" note) to record collapsed counts/ids. Deferred here to keep this change to the exact "don't occupy a bucket slot" scope the task asked for.
- **Recall-hook compaction.** Unchanged from the original chat design's out-of-scope note — the 5s budget (CLAUDE.md) makes any addition there higher-risk for a much smaller per-turn ceiling. Still not this spec's problem.
- **Preferring an existing `type=synthesis` memory over the mechanical representative.** Same YAGNI deferral as chat's design — not revisited here.

## Dependencies

Reads `memory.graph.memory_entities`/`entity_doc_freqs`/`total_indexed_memories` via the existing `compact_by_entity_overlap` delegate — same unconditional availability as chat's version (`GraphStore` is constructed unconditionally in `Memory.__init__`), so no hard dependency on the graph being populated; an empty graph just means the function never fires (fail-open, matches chat's dependency note verbatim). No new dependency on `src/memo/chat/*` beyond importing the one pure function `compact_by_entity_overlap` from `memo.chat.graph_compact` — that module has no chat-pipeline-specific state, so this is a plain function import, not a layering violation (context_pack does not depend on chat's `ChatConfig`, pipeline, or synthesis).
