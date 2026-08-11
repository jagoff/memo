# Emission Ledger — design

**Date**: 2026-08-10
**Status**: implemented, shipped dark — `MEMO_EMITTED_LEDGER=0` by default;
see "Measured result" below for why
**Origin**: analysis of [messkan/prompt-cache](https://github.com/messkan/prompt-cache),
asking what it has that memo could use to cut the user's token spend.

## Summary

memo tracks which memory bodies it has already put into the current context
window, and stops re-emitting them. A repeat hit returns `{id, title, ref}`
instead of the full body.

The saving is bounded by how much memo repeats itself inside one session. The
recall hook fires on every `UserPromptSubmit` and the MCP read tools pull from
the same corpus with the same embedder, so the overlap between them is the
largest single source of duplicate bytes memo produces.

## What prompt-cache actually is, and why this is not a port of it

prompt-cache is an OpenAI-compatible Go proxy. It embeds the incoming prompt,
finds the nearest cached prompt, and returns that prompt's stored completion
without calling the provider:

```
sim >= 0.70          -> hit
sim <  0.30          -> miss
0.30 <= sim < 0.70   -> gray zone, a cheap LLM (gpt-4o-mini) judges "same intent?"
```

Two things about that codebase are worth recording, because they are the reason
almost none of it is reused here.

**Its ANN index is not an ANN index.** `internal/ann/ann.go` is a linear scan
plus a full `sort.Slice` over every vector — O(n log n), despite the comment
claiming O(log n). Worse, `FindSimilar` in the ANN branch calls `Search()` and
then calls `Store.GetAllEmbeddings(ctx)` anyway — a full read of the embedding
table out of BadgerDB — to recompute exact cosine. The "fast" path does all the
work of the slow path plus a full disk read. memo already has sqlite-vec and a
warm MLX embedder; there is nothing to take here.

**Its default thresholds are unsafe.** Raw cosine over modern embedding models
puts unrelated text above 0.6 routinely, so at 0.70/0.30 nearly everything lands
in the gray zone and the verifier LLM carries all of the precision. Their own
docs then recommend `ENABLE_GRAY_ZONE_VERIFIER=false` for cost — which leaves a
cache that answers one question with another question's response.

The deeper mismatch is structural:

> prompt-cache caches to avoid an operation that is expensive (a provider API
> call). In memo the operation is cheap — sqlite-vec + BM25 + cross-encoder,
> all local. What is expensive is the **payload**.

So memo should not cache the operation. It should cache the **emission**. That
inverts the design: always run the query, then decide what to serialize based on
what is already in the window. No query embedding, no similarity threshold, no
verifier, no false positives.

What survives from prompt-cache is the framing — "a large percentage of requests
are repetitive, stop paying twice" — and the discipline of measuring hit rate
rather than assuming it.

## What memo already has (not rebuilt here)

| Surface | Existing mechanism |
| --- | --- |
| Recall injection size | `MEMO_RECALL_TOKEN_BUDGET`, `MEMO_RECALL_ADAPTIVE_BUDGET`, `session_budget_scale` |
| Recall redundancy | `recall_dedup.collapse_near_dups` (Jaccard), `_apply_mmr`, session dedup at `cli_recall_hook.py:816` |
| MCP payload size | `mcp_budget.py` — caps and fails loudly over budget |
| Memory tier eviction | `memory.cache` / `memo_cache_stats` / `memo_cache_evict` |

The gap is that nothing tracks emissions **across** these surfaces. The recall
hook does not know what the MCP tools emitted, the MCP tools do not know what
the hook injected, and neither remembers across turns.

## Architecture

### Components

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `src/memo/emitted_ledger.py` (new, ~120 lines) | append / read / reset. Leaf: no store, no MLX, no flag reads beyond its own | stdlib |
| `cli_recall_hook.py` (edit) | after render, append the ids it injected | ledger |
| `server_common.py` (edit) | `apply_ledger(hits, ref) -> (full, digest)`, called by participating tools before serializing | ledger |
| `cli_hooks.py` (edit) | reset on PreCompact and SessionStart | ledger |

### Session key

`identity._session_id()` reads `MEMO_SESSION_ID` / `CLAUDE_SESSION_ID` /
`CLAUDE_CODE_SESSION_ID`. Verified 2026-08-10 on live processes: the `memo-mcp`
stdio child **does** inherit `CLAUDE_CODE_SESSION_ID`, and two concurrent Claude
Code sessions carry distinct values (pid 88865 -> `e00b57f5…`, pid 97573 ->
`1d3fb1db…`). The hook receives the same id in its stdin payload. So both
writers key the same file without coordination.

Clients that export no session var (Claude Desktop, Codex, opencode, Cursor)
fall through to `server_session_patterns._fallback_session_id`, which is
process-scoped — correct, since a stdio MCP server lives exactly as long as its
client session. Those clients have no recall hook, so the ledger degrades to
MCP-to-MCP coverage.

### Storage

`state_dir/emitted/<session_id>.jsonl`, append-only:

```json
{"id":"mem_41f","h":"9c2a","n":400,"hp":"7b10","ref":"memo-h/1c2","t":1754820001,"src":"hook"}
{"id":"mem_902","h":"be71","n":912,"hp":"3fa2","ref":"memo-r/a3f9","t":1754820044,"src":"mcp"}
```

`h` is a short hash of **the text that was actually emitted**, and `n` is its
length in characters. `hp` is a hash of just that text's first 200 characters
— the prefix hash the length arm of the monotonic-emission rule below
requires; `null` on a ledger line written before this field existed. Not a
memo version number, and not a hash of the stored body — the distinction
matters:

- Hashing the emitted text is self-contained. A body changed by any route
  (`memo_update`, the nightly consolidate pass, vault-ingest) produces a
  different emission and therefore invalidates, without the ledger knowing
  anything about memo's versioning.
- Hashing the *stored* body would be a correctness bug. Tools render the same
  memory at different lengths — the recall hook truncates to
  `MEMO_RECALL_BODY_CHARS` (default 400), `memo_ask` may emit more. If the hook
  emitted 400 chars at turn 2 and a stored-body hash said "same" at turn 3, the
  model would be digested past content it never saw.

This gives the **monotonic-emission rule**: digest only when `h` matches, or
BOTH `new_len <= n` AND the new emission's prefix hash matches the recorded
`hp`. An entry with no recorded prefix hash (`hp` is `null`) is always sent in
full — unknown means unsafe means full, and that direction costs tokens, not
correctness. An emission that would be longer than what is already in the
window is always sent in full regardless, and replaces the ledger entry.

The prefix hash exists because length alone is not enough: a body that was
edited and happens to be shorter would satisfy `new_len <= n` while
describing text the model never saw. Requiring the first 200 characters to
match catches an edit that changes the start of a body, while still
accepting a prefix-preserving shortening — trailing truncation, where the
model has already seen a superset of the new text.

`ref` is a short token minted per emission batch — `memo-r/` + the first 6 hex of
`sha256(sorted_ids + t)` for MCP, `memo-h/` for hook injections — echoed in the
payload so the digest points at a specific prior message without needing turn
numbers.

No locking. Short lines written with `O_APPEND` are atomic; a reader that hits an
unparseable final line skips it. Capped at `MEMO_EMITTED_LEDGER_MAX` entries, FIFO.

### Flow

```
turn 2   UserPromptSubmit
         recall hook injects mem_41f, mem_902
         ledger.append(41f@9c2a, 902@be71, ref=memo-h/1c2)

turn 3   memo_search("cómo anda el chat")
         runs the full pipeline (sqlite-vec + bm25 + rerank)    <- ALWAYS
         -> hits [41f, 902, 7c3]
         ledger: 41f hash matches -> digest
                 902 hash matches -> digest
                 7c3 absent       -> full
         emits 1 body + 2 titles + ref     (illustrative: ~380 tok vs ~1800;
                                            the real ratio is what test 7 measures)

turn 9   memo_update(mem_41f)
         ledger untouched

turn 11  memo_ask("explicame el chat")
         -> hits [41f, 902, 7c3]
         ledger: 41f hash CHANGED -> full (re-emit)
                 902, 7c3 match   -> digest
         emits 1 body + 2 titles

PreCompact -> ledger.reset() -> everything emits full again
```

### Digest payload

```json
{
  "results": [ { "id": "mem_7c3", "title": "...", "body": "..." } ],
  "already_in_context": [
    {"id": "mem_41f", "title": "memo chat en :8767", "ref": "memo-h/1c2"},
    {"id": "mem_902", "title": "Gate pre-push mide drift", "ref": "memo-h/1c2"}
  ],
  "hint": "bodies already emitted above; memo_get(id) if you cannot see them"
}
```

Per-item, not per-call: a partially-overlapping result set emits the new bodies
in full and digests the rest.

### Tool scope

Participating (return bodies from a search): `memo_search`, `memo_ask`,
`memo_evidence_pack`.

Exempt: `memo_get` and `memo_history`. These mean "give me this one,
explicitly". If `memo_get` participated, the `hint` would deadlock.

Within `memo_ask`, only rows with `source == "memory"` participate — its
`sources` list is not memory-only. With `include_repos=True` (the tool's
default), `ask_ops._build_ask_context` also appends `source == "repo"` rows
from the indexed repo corpus into the same list. A repo row's id is not a
memory id, so `memo_get(id)` — the digest's own escape hatch — cannot resolve
it; digesting one would hand the model a pointer with no way back to the
content. `memo_evidence_pack`'s `items` cannot carry a non-memory row —
`evidence_ops._build_items` hardcodes `source="memory"` on every
`EvidenceItem` and evidence_pack never calls the repo corpus — so no
equivalent guard is needed there.

Also out of scope, discovered during Task 5's implementation rather than
anticipated here: `memo_context` and `memo_unified_briefing`. Neither has a
hit list this mechanism can suppress. `memo_context`'s structured `hits` key
(`context_surface.py`'s `_consult_hits_with_sections`) carries `id`/`title`/
`score`/`section` only — no body text to hash — and the body text it does
return lives inside the packed `prompt` string, which embeds full snippets
inline (`context_pack.py`'s `_format_section`); suppressing the bodyless
structured list while the prompt still carries every body in full would be a
no-op dressed up as a feature. `memo_unified_briefing` returns one
`compact_text`-squashed markdown string (`compose_unified_briefing`) with no
per-hit list to partition at all. Both are absent from
`MEMO_EMITTED_LEDGER_TOOLS`'s default rather than half-wired.

### Flags

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `MEMO_EMITTED_LEDGER` | bool | `0` | Master switch. Off until the success criteria below are met. |
| `MEMO_EMITTED_LEDGER_TOOLS` | csv | the three above | Participating tools. |
| `MEMO_EMITTED_LEDGER_MAX` | int | `500` | Entry cap, FIFO. |

## Invalidation

| Event | Behaviour | Mechanism |
| --- | --- | --- |
| body changes (`memo_update`, nightly consolidate, vault-ingest) | hash differs, re-emit full | emitted-text hash |
| a tool would emit *more* of a memory than was emitted before (hook truncated to 400, `memo_ask` has room for 900) | re-emit full, replace the entry | monotonic-emission rule (`new_len <= n`) |
| memory deleted | absent from hits | nothing needed |
| auto-compaction | reset | PreCompact, already wired at `cli_hooks.py:98` |
| `/clear` | reset | SessionStart |
| new session | new file | session-id partition |
| two sessions, same cwd | separate files | session-id partition |
| **subagent** | **inherits the parent ledger, has its own window** | see hazard 1 |

## Hazards

**1. Subagents — confirmed, no clean fix.** A subagent does not start its own
MCP server; it uses the parent's connection and shares the session id, but has
its own context window. It would receive a pointer to a `ref` it never saw.
Implementation should try to detect subagent calls via the fastmcp `Context`; if
that is not available, the behaviour is accepted, because the digest carries ids
and titles — it degrades to one extra `memo_get` round trip, not to blindness.
This is precisely why the digest carries titles rather than being a bare pointer.

**2. PreCompact double-fire.** `cli_hooks.py:101` already documents double-firing
against the plugin copy. `reset()` deletes the file, so it is idempotent.

**3. Clients with no session id.** Covered above: MCP-to-MCP only. Correct, not
broken.

**4. Orphaned ledger files.** Prune `state_dir/emitted/*.jsonl` with mtime older
than 48h, hung off the existing nightly gc passes in `memo-nightly.sh`.

**5. The recall hook must not fail because of this.** The append is wrapped and
fail-open, never propagates. ~1ms against a 5s budget, but the rule stands
independently of the cost.

**6. The eval gate will move.** This touches recall rendering. Per prior
experience, the pre-push gate measures corpus drift rather than code: isolate
with a worktree on master before attributing a regression to this diff.

## The metric that decides whether this is worth shipping

The failure mode is arithmetic, not semantic:

> If the model calls `memo_get` for every digested id, the change **loses**
> tokens. Digest + tool call + full body costs more than having sent the body
> once.

So "tokens suppressed" is not the measurement. The net is:

```
net_saved = tokens_suppressed
          − tokens_spent_on_digests
          − (memo_get_after_digest × roundtrip_cost)
```

`roundtrip_cost` is defined concretely as the tokens a recovery costs that the
baseline would not have paid: the `memo_get` tool-call block the model emits,
plus the returned body. Measured, not assumed — the counters record actual
`est_tokens` of both, so `roundtrip_cost` is a measured average over the
session, not a constant.

`memo_get_after_digest` counts a `memo_get` whose id appeared in an
`already_in_context` block earlier in the same session. Attribution is
best-effort: a `memo_get` the model would have issued anyway is
indistinguishable and is counted against us, which biases the metric
conservative. That is the correct direction for a gate.

Exposed by extending `memo_cache_stats` — **no new tool**, because a new tool
costs schema tokens on every request and would contradict the feature:

```json
"emit_ledger": {
  "entries": 42,
  "digests_served": 17,
  "tokens_suppressed": 11400,
  "tokens_digest": 890,
  "memo_get_after_digest": 3,
  "net_saved_est": 9200
}
```

## Success criteria

1. **≥25% fewer emitted tokens** over a replayed real transcript.
   The denominator is the total `mcp_budget.est_tokens` of everything memo put
   into that session's window with `MEMO_EMITTED_LEDGER=0` — recall-hook injections
   plus participating tool results. Not whole-session tokens; memo is not
   responsible for those. Verify: replay the same transcript with the flag off
   and on, diff the two totals.
2. **`memo_get_after_digest` < 20% of `digests_served`.**
   Verify: ledger counter. Above this the digest is too aggressive and the
   design loses.
3. **Recall hook p95 latency delta < 20ms.**
   Verify: `recall_metrics` before/after.
4. **Eval gate not regressed.**
   Verify: master worktree first, then the diff.

If 1 or 2 fail, the feature stays at `MEMO_EMITTED_LEDGER=0` and is not promoted.
That is the decision, not a deferral.

## Measured result

Task 10. Full methodology, denominator definition, and raw command output:
`docs/eval/emission-ledger-replay.md`. Harness: `scripts/eval_emission_ledger.py`.

| # | Criterion | Result | Measured |
| - | --- | --- | --- |
| 1 | ≥25% fewer emitted tokens on a replayed transcript | **PASS, with margin, under normal load** | Two independent clean runs of the same real transcript (`e00b57f5-8745-4462-a8dd-fbb60a6616b9.jsonl`, 7 participating calls) on a quiet system: 31.4% and 36.6% reduction. A third run measured 21.8% (below the floor) while executing under severe, self-inflicted resource contention (a duplicate `pytest` run competing for MLX/GPU resources) — attributable to memo's cross-encoder reranker falling back under GPU contention, not the ledger. Full run-by-run table: `docs/eval/emission-ledger-replay.md` |
| 2 | `memo_get_after_digest` < 20% of `digests_served` | **UNMEASURABLE by replay** | a replay has no model in the loop deciding whether to recover a digested id via `memo_get` — any rate a replay produces is an artifact of the harness's determinism, not evidence. Needs a live dogfooding period reading `memo_cache_stats`'s `emit_ledger.memo_get_after_digest` / `emit_ledger.digests_served` |
| 3 | Recall-hook p95 latency delta < 20ms | **PASS** | warm-daemon path (production path on this machine): delta p50 = -0.56ms, delta p95 = -102.64ms (negative at every percentile — ledger write is noise against a ~400ms round trip). Subprocess-fallback path: delta p50 ≈ +5.95ms, delta p95 ≈ +12.83ms, under the 20ms ceiling. Measured in Task 6, not re-measured here |
| 4 | Eval gate not regressed | **PASS** | `memo eval recall --gate` fails identically on an isolated `origin/master` worktree and on this branch (stale baseline pinned to a config — `H synth/0.05` — the current default selection doesn't run; predates this diff). The corpus-cancelling `memo eval recall --against origin/master` check — same live corpus, both runs uncached — shows zero delta: prec@k 0.724 vs ref 0.724, noise@k 0.000 vs ref 0.000 |

**Decision: KEEP `MEMO_EMITTED_LEDGER` AT `0`. Do not promote.**

This is not a criterion-1 failure — under normal (uncontended) conditions
criterion 1 passed with clear margin on two independent runs, with no harness
tuning, and criteria 3 and 4 are also clean passes. The reason not to
flip the default is criterion 2: it requires evidence a transcript replay
cannot produce by construction (no model in the loop to decide whether to
call `memo_get` on a digest), and no live-dogfooding data exists yet to
supply it. Promotion needs both criteria 1 and 2 to hold — an unmeasured
criterion is not the same as a passed one, and this task does not fabricate
the number to force a promotion.

All ten tasks' code ships regardless, exactly as planned: the feature is
fully implemented, tested, and available behind the flag for anyone who sets
`MEMO_EMITTED_LEDGER=1` explicitly (e.g. to start collecting the live data
criterion 2 needs). The default stays off until a real session population
supplies a measured `memo_get_after_digest` / `digests_served` ratio.

## Test plan

| # | Type | Covers |
| --- | --- | --- |
| 1 | unit | `emitted_ledger`: roundtrip, torn final line, FIFO cap, idempotent reset, missing dir |
| 2 | unit | `apply_ledger`: full/digest partition, changed hash -> full, empty ledger -> all full |
| 2b | unit | monotonic-emission rule: recorded `n=400`, new emission 900 chars -> full + entry replaced; new emission 200 chars -> digest |
| 3 | integration (MCP) | two identical `memo_search` -> second digests; with `memo_update` between -> full |
| 4 | integration (hook -> MCP) | hook appends, subsequent `memo_search` digests that id |
| 5 | integration | PreCompact -> reset -> full again |
| 6 | fail-open | unwritable `state_dir` -> hook still renders, tools still emit full |
| 7 | eval | transcript replay, measures criteria 1 and 2 |

TDD order: 1 and 2 first (RED), then the module. 3–6 after wiring.

## Related

- `docs/SPECS/2026-08-06-mcp-response-budget-plan.md` — the response cap this
  complements. That one bounds a single response; this one bounds repetition
  across a session.
- `src/memo/recall_dedup.py` — within-render dedup. This is the across-render
  counterpart.
