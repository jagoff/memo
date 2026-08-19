# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases before `2.0.0` are archived in [docs/CHANGELOG-archive.md](docs/CHANGELOG-archive.md).

## [Unreleased]

## [4.13.2] - 2026-08-18

### Fixed

- `memo ops checkpoint-wal` (new, called by the nightly pass) actually reclaims
  the sqlite write-ahead logs. `PRAGMA journal_size_limit`, added in 4.13.0,
  caps a WAL only after a successful checkpoint — and a checkpoint cannot
  advance past the oldest open reader, of which memo keeps several permanently
  (recall daemon, watcher, every memo-mcp session). Measured: graph.db-wal at
  74MB against a 127MB database, and back to 68MB hours after a manual
  truncate.
- The `hook-commands-resolve` release gate now also resolves the `memo`
  subcommands fired by `launchd/memo-nightly.sh`, not just `hooks/hooks.json`.
  That is the surface that actually drifted: `ops gc-emitted-ledgers` shipped in
  the template before the binary registered it, and for four nights the pass
  logged `Error: No such command` into a file nobody reads.

## [4.13.1] - 2026-08-18

### Fixed

Follow-up to the 4.13.0 QA sweep — the MEDIUM findings that release left on the
table.

- `Config.ensure_dirs` raised a bare `RuntimeError` on an unwritable
  data/state dir; `cli.main` only catches `MemoError`, so the clean "cannot
  create <dir>" message was replaced by a ~40-line chained traceback.
- `memo mcp-command --client codex` emitted `MEMO_MCP_PROFILE=agent` while
  `memo setup codex` wrote `core` and `memo doctor --agent codex` asserted
  `core`: the profile now comes from the agent registry unless explicitly set.
- The graph/contradiction/crossref sidecars share one sqlite connection across
  the FastMCP worker threadpool but locked only their writes; reads now take the
  same (re-entrant) lock and no cursor outlives it.
- `memo_graph verb="communities"` returned a silently truncated page (no
  `total`) with every community's full entity list — 4,327 communities and 156
  entities in the largest, ~12k tokens at the default limit.
- `consolidate` classified clusters from a prompt capped at 24k chars while
  returning the full member list; a truncated cluster now reports
  `members_seen_by_llm` and refuses to classify.
- The lifecycle metadata writers read the legacy vault copy and wrote to
  `memory_dir`, leaving two `.md` files with the same canonical id.
- The eval cache key omitted every inherited ranking knob, so an A/B run could
  return the other arm's cached rows.
- The nightly dream receipt blended curated and auto-harvested labels into one
  `prec@k` headline; both partitions are now recorded, curated first.
- The chat SPA's delete action hit an endpoint answering `501 "deferred to plan
  2"`; it now explains the refusal and names the CLI command instead.
- `eval/regression_labels.json` gains three project-tagged prompts: every prompt
  carried `project=null`, so the three project/global boost tiers were dead code
  for the entire regression gate.

## [4.13.0] - 2026-08-18

A full-project QA sweep (24 adversarially verified findings, measured
against the live corpus).

### Fixed

- **Recall injected one memory where three qualified.** The post-rank
  gap-trim compared an absolute 0.10 delta against `h.score`, which is not
  a bounded cosine — `search_scoring_ops` stacks three multiplicative
  boosts and the live range reaches ~6.8 — so it fired on 18 of 30
  multi-hit prompts and dropped a relevant rank-2 hit in 15 of them (one
  scoring 96.5% of rank-1). The gap is now RELATIVE to rank-1
  (`MEMO_RECALL_GAP_THRESHOLD` default 0.10 → 0.50, new semantics).
  Measured with `memo eval recall --config L --force --against
  origin/master`: precision@5 0.550 vs 0.370, noise@5 0.000 unchanged.
- **The nightly tuner could not fail.** `dream_tune`'s label sets dropped
  `relevant_terms`, so 34 of the 46 curated prompts could never score as
  relevant while still counting in the denominator: gate precision was
  pinned near 0.015 and every candidate auto-applied — which is how
  `MEMO_RECALL_MIN_SIM=0.8033` reached the live overlay. `_regressed` now
  also rejects a drop in `recall@k`/`canonical_hit_at_k` (the shipped
  receipt halved the latter while the gate said "ok"), and
  `estimate_noise_floor` calibrates on query-side encodings — the
  distribution the floor is actually compared against.
- **Data loss on the canonical Markdown.** `update(append=)` overwrote the
  `.md` with just the appended fragment when the file was unreadable and
  `fts.body` was NULL (the version snapshot recorded an empty body too, so
  rollback could not recover it); a metadata-only `update()` re-applied
  `max_content_chars` to a body it had not been asked to change, truncating
  notes hand-extended in Obsidian; absorb-on-recurrence committed a merge
  clipped by its 1024-token budget over a long note, and could fold a
  `project:B` save into a `project:A` memory. `gc(fix=True)` — run
  unattended by `sync_pull` — deleted index rows when an unmounted volume
  or an iCloud-evicted vault merely made existence unverifiable.
- **Silent partial results.** `reindex()` counted parse/embed failures as
  benign skips (new `errors` counter, non-zero CLI exit); hybrid search
  returned BM25-only results without marking `degraded` when the embedder
  raised; an absorbed save returned no `action`; `list()` filtered forgotten
  rows after the SQL LIMIT and returned a short page; a failed tantivy write
  left the on-disk index stale for every later process; the proactive
  refresh aborted on a duplicate nudge id and kept the previous night's
  candidates; a rolled-back hard delete re-opened a superseded fact.
- **MCP surface.** `memo_save` over the response budget returned
  `response_budget_exceeded` with no id although the record had committed
  (writes now return a reduced payload keeping `id`/`path`/`action`);
  `memo_history(limit=-1)` materialized the whole events table (SQLite reads
  a negative LIMIT as unbounded) — limits are clamped at the tool boundary;
  the write coordinator masked caller-input errors as storage failures;
  `memo_delete` skipped its irreversible-delete confirmation whenever the
  pre-read raised; `memo_event_bus_publish` always reported success and
  `memo_event_poll` re-delivered every event; the shipped mandate and server
  instructions named tools absent from the `agent`/`core` profiles.
- **Ops and runtime.** `watcher.render_plist` interpolated MEMO_* values
  into launchd XML unescaped (ProgramArguments injection); both plist
  renderers froze the installing terminal's `MEMO_AGENT_TTY` and copied
  `MEMO_HTTP_API_TOKEN` into a world-readable file (now filtered, plists
  written 0600); `MLXChat.chat()` held the machine-wide GPU flock for a
  whole generation, starving recall (per-token guard now), and
  `_ensure_model` raced unload/eviction into a KeyError;
  `memo-nightly.sh` stamped its 20h due-guard before running, so a killed
  run burned the day (stamp moved to the end, portable mkdir lock, log
  rotation added, dream moved to 03:30); sidecar sqlite stores bounded
  their WAL (graph.db-wal had reached 80MB); the two process-local caches
  are locked; `capture-tick`'s `no_pair` path never advanced the throttle;
  `MEMO_AUTO_UPDATE_REPO` is restricted to https.
- **Test isolation.** `MEMO_SAVE_ABSORB` and `MEMO_AUTO_PROJECT_TAG` are
  hard-pinned process-wide: with absorb on, the suite's stub embedders made
  every pair a near-duplicate and fired real MLX generations on Apple
  Silicon — 5 tests failed there while Linux CI stayed green.
- **Documentation drift**, including a recall-hook wiring pointer that named
  the wrong file, mixin/type/flag/route counts, `/memo` router subcommands
  that do not exist, and the Docker tag list.

## [4.12.2] - 2026-08-17

### Fixed

- Findings from an adversarial review of the (dark) emission ledger that
  touch live surfaces: `MEMO_EMITTED_LEDGER` now declares its gate in the
  dark-flag graduation contract (a dark bool without the `_ENABLED` suffix,
  it escaped the automatic completeness check); `MEMO_SESSION_ID` is no
  longer flagged as a typo by `memo config validate` (env-only per-session
  identity); the `memory_with_memories` test fixture pins
  `MEMO_SAVE_ABSORB=0` so its two cosine-identical seeds stop firing a real
  LLM absorb per test setup (integration module: ~436s → ~17s on Apple
  Silicon, and the two-hit premise can no longer be merged away). The
  emission-ledger spec's invalidation table now records that the promised
  `/clear` → SessionStart reset was never implemented, and lists the
  confirmed gaps as explicit promotion blockers.
- Homebrew reference formula (`docs/homebrew/mlx-memo.rb`) pinned to the
  published 4.12.1 source distribution (was 4.11.3).

## [4.12.1] - 2026-08-17

### Fixed

- Docker image builds again (`docker-publish` had been red since 4.11.2):
  the wheel force-includes `eval/regression_labels.json`, but the Dockerfiles'
  explicit COPY allowlist never copied `eval/` into the build context, so the
  in-image `uv build --wheel` failed with `Forced include not found`. Both
  `Dockerfile` and `Dockerfile.glama` now `COPY eval ./eval`, and the
  allowlist-style `.dockerignore` re-includes `eval/` so the directory
  actually reaches the build context (without that, the new COPY line would
  fail with `"/eval": not found`). This patch release exists to re-run the
  tag-gated `docker-publish` workflow from a ref that contains the fix —
  there are no code changes beyond the Docker build surface.

## [4.12.0] - 2026-08-17

### Added

- `MEMO_RECALL_CHUNK_PARENT` (default off): closes a gap where a chunked
  long durable memory was invisible to auto-recall — its fragments
  (type=reference) were excluded by `MEMO_RECALL_EXCLUDE_REFERENCE` before
  the search pipeline's chunk->parent rollup ever saw them. Runs one small,
  bounded, type-scoped search to resolve a winning chunk back to its
  canonical parent. Covers the recall-hook subprocess; the warm daemon path
  does not call this yet, a named follow-up.

### Fixed

- Consolidation clustering (`memo consolidate`) no longer splits above-threshold
  near-duplicates across different proposed clusters. `_greedy_cluster` compared
  each new memory only to each existing cluster's FIRST member (frozen forever
  as its "representative"), never to members added afterwards — so two
  near-duplicates could land in different clusters purely because of pull
  order. Measured on the live corpus: 38.4% of above-threshold pairs (861 of
  1450) split this way. Single-linkage (connected components of the threshold
  graph) was tried and rejected as the fix: it transitively chains anything
  reachable through a path of individually-strong pairs, and on the live
  corpus one (project, type) bucket alone chained 157 of its 950 memories into
  one unmergeable blob. Merge-consolidation (`_cluster_within_scope`) now uses
  average-link (UPGMA) agglomerative clustering instead: two clusters merge
  only when the AVERAGE similarity across every cross-pair clears the
  threshold, which fixes the greedy split (previously-split near-duplicates
  now co-cluster) without single-linkage's chaining (max cluster size stays
  bounded — 6 vs. 157 on the same corpus). Hand-checked purity on real
  title+body pairs roughly doubled (~30% -> ~55-60% correct near-duplicates)
  versus greedy's proposals. The merge itself uses Lance-Williams
  average-linkage updates (O(k³) at C speed): the naive
  recompute-every-block-mean formulation was O(k⁴) and turned one
  `memo_consolidate` call over a dense 500-member component into ~2h of
  GIL-holding work (caught by the corpus-scale conformance suite).
  `synthesize_cross_cluster` and `dream_distill.run_distill` (read-only
  insight generation, not merges) still use the original `_greedy_cluster`.
- Two bugs in `MEMO_SAVE_ABSORB` found by adversarial review right after it
  shipped default-on:
  - The absorb target's LLM merge call (up to `MEMO_CONSOLIDATE_TIMEOUT`,
    default 180s) runs unlocked. If a concurrent nightly consolidation merge
    archives+deletes the same target while the call is in flight, `update()`
    silently resolved to `None` with zero logging and the caller fell
    through to creating a brand-new near-duplicate record — undoing the
    consolidation that just ran, invisibly. Now logs a warning naming the
    likely cause, so the outcome is observable instead of a silent mystery.
  - Absorb had no type-match check: a near-duplicate of a *different* type
    (e.g. a `note` scoring >=0.88 against an existing `fact`) would still
    rewrite the existing record's body via LLM merge while keeping its
    original type label, silently blending cross-type content. Absorb now
    only triggers on a same-type match; a type mismatch falls through to
    the ordinary warn-and-create path.

### Changed

- `MEMO_SAVE_ABSORB` defaults ON: a near-duplicate save now rewrites the
  existing record (versioned, rollbackable) instead of creating a near-copy.
  Measured 2026-08-16: absorbs correctly at cosine 0.9691/matching type,
  ~24s per absorption (one bounded LLM call). Opt out with `=0`.
- **`MEMO_SAVE_DEDUP_THRESHOLD` default lowered 0.88 → 0.85.** The save-time
  near-duplicate cosine floor was measured against the live corpus in the
  *real* regime the check actually runs in — `embed_query` on the new
  candidate vs. the stored document embedding of each existing memory, not
  symmetric document-document cosine, which scores meaningfully higher and
  was masking the true precision of this band. A census of every real-regime
  candidate whose top hit fell in the newly-caught [0.85, 0.88) window (28
  pairs, not a sample — the whole population out of a 2,012-candidate scan)
  came out 71% genuine duplicates vs. 18% distinct-fact false-positive risk,
  dominated by same-session cross-language (es/en) and refined restatements;
  the two pairs already caught at 0.88 today were both genuine duplicates
  too. The event stays rare corpus-wide (~1.5% of candidates cross 0.85).
  Separately (not changed by this PR): the save-time dedup search has no
  type or project filter, so a near-duplicate hit — and, if
  `MEMO_SAVE_ABSORB` is enabled, an absorb — can in principle match across
  memory types or projects; this pre-exists the threshold at 0.88 too.

## [4.11.3] - 2026-08-16

### Fixed

- An `unload()` landing between `_ensure_loaded()` and the dereference could
  null the model or tokenizer mid-embed, surfacing as an `AttributeError` on
  `None` instead of a clean failure. Both embedders now snapshot the pair under
  the same lock that loads it, so an in-flight embed always runs against a
  consistent model/tokenizer or raises a named error (#256).

## [4.11.2] - 2026-08-15

### Fixed

- A NULL title or body no longer downgrades the BM25 backend. `_fold_diacritics`
  called `unicodedata.normalize` on whatever the caller passed; a nullable
  column reaching it raised `normalize() argument 2 must be str, not None`
  inside `add_document`, and the caller's except-branch then ran
  `_mark_tantivy_unhealthy()` — so one bad row silently downgraded BM25 to FTS5
  for the rest of the process. Fixed at the boundary rather than at the four
  `add_document` call sites in `queries.py`, since patching one would have left
  the other three. Only reachable with the `tantivy` extra installed.

## [4.11.1] - 2026-08-15

### Fixed

- `memo eval memory --gate` compared its own numbers against a baseline written
  by `memo eval recall`. The two measure different pipelines and shared one
  file; only `eval recall` ever wrote it, so the defect was on the read side,
  and `check_gate`'s k guard caught it only when the two happened to run at a
  different top-K. Each command now owns its baseline file, the payload is
  stamped with the command that wrote it, and a mismatched one is refused.
  `eval recall` keeps the historical filename so no machine has to re-seed, and
  an unstamped baseline counts as recall's because before the split only that
  command could produce one. `eval memory` also gains `--update-baseline` —
  refusing the foreign baseline without giving it a way to seed its own would
  have left its gate with no remedy at all.

## [4.11.0] - 2026-08-15

### Added

- `memo eval behavior` — measures whether a recalled memory actually *steers*
  the answer, the step after `memo eval recall`. Seeds an isolated store, runs
  the real `memo recall-hook` subprocess against it, and scores the answer a
  model gives with that injected block. A scenario whose gates are all
  recall-layer is rejected at load: that is retrieval, already covered.
- `L live/...` eval config — the configuration the recall hook actually runs.
  It pins nothing it can inherit, and its name carries the resolved mode/floor
  so a tuner moving the floor underneath the gate is visible. Included in the
  `pre-push` profile, which is the profile the blocking gate runs.
- Adapter drift checks folded into `memo release check`: hook commands still
  resolving in the CLI, `.mcp.json` embedder dims still matching their model
  (MLX invariant 3), and manifest paths still existing. Each has a test that
  mutates one surface and asserts the check flips to fail.

### Fixed

- The recall gate scored configurations nobody runs. Measured on the curated
  44-prompt set at k=5: the grid reported precision@5 0.568-0.716 and stayed
  green while the live hook was at 0.205 / recall@5 0.20, because the tuner's
  overlay had moved `MEMO_RECALL_MIN_SIM` to 0.8835 — a floor no grid config
  uses.
- The nightly tuner scored a pipeline it does not apply. All four measurement
  sites ran with the hook's post-rank injection filters OFF and then applied
  the winner into a hook that has them ON (measured: precision@5 0.363 vs
  0.205, a 44% gap the search was blind to). Since the tuner auto-applies, that
  was a mechanism for shipping regressions, not just a measurement error.
- `gc` could delete a row whose path it was never able to verify. A
  `StorageError` from `_resolve_existing` is a path-*safety* refusal (a symlink
  component, a path escaping `memory_dir`), not evidence the `.md` is gone. It
  ran unattended on every `sync_pull`, logged nothing, and `_reindex_locked`
  refuses the same paths — so the row never came back and the memory left
  search, recall and list permanently.
- `turn_store`'s BM25 query swallowed `sqlite3.OperationalError` into an empty
  list with no logger in the module, making a corrupt verbatim FTS index
  indistinguishable from "nothing matched" forever.

## [4.10.2] - 2026-08-15

### Added

- **`memo consolidate restore` — merges are reversible now.** Consolidation
  archives the memories it absorbs, and that was a one-way door: `_archive_memory`
  stripped the frontmatter `id` before writing to `archived/`, so `reindex`
  (which requires a canonical id) could never adopt the file again even if a
  human moved it back. Recovering a single wrongly-merged record meant
  hand-editing YAML and re-injecting its id. The archived copy now keeps its
  `id` and records the `archived_from` path it used to occupy, and
  `memo consolidate restore <id>…` / `--for <merged-id>` moves memories back
  into the live corpus — into their original path when it is still free,
  otherwise a freshly allocated one. Legacy archives written without an id are
  recovered from the filename. Restoring is additive: `--drop-merged` also
  deletes the merged record, and refuses unless that record carries the new
  `consolidated_from` provenance proving the merge created it, so a
  `keep_latest` survivor — a pre-existing memory, not a merge artifact — can
  never be deleted by an undo. `maintain`'s compaction has had the symmetric
  `_restore_archived` since it shipped; consolidation had nothing.

  Restoring is also honest about the index: `(namespace, topic_key)` is UNIQUE
  across live rows, and archiving a record soft-deletes it — which *frees* its
  topic reservation for a later save to claim. Restoring such a record
  verbatim made reindex's `deleted_at=NULL` un-delete violate that index, and
  reindex reports a per-file failure as a log warning and carries on: the `.md`
  landed in the live tree permanently unindexed while the restore claimed
  success. The record coming out of the archive is by definition the superseded
  one, so it now gives the slot up instead of colliding for it. As a general
  net, every restored id is re-checked after the reindex; anything the index
  would not adopt is reported in `unindexed_ids` with the `memo reindex
  --rebuild` remedy, never counted as restored.

### Fixed

- **A disk-only topic-reservation recovery could resurrect an archived memory.**
  `_recover_topic_reservation_locked` walks every `.md` under `memory_dir` and
  re-indexes the one holding a claimed `(namespace, topic_key)`. It never
  skipped `inactive/` or `archived/` — and `inactive/` has always kept its
  canonical id — so a compacted memory could be silently pulled back into the
  index behind the user's back. It now skips `LIFECYCLE_ARCHIVE_DIRS`, the same
  two directories `reindex` and the disk-orphan gc already refuse to absorb.
  This is what makes preserving the `id` on the archived copy safe.

## [4.10.1] - 2026-08-14

### Fixed

- **Consolidation proposed merges the write path then refused.** Clusters were
  built on cosine alone, but `apply_merge` saves the merged record with the
  *union* of its members' tags and `identity.namespace_for_write` rejects a
  union carrying more than one `project:` slug. Every cluster that spanned two
  projects was proposed, attempted, and lost to `memory identity conflict:
  ambiguous_namespace` — measured on a 6,135-memory corpus at the default
  threshold, **14 of 15 clusters died that way**, so the nightly pass looked
  like it ran while merging 9 memories instead of 138. Clustering now runs
  independently inside each project scope (`identity.cluster_scope`), which
  keeps every proposal writable *and* leaves project attribution untouched: a
  memory is only ever merged with memories of its own project, and untagged
  memories stay untagged rather than being absorbed into one. A record
  carrying two `project:` tags of its own is left out entirely — no partner
  makes its union writable.
- **Consolidation silently retyped what it merged.** The merged record takes
  the type of its newest member, so a cross-type cluster retypes everyone
  else — and type decides which surface a memory appears on:
  `failure_pattern` feeds the recall hook's ⛔ AVOID block, `procedure` feeds
  procedure promotion, `preference`/`feedback` get their own recall boost
  tier. One live pass merged `decision`, `procedure`, `failure_pattern`,
  `preference` and `note` into a single `decision`; across that pass 55 of 66
  archived records lost their type, and the committed retrieval label set
  caught it as an AVOID probe that stopped resolving (avoid@k 1.000 → 0.500).
  Clustering is now partitioned by type as well as by project scope: records
  that are near-identical in wording but differ in type are not duplicates,
  they are the same topic seen through different lenses. Measured on the same
  corpus, this trades 106 merged memories per pass for 79 — and 0 retyped
  instead of 57.

## [4.10.0] - 2026-08-13

### Added

- **A response budget on every MCP tool result.** A `limit` argument bounds an
  internal loop; it does not bound what the caller receives — `memo_graph
  verb=impact` returned 27.5k tokens *with* `limit=3`. Any tool result whose
  estimated size exceeds `MEMO_MCP_RESPONSE_BUDGET_TOKENS` (new flag, default
  10,000; `0` disables) is now replaced by a small structured
  `{"error": "response_budget_exceeded", "tool", "tokens", "cap", "hint"}`
  payload. It comes back as a normal successful call, not a protocol error, so
  callers read it the way they already read every other `{"error": ...}`
  refusal on this surface. Results are never truncated by this layer —
  truncating an arbitrary payload silently corrupts its contract; a tool that
  knows which of its fields is elastic trims that field itself and says so.
  The estimate covers both fields the wire carries (content blocks *and*
  structured content), which for a dict-returning tool is the same JSON twice.

- **An emission ledger, so memo stops re-sending bodies already in the window.**
  memo produces the same memory body from two places — the recall hook on every
  `UserPromptSubmit` and the MCP read tools — and nothing tracked emissions
  across them, so one body could enter a session's context three or four times.
  `MEMO_EMITTED_LEDGER` (new flag, **default `0`**) records what was emitted
  per session and replaces a repeat with `{id, title, ref}`, which the model
  expands via `memo_get` when it actually needs the text. Consulted by
  `memo_search`, `memo_ask` and `memo_evidence_pack`
  (`MEMO_EMITTED_LEDGER_TOOLS`); `memo_context` and `memo_unified_briefing` are
  deliberately excluded, not half-wired — neither has a hit list this mechanism
  can suppress. Writes are deferred until the response-budget middleware
  commits, so a truncated response never records an emission the model never
  saw, and the ledger is cleared at the PreCompact boundary because compaction
  invalidates every claim about what is in the window. Counters
  (`digests_served`, `memo_get_after_digest`, `net_saved_est`) surface in
  `memo_cache_stats`; stale ledgers are pruned by the nightly pass.

  Measured on a replayed real transcript: **31.4% and 36.6% fewer emitted
  tokens** across two clean runs, recall-hook p95 latency delta negative on the
  warm-daemon path, and zero delta on the retrieval eval. It stays default-off
  because promotion also requires a `memo_get_after_digest` rate that a replay
  cannot produce — there is no model in the loop deciding whether to recover a
  digested id. An unmeasured criterion is not a passed one. See
  `docs/SPECS/2026-08-10-emission-ledger-design.md` and
  `docs/eval/emission-ledger-replay.md`.

### Changed

MCP payloads that grew with the corpus instead of with the request. Each tool
below now trims its elastic field and reports what was really there. The
library and CLI paths are unchanged — they still return everything; only the
MCP surface, the one with a token budget, is trimmed.

- **`memo_consolidate`** — `max_clusters` default 20 → 10, and a new
  `member_limit` (default 2) bounds a cluster's sample members; each
  proposal's `memory_ids`/`archived_ids` and the `proposals` list are bounded
  too. Measured 74,155 tokens before, ~3,980 after, flat from 2,000 to 10,001
  memories. **The CLI keeps its old `--max-clusters 20`** and full member
  detail on purpose: it writes to a terminal, not into a model's context, so
  it has no token budget to blow. Note that passing `max_clusters=20,
  member_limit=20` back to the MCP tool does **not** restore the old response
  — at 20 × 20 the result is ~47k tokens and the response budget refuses it.
  Raise either argument as far as the budget allows; `total` always reports a
  cluster's real size and `memo_get` fetches any member by id.
- **`memo_synthesize_run`** — each result's `sources` list is capped at 20,
  with `shown`/`total`/`truncated` alongside. Measured 44,043 tokens before,
  2,143 after. The saved synthesis memory's `synthesis_sources` frontmatter
  still records every source.
- **`memo_graph_communities`** — new `limit` (default 20, largest first), and
  each community's `entities` list is capped at 50 with a per-community
  `entities_truncated`. `size` still reports the community's true entity
  count. The tool still returns a **list**, unchanged.
- **`memo_entity`** — new `limit` (default 200) on the memory ids returned.
  The query behind it has no `LIMIT`, so a hub entity's id list tracked the
  corpus: 12,429 tokens for a single entity with 700 mentions.
- **`memo_temporal_timeline`** — new `limit` (default 30; the most recent
  events are kept, still in chronological order) plus `event_count` and
  `truncated`. It emits one event per mention, each with a 200-char snippet:
  110,326 tokens for an entity with 700 mentions. `first_seen`/`last_seen`
  still span the whole timeline.

### Fixed

- **Reverted the metadata-boost ranking change (#223): it cost 22% of
  precision@5.** Measured A/B on the live corpus with the curated regression
  set (43 prompts, `--profile pre-push`, both sides uncached): **precision@5
  0.524 with it, 0.676 without**, noise@5 unchanged at 0.000. The scale
  argument behind it — hybrid's `h.score` is RRF-fused, so a cosine-calibrated
  floor cannot be compared against it — may be right on its own, but it shipped
  together with `_MAX_BOOST` 12.0 → 1.5 and a wholesale collapse of the
  curatorial boost weights, which made the regression unattributable. Either
  half can be re-landed separately, each measured against the label set.

- **A sub-threshold contradiction froze writes forever.** Every detected
  `semantic_contradiction` opened a conflict with `freeze_write=True`, but the
  only automatic passes that adjudicate one — `memo maintain --confidence`
  (default 0.9) and dream's contradictions pass — skip anything below 0.9. A
  0.80–0.85 pair therefore froze writes to its subject memories permanently:
  an agent cannot lift it (`write_policy` requires authenticated human
  authority), and no nightly ever would. Freeze only what the configuration
  says is actionable; an anomaly with no readable confidence keeps the
  conservative freeze.

- **The nightly's consolidate pass had never run.** `consolidate apply
  --force` still raises the interactive confirmation — `--force` is gated on
  `--yes` — and a LaunchAgent has no stdin, so the last pass of every nightly
  aborted with exit 1. The same script also still synced `memflow`, archived on
  the 2026-07-30 trinity deprecation.

- **Neither the nightly nor dream fired on a machine that sleeps at 03:00.**
  macOS does not wake for a `StartCalendarInterval` and does not reliably
  replay a slot it slept through, so memo's whole self-maintenance story could
  silently not run for days. Both agents now pair the 03:00 slot with an hourly
  `StartInterval` and an idempotent entrypoint: dream runs `memo dream if-due`
  (which already existed for this and was simply never wired), and the nightly
  script carries a matching stamp guard (`MEMO_NIGHTLY_MIN_INTERVAL_H`, default
  20h; `--force` bypasses).

- **A NULL FTS body silently downgraded the BM25 backend.** `update_meta` read
  `fts.body` — NULL for thousands of live rows — straight into tantivy's
  `add_document`, which rejects a non-str field. The error was caught, but the
  handler marks the whole backend unhealthy, so one such row downgraded BM25
  from tantivy to FTS5 for the rest of the process.

- **`memo eval recall --against <ref>` could not run from an installed
  memo.** It resolved the repo root from `Path(__file__)`, which under the
  isolated uv-tool install is site-packages — outside any git worktree — so it
  failed with `fatal: not a git repository` on precisely the install the
  pre-push gate's own failure message tells you to run it from. It now anchors
  on the invocation cwd.

## [4.9.3] - 2026-08-05

Found by a live health sweep of the installed runtime, not by the test suite.

### Fixed

- **memo answered "I couldn't find an answer" to questions it had answers
  for.** On the live corpus `memo search "cómo me conecto a la VPN de
  avature"` put the procedure at rank 2 while `memo ask` refused outright.
  Two independent causes on the answer path (`ask`, `chat ask`, and the
  `memo_ask` / `memo_chat_ask` MCP tools): the path defaulted to
  `disable_reranker=True` — "RRF is sufficient for synthesis" holds on a small
  corpus, not a real one — and `load_bodies=False`, an I/O optimisation,
  silently changed the ranking by handing the cross-encoder empty bodies, so
  it scored titles alone and the same document fell from rank 2 to rank 21.
- **Chunk→parent collapsing shrank results instead of refilling them.** A
  query could return eight hits that were eight chunks of one note; enabling
  `MEMO_SEARCH_CHUNK_PARENT` turned those eight into one rather than one plus
  the next seven distinct documents. Collapsing now runs on the wide pool
  before rerank and the trim. Still default-off.
- **`memo watch` flooded its log.** A memory the nightly GC or a merge deleted
  mid-scan was reported as `reindex: skipping <name> (parse error)` at warning
  level on every reindex — 15 MB of it on this machine. That race now logs at
  debug; real parse errors keep their warning and still abort a `--rebuild`.
- **Release commands rewrote the wrong checkout.** `memo release bump` run
  from an isolated worktree — the procedure that exists to keep releases out
  of the shared working tree — resolved the repo from the imported module and
  bumped the shared tree instead. Resolution order is now `MEMO_DEV_REPO` >
  the checkout containing the cwd > the module's own checkout.

## [4.9.2] - 2026-08-05

Nine defects found by running memo as an end user across the whole CLI (349
commands) and MCP (41 tools) surface, each fixed with a regression test.

### Fixed

- Operational conflicts are closed when their relation is judged. The
  contradiction scanner only ever emitted the `open` anomaly; nothing emitted
  `resolved` once the pair was judged in the canonical relation ledger, so
  conflicts accumulated forever and the SessionStart briefing listed settled
  ones as open. `Memory.judge_relation` — the chokepoint every judge path goes
  through — now closes the pair's conflict.
- `memo_operational_state` and `memo operational state` return only what is
  still open. They shipped the whole projection including resolved conflicts,
  consumed handoffs and acknowledged attention items, which grows without
  bound and could exceed an MCP client's token budget (91 KB on a real
  corpus). Pass `include_closed` / `--include-closed` for the full history.
- Memory content renders as data, not as Rich markup. A body containing
  `[#42]`, `array[index]` or `[bold]` printed with those tokens swallowed or
  applied as styling, and `memo context` dropped every citation id starting
  with a letter — roughly a third of them. The markdown on disk was always
  correct; only the rendered view was wrong.
- `memo operational signal remember` works on installs whose snapshot predates
  the `signals` section. Both snapshot readers returned the persisted document
  verbatim while the schema string and journal heads matched, so the writer
  raised `KeyError: 'signals'` — surfaced through MCP as the unactionable
  "coordinated MCP write failed safely". Missing sections are now backfilled,
  and the coordinator's mask names the failing exception type.
- The nightly tuner's curated no-regression gate no longer fails open on an
  installed runtime. The curated regression labels were resolved through path
  arithmetic that only lands on a repo root from `src/memo/`, so no installed
  runtime ever found them and the gate silently approved every candidate. The
  labels now ship in the wheel and sdist.
- `memo edit` offers the same edit shapes as the `memo_update` MCP tool:
  `--append` and a surgical `--replace-old` / `--replace-new` pair, not just a
  full-body `--content` replace.
- `--limit` and `--k` reject out-of-range values (the 1..500 bound the MCP
  tools already clamped to) instead of silently printing an empty table, and
  `memo ask ""` fails with a clear message instead of an empty answer panel.
- Renaming a `fact` stops it asserting its old title. The coarse
  `memory asserts <title>` edge was only ever written by the save paths, so
  the graph and the briefing kept the pre-rename title indefinitely. The stale
  assertion is invalidated (not deleted — the edges are bi-temporal) and one
  for the current title is opened.

## [4.9.1] - 2026-08-04

### Fixed

- Topic-scoped conflicts no longer freeze unrelated writes. A manually-opened
  conflict matched whenever any 3+ character token of the incoming write was a
  *substring* of the conflict topic, so a conflict on `test_conflict` refused
  every durable write whose topic contained the word "test". Matching now
  requires whole-token containment of the conflict topic, and a topic with no
  significant token freezes nothing while staying resolvable by id.
- `memo maintain` reports pass failures instead of hiding them. A refused
  synthesis save was logged at warning level and dropped while the command
  printed a success banner and exited 0. The failure is now recorded on the
  cluster result, folded into `receipt["errors"]` before the receipt is
  persisted, and the run exits non-zero. A dry run still exits 0 — it changed
  nothing.
- A missing `crush_cache` directory is no longer recorded as a maintain error.
  It is the normal state of a fresh install, and with the new exit code it would
  have made every first run report failure.

### Added

- `memo operational conflict list [--all]` lists conflicts, newest first,
  showing only write-freezing ones by default. `conflict resolve` previously
  required an id that no CLI command could produce.

## [4.9.0] - 2026-08-04

### Added

- Graph-aware source compaction for chat/ask: chat sources that share a rare,
  IDF-weighted entity are collapsed into one representative source plus a
  `related_ids` pointer, cutting synthesis-prompt token cost. IDF weighting
  avoids the ubiquitous-entity landmine hit by a prior graph-injection
  experiment (common entities no longer force unrelated sources together).
  Gated by new default-off `MEMO_CHAT_GRAPH_COMPACT` /
  `MEMO_CHAT_GRAPH_COMPACT_MIN_IDF` env vars. (#186)

### Changed

- Bumped `aiohttp` 3.14.1 -> 3.14.3 and `cryptography` 49.0.0 -> 50.0.0 to
  patched versions (CVE fixes).

## [4.8.1] - 2026-08-03

### Fixed

- The MCP write-coordinator no longer masks legitimate business-rule
  validation errors (e.g. `memo_procedure_promote` rejecting a memory that
  lacks sufficient outcome evidence) behind a generic "coordinated MCP write
  failed safely" message. Root cause: FastMCP's own tool dispatch wraps any
  exception a tool raises into its own `ToolError` (chained via `__cause__`)
  before the coordinator's exception-type check ever sees it, so the
  existing `except MemoError` branch could never match in production — only
  the generic mask branch could. The coordinator's generic handler now
  inspects `e.__cause__` and, when it is a `MemoError`, propagates that
  original error and message instead of the opaque mask; truly
  unknown/unexpected exceptions are still masked as before. Found via live
  MCP tool-by-tool testing during a production-readiness audit.

## [4.8.0] - 2026-08-02

### Added

- An explicit opt-in receiver-bound terminal transport (`memo terminal
  receiver ...` and matching MCP tools) now owns a nested PTY, authenticates
  Unix-socket peers with a mode-0600 capability, revalidates the child identity
  before every write, and deduplicates message IDs. Legacy exact-TTY mutators
  remain disabled by default.
- `memo events list --cursor` now exposes bounded, resumable event pages from
  the append-only JSONL journal, with an opaque cursor backed by a verified byte
  offset, while the no-cursor command preserves its legacy JSON list contract.
  `memo chat-session list --cursor` similarly supports stable
  session-id pagination so consumers can drain histories larger than 1,000
  records without replaying earlier pages.
- MCP saves support deferred embedding over stdio, preserving fast writes while
  keeping the markdown source of truth and later index convergence intact.
- Read-only live-terminal inventory and receipt diagnostics are available over
  CLI and MCP. Unsafe input delivery remains fail-closed and is not exposed.

### Fixed

- The archived Memflow and Synapse agent-integration state now migrates fully
  into memo, including exact, idempotent chat feedback signals, without leaving
  a second runtime or database in service.
- Runtime provenance, SQLite and audit resource cleanup, graph projection
  teardown, bounded GPU-lock waits, and local Docker builds are hardened for
  production and CI isolation.
- Vault re-ingest now self-heals missing legacy FTS bodies even when the source
  hash is unchanged, and the integrity auditor validates Markdown, PDF, image,
  and audio references without misclassifying external media paths.
- Tantivy uses short, cross-process writer leases and refreshes long-lived
  readers per query, so MCP, CLI, and daemons can update one index without lock
  starvation or stale search snapshots.
- Chat HTTP/SSE boundaries, destructive session resets, rejection quotas, and
  loopback enforcement are safe under concurrent requests.
- `memo config validate` accepts the retired live-terminal shim variables in
  agent processes that were already running when the fail-closed upgrade was
  installed; fresh shims still remove those variables.
- Live terminal input now fails closed: the CLI and MCP legacy `send`/`enter`
  mutators and automatic shim registration are disabled, and legacy terminal
  registrations are non-deliverable. Read-only diagnostics remain available;
  the receiver-bound API is explicit opt-in only.
- Unix socket creation now falls back to a short per-user temporary root when
  the configured state path would exceed the platform AF_UNIX path limit.
- Live terminal coordination now recognizes the native `Apple_Terminal` and
  `iTerm.app` `TERM_PROGRAM` values, rotates a registration id when a new
  process replaces an agent on the same TTY, and records target-validation
  failures instead of leaving them outside receipt history.
- macOS terminal presentation keeps message bodies out of process arguments,
  converts AppleScript timeouts into failed receipts, submits Ghostty input to
  the exact terminal without a global-focus race, and removes unsafe
  Terminal.app delivery entirely. A partially successful `TIOCSTI`
  injection is never replayed through a fallback transport.
- The fail-closed internal terminal bridge preserves safe transport diagnostics
  in receipts while redacting unexpected exception details; no delivery mutator
  is exposed through CLI or MCP.
- `memo ask`, `memo diff`, `memo entity`, `memo invalidate` (preview), `memo
  temporal timeline`, and `memo synthesize` no longer silently drop `[id]`
  citations, memory ids, or titles from their plain-text output: Rich's
  console markup parser was swallowing any bracketed substring it couldn't
  resolve as a style tag, most visibly ids starting with a hex letter (`a`-`f`).
  `--json` output was always correct.
- `memo events ingest` returns a clean CLI error instead of an unhandled
  traceback on invalid input (missing `event_id`, invalid `kind`, or a
  duplicate `event_id` with a conflicting payload).
- `memo chat-session get`/`list` now surface sessions created through `memo
  chat serve`'s HTTP/SSE API, which previously lived in a separate,
  unconnected per-session store invisible to the CLI.
- `memo related`'s `via` attribution is now deterministic across repeated runs
  of the same query; tied candidate scores no longer depend on set/dict
  iteration order.

### Changed

- `memo stats --json` now emits `memo.stats.v2`: bounded context activity is
  reported honestly as `context_tokens_injected` and `memories_surfaced`,
  leaving estimated savings exclusively to `memo roi`. Recall dashboards and
  CLIs now name the boosted final ranking value a composite score instead of
  presenting it as semantic confidence; the ambiguous v1 score fields remain
  compatibility aliases.

## [4.7.0] - 2026-07-31

### Added

- **Native chat UI over your memory** (`memo chat serve`, rescued from the archived
  synapse chat): retrieval pipeline (`memo.chat`) with RRF fusion, per-group score
  normalization, chunk dedup, rules-based follow-up rewrite, gated multi-query
  expansion, 👍/👎 feedback with exact + semantic vote boosts, relevance floor,
  and fulldoc inline — streamed as SSE with a vendored React SPA (`web-chat/`).
- `memo eval chat` — regression gate over the rescued chat corpus
  (`eval/chat_regression_corpus.json`).
- `memo ops install|uninstall|status` — launchd lifecycle for the `com.memo.chat`
  agent (env-forwarding plist, validated `--dist`).
- One-off migration of synapse chat feedback signals
  (`scripts/migrate_synapse_chat_state.py`).

### Fixed

- `memo config validate` no longer flags the documented `MEMO_CHAT_*` knobs as typos.

## [4.6.2] - 2026-07-31

### Fixed

- MCP registry publish works again: the `mcp-name: io.github.jagoff/memo`
  ownership marker the registry requires in the PyPI README was dropped by
  the #130 rewrite, so the v4.6.1 registry step failed validation. The marker
  is restored and now pinned to `server.json` by a supply-chain test (#147).

## [4.6.1] - 2026-07-31

### Fixed

- PyPI project page renders again: README images and doc links use absolute
  GitHub URLs, restoring the a44890c9 fix that the conversion README dropped —
  the v4.6.0 page shipped with a broken banner because of it (#145).
- README banner served as WebP at full 1600 px resolution (152 KB, down from
  the original 372 KB JPEG); `banner.jpg` stays for older PyPI pages (#140,
  #145).

### Added

- Docker image now publishes a `linux/arm64` manifest alongside `linux/amd64`:
  the README one-liner works natively on Apple Silicon and ARM Linux instead
  of failing with `no matching manifest` (#144).

### Changed

- Dependency bumps: python-runtime group, GitHub Actions group, Python base
  image digest, and `@types/node` (#141, #86, #58, #125).

## [4.6.0] - 2026-07-31

### Added

- Graph-backed repository intelligence: repo search and context packs are now
  enriched with the CodeGraph structure layer (#134).
- Synapse/memflow daemons adopted natively: the `com.memo.*` launchd fleet
  (recall daemon, nightly, vault ingest, dream, watch) replaces the deprecated
  trinity stack; handoffs and operational continuity now live in memo (#136).

### Fixed

- Reindex no longer emits `invalid memory id` warnings for `_chronicle` diary
  files: the bucket is skipped silently via the reindex skip-dirs list (#138).
- Nightly launchd template now passes the correct `--max-memories` flag to
  `memo contradict scan` (#137).
- Hardened code intelligence and the dream pipeline against real-run failures
  (#135).

### Changed

- Normal MCP startup is fully offline again: remote update checks and
  background auto-update now require explicit opt-in with
  `MEMO_UPDATE_CHECK_ENABLED=1` or `MEMO_AUTO_UPDATE=1`.
- Graph-enriched repo search now uses CodeGraph's indexed identifier-segment
  vocabulary instead of scanning every symbol once per query term, ranks
  multi-term identifier matches first, and ignores generic test/repo boilerplate
  that previously crowded specific results out of the top ten. Unified fusion
  also caps each file at two chunks so one verbose file cannot consume the
  entire result window, and uses a stable candidate floor so increasing the
  requested limit does not reveal candidates that should already rank in a
  smaller window.
- Repository embeddings now default to 16 chunks per MLX batch (configurable
  with `MEMO_REPO_EMBED_BATCH`) after a real self-index run showed the previous
  batch of 64 exceeding 19 GiB.
- Repository watchers now observe a local source checkout (when the indexed URL
  is local) and accept Git branch-ref events, so a commit reliably triggers an
  incremental refresh instead of watching an otherwise idle managed clone.
- Tantivy shutdown now commits pending writes, joins background merge threads,
  and is idempotent, preventing immediate index-directory cleanup from racing
  live merge files.

### Security

- Release workflow contracts now pin all platform smoke jobs as mandatory and
  prohibit downstream publishing from bypassing skipped prerequisites.

## [4.5.0] - 2026-07-29

### Changed

- CLI `memo briefing` and MCP `memo_unified_briefing` now share one
  unified-briefing composer, so both surfaces render the same sections and
  dispute markers; dispute-aware ask gained an MLX integration test.
- Code-ref verification is unified in the new `code_intel` engine: the recall
  renderer and the dream code-drift pass now share one implementation of the
  vigente/desaparecido/no-verificado semantics (the two previous copies had
  already diverged once).

### Added — codegraph integration (two rounds)

- Project-aware codegraph loader: nearest `.codegraph/codegraph.db` discovered
  upward from cwd, `MEMO_CODEGRAPH_DB` pin for daemons/pipx installs,
  per-DB mtime-cached graphs, `MEMO_CODEGRAPH_MAX_EDGES` hot-path cap.
- `memo doctor` checks the codegraph index (freshness, WAL, counts, CLI
  version) — WARN-only, never blocks.
- `memo code-facts [--project PATH]`: mines durable architectural facts
  (call hubs, CLI surface, package dependencies) from any indexed repo into
  memories with verifiable `code_refs` provenance.
- Dream pass `code_drift` (`MEMO_DREAM_CODE_DRIFT_ENABLED`, default off):
  re-verifies memories citing code against the live index nightly; memories
  whose refs are all gone are reversibly archived, never hard-deleted.
  Optional auto-repair (`MEMO_DREAM_CODE_REPAIR_ENABLED`, default off)
  re-points a dead ref when exactly one name-similar candidate exists,
  preserving the old ref in `code_refs_history`.
- Verified code citations in recall (`MEMO_RECALL_CODE_REFS_ENABLED`, default
  off): `↳ code: path:line (vigente)` lines under recalled memories, checked
  live against the index (full and balanced formats).
- Symbol-aligned repo chunking (`MEMO_REPO_CHUNK_SYMBOL_ALIGNED`, default
  off): chunk boundaries follow codegraph symbol boundaries.
- `memo code-nudge`: after a commit, surfaces memories that cite the files
  just touched (wired into the trinity git hooks; silent when none).
- `memo code-health`: ref-status summary, dead-knowledge report (memories
  citing 0-caller symbols) and undocumented-hub report in one command.
- Briefing shows the nightly code-drift outcome (`MEMO_BRIEFING_CODE_DRIFT`,
  default on; reads the dream receipt — zero graph queries at SessionStart).
- ask-gaps flags top call hubs with no memory documenting them
  (`MEMO_GAPS_CODE_HUBS`, default on).
- Recall code-proximity boost (`MEMO_RECALL_CODE_PROXIMITY_BOOST`, default
  0.0/off): memories citing symbols near your uncommitted changes rank higher.
- `memo context-pack --code <symbol|path>`: adds a graph-neighborhood section
  (symbols ≤1 hop + memories citing them) to the pack.
- Scripts `cg_impact_gate.sh` (pre-refactor caller checklist; aborts on
  unknown symbols) and `cg_affected_tests.sh` (SQL reverse-dependency test
  selection with fail-safe FULL_SUITE fallback) + report-only CI shadow job.

### Added

- MCP elicitation confirm on the six irreversible tools (`memo_delete`,
  `memo_synthesize_delete`, `memo_backup_restore`, `memo_feedback_clear`,
  `memo_repo_delete`, `memo_cache_evict`): elicitation-capable clients get an
  in-band confirmation stating the blast radius before the operation runs;
  clients without the capability proceed unchanged (fail-open, capability
  check first). An explicit decline (not cancel) is persisted as a durable
  `type=feedback` memory — decline-as-signal, itself fail-open. Flags:
  `MEMO_ELICIT_CONFIRM` + `MEMO_ELICIT_DECLINE_SIGNAL`, both default on.
  Review-hardened: once the question has been sent the gate fails closed (a
  mid-elicit error/disconnect aborts instead of running unconfirmed); the
  HTTP daemon's `json_response` transport skips eliciting up-front (it cannot
  carry it and would deadlock); prompt fragments from untrusted titles are
  sanitized + capped; the decline signal dedupes via a stable `topic_key`
  and goes through the normal write policy; `memo_repo_delete` executes
  against the exact row the user confirmed (id-bound, no re-resolve).

- Dispute-aware ask (`MEMO_ASK_DISPUTES`, default on): retrieved memories with
  open/competing contradiction pairs are marked `⚔ disputed-by` in the ask
  context and sources (`disputed_by`), the answer is steered to present them
  as contested, and when an answer rests only on disputed evidence ask
  abstains deterministically (`abstained: "disputed"`, no extra LLM call).
  Disputed top hits no longer bypass the gate via the verbatim short-circuit.
- MCP ambient-recall parity for MCP-only clients: pinnable `briefing` and
  `recall <topic>` MCP prompts (fail-open, consult-attributed as
  `mcp-prompt`), a memory-first contract in the server instructions, and an
  install-mcp tip pointing at `--with-mandate`.

## [4.4.6] - 2026-07-28

### Added

- MCP tool schemas now describe **every** parameter on the full surface:
  `Annotated` + pydantic `Field(description=...)` on the remaining 46 tools
  (157 parameters) across all `server_*` modules — completing the pass
  v4.4.5 started on the operational tools. Descriptions are grounded in the
  delegate implementations (clamps, allowed values, id-resolution rules,
  time-travel semantics) and verified by a FastMCP client probe: zero
  undescribed parameters remain.
- Disambiguation docstrings for overlapping tools: `memo_save` vs
  `memo_offload`, and `memo_search` / `memo_context` / `memo_search_trace` /
  `memo_rerank`.

## [4.4.5] - 2026-07-28

### Fixed

- PyPI project page rendering: the README banner and diagrams used relative
  paths that PyPI cannot resolve — every image `src` and doc link is now an
  absolute GitHub URL, so the page renders the banner, diagrams, and links.
- MCP tool schemas now describe every parameter: `Annotated` + pydantic
  `Field(description=...)` on all `server_operational` tools
  (`memo_evidence_pack`, `memo_handoff_consume`, `memo_attention_ack`,
  `memo_conflict_open`, `memo_outcome_record`, `memo_procedure_promote`, …)
  and on `memo_ask` / `memo_chat_ask`. Docstrings now state side effects,
  idempotency, and abstention behavior. Addresses the external MCP-directory
  quality audit that flagged 0% parameter-description coverage on the
  lowest-scoring tools.

### Changed

- Package `description` refreshed to match the README tagline (time-travel,
  contradiction radar, automatic synthesis).

## [4.4.4] - 2026-07-27

A second full file-by-file production audit of the whole `src/memo/` tree
(v4.4.3) surfaced one critical and several high/medium issues the v4.4.1–4.4.3
remediation had missed. All fixes below; the full test suite stays green.

### Security

- Fixed a **critical** remote-code-execution vector in the git sync path —
  the identical `ext::` / `fd::` remote-helper RCE that v4.4.2 closed for
  `memo_repo_index` was still open in `memo sync clone` / `setup` / `bootstrap`,
  which passed a caller-supplied URL to `git` with no protocol allow-list.
  Sync URLs are now validated (scheme allow-list, leading-dash and `<scheme>::`
  rejection) and every sync git call runs with `GIT_ALLOW_PROTOCOL` /
  `GIT_TERMINAL_PROMPT` set. Added a regression test.
- `memo_repo_index` now rejects a git `ref` with a leading dash (checkout
  argument injection), mirroring the URL guard.
- The per-machine secret salt is now created with `O_CREAT|O_EXCL` at `0600`
  instead of `write_text()`+`chmod`, closing a world-readable TOCTOU window.
- `memo_ask` / `memo_chat_ask` MCP tools now enforce the same size/shape bounds
  (question/history/context caps) the HTTP `/chat` route applies; `memo_search`
  and the session-pattern / synthesis MCP tools clamp their `limit` so an
  unbounded value can no longer exhaust the shared DB or blow the latency budget.

### Fixed

- The nightly dream pipeline no longer aborts its remaining passes when the
  curated graph-projection step raises an unexpected error class; the failure is
  isolated to a receipt entry like every sibling pass.
- `memo doctor` now reports a machine that is stuck *behind* the remote (a
  persistent `.md` rebase conflict), and `sync_pull` stamps a reasoned pending
  marker on conflict — previously doctor printed "up to date" while commits
  stayed unpulled.
- Hard `delete()` now purges a memory's temporal fact edges; orphaned facts no
  longer resurface in the SessionStart briefing and temporal CLI/MCP reads.
- `memo journey-check`'s isolated subprocess now strips `CLAUDE_*` env and pins
  `CLAUDE_CONFIG_DIR` inside its sandbox, so hook-wiring during the spawned
  `memo onboard` can never mutate the real `settings.json`.
- Explicit contradiction resolution (`kept_newer` / `kept_older`) compares
  timestamps as timezone-aware instants, fixing a `supersedes`-edge inversion
  across a DST boundary in positive-UTC-offset zones.
- The recall-hook `MEMO_RECALL_RERANK_INPUT_K` flag is bounded 1–200 (the hook
  injects it via `model_copy`, which skips pydantic validation); the floor-
  calibration dream pass now registers its overlay write with the online proof
  loop so a live-grounding regression it causes is auto-reverted.
- Robustness: `memo reflect` survives a malformed-encoding transcript;
  session-resume JSONL parsers tolerate non-object lines; the eval baselines are
  written atomically; the detached autosave inherits stderr so capture failures
  are no longer black-holed; `memo config validate` derives the valid model
  profiles from the config source of truth; `memo backup list` surfaces a
  corrupt archive; `MEMO_ASK_SNIPPET_CHARS=0` is honored.

## [4.4.3] - 2026-07-27

### Fixed

- Audit round-2 secondaries: the `snippet_chars` sentinel now honors an
  explicit value over the flag default; the Homebrew formula is synced; CI runs
  the tantivy/OCR extras plus the reranker real-model MLX tests; and the nightly
  contradiction pass resolves the pair before archiving (fixes a
  `NotFoundError`).

## [4.4.2] - 2026-07-26

### Security

- Fixed a **critical** remote-code-execution vector via git remote-helper
  transports (`ext::` / `fd::`) reachable from the `memo_repo_index` MCP tool —
  the repo URL is now validated against an allowed-scheme list and clones run
  with `GIT_ALLOW_PROTOCOL` set.
- Fixed a quadratic-complexity ReDoS-class DoS in the PEM secret-redaction
  regex; the scan is now linear. Added Stripe/npm/GitLab secret-prefix
  redaction.

### Fixed

- `memo sync once` / `memo sync auto` now surface push/pull/commit errors
  instead of silently reporting success.
- The nightly dream pipeline writes a receipt even on a hard crash, and
  contradiction-supersede provenance is stamped correctly.
- The capture incremental watermark no longer loses a long turn's insight.
- Store sidecar connections (history/graph/contradictions/crossref/fact_edges)
  close deterministically.
- Recall-hook rerank-pool shrink and ranking parity fixes.
- The query-embedding cache is now thread-safe.
- Config/flag resolution no longer silently diverges between call sites.
- Contradiction supersede timestamps are UTC-safe.
- The eval regression gate pins to the baseline config and guards `k`.

This release is the production-audit remediation, merged as #108.

## [4.4.1] - 2026-07-26

### Fixed

- MCP: `promote_learning` / `record_task_outcome` validation errors are now
  surfaced with their real message over the MCP write coordinator instead of
  being masked as the generic "coordinated MCP write failed safely". The
  input-validation raises migrated from bare `ValueError` to `ValidationError`
  (a `MemoError` subclass), which the single-writer coordinator passes through
  verbatim (#104).
- `memo backup list` / restore no longer crash on backups whose metadata carries
  the legacy `memoria_count` key: `BackupMetadata.from_dict` aliases it to
  `memory_count` and drops unknown keys instead of raising `TypeError` (#105).

## [4.4.0] - 2026-07-24

### Fixed

- config_md: `memo config set` no longer clobbers other flag groups that share a
  config file (misc/behavior in advanced-config.md). The write path now parses
  every `[group]` section in the target file, updates only the affected key, and
  rewrites all sibling sections intact instead of overwriting the file with a
  single group.

## [4.3.0] - 2026-07-24

### Changed

- **Eight recommended flag defaults flipped from OFF to ON** — features that
  earned their keep are now the product default; each opts out with `=0` (env
  var) or `memo config set <key> false`. The two recall-path flips were gated on
  `memo eval recall` (precision@5 and noise@5 unchanged vs OFF — no regression);
  the other six flip unconditionally.
  - `MEMO_RECALL_UNMATCHED_TERM_GATE` — honest-empty recall gate: suppresses weak
    unmatched-term noise so a zero-hit turn stays empty (and feeds `memo gaps`)
    instead of injecting distractors. Paraphrase/semantic matches are never gated.
  - `MEMO_RECALL_INTRA_DEDUP` — collapses near-duplicate hits within a single
    recall injection (lexical Jaccard), so an injection doesn't repeat itself.
  - `MEMO_VERIFICATION_STATE_TRACKING` — verification-state lifecycle: `memo
    maintain` marks due VERIFIED memories STALE, and recall applies a state-decay
    factor so fresh facts outrank stale ones (no-op on an all-UNVERIFIED corpus).
  - `MEMO_CROSSREF_INDEX` — indexes `[[wikilinks]]` and typed relation edges into
    the crossref backlinks table on save/update/delete/reindex, enabling
    cascade-aware supersede/delete warnings.
  - `MEMO_SAVE_NORMALIZE_DATES` — annotates relative date expressions in durable
    saves with absolute ISO dates (`ayer` → `ayer (2026-07-02)`, ES+EN).
  - `MEMO_DREAM_VALIDITY_EXTRACT_ENABLED` — nightly dream pass that extracts
    explicit bi-temporal `valid_at`/`invalid_at` windows stated verbatim in a
    note's text (never hallucinated).
  - `MEMO_DREAM_GRADUATION_ENABLED` — nightly dream pass that graduates
    `_uncertain` auto-captures proven by grounding or corroboration back into
    auto-recall (reversible via `memo version` rollback).
  - `MEMO_DREAM_FLAG_GRADUATION_ENABLED` — nightly dark-feature graduation pass:
    A/B-measures remaining default-off `*_ENABLED` flags and graduates winners via
    the tuned overlay (reversible on regression).

## [4.2.0] - 2026-07-24

### Added

- **Negative Recall — a preemptive ⛔ avoidance channel.** Memo now remembers
  what *not* to do and surfaces it before you repeat it. A new
  `type=failure_pattern` anti-memory tier is captured, retrieved, reinforced,
  measured, and surfaced as a distinct ⛔ AVOID block — everything OFF by
  default (six `MEMO_NEGATIVE_RECALL_*` flags), so behavior is unchanged until
  opted in.
  - **Capture** (`negative_capture.py`, `MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED`):
    the nightly dream passes auto-derive anti-memories from supersede/reversal
    decisions (the superseded side = the mistake) and from graduated *avoid*
    verdicts, with provenance recorded in `extra`.
  - **Retrieval** (`negative_recall.py`, `MEMO_NEGATIVE_RECALL_ENABLED`): a
    bounded, high-precision pass over `failure_pattern` anti-memories that
    **reuses the already-computed query embedding** (LRU cache hit — no second
    MLX forward, budget-gated as the first stage to skip), excludes
    `failure_pattern` from normal recall (no duplication), and injects at most
    `MEMO_NEGATIVE_RECALL_K` (default 2) hits above a strict
    `MEMO_NEGATIVE_RECALL_MIN_SIM` (default 0.6) cosine floor.
  - **Reinforcement** (`MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED`): the dream
    ROI-reconcile pass strengthens an anti-memory whose ⛔ surfaced but the
    mistake repeated anyway, and mildly rewards it when heeded — off the 5s hook
    path.
  - **Risky-context trigger** (`MEMO_NEGATIVE_RECALL_TRIGGER_ENABLED`): loosens
    the floor / raises K in detected high-risk contexts
    (release/delete/deploy/refactor/migrate) via a pure `O(len(prompt))` keyword
    scan — never re-embeds.
  - **Measurement**: an `avoid@k` gate in `memo eval recall` scores the ⛔
    channel, and every `*_ENABLED` flag declares its graduation gate in
    `dream_flags.GATES` (CI-enforced by `tests/test_dream_flags.py`).
  - **Surfacing**: the ⛔ AVOID block renders in the recall hook and in El
    Briefing, kept distinct from normal recall.

## [4.1.0] - 2026-07-24

### Changed

- **Auto-update is now ON by default** (`MEMO_AUTO_UPDATE`, previously opt-in).
  Every install now keeps itself current: on `memo-mcp` start memo makes a
  throttled `git ls-remote` tag probe and installs a newer tagged release in the
  background for the next start. The probe sends no memory content, paths,
  identity, or IP — only the git tag check. Memory operations remain fully
  offline. Set `MEMO_AUTO_UPDATE=0` to opt out and keep startup fully offline.
- Privacy policy and README reworded from "offline by default" to "local-first,
  offline by default with auto-update as the one default-on outbound call",
  documenting the opt-out.
- Update offer is now channel-aware. Homebrew installs are detected as their own
  channel: the banner offers `brew upgrade mlx-memo` and `memo update` runs
  `brew upgrade` instead of failing over the Cellar; background auto-update on
  Homebrew offers (notifies) rather than running brew unattended. pipx / uv tool
  / PyPI continue to self-install via `memo update`.

## [4.0.1] - 2026-07-24

### Fixed

- MCP write tools no longer surface database lock contention as the opaque
  `coordinated MCP write failed safely`. `VecStore._tx()` now translates a
  lock-class `sqlite3.OperationalError` (a concurrent writer — another agent
  session, or an external tool syncing the DB inside an Obsidian vault — holding
  the write lock) into a typed, retryable `StorageError`, so the write
  coordinator reports the real cause and retry semantics. `save()` keeps its
  graceful `_memo_embed_pending` fallback; all other write tools (`update`,
  `delete`, `supersede`, `focus`, `attention`, `outcome`, `handoff`, …) now fail
  with an actionable message instead of the generic wrapper. Non-lock
  `OperationalError`s (e.g. `no such column`) still propagate unchanged.
- The write coordinator now logs the original traceback via `_log.exception`
  before masking an unexpected exception, so a swallowed root cause is no longer
  unrecoverable from the server stderr / MCP logs.

## [4.0.0] - 2026-07-23

### Added

- A self-contained operational-memory kernel: Memo-owned versioned contracts,
  trace propagation, append-only hash-chained operation journal, conflict-aware
  write policy, attention queue, focus state, handoffs, and crash recovery.
- `EvidencePack`, a bounded retrieval result with source identifiers, evidence
  URIs, coverage, freshness, and an explicit abstention result when the corpus
  cannot support an answer.
- Outcome-driven learning. Agents can record task success, partial success, or
  failure against the memories they used; repeated evidence updates utility and
  priority, and can promote memories into procedures or failure patterns.
- Deny-by-default federation with per-memory `visible_to` ACLs, signed and
  recipient-bound bundles, tamper detection, idempotent import, provenance, and
  complete causal-journal transfer for owner backups.
- `memo definitive check` and `memo definitive benchmark`, plus a dedicated CI
  lane that proves the package has no retired runtime imports or sibling-repo
  dependency.
- A one-way compatibility migration for legacy trace, cache, provenance, and
  operational metadata. Markdown remains authoritative throughout migration.
- Namespaced memory identity (`project:<slug>`, `_global`, `_unscoped`) with
  deterministic create/corroborate/revise outcomes, additive `action` and
  `index_pending` save fields, schema-v5 diagnostics, and read-only
  `memo doctor --db` trust preflight.
- Canonical relation convergence in `memory_relations`: deterministic unordered
  pair identity, bounded/no-LLM post-save candidates, auditable idempotent
  judgments, `supersedes` validity transitions, orphan retention, compact
  search/ask/briefing annotations, and an idempotent legacy contradiction import.
- Explicit freshness lifecycle with first-class `review_after`, durable review
  evidence, policy schedules, `memo review due|mark`, and MCP review,
  invalidation, and supersession tools. Review timing is separate from truth
  validity and never auto-invalidates a memory.
- Bounded FIFO coordination for mutating MCP calls, including typed
  retryable saturation, cancellation semantics, safe error translation, and
  queue/wait/rejection diagnostics via `memo_write_queue_status`.
- Declarative Codex and Claude Code adoption through `memo setup`, including
  detection/dry-run plans, atomic managed instructions, backups, compensating
  rollback, partial-install receipts, and `memo doctor --agent` verification
  with an isolated save/search smoke test.
- A fixed 12-case relation policy corpus and `memo eval relations --gate` for
  candidate recall, noise, namespace isolation, and reproducible fingerprinting.

### Changed

- Memo now owns its complete runtime contract and operational state. Core
  behavior no longer delegates to Synapse, Memflow, or private
  `consciousness-contracts` packages.
- Lifecycle promotion and demotion now consume durable task outcomes and persist
  their priority changes instead of producing advisory-only recommendations.
- Unified briefing, cache selection, tracing, provenance, CLI surfaces, and MCP
  tools use Memo-native schemas. Legacy field names are accepted only at the
  read-time migration boundary.
- The contradiction scanner and legacy CLI now write/project the canonical
  relation ledger. `contradictions.db` is an import-only compatibility source;
  no new path dual-writes it.
- Verification decay now moves only explicitly scheduled, due `VERIFIED`
  records to `STALE`; stale records remain stale until review or invalidation.
- MCP profile inventories are now 30 tools for `agent`, 50 for `core`, and 158
  for `full`, including always-on write-queue diagnostics.
- Canonical relation candidate generation and judged-relation annotations are
  now on by default after passing the fixed 12-case policy corpus and measured
  save-latency gate. Both remain independently reversible with their
  `MEMO_RELATION_*_ENABLED=0` opt-outs.
- Mutating MCP calls now use the process-local FIFO coordinator by default with
  capacity 32. `MEMO_MCP_WRITE_QUEUE_SIZE=0` restores direct dispatch; the
  data-directory lock remains the cross-process authority.
- Internal post-save candidate searches no longer record user access or
  co-recall signals, preventing derived maintenance work from contaminating
  usage telemetry.
- Existing vaults with the experimental integer-keyed `memory_relations` table
  are migrated transactionally to canonical text identities before default-on
  relation writes begin; legacy rows and lookup aliases are preserved.

### Removed

- Runtime modules and adapters for Synapse, Memflow cache delegation,
  consciousness ledgers, external receipts, and cross-dedup orchestration.
- Dead fixed-age verification flags `MEMO_VERIFICATION_STALE_DAYS` and
  `MEMO_VERIFICATION_UNVERIFY_DAYS`; record-specific `review_after` owns
  freshness policy.

### Security

- Federation excludes secrets and unapproved metadata, rejects unsafe bundle
  sizes and device identifiers, verifies HMAC signatures before import, and
  rejects divergent foreign journal chains.
- All memory save, update, reindex, capture, and ingest persistence paths now
  enforce known-secret masking and `<private>` stripping at the final storage
  boundary. Entropy masking remains opt-in; legacy redaction flags now control
  only earlier defense-in-depth passes.

## [3.12.1] - 2026-07-22

### Fixed

- **Recall under-fill under the validity gate** (finding-7). Default recall and
  `--as-of` apply the bi-temporal validity gate *after* the vec kNN, so a plain
  `vec.k = limit` could return fewer than `limit` results when the nearest
  neighbours were the rows being filtered out (contradiction-superseded or
  out-of-as-of). vec-search now widens the candidate pool the same way it
  already does for date/tag post-filters — but only when the gate can actually
  drop rows (an `as_of` query, or a corpus that holds any invalid row, checked
  via the `idx_meta_invalid_at` partial index), so the common all-valid recall
  path pays nothing on the hot 5s budget.

## [3.12.0] - 2026-07-22

### Added

- **Graph mindmap visualization** (#80). New `memo graph mindmap [ENTITY]`
  renders a graph neighborhood as a self-contained, offline-clean interactive
  HTML mindmap (pan / zoom / fold). A pure-Python tree builder converts the
  existing `GraphNavigator.export_json()` subgraph into a nested tree (BFS,
  depth-bounded, node-capped, cycle-safe); the renderer mirrors the Health
  Dashboard security pattern (inlined CSS/JS, CSP-nonce,
  `atomic_write_text(mode=0o600)`, no external `<script src>`/CDN) with a small
  memo-owned vanilla-SVG drawer — no vendored third-party JS. Defaults the
  center to the highest-degree entity; empty graph is a graceful no-op. Viz
  output only — not an MCP tool (respects the no-cognition invariant).
- **Record-level bi-temporal validity** (#79). Memories now carry world-validity
  (`valid_at` / `invalid_at`) distinct from learned-time (`created`).
  Supersede-by-contradiction closes the loser's interval in place instead of
  hiding it; default recall filters to currently-valid records; valid-time
  queries via `memo search --as-of T` and MCP `memo_search_valid_as_of` /
  `memo_ask_valid_as_of`. A gated dream pass
  (`MEMO_DREAM_VALIDITY_EXTRACT_ENABLED`) extracts validity windows from bodies
  (guarded against hallucination), and `memo migrate --backfill-valid-time`
  backfills an existing corpus.
- **Sharper memory extraction** (#79). The shared capture/dream extraction
  prompt now preserves proper nouns and numbers verbatim, folds "switched X→Y"
  transitions into a single memory, and noise-gates one-off/volatile turns.

### Fixed

- **Validity recall gate** (#79): the validity filter now covers all candidate
  legs (fact-retrieval and hype folds, not just the SQL vec/bm25 legs), so a
  superseded record with a fact edge can no longer leak into default or as-of
  recall. Timezone handling for the validity filter binds `now`/`as-of` in the
  same local offset the records stamp, fixing a lexicographic-compare skew on
  UTC-offset machines.

## [3.11.0] - 2026-07-21

### Added

- **Verification-state decay** (opt-in via `MEMO_VERIFICATION_STATE_TRACKING`,
  default OFF). A verification lifecycle that had been implemented but never
  wired is now live end-to-end: a memory marked `verification_state: verified`
  without a `verified_at` gets one stamped on reindex; `memo maintain` ages
  VERIFIED→STALE→UNVERIFIED by `verified_at` (thresholds
  `MEMO_VERIFICATION_STALE_DAYS` = 30, `MEMO_VERIFICATION_UNVERIFY_DAYS` = 60),
  recording the count in its receipt; and live recall multiplies each hit's
  score by a state decay factor (VERIFIED ≈ 1.0, STALE 0.7, UNVERIFIED 0.8) so
  fresh facts outrank stale ones. Pure in-memory on the hot path; a no-op for an
  all-UNVERIFIED corpus. Replaces the earlier, thread-unsafe `memory_map`
  approach with per-call data sourced from the record/store.

### Fixed

- **Recall hook budget:** the subprocess fallback re-embedded over the daemon
  socket at a 30 s timeout — a busy/warming daemon still answers `ping`, so a
  stalled embed could exceed the hook kill. The fallback now caps the embed
  timeout, requires the daemon (fail fast, no cold-load GPU fight), and
  downgrades vec/hybrid → bm25 on embed failure instead of returning nothing.
- **Recall daemon/subprocess parity:** ported the daemon-only post-rank gates to
  the subprocess fallback — `MEMO_RECALL_DEDUP_COLLAPSE` (default ON), the
  unmatched-term gate, and the recency band — so recall no longer differs when
  the daemon is down.
- **Graph density boost:** implemented `GraphStore.memory_degree()` — it was
  called by the density-boost rerank but never defined, so
  `MEMO_GRAPH_DENSITY_BOOST` silently did nothing.
- **`memo migrate --consolidate-db`:** now merges all six of `graph.db`'s tables
  (previously only two), so co-recall / entity-edge / alias / semantic-relation
  data is no longer orphaned into `graph.db.bak`.
- **`memo links reindex`:** uses `index_source` (delete-then-insert, including
  typed `- rel [[target]]` edges) instead of the append-only `index_wikilinks`,
  which dropped typed edges and left stale rows.
- **ProactiveStore:** aligned with the sidecar connection model (WAL,
  `check_same_thread=False`, busy timeout); the briefing path now closes its
  connection instead of leaking it on the SessionStart hot path.

### Removed

- Dead, unreachable code with no callers: the never-wired token-economy Wave 2
  modules (`stream_compress`, `prefix_optimizer`), the graph-distance rerank
  (a stub `distance_to_nearest_fact` + an in-memory `Graph` used only by tests),
  and the dead flags `MEMO_STREAM_COMPRESS`, `MEMO_PREFIX_CACHE_ALIGN`,
  `MEMO_MEMFLOW_DIR`, `MEMO_GRAPH_DISTANCE_DECAY`, `MEMO_GRAPH_DISTANCE_DECAY_RATE`.

## [3.10.0] - 2026-07-21

### Changed

- **`MEMO_VEC_QUANTIZE` now defaults to `int8`** (was `off`). Fresh indexes are
  created as `int8[dims]` (1 B/dim, ~4× smaller on disk and in the sync shard).
  **Existing installs are never broken:** the store is now self-describing — it
  adopts the on-disk `vec` precision, so a float32 index keeps working as
  float32 under the new default (an INFO log notes the adoption). To move an
  existing index to int8, run `memo reindex --rebuild` (which honors the
  configured precision). Backward-compatible graduation of the dark flag shipped
  in 3.9.0.

## [3.9.0] - 2026-07-21

### Added

- **Proactive engine** (`MEMO_PROACTIVE_ENABLED`, default `off`): a unified
  engine that surfaces useful, non-annoying messages to the human and the agent.
  Guarded detectors — **reliability** (superseded facts), **continuity** (open
  loops), **health** (low-confidence memories), **roi** (never-surfaced
  memories), **dejavu** (recurring patterns) — feed a precomputed sqlite pool
  written by the nightly `dream` pass (never the 5s recall path). A single
  arbiter scores, budgets, and suppresses candidates with adaptive per-kind
  demotion learned from feedback, and routes them to three surfaces: `memo
  digest` (pull), a compact El Briefing section, and a Stop-hook urgent push
  (reliability only, cooldown + daily-cap gated). Never fabricates — every nudge
  cites a real memory; an empty corpus surfaces nothing.
- `MEMO_VEC_QUANTIZE=int8` (default `off`, dark flag): store the main `vec`
  table as `int8[dims]` (1 B/dim) instead of `float32[dims]` (4 B/dim) via
  sqlite-vec `vec_quantize_int8(...,'unit')`, cutting the vec table and the
  cross-machine embed-cache sync shard ~4×. Safe only for L2-normalised vectors
  (memo's are). It changes the vec0 column *type*, so it takes effect on
  `memo reindex --rebuild` only; a DDL-derived guard raises `StorageError` on a
  precision/index mismatch. Manual, eval-gated graduation — never auto-tuned.

### Fixed

- int8: `vec` embedding blobs are now decoded dtype-aware in the consolidate
  pass and the viz tools (`web_build`, `cli_viz`) — reading a 1 B/dim int8 blob
  as float32 produced garbage (numpy overflow in the nightly consolidate). The
  reliability/continuity urgent-push feedback loop now also records `acted`
  (previously the multiplier could only decay), and a demoted reliability kind
  can still break silence for an urgent push.

## [3.8.2] - 2026-07-21

Second-round QA hardening pass (bug fixes only, no behavior/API changes).

### Fixed

- Project bucket names can no longer collide with the lifecycle-archive dirs:
  a memory tagged `project:inactive`/`project:archived` (or saved in a repo of
  that name) with `MEMO_STORE_BY_PROJECT` on used to be written where reindex/gc
  skip, making it invisible to search and unrecoverable — those slugs now remap
  to `_inactive`/`_archived`.
- `defer_embed` saves without a `topic_key` now route an index failure through
  the `_memo_embed_pending` recovery path (the `.md` is never silently missing
  from the index until a manual reindex).
- Hard-delete rollback (`MEMO_SOFT_DELETE=0`) preserves the user-signal tables
  (access / health / source_feedback) instead of losing them.
- `memo_search_as_of` applies its type filter before the limit, so a
  type-scoped as-of search returns up to `limit` matching rows.
- Incremental capture no longer advances its watermark past turns whose helper
  LLM extraction failed transiently (those insights are retried, not lost).
- The belief supersede support gate fails closed: a support-lookup error holds
  the memory open instead of archiving a possibly-supported one.
- `recall-daemon`/`ingest-daemon`/`maint-daemon` start now confirm readiness by
  connecting to the socket (with fail-fast on child death) rather than trusting
  a stale socket file, and no longer unlink a live daemon's socket.
- The dashboard JSONL trim holds an exclusive lock over the append+truncate so a
  concurrent write under a shared `data_dir` can't be lost or corrupt the file;
  the dashboard health probe is backend-aware (no false "MLX not importable" on
  a healthy Linux/CPU install); and the cheap live poll no longer rescans the
  whole corpus for body-hash drift.
- `.capture_stop`/`.capture_watermark` sidecar files are pruned on a 30-day TTL
  instead of accumulating forever.
- Tuned-overlay state-dir resolution mirrors `Config.from_env` exactly (legacy
  `[storage]` TOML, key-presence `MEMO_VAULT_PATH`).

## [3.8.1] - 2026-07-20

### Security

- Treat recalled memories, repository excerpts, and prompt overrides as untrusted
  data in every ask path; immutable trust instructions are appended after
  overrides and delimiter-like content is escaped.
- Quarantine unsupported assistant outcome claims by default, recording claim
  kind/reason and lowering confidence instead of silently promoting them to
  durable fact.
- Bound and pre-buffer every HTTP request body, reject incomplete/disconnected
  requests before side effects, validate Unicode and chat payload depth/size,
  and install the loopback Host/Origin guard in direct/reload no-auth workers.
- Gate PyPI, GitHub Release, Docker, and MCP registry publication on the exact
  tagged SHA passing static QA plus real Linux CPU and Apple Silicon MLX smoke
  suites; PyPI also waits for successful GitHub Release creation.
- Require HTTPS for remote benchmark judges (HTTP remains available only on
  loopback) and reject redirects so bearer credentials cannot cross origins or
  downgrade to plaintext.
- Require HTTPS for benchmark dataset overrides as well, and reject redirect
  downgrades; local datasets remain available through the explicit file option.
- Resolve every shipped Hugging Face model through an audited 40-character
  commit SHA before loading; unknown remote overrides now require an exact
  inline or role-specific revision, while local model paths remain supported.

### Fixed

- Preserve Markdown/SQLite consistency when updates fail, including legacy
  vault-only sources; serialize history reads and repair sync ordering so
  backlogs larger than 1,000 events cannot skip the oldest entries.
- Abort stale Git rebases without ever resuming them via `--skip`, preserve the
  local commit, and skip pull/autostash after unrelated commit failures so
  uncommitted Markdown cannot be lost.
- Export complete corpora beyond 10,000 records and replace JSON, CSV, and ZIP
  exports atomically.
- Serialize backup creation, listing, and restore; publish compressed archives
  atomically so readers never observe scratch directories or partial tarballs.
- Make capture state and cooldowns per-session, serialize snapshot refreshes
  across processes, keep failed captures retryable, and prevent duplicate Stop
  hooks from extracting/saving twice. Incremental capture also retains its
  watermark after partial or total save failure so the exchange can be retried,
  with a separate retry backoff preventing repeated LLM work every few seconds.
- Keep MCP read-only notification tools non-consuming, avoid loading MLX for
  text-only saves, cancel timed-out sampling work, and preserve exact-boundary
  offloaded payloads losslessly.
- Make auto-update leases cross-process atomic and crash-safe with child-side
  handoff, OS process identity, stale-start recovery, retryable persistence
  failures, and process-group termination.
- Close owned Memory, sqlite, socket, server, runner, and lock resources on
  finite commands, normal daemon shutdown, constructor failures, registration
  failures, bind failures, and early setup exceptions.
- Reject invalid persisted MCP transports, profiles, ports, and flag choices;
  remove silent profile fallback.
- Install the CPU sentence-transformers backend for bare Linux distributions
  and MCP bundles, while keeping MLX imports deferred. Restrict the dependency
  marker and support claims to Linux because PyTorch has no Python 3.13 wheels
  for Intel macOS.
- Route recommended Linux uv/pipx installs and the release smoke through the
  official PyTorch CPU wheel page, avoiding CUDA, Triton, and NVIDIA packages.
- Pin the one-line installer to this release, preserve existing uv/pipx tools
  when replacement installation fails, and remove destructive pre-uninstalls.

### Changed

- Refactored capture, configuration, release validation, server routing, and
  update paths to keep the progressive complexity/exception quality budget
  green without widening its baseline.

## [3.8.0] - 2026-07-20

### Added

- **Embed-cache sync** — per-machine shards (`embed_cache/<device_id>.json`) carry
  document/chunk embeddings through the memo-sync repo; a pulling or bootstrapping
  machine imports them before the post-pull reindex and indexes peer memories with
  ~zero local MLX embed calls. Capped by recency (`MEMO_SYNC_EMBED_CACHE_MAX_ROWS`,
  default 1000); vault/reference tier never exports; disable with
  `MEMO_SYNC_EMBED_CACHE=0`.
- **`memo eval ab`** — blind-judge A/B eval: same prompts answered with and without
  memo's recall context by the local LLM, scored by a blind judge (symmetric
  prompts, pre-judge leak scrub, recall-faithful ON condition). Raw runs persist
  under `state_dir/eval/` for audit.
- **Dream edge-verify pass** (`MEMO_DREAM_EDGE_VERIFY_ENABLED`, default off) —
  knowledge-graph edges earn confidence from grounded co-use evidence, with an
  idempotent sidecar ledger, gentle reversible decay, and rebuild survival; the
  associative recall nudge drops its "· unverified" qualifier only above the shared
  verified threshold.
- **Empty-recall epistemic marker** (`MEMO_RECALL_EMPTY_MARKER`, default on) — a
  successful search with zero relevant hits now injects "absence of record, not
  evidence of absence" instead of silence; a failed search still emits nothing.
- **Measured benchmark** — `docs/BENCHMARK.md` with real command output only,
  negative results included, plus a published adversarial challenger review of the
  doc's own claims; README carries the surviving numbers.

## [3.7.0] - 2026-07-16

### Added

- MCP client-sampling synthesis (dark flag `MEMO_SAMPLING_SYNTH_ENABLED`):
  `memo_ask`, `memo_chat_ask`, `memo_reflect`, `memo_synthesize_run`, and
  `memo_consolidate` delegate synthesis to the connected client's model via
  MCP sampling, with sticky per-request MLX fallback, a per-request call cap
  (`MEMO_SAMPLING_MAX_CALLS`), and a `synthesizer` attribution field on
  dict-shaped responses. Grounding judgement stays local by design.
  Companion flags: `MEMO_SAMPLING_TIMEOUT_S`, `MEMO_SAMPLING_MAX_TOKENS`.
- Chunk emission at save/update time (`MEMO_CHUNK_INGEST`): long documents
  get their heading-aware chunk records immediately on `save()`/`update()`
  instead of waiting for the next manual `memo reindex`. Best-effort — a
  chunk-emission failure never fails the write; metadata-only updates skip
  emission. Docs for the chunker refreshed to match its real wired state.

### Fixed

- Release metadata realigned for v3.6.0: rebuilt `.mcpb` archives,
  regenerated `uv.lock`, SECURITY.md supported line bumped to `3.6.x`.

## [3.6.0] - 2026-07-15

### Added

- Dark-feature graduation pipeline (`memo dream graduate-flags`, module
  `dream_flags.py`): every default-off `*_ENABLED` flag now declares a
  graduation gate (`recall` = nightly ON/OFF A/B through the recall-faithful
  eval; `tuner` = owned by an existing tuner pass; `manual` = documented
  reason). Recall-gated flags that win `MEMO_FLAG_GRADUATION_WIN_NIGHTS`
  consecutive measurements (latency + curated no-regression gated) graduate
  to ON via the tuned overlay, reversibly — a later regression reverts them.
  Flags still dark after `MEMO_FLAG_GRADUATION_DEADLINE_DAYS` surface as
  cull candidates in `--status`. Gate completeness is CI-enforced: a new
  dark flag cannot merge without declaring its gate. Nightly pass gated by
  `MEMO_DREAM_FLAG_GRADUATION_ENABLED` (default off).

## [3.5.2] - 2026-07-15

### Security

- Hardened local HTTP surfaces against DNS rebinding, cross-site requests,
  reflected errors, unsafe content types, XSS, and accidental unauthenticated
  dashboard access; generated health pages are now offline-safe with a strict
  CSP and capability-protected live data.
- Hardened backups, restores, runtime state, release staging, and secret files
  against archive bombs, path traversal, symlink races, partial publication,
  permissive modes, and accidental credential inclusion.
- Release and auto-update paths now require trusted tags descended from
  `origin/master`; Docker builds use an explicit context allowlist, pinned
  tooling, locked hashed dependencies, and the CPU-only PyTorch index.

### Fixed

- Made save, update, delete, reindex, backup, and restore operations atomic
  across threads and processes, including topic-key compare-and-swap recovery,
  path-collision rollback, live SQLite restore coherence, and stale pending
  marker cleanup.
- Rebuild now migrates embedding model, revision, and dimensions consistently
  across the primary, HyPE, and episode indexes while preserving durable
  deduplication and verification metadata and invalidating incompatible
  derived watermarks.
- Closed request-validation, session-isolation, traversal, timestamp, receipt,
  resource-lifecycle, slow-test cleanup, and error-sanitization gaps across the
  CLI, MCP transports, runtime updater, and background dashboard.

## [3.5.1] - 2026-07-13

### Fixed

- **embedder_client per-op socket timeouts** (query 30s, batch 120s,
  ping/stats 5s): the flat 8s default was calibrated for the 0.6B profile;
  on the 4B `quality` profile `embed_batch` routinely exceeds it, so clients
  timed out, retried (abandoned requests kept computing daemon-side), then
  fell back in-process — loading a second model copy that fought the daemon
  for the cross-process GPU lock and drove warm `embed_query` past 10s.
  `MEMO_EMBEDDER_CLIENT_TIMEOUT` still overrides every op when set.
- HyPE dark-index hardening: type filters respected on fold-appended
  candidates, `_hype_store` closed in `Memory.close()`, markdown fences
  stripped before parsing LLM questions, honest `backlog_remaining` +
  `all_items_failed` status on real runs.
- Capture: skip LLM insight items with non-string title/body/type.

## [3.5.0] - 2026-07-13

### Added

- **HyPE question-space index** (dark, read fold default off): nightly dream
  pass (`MEMO_DREAM_HYPE_ENABLED`) generates 2-3 "questions this memory
  answers" per durable memory with the local LLM (ROI-prioritized backlog,
  `body_hash` watermark, per-item failure isolation) into a rebuildable
  vec0 sidecar. Read-path max-fold behind `MEMO_HYPE_ENABLED` (off — measured
  no-flip at 10.6% index coverage; re-evaluated once coverage grows). New:
  `memo dream hype`, `memo hype status`, eval config K + profile `hype`.
- **MCPB Node bundle**: `build_mcpb_node()` produces `memo-node.mcpb` with a
  zero-dependency `bootstrap.js` that installs `uv` and the pinned `mlx-memo`
  on first launch and then execs `memo-mcp` (stdout reserved for MCP) — no
  preinstalled Python runtime required. Version pin-chain enforced by test,
  `memo release bump`, and `memo release check`.

### Fixed

- `bootstrap.js` pin check matches the real `uv tool list` output format
  (`mlx-memo vX.Y.Z`), keeping the no-network fast path and offline starts.

## [3.4.0] - 2026-07-13

### Added

- **Chronicle** — nightly engineering diary (default off): a new dream pass
  (`MEMO_DREAM_CHRONICLE_ENABLED`) writes a human markdown diary per day under
  `<memory_dir>/_chronicle/` from memo's own logs (episodes, new memories,
  grounding, maintenance receipt), narrated by the local LLM with mandatory
  per-id provenance — bullets without a valid citation are dropped, and a
  fully-uncited narrative is not written at all (`low_provenance`).
  `MEMO_CHRONICLE_WEEKLY` adds a deterministic ISO-week rollup. New commands:
  `memo chronicle [--date|--week]` (reader) and `memo dream chronicle` (one-off).
- **`memo onboard`** — Day-0 wizard: wires the recall hook + shims, backfills
  memories from transcripts already on disk (`MEMO_ONBOARD_BACKFILL_DAYS`,
  default 90, resumable, secret-redacted), points at importers, and shows the
  first "3 things memo already knows about you" — all without loading MLX in
  the wizard itself. `--yes` for headless, `--dry-run` never touches settings.

### Fixed

- Disk scanners (chronicle day-facts, onboard recent-memories) now include the
  `_global/` bucket, where all untagged/backfilled memories live under the
  default per-project layout.

## [3.3.0] - 2026-07-13

### Added

- **Cognition-layer program (Phases 0-3), all default-OFF.** A staged path from
  experimental to trusted behavior, gated end-to-end by dark flags with a
  reversible flip:
  - **Graduation controller (Phase 0):** shadow-proves dark flags against real
    traffic before they can affect behavior, then flips them on via the tuned
    overlay once proof holds — never a direct env-var mutation, so any
    graduation is reversible by deleting the overlay entry.
  - **Phase 1 — activation:** numeric self-tuning candidates move from
    proposal to applied value through the same graduation gate; PAVA
    (pool-adjacent-violators) confidence calibration turns raw scores into
    calibrated confidence; `MEMO_RECALL_CONFIDENCE_GATE` withholds low-confidence
    recall results instead of surfacing them as if authoritative.
  - **Phase 2 — distillation:** `dream_distill` performs upward re-abstraction
    of accumulated memories into higher-altitude summaries during the nightly
    dream pass, feeding recall altitude selection; fully reversible (distilled
    output is additive, not destructive of source memories).
  - **Phase 3 — proactive:** a contradiction-gated interject sits alongside the
    existing accountable-guard, and an ask-one-gap briefing surfaces the single
    highest-value unknown; both start in shadow mode and only reach the user
    after a human flips them on.

## [3.2.0] - 2026-07-12

### Added

- **`memo config` terminal configuration center.** Bare `memo config` now opens
  a Textual TUI on interactive terminals, with a four-step first-run wizard,
  domain navigation, global search across every registered setting, typed
  controls, source/effective-value badges, cross-setting validation, draft
  review, explicit post-save activation, and conflict/recovery screens. Markdown
  remains the editable source of truth. Multi-file writes preserve surrounding
  prose and use staged transaction manifests plus rollback backups. Headless and
  `MEMO_NONINTERACTIVE=1` invocations keep the existing config subcommands and do
  not load Textual.

## [3.1.0] - 2026-07-11

### Added

- **`memo sync setup` — guided onboarding for cross-machine memory sharing.** One
  interactive command to share your memory corpus over git: *create a new corpus*
  (auto-creates a private repo when the GitHub CLI is present, otherwise guides you
  to paste any empty repo URL — GitHub/GitLab/self-hosted) or *join an existing one*
  (paste the URL). Skippable at every step; `--never` silences the nudge.
- **Discoverable, never restrictive.** A one-line first-run tip, a dismissable
  SessionStart briefing line shown only when sync is off, and a richer `memo doctor`
  hint all point at `memo sync setup` — so sharing is easy to find without ever
  blocking a local-only user or prompting from a hook.
- `sync_init_home_byo` — the no-`gh` "bring your own empty repo" initializer
  (initializes cleanly even for a brand-new empty corpus).

## [3.0.1] - 2026-07-11

### Fixed

- **`memo sync status` / `memo doctor` no longer report a false STRANDED.**
  Ahead/behind were computed against the last-fetched tracking ref, and a
  `sync_pending` marker survived after a later trigger already pushed the commit —
  so the diagnostics cried STRANDED long after the remote was caught up.
  `sync_status(check_remote=True)` now fetches the ref first (truthful counts) and
  self-heals a provably-stale marker (reachable + nothing unpushed + clean tree +
  no block reason); blocked markers from the secret gate survive. The CLI `sync
  status` fetches by default (`--offline` skips the network; `--check-remote` kept
  as a hidden deprecated alias), and `doctor` probes the remote.

## [3.0.0] - 2026-07-11

### Changed

- **BREAKING — `MEMO_RECALL_DEDUP_COLLAPSE` now defaults ON.** Recall collapses
  lexical paraphrase near-duplicates in the over-fetched pool before top-K
  truncation, so redundant near-dups no longer crowd out distinct results. Proven
  net-positive in the paraphrase-crowding state (controlled fixture recall@5
  0.5→0.833, zero added noise) and provably never-negative on the committed
  regression corpus (Δ0.0). Set `MEMO_RECALL_DEDUP_COLLAPSE=0` to restore the
  prior behaviour.

### Added

- **Trust & Belief-Revision program (P0–P3).** Every unit is opt-in / default-OFF
  except the collapse flip above; each has deterministic fail→pass test proof.
  - Belief-integrity maintenance (Dream + `memo maintain`): a `competing`
    contradiction status and a shared trust-margin `supersede_decision`,
    corroboration support-gate, and N-way cluster detection
    (`MEMO_BELIEF_COMPETING`, `MEMO_SUPERSEDE_MARGIN`, `MEMO_BELIEF_NWAY`,
    `MEMO_SUPERSEDE_SUPPORT_GATE`).
  - Write-path grounding: a capture-time grounding judge quarantines
    low-grounding memories as `_uncertain`, claim-support downgrades unsupported
    evidence-refs, per-type save-gate presets, and ask-path abstention on
    unentailed answers (`MEMO_GROUNDING_JUDGE`, `MEMO_CLAIM_SUPPORT`,
    `MEMO_SAVE_GATE_PRESETS`, `MEMO_GROUNDING_ASK_MIN`).
  - Honest recall: a per-hit trust dossier, declare-competing-disputes (surface
    both sides of a dispute instead of silently demoting the older side), and
    Dream noise-quantile floor calibration (`MEMO_HIT_DOSSIER`,
    `MEMO_DECLARE_DISPUTES`, `MEMO_FLOOR_CALIBRATION`).
  - `memo eval bench run --regime oracle` and a per-QA judge guard.
- Graph phase 2: deterministic semantic-relation extraction/backfill
  (`memo graph relations rebuild`), hub diagnostics (`memo graph hubs`),
  graph-on/off recall eval comparison (`memo eval recall --graph-ab`),
  human `memo search --explain` graph reasons, and optional outcome-informed
  graph boost modulation (`MEMO_GRAPH_OUTCOME_SIGNAL_ENABLED`).

### Fixed

- Recall-eval gate (`memo eval recall`) no longer appears to hang mid-grid: its
  two hybrid configs invoked the production cross-encoder reranker
  (~7–11s/prompt cold, thrashing under the concurrent MLX fleet), contradicting
  the documented "fast, retrieval-only" contract. It now runs retrieval-only —
  the gate re-ranks the candidate pool with the shared `rank_hits` — completing
  the full 4-config grid in ~15s instead of >300s.
- Documentation corrected to match code: the HTTP MCP transport and
  `memo http-api` have **no built-in authentication** (previously documented as
  bearer-auth with `MEMO_HTTP_API_TOKEN`, which does not exist); MCP profile tool
  counts are 14 / 34 / 129 (agent / core-slim / full); the CPU embedder is
  1024-dim; the record editor is `memo edit` (not `memo update <id>`); and the
  Docker HTTP transport uses `MEMO_MCP_TRANSPORT=http` (no `memo-mcp --http`).

## [2.12.20] - 2026-07-10

### Added

- `memo eval bench run --contradict-scan`: runs memo's contradiction scanner on
  each isolated bench store after ingest and enables the contradiction penalty
  during scoring, so the `knowledge_update` capability bucket measures memo's
  real conflict handling instead of raw retrieval. Pair classification uses
  `cfg.helper_model` (~4B), not the answer LLM — off the 30B OOM path. Default
  off keeps raw-retrieval scoring identical.

### Docs

- `docs/eval/capability-baseline-and-levers.md`: per-bucket LongMemEval-oracle
  retrieval baseline plus the measured lever results — MMR is negative for
  single-evidence retrieval, recency decay is N/A for old-dated corpora, and the
  contradiction penalty over-detects on turn-granular ingestion (measured
  regression). Documents why BEAM does not fit the evidence-labeled harness and
  the local-GPU wall for the abstention QA number.

## [2.12.19] - 2026-07-10

### Added

- **Capability-taxonomy rollup** for `memo eval bench` (Memoria-style 6-bucket
  view): per-category retrieval/QA metrics roll up into fixed capability buckets
  — single-session grounding, preference understanding, multi-session synthesis,
  temporal state tracking, knowledge-update/conflict, abstention/constraint — via
  the new pure leaf module `eval_bench_taxonomy.py` (`capability/<bucket>/…` rows
  in the receipt + report). Cross-dataset: maps both LoCoMo and LongMemEval
  category labels.
- **First-class abstention metric** in `memo eval bench` — correct-abstentions,
  hallucinations, abstention-accuracy, hallucination-rate — plus per-bucket QA
  accuracy. An abstention question routes to the `abstention_constraint` bucket
  regardless of its source category.
- **Typed feedback lifecycle routing**: `memo feedback flag --kind outdated|wrong`
  (CLI) and `memo_feedback_flag` (MCP) route a stale/wrong memory through the
  reversible archive — `outdated` archives it (hidden from search, restorable),
  `wrong` archives **and** supersedes it (optionally by a replacement id). Steers
  non-Claude clients to self-correct via a new mandate bullet.

### Fixed

- LongMemEval benchmark download 404 — the HuggingFace blobs (`longmemeval_oracle`,
  `longmemeval_s`) have no `.json` extension; appending it 404s.

## [2.12.18] - 2026-07-10

### Added

- Accountable memo — proactive **Guard** + **Capture receipt** (both default-OFF):
  - Guard (`MEMO_GUARD_ENABLED`): flags a prior `decision`/`preference` the prompt
    looks to be reversing (reversal-signal gate over recalled hits) with a ⚠ banner
    at the top of the recall block, in both recall paths (subprocess hook + warm
    daemon). Advisory, never blocks; pure-Python, respects the 5s recall budget.
    `memo guard stats` reports fires. A live-LLM verdict is specified but deferred —
    measurement showed the bottleneck is decision *retrieval*, not classification.
  - Capture receipt (`MEMO_CAPTURE_RECEIPT`): makes auto-capture visible and
    correctable — a Stop-hook receipt lists saved titles+ids with `memo undo` /
    `memo fix` verbs to remove or correct a wrongly-captured memory.
- Design and gated implementation path for Memory Quality Loop:
  quality-aware reranking, context packs, and reversible quality compaction.
- **Temporal fact edges.** Subject–predicate–object fact edges are extracted
  from memories into a `fact_edges` store, rebuilt on `memo reindex`, and fused
  into hybrid search as a temporal fact-edge leg (`MEMO_FACT_RETRIEVAL_ENABLED`,
  weight `MEMO_FACT_RETRIEVAL_WEIGHT=0.6`, both default-on). Query-relevant facts
  attach to search/ask results (`MEMO_FACT_SURFACE_ENABLED`) and surface in the
  SessionStart briefing's **Temporal facts** section. New CLI:
  `memo temporal facts add|list|invalidate` with validity windows
  (`--valid-at` / `--invalid-at` / `--expired-at`, `list --as-of <date>`) and
  full-profile MCP tools (`memo_fact_edges`, `memo_fact_edge_save`,
  `memo_fact_edge_invalidate`).
- **Visible memory context surface.** `memo context "<question>"` builds the
  prompt-ready recall block without calling the answer LLM; exposed as the
  default-profile `memo_context` MCP tool and gated by `MEMO_CONTEXT_SURFACE`
  (default on). `memo search --explain` (and the `memo_search` `trace` option)
  report per-hit ranking rationale.

## [2.12.17] - 2026-07-06

### Fixed

- Ship Glama/Docker builds with the 13-tool `agent` MCP surface instead of the
  broader `core` surface, keeping the public directory profile focused on the
  agent workflow while preserving the full installed CLI/server for users.
- Improve MCP metadata for agent-profile tools by expanding write/destructive
  tool descriptions and marking `memo_pop_notification` as destructive only for
  its transient notification queue.

## [2.12.16] - 2026-07-06

### Fixed

- Keep Linux CI hermetic on Python 3.13 and 3.14 by installing CPU runtime
  extras and pinning unit-test runtime assumptions.
- Stabilize capture meta-filter coverage by patching the extraction boundary
  instead of an LLM-shaped mock.

## [2.12.15] - 2026-07-06

### Fixed

- Add MCP descriptions for every core-profile tool so directory quality
  scanners and MCP clients receive useful tool metadata instead of empty
  descriptions.

## [2.12.14] - 2026-07-06

Post-Wave-5 robustness fast-follow for the history importers (from the Wave 5
whole-branch review).

### Fixed

- **Per-record error isolation in streaming importers.** A single malformed
  record no longer aborts the whole import. `iter_codex/opencode/chatgpt/
  claude_export_exchanges` now skip-and-continue on any per-record parse error
  (previously e.g. a non-object `message.data` in opencode raised an uncaught
  `AttributeError`). Valid records still import; matches the already-guarded
  mem0/zep path.

### Added

- **Source provenance on imports.** Memories imported by the streaming
  importers (codex / opencode / chatgpt / claude-export) now carry
  `extra.source = "imported:<source>"`, so imports are attributable and
  cleanable later — matching the `imported:mem0` / `imported:zep` markers.

## [2.12.13] - 2026-07-06

Ecosystem roadmap **Wave 5** (final) — benchmarks (workstream I, 7 tasks) +
maintenance-extras (workstream K, 7 tasks) + multimodal-import (workstream H,
8 tasks). **This completes the 11-workstream / 92-task ecosystem-learnings
roadmap** (Waves 1–5). Every new flag is default-off; the recall hot path is
untouched (empty diff); full non-MLX suite green (2960), mypy clean,
whole-branch opus review = SHIP.

### Added

- **Benchmark harness (`memo eval bench`).** LoCoMo / LongMemEval dataset
  schema + parsers, fetch/cache into an isolated store, an idempotent ingestion
  adapter (provenance + back-dating), per-category retrieval scoring via the
  shared `rank_hits` path, and a pluggable QA judge (local MLX default, API
  judge env-gated). `eval_recall` gains ranked metrics **Recall@K / NDCG@K /
  MRR** alongside precision@K.
- **Chunk→parent recall** (`MEMO_SEARCH_CHUNK_PARENT`, default off, eval-gated).
  In explicit `Memory.search`, a chunk hit maps to its parent memory. The 5s
  recall hook is untouched and excludes the reference tier, so this is a no-op
  there even when enabled.
- **Entity de-duplication.** MinHash+LSH blocking (pure-Python, entropy-gated)
  before any LLM merge, `graph.list_entities` / `merge_entity_pair`, and a
  nightly entity-canon dream pass (`MEMO_DREAM_ENTITY_CANON`, default off) that
  records merge counts in the receipt.
- **Consolidation bounds** — per-topic size invariant + synthesis body cap
  (memobase-style; `MEMO_SYNTHESIS_MAX_MEMBERS` / `_BODY_MAX_CHARS`, default 0 =
  off) and a per-folder vault-abstract dream pass (`synthesis_kind=folder_abstract`).
- **Multimodal & imports.** mlx-vlm image captions + mlx-whisper audio
  transcription into ingest (guarded optional deps, deferred imports, SHA256
  cache, opt-in `--include-audio` / caption flags); history importers for
  **Codex / opencode / ChatGPT / claude-export / mem0 / zep** (`memo import …`,
  CLI-gated, normal save path with dedup + provenance).

### Removed

- Placeholder CLIP multimodal stub — captions/transcripts flow through the text
  index, which is the real cross-modal retrieval path.

- Cold-start importers: `memo import codex` (Codex CLI rollouts, resumable
  cursors), `memo import opencode` (opencode SQLite), `memo import
  chatgpt` / `memo import claude-export` (data-export conversations.json) —
  all replay through the same mine-history extraction pipeline — plus
  `memo import mem0` / `memo import zep` store migrators (invalidated Zep
  facts are skipped).
- Ingest: optional mlx-vlm caption pass for images whose OCR yields
  little/no text (`MEMO_VLM_CAPTION_ENABLED`, default off) — text-free
  diagrams/photos become searchable via `<!-- VLM: … -->` blocks and
  caption-bodied orphan-image records (`vlm-caption` tag). Install deps
  with `pip install "mlx-memo[multimodal]"`.
- `memo ingest --include-audio`: vault audio files (m4a/mp3/wav/…) are
  transcribed via mlx-whisper (SHA256-cached) and indexed through the same
  pipeline as notes/PDFs, keeping the raw file as provenance. Opt-in.

### Removed

- The placeholder multi-modal store (`memo multimodal` CLI group and the
  `memo_multimodal_add_image/add_audio/search_images/search_audio/search_all`
  MCP tools). Its "universal embeddings" were a hash-based CLIP stub —
  cross-modal search over them was noise. VLM captions and whisper
  transcripts indexed through the normal text pipeline replace it.
  `memo_ocr_image` is unaffected.

## [2.12.12] - 2026-07-05

Ecosystem roadmap **Wave 4** — profile distillation (workstream B, 5 tasks) +
interfaces (workstream G, 14 tasks). Every new flag is **default-off** except
`MEMO_BRIEFING_PROFILE` (default on but a no-op until the default-off dream pass
produces a profile). Retrieval hot path untouched (empty diff); full non-MLX
suite green, mypy clean.

### Added

- **profile.md distillation (Tier-1 #1).** A nightly `memo dream run` pass
  (`MEMO_DREAM_PROFILE_ENABLED`, default off) distills preference/feedback/
  decision/synthesis memories into char-budgeted, rewritten-in-place profile
  documents (global + per-project) under `memory_dir/_profile/` with memory-id
  provenance, plus a **Standing rules** block graduated from grounding.log
  (cited in ≥K distinct sessions) and retired on resolved contradictions. El
  Briefing injects them at SessionStart via a zero-MLX file read
  (`MEMO_BRIEFING_PROFILE`, opt-out) — the "facts you wouldn't think to search
  for" channel that similarity recall structurally misses.
- **Typed knowledge-graph edges.** `- relation_type [[target]]` link grammar +
  bare `[[wikilinks]]` parsed into the crossref backlinks table at
  save/update/delete/reindex (`MEMO_CROSSREF_INDEX`, default off), with
  prefix-aware reverse traversal and cascade-aware warnings on supersede
  (`memo maintain`) and `memo_delete`.
- **Surgical edits.** `Memory.update(replace=(old,new))` / `append=` +
  `memo_update` params — edit a memory's body without a full rewrite.
- **`memo_offload` / `memo.offload`.** Content-addressed offload of large
  payloads to a `reference`-tier memory with a deterministic typed synopsis
  (recall-excluded; retrievable via drill-down).
- **`memo.integrations.wrap(client)`.** Wrap an LLM client for automatic
  pre-call recall injection + post-call capture (sync + async, idempotent
  re-wrap).
- **Prompt overrides.** All LLM system prompts route through `resolve_prompt`
  (`state_dir/prompts/<name>.md`), byte-identical to the built-in default until
  a user drops an override; captured memories carry a `prompt_version` stamp.
- **`memo maintain undo`.** `memo maintain` is now a Click group with
  timestamped receipts; `maintain undo [--run <ts>]` batch-restores from them.
- **FastMCP tool annotations.** `ToolAnnotations` (readOnly/idempotent/
  destructive hints) swept across every registered MCP tool, with a
  completeness gate.

## [2.12.11] - 2026-07-05

### Fixed

- **Recall daemon starvation on the GPU flock.** The daemon's embed thread
  could block indefinitely in `flock()` on the machine-global GPU lock behind
  batch MLX jobs (capture-stop, refresh-summary, idle-daemon, test suites)
  while holding its internal PriorityLock — every recall bailed at 2.5s
  (`recall_lock_bail` storms) and hooks degraded to subprocess+bm25. The
  daemon now takes a priority lane (`memo-mlx-gpu.prio.lock`): it holds the
  priority flock while waiting for / holding the main GPU lock, and
  non-priority processes probe it and back off, yielding at their next chunk
  boundary. Liveness is kernel-managed (flock drops on process death).
- New `mlx_gpu.gpu_deadline(seconds)` context manager; the daemon wraps all
  MLX work in it so a busy GPU raises `TimeoutError` instead of wedging the
  PriorityLock indefinitely.
- **Reindex `UNIQUE constraint failed: meta.path` skips.** With soft delete
  enabled, a deleted memory's tombstone row still occupies the
  `UNIQUE(meta.path)` index; when a new file (new id) reclaimed the same
  path, the path-collision guard could not see the tombstone and the INSERT
  failed. The guard now looks up stale rows with `include_deleted=True` and
  purges them with `hard_delete`.

## [2.12.10] - 2026-07-05

Ecosystem roadmap **Wave 3** — retrieval UX (workstream F, 13 tasks) + feedback
& eval instrumentation (workstream D, 9 tasks). Every new flag is **default-OFF
or opt-out**; retrieval ranking is unchanged at the shipped defaults (full suite
2813 passed, mypy clean, retrieval eval baseline prec@5 / noise@5 0.0). This
release also carries concurrent default-off recall/token-metering levers already
merged to master (precision-gate, intra-injection dedup, per-session budget
decay, token-meter).

### Added

- **Temporal retrieval.** `save()` can normalize relative dates
  (`MEMO_SAVE_NORMALIZE_DATES`); `store.search` / `Memory.search` gain
  `date_from` / `date_to` filters; `memo_search` accepts a natural-language
  `when` range (new leaf `nl_dates.py`).
- **Timeline primitives + `memo_around`.** `chunks_adjacent` /
  `records_around_created` + `Memory.around()` and an MCP `memo_around` tool for
  neighbor/around-a-memory navigation (`idx_meta_created` now back-fills on
  existing DBs).
- **Epistemic recall labels** (`MEMO_RECALL_EPISTEMIC_LABELS`) — presentation-only
  `⟨type · YYYY-MM⟩` / `⟨~inferred⟩` / `⟨?unverified⟩` prefixes; ranking untouched.
- **Omissions tail** (`MEMO_RECALL_OMISSIONS_TAIL`) — a budget-checked `+N more`
  line so agents don't read budget-trimmed hits as absent.
- **Recency band** (`MEMO_RECALL_RECENCY_BAND_DAYS`) — khoj-style union of the
  newest durable memories at the min_sim floor (eval-gated, default 0/off).
- **Unmatched-term honest-empty gate** (`MEMO_RECALL_UNMATCHED_TERM_GATE`) —
  inject nothing when the top score is weak and no distinctive term matches.
- **Multi-round ask** (`MEMO_ASK_MULTI_ROUND`) — one flag-gated extra retrieval
  round in `_build_ask_context` (recall hook untouched).
- **Next-turn verdicts → implicit feedback.** A heuristic ES+EN reaction
  classifier correlates the user's next turn to the prior recall and, behind
  `MEMO_VERDICT_ENABLED` (Stop-hook only), writes implicit `source_feedback`
  (never clobbers a manual vote).
- **Ablation instrumentation.** `MEMO_RECALL_DISABLE` turns are now stamped as a
  `via="disabled"` cohort; `memo roi` / `memo tokens` report with-vs-without
  cohort deltas (per-cohort turns, grounded-per-turn, re-ask rate).
- **Eval/tuner depth.** Negative labels (`avoid_ids` + `harvest_negative_labels`
  from verdict.log, `memo eval harvest --negatives`); a HyDE eval seam
  (`Cfg.flag_overrides` env-pin + named config `J`); a nightly HyDE A/B pass
  (`MEMO_DREAM_HYDE_TUNE_ENABLED`, overlay-applied, hybrid-hook vetoed); and an
  MLX paraphrase label expander (`memo eval expand-labels`).
- **Quarantine + graduation.** `_uncertain` captures are recall-excluded
  (`MEMO_RECALL_EXCLUDE_UNCERTAIN`, opt-out) and a nightly graduation pass
  (`MEMO_DREAM_GRADUATION_ENABLED`) promotes corroborated ones.

## [2.12.9] - 2026-07-04

Internal simplification of the Wave 2 code — behavior-identical (full suite,
mypy, ruff, and the retrieval eval all unchanged: baseline prec@5, noise@5 0.0).

### Changed

- **Corroboration bump deduplicated.** The `if MEMO_SUPPORT_COUNT: suppress:
  bump_support_batch(..., lift=support_lift())` block was copy-pasted at four
  call sites (near-duplicate save, `topic_key` upsert, and both consolidation
  merge paths). Collapsed into one `record.bump_support_if_enabled(store, ids)`
  helper that owns the flag gate + lift resolution — the store SQL layer stays
  flag-free and the policy lives in exactly one place.
- **Capture provenance deduplicated.** The identical provenance-bag construction
  in `run_capture` and `run_capture_incremental` (session/turn stamp + opt-in
  tool-file arrays) is now one `capture._capture_provenance()` helper.

## [2.12.8] - 2026-07-04

Ecosystem roadmap **Wave 2** — confidence lifecycle (workstream C) + capture
depth (workstream E). Every new flag is **default-OFF** except the signed-off
pure counter `MEMO_SUPPORT_COUNT`; nothing changes retrieval ranking at the
shipped defaults.

### Added

- **Corroboration counter (`support_count`).** A re-asserted memory now
  *counts* instead of discarding the signal: a new `support_count` column in
  `memory_health` (survives `reindex --rebuild`, syncs cross-machine via
  `dump_signal`/`merge_signal` with `max()` merge) is bumped at the three
  existing corroboration sites — near-duplicate save, `topic_key` upsert, and
  consolidation merge. Gated by `MEMO_SUPPORT_COUNT` (default on, pure counter).
- **Bounded confidence lift from corroboration.** `MEMO_SUPPORT_CONFIDENCE_LIFT`
  (default `0.0` = off) lets a re-assertion restore confidence toward — never
  past — neutral `1.0`, so a repeatedly-confirmed fact recovers from an earlier
  contradiction/quality penalty. A ranking input only when enabled.
- **Supersede gate.** `MEMO_SUPERSEDE_SUPPORT_GATE` (default `0` = off): when
  set, `memo maintain` refuses to auto-archive the losing side of a
  contradiction whose `support_count` meets the gate — the pair stays open for
  triage and is reported under `flagged_for_review` in the receipt.
- **Supersede provenance.** Archiving a contradiction loser now stamps
  `superseded_by` + `superseded_at` into the archived `.md`'s `extra` bag
  (portable, best-effort — a stamp failure never un-archives).
- **Mutability classes.** `MEMO_CONTRADICT_MUTABILITY` (default off): an LLM
  "contradiction" verdict between two volatile-class bodies (ports, versions,
  status) is downgraded to an `evolution` (a normal update, not a conflict).
  The volatile regex is deliberately conservative on Spanish text.
- **Absorb-on-recurrence.** `MEMO_SAVE_ABSORB` (default off): a near-duplicate
  save folds into the *existing* record via one bounded LLM call routed through
  the versioned `update()` (rollbackable), growing `proof_count` — instead of
  creating a near-copy. Best-effort; any failure falls back to warn-and-create.
- **`memo invalidate <pattern> --reason`.** Reversible bulk confidence
  weakening: one event (a stack migration, a port change) weakens every
  matching memory (confidence penalty + `_invalidated` tag/stamp), writes a
  receipt of prior confidences, and `memo invalidate --undo` restores them.
  Preview-only without `--yes`. Flag `MEMO_INVALIDATE_PENALTY` (default `0.3`).
- **Procedural memory types.** New durable types `procedure` (how-to workflows)
  and `failure_pattern` (structured Pattern/Context/Wrong/Right mistake notes);
  the capture extractor now mines both.
- **Capture provenance.** Every capture- and mine-extracted memory is stamped
  with `session_id` / `transcript_path` / `turn_hash`, plus structured
  `files_read` / `files_modified` arrays from the session's tool stream (gated
  by the existing `MEMO_CAPTURE_TOOL_EVIDENCE`).
- **By-file search lane.** `Memory.search_by_file` + a `file=` parameter on the
  MCP `memo_search` tool: a high-precision post-filter over the captured
  file arrays. `search()` itself is unchanged.
- **`memo mine-git`.** A deterministic (no LLM) miner that turns a repo's
  fix/revert commits into `failure_pattern` seed memories with commit-SHA
  provenance; resumable per repo.
- **PreCompact capture flush.** A `PreCompact` hook runs `capture-tick --force`
  at the compaction boundary so a long session's insight reaches `.md` before
  early context is destroyed, and the briefing re-fires after a compaction.
  Wired in the plugin and self-healed into `settings.json`.

### Changed

- `bug` memories now join `failure_pattern` and `procedure` in
  `EVICTION_PROTECTED_TYPES` — hard-won failure/how-to knowledge is exempt from
  the nightly prune-floor and LFU-eviction passes even when rarely accessed.

## [2.12.7] - 2026-07-04

### Added

- **Privacy layer (default-on secret redaction).** A new pure `memo.redact`
  module masks provider-prefixed API keys (AWS, GitHub, OpenAI, Anthropic,
  Slack, GCP) and PEM private-key blocks to `****<last4>` and strips
  `<private>…</private>` spans. Redaction runs before persisting on every
  capture path (Stop-hook capture, `capture-tick`, `mine-history`) and on
  `memo ingest` index rows — the vault `.md` on disk is never rewritten, and
  redaction is deterministic so `reindex` reproduces identical masked rows.
  Redacted memories carry a `_redacted` tag. Flags: `MEMO_REDACT_SECRETS`
  (on), `MEMO_PRIVATE_MARKERS` (on), `MEMO_REDACT_ENTROPY` (off, opt-in
  high-entropy tier; hex hashes/ids always exempt).
- **Pre-push secret gate for git sync (`MEMO_SYNC_SECRET_GATE`, on).** Before
  the sync commit, staged `.md` additions are scanned (pattern tier only);
  on a hit the commit+push are blocked, `sync_pending` is stamped with the
  reason, and `memo sync status` / `memo doctor` surface it. Blocks the
  commit (not just the push) so a secret never enters git history.
- **Per-call write scope on `memo_save` (MCP).** New `scope` param:
  `"global"` skips the auto `project:<repo>` tag (memory lands in the global
  recall tier); `"project"`/omitted keep auto-detection; an explicit
  `project:` tag always wins.
- **Project→global retag dream pass (`memo dream retag`, default-off).**
  `MEMO_DREAM_RETAG_GLOBAL_ENABLED` promotes memories that were grounded from
  ≥ `MEMO_DREAM_RETAG_MIN_PROJECTS` other projects to global via the
  tag-only update path (no re-embed, reversible with `memo version rollback`).

### Fixed

- **Git worktrees no longer mint a bogus project tag.** `project.py` now
  canonicalizes a linked-worktree `.git` file to the main repo's toplevel, so
  memories saved from a release worktree (e.g. `/tmp/rel`) tag as the real
  project instead of `project:rel` forever. Pure file reads — the 5s
  recall-hook budget is untouched. Submodule and non-worktree `.git` files
  keep their own-basename behavior.

## [2.12.6] - 2026-07-03

### Fixed

- Pin `transformers<5.13` on the Apple Silicon install: transformers 5.13
  broke `AutoTokenizer.register(str)` which mlx-lm (≤0.31.3) still calls at
  import, so any fresh `mlx-memo` install crashed on the first embed
  (`AttributeError: 'str' object has no attribute '__module__'`). 5.12 and
  below verified working; lift the cap when mlx-lm ships a compatible release.

## [2.12.5] - 2026-07-03

### Fixed

- eval recall now mirrors the hook's reference-tier SQL exclusion
  (`MEMO_RECALL_EXCLUDE_REFERENCE`): the eval's explicit `mem.search()` pool
  included bulk `reference` chunks the live hook never surfaces, so ingested
  vault content (e.g. WhatsApp conversations) crowded top-K and false-failed
  the regression gate (prec@5 0.884 → 0.845 with zero production impact).
  With the exclusion mirrored the gate passes at 0.903.

## [2.12.4] - 2026-07-03

### Fixed

- `build_labels()` (tuner objective) now passes the curated document's
  `noise_tags`/`noise_path_fragments` into the LabelSet — same fix as the
  curated gate: the knob line-searches' noise@K was a vacuous 0.0.
- Capture `retyped` counter compares against the normalized claimed type —
  a whitespace-only extractor type no longer counts as a spurious retype.

### Changed

- Coverage floor ratcheted 64 → 68 (measured 70% after the Q3 programs).

## [2.12.3] - 2026-07-03

### Fixed

- **Recall daemon latency tail.** Two real causes fixed: (1) priority drift —
  the `embed_query` branch acquired the daemon lock at priority 1 (same as
  interactive recall) with a 60s timeout, so an embed burst (memflow vec
  indexing, eval runs, grounding scoring) queued at recall's own priority and
  burned the hook's 2.5s budget; `embed_query`/`search`/`embed_batch` now run
  at priority 0 and interactive recall genuinely outranks them. (2) Cold MLX
  load no longer counts against queued recalls: the model warms in a
  background thread at daemon start; while warming, recall ops bail `{}` fast
  (hook falls back to subprocess, same as socket-absent) and non-latency-bound
  ops wait the event out. The socket still binds immediately, so
  `memo recall-daemon start`'s probe succeeds on cold starts. Observability: a
  recall waiting >500ms or bailing on lock-busy emits one structured stderr
  line (`recall_lock_wait`/`recall_lock_bail`/`recall_warming` with `held_by`).
  A timed-out high-priority waiter now wakes re-slept priority-0 waiters
  (liveness gap).
- `memo dream status` renders the `capture_weights` fragment; `memo
  debug-recall` renders the `synthesis_boost`/`mmr` explain stages; capture
  gains a `retyped` counter; `type_weights.json` staging file is pid-suffixed;
  the tuner's curated gate now measures real noise@K (document-level
  `noise_tags`/`noise_path_fragments` reach the LabelSet).

## [2.12.2] - 2026-07-03

### Fixed

- Hook-side session writers (`stamp_recall_turn`, `mark_ids_recalled`) now
  refresh the snapshot's `updated` timestamp — a session that only ever got
  hook stamps sorted oldest-by-`updated` and the session-cap GC evicted it,
  wiping its `recalled_ids` before Stop-hook cited-grounding could match.

## [2.12.1] - 2026-07-03

### Fixed

- **Cited-grounding was dead on the daemon (production) recall path.** Only
  the subprocess fallback marked session `recalled_ids`; the warm daemon —
  which serves virtually every recall — never did, so `match_cited` always
  compared against an empty map and not one `method="cited"` grounding row
  existed (409/409 recalled-never-cited). `_recall_logic` now mirrors the
  subprocess path exactly: session dedup (already-recalled hits are not
  re-injected — also a token saving every turn) + `mark_ids_recalled`.
  End-to-end regression test covers daemon recall → citing answer →
  `method=cited` row.

## [2.12.0] - 2026-07-02

### Added

- **Recall-faithful eval for boosts.** `knobs_from_flags()` is the single
  source of `RankKnobs` resolution (env > tuned overlay > registry default),
  shared by the daemon recall path, the eval harness, and the hook; the
  extracted `apply_injection_filters()` (skip-below / gap trim) is shared by
  hook and eval so injection semantics cannot diverge. The eval grid gains
  MMR / synthesis variants (`E mmr/0.3` … `I synth/0.10`) and
  `Cfg.knob_overrides`, so any ranking knob is measurable offline.
- **Project-aware grounding and labels.** `grounding.log` entries record the
  session's `project` tag; harvested eval labels carry it through, and the
  eval ranks project-carrying labels with the hook-faithful `project_tag` —
  project/global boosts are now measurable offline per-label.
- **Nightly tuner searches the new knobs.** `MEMO_RECALL_MMR_LAMBDA`
  (0/0.3/0.5/0.7) and `MEMO_RECALL_SYNTHESIS_BOOST` (0/0.05/0.10) join the
  line-search under `MEMO_DREAM_TUNE_ENABLED`, with a curated no-regression
  gate, a latency gate (candidate p50 must stay within +25% of baseline),
  the single-apply guard (one knob change per night across all knobs),
  per-knob baselines, and the knob-generic online revert.

### Changed

- **Recall-hook subprocess fallback unified onto `rank_hits`.** The inline
  ranking chain that predated `rank_hits` is gone; daemon and subprocess
  paths now produce identical injections for identical inputs (parity-tested,
  including with MMR/synthesis/preference knobs on).
  `MEMO_RECALL_STALENESS_DAYS` and `MEMO_RECALL_ADAPTIVE_CONTEXT` had no
  daemon-path consumer and are now inert — marked deprecated in the registry.

### Fixed

- Tuner online-revert now merges into the scalar overlay, so bool/str levers
  (e.g. `MEMO_RECALL_MODE`) survive a revert instead of being dropped.

## [2.11.0] - 2026-07-02

### Added

- **Recall observability pipeline (Q3 Mes 1).** Two nightly dream passes:
  `harvest_labels` mines eval labels from real citations in `grounding.log`,
  `eval_recall` runs a retrieval-only eval (prec@K / noise@K) against
  curated + harvested labels, recording to the dream receipt and
  `state_dir/eval/history.jsonl` (`MEMO_DREAM_EVAL_ENABLED`, default on;
  `MEMO_DREAM_EVAL_MAX_LABELS`, default 200).
- **Recall-hook latency metrics.** Both hook paths (warm daemon and
  subprocess fallback) stamp `total_ms`/`path`/`hits` to
  `state_dir/recall_metrics.jsonl` (`MEMO_RECALL_METRICS`, default on);
  `memo stats` shows p50/p95/p99 per path over the last 7 days.
- **`memo debug-recall <prompt>`.** Diagnostic command reproducing the
  recall pipeline outside a session: per-hit vec / BM25 / rerank-fused
  scores, boost deltas, floor verdicts, and active thresholds
  (including `skip_below` / `gap_threshold`); `--json` for scripting.
- **TUI recall-quality panel.** prec@5 trend sparkline, top cited
  memories, and recalled-but-never-cited count in `memo tui`.
- **Capture hygiene (Q3 Mes 2).** Meta-commentary filter drops process
  narration and trims filler openers keeping the substance
  (`MEMO_CAPTURE_META_FILTER`, default on); intra-batch near-dup window
  collapses retry twins before the store check
  (`MEMO_CAPTURE_BATCH_DEDUP`, default on); type-classification
  confidence stamped in `extra.capture_confidence`, low-confidence
  captures tagged `_uncertain` below `MEMO_CAPTURE_MIN_CONFIDENCE`
  (default 0.0 = off).
- **Citation-type feedback.** Nightly `capture_weights` dream pass
  computes per-type citation rates into
  `state_dir/capture/type_weights.json`; the capture classifier consults
  them as a tie-breaker in ambiguous classifications
  (`MEMO_CAPTURE_TYPE_FEEDBACK`, default off).
- **Reference-tier search floor.** `MEMO_REFERENCE_SEARCH_FLOOR` (default
  0.0 = off) keeps bulk-ingested reference chunks out of
  search/ask results unless they clear a higher score bar; durable
  memories unaffected.
- **Ranking knobs, evidence-gated (Q3 Mes 3), all default off.** MMR
  diversity re-rank (`MEMO_RECALL_MMR_LAMBDA`), synthesis-type boost
  (`MEMO_RECALL_SYNTHESIS_BOOST`), and cited-weighted outcome utility
  (`MEMO_OUTCOME_CITED_WEIGHT`, active only inside the
  `MEMO_OUTCOME_RANKING_ENABLED`-gated path).

### Fixed

- `packaging/mcpb/manifest.json` was left at 2.10.1 by the v2.10.2
  release; version-sync test now green.
- Recall-metrics `hits` now counts the post-dedup injected hits on both
  hook paths (comparable daemon vs subprocess).
- Nightly eval cross-dedups harvested labels against the curated set so
  a prompt never double-weights prec@K.

## [2.10.2] - 2026-07-02

### Changed

- Test suite: `ruff format` pass across all 143 test files (style-only, no logic changes).

## [2.10.1] - 2026-07-02

### Fixed

- Presence polish pass: per-process tmp file for `presence_today.json` writes
  (no lost updates between concurrent writers); `systemMessage` titles are
  forced single-line; cited-id parsing accepts uppercase hex (normalized);
  memories cited from earlier turns are grounded even when the current turn
  had no recall hits; statusline date extraction is GNU/BSD-portable;
  "stale memories archived" label; broader statusline + full-hook test
  coverage.

## [2.10.0] - 2026-07-02

### Added

- **Presence program** — memo's background work is now visible to the human:
  - Recall hook emits a human-visible `systemMessage` line (`🧠 memo · N: titles`)
    on every prompt with recall hits, on both the subprocess and warm-daemon
    paths (`MEMO_RECALL_SYSTEM_MESSAGE`, default on).
  - Injected recall block asks the model to cite used memories inline by short
    id (`MEMO_RECALL_CITE_INSTRUCTION`, default on); cited `[id8]` references
    are validated against session-recalled ids and logged as the strongest
    grounding signal (`method="cited"`, used_score=1.0), feeding roi/usefulness.
  - Statusline badge shows today's live activity —
    `[Memo <ver> · 🧠recalls · 💾saves · ~tokens saved]` — from a new
    `presence_today.json` state file written by the recall hook, `Memory.save()`,
    and capture-stop (`MEMO_STATUSLINE_ACTIVITY`, default on).
  - SessionStart briefing gains a one-shot "☾ Last night" digest of the nightly
    dream receipt (`MEMO_BRIEFING_DREAM_DIGEST`, default on).
  - `memo install-mcp --write` seeds one real memory recording the install and
    the first briefing surfaces it once ("🧠 memo remembers…") — onboarding
    proof powered by the real recall mechanism.

### Fixed

- Cited-id grounding matches the 8-char truncated ids stored in
  `recall_hook.log` (prefix-normalized) — no dead comparisons, no duplicate
  grounding rows.
- Install-seed side effect is isolated from the shared test state dir.

## [2.9.8] - 2026-07-02

### Fixed

- **Sync can no longer destroy memories through a stale interrupted rebase.** A rebase killed mid-conflict (git timeout, machine sleep, session teardown) left `.git/rebase-merge` behind; the next sync matched `--skip` in git's "already a rebase in progress" error and ran `git rebase --skip` — silently dropping the local memories commit and committing raw conflict markers into `.md` files, which then pushed to every other machine. Sync now aborts a stale rebase before pulling (reported as `stale_rebase_aborted`), `_commit_local` refuses to commit unmerged paths, and the recovery loop treats "already a rebase" as a hard error instead of a skippable conflict.
- **`memo sync pull` / `memo sync push` now take the machine sync lock.** Both called the git layer directly, bypassing the `.sync.lock` flock that `memo sync once` holds — a SessionStart pull could drive or abort a concurrent Stop-hook rebase mid-flight. They now route through the same locked coordinator and soft-skip when another sync owns the lock. Git subprocess timeouts (`subprocess.TimeoutExpired`) are wrapped as `SyncGitError`, so `sync_once`'s "never raises" contract and the `--quiet` hook paths hold under a blackholed network.
- **Path traversal via `project:` tags is closed.** A tag like `project:../../evil` was used verbatim as the on-disk bucket, letting `memo save`/`memo_save` create directories and plant `.md` files outside the vault (and `memo migrate --bucket-by-project` rename memories out of it — data loss on the next reindex). The derived folder is now slugified (the tag itself is stored unchanged), and both the save and migrate paths enforce a containment check against `memory_dir`.
- **Recall daemon startup is race-free.** `recall-daemon start` fired concurrently (SessionStart hooks, launchd respawns) could orphan a live MLX daemon on an unlinked socket and later have the orphan's shutdown unlink the survivor's socket, silently killing warm recall. `run_server` now takes the same non-blocking startup flock as the maint/ingest daemons, and daemon cleanup only removes socket/pid files it owns (a live foreign owner's files are left alone). The idle-capture daemon got the same child-side guard against duplicate loops.
- **`gpu_guard(timeout=...)` actually times out across processes.** The deadline only covered the in-process RLock; the cross-process file lock was a plain blocking `flock`, so one stuck MLX pass wedged every other memo process (recall daemon, CLI, MCP) machine-wide with no `TimeoutError`. The flock acquisition is now bounded by the same deadline.
- **Token ledger survives concurrent Stop hooks.** `write_ledger` used a fixed tmp filename, so two sessions rolling up simultaneously could replace the ledger with a half-written file — `read_ledger` then silently reset to empty and historic days beyond the grounding-log window were permanently lost. Writers now use unique temp files and the read-merge-write runs under a sidecar flock, matching the repo's dashboard-logs pattern.
- `memo release bump`/`sync` now edit `packaging/mcpb/manifest.json` (version + pinned `mlx-memo` spec) — `release check` validated it but `bump` never wrote it, so every release tripped the check.
- **`device_id` first-run mint is atomic.** Concurrent first sessions could mint divergent machine ids (phantom attribution in history events) or read an empty id mid-write; the id is now published via an exclusive atomic link and losers adopt the winner's id.

## [2.9.7] - 2026-07-01

### Fixed
- Release version alignment now includes the internal Codex plugin bundle, with `memo release sync` to realign drifted manifests to `pyproject.toml` and `memo update` refreshing installed static agent artifacts for Claude Code, Codex, and Devin after a successful runtime update.
- The retired desktop-agent surface was removed from CLI, docs, tests, and installer flows; use Devin Desktop via `--client/--agent devin-desktop`, `DEVIN_DESKTOP_MCP_CONFIG`, and `~/.devin/mcp.json`.
- The HTTP API no longer hardcodes its OpenAPI and `/health` version; both use the runtime `memo.__version__`.
- `memo release bump` detects surplus version fields again (the `re.subn` count cap had silently disabled over-match detection), and `memo release check` validates every `server.json` package instead of only the first.
- `memo update` no longer masks a failed `claude plugin install` as "already handled" — a real failure now surfaces as a warning and the refresh is reported honestly.
- Three dead flags are actually wired now: `MEMO_TANTIVY_ENABLED=0` really forces FTS5-only (it previously did nothing — only `MEMO_FTS_BACKEND` gated the index), `MEMO_SEARCH_JSON_BODY_CHARS` drives the default `--body-chars` of `memo search`/`memo recall`, and `MEMO_DREAM_MINE_LIMIT` caps the labels mined per dream tuning pass (was hardcoded at 200).
- Unix-socket daemons (recall/ingest/maint) can no longer hang indefinitely on SIGTERM: handler threads are daemonic, so the shutdown `join_timeout` is a real bound (stdlib `ThreadingMixIn` defaults joined in-flight handlers unbounded in `server_close()`).
- Dream `eviction`/`compress` DB errors now propagate into the receipt's `errors` list instead of reading as "nothing to do".
- The HTTP API's lazy `Memory` init is thread-safe — concurrent first requests no longer construct duplicate `Memory` instances (duplicate sqlite connections + embedder load).
- Historical CHANGELOG entries no longer retroactively claim Devin Desktop support for releases that actually shipped the retired client's paths.
- Test hygiene: runtime-isolation tests run against a sandboxed `$HOME` (they wrote real `~/.memo/bin` shims and `~/.zshrc` PATH snippets on every run) and no longer depend on the developer machine's live index for `MEMO_EMBEDDER_DIMS`; the agent-artifact "skip" test is hermetic and uses recording stubs that can actually fail.

## [2.9.6] - 2026-07-01

### Fixed
- **`install-mcp` now merges JSONC agent configs.** Comments (`//`, `/* */`) and trailing commas are tolerated when merging an existing config (Zed `settings.json`, VS Code, Cursor). Previously these raised "not valid JSON". Comments are not preserved on rewrite; the config data is.

## [2.9.5] - 2026-07-01

### Fixed
- **CI `mypy` step is green again.** `fastapi` / `uvicorn` are optional (`[http]`) dependencies not installed in the default `[dev]` CI runtime, but `server_http.py` / `cli_http.py` import them at module level — so `mypy src/memo` failed with `import-not-found` (a pre-existing gap, unrelated to the 2.9.4 changes). Added them to the `[[tool.mypy.overrides]]` `ignore_missing_imports` list, matching how the `[cpu]`/MLX optional extras are already handled. No runtime change.

## [2.9.4] - 2026-07-01

### Added
- **MCP server surface is now under test.** 21 new `tests/test_server_*.py` files cover the previously-untested `server_*` domains (analytics, asof, backup, cache, collaborative, contextual, contradict, core_history, entities, episodes, feedback, graph, import_export, links, multimodal, query, reflect, repo, sync, temporal, version) — each mocks `Memory`, invokes every tool through its `register()`, and asserts the returned envelope. Measured suite coverage rose ~60% → 67%, and the coverage floor (`fail_under`) is ratcheted **58 → 64**.

### Fixed
- **`delete()` rollback no longer drops the sqlite-only dedup keys.** `topic_key` / `normalized_hash` live only in the sqlite index (never in the `.md` frontmatter), and `store.get()` omits them — so a failed-final-unlink rollback restored the row *without* them, and a later same-topic save would then create a **duplicate** instead of updating in place. A new `VecStore.get_dedup_keys()` (mirroring `get_embedding_blob` / `get_fts_body`) pre-fetches them so the rollback restores the row faithfully.
- **`delete()` mutates derived/audit state only after the authoritative unlink.** History logging, graph-edge drop, and receipt/event emission now run strictly *after* the canonical `.md` is removed. A failed unlink therefore rolls back cleanly with **no spurious `delete` audit event** and no dropped graph edges for a memory that actually survives the failure.

### Changed
- **`cli_dream.py` god-file thinned** (1228 → 1142 LOC): five extractable helpers (`_state_path`, `_older_id`, `_corpus_fingerprint`, `_make_progress`, `_render_run_summary`) moved into `cli_dream_passes.py` and re-exported, so existing imports keep resolving. Behaviour-preserving; the CLI file stays wiring-only per the repo convention.

## [2.9.3] - 2026-07-01

### Fixed
- **`memo dream run` no longer emits misleading warnings for by-design behavior.** Two `WARNING`-level log lines fired on every manual dream run despite the run succeeding (`receipt["errors"]` stayed empty):
  - *Near-duplicate save nag.* The save-time dedup advisory (`consider `memo update` instead`) is only actionable for an interactive human, but dream's signal-gather/synthesize passes save near-duplicates by construction (the same run's consolidate pass merges them). A new `derived_save_scope()` (a `ContextVar` in `memory/record.py`) marks batch/derived saves; `dream run` wraps its whole pipeline and `apply_merge` wraps its merged save, so the nag drops to `DEBUG` in those paths. A human's direct `memo save` still gets the warning.
  - *Consolidation merge-proposal JSON unparseable.* The retry only flipped decode temperature (0.0→0.3), which cannot recover a proposal truncated by too small a token budget. The retry now escalates **both** temperature and `max_tokens` (1536→3072) so a long merged body completes instead of being cut mid-value; skipping the cluster remains the safe final fallback (originals untouched, retried next night).

## [2.9.2] - 2026-07-01

### Fixed
- Recover `transcript_path` by `session_id` when hook payloads omit it — since 2026-06-27 some Stop/UserPromptSubmit hook payloads stopped carrying `transcript_path` while `session_id` kept arriving, silently starving `capture-stop`'s grounding scoring (and therefore the token-savings ledger `memo tokens` reads), session autosave's payload persistence, and session checkpoint's snapshot fields. Recovered via a `~/.claude/projects/*/<session_id>.jsonl` glob fallback.

## [2.9.1] - 2026-07-01

### Changed
- The nightly `memo dream run` LaunchAgent template now enables the self-improving recall tuner by default (`MEMO_DREAM_TUNE_ENABLED` + `MEMO_DREAM_TUNE_BOOST_ENABLED`). Every applied change is verified against real grounding by the online proof loop and reverted if it regresses, so on-by-default is safe. Reversible by removing the flags from the plist.

## [2.9.0] - 2026-07-01

### Added
- **F4 consolidate-reuse metric** (`memo dream consolidate-reuse`): read-only report of whether the `type=synthesis` memories the episodic-consolidation pass creates actually get grounded/reused in real recall (n_consolidated / n_reused / reuse_fraction). Measures the value of consolidation with real data.
- **Online-only project-boost explorer** (`MEMO_DREAM_TUNE_BOOST_ENABLED`, OFF by default): nudges `MEMO_RECALL_PROJECT_BOOST` and lets the online proof loop confirm/revert it against real grounding. Boosts are not offline-measurable (the label eval has no project context), so this knob is tuned purely by real outcomes — no offline gate. Rides the generic proof loop (generic per-knob revert). Direction is hill-climbed from the ledger (repeat confirmed, reverse reverted).

### Fixed
- Online proof-loop revert self-heals the overlay's `_meta.prev` so the offline rollback-guard cannot resurrect a just-reverted config under index drift.

## [2.8.2] - 2026-07-01

### Fixed
- Online proof-loop revert now self-heals the tuned-params overlay's one-step `_meta.prev`, so a later offline rollback-guard can no longer resurrect the config the online loop just reverted away under index drift. (Gated behind `MEMO_DREAM_TUNE_ENABLED`, OFF by default.)

## [2.8.1] - 2026-07-01

### Fixed
- Proof-loop deferral is now checked BEFORE the (expensive) MLX search in the graph tuner passes: when a min_sim change is being proven or a revert cooldown is active, the graph passes skip the search entirely instead of grid-searching and only then deferring. Surfaced by end-to-end empirical testing of the proof loop. (Still gated behind `MEMO_DREAM_TUNE_ENABLED`, OFF by default.)

## [2.8.0] - 2026-07-01

### Added
- **Generic online proof loop** for the recall self-tuner — out-of-sample grounding verification now covers any tuned knob, not just `min_sim`. The graph-proximity-weight tuner joins the proof loop: each applied change is confirmed or reverted by real grounding under its new params version, with a knob-generic revert that restores the correct per-knob offline baseline. A one-cycle revert cooldown stops a co-gated pass from re-applying a just-reverted value the same night. (All gated behind `MEMO_DREAM_TUNE_ENABLED`, OFF by default — no default behavior change.)
- Proof-loop ledger, `memo dream status`, and `memo dream timeline` now label each entry by the tuned knob (`min_sim` / `graph_proximity_weight`).

## [2.7.0] - 2026-07-01

### Added
- **Self-improvement proof loop** for the recall self-tuner — all gated behind `MEMO_DREAM_TUNE_ENABLED` (OFF by default), no default behavior change. Each nightly `min_sim` change the tuner applies is now judged out-of-sample by the real grounding accumulated under its new tuned-params version, reverted if that regresses, and recorded in a durable ledger.
  - `params_version` attribution stamped on every grounding row; `memo eval baseline` snapshots offline precision/noise + online grounded/tokens (7d/30d) + the active params version.
  - Online-guarded confirm/revert/wait/expire verdicts (`resolve_pending`), with a self-contained revert that restores the pre-apply floor and offline baseline.
  - One overlay change per proof cycle: the graph tuner passes defer (`deferred_pending`) while a `min_sim` change is being proven, so its grounding cohort is never orphaned.
  - Graduation-readiness checker surfaced in `memo dream status`; `memo dream timeline` renders the proof-loop history with realized online impact.
- Flags: `MEMO_DREAM_TUNE_MIN_COHORT` (20), `MEMO_DREAM_TUNE_ONLINE_EPS` (0.02), `MEMO_DREAM_TUNE_GRADUATION_K` (5).

## [2.6.11] - 2026-06-30

### Fixed
- `memo init` now exits cleanly with guidance on a non-TTY / non-interactive shell instead of crashing inside the interactive picker.
- LLM features (`ask` / `synthesize` / `dream`) on the CPU (non-MLX) backend now print the "requires the MLX runtime" guidance as a clean error instead of an uncaught traceback.
- MCP `serverInfo.version` now reports memo's own version instead of the FastMCP framework version.

### Changed
- README: corrected the CLI command count (95 → 105) and the full MCP surface count (123 → 126).

## [2.6.10] - 2026-06-30

### Fixed
- Dashboard historic tokens-saved now reads from the durable per-day ledger (`token_ledger`) instead of the capped `grounding.log`, so the gerencial headline keeps growing like `memo tokens` instead of plateauing once old grounded rows rotate out of the log.

### Performance
- Reranker: reverted the batched cross-encoder forward (it regressed the configured 4B model) while keeping the per-pair head-slice in `score()` — projecting only the last token through the LM head.

## [2.6.9] - 2026-06-30

### Added
- `memo tokens` — TUI showing how many tokens memo saved today / this month / all-time, with big-number panels (HOY/MES/HISTÓRICO) plus daily and monthly bar charts and a month-over-month growth indicator (`--json` for machine output). Savings are attributable to memo alone: it counts *grounded* recalls — surfaced memories the answer actually used (re-derivations memo prevented) — times `MEMO_ROI_TOKENS_PER_GROUNDED`, so the total rises as memo accumulates more useful memories.
- Durable token-savings ledger (`token_ledger.py`, `state_dir/token_savings_daily.json`). `grounding.log` is capped (~12 days) and rotates, so an all-time total read from it alone would plateau; the ledger folds grounded events into a monotonic per-day file before they evict, giving `memo tokens` a durable, ever-growing historic total. Rolled up on the Stop hook and on demand.

## [2.6.8] - 2026-06-30

### Added
- `[Memo <ver>]` badge in opencode. opencode has no native statusline/tagline slot for custom text (plugin status-bar widgets are an open feature request), so `startup-banner --agent opencode` now stamps the live memo version into opencode's `username` config (`<base> · [Memo <ver>]`), shown next to each user message. Idempotent upsert into the pure-JSON `opencode.json` (merges with `opencode.jsonc`); no-op when opencode is absent.

## [2.6.7] - 2026-06-30

### Added
- `memo recall-daemon restart` subcommand (was only start/stop/status). Launchd-aware: SIGTERM via stop, then waits up to ~5s for a KeepAlive respawn — if a new live PID appears it defers to launchd instead of spawning a competing daemon; otherwise it starts a fresh one. Use after a runtime upgrade so the daemon reloads new code.
- Successful memo updates notify Codex with `Plugin updated: memo · Run /reload_plugins to apply`.

### Changed
- Startup banner and statusline badges now render as `[Memo <version>]` instead of `[MEMO <version>]`.

### Fixed
- Codex shims now also emit a delayed `[Memo <version>]` Codex/Supacode notification so the memo version remains visible after the Codex TUI takes over the terminal.
- Agent shims now always render memo's own startup banner before delegating to downstream wrappers or binaries, and refresh stale PATH snippets so memo stays ahead of downstream wrappers.

## [2.6.6] - 2026-06-30

### Fixed
- Recall-daemon `embed_query` no longer bails after a hardcoded 5s wait for the shared embedder lock — shorter than a single 4B `embed_batch` chunk hold (60s). A heavy job like `memo dream run`, routing through the auto-detected warm daemon, self-contended: its own batch embed held the lock while its interleaved `embed_query` timed out at 5s and paid a redundant ~2s in-process cold MLX load (the `timeout acquiring lock` → `falling back to in-process` warning pair). New `MEMO_EMBED_LOCK_TIMEOUT_MS` (default 60000) matches the `embed_batch` hold so `embed_query` waits out the in-flight chunk instead of bailing. Mirrors the existing `MEMO_RECALL_LOCK_TIMEOUT_MS` pattern; background callers aren't latency-bound so the longer wait is free.

## [2.6.5] - 2026-06-30

### Fixed
- The MCP-registry `server.json` description is back under the registry's 100-character limit (2.6.4's longer Linux-mentioning description was rejected with HTTP 422, so the registry stayed at 2.6.3). Shortened to "Memory for AI agents — MLX (Apple Silicon) or CPU (Linux), sqlite-vec + BM25, zero cloud." — still names the Linux path. PyPI/plugin descriptions (no length cap) keep the fuller wording.

## [2.6.4] - 2026-06-30

### Changed
- Documentation sweep for Linux/Ubuntu: the PyPI/plugin/MCP-registry package descriptions, the README (hero, hybrid-search line, diagram alt text), and `docs/reference.md` + `docs/install-new-mac.md` now state that memo runs on Apple Silicon (MLX) **or** Linux/Ubuntu (CPU `sentence-transformers`), instead of implying macOS-only. The "when not to pick memo" note no longer lists "not on Apple Silicon" as a disqualifier (memo runs standalone on Linux; only the reranker + LLM features are MLX-only).

## [2.6.3] - 2026-06-30

### Fixed
- The MLX-only cross-encoder reranker is now correctly forced **off** on non-Apple-Silicon hosts. The default `MODEL_PROFILES` set `reranker_enabled: True`, which overrode the platform-aware field default — so on Linux every hybrid search tried to load the MLX reranker, failed (`No module named 'mlx_lm'`), and fell back. `Config.from_env()` now gates it: an explicit `MEMO_RERANKER_ENABLED` wins, otherwise it is off when MLX is unavailable. Found by the new `linux-cpu-smoke` CI job running a real Ubuntu `[cpu]` install.

### Added
- `linux-cpu-smoke` CI job (ubuntu-latest): real `pip install ".[cpu]"`, sqlite-vec/FTS5 load, the actual Qwen3-Embedding-0.6B model, `memo doctor`, a save→search roundtrip, and `memo-mcp` build — the end-to-end Linux validation the (stubbed) unit suite can't give.

## [2.6.2] - 2026-06-30

### Fixed
- `memo install-mcp claude-desktop` writes to the per-OS Claude Desktop config path (`~/.config/Claude/...` on Linux, `%APPDATA%` on Windows) instead of the hardcoded macOS path.
- The MCP-config scan (`memo doctor` / mcp-config) now also checks the Linux Windsurf path (`~/.config/Windsurf/User/mcp_config.json`).

### Added
- Linux systemd **user** units in `systemd/` (`memo-dream.service` + `memo-dream.timer` for nightly maintenance, `memo-watch.service` for auto-reindex) — the counterpart of memo's macOS launchd agents. See `systemd/README.md` and `docs/ubuntu.md`.

## [2.6.1] - 2026-06-30

### Fixed
- `memo doctor` / `memo doctor --json` no longer false-fail (exit 1) on a healthy Linux/Ubuntu `[cpu]` install: the import probe is now backend-aware — it checks `sentence-transformers` on the CPU backend instead of MLX, and the model-cache report lists `st_embedder_model`.
- `memo install-watcher` / `uninstall-watcher` give a clean "launchd is macOS-only" message on Linux instead of an uncaught `FileNotFoundError` (no `launchctl`).
- `memo search --rerank` warns and skips on non-Apple-Silicon instead of trying to load the MLX-only cross-encoder.
- `RepoCorpus` (and the opt-in ingest daemon) routes the embedder through `make_embedder`, so it uses the CPU backend on Linux instead of hard-constructing `MLXEmbedder`. SessionStart prewarm likewise warms the active backend.
- Obsidian vault auto-detection reads the per-OS registry path (`~/.config/obsidian/obsidian.json` on Linux, `%APPDATA%` on Windows).

## [2.6.0] - 2026-06-30

### Added
- Linux/Ubuntu (and Intel mac) support via a CPU `sentence-transformers` embedder backend (`STEmbedder`), auto-selected off Apple Silicon. Semantic search, recall, and save work without MLX as a **standalone** corpus. New `embedder_select.make_embedder` is the single MLX-vs-CPU decision point, routed through the `Memory` facade, the `embedder_client` daemon fallback, and `memo ingest`.
- `MEMO_EMBEDDER_BACKEND` (`auto`/`mlx`/`st`) and `MEMO_ST_EMBEDDER_MODEL` (default `Qwen/Qwen3-Embedding-0.6B`, 1024-dim — same family/dims as the MLX quant, so the vec0 schema and asymmetric query prefix are unchanged).
- `pip install "mlx-memo[cpu]"` extra, `scripts/install-ubuntu.sh` (uv/pipx), and `docs/ubuntu.md`.

### Changed
- `reranker_enabled` now defaults ON only on Apple Silicon (the cross-encoder is MLX-only); on other hosts hybrid search returns fusion-ranked results without the rerank pass.
- LLM features (`ask`/`synthesize`/`dream`) raise a clear `MemoError` off Apple Silicon instead of an opaque import failure; search/recall/save are unaffected.
## [2.5.0] - 2026-06-30

### Added
- Graph-native program: associative recall nudge (spreading activation over the merged entity + codegraph graph with IDF hub-damping), entity-centric "Knowledge map" briefing section, nightly community-synthesis dream pass (off by default, `MEMO_DREAM_COMMUNITIES_ENABLED`), and `memo graph explore` / `memo_explore` knowledge exploration.
- codegraph migration: memo navigation merges colbymchenry/codegraph's symbol graph as the primary layer (`MEMO_GRAPH_USE_CODEGRAPH`).

### Fixed
- Associative recall nudge now renders on the warm-daemon (primary) recall path, not only the subprocess fallback (it was dormant in normal operation).
- `compute_centrality` rebuilt the graph adjacency O(N²) times (hung on real corpora); now builds it once.
- Storage: `get_by_path` / `get_by_path_ci` and prune/eviction candidate queries no longer return soft-deleted rows (path-collision false positives, UNIQUE constraint crashes, inflated eviction pools).
- HyDE search reuses the shared MLX chat instead of cold-loading a second copy of the LLM.
- `llm.chat_stream` no longer holds the GPU lock across the entire streamed response.
- `AmbiguousIdError` guards on `memo_related` (and the HTTP/CLI paths); restore tolerates a corrupt manifest; numerous CLI/MCP/flags/dream correctness fixes from a whole-project adversarial audit.

### Changed
- English-only user-facing strings (associative nudge); removed dead code (`find_node_fuzzy`, `auto_update_on_commit`, `get_session_metadata`); gitignore scratch output.

## [2.4.3] - 2026-06-29

### Added
- **Dream convergence guard**: `memo dream run` stamps a corpus fingerprint and, on a re-run where nothing changed (signal-gather found nothing new and the corpus is unchanged), skips the expensive contradict/synthesize/consolidate passes — a re-run on an idle corpus is now near-instant instead of redoing identical LLM work. `--force` runs every pass regardless.
- **Dream single-owner lock**: a second `dream run` (manual, or the `com.memo.dream` LaunchAgent firing while one is in flight) now skips instead of racing on the shared sidecar DBs and clobbering the receipt.

### Changed
- Dream signal-gather now uses the exact last-run timestamp for its transcript lookback instead of a day-rounded window that re-mined ~1–2 days every run.

## [2.4.2] - 2026-06-29

### Changed
- Removed the Spanish command aliases `memo historia` / `memo mapa` (use `memo record-history` / `memo map`); the CLI surface is now English-only.

### Fixed
- Silenced HuggingFace hub progress bars globally (set `HF_HUB_DISABLE_PROGRESS_BARS` at import). Model loads, prewarm, daemon startup and self-update no longer leak repeated "Fetching N files / Download complete 0.00B" noise for already-cached models.

## [2.4.1] - 2026-06-29

### Fixed
- **Split graph DB healing**: an install that ran an interim build (the table rename without a migration) ended up with both `entity_memoria` (legacy data) and an empty `entity_memory`, splitting the knowledge graph and losing historical entity links to recall. The migration now folds the legacy rows into `entity_memory` (deduped) and drops the legacy table when both are present, not only when the new table is absent.

## [2.4.0] - 2026-06-29

### Changed
- **English-only codebase** for public release: Spanish identifiers renamed to English across the code (DB table `entity_memoria`→`entity_memory`, column `memoria_id`→`memory_id`, output/API keys), remaining UI strings translated, and the `engram` name removed (module → `session_patterns`). On-disk corpus dir default `memorias/`→`memories/`.

### Fixed
- **Existing-database migrations**: pre-rename SQLite DBs are auto-migrated in place (`pairs.memoria_id_a/b`, the `entity_memoria` table + `memoria_id` column, `versions.memoria_id`) so upgrades no longer crash with `no such column` (which surfaced as a warn in `memo dream`) or silently orphan graph data.
- **Soft directory migration**: new installs use `memories/`; existing `memorias/` installs keep working — config, sync clone/bootstrap, backup/restore and import all read both names.
- 24 bugs from an exhaustive line-by-line review: `memo backup`/`memo health` crashing on their non-JSON paths, consolidation resurrecting archived duplicates, the `mem_review` SQL query, the tantivy rebuild for >5000-memory corpora, the synapse `conflicts` CLI args, and more.
- Dropped a personal locale default ("Spanish rioplatense") from the ask prompt; answers follow the question's language generically.

## [2.3.12] - 2026-06-29

### Fixed
- Dream cross-session consolidation (`memo dream consolidate-episodes`) now saves its synthesized memories — a keyword-only `save()` was being called positionally, raising `TypeError`. Regression-tested.

## [2.3.11] - 2026-06-29

### Added
- **Dream v2 — self-improving recall tuner** (off by default): nightly `MEMO_RECALL_MIN_SIM` line-search from ground-truth-by-use labels, gated by the regression set, auto-applied and auto-reverted on a later regression (`memo dream tune`).
- **Dream v2 — episodic→semantic consolidation** (off by default): abstracts recurring cross-session work into one durable synthesis memory (`memo dream consolidate-episodes`).
- **Dream v2 — anticipatory pass** (off by default): surfaces recurring knowledge gaps and hot queries into the briefing and pre-warms their embeddings; never fabricates (`memo dream anticipate`).

## [2.3.10] - 2026-06-29

### Changed
- `memo doctor` auto-repairs a stale MCP config path (a dead pipx/uv launch path is repointed to the stable `~/.local/bin` shim, with a `.bak` backup). `--check` and `--json` stay read-only.

## [2.3.9] - 2026-06-29

### Changed
- The injected recall block now carries a relevance gate and stronger prompt-injection framing — recalled memory is treated as data, never as instructions.

## [2.3.8] - 2026-06-28

### Added
- Per-project memory storage (default on): new memories are filed under `memory_dir/<project>/`, while search stays global. `memo migrate --bucket-by-project` re-buckets an existing install.
- 3-tier recall relevance: current project > global / cross-cutting > other projects.

## [2.3.7] - 2026-06-28

### Fixed
- `ask` now surfaces both sides of a contradiction instead of silently picking one, and respects `k`.
- Spanish accented queries keep their relevance boost (diacritics folded).
- The embed socket handles full batches on 4B/8B models without truncating.

## [2.3.6] - 2026-06-28

### Fixed
- The statusline no longer duplicates the `[MEMO <version>]` badge when wrapping an existing statusline.

## [2.3.5] - 2026-06-28

### Fixed
- Broad CLI hardening: corrected contradiction/temporal/stale defaults, short-id resolution in `links`, populated `version history/diff/rollback`, date validation, and crash guards across `feedback`, `import`, `eval`, `query`, and the TUI.

## [2.3.4] - 2026-06-28

### Fixed
- **CRITICAL:** archive no longer deletes memories (the `.md` is moved to an archive dir before the index row is dropped).
- **CRITICAL:** zip-slip path traversal in `memo restore` and `../` traversal in Obsidian image embeds are blocked.
- Concurrency fixes, atomic config/state writes, subprocess timeouts, and correct exit codes across the runtime.

## [2.3.3] - 2026-06-27

### Added
- The statusline self-heals its wiring on session start (gated, default on).

### Fixed
- `memo install-statusline` now **wraps** an existing statusline instead of skipping it, so the `[MEMO]` badge coexists with other tools' badges.

## [2.3.2] - 2026-06-27

### Removed
- Reverted the cross-machine version-file auto-update added in 2.2.0 — it was redundant with the existing tag-based auto-update and shipped broken. Tag-based auto-update is unaffected.

## [2.3.1] - 2026-06-27

### Added
- The `memo resume` picker now previews what changed in a session's cwd since you last worked there, plus that session's open loops.

## [2.3.0] - 2026-06-27

### Added
- `memo episodes search` (CLI) and `memo_episodes_search` (MCP) find past work sessions by meaning.

## [2.2.0] - 2026-06-27

### Added
- Semantic Resume: `memo resume` searches the full session history by meaning via a derived episode index.

## [2.1.1] - 2026-06-27

### Added
- `memo resume` opens an interactive cross-agent picker in a TTY (resumes `claude` / `codex` / …); piped or `--json` still emits the candidate list.

### Fixed
- `memo update` from an editable/dev install now refuses with clear guidance instead of silently updating the wrong runtime.

## [2.1.0] - 2026-06-27

### Added
- `memo eval harvest` mines ground-truth recall labels from real grounded outcomes.
- The outcome loop closes nightly so ranking learns from real usage; continuous entity extraction on save; the unified briefing composes from memo's own corpus.

### Changed
- The reranker sees more candidates (`rerank_input_k` 5→30; validated precision@5 0.786→0.834), with candidate bodies served from the index instead of re-reading `.md` files.

### Fixed
- Hybrid recall mode is usable (its similarity gate compares the true vec cosine).
- 34 correctness bugs across store, memory ops, sync, embedder, capture, temporal, and analytics.

## [2.0.0] - 2026-06-26

First stable release.

### Added
- **Time-machine** — rewind the corpus to any past date (`memo as-of`, `memo diff`).
- **Contradiction radar** — `memo contradict scan` / `triage` detects and resolves conflicting facts.
- **Synthesis pipeline** — `memo synthesize` and the nightly `memo dream` infer cross-cluster insights (opt-in).
- **Cross-Mac git sync** — `memo sync bootstrap <url>` shares a corpus over a private git remote (rebase-before-push, single-owner `flock`).
- **Obsidian vault as source-of-truth** — `MEMO_MEMORIES_IN_VAULT=1`; human edits win on the next reindex.
- **Knowledge graph** — entity extraction, neighbors/path/centrality/communities, backlinks/outlinks.
- **Health scoring & eval gates** — `memo health` and `memo eval recall --gate` for CI or pre-commit.
- **Multi-modal ingestion** — images and audio (macOS Vision OCR), searchable.
- **Session continuity** — open loops, session state, `memo resume`.
- **Warm daemons** — recall (<200 ms) and idle-capture for hookless MCP clients.
- 95 CLI commands and 109 MCP tools across three surface profiles (slim / agent / full).
- `install.sh` uv-first installer and `memo doctor --strict-runtime`.

### Changed
- `MEMO_AUTO_UPDATE` is on by default (tag-based self-update); the `agent` MCP profile (5 tools) is the default.

### Removed
- Purged experimental surfaces (encryption, sharing, federation) and the unused GLiNER entity backend.

---

Releases before `2.0.0`: see [docs/CHANGELOG-archive.md](docs/CHANGELOG-archive.md).
