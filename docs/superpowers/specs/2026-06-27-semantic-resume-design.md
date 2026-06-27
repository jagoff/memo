# Semantic Resume — memo Episodic Memory (Phase 1)

**Date:** 2026-06-27
**Status:** Design — pending review
**Builds on:** the cross-agent `memo resume` picker shipped in v2.1.1 (Forma B).

## One-liner

Turn `memo resume` from a recency launcher into a **meaning-based picker over your
entire session history**: type "the vec0 timeout bug" and the session from three
weeks ago surfaces — even though it's far past the recency cap. This is the first
slice of a larger idea: memo gains an **episodic memory** layer (what you *did*),
alongside its existing semantic memory (what you *know*).

## Goals

1. Semantic search in the resume picker over **all** sessions (not just the
   mtime-capped recent set), across all agents (claude/codex/devin/gemini/opencode).
2. Seamless UX: substring filters instantly as you type; after a ~300 ms pause the
   list re-ranks by meaning. Degrades to substring when no warm embedder is available.
3. Leverage memo's unique asset (MLX embeddings + sqlite-vec) — the capability
   `synapse resume` structurally cannot have.
4. Respect every memo invariant: MLX asymmetric-prefix, the 5 s recall-hook budget,
   "markdown is source of truth", sovereignty (no memflow coupling), and "sessions
   are not durable facts".

## Non-goals (Phase 1)

- Exposing episodes to `memo_ask` / `memo search` / the recall hook / MCP. (Phase 2.)
- Injecting episode context into the resumed agent ("resume with grounding"). (Phase 2.)
- Repo-delta / open-loops preview panel. (Phase 2.)
- Chunked per-turn embedding ("find the exact moment I discussed X"). (Future.)

## Key decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Where the index lives | **Separate derived index** (`episode_vec` + `episode_meta` tables), NOT the `.md`-sourced memory store | Episodes' source of truth is the **transcript**, not a `.md`. Keeps the vault clean (no 1683 generated `.md`) and the memory-store rebuild (`reindex --rebuild` from `.md`) untouched. Episodes get their own rebuild path. |
| What's embedded | **Prompt-arc + running_summary**, one embed/session, as a **document** (no query prefix) | Uniform across agents (no dependency on running_summary, which only memo sessions have); captures intent. One embed keeps backfill cheap (~1683 one-time). |
| Picker trigger | **Seamless debounced** semantic re-rank; substring is the instant + fallback layer | Most "magical"; one embed per typing pause via the warm daemon (~50 ms), not per keystroke. |
| Embedder for the picker | **Warm recall daemon only** (`MEMO_EMBEDDER_VIA_DAEMON` socket). If unavailable → substring only | The picker must never cold-load MLX (~2 s) on a keystroke pause. The warm daemon is always-on per the trinity. |
| Framing | Logical = **episodic memory tier**; physical = separate derived index | Lets Phase 2 open episodes to ask/MCP as an additive surface without a migration. |

## Architecture

### 1. Episode index (storage)

A sidecar store following memo's existing multi-file / `MEMO_SINGLE_DB` convention
(folds into `memvec.db` when single-db is on; otherwise `episodes.db` under `data_dir`):

- **`episode_vec`** — vec0 table, dims = `MEMO_EMBEDDER_DIMS` (must match the model;
  guard on mismatch like the existing vec store).
- **`episode_meta`** — `rowid, agent, session_id, content_hash, cwd, updated_at,
  summary, resume_command (json), turn_count, indexed_at`.
- Keyed logically by `(agent, session_id)`. `content_hash = sha256(prompt_arc)` →
  re-embed only when the session grew. Reuses the content-addressed
  `repo_embedding_cache` so a warm rebuild issues ~zero embedder calls.
- Derived + rebuildable: `memo episodes index --rebuild` re-scans transcripts. The
  index is never the source of truth.

New file: `src/memo/store/episode_store.py` (mirrors the `VecStore` patterns —
thread-local conn, `_tx()`, packed float32 blobs, dims guard).

### 2. Indexer

`src/memo/resume/_index.py`:

- **`prompt_arc(candidate) -> str`** — given a `ResumeCandidate` (has
  `metadata.path` = transcript), read it, collect the last ~40 user prompts
  (generalize `_extract_user_text` / `_jsonl_latest_user_text` to *gather*, not
  just "latest"), prepend `running_summary`/`summary` when present, join, clip
  ~2000 chars.
- **`upsert(candidate)`** — compute `content_hash`; skip if unchanged; else
  `embed([prompt_arc])` (document, no prefix) → write `episode_vec` + `episode_meta`.
- **Incremental (Stop):** `capture-stop` already runs on the Stop hook and computes
  `running_summary`. After that, best-effort `episode_index.upsert(<this session>)`.
  One embed, content-addressed skip → effectively free. Never fails the hook.
- **Backfill:** `memo episodes index [--rebuild] [--limit N] [--agent A]` — reuse the
  resume providers' discovery to enumerate sessions, build prompt-arcs, batch-embed
  (cold-load once is fine for a batch command), content-addressed, bounded by
  `MEMO_RESUME_INDEX_BATCH`, resumable (skip indexed-by-hash). Folded into
  `memo-nightly` so it stays warm without manual runs.

### 3. Picker semantic search

`src/memo/resume/_index.py` `semantic_search(query, k) -> list[ResumeCandidate]`:

- Embed the query **with** the asymmetric query prefix (`embed_query`) via the warm
  daemon socket; KNN over `episode_vec` (top `MEMO_RESUME_SEMANTIC_K`, default 50);
  join `episode_meta`; **hydrate a `ResumeCandidate` straight from the meta row**
  (agent, session_id, cwd, summary, resume_command) — no transcript re-parse, so
  old sessions beyond the recency cap surface fully.

Wiring in `_tui.py` + `cli_capture.py::_resume_federated`:

- Query empty → recency order (unchanged).
- On query change → substring filter renders instantly (today's behaviour).
- After ~300 ms with no keypress → if a warm embedder is reachable, run
  `semantic_search(query)`; **merge** its hits (which may include sessions not in the
  loaded recency set) with the substring-filtered loaded set; re-rank by similarity.
- No warm embedder / embed error → stay on substring. Semantic is strictly additive;
  the picker never breaks or hangs.
- Inline embed on the debounce tick (warm ~50 ms) — no background threads; one render
  frame's latency is imperceptible.

### Data flow

```
Stop hook ─► capture-stop ─► running_summary ─► episode_index.upsert(session)
                                                        │  (1 embed, hash-skip)
                                                        ▼
nightly / `memo episodes index` ─► discover all sessions ─► prompt_arc ─► batch embed
                                                        ▼
                                                  episode_vec + episode_meta
                                                        ▲
picker: type query ─(300ms pause)─► embed_query (warm socket) ─► KNN ─► hydrate ─► re-rank
                   └─ instant substring (fallback if no warm embedder) ─┘
```

## Components & files

| File | Change |
|---|---|
| `src/memo/store/episode_store.py` | **new** — `EpisodeStore` (vec0 + meta, dims guard, `_tx`) |
| `src/memo/resume/_index.py` | **new** — `prompt_arc`, `upsert`, `semantic_search`, backfill |
| `src/memo/resume/_tui.py` | debounce tick → semantic re-rank; degrade path |
| `src/memo/cli_capture.py` | `memo episodes index` command; picker wiring in `_resume_federated` |
| `src/memo/capture.py` / capture-stop | best-effort `upsert` after `running_summary` |
| `src/memo/flags_behavior.py` | new flags |
| `src/memo/config.py` | `episode_db` path property (single-db aware) |
| `tests/test_resume_episodes.py` | **new** |

## Flags

- `MEMO_EPISODIC_ENABLED` (bool, default `true`) — master switch; off ⇒ picker is
  recency+substring only, no indexing.
- `MEMO_RESUME_SEMANTIC_K` (int, 50) — KNN top-k.
- `MEMO_RESUME_INDEX_BATCH` (int, 500) — backfill cap per run.
- Debounce (300 ms) hardcoded unless a need to tune emerges (YAGNI).

## Invariants & safety

- **MLX asymmetric prefix:** query embed uses `embed_query` (prefixed); episode
  document embed uses `embed([arc])` (no prefix). Never both.
- **`embed()` takes `Sequence[str]`**, never a bare str.
- **Dims = model:** `episode_vec` dims = `MEMO_EMBEDDER_DIMS`; mismatch guard.
- **MLX imports deferred** (inside functions).
- **5 s recall-hook budget:** untouched — the picker is not on the hook path; the
  Stop-hook upsert is one hash-skippable embed, best-effort.
- **Markdown source of truth:** episodes are a *derived* index over transcripts
  (their source), rebuildable independently; no `.md` generated, memory-store rebuild
  unaffected.
- **Sovereignty:** zero memflow/synapse coupling (stays Forma B). Episodes are an
  index, not a new store of durable truth.
- **"Sessions are not durable facts":** episodes are physically separate from the
  `.md` memory store and (Phase 1) invisible to recall/ask/stats.

## Error handling / degradation

- No warm embedder → picker is substring-only (today's behaviour). Logged once, never
  raised.
- Index missing/empty → semantic returns nothing → substring-only. Backfill populates
  lazily via nightly + Stop.
- Dims mismatch (model changed) → guard rejects writes; `--rebuild` re-embeds at the
  new dims.
- Corrupt/locked `episode_db` → caught, picker degrades; `--rebuild` recovers.

## Testing

- `prompt_arc` extraction: memo snapshot (running_summary present) + native transcript
  (prompts only). Pin `MEMO_EMBEDDER_DIMS` when stubbing the embedder.
- `upsert`: writes vec+meta; re-upsert with same content → hash-skip (no embed);
  grown session → re-embed.
- `semantic_search`: stub embedder (fixed vectors), seed episodes, assert ranking +
  that a session **outside** the recency set is hydrated and returned.
- Degrade: no warm embedder → picker yields substring results, no error.
- Backfill: incremental (skip indexed), `--rebuild` re-embeds, bounded by batch.
- Full suite + mypy + ruff green; a perf check that picker open stays sub-second.

## Phase 2 (roadmap — separate spec)

1. **Episodes in `memo_ask` / `memo search` / MCP** — query the episode index via a
   `tier=episode` / `--episodes` path (additive, still excluded from auto-recall), so
   "what did I decide about the vec0 timeout" answers from past sessions, and agents
   reach it over MCP.
2. **Resume with grounding** — opt-in: on Enter, write a primer the resumed agent
   reads (claude-first, best-effort) so it restarts already knowing the episode.
3. **Enriched preview** — highlight a session → episode summary + open-loops
   (`prompt_trail`) + `git diff --stat` since that session.
4. **Chunked episodes** — per-turn embeds for "find the exact moment I discussed X".

## Risks

- **Index staleness vs reality:** a session's `resume_command` could point at a
  transcript later deleted. Mitigation: `execute_resume_candidate` already handles
  "executable/target not found"; the meta row is best-effort and refreshed on rebuild.
- **Backfill cost (1683 sessions):** one-time, batched, content-addressed, bounded —
  runs in the nightly window, not interactively. `log()` what was skipped/capped.
- **Scope creep into Phase 2:** the episodic-memory framing is large; Phase 1 is
  deliberately walled (index + picker only) so it ships independently.
```
