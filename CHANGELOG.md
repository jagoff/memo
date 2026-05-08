# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] - 2026-05-08

Patch release — install-from-git fixes surfaced while validating the
distributed-install flow on a clean machine.

### Fixed

- **`memo --version` crashed** with `RuntimeError: 'memo' is not
  installed. Try passing 'package_name' instead.` because click's
  `version_option` defaulted to `package_name="memo"` while the actual
  PyPI/wheel dist is `memo-mcp`. Pinned the lookup explicitly.
- **`DEFAULT_MEMORY_SUBDIR` pointed at the deprecated archive path**
  (`04-Archive/99-obsidian-system/99-AI/memory`). Updated to the
  current `99-obsidian/99-AI/memory` location, matching the user-facing
  vault reorganization done on 2026-05-08. Existing installs that
  override via `MEMO_MEMORY_SUBDIR` are unaffected.

## [0.3.2] - 2026-05-07

Patch release — BM25 recall fix.

### Fixed

- **BM25 search wrapped query in phrase quotes** (`"foo bar"`) which
  required the words to appear consecutively. This killed recall on
  natural multi-word Spanish queries:
  - `"Astor terapia ocupacional"` did NOT match the document titled
    `"Informe Terapia Ocupacional — Astor Ferrari"` because the words
    don't appear in that exact consecutive order.

  Fix: tokenize via `\w+` (Unicode-aware), wrap each token in its own
  phrase quotes, join with whitespace (FTS5's implicit AND). Now the
  query is `"Astor" "terapia" "ocupacional"` — matches any doc
  containing all 3 words anywhere, any order.

### Verified

Same query post-fix:
- BM25: `Astor — Informe Terapia Ocupacional feb 2026` returns first hit ✓
- Hybrid (vec+BM25 RRF): same doc in top-3 ✓

Other corpus queries that benefit:
- `MLX migration` → `obsidian-rag: migración Ollama → MLX` (was missing pre-fix)
- `obsidian-rag bug fix` → `Bug pattern — sqlite3 'database is locked'` (was missing pre-fix)

## [0.3.1] - 2026-05-07

Patch release — bug fixes for the v0.3.0 ingest pipeline.

### Added

- `memo ingest <vault-path>` CLI command — bulk-ingest .md files from any
  Obsidian vault into the memo index. Synthesizes ids from path hash so
  user .md files are not modified. Idempotent. See README "Ambient memory"
  section for usage.
- `MEMO_INGEST_MIN_CHARS` env var (default 200) — skip notes shorter than
  this threshold. Tag-only stubs (`#tagA #tagB` + 1-line question)
  produce noisy embeddings near the corpus centroid that match generic
  queries with high false-positive rate. Filtering them improves recall
  precision on queries with proper nouns.

### Fixed

- **Embedder API misuse** in `recall-hook`, `prewarm`, and `ingest`:
  `MLXEmbedder.embed()` is batched (signature `Sequence[str] →
  list[list[float]]`); passing a bare string iterated per-char,
  producing variable-dim outputs (135, 512, 2465...) instead of 1024.
  Cascade Metal GPU error after several mismatches. Fix: wrap input
  in `[composed]` and take `[0]`.
- **Plugin install path** — bumping from 0.3.0 to 0.3.1 so users on the
  Claude Code plugin marketplace pick up these fixes via `/plugin update`.

## [0.3.0] - 2026-05-07

**Game-changer release**: memo turns into an *ambient* context layer.
Memorias auto-inject as `additionalContext` on every Claude Code prompt,
without the user invoking `/memo` at all.

### Added

- `memo recall-hook` CLI command — Claude Code `UserPromptSubmit` hook that
  embeds the user prompt via the MLX embedder, runs vec-only search against
  the memo index, and emits relevant memorias as `additionalContext` markdown
  on stdout. Sub-1.7s warm latency on a 223-doc corpus, well within the
  default 5s hook timeout.
- `memo prewarm` CLI command — `SessionStart` hook that pre-loads the MLX
  embedder so the first `recall-hook` invocation of the session is fast
  (warm load: ~500ms vs ~2s cold).
- `hooks/hooks.json` bundled in the plugin — auto-wires the two hooks above
  when users install via `/plugin install memo@memo`.
- 6 env vars to tune ambient memory behaviour:
  - `MEMO_RECALL_DISABLE` — kill switch (default: enabled).
  - `MEMO_RECALL_TOP_K` — default 3.
  - `MEMO_RECALL_MIN_SIM` — cosine similarity floor, default 0.6.
  - `MEMO_RECALL_MIN_PROMPT_CHARS` — default 12.
  - `MEMO_RECALL_BODY_CHARS` — snippet length, default 240.
  - `MEMO_RECALL_SKIP_SLASH` — skip recall on `/` prompts, default 1.
  - `MEMO_RECALL_DEBUG` — print failure reasons to stderr, default 0.

### Why this is the game changer

Before: user types `/memo save 'X'` to save and `/memo search 'Y'` to recall.
Friction = adoption blocker; memory only helps when remembered to invoke.

After: memo silently consults the user's past on every prompt and injects
the most relevant 3 memorias if any score above 0.6 cosine similarity.
Zero `/memo` invocations needed for the recall side. The agent "knows
your past" automatically.

### Empirical tuning

Threshold 0.6 was picked after testing the 223-doc corpus:
- "qué decidí sobre MLX vs Ollama" → 3 hits at 0.71-0.74 (all relevant).
- "how to bake apple pie" (corpus has zero food memorias) → 0 hits at 0.6
  (3 noise hits at 0.5-0.6 cut by the floor).
- "qué hice con whatsapp" → 3 hits at 0.6-0.75 (whatsapp work).

### Privacy / local-first

Hot path is 100% MLX in-process. Embedder = `Qwen3-Embedding-0.6B-4bit-DWQ`,
search = `sqlite-vec`. Zero network calls, zero cloud APIs, zero telemetry.
The hook input (your prompt) never leaves the machine.

### Save side (Phase B)

Passive memory extraction (auto-`memo save` from chat transcripts) is NOT
in 0.3.0 — recall side first to validate the architecture, save side in
0.4.0 once we have signal on whether the recall hook is helpful in real use.

## [0.2.0] - 2026-05-07

First public release. Distribution name on PyPI is `memo-mcp`
(`memo` was already taken).

### Added

- Public PyPI distribution as [`memo-mcp`](https://pypi.org/project/memo-mcp/).
- Claude Code plugin format (`.claude-plugin/plugin.json`) — single-step install
  via `/plugin install memo@jagoff/memo`.
- `.mcp.json` bundled in repo root so MCP-aware clients can auto-register.
- `skills/memo/SKILL.md` bundled in repo — slash-command UX layer for Claude
  Code CLI users (optional).
- LICENSE file (MIT).
- README polish: install, quickstart, architecture diagram, comparison vs
  `mem-vault` / `mem0` / `engram`.
- Migration script `scripts/migrate-from-mem-vault.py` (already shipped in
  0.1.0 development; documented in README for this release).

### Changed

- Bumped Development Status from `3 - Alpha` to `4 - Beta` in pyproject classifiers.
- Author name expanded from "Fer" to "Fernando Ferrari" for PyPI metadata clarity.

## [0.1.0] - 2026-04-28

Initial development release (private — not on PyPI).

### Added

- MLX-native memory MCP for Apple Silicon. Stack: `mlx-lm` + `mlx`
  (Qwen2.5-7B/3B-Instruct-4bit + Qwen3-Embedding-0.6B-4bit-DWQ),
  `sqlite-vec` for vectors, markdown files in Obsidian vault for storage.
- Tools exposed: `memory_save`, `memory_search`, `memory_list`, `memory_get`,
  `memory_update`, `memory_delete`, `memory_stats`, `memory_reindex`,
  `memory_ask` (RAG over memorias with inline citations), `memory_consolidate`
  (cluster + LLM merge proposals), graph queries (entity extraction).
- CLI (`memo`) with subcommands matching MCP tools.
- MCP server entry point (`memo-mcp`) using FastMCP framework.
- History tracking via SQLite for memory edits + accesses.
