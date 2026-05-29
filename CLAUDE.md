# memo — project instructions

> **Note:** This file was inlined into the repo from the previous machine-local
> `@import` (an iCloud/Obsidian path that doesn't resolve in clones or on every
> Mac). The content below is reconstructed from the project's own subagent
> definitions (which quote these invariants verbatim) and verified against the
> code. If you keep a richer canonical copy in your Obsidian vault, reconcile it
> here so the repo stays self-contained.

memo is a local-first semantic memory: MLX embeddings + sqlite-vec hybrid
(vec + BM25) search, reranking, a knowledge graph, temporal reasoning, an MCP
server, and a CLI. Single-user, runs offline.

## MLX invariants (do not violate)

The embedder/LLM path depends on four invariants. A diff touching
`src/memo/embedder.py`, `src/memo/llm.py`, `src/memo/memory.py`, or anything
that calls `.embed()` / `.chat()` must preserve all four:

1. **Asymmetric retrieval prefix on QUERIES only.** Query text gets
   `_QUERY_INSTRUCTION_PREFIX` (see `embedder.embed_query`); stored documents do
   **not**. Prefixing both (or neither) collapses cosine similarity — the model
   places prefixed and raw inputs in different regions of the space.
2. **`MLXEmbedder.embed()` takes `Sequence[str]`, never a bare `str`.** A bare
   string is iterated as characters and silently produces garbage. Always wrap:
   `embed([text])`.
3. **`MEMO_EMBEDDER_DIMS` must match the model.** 1024 / 2560 / 4096 for the
   0.6B / 4B / 8B Qwen3-Embedding models. Default model is
   `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` (1024-dim). A mismatch corrupts
   the vec0 table or fails the dims-validation guard in `store.py`.
4. **`mlx` / `mlx-lm` imports stay deferred** (inside functions, not at module
   level) — module-level import drags the MLX runtime into every CLI invocation
   and blows the recall-hook cold-start budget.

When stubbing `MLXEmbedder.embed` in tests, also pin `MEMO_EMBEDDER_DIMS` to the
stub's output dim.

## Recall hook (UserPromptSubmit) — 5s budget

`memo recall-hook` is wired into Claude Code's UserPromptSubmit hook
(`hooks/hooks.json`). End-to-end it must stay under ~5s. Cold MLX load is ~2s,
leaving ~3s for embed + vec search + format. The rerank candidate pool
auto-shrinks (and `MEMO_RECALL_TOKEN_BUDGET` caps output) so warm latency stays
under the timeout. Anything added to the hook path — `hooks/hooks.json`, the
`recall-hook` command in `src/memo/cli.py`, `embedder.py`, `store.py` search —
must respect that budget. A warm `memo-mcp` HTTP/recall daemon serves embeds
over a socket (`MEMO_EMBEDDER_VIA_DAEMON`) to skip cold load.

## Runtime

`memo` and `memo-mcp` must resolve from the **same isolated runtime**
(pipx / `uv tool` / Homebrew) — **not** a project `.venv`. A mixed runtime is the
usual cause of "works in CLI, broken in MCP". `memo doctor --strict-runtime`
checks this.

## BM25 / Spanish search

FTS5's tokenizer wraps each `\w+` token in its own phrase quotes, so a
multi-token query becomes an AND-of-tokens (not a phrase match). `store.py`
tokenizes, AND-joins, and falls back to OR only on zero recall. Diacritics are
folded (`unicode61 remove_diacritics 2`) so "decision" matches "decisión".

## Storage

`VecStore` (`src/memo/store.py`) is sqlite-vec backed, one DB file, **one
connection per thread** (thread-local — required for the FastMCP HTTP transport's
worker threadpool). Writes go through `_tx()` (`BEGIN IMMEDIATE`); vectors are
packed float32 blobs; WAL mode + `busy_timeout` give concurrent readers + a
writer.

## Test isolation (see `tests/conftest.py`)

- Use the `tmp_cfg` fixture or build an isolated `Config` — never call
  `Config.from_env()` without controlling the environment.
- `CliRunner` invocations set `MEMO_NONINTERACTIVE=1`, `MEMO_DATA_DIR`, and
  `MEMO_STATE_DIR` in `env=` (conftest defaults `MEMO_NONINTERACTIVE=1`).
- Real MLX forward passes are gated by `@pytest.mark.requires_mlx` (auto-skipped
  when `mlx_lm` isn't importable).
- Never read or write the developer's real vault.

## Config & errors

- `MEMO_*` flags live in a central registry (`src/memo/config.py` +
  `src/memo/flags.py`); prefer the typed `flag_bool/int/float/str` accessors over
  raw `os.environ`. `memo config validate` catches typos.
- Domain errors live in `src/memo/errors.py` (`MemoError` base). Raise/catch
  those rather than bare `Exception` in non-defensive code.

## Releasing

Bump the version in sync across **four** source-of-truth files:
`pyproject.toml` `[project].version`, `.claude-plugin/plugin.json`,
`server.json` (version + package version), and `CHANGELOG.md`
(Keep-a-Changelog). Commit / tag / push stays manual.

## CI gates

`pytest`, `mypy`, and coverage run per commit. Keep the suite green.
