# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`memo` (PyPI dist name: `mlx-memo` as of 0.5.0; previously `memo-mcp`) is a local MCP memory server for Apple Silicon. Two entry points share the same `Memory` API:

- `memo` — Click-based CLI in `src/memo/cli.py` (~25 commands).
- `memo-mcp` — FastMCP stdio server in `src/memo/server.py` (~13 tools, all prefixed `memory_`).

The Python module, CLI binary, and GitHub repo are all named `memo`. The PyPI distribution is `mlx-memo` because `memo` was already taken and `memo-mcp` collided with another project. Don't conflate the distribution name with the import/module/binary names when grepping or renaming.

## Common commands

```bash
# Dev install (the project requires Python ≥ 3.13; the Linux/x86 path is gated by pyproject markers)
uv pip install -e '.[dev]'

# Tests — fast suite (skips slow real-MLX smoke)
.venv/bin/python -m pytest -q -m "not slow"

# Single test
.venv/bin/python -m pytest tests/test_memory.py::test_save_then_search -q

# Real-MLX smoke (Apple Silicon only — auto-skips elsewhere via conftest)
.venv/bin/python -m pytest -q -m requires_mlx

# Lint
.venv/bin/ruff check src/ tests/

# CLI smoke against the real vault
memo doctor --strict-runtime
memo stats

# Real isolated system install from this checkout
pipx install --force .
memo doctor --strict-runtime

# Installer script syntax check
bash -n install.sh
```

CI (`.github/workflows/test.yml`) runs ruff + `pytest -m "not slow"` on Ubuntu — this works because `mlx`/`mlx-lm` deps are gated to `darwin`/`arm64` in `pyproject.toml`, and `tests/conftest.py` auto-skips `requires_mlx` tests when `mlx_lm` isn't importable.

Pytest markers (declared in `pyproject.toml`):
- `requires_mlx` — loads real MLX models. Auto-skipped without `mlx_lm`. Default run includes them on Apple Silicon.
- `slow` — anything >1s. Default run excludes via `-m "not slow"`.

## Architecture

Layered, with the `.md` file in the Obsidian vault as the storage of record. Three sqlite files intentionally split (different WAL, independent reset).

```
CLI / MCP server
       │
       ▼
   memory.py  (Memory: save / search / list / get / update / delete / reindex / consolidate / ask)
       │
   ┌───┼─────────────┬──────────────┬──────────────┐
   ▼   ▼             ▼              ▼              ▼
embedder.py  store.py        history.py      graph.py     llm.py
 (MLX        (sqlite-vec     (events DB)     (entities    (MLX two-tier:
  embed)      memvec.db                       graph.db)    Qwen2.5-7B chat
                                                            + 3B helper)
       │
       ▼
markdown files in cfg.data_dir (default ~/Documents/memo/)  ← source of truth
```

### Storage of record vs index

The `.md` files are authoritative. `memvec.db` (sqlite-vec) is a rebuildable index. The user can edit memorias directly in Obsidian or any editor; on next `memo reindex` (or watcher run), `body_hash` mismatches drive re-embedding. `rm ~/.local/share/memo/memvec.db && memo reindex` is a safe full rebuild — it never touches the `.md` files.

Three separate sqlite files in `~/.local/share/memo/`:
- `memvec.db` — `meta` table + `vec0` virtual table (sqlite-vec). Hot-path read.
- `history.db` — append-only audit of save/update/delete events.
- `graph.db` — entity index (proper-noun NER over memorias).

Splitting them avoids WAL contention between hot vec reads and batch writes (entity extraction, history). Don't merge them back without good reason.

### MLX in-process — Apple Silicon only

`embedder.py` uses `mlx_lm.load()` to load a Qwen3-Embedding model, bypasses `lm_head`, last-token-pools the hidden states, and L2-normalises. `llm.py` runs Qwen2.5-Instruct in two tiers (7B chat / 3B helper) via `mlx_lm.generate()`. Both lazy-load under `_load_lock`.

Key invariants — don't break these:
- **Asymmetric retrieval.** Queries get a `Instruct: ...\nQuery: ...` prefix prepended in `embedder.py`; documents go raw. Without the prefix, cosine collapses toward 0. The constant lives in `_QUERY_INSTRUCTION_PREFIX`.
- **`MLXEmbedder.embed()` is batched.** Signature is `Sequence[str] → list[list[float]]`. Passing a bare string iterates per-character (Python's string-as-iterable), produces variable-dim outputs, and cascades into a Metal GPU error. Always wrap: `embedder.embed([text])[0]`. This was the v0.3.1 bug — don't reintroduce it.
- **`MEMO_EMBEDDER_DIMS` must match the model.** Asserted at load (1024 for 0.6B, 2560 for 4B, 4096 for 8B). The vec0 schema bakes in `FLOAT[1024]`; swapping models requires `rm memvec.db && memo reindex` — see README "Upgrading the embedder".
- **`mlx`/`mlx-lm` imports are deferred.** Module-level imports would break Linux CI. Defer until `_ensure_loaded()` (already the convention in `embedder.py` / `llm.py`).

### Search modes

`Memory.search(query, mode=...)` supports three modes (default `hybrid`):
- `vec` — cosine over sqlite-vec embeddings.
- `bm25` — FTS5 over title + tags + body. Uses unicode61 + diacritic stripping for Spanish. The query tokenizer wraps each `\w+` token in its own phrase quotes so multi-word queries become AND-of-tokens, not phrase-match — see v0.3.2 changelog. If you touch BM25 tokenization, re-run `tests/test_memory.py` BM25 cases against multi-word Spanish queries (`"Astor terapia ocupacional"`).
- `hybrid` — Reciprocal Rank Fusion of vec + BM25 (default).

### Ambient memory hooks (`hooks/hooks.json`)

When the plugin is installed, two Claude Code hooks are wired:
- `SessionStart` (matcher `startup|clear`) → `memo prewarm` (async, 30s timeout). Pre-loads MLX so the first recall is fast.
- `UserPromptSubmit` → `memo recall-hook` (5s timeout). Embeds the prompt, runs vec-only search, prints the top-3 memorias above `MEMO_RECALL_MIN_SIM` (default 0.6) as `additionalContext` markdown on stdout.

The 5s timeout is tight. Cold MLX load is ~2s. If you add work to the hook, measure end-to-end latency against the 5s budget with `MEMO_RECALL_DEBUG=1`.

## Testing conventions

`tests/conftest.py` enforces two rules — preserve them:

- **Never touch the real vault.** The `tmp_cfg` fixture builds a `Config` with `data_dir`, `vault_path`, and `state_dir` under pytest's `tmp_path`, and pins `MEMO_CONFIG_FILE` so `Config.from_env()` doesn't read the dev's real `~/.config/memo/config.toml`. Any new test must depend on `tmp_cfg` (or build its own isolated `Config`) — never call `Config.from_env()` in a test without controlling these env vars.
- **CliRunner-based tests must set `MEMO_NONINTERACTIVE=1`** in the `env=` arg, otherwise the CLI's first-run gate may try to fire the picker mid-test. Also override `MEMO_DATA_DIR`/`MEMO_STATE_DIR` to keep state under `tmp_path`. If your test exercises the embedder via the stub (`monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", ...)`), also pin `MEMO_EMBEDDER_DIMS` to match the stub's output dim — the dev's shell may have `MEMO_EMBEDDER_DIMS=2560` (4B model) exported and CliRunner inherits it.
- **`requires_mlx` is auto-applied.** A test marked `@pytest.mark.requires_mlx` skips automatically when `mlx_lm` can't be imported. Use it for anything that does a real forward pass; everything else should monkeypatch `MLXEmbedder.embed` / `MLXChat.chat`.

`tests/test_smoke_mlx.py` is the canonical example of `requires_mlx` smoke tests.

## Versioning + release

Version lives in three places — bump together:
- `pyproject.toml` `[project] version`
- `.claude-plugin/plugin.json` `version`
- `server.json` `version` and package `version`
- `CHANGELOG.md` (Keep-a-Changelog format)

`src/memo/__init__.py:__version__` is read from installed package metadata via `importlib.metadata.version("mlx-memo")`; `pyproject.toml` is the source of truth for package builds.

Release/runtime invariant: production use should install memo as an isolated tool (`curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash`, `pipx install mlx-memo`, `uv tool install mlx-memo`, Homebrew, or `pipx install --force .` from this checkout). The curl installer tracks GitHub `master` by default; use `MEMO_INSTALL_FROM_PYPI=1` or `MEMO_VERSION=...` only when the published PyPI build is the target. Avoid installing it into another project's `.venv`; `memo doctor --strict-runtime` is expected to fail if `memo` and `memo-mcp` resolve from mixed runtimes.

## Conventions specific to this repo

- **The memoria storage location is user-configurable.** The default is `~/Documents/memo/` (a standard macOS path so the tool works for users who don't use Obsidian). On first interactive run, `memo` prompts an arrow-key picker (`memo init` re-runs it). The chosen path is persisted in `~/.config/memo/config.toml` `[storage] data_dir`. Tests must override via `tmp_cfg` (which sets `data_dir`, `vault_path`, `state_dir` and pins `MEMO_CONFIG_FILE` to a test-only path so the developer's real config doesn't leak in). Resolution order in `Config.from_env()`: explicit kwargs → `MEMO_*` env vars → config file → legacy `MEMO_VAULT_PATH` + `MEMO_MEMORY_SUBDIR` → default. `vault_path` is now optional and only used by `memo ingest`.
- **Frontmatter schema is fixed.** `id`, `title`, `type`, `tags`, `created`, `updated`. `type` is one of `decision | fact | bug | feedback | preference | note | manual`. Adding a new type means updating the Click `Choice` in `cli.py:save`/`update` AND the docstring in `server.py:memory_save`.
- **Filename slug is `<YYYY-MM-DD>-<slug>.md`.** Mirrors obsidian-rag's conversation writer. The slugifier lives in `memory.py`.
- **No Ollama anywhere.** `pyproject.toml` does not declare it; `doctor` does not probe `:11434`. The README leans on this as a selling point — don't reintroduce it as a dependency.
