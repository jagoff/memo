# Memo Markdown Configuration Design

## Context

Memo's configuration is currently split across several surfaces:

- `Config.from_env()` in `src/memo/config.py` handles storage, model, reranker, and limit settings.
- `src/memo/flags.py` and domain files such as `flags_recall.py` and `flags_search.py` register behavioral `MEMO_*` flags.
- `~/.config/memo/config.toml` stores a small `[storage]` section written by `memo init`.
- Hooks and docs still expose many settings as environment variables.
- The tuned overlay in `src/memo/tuned_overlay.py` can adjust selected flags below explicit environment overrides.

This makes the system hard for users to configure because settings are discoverable only by reading docs, code, or environment variable tables. The new goal is a Linux-style editable configuration directory where users can inspect and change the full memo configuration in plain Markdown.

Prior project decisions also matter:

- Markdown should remain a human-readable source of truth for user-owned state.
- File and key names should stay in English for cross-machine compatibility.
- The repo is mature and broad, so this should be introduced through the existing `Config.from_env()` and `flags.flag()` seams instead of rewriting all call sites at once.

## Goals

- Make Markdown configuration the persistent source of truth for memo settings.
- Cover 100% of user-configurable settings, including advanced and experimental knobs.
- Keep config readable for normal users without hiding advanced settings.
- Preserve environment variables as temporary runtime overrides for CI, tests, hooks, and one-off commands.
- Treat `~/.config/memo/config.toml` as legacy fallback during migration, with the explicit architectural direction that it will disappear later.
- Avoid adding latency to recall/search hot paths.
- Add CLI commands so users can create, inspect, validate, edit, and migrate configuration without manually finding files.

## Non-Goals

- Do not remove `config.toml` in the first implementation.
- Do not rename all internal `MEMO_*` flags at once.
- Do not rewrite all call sites that already use `Config.from_env()` or `flag_bool()`/`flag_int()`/`flag_float()`/`flag_str()`.
- Do not change default behavior merely because the configuration surface changes.
- Do not run retrieval evals unless implementation changes ranking defaults or retrieval behavior beyond configuration resolution.

## Chosen Approach

Use Markdown files as the persistent user-facing source of truth, parsed through fenced TOML blocks. Insert a new loader below the existing config seams:

- `Config.from_env()` reads Markdown values for storage/model/config fields.
- `flags.flag()` reads Markdown values for behavioral flags.

This keeps existing runtime code stable while changing where persistent configuration comes from.

The chosen approach is an overlay-compatible migration, not a full configuration rewrite. It gives users a real editable config directory now and leaves room to remove legacy `config.toml` later.

## File Layout

Default location:

```text
~/.config/memo/
  memo-config.md
  config/
    storage-config.md
    models-config.md
    search-config.md
    recall-config.md
    capture-config.md
    graph-config.md
    hooks-config.md
    advanced-config.md
```

`memo-config.md` is a human index. It explains precedence, points to each domain file, and lists useful commands.

Domain files:

- `storage-config.md`: data/state paths, vault settings, `memories_in_vault`, `single_db`.
- `models-config.md`: model profile, LLM/helper/embedder models, dims, backend, reranker settings.
- `search-config.md`: search limits, decay, BM25, search-level reranking.
- `recall-config.md`: recall hook, daemon, formatting, budgets, boosts, dedup collapse.
- `capture-config.md`: auto-capture, idle capture, redaction, secrets, capture quality gates.
- `graph-config.md`: graph signal, semantic relations, hubs, fact retrieval/surface controls.
- `hooks-config.md`: hook lifecycle switches, noninteractive behavior, self-heal, sync and idle timings.
- `advanced-config.md`: debug, experimental, rare, compatibility, and low-level operational knobs.

## Format

Files are Markdown for humans. Memo parses only fenced TOML blocks:

````markdown
# Recall config

Common settings.

```toml
[recall]
disable = "off"
top_k = 3
min_sim = 0.5
format = "compact"
dedup_collapse = "on"
```

Advanced settings.

```toml
[recall.advanced]
debug = "off"
force_mode = "off"
intra_dedup_threshold = 0.8
```
````

Rules:

- Boolean values are rendered by default as `"on"` and `"off"`.
- The parser accepts `on/off`, `true/false`, `1/0`, and `yes/no`, with or without quotes when TOML allows it.
- User-facing keys are domain-oriented, for example `recall.top_k`.
- Each user-facing key maps to an existing `MEMO_*` name, for example `MEMO_RECALL_TOP_K`.
- Unknown TOML keys are validation warnings or errors, not silent no-ops.
- Free Markdown text outside TOML blocks is ignored by the parser.
- Invalid TOML should be surfaced clearly. Runtime should avoid breaking hooks: warn and continue with higher/lower precedence sources where possible.

## Precedence

Effective resolution order:

1. Explicit internal kwargs.
2. Environment variables, such as `MEMO_RECALL_TOP_K`.
3. Markdown config files in `~/.config/memo/config/`.
4. Tuned overlay, only for flags currently supported by `tuned_overlay.py`.
5. Legacy `~/.config/memo/config.toml`.
6. Registry/model/default values.

Markdown intentionally wins over the tuned overlay. If a user writes a value in config, that persistent explicit choice should beat auto-tuning.

Environment variables intentionally win over Markdown. They remain the right tool for CI, tests, hooks, and temporary one-off overrides.

## Loader Design

Add a new module, `src/memo/config_md.py`, with responsibilities:

- Resolve the config home and domain file paths.
- Parse fenced TOML blocks from Markdown.
- Merge all domain files into one structured document.
- Map user-facing paths to existing config fields and `MEMO_*` flags.
- Coerce values through the same type rules and bounds used by `Config` and `FlagSpec`.
- Report unknown keys, parse failures, invalid values, and shadowed legacy values.
- Cache parsed files by path and mtime to avoid hot-path file reads.
- Provide write helpers for `init`, `migrate`, `set`, and `unset`.

Add a new environment override for tests and advanced installs:

```text
MEMO_CONFIG_DIR=/path/to/memo-config-home
```

This points to the directory containing `memo-config.md` and `config/`. `MEMO_CONFIG_FILE` remains only for legacy TOML during the transition.

## Integration Points

### `Config.from_env()`

`Config.from_env()` should load Markdown config before legacy TOML. It should use Markdown values for known storage/model fields, then apply env vars, platform gates, repo-mode behavior, legacy fallback, kwargs, and index embedder adoption without changing existing semantics.

Storage/model fields in Markdown should map to the same Pydantic `Config` fields currently populated by env vars and `config.toml`.

### `flags.flag()`

`flags.flag()` should read Markdown values after checking env vars and before consulting the tuned overlay/default. It should reuse `FlagSpec` for kind coercion and numeric bounds.

`active_flags()` should remain environment-only. Add a clearly named companion, `active_config_values()`, for Markdown-backed explicit config values so callers that need environment overrides do not accidentally treat Markdown as env.

`validate()` should validate both env vars and Markdown values, and should report unknown `MEMO_*` env vars as it does today.

### Tuned Overlay

The tuned overlay remains machine-local and below Markdown. Existing overlay mechanics and rollback behavior should not be removed in this change.

## CLI Design

Expand `memo config` into the main configuration console:

- `memo config init`
  - Creates `memo-config.md` and all domain files with defaults.
  - Does not overwrite existing files unless `--force`.

- `memo config migrate`
  - Reads legacy `config.toml`.
  - Writes Markdown config files.
  - Creates `config.toml.pre-md-config.bak`.
  - Leaves `config.toml` in place as legacy fallback.

- `memo config validate`
  - Validates Markdown, env vars, legacy TOML, types, ranges, unknown keys, conflicts, and important action warnings.
  - Reports when legacy values are shadowed by Markdown.

- `memo config show`
  - Shows configured values by domain.

- `memo config show --effective`
  - Shows final values after precedence.
  - Includes value, source, domain path, and equivalent `MEMO_*` name.

- `memo config set recall.top_k 5`
  - Edits the correct domain file and validates the result.

- `memo config unset recall.top_k`
  - Removes the Markdown override so the value falls back to env, overlay, legacy, or default.

- `memo config path`
  - Prints active config home, index file, domain directory, and legacy TOML path.

Existing `memo config flags` may remain as a technical view or alias, but the primary UX should become `show`, `show --effective`, and `validate`.

## Migration Behavior

`memo init` should write Markdown config for new installs. It should not write legacy `config.toml` unless a specific compatibility mode is added.

During migration:

- Existing `config.toml` is read as fallback.
- Markdown values win when both exist.
- `memo config validate` reports legacy values shadowed by Markdown.
- Documentation marks `config.toml` as deprecated.
- A later design/implementation can remove `config.toml` support after sufficient compatibility time.

## Error Handling

- Invalid Markdown text outside TOML blocks is irrelevant.
- Invalid TOML blocks are reported by file and block location when possible.
- Runtime should warn and continue rather than breaking recall hooks.
- CLI validation should fail with non-zero exit when parse/type/range errors exist.
- Unknown TOML keys should fail validation unless explicitly marked as future-compatible.
- Unknown `*-config.md` files under `config/` should produce validation warnings. Files that do not match `*-config.md` are ignored so users can keep notes or backups nearby.

## Performance

Recall and search hot paths cannot parse all config files for every flag read.

The Markdown config loader should maintain an mtime cache similar in spirit to `tuned_overlay.py`. Cache invalidation can rely on file mtimes. CLI write operations naturally update mtimes; no long-lived daemon invalidation mechanism is required for the first version.

## Testing

Add focused tests for:

- Parsing fenced TOML from Markdown.
- Multiple TOML blocks per file.
- `on/off`, `true/false`, `1/0`, `yes/no` boolean coercion.
- Unknown key detection.
- Invalid TOML behavior.
- `Config.from_env()` precedence:
  - kwargs > env > Markdown > legacy TOML > defaults.
- `flags.flag()` precedence:
  - env > Markdown > tuned overlay > default.
- Legacy TOML shadow warnings.
- `memo config init`, `migrate`, `show --effective`, `set`, `unset`, and `path`.
- Test isolation through `MEMO_CONFIG_DIR`.

Recommended focused checks:

```bash
uv run --no-sync pytest tests/test_config.py tests/test_flags.py tests/test_setup_config_io.py tests/test_cli_init.py -v
uv run --no-sync ruff check src/ tests/
uv run --no-sync mypy src/memo
```

Run broader CI-parity checks once the integration touches both `Config` and `flags`.

## Rollout Plan

1. Build `config_md.py` parser/loader/writer with tests.
2. Integrate Markdown storage/model values into `Config.from_env()`.
3. Integrate Markdown behavioral values into `flags.flag()`.
4. Add CLI commands under `memo config`.
5. Update `memo init` to write Markdown config.
6. Add migration from `config.toml`.
7. Update docs and reference tables.
8. Keep `config.toml` fallback.
9. In a future release, remove or hard-deprecate legacy TOML after migration has been proven.

## Fixed Implementation Choices

- `active_flags()` remains strictly env-only. Markdown config gets a separate `active_config_values()` API.
- Unknown registered-looking files such as `custom-config.md` warn during validation unless their domain is registered.
- `memo config set` rewrites the relevant TOML block canonically and preserves surrounding Markdown. Exact TOML whitespace/comment preservation inside the block is not required for the first version.
