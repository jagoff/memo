# Task 1 Report: Markdown Config Parser, Mapping, and Validation Core

Status: DONE

## Files Changed

- `src/memo/config_md.py` (created)
- `tests/test_config_md.py` (created)

## Commits Created

- `438d68c feat(config): parse markdown config files`

## Tests Run

1. `uv run --no-sync pytest tests/test_config_md.py -v`
   - Initial TDD red run: expected collection failure, `ImportError: cannot import name 'config_md' from 'memo'` (1 error).
2. `uv run --no-sync pytest tests/test_config_md.py -v`
   - Result: `7 passed in 1.91s`.
3. `git diff --check`
   - Result: passed with no whitespace errors.
4. `uv run --no-sync ruff check src/memo/config_md.py tests/test_config_md.py`
   - Result: `All checks passed!`.

## Self-Review Notes

- Parses only fenced TOML blocks from known `*-config.md` domain files.
- Maps storage/model/search values to Config fields and registry-backed values to `MEMO_*` flags without import-time `memo.flags` access.
- Normalizes supported boolean spellings for boolean flags.
- Reports invalid TOML, unknown keys, and unknown `*-config.md` files while ignoring ordinary Markdown notes.
- Includes a signature cache keyed by config home and domain file modification times.

## Concerns

- `.superpowers/sdd/task-1-brief.md` was already modified before this task and remains uncommitted; it was not changed or committed by this task.

## Boolean Canonicalization Fix

### Status

DONE

`flag_values()` now emits `"on"` and `"off"` for every accepted boolean
spelling. Invalid boolean spellings continue to be reported as `ConfigProblem`s.

### Commit

- `fix(config): normalize markdown boolean flags`

### Tests Run

1. `uv run --no-sync pytest tests/test_config_md.py -v`
   - Result: `11 passed in 0.04s`.
2. `uv run --no-sync ruff check src/memo/config_md.py tests/test_config_md.py`
   - Result: `All checks passed!`.

### Concerns

- `.superpowers/sdd/task-1-brief.md` was already modified before this fix and remains uncommitted; it was not changed or committed by this fix.

## Review Fix

### Fix Status

DONE

`validate_markdown_config()` now validates mapped flag values with the same
kind and inclusive bound rules as `memo.flags._coerce`, and validates mapped
Config fields through Pydantic without calling `Config.from_env()`. Invalid
values are returned as `ConfigProblem`s. TOML fences without a newline before
their closing delimiter are also accepted.

### Files Changed

- `src/memo/config_md.py`
- `tests/test_config_md.py`
- `.superpowers/sdd/task-1-report.md`

### Commit Created

- This commit: `fix(config): validate markdown config values`

### Tests Run

1. `uv run --no-sync pytest tests/test_config_md.py -v`
   - Result: `11 passed in 0.03s`.
2. `uv run --no-sync ruff check src/memo/config_md.py tests/test_config_md.py`
   - Result: `All checks passed!`.
3. `uv run --no-sync mypy src/memo/config_md.py`
   - Result: `Success: no issues found in 1 source file`.
4. `git diff --check`
   - Result: passed with no whitespace errors.

## Index TOML Review Fix

### Status

DONE

`memo-config.md` is now included in the parsed and signatured configuration
paths when it exists. Its fenced TOML is validated, while `config/*-config.md`
files remain later in parse order and therefore retain precedence. Unknown
`*-config.md` warnings under `config/` are unchanged; the top-level index is
recognized without warning.

### Commit

- `fix(config): parse fenced TOML from config index`

### Tests Run

1. `uv run --no-sync pytest tests/test_config_md.py -v`
   - Result: `12 passed in 0.06s`.
2. `uv run --no-sync ruff check src/memo/config_md.py tests/test_config_md.py`
   - Result: `All checks passed!`.
3. `git diff --check`
   - Result: passed with no whitespace errors.

### Concerns

- `.superpowers/sdd/task-1-brief.md` was already modified before this fix and remains uncommitted; it was not changed or committed by this fix.

## Validation Override Fix

### Status

DONE

Each parsed TOML value is now validated before it enters the effective-value
map. Later domain files still overwrite earlier index values for `load_values()`
and `flag_values()`.

### Commit

- `fix(config): validate overridden markdown config values`

### Tests Run

1. `uv run --no-sync pytest tests/test_config_md.py -v`
   - Result: `13 passed in 0.05s`.
2. `uv run --no-sync ruff check src/memo/config_md.py tests/test_config_md.py`
   - Result: `All checks passed!`.

### Concerns

- `.superpowers/sdd/task-1-brief.md` was already modified before this fix and remains uncommitted; it was not changed or committed by this fix.
