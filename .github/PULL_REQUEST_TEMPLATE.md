## What

<!-- One paragraph: what changes and why. Link the issue if there is one. -->

## Verification

<!-- Paste the commands you actually ran and their outcome. "Should pass" is not verification. -->

```
uv run --no-sync pytest tests/
uv run --no-sync mypy src/memo/
uv run --no-sync ruff check src/ tests/
```

## Invariant checklist

Tick only what applies to this diff. An unticked box on a line the diff touches
blocks the merge.

- [ ] **Retrieval / ingest / ranking changed** — the fix is systemic, not a patch
      for one query, and `memo eval recall --labels eval/regression_labels.json --k 5 --gate`
      holds against the saved baseline. A new incident added a labeled prompt.
- [ ] **MLX path changed** (`embedder.py`, `llm.py`, `memory/`, anything calling
      `.embed()`/`.chat()`) — all four invariants preserved: query-only retrieval
      prefix, `embed()` takes `Sequence[str]`, `MEMO_EMBEDDER_DIMS` matches the
      model, `mlx`/`mlx-lm` imports stay deferred.
- [ ] **Recall hook path changed** (`hooks/hooks.json`, `recall-hook` in `cli.py`,
      `embedder.py`, `store/queries.py` search) — measured end-to-end under the
      ~5s budget.
- [ ] **Store / markdown authority changed** — a failed index still stamps
      `_memo_embed_pending`; `delete()` still drops the index first and unlinks
      the `.md` last; `memo reindex --rebuild` still preserves the user-signal
      tables (`access`, `memory_health`, `source_feedback*`).
- [ ] **Release surfaces changed** — `python3 scripts/adapter_matrix.py --check`
      passes (versions locked across `pyproject.toml`, `.claude-plugin/plugin.json`,
      `plugins/memo/.codex-plugin/plugin.json`, `server.json`, `CHANGELOG.md`).
- [ ] **A new `MEMO_*` flag was added** — registered in the matching
      `flags_<group>.py`, accessed via `flag_bool/int/float/str`, and (if it is a
      default-off `*_ENABLED` flag) declares its gate in `dream_flags.GATES`.
- [ ] **Tests added/changed** — isolated per `tests/conftest.py` (`tmp_cfg` or an
      explicit `Config`; `CliRunner` sets `MEMO_NONINTERACTIVE=1` +
      `MEMO_DATA_DIR` + `MEMO_STATE_DIR`; never touches the real vault).

## Risk / rollback

<!-- What breaks if this is wrong, and how it gets reverted. Say "none" only if
     the change is genuinely inert. -->
