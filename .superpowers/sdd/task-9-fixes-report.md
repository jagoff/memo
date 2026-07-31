# Task 9 — Fixes for the two Critical findings (memo presence program)

Date: 2026-07-02
Repo: `/Users/fer/repos/memo` (branch `master`, shared worktree)

---

## Finding 1 (Critical): `systemMessage` unreachable on the recall-DAEMON path

### Investigation

- **Daemon short-circuit** (`src/memo/cli_recall_hook.py:184-189`): when the warm
  recall daemon answers, `connect_and_recall()` returns a pre-built JSON string and
  the hook does `print(_daemon_result); sys.exit(0)` — never reaching the
  subprocess-path systemMessage injection at `cli_recall_hook.py:484-490`. So in the
  NORMAL production config (warm daemon), the top-level `systemMessage` was dropped.

- **Where the daemon BUILDS the JSON**: `src/memo/recall_logic.py::_recall_logic`
  (the daemon's response builder). It assembles the `output` dict at
  `recall_logic.py:704-710` with only `hookSpecificOutput` — no `systemMessage`.
  The socket handler (`src/memo/recall_socket.py:220-266`) calls `_recall_logic`,
  gets `(json_string, log_fn)`, and writes the string **verbatim** to the client
  (`_write_response`) — no transformation. So the systemMessage must be added inside
  `_recall_logic`, where the hit objects (`relevant`, with titles) are in scope.

- `_recall_logic` is daemon-only: its sole caller is `recall_socket.py:220`. The
  subprocess path in `cli_recall_hook.py` inlines its own logic and does NOT call
  `_recall_logic`. So adding systemMessage there affects ONLY the daemon path; the
  subprocess (cold) path is untouched and keeps working exactly as before.

- **Verification asks:**
  - **(a) CITE_INSTRUCTION on the daemon path?** NO — `_recall_logic` never appends
    `CITE_INSTRUCTION`; only the subprocess path does (`cli_recall_hook.py:455-458`).
    This is the SAME regression class (both systemMessage and CITE were added to the
    subprocess path only, by the presence commits `1f4d2cc` / `00b43f3`). **Not fixed
    here** — appending CITE to `additionalContext` after the token-budget render would
    break the legitimate budget-cap invariant test
    `tests/test_recall_hooks.py::test_recall_logic_caps_total_context_and_logs_exact_cost`
    (asserts `len(context) <= budget*4`; CITE default-on adds ~186 chars past the cap),
    and raises a real design question (should CITE count toward the token budget?)
    beyond this finding. Flagged as follow-up. systemMessage is a SEPARATE top-level
    field, so it does NOT affect `additionalContext` length and is safe.
  - **(b) presence bump on the daemon path?** NO. `presence.bump(recalls=N)` is called
    ONLY on the subprocess path (`cli_recall_hook.py:508`), AFTER the daemon
    short-circuit. On the daemon path the hook exits at line 189 before ever bumping.
    So the daemon path currently does NOT bump `recalls` at all. The live `recalls=6`
    came from the subprocess-path fallbacks (cold starts / daemon misses), which DO
    bump. **No double-bump risk** from this fix: I did NOT add any presence bump to
    `_recall_logic`; the only bump site remains the subprocess path. (Pre-existing gap
    noted: the daemon path under-counts `recalls`; out of scope for the systemMessage
    finding.)

### Change

`src/memo/recall_logic.py` — in `_recall_logic`, at the output-dict assembly:

- Annotated `output: dict[str, Any]` (was inferred `dict[str, dict[str, str]]`; needed
  so mypy accepts the string `systemMessage` value).
- Added a flag-gated (`MEMO_RECALL_SYSTEM_MESSAGE`, default-on) `systemMessage` built
  from `build_system_message(relevant)`, wrapped in try/except that degrades to
  omitting the field. `build_system_message` and the flag helper already live in this
  module / are already imported — **no new module-level imports touch the hot path.**

The no-hits case still returns `"{}"` early (recall_logic.py:648) before the output
dict is built, so systemMessage is only emitted when there are hits — matching the
subprocess path.

### Tests

`tests/test_recall_sysmsg.py` — added two daemon-path tests exercising the REAL
builder (`_recall_logic`) via the `StubMemory` harness (vec mode never calls the
embedder):

- `test_daemon_path_emits_system_message_when_flag_on` — asserts the daemon-built
  JSON has `systemMessage` starting `🧠 memo · 1: ` with the hit title, and that
  `additionalContext` is still present.
- `test_daemon_path_omits_system_message_when_flag_off` — with
  `MEMO_RECALL_SYSTEM_MESSAGE=0`, asserts `systemMessage` is absent and
  `additionalContext` remains.

Existing `_recall_logic` tests (substring / `== "{}"` assertions on the returned JSON
string) still pass — systemMessage contains only the title, and the no-hit path
returns before the dict is built.

---

## Finding 2 (Critical): install-seed side effect leaked into the shared test state dir

### Investigation / root cause

`install-mcp … --write` calls `_seed_install_memory()`
(`src/memo/cli_install_mcp.py:355-356`), which does a real `Memory.save()` resolved via
`Config.from_env()` → conftest's SHARED `MEMO_STATE_DIR`
(`tests/conftest.py:59`, `$TMPDIR/memo-test-nonexistent-state`). `Memory.save()` calls
`_presence_bump_save()` (`src/memo/memory/write_ops.py:119-123`) which writes
`presence_today.json` with `saves:1`. The two `--write` install-mcp tests
(`test_install_mcp_jetbrains_prints_snippet`, `test_install_mcp_only_present_filters`)
used a plain `CliRunner().invoke(...)` with NO `env=` isolation, so the seed leaked into
the shared dir. `tests/test_statusline.py::test_script_standalone_still_emits_badge` and
`::test_script_wrap_prepends_memo_badge` inherit that dir via `{**os.environ, ...}` and
assert the exact substring `[Memo 9.9.9]`, which the leaked presence renders as
`[Memo 9.9.9 · 💾1]` → deterministic 2-failure in alphabetical suite order.

### Changes (both layers + artifact cleanup)

1. **Root cause** — `tests/test_cli_install_mcp.py`: added a function-scoped autouse
   fixture `_no_seed_install_memory` that monkeypatches
   `memo.cli_install_mcp._seed_install_memory` to a no-op for the whole module. The
   seed keeps its own coverage in `tests/test_install_seed.py`. (Chosen over per-test
   `env=` isolation because it is the smallest, style-matching change and covers any
   future `--write` test in the file.)

2. **Defense-in-depth** — `tests/test_statusline.py`: added
   `"MEMO_STATUSLINE_ACTIVITY": "0"` to the `env` of the two exact-bracket tests
   (`test_script_wrap_prepends_memo_badge`, `test_script_standalone_still_emits_badge`),
   so ambient presence can never alter the `[Memo <ver>]` bracket. (The bundled script
   honors this flag at `src/memo/agent_assets/statusline/memo-statusline.sh:44`.)

3. **Artifact cleanup NOW** — removed `presence_today.json` / `.install_seed.json` (and
   any `presence_today.json.tmp`) from the shared dir
   `/var/folders/_v/8v3nrp2974d167pcsrglrv4r0000gn/T/memo-test-nonexistent-state`.
   They were already absent when checked (a concurrent session had cleaned them); the
   defensive removal was a no-op. Nothing else in that dir was touched (`.device_id`,
   `graph.db`, `history.db`, `memvec.db` left as-is).

---

## Verification

- **Targeted:** `pytest tests/test_recall_sysmsg.py tests/test_cli_install_mcp.py
  tests/test_statusline.py tests/test_install_seed.py tests/test_recall_hooks.py
  tests/test_recall_logic_synthesis.py tests/test_recall_server.py` →
  **76 passed, 5 skipped**.
- **Full suite:** `uv run --no-sync pytest tests/` → SEE FINAL LINE BELOW.
- **mypy:** `uv run --no-sync mypy src/memo/recall_logic.py` → Success, no issues.
- **ruff:** `uv run --no-sync ruff check` on all four touched files → All checks passed.

### Live daemon smoke — IMPORTANT caveat

The launchd recall daemon (`com.synapse.memo-recall-daemon`) on this machine runs the
**installed, non-editable `mlx-memo 2.9.8`** from
`~/.local/share/uv/tools/mlx-memo/lib/python3.14/site-packages/memo`, NOT the repo
working tree. (My interactive shell exports `PYTHONPATH=src`, which is why direct calls
resolve the repo source; launchd runs a clean env and imports site-packages.) The
installed 2.9.8 predates the ENTIRE presence program — its `_recall_logic` has no
systemMessage and its flag registry lacks `MEMO_RECALL_SYSTEM_MESSAGE` (KeyError). So
`kickstart`-ing the daemon and running `memo recall-hook` end-to-end can NOT reflect the
fix until memo is re-released and reinstalled — a separate manual release step, out of
scope, and not done here.

Authoritative daemon-path proof instead: executing the REAL repo-source `_recall_logic`
in the daemon's OWN Python 3.14 interpreter (with `PYTHONPATH=src`) returned
`keys: ['hookSpecificOutput', 'systemMessage']`,
`systemMessage: 🧠 memo · 1: sync tier decision`. The socket handler writes
`_recall_logic`'s return value verbatim, so the socket path emits identically once the
daemon runs this code. The flag-off omission is covered by the new unit test. Direct
socket probe against the running daemon confirmed it lacks the flag entirely (proving it
is the pre-presence 2.9.8, not a fault in the fix).

---

## Commits

1. `fix(presence): emit systemMessage on the recall-daemon path`
   — `src/memo/recall_logic.py`, `tests/test_recall_sysmsg.py`
2. `test(presence): isolate install-seed side effect from shared state dir`
   — `tests/test_cli_install_mcp.py`, `tests/test_statusline.py`

## Follow-ups (not done — out of scope)

- CITE_INSTRUCTION is also missing on the daemon path (same regression). Fixing it needs
  a decision on whether it counts toward the recall token budget; the current budget-cap
  test would fail if it's appended naively.
- The daemon path does not `presence.bump(recalls=…)`, so `recalls` under-counts under
  the warm-daemon config. Separate from this finding.

---

## Task-9 Round 2: CITE_INSTRUCTION + presence recalls (commit 08518db)

Date: 2026-07-02

### Caller audit

`grep -rn "_recall_logic" src/memo/` shows three files reference the symbol:
- `src/memo/recall_socket.py:220` — the sole **caller** (socket handler, daemon-only)
- `src/memo/recall_logic.py:427` — the definition
- `src/memo/recall_server.py:22,68` — a facade re-export (`__all__`), never calls it

`cli_recall_hook.py` imports `CITE_INSTRUCTION` and `build_system_message` from
`recall_logic` but has its own inlined recall path and does NOT call `_recall_logic`.
Therefore `_recall_logic` is daemon-only — adding both fixes inside it causes no
double-bump on the subprocess path.

### Gap 1: CITE_INSTRUCTION

Added after the associative-nudge block in `_recall_logic` (src/memo/recall_logic.py),
before `hits_snapshot` is built:

```python
# Cite instruction — budget-exempt (~30 tokens), appended after any token-cap.
if flag_bool("MEMO_RECALL_CITE_INSTRUCTION"):
    context = f"{context}\n{CITE_INSTRUCTION}"
```

`CITE_INSTRUCTION` is already defined at module level in `recall_logic.py` — no new import.
The budget-cap test `test_recall_logic_caps_total_context_and_logs_exact_cost`
(`tests/test_recall_hooks.py:170`) was adjusted to strip the cite suffix before the
`len(core_context) <= budget * 4` assertion, proving the core content respects the budget
while the instruction is explicitly exempt.

### Gap 2: Presence recalls

Added at the end of `_recall_logic` before the return, matching the subprocess-path
guard pattern exactly:

```python
try:
    from memo import presence as _presence_mod

    _presence_mod.bump(cfg.state_dir, recalls=len(relevant))
except Exception as exc:
    _logger.debug("presence bump failed: %s", exc)
```

Placed in `_recall_logic` (not the socket handler) because the `relevant` list and `cfg`
are both in scope there, and the function is daemon-only so no double-bump risk.

### Tests added (tests/test_recall_sysmsg.py)

- `test_daemon_path_emits_cite_instruction_when_flag_on` — CITE_INSTRUCTION in
  `additionalContext` with flag default-on
- `test_daemon_path_omits_cite_instruction_when_flag_off` — absent with
  `MEMO_RECALL_CITE_INSTRUCTION=0`; context still present
- `test_daemon_path_bumps_presence_recalls` — `presence.read_today(tmp_path)["recalls"]`
  increments by 1 after invoking `_recall_logic` with a hit, using isolated `tmp_path`

### Suite results

Full suite (excluding test_recall_hook.py which serialises against the live MLX GPU):
**2314 passed, 29 skipped** (same baseline). mypy + ruff clean on all touched files.
