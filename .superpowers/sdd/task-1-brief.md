### Task 1: Flag registry

**Files:**
- Modify: `src/memo/flags_misc.py` (append `_spec(...)` entries to `SPECS`)
- Test: `tests/test_proactive_flags.py`

**Interfaces:**
- Produces: env flags `MEMO_PROACTIVE_ENABLED` (bool, default False), `MEMO_PROACTIVE_PUSH_COOLDOWN_H` (int, default 6), `MEMO_PROACTIVE_DAILY_CAP` (int, default 3), `MEMO_PROACTIVE_MULT_FLOOR` (float, default 0.2), `MEMO_PROACTIVE_URGENT_MIN` (float, default 0.7), `MEMO_PROACTIVE_DIGEST_TOP` (int, default 7). Read via `flag_bool/int/float`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_proactive_flags.py
from memo.flags import flag_bool, flag_int, flag_float


def test_proactive_flags_defaults():
    assert flag_bool("MEMO_PROACTIVE_ENABLED") is False
    assert flag_int("MEMO_PROACTIVE_PUSH_COOLDOWN_H") == 6
    assert flag_int("MEMO_PROACTIVE_DAILY_CAP") == 3
    assert flag_float("MEMO_PROACTIVE_MULT_FLOOR") == 0.2
    assert flag_float("MEMO_PROACTIVE_URGENT_MIN") == 0.7
    assert flag_int("MEMO_PROACTIVE_DIGEST_TOP") == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_proactive_flags.py -v`
Expected: FAIL — unknown flag / KeyError.

- [ ] **Step 3: Add the specs**

Append to the `SPECS` tuple in `src/memo/flags_misc.py` (before its closing `)`), matching the `_spec(name, kind, default, group, help, ...)` signature:
```python
(
    _spec(
        "MEMO_PROACTIVE_ENABLED",
        "bool",
        False,
        "misc",
        "Master switch for the proactive engine (statusline badge, urgent push, "
        "`memo digest`). Default off — dark flag, graduates via dream_flags.",
    ),
)
(
    _spec(
        "MEMO_PROACTIVE_PUSH_COOLDOWN_H",
        "int",
        6,
        "misc",
        "Minimum hours between urgent pushes.",
        min_val=0,
    ),
)
(
    _spec(
        "MEMO_PROACTIVE_DAILY_CAP",
        "int",
        3,
        "misc",
        "Hard cap on proactive pushes per day.",
        min_val=0,
    ),
)
(
    _spec(
        "MEMO_PROACTIVE_MULT_FLOOR",
        "float",
        0.2,
        "misc",
        "Floor for the adaptive per-kind multiplier (reliability can never be fully muted).",
        min_val=0.0,
        max_val=1.0,
    ),
)
(
    _spec(
        "MEMO_PROACTIVE_URGENT_MIN",
        "float",
        0.7,
        "misc",
        "Minimum score for a reliability nudge to qualify for an urgent push.",
        min_val=0.0,
        max_val=1.0,
    ),
)
(
    _spec(
        "MEMO_PROACTIVE_DIGEST_TOP",
        "int",
        7,
        "misc",
        "Max items shown in `memo digest`.",
        min_val=1,
    ),
)
```

- [ ] **Step 4: Run test + config validate**

Run: `uv run --no-sync pytest tests/test_proactive_flags.py -v && uv run --no-sync memo config validate`
Expected: PASS; validate reports all flags valid.

- [ ] **Step 5: Commit**
```bash
git add src/memo/flags_misc.py tests/test_proactive_flags.py
git commit -m "feat(proactive): register MEMO_PROACTIVE_* flags (default off)"
```

---

