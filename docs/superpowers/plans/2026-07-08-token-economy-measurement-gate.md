# Token-Economy Measurement Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `memo eval tokens` — a gate that measures, per token-economy lever, whether it saves tokens without degrading recall, so the data (not vibes) decides which levers get wired ON or pruned.

**Architecture:** A new pure-Python harness `src/memo/eval_tokens.py` measures two planes memo controls directly and that need no LLM: **P1** recall-output block size (quality guard = precision over ids surviving into the block) and **P2** capture size (quality guard = a labeled must-keep row surviving the crush). A `memo eval tokens` subcommand clones the `memo eval recall` gate wiring (baseline file under `state_dir`, `--update-baseline`/`--gate`, exit 0/1). `cli_token_savings.py` is rewritten to report the measured deltas instead of a hardcoded 65%.

**Tech Stack:** Python 3.11+, Click, pytest, `dataclasses`. Reuses `memo.eval_recall` (labels), `memo.recall_logic.render_recall_context` (P1 render), `memo.capture_core.maybe_crush_json_capture` (P2 crush), `memo.token_meter._CHARS_PER_TOKEN` (token counting).

## Global Constraints

- Pure helpers must be testable **without MLX**; only the live search path (P1 in the CLI) touches the embedder. Never call `Config.from_env()` in tests without env control — use the `tmp_cfg` fixture (`tests/conftest.py`).
- Token count is `ceil(chars / 4)` via the single `_CHARS_PER_TOKEN = 4` heuristic — consistency over absolute accuracy for a *delta* gate.
- The gate is **machine-local, opt-in** — baseline lives under `state_dir/eval/`, never a committed repo file (mirrors `memo eval recall --gate`).
- Measure the transformation function; do **not** wire any lever into the live pipeline in this plan. Wiring winners is a separate follow-up.
- New Python files ≤ 800 lines. Match existing style (type annotations on all signatures, `from __future__ import annotations`).
- Constants: `MIN_SAVING_FRAC = 0.05` (a lever must cut ≥5% tokens on its plane), `QUALITY_EPS = 0.0` (precision must not drop; must-keep row must survive).

---

### Task 1: Token counting + surviving-id helpers

**Files:**
- Create: `src/memo/eval_tokens.py`
- Test: `tests/test_eval_tokens.py`

**Interfaces:**
- Produces: `count_tokens(text: str) -> int`; `surviving_ids(block_text: str, candidate_ids: list[str]) -> set[str]`; module constants `MIN_SAVING_FRAC: float`, `QUALITY_EPS: float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_tokens.py
from memo import eval_tokens


def test_count_tokens_is_ceil_chars_over_four():
    assert eval_tokens.count_tokens("") == 0
    assert eval_tokens.count_tokens("abcd") == 1
    assert eval_tokens.count_tokens("abcde") == 2  # 5 chars -> ceil(5/4) == 2


def test_surviving_ids_matches_eight_char_prefix_in_block():
    block = "**[5d7d253a] Some title**\n> body text with [ee73e5e9] too"
    candidates = ["5d7d253a1122", "ee73e5e9ffff", "deadbeefcafe"]
    assert eval_tokens.surviving_ids(block, candidates) == {"5d7d253a1122", "ee73e5e9ffff"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'memo.eval_tokens'` (or `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/memo/eval_tokens.py
"""`memo eval tokens` — token-economy measurement gate.

Mirrors the `memo eval recall --gate` retrieval-regression discipline for the
token economy. Measures, per lever, whether it saves tokens WITHOUT degrading
recall, over a committed corpus. Two planes memo controls directly (no LLM):

  P1 recall-output block size — quality guard = precision over ids that survive
     into the injected block.
  P2 capture size — quality guard = a labeled must-keep row surviving the crush.

Design note: we measure the transformation FUNCTION at unit level; a lever need
not be wired into the live pipeline to be measured. The data decides wiring.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4  # keep in lockstep with token_meter._CHARS_PER_TOKEN

MIN_SAVING_FRAC = 0.05  # a lever must cut >=5% tokens on its plane to count
QUALITY_EPS = 0.0  # precision must not drop; must-keep row must survive


def count_tokens(text: str) -> int:
    """Estimated tokens for `text` via the chars/4 heuristic (ceil)."""
    if not text:
        return 0
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def surviving_ids(block_text: str, candidate_ids: list[str]) -> set[str]:
    """Which candidate memory ids survived into a rendered recall block.

    The renderer prints each hit as `**[<id[:8]>] title**`, so an id counts as
    surviving iff its 8-char short prefix appears anywhere in the block.
    """
    return {cid for cid in candidate_ids if cid[:8] in block_text}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/memo/eval_tokens.py tests/test_eval_tokens.py
git commit -m "feat(eval-tokens): token counting + surviving-id helpers"
```

---

### Task 2: P1 recall-output measurement (pure)

**Files:**
- Modify: `src/memo/eval_tokens.py`
- Test: `tests/test_eval_tokens.py`

**Interfaces:**
- Consumes: `count_tokens`, `surviving_ids` (Task 1).
- Produces: `@dataclass LeverRow(lever: str, plane: str, tokens_off: int, tokens_on: int, quality_off: float, quality_on: float)` with properties `saved_frac: float`, `quality_delta: float`, `passed: bool`; `measure_recall_sample(block_off: str, block_on: str, expect_ids: list[str]) -> P1Sample` where `@dataclass P1Sample(tokens_off, tokens_on, prec_off, prec_on)`; `aggregate_recall(lever: str, samples: list[P1Sample]) -> LeverRow`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_tokens.py  (append)
def test_lever_row_passed_requires_saving_and_no_quality_drop():
    # 100 -> 90 tokens (10% saving), precision unchanged -> PASS
    good = eval_tokens.LeverRow("compact", "recall_output", 100, 90, 1.0, 1.0)
    assert good.saved_frac == 0.1
    assert good.quality_delta == 0.0
    assert good.passed is True
    # saves tokens but drops precision -> FAIL
    lossy = eval_tokens.LeverRow("aggressive", "recall_output", 100, 50, 1.0, 0.5)
    assert lossy.passed is False
    # keeps precision but no saving -> FAIL
    nosave = eval_tokens.LeverRow("noop", "recall_output", 100, 99, 1.0, 1.0)
    assert nosave.passed is False


def test_measure_recall_sample_scores_surviving_expect_ids():
    off = "**[aaaaaaaa] t**\n> long body here that is bigger\n**[bbbbbbbb] u**"
    on = "**[aaaaaaaa] t**"  # smaller block, but bbbbbbbb dropped
    s = eval_tokens.measure_recall_sample(off, on, expect_ids=["aaaaaaaa11", "bbbbbbbb22"])
    assert s.tokens_on < s.tokens_off
    assert s.prec_off == 1.0  # both expected ids present in off
    assert s.prec_on == 0.5  # only aaaaaaaa survived in on


def test_aggregate_recall_sums_tokens_and_means_precision():
    samples = [
        eval_tokens.P1Sample(tokens_off=100, tokens_on=80, prec_off=1.0, prec_on=1.0),
        eval_tokens.P1Sample(tokens_off=60, tokens_on=60, prec_off=1.0, prec_on=0.0),
    ]
    row = eval_tokens.aggregate_recall("compact", samples)
    assert row.tokens_off == 160 and row.tokens_on == 140
    assert row.quality_off == 1.0
    assert row.quality_on == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -q`
Expected: FAIL — `AttributeError: module 'memo.eval_tokens' has no attribute 'LeverRow'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/memo/eval_tokens.py  (append after the helpers)
from dataclasses import dataclass


@dataclass
class LeverRow:
    lever: str
    plane: str  # "recall_output" | "capture"
    tokens_off: int
    tokens_on: int
    quality_off: float
    quality_on: float

    @property
    def saved_frac(self) -> float:
        return (self.tokens_off - self.tokens_on) / self.tokens_off if self.tokens_off else 0.0

    @property
    def quality_delta(self) -> float:
        return self.quality_on - self.quality_off

    @property
    def passed(self) -> bool:
        return self.saved_frac >= MIN_SAVING_FRAC and self.quality_delta >= -QUALITY_EPS


@dataclass
class P1Sample:
    tokens_off: int
    tokens_on: int
    prec_off: float
    prec_on: float


def _precision(block: str, expect_ids: list[str]) -> float:
    if not expect_ids:
        return 1.0
    return len(surviving_ids(block, expect_ids)) / len(expect_ids)


def measure_recall_sample(block_off: str, block_on: str, expect_ids: list[str]) -> P1Sample:
    """One prompt: token + precision measurement of the OFF vs ON recall block."""
    return P1Sample(
        tokens_off=count_tokens(block_off),
        tokens_on=count_tokens(block_on),
        prec_off=_precision(block_off, expect_ids),
        prec_on=_precision(block_on, expect_ids),
    )


def aggregate_recall(lever: str, samples: list[P1Sample]) -> LeverRow:
    """Fold per-prompt P1 samples into one LeverRow (sum tokens, mean precision)."""
    n = len(samples) or 1
    return LeverRow(
        lever=lever,
        plane="recall_output",
        tokens_off=sum(s.tokens_off for s in samples),
        tokens_on=sum(s.tokens_on for s in samples),
        quality_off=sum(s.prec_off for s in samples) / n,
        quality_on=sum(s.prec_on for s in samples) / n,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/memo/eval_tokens.py tests/test_eval_tokens.py
git commit -m "feat(eval-tokens): P1 recall-block token+precision measurement"
```

---

### Task 2b: P1 render seam + verbosity-lever env pins

**Files:**
- Modify: `src/memo/eval_tokens.py`
- Test: `tests/test_eval_tokens.py`

**Interfaces:**
- Consumes: `measure_recall_sample`, `aggregate_recall` (Task 2).
- Produces: context manager `env_pins(overrides: dict[str, str])`; `render_block(hits: list, env: dict[str, str], *, body_chars: int, token_budget: int) -> str`; `RECALL_LEVERS: list[dict]` (each `{"name": str, "env": dict[str, str]}`).

**Note:** `render_block` wraps `recall_logic.render_recall_context(hits, [], turn=2, body_chars=..., token_budget=...)` under env pins, then applies L4 verbosity steering when `MEMO_RECALL_VERBOSITY_LEVEL > 0` — modelling L4 faithfully (it *appends* a steering block, so it costs tokens on P1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_tokens.py  (append)
from dataclasses import dataclass as _dc


@_dc
class _FakeHit:
    id: str
    title: str
    body: str
    score: float | None = 0.9
    tags: tuple[str, ...] = ()


def test_env_pins_sets_and_restores(monkeypatch):
    monkeypatch.delenv("MEMO_RECALL_FORMAT", raising=False)
    with eval_tokens.env_pins({"MEMO_RECALL_FORMAT": "compact"}):
        import os
        assert os.environ["MEMO_RECALL_FORMAT"] == "compact"
    import os
    assert "MEMO_RECALL_FORMAT" not in os.environ


def test_render_block_verbosity_level_appends_steering():
    hits = [_FakeHit(id="aaaaaaaa11", title="T", body="b")]
    plain = eval_tokens.render_block(hits, {}, body_chars=200, token_budget=200)
    steered = eval_tokens.render_block(
        hits, {"MEMO_RECALL_VERBOSITY_LEVEL": "2"}, body_chars=200, token_budget=200
    )
    # L4 adds a steering block -> steered is LONGER (the paradox: L4 costs P1 tokens)
    assert len(steered) > len(plain)
    assert "aaaaaaaa" in plain  # id short-prefix present for surviving_ids()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -k "env_pins or render_block" -q`
Expected: FAIL — `AttributeError: ... has no attribute 'env_pins'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/memo/eval_tokens.py  (append)
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def env_pins(overrides: dict[str, str]) -> Iterator[None]:
    """Temporarily set MEMO_* env vars, restoring prior state on exit."""
    prior: dict[str, str | None] = {k: os.environ.get(k) for k in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def render_block(
    hits: list[Any], env: dict[str, str], *, body_chars: int, token_budget: int
) -> str:
    """Render the recall block for `hits` under a lever's env pins.

    Applies L4 verbosity steering when MEMO_RECALL_VERBOSITY_LEVEL > 0, so the
    lever's true effect on the injected block size is measured.
    """
    from memo.cli_recall_hook import maybe_inject_verbosity_steering
    from memo.flags_recall import flag_recall_verbosity_level
    from memo.recall_logic import render_recall_context

    with env_pins(env):
        block = render_recall_context(
            hits, [], turn=2, body_chars=body_chars, token_budget=token_budget
        )
        level = flag_recall_verbosity_level()
        if level > 0:
            block = maybe_inject_verbosity_steering(block, level)
        return block


# The P1 levers to measure. Env-expressible knobs that transform the injected
# recall block. Baseline (OFF) is the empty env; each lever is its ON delta.
RECALL_LEVERS: list[dict[str, Any]] = [
    {"name": "recall_format_compact", "env": {"MEMO_RECALL_FORMAT": "compact"}},
    {"name": "verbosity_steer_L2", "env": {"MEMO_RECALL_VERBOSITY_LEVEL": "2"}},
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -k "env_pins or render_block" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/memo/eval_tokens.py tests/test_eval_tokens.py
git commit -m "feat(eval-tokens): P1 render seam + verbosity/format levers"
```

---

### Task 3: P2 capture corpus + crush measurement (pure)

**Files:**
- Create: `eval/token_corpus.json`
- Modify: `src/memo/eval_tokens.py`
- Test: `tests/test_eval_tokens.py`

**Interfaces:**
- Consumes: `count_tokens`, `LeverRow` (Tasks 1–2).
- Produces: `load_capture_corpus(path: Path) -> list[CaptureCase]` where `@dataclass CaptureCase(name: str, rows: list, must_keep_index: int)`; `measure_crush_case(case: CaptureCase, crush_fn: Callable[[str], tuple[str, str | None]]) -> P2Sample` where `@dataclass P2Sample(tokens_off, tokens_on, survived)`; `aggregate_capture(lever: str, samples: list[P2Sample]) -> LeverRow`.

**Note:** `crush_fn` is a closure over the crusher; passing content through it must run under `MEMO_CRUSHER_ENABLED=1`. With the current placeholder `0.5` scorer, the crusher keeps the first `keep_count` rows in original order, so a must-keep row near the end is dropped — this is exactly the honest verdict v1 should surface (L1 needs a real scorer to pass P2).

- [ ] **Step 1: Create the committed corpus fixture**

Author `eval/token_corpus.json` with the schema below. Rows must be realistic tool-output objects; each case is a JSON array ≥ 10 rows with the answer (`must_keep_index`) placed **late** so the placeholder scorer drops it, plus one case where it survives (index 0):

```json
{
  "schema": "memo.token_corpus.v1",
  "_doc": "Realistic tool-output JSON arrays for the P2 capture plane. Each case's must_keep_index marks the row that MUST survive the crush (the 'answer'). Placed late in most cases so a position-only scorer drops it -> the gate's quality guard FAILs, documenting that L1 needs a real relevance scorer.",
  "cases": [
    {
      "name": "grep_hits_answer_last",
      "must_keep_index": 29,
      "rows": [
        {"file": "src/a01.py", "line": 12, "match": "def unrelated_a(): pass"},
        {"file": "src/a02.py", "line": 8, "match": "def unrelated_b(): pass"},
        {"file": "src/a03.py", "line": 3, "match": "import os"},
        {"file": "src/a04.py", "line": 44, "match": "return None"},
        {"file": "src/a05.py", "line": 5, "match": "x = 1"},
        {"file": "src/a06.py", "line": 9, "match": "y = 2"},
        {"file": "src/a07.py", "line": 1, "match": "# header"},
        {"file": "src/a08.py", "line": 7, "match": "pass"},
        {"file": "src/a09.py", "line": 2, "match": "z = 3"},
        {"file": "src/a10.py", "line": 6, "match": "log.info('x')"},
        {"file": "src/a11.py", "line": 4, "match": "raise ValueError"},
        {"file": "src/a12.py", "line": 11, "match": "continue"},
        {"file": "src/a13.py", "line": 13, "match": "break"},
        {"file": "src/a14.py", "line": 14, "match": "yield"},
        {"file": "src/a15.py", "line": 15, "match": "assert True"},
        {"file": "src/a16.py", "line": 16, "match": "print('a')"},
        {"file": "src/a17.py", "line": 17, "match": "del q"},
        {"file": "src/a18.py", "line": 18, "match": "global g"},
        {"file": "src/a19.py", "line": 19, "match": "nonlocal n"},
        {"file": "src/a20.py", "line": 20, "match": "lambda: 0"},
        {"file": "src/a21.py", "line": 21, "match": "with open('f'): pass"},
        {"file": "src/a22.py", "line": 22, "match": "try: pass"},
        {"file": "src/a23.py", "line": 23, "match": "except: pass"},
        {"file": "src/a24.py", "line": 24, "match": "finally: pass"},
        {"file": "src/a25.py", "line": 25, "match": "class Foo: pass"},
        {"file": "src/a26.py", "line": 26, "match": "@decorator"},
        {"file": "src/a27.py", "line": 27, "match": "async def h(): pass"},
        {"file": "src/a28.py", "line": 28, "match": "await x"},
        {"file": "src/a29.py", "line": 29, "match": "return sum(v)"},
        {"file": "src/target.py", "line": 42, "match": "def maybe_crush_json_capture(content, context, config):"}
      ]
    },
    {
      "name": "search_results_answer_first",
      "must_keep_index": 0,
      "rows": [
        {"rank": 1, "id": "hit_the_answer", "text": "the row we must keep"},
        {"rank": 2, "id": "n02", "text": "noise 2"},
        {"rank": 3, "id": "n03", "text": "noise 3"},
        {"rank": 4, "id": "n04", "text": "noise 4"},
        {"rank": 5, "id": "n05", "text": "noise 5"},
        {"rank": 6, "id": "n06", "text": "noise 6"},
        {"rank": 7, "id": "n07", "text": "noise 7"},
        {"rank": 8, "id": "n08", "text": "noise 8"},
        {"rank": 9, "id": "n09", "text": "noise 9"},
        {"rank": 10, "id": "n10", "text": "noise 10"},
        {"rank": 11, "id": "n11", "text": "noise 11"},
        {"rank": 12, "id": "n12", "text": "noise 12"}
      ]
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_eval_tokens.py  (append)
import json as _json
from pathlib import Path


def _write_corpus(tmp_path: Path) -> Path:
    corpus = {
        "schema": "memo.token_corpus.v1",
        "cases": [
            {"name": "late", "must_keep_index": 11,
             "rows": [{"i": i, "text": f"row {i}"} for i in range(11)]
                     + [{"i": 11, "text": "THE ANSWER"}]},
            {"name": "early", "must_keep_index": 0,
             "rows": [{"i": 0, "text": "THE ANSWER"}]
                     + [{"i": i, "text": f"row {i}"} for i in range(1, 12)]},
        ],
    }
    p = tmp_path / "token_corpus.json"
    p.write_text(_json.dumps(corpus), encoding="utf-8")
    return p


def test_load_capture_corpus(tmp_path):
    cases = eval_tokens.load_capture_corpus(_write_corpus(tmp_path))
    assert [c.name for c in cases] == ["late", "early"]
    assert cases[0].must_keep_index == 11
    assert len(cases[1].rows) == 12


def test_measure_crush_case_flags_dropped_answer():
    case = eval_tokens.CaptureCase(
        name="late", must_keep_index=11,
        rows=[{"i": i, "text": f"row {i}"} for i in range(11)] + [{"i": 11, "text": "ANSWER"}],
    )

    def crush_fn(content: str) -> tuple[str, str | None]:
        # Simulate a position-only crusher: keep first 10 rows, drop the rest.
        arr = _json.loads(content)
        return _json.dumps(arr[:10]), "hash"

    s = eval_tokens.measure_crush_case(case, crush_fn)
    assert s.tokens_on < s.tokens_off  # crushing saved tokens
    assert s.survived is False  # index 11 was dropped -> quality FAIL

    row = eval_tokens.aggregate_capture("crusher", [s])
    assert row.plane == "capture"
    assert row.quality_on == 0.0
    assert row.passed is False  # saved tokens but dropped the answer
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -k "corpus or crush_case" -q`
Expected: FAIL — `AttributeError: ... has no attribute 'load_capture_corpus'`.

- [ ] **Step 4: Write minimal implementation**

```python
# src/memo/eval_tokens.py  (append)
import json
from collections.abc import Callable
from pathlib import Path

CAPTURE_CORPUS_SCHEMA = "memo.token_corpus.v1"


@dataclass
class CaptureCase:
    name: str
    rows: list[Any]
    must_keep_index: int


@dataclass
class P2Sample:
    tokens_off: int
    tokens_on: int
    survived: bool


def load_capture_corpus(path: Path) -> list[CaptureCase]:
    """Load the committed P2 capture corpus (schema memo.token_corpus.v1)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise ValueError(f"capture corpus {path} must be an object with a `cases` list")
    cases: list[CaptureCase] = []
    for c in raw["cases"]:
        cases.append(
            CaptureCase(
                name=str(c["name"]),
                rows=list(c["rows"]),
                must_keep_index=int(c["must_keep_index"]),
            )
        )
    return cases


def measure_crush_case(
    case: CaptureCase, crush_fn: Callable[[str], tuple[str, str | None]]
) -> P2Sample:
    """One capture case: token delta + whether the must-keep row survived."""
    original = json.dumps(case.rows, ensure_ascii=False)
    crushed, _hash = crush_fn(original)
    survived = _row_survived(case.rows[case.must_keep_index], crushed)
    return P2Sample(
        tokens_off=count_tokens(original),
        tokens_on=count_tokens(crushed),
        survived=survived,
    )


def _row_survived(row: Any, crushed_content: str) -> bool:
    try:
        arr = json.loads(crushed_content)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(arr, list) and row in arr


def aggregate_capture(lever: str, samples: list[P2Sample]) -> LeverRow:
    """Fold per-case P2 samples into one LeverRow (mean survival = quality)."""
    n = len(samples) or 1
    return LeverRow(
        lever=lever,
        plane="capture",
        tokens_off=sum(s.tokens_off for s in samples),
        tokens_on=sum(s.tokens_on for s in samples),
        quality_off=1.0,  # uncrushed content always retains the must-keep row
        quality_on=sum(1.0 if s.survived else 0.0 for s in samples) / n,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -k "corpus or crush_case" -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add -f eval/token_corpus.json
git add src/memo/eval_tokens.py tests/test_eval_tokens.py
git commit -m "feat(eval-tokens): P2 capture corpus + crush measurement"
```

*(`eval/` is tracked, but `git add -f` is harmless if a broad ignore applies; drop `-f` if the plain add succeeds.)*

---

### Task 4: Gate metrics + no-regression check

**Files:**
- Modify: `src/memo/eval_tokens.py`
- Test: `tests/test_eval_tokens.py`

**Interfaces:**
- Consumes: `LeverRow` (Task 2).
- Produces: `gate_metrics(rows: list[LeverRow]) -> dict[str, dict[str, float | bool]]`; `@dataclass GateResult(passed: bool, message: str, regressions: list[str])`; `check_gate(rows: list[LeverRow], baseline: dict) -> GateResult`.

**Gate semantics:** `gate_metrics` snapshots each lever's `{saved_frac, quality_delta, passed}`. `check_gate` only guards levers that were **passing** at baseline: it FAILS if such a lever now fails, disappears, or its `saved_frac` shrank beyond tolerance. A brand-new lever cannot fail the gate (nothing to regress against).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_tokens.py  (append)
def test_gate_metrics_snapshots_each_lever():
    rows = [eval_tokens.LeverRow("compact", "recall_output", 100, 80, 1.0, 1.0)]
    m = eval_tokens.gate_metrics(rows)
    assert m["compact"]["passed"] is True
    assert round(m["compact"]["saved_frac"], 3) == 0.2


def test_check_gate_passes_when_no_regression():
    baseline = {"compact": {"saved_frac": 0.2, "quality_delta": 0.0, "passed": True}}
    rows = [eval_tokens.LeverRow("compact", "recall_output", 100, 78, 1.0, 1.0)]  # 22% now
    res = eval_tokens.check_gate(rows, baseline)
    assert res.passed is True


def test_check_gate_fails_when_passing_lever_regresses():
    baseline = {"compact": {"saved_frac": 0.2, "quality_delta": 0.0, "passed": True}}
    rows = [eval_tokens.LeverRow("compact", "recall_output", 100, 95, 1.0, 1.0)]  # 5% now
    res = eval_tokens.check_gate(rows, baseline)
    assert res.passed is False
    assert "compact" in res.message


def test_check_gate_ignores_levers_that_never_passed():
    baseline = {"verbosity": {"saved_frac": -0.1, "quality_delta": 0.0, "passed": False}}
    rows = [eval_tokens.LeverRow("verbosity", "recall_output", 100, 130, 1.0, 1.0)]
    res = eval_tokens.check_gate(rows, baseline)
    assert res.passed is True  # a never-passing lever can't regress
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -k gate -q`
Expected: FAIL — `AttributeError: ... has no attribute 'gate_metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/memo/eval_tokens.py  (append)
def gate_metrics(rows: list[LeverRow]) -> dict[str, dict[str, float | bool]]:
    """Per-lever snapshot the gate tracks: saved_frac, quality_delta, passed."""
    return {
        r.lever: {
            "saved_frac": round(r.saved_frac, 4),
            "quality_delta": round(r.quality_delta, 4),
            "passed": r.passed,
        }
        for r in rows
    }


@dataclass
class GateResult:
    passed: bool
    message: str
    regressions: list[str]


def check_gate(rows: list[LeverRow], baseline: dict, *, tol: float = 1e-9) -> GateResult:
    """FAIL if any lever that was PASSING at baseline regressed (now fails,
    disappeared, or its token saving shrank beyond `tol`)."""
    cur = gate_metrics(rows)
    regressions: list[str] = []
    for lever, base in baseline.items():
        if not isinstance(base, dict) or not base.get("passed"):
            continue  # only guard previously-passing levers
        c = cur.get(lever)
        if c is None:
            regressions.append(f"{lever}: missing from current run")
        elif not c["passed"]:
            regressions.append(f"{lever}: was passing, now FAIL")
        elif float(c["saved_frac"]) < float(base["saved_frac"]) - tol:
            regressions.append(
                f"{lever}: saving dropped {float(base['saved_frac']):.3f}"
                f"→{float(c['saved_frac']):.3f}"
            )
    passed = not regressions
    message = "PASS — no token/quality regression" if passed else "FAIL — " + "; ".join(regressions)
    return GateResult(passed, message, regressions)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -k gate -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/memo/eval_tokens.py tests/test_eval_tokens.py
git commit -m "feat(eval-tokens): gate metrics + no-regression check"
```

---

### Task 5: `run_all` orchestrator (P1 live-search seam + P2 corpus)

**Files:**
- Modify: `src/memo/eval_tokens.py`
- Test: `tests/test_eval_tokens.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_all(*, prompts: list, search: Callable[[str], list], corpus: list[CaptureCase], crush_fn: Callable[[str], tuple[str, str | None]], k: int = 5, body_chars: int = 200, token_budget: int = 400) -> list[LeverRow]`. `search(text)` returns ranked hits (objects with `.id/.title/.tags/.body/.score`); `prompts` are `eval_recall.Prompt`. The CLI supplies real `search`/`crush_fn`; tests supply fakes — so `run_all` needs no MLX.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_tokens.py  (append)
def test_run_all_measures_p1_and_p2_without_mlx(monkeypatch):
    from memo.eval_recall import Prompt

    hits = [_FakeHit(id="aaaaaaaa11", title="Answer", body="the answer body " * 10)]

    def fake_search(text: str) -> list:
        return hits

    def fake_crush(content: str) -> tuple[str, str | None]:
        import json as j
        arr = j.loads(content)
        return j.dumps(arr[:10]), "h"  # position-only: drops late rows

    corpus = [
        eval_tokens.CaptureCase(
            "late", must_keep_index=11,
            rows=[{"i": i} for i in range(11)] + [{"i": 11, "answer": True}],
        )
    ]
    rows = eval_tokens.run_all(
        prompts=[Prompt("q", expect_ids=["aaaaaaaa11"])],
        search=fake_search,
        corpus=corpus,
        crush_fn=fake_crush,
    )
    planes = {r.plane for r in rows}
    assert planes == {"recall_output", "capture"}
    # The crusher lever dropped the answer -> capture lever FAILs quality.
    cap = next(r for r in rows if r.plane == "capture")
    assert cap.passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -k run_all -q`
Expected: FAIL — `AttributeError: ... has no attribute 'run_all'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/memo/eval_tokens.py  (append)
def run_all(
    *,
    prompts: list[Any],
    search: Callable[[str], list[Any]],
    corpus: list[CaptureCase],
    crush_fn: Callable[[str], tuple[str, str | None]],
    k: int = 5,
    body_chars: int = 200,
    token_budget: int = 400,
) -> list[LeverRow]:
    """Measure every P1 recall lever and the P2 crusher over the given corpus.

    `search`/`crush_fn` are injected so tests run without MLX; the CLI wires the
    real live-index search and the real `maybe_crush_json_capture`.
    """
    rows: list[LeverRow] = []

    # --- P1: recall-output levers ---------------------------------------
    hits_by_prompt = {p.text: (search(p.text) or [])[:k] for p in prompts}
    for lever in RECALL_LEVERS:
        samples: list[P1Sample] = []
        for p in prompts:
            hits = hits_by_prompt[p.text]
            block_off = render_block(hits, {}, body_chars=body_chars, token_budget=token_budget)
            block_on = render_block(
                hits, lever["env"], body_chars=body_chars, token_budget=token_budget
            )
            samples.append(measure_recall_sample(block_off, block_on, p.expect_ids))
        rows.append(aggregate_recall(lever["name"], samples))

    # --- P2: capture crusher --------------------------------------------
    p2 = [measure_crush_case(c, crush_fn) for c in corpus]
    if p2:
        rows.append(aggregate_capture("crusher_L1", p2))
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -k run_all -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the whole module suite**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py -q`
Expected: PASS (all green).

- [ ] **Step 6: Commit**

```bash
git add src/memo/eval_tokens.py tests/test_eval_tokens.py
git commit -m "feat(eval-tokens): run_all orchestrator (P1 seam + P2 corpus)"
```

---

### Task 6: `memo eval tokens` CLI subcommand

**Files:**
- Modify: `src/memo/cli_eval.py` (add subcommand; imports)
- Test: `tests/test_cli_eval_tokens.py`

**Interfaces:**
- Consumes: `eval_tokens.run_all`, `eval_tokens.gate_metrics`, `eval_tokens.check_gate` (Tasks 4–5); `eval_recall.load_labels`; `Config`, `_get_memory`.
- Produces: `memo eval tokens` with `--labels`, `--corpus`, `--k`, `--json`, `--update-baseline`, `--gate`, `--force`. Baseline at `state_dir/eval/token_baseline.json`. Real `search = lambda t: mem.search(t, k=k)`; real `crush_fn` wraps `maybe_crush_json_capture` under `MEMO_CRUSHER_ENABLED=1`.

**Note:** tests patch `eval_tokens.run_all` to return canned rows (no MLX, no live index), exercising baseline write, gate pass, gate fail, and the missing-baseline error — the same surface `memo eval recall` guards.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_eval_tokens.py
import json
from pathlib import Path

from click.testing import CliRunner

from memo import eval_tokens
from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_tokens_update_baseline_then_gate_pass(tmp_path, monkeypatch):
    canned = [eval_tokens.LeverRow("recall_format_compact", "recall_output", 100, 80, 1.0, 1.0)]
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: canned)
    # Skip the live-index/labels wiring the stub doesn't need.
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())

    runner = CliRunner()
    r1 = runner.invoke(cli, ["eval", "tokens", "--update-baseline"], env=_env(tmp_path))
    assert r1.exit_code == 0, r1.output
    baseline = tmp_path / "state" / "eval" / "token_baseline.json"
    assert baseline.exists()
    saved = json.loads(baseline.read_text())
    assert saved["recall_format_compact"]["passed"] is True

    r2 = runner.invoke(cli, ["eval", "tokens", "--gate"], env=_env(tmp_path))
    assert r2.exit_code == 0, r2.output


def test_tokens_gate_fails_on_regression(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())
    good = [eval_tokens.LeverRow("recall_format_compact", "recall_output", 100, 80, 1.0, 1.0)]
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: good)
    runner = CliRunner()
    runner.invoke(cli, ["eval", "tokens", "--update-baseline"], env=_env(tmp_path))

    bad = [eval_tokens.LeverRow("recall_format_compact", "recall_output", 100, 99, 1.0, 1.0)]
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: bad)
    r = runner.invoke(cli, ["eval", "tokens", "--gate"], env=_env(tmp_path))
    assert r.exit_code == 1
    assert "FAIL" in r.output


def test_tokens_gate_without_baseline_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: [])
    r = CliRunner().invoke(cli, ["eval", "tokens", "--gate"], env=_env(tmp_path))
    assert r.exit_code != 0
    assert "baseline" in r.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_cli_eval_tokens.py -q`
Expected: FAIL — `No such command 'tokens'`.

- [ ] **Step 3: Add the subcommand**

Add to `src/memo/cli_eval.py` (after `eval_recall_cmd`, before `eval_baseline_cmd`). Reuse the existing module imports (`json`, `sys`, `Path`, `click`, `console`, `_get_memory`, `Config`, `eval_recall`):

```python
def _tokens_baseline_path(cfg: Config) -> Path:
    return cfg.state_dir / "eval" / "token_baseline.json"


@eval_group.command(name="tokens")
@click.option("--k", type=int, default=5, help="Top-K hits to render per P1 prompt.")
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(exists=True, dir_okay=False),
    default="eval/regression_labels.json",
    help="P1 label set (schema memo.eval_recall.labels.v1).",
)
@click.option(
    "--corpus",
    "corpus_path",
    type=click.Path(exists=True, dir_okay=False),
    default="eval/token_corpus.json",
    help="P2 capture corpus (schema memo.token_corpus.v1).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON.")
@click.option("--update-baseline", is_flag=True, help="Save current per-lever metrics as baseline.")
@click.option("--gate", is_flag=True, help="Exit non-zero if a passing lever regressed vs baseline.")
@click.option("--force", is_flag=True, help="(accepted for parity; runs are never cached).")
def eval_tokens_cmd(
    k: int,
    labels_path: str,
    corpus_path: str,
    as_json: bool,
    update_baseline: bool,
    gate: bool,
    force: bool,
) -> None:
    """Measure each token-economy lever: Δtokens + Δquality, per plane.

    P1 (recall-output): render OFF vs ON under each lever, precision = expect_ids
    surviving into the injected block. P2 (capture): crush the corpus, quality =
    the labeled must-keep row surviving. A lever PASSes iff it cuts >=5% tokens
    AND does not drop quality.

    Gate (local, runs against the live index):
      memo eval tokens --update-baseline
      memo eval tokens --gate
    """
    from memo import eval_tokens

    cfg = Config.from_env()
    labels = eval_recall.load_labels(Path(labels_path))
    corpus = eval_tokens.load_capture_corpus(Path(corpus_path))
    mem = _get_memory(cfg)

    def _search(text: str) -> list:
        return list(mem.search(text, k=k))

    def _crush(content: str) -> tuple[str, str | None]:
        from memo.capture_core import maybe_crush_json_capture

        with eval_tokens.env_pins({"MEMO_CRUSHER_ENABLED": "1"}):
            return maybe_crush_json_capture(content, context="", config=cfg)

    rows = eval_tokens.run_all(
        prompts=labels.prompts, search=_search, corpus=corpus, crush_fn=_crush, k=k
    )
    metrics = eval_tokens.gate_metrics(rows)

    if update_baseline:
        bp = _tokens_baseline_path(cfg)
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]✓[/green] token baseline saved → {bp}")
        return

    if gate:
        bp = _tokens_baseline_path(cfg)
        if not bp.exists():
            raise click.ClickException(
                f"no token gate baseline at {bp} — seed it with "
                "`memo eval tokens --update-baseline`"
            )
        baseline = json.loads(bp.read_text(encoding="utf-8"))
        result = eval_tokens.check_gate(rows, baseline)
        if as_json:
            click.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        else:
            color = "green" if result.passed else "red"
            mark = "✓" if result.passed else "✗"
            console.print(f"[{color}]{mark}[/{color}] token gate: {result.message}")
        sys.exit(0 if result.passed else 1)

    if as_json:
        click.echo(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    for r in rows:
        verdict = "PASS" if r.passed else "FAIL"
        color = "green" if r.passed else "yellow"
        console.print(
            f"[{color}]{verdict}[/{color}] {r.lever} [{r.plane}]  "
            f"saved {r.saved_frac * 100:+.1f}%  Δquality {r.quality_delta:+.2f}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_cli_eval_tokens.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/memo/cli_eval.py tests/test_cli_eval_tokens.py
git commit -m "feat(eval-tokens): memo eval tokens CLI subcommand + gate"
```

---

### Task 7: Replace hardcoded estimate in `memo token-savings`

**Files:**
- Modify: `src/memo/cli_token_savings.py`
- Test: `tests/test_cli_token_savings.py`

**Interfaces:**
- Consumes: the `token_baseline.json` written by Task 6 (`{lever: {saved_frac, quality_delta, passed}}`).
- Produces: `_measured_savings(cfg) -> list[tuple[str, float]]` returning `(lever, saved_frac)` for levers that PASSED; `token_savings_cmd` reports measured deltas and prints a clear "not yet measured — run `memo eval tokens --update-baseline`" line when the baseline is absent, instead of the hardcoded 65%.

**Note:** this removes the fabricated `compact_savings_pct = 65` and the estimate arithmetic (`cli_token_savings.py:48-59`). A lever that never passed the gate reports nothing — honest zero, not an estimate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_token_savings.py
import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_reports_measured_savings_from_baseline(tmp_path):
    eval_dir = tmp_path / "state" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "token_baseline.json").write_text(
        json.dumps(
            {
                "recall_format_compact": {"saved_frac": 0.31, "quality_delta": 0.0, "passed": True},
                "verbosity_steer_L2": {"saved_frac": -0.12, "quality_delta": 0.0, "passed": False},
            }
        )
    )
    r = CliRunner().invoke(cli, ["token-savings"], env=_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "recall_format_compact" in r.output
    assert "31" in r.output  # measured 31% saving surfaced
    assert "verbosity_steer_L2" not in r.output  # a non-passing lever is not claimed


def test_reports_unmeasured_when_no_baseline(tmp_path):
    r = CliRunner().invoke(cli, ["token-savings"], env=_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "memo eval tokens --update-baseline" in r.output
    assert "65%" not in r.output  # the fabricated estimate is gone
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_cli_token_savings.py -q`
Expected: FAIL — both assertions fail (old output has the hardcoded 65% and no baseline read).

- [ ] **Step 3: Rewrite the command body**

Replace the body of `token_savings_cmd` and add the helper in `src/memo/cli_token_savings.py`. Keep the existing module docstring and the `_parse_ts` helper; replace the estimate block (lines ~44–80):

```python
def _measured_savings(cfg: Config) -> list[tuple[str, float]]:
    """PASSED levers and their measured token-saving fraction, from the last
    `memo eval tokens --update-baseline`. Empty when never measured."""
    import json

    path = cfg.state_dir / "eval" / "token_baseline.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[tuple[str, float]] = []
    for lever, m in data.items():
        if isinstance(m, dict) and m.get("passed"):
            out.append((str(lever), float(m.get("saved_frac", 0.0))))
    return sorted(out, key=lambda t: t[1], reverse=True)


@click.command(name="token-savings")
def token_savings_cmd() -> None:
    """Show measured per-lever token savings (from `memo eval tokens`)."""
    cfg = Config.from_env()
    savings = _measured_savings(cfg)

    click.echo("memo token savings (measured)")
    click.echo("")
    if not savings:
        click.echo("  No measured savings yet.")
        click.echo("  Seed the gate:  memo eval tokens --update-baseline")
        click.echo("  Then re-run:    memo token-savings")
        return
    for lever, frac in savings:
        click.echo(f"  {lever:<28} {frac * 100:+.1f}%  (measured, gate-passed)")
    click.echo("")
    click.echo("  Re-measure after any change:  memo eval tokens --gate")
```

Remove the now-unused imports if they become orphaned (e.g. `read_context_cost_log`, `read_recall_log`, `flag_str`, `datetime`/`timedelta`/`UTC`) — only those your rewrite no longer references. Keep `_parse_ts` only if still used; if not, delete it and its imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_cli_token_savings.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/memo/cli_token_savings.py tests/test_cli_token_savings.py
git commit -m "fix(token-savings): report measured deltas, drop hardcoded 65% estimate"
```

---

### Task 8: Full-suite verification + lint/type

**Files:** none (verification only).

- [ ] **Step 1: Run the new tests together**

Run: `uv run --no-sync pytest tests/test_eval_tokens.py tests/test_cli_eval_tokens.py tests/test_cli_token_savings.py -q`
Expected: PASS (all green).

- [ ] **Step 2: Lint + type only the files this plan touched**

Run:
```bash
uv run --no-sync ruff check src/memo/eval_tokens.py src/memo/cli_eval.py src/memo/cli_token_savings.py tests/test_eval_tokens.py tests/test_cli_eval_tokens.py tests/test_cli_token_savings.py
uv run --no-sync ruff format src/memo/eval_tokens.py src/memo/cli_eval.py src/memo/cli_token_savings.py tests/test_eval_tokens.py tests/test_cli_eval_tokens.py tests/test_cli_token_savings.py
uv run --no-sync mypy src/memo/eval_tokens.py src/memo/cli_eval.py src/memo/cli_token_savings.py
```
Expected: no errors. Fix any surfaced issues, re-run.

- [ ] **Step 3: Config-flag sanity (no new flags introduced, but validate clean)**

Run: `uv run --no-sync memo config validate`
Expected: no unknown-flag errors.

- [ ] **Step 4: Commit any lint/format fixups**

```bash
git add src/memo/eval_tokens.py src/memo/cli_eval.py src/memo/cli_token_savings.py tests/
git commit -m "chore(eval-tokens): lint/format/type fixups"
```

---

## Self-Review

**1. Spec coverage** (each spec section → task):
- Measurement gate mirroring `eval recall` → Tasks 4, 6. ✓
- P1 recall-output plane + precision guard → Tasks 2, 2b, 5. ✓
- P2 capture plane + must-keep guard + committed corpus → Task 3. ✓
- Token counting via `_CHARS_PER_TOKEN` → Task 1. ✓
- Baseline under `state_dir`, `--update-baseline`/`--gate`, exit 0/1 → Task 6. ✓
- Honesty fix: drop hardcoded 65%, report measured deltas, 0 when unmeasured → Task 7. ✓
- L4 paradox surfaced (verbosity shows a token cost on P1) → Task 2b (`test_render_block_verbosity_level_appends_steering`) + `verbosity_steer_L2` lever. ✓
- L1 verdict by data (placeholder scorer fails P2 must-keep) → Task 3 + Task 5. ✓
- Non-goals honored: no LLM (all tests MLX-free via injected `search`/`crush_fn`); no live-path wiring of any lever. ✓
- Tests reuse `eval_recall` patterns + `tmp_cfg`/env isolation → Tasks 6, 7. ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N" — every code step carries full code. ✓

**3. Type consistency:** `LeverRow`/`P1Sample`/`P2Sample`/`CaptureCase`/`GateResult` defined in Tasks 2–4, consumed with the same fields/props in Tasks 5–6. `run_all` keyword signature in Task 5 matches the CLI call in Task 6 and the test stubs. `gate_metrics` shape (`{lever: {saved_frac, quality_delta, passed}}`) is written by Task 6 and read identically by Task 4's `check_gate` and Task 7's `_measured_savings`. ✓

**Open follow-ups (out of scope, noted for later):** wire winning levers into the live path (compact format / token_budget if they pass; a real relevance scorer for L1); v2 planes P3 (model-output tokens, needs LLM) and P4 (latency/KV-cache).
</content>
