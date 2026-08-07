# Eval Gate Attribution + Close The Red — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the recall regression gate attribute a failure to *code* or to *corpus drift*, and add a same-corpus two-code comparison so ranking changes become verifiable — plus close the broad-exception failure blocking PR #211.

**Architecture:** The gate today compares `(current code, current corpus)` against a machine-local baseline snapshot taken at `(old code, old corpus)`. The two deltas are confounded, so a corpus-driven drop looks identical to a code regression, and the documented workaround is to reseed the baseline — which destroys the signal. Two changes fix this. First, the baseline records the corpus fingerprint it was measured on, so `check_gate` can say whether the corpus moved and word its remedy accordingly. Second, a new `--against <git-ref>` mode evaluates the *same live corpus* twice — once with the current worktree's code, once with the code at `<ref>` — so the corpus term cancels and the remaining delta is attributable to the diff.

**Tech Stack:** Python 3.13/3.14, click, pytest, git worktrees, existing `memo.eval_recall` / `memo.cli_eval` modules.

## Global Constraints

- Dev commands run as `uv run --no-sync <cmd>` from the repo root.
- `ruff check`, `ruff format --check` and `mypy src/memo` must stay clean; all three are CI gates.
- The gate baseline is machine-local at `cfg.state_dir / "eval" / "recall_baseline.json"` and is never committed.
- The eval result cache at `cfg.state_dir / "eval" / "recall.json"` is **shared across git worktrees on this machine**. Any comparison that runs two checkouts must pass `--no-cache` on both sides, or one side reads the other's numbers.
- Broad `except Exception` sites in `recall_logic.py`, `memory/write_ops.py`, `cli_recall_hook.py` and `store/queries.py` must be listed in `memo.dev_audit.BROAD_EXCEPTION_ALLOWED`, keyed by `(relpath, lexical scope, ordinal within scope)`, with a source comment stating the fail-open contract. Policy: `docs/engineering/exception-policy.md`.
- Existing baseline files lack the new keys. Every new check must be non-enforcing when its key is absent, matching how `avoid_at_k` / `config` already degrade.

## Branch routing

Task 1 lands on `feat/parallel-search-and-graph-recall` (the PR #211 branch — that is where the red is). Tasks 2–6 land on a new branch `fix/eval-gate-attribution` cut from `origin/master`. Do not mix them.

---

### Task 1: Close the broad-exception failure on PR #211

**Files:**
- Modify: `src/memo/recall_logic.py` (the `_apply_graph_compact` helper)
- Modify: `src/memo/dev_audit.py:57-165` (`BROAD_EXCEPTION_ALLOWED`)
- Test: `tests/test_dev_audit.py` (existing, no new test needed — it is the gate)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing other tasks rely on. Independent.

**Context:** commit `a2706251` added two broad catches to `recall_logic.py`. The working tree already narrowed the one inside `_graph_compact_clusters` to `except (_sqlite3.Error, ValueError, TypeError, KeyError):` and moved the other into a new `_apply_graph_compact` helper. One unclassified site remains: `("recall_logic.py", "_apply_graph_compact", 1)`. It is legitimately fail-open — optional recall compaction on the 5s hook hot path — so it gets classified, not narrowed.

- [ ] **Step 1: Run the failing test to see the current state**

Run: `uv run --no-sync pytest tests/test_dev_audit.py -q`
Expected: FAIL on `test_broad_exception_policy_targets_are_classified`, listing `recall_logic.py:...:_apply_graph_compact:1`.

- [ ] **Step 2: Add the classification**

In `src/memo/dev_audit.py`, inside `BROAD_EXCEPTION_ALLOWED`, immediately after the `("recall_logic.py", "_code_ref_lines", 1),` entry:

```python
    # Graph-cluster recall compaction (MEMO_RECALL_GRAPH_COMPACT): optional
    # token-budget work on the hook hot path. Any projection/store failure
    # degrades to the uncompacted relevant/nudge lists and must never break
    # the recall payload or blow the 5s hook budget.
    ("recall_logic.py", "_apply_graph_compact", 1),
```

- [ ] **Step 3: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/test_dev_audit.py -q`
Expected: 4 passed. The reverse-direction assertion (`stale == []`) also passes, confirming the new key resolves to a real `except Exception` site.

- [ ] **Step 4: Run the checks CI runs**

Run: `uv run --no-sync ruff check src/ tests/ && uv run --no-sync mypy src/memo && uv run --no-sync pytest tests/test_recall_hooks.py -q`
Expected: all clean, `tests/test_recall_hooks.py` fully green.

- [ ] **Step 5: Commit**

```bash
git add src/memo/recall_logic.py src/memo/dev_audit.py tests/test_recall_hooks.py
git commit -m "fix(recall): classify the graph-compaction fail-open catch

The graph-compaction commit added two broad except sites to recall_logic.py,
one of the four files the broad-exception policy test guards, turning PR #211
red on test (3.13), test (3.14) and test-int8.

Narrow the _graph_compact_clusters catch to the concrete sqlite/parse errors it
actually handles, and classify the _apply_graph_compact catch, which is
fail-open by contract on the 5s hook hot path."
```

- [ ] **Step 6: Verify the PR goes green**

Run: `git push && gh pr checks 211 --watch`
Expected: `test (3.13)`, `test (3.14)` and `test-int8` all pass.

---

### Task 2: The baseline records the corpus it was measured on

**Files:**
- Modify: `src/memo/eval_recall.py` (add `baseline_payload`, near `full_gate_metrics` at line 1183)
- Modify: `src/memo/cli_eval.py:480-494` (the `update_baseline` branch)
- Test: `tests/test_eval_recall.py`

**Interfaces:**
- Consumes: `eval_recall.full_gate_metrics(rows) -> dict[str, Any]`, `eval_recall.fingerprint_corpus(mem) -> str` (both exist).
- Produces: `eval_recall.baseline_payload(rows, *, k: int, labels_fingerprint: str, corpus_fingerprint: str) -> dict[str, Any]` — the exact dict written to the baseline file. Task 3 reads the `corpus_fingerprint` key it adds.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_recall.py`:

```python
def test_baseline_payload_records_the_corpus_it_was_measured_on() -> None:
    rows = _rows((0.6, 0.1))

    payload = eval_recall.baseline_payload(
        rows,
        k=5,
        labels_fingerprint="labels-abc",
        corpus_fingerprint="corpus-123",
    )

    assert payload["corpus_fingerprint"] == "corpus-123"
    assert payload["labels_fingerprint"] == "labels-abc"
    assert payload["k"] == 5
    # The existing metric contract is unchanged.
    assert payload["precision_at_k"] == 0.6
    assert payload["noise_at_k"] == 0.1
    assert "config" in payload
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --no-sync pytest tests/test_eval_recall.py::test_baseline_payload_records_the_corpus_it_was_measured_on -v`
Expected: FAIL with `AttributeError: module 'memo.eval_recall' has no attribute 'baseline_payload'`.

- [ ] **Step 3: Implement**

In `src/memo/eval_recall.py`, directly after `full_gate_metrics`:

```python
def baseline_payload(
    rows: list[Row],
    *,
    k: int,
    labels_fingerprint: str,
    corpus_fingerprint: str,
) -> dict[str, Any]:
    """The exact dict `--update-baseline` persists.

    Adds ``corpus_fingerprint`` to ``full_gate_metrics`` so :func:`check_gate`
    can tell a code regression from corpus drift. Without it, a drop in
    precision@K is unattributable and the only available remedy is reseeding
    the baseline, which discards the signal it was meant to carry.
    """
    return {
        **full_gate_metrics(rows),
        "k": k,
        "labels_fingerprint": labels_fingerprint,
        "corpus_fingerprint": corpus_fingerprint,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/test_eval_recall.py::test_baseline_payload_records_the_corpus_it_was_measured_on -v`
Expected: PASS.

- [ ] **Step 5: Route the CLI through it**

In `src/memo/cli_eval.py`, replace the two payload lines in the `if update_baseline:` branch:

```python
        metrics = eval_recall.full_gate_metrics(rows)
        payload = {**metrics, "k": k, "labels_fingerprint": labels.fingerprint()}
```

with:

```python
        metrics = eval_recall.full_gate_metrics(rows)
        payload = eval_recall.baseline_payload(
            rows,
            k=k,
            labels_fingerprint=labels.fingerprint(),
            corpus_fingerprint=corpus_fp,
        )
```

`corpus_fp` is already in scope — `cli_eval.py:410` computes it as `eval_recall.fingerprint_corpus(mem)`.

- [ ] **Step 6: Run the eval test module and the checks**

Run: `uv run --no-sync pytest tests/test_eval_recall.py tests/test_cli_eval.py -q && uv run --no-sync mypy src/memo`
Expected: all pass, mypy clean.

- [ ] **Step 7: Commit**

```bash
git add src/memo/eval_recall.py src/memo/cli_eval.py tests/test_eval_recall.py
git commit -m "feat(eval): baseline records the corpus fingerprint it was measured on"
```

---

### Task 3: `check_gate` attributes a failure to code or to corpus drift

**Files:**
- Modify: `src/memo/eval_recall.py:1153-1160` (`GateResult`), `1211-1325` (`check_gate`)
- Test: `tests/test_eval_recall.py`

**Interfaces:**
- Consumes: `baseline_payload`'s `corpus_fingerprint` key from Task 2.
- Produces: `check_gate(rows, baseline, *, labels_fingerprint: str = "", k: int | None = None, corpus_fingerprint: str = "", tol: float = ...) -> GateResult`, where `GateResult` gains `corpus_changed: bool`. Task 4 reads `result.corpus_changed`.

**Design note:** a confounded failure still fails. Blocking was never the defect — the message was, because it sent the reader to `--update-baseline`, which resets the very number being defended. A confounded failure now names the two-run comparison from Task 5 as the first remedy and re-baselining as the second.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_recall.py`:

```python
def test_check_gate_blames_code_when_the_corpus_did_not_move() -> None:
    rows = _rows((0.5, 0.1))

    res = eval_recall.check_gate(
        rows,
        {"precision_at_k": 0.6, "noise_at_k": 0.1, "corpus_fingerprint": "corpus-123"},
        corpus_fingerprint="corpus-123",
    )

    assert not res.passed
    assert res.corpus_changed is False
    assert "code" in res.message
    assert "precision@k" in res.message


def test_check_gate_flags_a_drop_as_confounded_when_the_corpus_moved() -> None:
    rows = _rows((0.5, 0.1))

    res = eval_recall.check_gate(
        rows,
        {"precision_at_k": 0.6, "noise_at_k": 0.1, "corpus_fingerprint": "corpus-123"},
        corpus_fingerprint="corpus-456",
    )

    assert not res.passed
    assert res.corpus_changed is True
    assert "confounded" in res.message
    assert "--against" in res.message


def test_check_gate_is_non_enforcing_for_a_baseline_without_a_corpus_fingerprint() -> None:
    rows = _rows((0.5, 0.1))

    res = eval_recall.check_gate(
        rows,
        {"precision_at_k": 0.6, "noise_at_k": 0.1},
        corpus_fingerprint="corpus-456",
    )

    assert not res.passed
    assert res.corpus_changed is False
    assert "confounded" not in res.message


def test_check_gate_passing_run_reports_no_corpus_change() -> None:
    rows = _rows((0.8, 0.0))

    res = eval_recall.check_gate(
        rows,
        {"precision_at_k": 0.6, "noise_at_k": 0.1, "corpus_fingerprint": "corpus-123"},
        corpus_fingerprint="corpus-456",
    )

    assert res.passed
    assert res.corpus_changed is True
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --no-sync pytest tests/test_eval_recall.py -k check_gate -v`
Expected: the four new tests FAIL with `TypeError: check_gate() got an unexpected keyword argument 'corpus_fingerprint'`; the pre-existing `check_gate` tests still pass.

- [ ] **Step 3: Add the field to `GateResult`**

In `src/memo/eval_recall.py`, extend the dataclass:

```python
@dataclass
class GateResult:
    passed: bool
    message: str
    precision_at_k: float
    noise_at_k: float
    baseline_precision: float
    baseline_noise: float
    corpus_changed: bool = False
```

The default keeps every existing construction site valid.

- [ ] **Step 4: Accept the fingerprint and compute the verdict**

In `check_gate`'s signature add `corpus_fingerprint: str = "",` after the existing `k` parameter. Immediately after the `current_best = best_row(rows)` line, add:

```python
    baseline_corpus = str(baseline.get("corpus_fingerprint") or "")
    corpus_changed = bool(
        baseline_corpus and corpus_fingerprint and baseline_corpus != corpus_fingerprint
    )
```

Every `return GateResult(...)` inside the identity guards keeps its current arguments — those failures are about the label set or K, not about metrics, so attribution does not apply.

- [ ] **Step 5: Word the two failure modes differently**

Replace the final `message = f"FAIL [{config_note}] — " + "; ".join(parts)` line and the `return` that follows it with:

```python
        if corpus_changed:
            attribution = (
                f"FAIL [confounded · {config_note}] — "
                + "; ".join(parts)
                + f". The corpus also changed since the baseline "
                f"({baseline_corpus} -> {corpus_fingerprint}), so this drop is not "
                "attributable to the diff. Isolate the code delta with "
                "`memo eval recall --against origin/master`; refresh the baseline "
                "with --update-baseline only once the drift is confirmed expected."
            )
        else:
            attribution = (
                f"FAIL [code · {config_note}] — "
                + "; ".join(parts)
                + ". The corpus is unchanged since the baseline, so this drop is "
                "attributable to the diff."
            )
        message = attribution
    return GateResult(
        passed,
        message,
        gated.precision_at_k,
        gated.noise_at_k,
        bp,
        bn,
        corpus_changed,
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_eval_recall.py -k check_gate -v`
Expected: all PASS, including the pre-existing gate tests.

- [ ] **Step 7: Commit**

```bash
git add src/memo/eval_recall.py tests/test_eval_recall.py
git commit -m "feat(eval): gate attributes a failure to code or to corpus drift

A drop in precision@K used to be unattributable: the baseline snapshot pinned
old code AND an old corpus, so drift and regression were indistinguishable and
the only offered remedy was --update-baseline, which discards the signal.

The gate now compares the recorded corpus fingerprint against the live one and
words the failure accordingly. Confounded failures still block; they just point
at the two-run comparison instead of at reseeding."
```

---

### Task 4: The CLI and the pre-push hook carry the attribution through

**Files:**
- Modify: `src/memo/cli_eval.py:497-519` (the `if gate:` branch)
- Modify: `.git/hooks/pre-push` (machine-local, not committed — edit in place)
- Test: `tests/test_cli_eval.py`

**Interfaces:**
- Consumes: `GateResult.corpus_changed` from Task 3.
- Produces: `cli_eval._run_gate(rows, cfg, *, labels_fingerprint: str, k: int, corpus_fingerprint: str) -> GateResult` — loads the machine-local baseline and delegates to `check_gate`. Extracted so the wiring is testable without running a real evaluation.

**Why extract a helper:** the `if gate:` branch sits after a full evaluation that needs the live index, so a CliRunner test of the whole command would be slow and machine-dependent. Pulling the baseline-load-and-check into a function makes the one thing this task changes — that `corpus_fingerprint` reaches `check_gate` — directly testable, and shrinks `eval_recall_cmd`, which is already long.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_eval.py`:

```python
def test_run_gate_passes_the_live_corpus_fingerprint_to_check_gate(tmp_path, monkeypatch) -> None:
    import json as _json

    from memo import cli_eval, eval_recall

    baseline_dir = tmp_path / "eval"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "recall_baseline.json").write_text(
        _json.dumps(
            {
                "precision_at_k": 0.6,
                "noise_at_k": 0.1,
                "corpus_fingerprint": "corpus-OLD",
                "config": "A",
            }
        ),
        encoding="utf-8",
    )

    class _Cfg:
        state_dir = tmp_path

    captured: dict[str, object] = {}
    real_check_gate = eval_recall.check_gate

    def _spy(rows, baseline, **kwargs):
        captured.update(kwargs)
        return real_check_gate(rows, baseline, **kwargs)

    monkeypatch.setattr(eval_recall, "check_gate", _spy)

    rows = [
        eval_recall.Row(**{**_ROW_TEMPLATE, "config": "A", "precision_at_k": 0.5, "noise_at_k": 0.1})
    ]
    result = cli_eval._run_gate(
        rows, _Cfg(), labels_fingerprint="labels-1", k=5, corpus_fingerprint="corpus-NEW"
    )

    assert captured["corpus_fingerprint"] == "corpus-NEW"
    assert result.corpus_changed is True
    assert not result.passed
```

`_ROW_TEMPLATE` is a dict of the `Row` dataclass's remaining required fields with zero values. Build it once at the top of the test module by reading the dataclass rather than hand-listing fields, so a new `Row` field cannot silently break it:

```python
import dataclasses

from memo import eval_recall

# `eval_recall` uses `from __future__ import annotations`, so dataclasses
# reports field types as strings ("str", "float") — not as the types
# themselves. Comparing against `str` alone would silently match nothing.
_ROW_TEMPLATE = {
    f.name: ("" if f.type in (str, "str") else 0.0)
    for f in dataclasses.fields(eval_recall.Row)
}
```

`Row` has 20 fields, of which exactly one (`config`) is `str`; the rest are floats. Verify that still holds when you run it — if a later field breaks the assumption, the template is where it shows up.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --no-sync pytest tests/test_cli_eval.py::test_run_gate_passes_the_live_corpus_fingerprint_to_check_gate -v`
Expected: FAIL with `AttributeError: module 'memo.cli_eval' has no attribute '_run_gate'`.

- [ ] **Step 3: Extract the helper and pass the fingerprint through**

In `src/memo/cli_eval.py`, add above `eval_recall_cmd`:

```python
def _run_gate(
    rows: list[Any],
    cfg: Config,
    *,
    labels_fingerprint: str,
    k: int,
    corpus_fingerprint: str,
) -> eval_recall.GateResult:
    """Load the machine-local baseline and check `rows` against it.

    `corpus_fingerprint` is what lets the result distinguish a code regression
    from corpus drift; see `eval_recall.check_gate`.
    """
    bp = _baseline_path(cfg)
    if not bp.exists():
        raise click.ClickException(
            f"no gate baseline at {bp} — seed it once with "
            "`memo eval recall --labels <set> --update-baseline`"
        )
    try:
        baseline = json.loads(bp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"unreadable baseline {bp}: {exc}") from exc
    return eval_recall.check_gate(
        rows,
        baseline,
        labels_fingerprint=labels_fingerprint,
        k=k,
        corpus_fingerprint=corpus_fingerprint,
    )
```

Then replace the body of the `if gate:` branch — everything from `bp = _baseline_path(cfg)` through the `result = eval_recall.check_gate(...)` call — with:

```python
        result = _run_gate(
            rows,
            cfg,
            labels_fingerprint=labels.fingerprint(),
            k=k,
            corpus_fingerprint=corpus_fp,
        )
```

Leave the `if as_json: ... else: ... sys.exit(...)` lines that follow unchanged. `result.__dict__` is what `--json` echoes, so `corpus_changed` appears without further work.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/test_cli_eval.py -q`
Expected: the new test PASSES and every pre-existing test in the module still passes.

- [ ] **Step 5: Update the pre-push hook's remedy text**

The hook lives at `.git/hooks/pre-push` and is machine-local (not committed). Replace its failure branch:

```zsh
    echo "✗ memo eval gate FAILED — retrieval regression vs saved baseline."
    echo "  Inspect: memo eval recall --labels eval/regression_labels.json --k 5 --force --profile pre-push"
    echo "  Bypass (emergencies): git push --no-verify"
```

with:

```zsh
    echo "✗ memo eval gate FAILED — see the attribution in the message above."
    echo "  [code]       the corpus is unchanged: the diff caused this. Fix it."
    echo "  [confounded] the corpus also moved. Isolate the code delta first:"
    echo "     memo eval recall --labels eval/regression_labels.json --k 5 --against origin/master"
    echo "  Reseed (--update-baseline) ONLY after confirming the drift is expected."
    echo "  Bypass (emergencies): git push --no-verify"
```

- [ ] **Step 6: Commit**

```bash
git add src/memo/cli_eval.py tests/test_cli_eval.py
git commit -m "feat(eval): surface gate attribution in CLI output and hook remedy"
```

Note: `.git/hooks/pre-push` is untracked by design and is not part of this commit.

---

### Task 5: `memo eval recall --against <ref>` — same corpus, two code revisions

**Files:**
- Create: `src/memo/eval_against.py`
- Create: `src/memo/__main__.py`
- Modify: `src/memo/cli_eval.py` (add the `--against` option and its branch)
- Test: `tests/test_eval_against.py`

**Interfaces:**
- Consumes: `eval_recall.Row` (existing dataclass with `config`, `precision_at_k`, `noise_at_k`, `avoid_at_k`, `avoid_leak_at_k`), `eval_recall.best_row(rows)`.
- Produces:
  - `eval_against.build_eval_argv(*, labels_path: str | None, k: int, profile: str | None, configs: tuple[str, ...]) -> list[str]`
  - `eval_against.AgainstResult` — dataclass with `passed: bool`, `message: str`, `config: str`, `current_precision: float`, `ref_precision: float`, `current_noise: float`, `ref_noise: float`
  - `eval_against.compare_rows(current: list[dict], ref: list[dict], *, tol: float = 0.01) -> AgainstResult`
  - `eval_against.run_against(ref: str, *, repo_root: Path, argv: list[str], runner: Callable[[list[str], dict[str, str], Path], str] | None = None) -> list[dict]`

**Why this exists:** it is the only construct in which a ranking change is verifiable. Both runs hit the same live index at the same instant, so the corpus term cancels and the delta is the diff.

**Two traps this task must avoid.** The eval result cache lives in `state_dir` and is shared across worktrees, so both runs pass `--no-cache` — otherwise the second run reads the first's numbers and the comparison is always a tie. And the installed `memo` binary is the global uv tool, so invoking `memo` inside a worktree runs the *installed* code, not the worktree's; the ref-side run must go through `PYTHONPATH=<worktree>/src <sys.executable> -m memo`, which is why `__main__.py` is created here.

- [ ] **Step 1: Write the failing tests for the pure functions**

Create `tests/test_eval_against.py`:

```python
from __future__ import annotations

from pathlib import Path

from memo import eval_against


def _row(config: str, precision: float, noise: float) -> dict[str, float | str]:
    return {
        "config": config,
        "precision_at_k": precision,
        "noise_at_k": noise,
        "avoid_at_k": 1.0,
        "avoid_leak_at_k": 0.0,
    }


def test_build_eval_argv_always_disables_the_shared_cache() -> None:
    argv = eval_against.build_eval_argv(
        labels_path="eval/regression_labels.json", k=5, profile="pre-push", configs=()
    )

    # The eval cache is keyed on corpus+labels+configs+k and lives in state_dir,
    # shared across worktrees — without --no-cache the ref run reads the
    # current run's numbers and every comparison is a false tie.
    assert "--no-cache" in argv
    assert "--json" in argv
    assert argv[:3] == ["eval", "recall", "--k"]
    assert "--profile" in argv and "pre-push" in argv


def test_build_eval_argv_passes_explicit_configs_through() -> None:
    argv = eval_against.build_eval_argv(labels_path=None, k=3, profile=None, configs=("A", "B"))

    assert argv.count("--config") == 2
    assert "A" in argv and "B" in argv
    assert "--profile" not in argv


def test_compare_rows_passes_when_the_diff_holds_precision() -> None:
    result = eval_against.compare_rows([_row("A", 0.70, 0.10)], [_row("A", 0.70, 0.10)])

    assert result.passed
    assert result.config == "A"
    assert "PASS" in result.message


def test_compare_rows_fails_when_the_diff_drops_precision() -> None:
    result = eval_against.compare_rows([_row("A", 0.55, 0.10)], [_row("A", 0.70, 0.10)])

    assert not result.passed
    assert "precision@k" in result.message
    assert result.current_precision == 0.55
    assert result.ref_precision == 0.70


def test_compare_rows_fails_when_the_diff_raises_noise() -> None:
    result = eval_against.compare_rows([_row("A", 0.70, 0.25)], [_row("A", 0.70, 0.10)])

    assert not result.passed
    assert "noise@k" in result.message


def test_compare_rows_pins_the_comparison_to_the_same_config() -> None:
    current = [_row("A", 0.55, 0.10), _row("B", 0.90, 0.05)]
    ref = [_row("A", 0.70, 0.10)]

    result = eval_against.compare_rows(current, ref)

    # B winning the current run must not mask A's regression.
    assert not result.passed
    assert result.config == "A"


def test_compare_rows_fails_loudly_when_the_ref_run_produced_nothing() -> None:
    result = eval_against.compare_rows([_row("A", 0.70, 0.10)], [])

    assert not result.passed
    assert "no rows" in result.message


def test_run_against_puts_the_worktree_src_first_on_pythonpath(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_runner(argv: list[str], env: dict[str, str], cwd: Path) -> str:
        seen["argv"] = argv
        seen["env"] = env
        seen["cwd"] = cwd
        return '{"rows": [{"config": "A", "precision_at_k": 0.7, "noise_at_k": 0.1}]}'

    monkeypatch.setattr(eval_against, "_add_worktree", lambda ref, root, dest: dest)
    monkeypatch.setattr(eval_against, "_remove_worktree", lambda root, dest: None)
    monkeypatch.setattr(eval_against, "_worktree_dest", lambda root: tmp_path / "wt")

    rows = eval_against.run_against(
        "origin/master",
        repo_root=tmp_path,
        argv=["eval", "recall", "--json", "--no-cache"],
        runner=_fake_runner,
    )

    assert rows == [{"config": "A", "precision_at_k": 0.7, "noise_at_k": 0.1}]
    env = seen["env"]
    assert isinstance(env, dict)
    assert env["PYTHONPATH"].startswith(str(tmp_path / "wt" / "src"))
    argv = seen["argv"]
    assert isinstance(argv, list)
    assert argv[1:3] == ["-m", "memo"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --no-sync pytest tests/test_eval_against.py -v`
Expected: every test FAILs with `ModuleNotFoundError: No module named 'memo.eval_against'`.

- [ ] **Step 3: Create the module entrypoint**

Create `src/memo/__main__.py`:

```python
"""`python -m memo` entrypoint.

Exists so a checkout can be run without installing it: the `--against` eval
comparison invokes a second git worktree's code via
``PYTHONPATH=<worktree>/src python -m memo``. Going through the installed
`memo` console script would run the globally installed uv tool instead, and the
comparison would silently evaluate the same code twice.
"""

from memo.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement the module**

Create `src/memo/eval_against.py`:

```python
"""Same-corpus, two-revision recall comparison.

The saved-baseline gate compares (current code, current corpus) against
(old code, old corpus). The two deltas are confounded, so it cannot approve a
ranking change. This module evaluates the SAME live corpus twice — once with
the working tree's code, once with the code at a git ref — so the corpus term
cancels and the remaining delta is attributable to the diff.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Runner = Callable[[list[str], dict[str, str], Path], str]


@dataclass
class AgainstResult:
    passed: bool
    message: str
    config: str
    current_precision: float
    ref_precision: float
    current_noise: float
    ref_noise: float


def build_eval_argv(
    *,
    labels_path: str | None,
    k: int,
    profile: str | None,
    configs: tuple[str, ...],
) -> list[str]:
    """The `memo eval recall` argv both sides of the comparison run.

    ``--no-cache`` is not optional: the result cache is keyed on
    corpus+labels+configs+k and lives in ``state_dir``, which every worktree on
    this machine shares. Cached, the ref run would read the current run's
    numbers and every comparison would tie.
    """
    argv = ["eval", "recall", "--k", str(k), "--json", "--no-cache"]
    if labels_path:
        argv += ["--labels", labels_path]
    if profile:
        argv += ["--profile", profile]
    for name in configs:
        argv += ["--config", name]
    return argv


def _by_config(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("config") or ""): r for r in rows}


def compare_rows(
    current: list[dict[str, Any]],
    ref: list[dict[str, Any]],
    *,
    tol: float = 0.01,
) -> AgainstResult:
    """Compare the two runs on the config the REF run ranked best.

    Pinning to the ref's best config stops a different config winning the
    current run from masking a regression in the one that was shipping.
    """
    if not ref:
        return AgainstResult(False, "FAIL — the ref run produced no rows", "", 0.0, 0.0, 0.0, 0.0)
    if not current:
        return AgainstResult(
            False, "FAIL — the current run produced no rows", "", 0.0, 0.0, 0.0, 0.0
        )

    ref_best = max(ref, key=lambda r: float(r.get("precision_at_k") or 0.0))
    config = str(ref_best.get("config") or "")
    cur = _by_config(current).get(config)
    if cur is None:
        ran = ", ".join(sorted(_by_config(current)))
        return AgainstResult(
            False,
            f"FAIL — config {config!r} was not evaluated in the current run (ran: {ran})",
            config,
            0.0,
            float(ref_best.get("precision_at_k") or 0.0),
            0.0,
            float(ref_best.get("noise_at_k") or 0.0),
        )

    cp = float(cur.get("precision_at_k") or 0.0)
    rp = float(ref_best.get("precision_at_k") or 0.0)
    cn = float(cur.get("noise_at_k") or 0.0)
    rn = float(ref_best.get("noise_at_k") or 0.0)

    parts: list[str] = []
    if cp < rp - tol:
        parts.append(f"precision@k {cp:.3f} < ref {rp:.3f}")
    if cn > rn + tol:
        parts.append(f"noise@k {cn:.3f} > ref {rn:.3f}")

    if parts:
        message = f"FAIL [config {config!r}] — " + "; ".join(parts)
        return AgainstResult(False, message, config, cp, rp, cn, rn)

    message = (
        f"PASS [config {config!r}] — prec@k {cp:.3f} vs ref {rp:.3f}, "
        f"noise@k {cn:.3f} vs ref {rn:.3f} (same corpus, both runs uncached)"
    )
    return AgainstResult(True, message, config, cp, rp, cn, rn)


def _worktree_dest(repo_root: Path) -> Path:
    return repo_root / ".git" / "memo-eval-against"


def _add_worktree(ref: str, repo_root: Path, dest: Path) -> Path:
    subprocess.run(
        ["git", "worktree", "add", "--detach", "--force", str(dest), ref],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return dest


def _remove_worktree(repo_root: Path, dest: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(dest)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _default_runner(argv: list[str], env: dict[str, str], cwd: Path) -> str:
    proc = subprocess.run(argv, cwd=cwd, env=env, check=True, capture_output=True, text=True)
    return proc.stdout


def run_against(
    ref: str,
    *,
    repo_root: Path,
    argv: list[str],
    runner: Runner | None = None,
) -> list[dict[str, Any]]:
    """Evaluate `argv` with the code at `ref`, against the live corpus.

    The ref's code is reached through ``PYTHONPATH=<worktree>/src python -m
    memo``. Invoking the installed `memo` script instead would run the globally
    installed build in both halves of the comparison.
    """
    import os

    run = runner or _default_runner
    dest = _worktree_dest(repo_root)
    _add_worktree(ref, repo_root, dest)
    try:
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        wt_src = str(dest / "src")
        env["PYTHONPATH"] = f"{wt_src}{os.pathsep}{existing}" if existing else wt_src
        raw = run([sys.executable, "-m", "memo", *argv], env, dest)
    finally:
        _remove_worktree(repo_root, dest)
    payload = json.loads(raw)
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    return list(rows or [])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_eval_against.py -v`
Expected: all 8 PASS.

- [ ] **Step 6: Wire the CLI option**

In `src/memo/cli_eval.py`, add after the `--update-baseline` option declaration:

```python
@click.option(
    "--against",
    "against_ref",
    default=None,
    help="Compare this worktree's code against a git ref on the SAME live corpus. "
    "Both runs are uncached, so the corpus term cancels and the delta is the diff. "
    "This — not --gate — is the check a ranking change has to clear.",
)
```

Add `against_ref: str | None,` to `eval_recall_cmd`'s signature, and insert this branch immediately before the existing `if update_baseline:` block:

```python
    if against_ref:
        from memo import eval_against

        repo_root = Path(__file__).resolve().parents[2]
        ref_argv = eval_against.build_eval_argv(
            labels_path=labels_path, k=k, profile=profile, configs=tuple(config_names)
        )
        ref_rows = eval_against.run_against(against_ref, repo_root=repo_root, argv=ref_argv)
        result = eval_against.compare_rows([r.__dict__ for r in rows], ref_rows)
        if as_json:
            click.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        else:
            color = "green" if result.passed else "red"
            mark = "✓" if result.passed else "✗"
            console.print(f"[{color}]{mark}[/{color}] vs {against_ref}: {result.message}")
        sys.exit(0 if result.passed else 1)
```

`Path` is already imported in `cli_eval.py`; `json`, `sys` and `console` are too.

- [ ] **Step 7: Run the full eval test surface and the static checks**

Run: `uv run --no-sync pytest tests/test_eval_against.py tests/test_eval_recall.py tests/test_cli_eval.py -q && uv run --no-sync ruff check src/ tests/ && uv run --no-sync ruff format --check src/ tests/ && uv run --no-sync mypy src/memo`
Expected: all pass, all three static checks clean.

- [ ] **Step 8: Commit**

```bash
git add src/memo/eval_against.py src/memo/__main__.py src/memo/cli_eval.py tests/test_eval_against.py
git commit -m "feat(eval): --against <ref> compares two revisions on one corpus

The saved-baseline gate cannot approve a ranking change: its baseline pins old
code AND an old corpus. --against evaluates the same live index twice, once
with the working tree's code and once with the code at a git ref, so the corpus
term cancels.

Both runs pass --no-cache because the result cache lives in state_dir and is
shared across worktrees. The ref side runs through PYTHONPATH + python -m memo
because the installed console script would run the global uv tool in both
halves."
```

---

### Task 6: Verify the whole gate end to end, on this machine

**Files:** none modified. This task is the acceptance evidence for Phase 0.

**Interfaces:**
- Consumes: everything from Tasks 2–5.
- Produces: the numbers recorded in the spec's verification section.

- [ ] **Step 1: Reseed the baseline from a clean checkout of master**

The baseline must not be seeded from a working tree carrying uncommitted ranking changes.

```bash
cd ~/repos/memo
git worktree add --detach /tmp/memo-baseline origin/master
cd /tmp/memo-baseline
PYTHONPATH=/tmp/memo-baseline/src python -m memo eval recall \
  --labels eval/regression_labels.json --k 5 --profile pre-push --update-baseline
cd ~/repos/memo && git worktree remove --force /tmp/memo-baseline
```

Expected: `✓ baseline saved: config '<name>' · prec@5 <p> / noise@5 <n> → …/eval/recall_baseline.json`

- [ ] **Step 2: Confirm the baseline now records the corpus**

Run: `python -c "import json,pathlib;print(sorted(json.loads(pathlib.Path('/Users/fer/.local/share/memo/eval/recall_baseline.json').read_text())))"`

Expected: the key list includes `corpus_fingerprint`. (`state_dir` on this machine is `/Users/fer/.local/share/memo`; on another machine resolve it with `python -c "from memo.config import Config; print(Config.from_env().state_dir)"`.)

- [ ] **Step 3: Determinism — the same commit twice yields the same verdict**

```bash
memo eval recall --labels eval/regression_labels.json --k 5 --gate --profile pre-push --no-cache
memo eval recall --labels eval/regression_labels.json --k 5 --gate --profile pre-push --no-cache
```
Expected: both print `✓ recall gate: PASS …` and exit 0.

- [ ] **Step 4: A deliberate ranking regression fails, and is blamed on code**

Neutralize the curatorial boost for one run. `MEMO_RETRIEVAL_BOOST` is a bool defaulting to `True`, read at `search_ops.py:1003`:

```bash
MEMO_RETRIEVAL_BOOST=0 memo eval recall \
  --labels eval/regression_labels.json --k 5 --gate --profile pre-push --no-cache
```

Expected: `✗ recall gate: FAIL [code · config …]` — the `[code]` tag is the assertion; the corpus did not move between this run and step 3's.

Note the direction: this run is the same *code* with a flag off, not a different commit, so it exercises the attribution logic rather than the `--against` path. If it unexpectedly PASSES, that is itself a finding worth recording — it means the boost is not load-bearing for the label set, which bears directly on Phase 2a.

- [ ] **Step 5: The two-run comparison agrees with itself on a no-op diff**

```bash
memo eval recall --labels eval/regression_labels.json --k 5 --profile pre-push --against HEAD
```
Expected: `✓ vs HEAD: PASS [config …] — prec@k X vs ref X` with identical numbers on both sides. Any difference here means the cache leaked between the runs and `--no-cache` is not reaching one of them — stop and fix that before trusting Phase 2.

- [ ] **Step 6: Record the evidence in the spec**

Append the four measured outcomes (baseline numbers, the two determinism runs, the deliberate-regression message, the no-op `--against` result) to `docs/SPECS/2026-08-07-repair-program-design.md` under a new `## Phase 0 verification (2026-08-XX)` section, then commit:

```bash
git add docs/SPECS/2026-08-07-repair-program-design.md
git commit -m "docs: record Phase 0 gate verification evidence"
```

---

## What this plan does NOT cover

Phases 2–5 of `2026-08-07-repair-program-design.md` get their own plans:

- **Phase 2 (retrieval)** — depends on Task 5 existing, which is the whole reason Phase 0 comes first. Plan it after Task 6's evidence is recorded.
- **Phase 3 (latency)** — independent; can be planned and executed in parallel with Phase 2.
- **Phase 4 (instrumentation)** — independent of both.
- **Phase 5 (reduction)** — must be last; its central question is answered by the harness this plan repairs.
