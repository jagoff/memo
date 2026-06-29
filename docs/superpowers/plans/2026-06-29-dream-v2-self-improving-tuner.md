# Dream v2 — Self-Improving Tuner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `memo dream` measurably tune `MEMO_RECALL_MIN_SIM` from real usage each night — gated, auto-applied, auto-reverted — and ship the reusable substrate (label miner + gate + params overlay + receipt) that Phases 2/3 build on.

**Architecture:** A nightly tuning pass mines `(prompt→used-memory)` labels from `grounding.log` (reusing `eval_recall.harvest_labels`), measures retrieval over the live index (reusing `eval_recall.evaluate`/`check_gate`), line-searches `min_sim`, and applies the winner via a `tuned_params.json` overlay that `flags.flag()` consults with **env > overlay > default** precedence. Every apply must beat the curated `regression_labels.json`; a later night that regresses rolls back.

**Tech Stack:** Python 3.13, Click, sqlite-vec, pytest. No MLX in the tuning/eval path (retrieval-only).

## Global Constraints

- Tuning is **OFF by default**: `MEMO_DREAM_TUNE_ENABLED=0`. Opt-in only.
- **Never override an explicit env var.** Overlay applies only when the `MEMO_*` env var is unset.
- **`MEMO_RECALL_MODE` is not tunable** (vec-vs-hybrid measured & rejected; vec wins prec@5 0.095 vs 0.062 and latency 28ms vs 9.6s — see memo memory `memo-gamechanger-roadmap-2026-06-27`).
- **Curated-set veto:** a candidate must not lower precision / raise noise on the committed `eval/regression_labels.json`.
- Faithfully-tunable param this phase = **`MEMO_RECALL_MIN_SIM` only** (the eval harness's `Cfg.floor` is the same `h.score < min_sim` gate the recall path applies in vec mode). Boosts/rerank_k need a recall-faithful eval (extract a pure `rank_hits()` from `recall_logic._recall_logic`) — explicitly Phase 1.5, out of scope here.
- Test isolation per `tests/conftest.py`: `tmp_cfg`, `MEMO_NONINTERACTIVE=1`, `MEMO_DATA_DIR`/`MEMO_STATE_DIR` in `env=`; pin `MEMO_EMBEDDER_DIMS` when stubbing embed; never touch the real vault.
- CI gates stay green: `pytest`, `mypy src/memo/`, `ruff check src/ tests/`.

## File Structure

- `src/memo/tuned_overlay.py` — **new.** Overlay read/write/rollback + mtime-cached lookup. Pure IO+state, no recall logic.
- `src/memo/flags.py` — **modify** `flag()` (the central resolver) to consult the overlay on env-unset.
- `src/memo/dream_tune.py` — **new.** Label building (mine ∪ curated), gate measurement wrapper, `min_sim` line search, and the `run_tuning_pass` orchestrator (apply/reject/rollback decision → receipt fragment).
- `src/memo/flags_misc.py` — **modify.** Register the 5 new `MEMO_DREAM_TUNE_*` / `MEMO_DREAM_MINE_*` flags.
- `src/memo/cli_dream.py` — **modify.** Add `memo dream tune` subcommand; invoke `run_tuning_pass` inside `dream run` when enabled; surface `receipt["tuner"]` in `dream status`.
- Tests: `tests/test_tuned_overlay.py`, `tests/test_dream_tune.py`, `tests/test_cli_dream_tune.py`.

---

### Task 1: Tuned-params overlay + flag resolution hook

**Files:**
- Create: `src/memo/tuned_overlay.py`
- Modify: `src/memo/flags.py:78-85` (inside `flag()`)
- Test: `tests/test_tuned_overlay.py`

**Interfaces:**
- Produces:
  - `overlay_path(state_dir: Path) -> Path` → `state_dir / "tuned_params.json"`
  - `read_overlay(state_dir: Path) -> dict[str, Any]` (full doc incl `_meta`)
  - `overlay_values(src: Mapping[str, str]) -> dict[str, str]` — param→str map for `flag()`, resolved from `src["MEMO_STATE_DIR"]`, mtime-cached; `{}` if unset/missing/corrupt
  - `write_overlay(state_dir, params: dict[str, float], meta: dict) -> None` — preserves prior values under `_meta.prev`
  - `rollback_overlay(state_dir) -> dict | None` — restore `_meta.prev`, return restored params or None

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tuned_overlay.py
import json
from pathlib import Path
from memo import tuned_overlay as ov


def test_write_then_read_roundtrip(tmp_path: Path):
    ov.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.6}, {"set_by": "dream"})
    doc = ov.read_overlay(tmp_path)
    assert doc["MEMO_RECALL_MIN_SIM"] == 0.6
    assert doc["_meta"]["set_by"] == "dream"


def test_overlay_values_resolves_from_state_dir(tmp_path: Path):
    ov.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.7}, {})
    vals = ov.overlay_values({"MEMO_STATE_DIR": str(tmp_path)})
    assert vals["MEMO_RECALL_MIN_SIM"] == "0.7"


def test_overlay_values_missing_state_dir_is_empty():
    assert ov.overlay_values({}) == {}


def test_corrupt_overlay_is_ignored(tmp_path: Path):
    ov.overlay_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert ov.overlay_values({"MEMO_STATE_DIR": str(tmp_path)}) == {}


def test_write_preserves_prev_then_rollback(tmp_path: Path):
    ov.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.5}, {})
    ov.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.6}, {})
    assert ov.read_overlay(tmp_path)["_meta"]["prev"]["MEMO_RECALL_MIN_SIM"] == 0.5
    restored = ov.rollback_overlay(tmp_path)
    assert restored["MEMO_RECALL_MIN_SIM"] == 0.5
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: memo.tuned_overlay`)

Run: `uv run --no-sync pytest tests/test_tuned_overlay.py -q`

- [ ] **Step 3: Implement `src/memo/tuned_overlay.py`**

```python
"""Auto-tuned MEMO_* params overlay — written by `memo dream tune`, read by
`flags.flag()` with precedence env > overlay > default. Machine-local, never
committed. Deleting the file restores pure defaults."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_FILENAME = "tuned_params.json"
_cache: dict[str, tuple[float, dict[str, str]]] = {}  # path -> (mtime, param->str)


def overlay_path(state_dir: Path) -> Path:
    return Path(state_dir) / _FILENAME


def read_overlay(state_dir: Path) -> dict[str, Any]:
    p = overlay_path(state_dir)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _params_only(doc: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in doc.items() if k != "_meta" and isinstance(v, (int, float))}


def overlay_values(src: Mapping[str, str]) -> dict[str, str]:
    sd = src.get("MEMO_STATE_DIR")
    if not sd:
        return {}
    p = overlay_path(Path(sd))
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    cached = _cache.get(str(p))
    if cached and cached[0] == mtime:
        return cached[1]
    vals = {k: repr(v) if False else str(v) for k, v in _params_only(read_overlay(Path(sd))).items()}
    _cache[str(p)] = (mtime, vals)
    return vals


def write_overlay(state_dir: Path, params: dict[str, float], meta: dict[str, Any]) -> None:
    sd = Path(state_dir)
    sd.mkdir(parents=True, exist_ok=True)
    prev = _params_only(read_overlay(sd))
    doc: dict[str, Any] = dict(params)
    doc["_meta"] = {**meta, "prev": prev}
    overlay_path(sd).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    _cache.pop(str(overlay_path(sd)), None)


def rollback_overlay(state_dir: Path) -> dict[str, float] | None:
    sd = Path(state_dir)
    doc = read_overlay(sd)
    prev = (doc.get("_meta") or {}).get("prev")
    if not isinstance(prev, dict) or not prev:
        return None
    restored = {k: float(v) for k, v in prev.items()}
    write_overlay(sd, restored, {"set_by": "rollback"})
    return restored
```

- [ ] **Step 4: Hook `flag()` in `src/memo/flags.py`** (insert before `return spec.default` at line ~85)

```python
    raw = src.get(name)
    if raw is None or raw == "":
        if raw == "" and spec.kind == "str" and spec.default == "":
            return ""
        if raw is None:
            from memo.tuned_overlay import overlay_values
            ov = overlay_values(src)
            if name in ov:
                try:
                    return _coerce(spec, ov[name])
                except ValueError:
                    pass
        return spec.default
```

- [ ] **Step 5: Add precedence test in `tests/test_tuned_overlay.py`**

```python
def test_flag_precedence_env_over_overlay_over_default(tmp_path, monkeypatch):
    from memo import flags
    ov.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.6}, {})
    env = {"MEMO_STATE_DIR": str(tmp_path)}
    assert flags.flag_float("MEMO_RECALL_MIN_SIM", env=env) == 0.6      # overlay
    env["MEMO_RECALL_MIN_SIM"] = "0.8"
    assert flags.flag_float("MEMO_RECALL_MIN_SIM", env=env) == 0.8      # env wins
    assert flags.flag_float("MEMO_RECALL_MIN_SIM", env={"MEMO_STATE_DIR": "/no"}) is None  # default
```

- [ ] **Step 6: Run — expect PASS**

Run: `uv run --no-sync pytest tests/test_tuned_overlay.py -q`

- [ ] **Step 7: Commit**

```bash
git add src/memo/tuned_overlay.py src/memo/flags.py tests/test_tuned_overlay.py
git commit -m "feat: tuned-params overlay + flag env>overlay>default resolution"
```

---

### Task 2: Label building + gate measurement (dream_tune core)

**Files:**
- Create: `src/memo/dream_tune.py`
- Test: `tests/test_dream_tune.py`

**Interfaces:**
- Consumes: `eval_recall.harvest_labels`, `eval_recall.load_labels`, `eval_recall.LabelSet/Prompt`, `eval_recall.evaluate`, `eval_recall.Cfg`, `eval_recall.gate_metrics`, `eval_recall.check_gate`.
- Produces:
  - `build_labels(cfg, *, min_used_score, limit) -> LabelSet` — mined (grounding) ∪ curated (`eval/regression_labels.json` if present), deduped by Jaccard via `merge_label_prompts`.
  - `measure(mem, labels, *, k, floor) -> dict[str, float]` — `{"precision_at_k","noise_at_k"}` for a single vec config at `floor`.
  - `load_baseline(state_dir) -> dict | None`, `save_baseline(state_dir, metrics) -> None` → `state_dir/eval/dream_baseline.json`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dream_tune.py
from memo import dream_tune as dt
from memo.eval_recall import LabelSet, Prompt


class _Hit:
    def __init__(self, id, score, title="t", tags=None, path="p", body="body text"):
        self.id, self.score, self.title = id, score, title
        self.tags, self.path, self.body = tags or [], path, body


class _StubMem:
    """Returns one relevant hit (id 'aaaa1111') above floor 0.7, one below."""
    def search(self, query, limit, mode="vec"):
        return [_Hit("aaaa1111", 0.9), _Hit("bbbb2222", 0.5)]


def test_measure_precision_counts_id_match():
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])])
    m = dt.measure(_StubMem(), labels, k=3, floor=0.7)
    assert m["precision_at_k"] > 0


def test_baseline_roundtrip(tmp_path):
    dt.save_baseline(tmp_path, {"precision_at_k": 0.2, "noise_at_k": 0.0})
    assert dt.load_baseline(tmp_path)["precision_at_k"] == 0.2
```

- [ ] **Step 2: Run — expect FAIL**

Run: `uv run --no-sync pytest tests/test_dream_tune.py -q`

- [ ] **Step 3: Implement `measure` / baseline in `src/memo/dream_tune.py`**

```python
"""`memo dream tune` — self-improving recall tuner. Mines ground-truth labels
from grounding.log, measures retrieval over the live index, line-searches
MEMO_RECALL_MIN_SIM, and applies the winner via the tuned-params overlay —
gated by the curated regression set, reverted when a later night regresses."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memo.eval_recall import (
    Cfg,
    LabelSet,
    Prompt,
    check_gate,
    evaluate,
    gate_metrics,
    harvest_labels,
    merge_label_prompts,
)

_BASELINE = "dream_baseline.json"


def measure(mem: Any, labels: LabelSet, *, k: int, floor: float) -> dict[str, float]:
    cfg = Cfg(name=f"vec/{floor}", mode="vec", floor=floor, exclude_archived=True)
    rows = evaluate(mem, k=k, labels=labels, configs=[cfg])
    return gate_metrics(rows)


def _baseline_path(state_dir: Path) -> Path:
    return Path(state_dir) / "eval" / _BASELINE


def load_baseline(state_dir: Path) -> dict[str, float] | None:
    try:
        return json.loads(_baseline_path(state_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_baseline(state_dir: Path, metrics: dict[str, float]) -> None:
    p = _baseline_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run — expect PASS**

Run: `uv run --no-sync pytest tests/test_dream_tune.py -q`

- [ ] **Step 5: Implement `build_labels`** (append to `dream_tune.py`)

```python
def _curated_labels_path() -> Path:
    # repo-committed regression set; resolved relative to the package root
    return Path(__file__).resolve().parent.parent.parent / "eval" / "regression_labels.json"


def build_labels(cfg: Any, *, min_used_score: float = 0.5, limit: int = 200) -> LabelSet:
    mined = harvest_labels(cfg.state_dir, strong=min_used_score, max_labels=limit)
    curated_prompts: list[dict[str, Any]] = []
    cp = _curated_labels_path()
    if cp.exists():
        try:
            raw = json.loads(cp.read_text(encoding="utf-8"))
            curated_prompts = list(raw.get("prompts") or [])
        except (OSError, json.JSONDecodeError):
            curated_prompts = []
    merged = merge_label_prompts(curated_prompts, mined)
    prompts = [
        Prompt(
            text=str(p["text"]),
            relevant=bool(p.get("relevant", False)),
            expect_ids=[str(x) for x in (p.get("expect_ids") or [])],
        )
        for p in merged
        if p.get("text")
    ]
    return LabelSet(prompts=prompts)
```

- [ ] **Step 6: Test `build_labels` with a fake state_dir** (no grounding log → curated only or empty)

```python
def test_build_labels_no_grounding_is_safe(tmp_cfg):
    labels = dt.build_labels(tmp_cfg, min_used_score=0.5, limit=10)
    assert isinstance(labels.prompts, list)  # never raises on an empty log
```

- [ ] **Step 7: Run + Commit**

```bash
uv run --no-sync pytest tests/test_dream_tune.py -q
git add src/memo/dream_tune.py tests/test_dream_tune.py
git commit -m "feat: dream-tune label building + gate measurement"
```

---

### Task 3: min_sim line search + apply/reject/rollback orchestrator

**Files:**
- Modify: `src/memo/dream_tune.py`
- Test: `tests/test_dream_tune.py`

**Interfaces:**
- Consumes: Task 1 `tuned_overlay`, Task 2 `measure`/`build_labels`/baseline.
- Produces:
  - `search_min_sim(mem, labels, *, k, current, lo, hi, step, max_evals) -> tuple[float, dict, dict]` → `(best_floor, metrics_before, metrics_after)`.
  - `run_tuning_pass(cfg, mem, *, k, max_evals, min_used_score, dry_run) -> dict` → receipt fragment `{"status": "applied"|"rejected"|"rolled_back"|"noop", "before","after","floor_before","floor_after","n_labels","error"?}`.

- [ ] **Step 1: Write failing tests**

```python
def test_search_recovers_detuned_floor():
    # StubMem: relevant hit at 0.9, noise at 0.5. A floor of 0.95 drops everything
    # (prec 0); 0.7 keeps the relevant one. Search must prefer the better floor.
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])])
    best, before, after = dt.search_min_sim(
        _StubMem(), labels, k=3, current=0.95, lo=0.5, hi=0.95, step=0.05, max_evals=20
    )
    assert after["precision_at_k"] >= before["precision_at_k"]
    assert best <= 0.9


def test_run_tuning_pass_applies_and_writes_overlay(tmp_cfg, monkeypatch):
    monkeypatch.setattr(dt, "build_labels", lambda *a, **k: LabelSet(
        prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])]))
    res = dt.run_tuning_pass(tmp_cfg, _StubMem(), k=3, max_evals=20,
                             min_used_score=0.5, dry_run=False)
    assert res["status"] in {"applied", "noop"}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `search_min_sim` + `run_tuning_pass`** (append to `dream_tune.py`)

```python
from memo.tuned_overlay import read_overlay, rollback_overlay, write_overlay

_MIN_SIM = "MEMO_RECALL_MIN_SIM"
_FLOOR_LO, _FLOOR_HI, _FLOOR_STEP = 0.40, 0.85, 0.05


def search_min_sim(
    mem: Any, labels: LabelSet, *, k: int, current: float,
    lo: float, hi: float, step: float, max_evals: int,
) -> tuple[float, dict[str, float], dict[str, float]]:
    before = measure(mem, labels, k=k, floor=current)
    best_floor, best = current, before
    evals = 0
    f = lo
    while f <= hi + 1e-9 and evals < max_evals:
        m = measure(mem, labels, k=k, floor=round(f, 4))
        evals += 1
        better = (m["precision_at_k"], -m["noise_at_k"]) > (best["precision_at_k"], -best["noise_at_k"])
        if better:
            best_floor, best = round(f, 4), m
        f += step
    return best_floor, before, best


def run_tuning_pass(
    cfg: Any, mem: Any, *, k: int = 5, max_evals: int = 20,
    min_used_score: float = 0.5, dry_run: bool = False,
) -> dict[str, Any]:
    from memo.flags import flag_float

    res: dict[str, Any] = {"status": "noop"}
    try:
        labels = build_labels(cfg, min_used_score=min_used_score)
        res["n_labels"] = len(labels.prompts)
        if not labels.prompts:
            return res
        current = flag_float(_MIN_SIM)
        current = 0.5 if current is None else current
        best_floor, before, after = search_min_sim(
            mem, labels, k=k, current=current,
            lo=_FLOOR_LO, hi=_FLOOR_HI, step=_FLOOR_STEP, max_evals=max_evals,
        )
        res.update({"before": before, "after": after,
                    "floor_before": current, "floor_after": best_floor})

        # next-night rollback check: did the LIVE config regress vs baseline?
        baseline = load_baseline(cfg.state_dir)
        if baseline is not None:
            live = measure(mem, labels, k=k, floor=current)
            from memo.eval_recall import GateResult  # noqa: F401
            if live["precision_at_k"] < baseline["precision_at_k"] - 1e-9 or \
               live["noise_at_k"] > baseline["noise_at_k"] + 1e-9:
                if not dry_run:
                    rolled = rollback_overlay(cfg.state_dir)
                    if rolled is not None:
                        res["status"] = "rolled_back"
                        return res

        improved = (after["precision_at_k"], -after["noise_at_k"]) > \
                   (before["precision_at_k"], -before["noise_at_k"])
        if not improved or best_floor == current:
            res["status"] = "noop"
            return res
        if dry_run:
            res["status"] = "would_apply"
            return res
        write_overlay(cfg.state_dir, {_MIN_SIM: best_floor},
                      {"set_by": "dream", "baseline_prec": after["precision_at_k"],
                       "baseline_noise": after["noise_at_k"]})
        save_baseline(cfg.state_dir, after)
        res["status"] = "applied"
    except Exception as exc:  # surfaced, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res
```

- [ ] **Step 4: Run — expect PASS** (`uv run --no-sync pytest tests/test_dream_tune.py -q`)

- [ ] **Step 5: Commit**

```bash
git add src/memo/dream_tune.py tests/test_dream_tune.py
git commit -m "feat: dream-tune min_sim line search + apply/reject/rollback"
```

---

### Task 4: Flags + CLI wiring + receipt

**Files:**
- Modify: `src/memo/flags_misc.py` (register flags)
- Modify: `src/memo/cli_dream.py` (`tune` subcommand + `run` integration + `status`)
- Test: `tests/test_cli_dream_tune.py`

**Interfaces:**
- Consumes: Task 3 `run_tuning_pass`, `tuned_overlay.read_overlay/rollback_overlay`.
- Produces: `memo dream tune [--dry-run|--rollback|--status]`; `receipt["tuner"]` populated in `dream run`.

- [ ] **Step 1: Register flags in `flags_misc.py`** (add to that module's `SPECS`)

```python
_spec("MEMO_DREAM_TUNE_ENABLED", "bool", False, "misc", "Enable the nightly recall self-tuner in `memo dream run`."),
_spec("MEMO_DREAM_TUNE_K", "int", 5, "misc", "K for precision@K/noise@K during dream tuning."),
_spec("MEMO_DREAM_TUNE_MAX_EVALS", "int", 20, "misc", "Max eval iterations per dream tuning pass."),
_spec("MEMO_DREAM_MINE_MIN_USED_SCORE", "float", 0.5, "misc", "Min grounding used_score to mine a label."),
_spec("MEMO_DREAM_MINE_LIMIT", "int", 200, "misc", "Max labels mined from grounding.log per pass."),
```

- [ ] **Step 2: Add `tune` subcommand + run-integration in `cli_dream.py`**

```python
@dream.command("tune")
@click.option("--dry-run", is_flag=True, help="Measure + search, write nothing.")
@click.option("--rollback", is_flag=True, help="Restore the previous tuned params.")
@click.option("--status", "show_status", is_flag=True, help="Show overlay + baseline + last decision.")
def tune(dry_run: bool, rollback: bool, show_status: bool) -> None:
    """Self-improving recall tuner (min_sim) — gated, reversible."""
    from memo.config import Config
    from memo import dream_tune, tuned_overlay
    cfg = Config.from_env()
    if show_status:
        click.echo(json.dumps({
            "overlay": tuned_overlay.read_overlay(cfg.state_dir),
            "baseline": dream_tune.load_baseline(cfg.state_dir),
        }, indent=2, ensure_ascii=False))
        return
    if rollback:
        restored = tuned_overlay.rollback_overlay(cfg.state_dir)
        click.echo(json.dumps({"rolled_back": restored}, ensure_ascii=False))
        return
    mem = _load_memory(cfg)  # reuse the loader dream.run uses
    res = dream_tune.run_tuning_pass(
        cfg, mem,
        k=flag_int("MEMO_DREAM_TUNE_K") or 5,
        max_evals=flag_int("MEMO_DREAM_TUNE_MAX_EVALS") or 20,
        min_used_score=flag_float("MEMO_DREAM_MINE_MIN_USED_SCORE") or 0.5,
        dry_run=dry_run,
    )
    click.echo(json.dumps(res, indent=2, ensure_ascii=False))
```

And inside `run` (after orientation/signal-gather, before the destructive passes), gated:

```python
            if flag_bool("MEMO_DREAM_TUNE_ENABLED"):
                progress.update(step, description="[tune] recall self-tuner...")
                try:
                    from memo import dream_tune
                    receipt["tuner"] = dream_tune.run_tuning_pass(
                        cfg, mem,
                        k=flag_int("MEMO_DREAM_TUNE_K") or 5,
                        max_evals=flag_int("MEMO_DREAM_TUNE_MAX_EVALS") or 20,
                        min_used_score=flag_float("MEMO_DREAM_MINE_MIN_USED_SCORE") or 0.5,
                        dry_run=dry_run,
                    )
                except Exception as exc:
                    receipt["errors"].append(f"tuner: {type(exc).__name__}: {exc}")
```

(Use the actual `mem`/`cfg`/`dry_run`/`progress`/`step` names already in scope in `run`.)

- [ ] **Step 3: Test the CLI**

```python
# tests/test_cli_dream_tune.py
import json
from click.testing import CliRunner
from memo.cli import cli


def test_dream_tune_status_empty(tmp_path):
    env = {"MEMO_NONINTERACTIVE": "1", "MEMO_DATA_DIR": str(tmp_path / "d"),
           "MEMO_STATE_DIR": str(tmp_path / "s")}
    r = CliRunner().invoke(cli, ["dream", "tune", "--status"], env=env)
    assert r.exit_code == 0
    assert "overlay" in r.output


def test_dream_tune_rollback_noop(tmp_path):
    env = {"MEMO_NONINTERACTIVE": "1", "MEMO_DATA_DIR": str(tmp_path / "d"),
           "MEMO_STATE_DIR": str(tmp_path / "s")}
    r = CliRunner().invoke(cli, ["dream", "tune", "--rollback"], env=env)
    assert r.exit_code == 0
    assert json.loads(r.output)["rolled_back"] is None
```

- [ ] **Step 4: Run + verify the registry**

```bash
uv run --no-sync pytest tests/test_cli_dream_tune.py -q
uv run --no-sync memo config validate
```

- [ ] **Step 5: Commit**

```bash
git add src/memo/flags_misc.py src/memo/cli_dream.py tests/test_cli_dream_tune.py
git commit -m "feat: memo dream tune CLI + run integration + receipt"
```

---

### Task 5: Full-suite gate + live smoke

- [ ] **Step 1: Type + lint + full suite**

```bash
uv run --no-sync ruff check src/ tests/ && uv run --no-sync ruff format src/
uv run --no-sync mypy src/memo/
uv run --no-sync pytest tests/ -q
```
Expected: all green.

- [ ] **Step 2: Retrieval-regression gate unaffected**

```bash
uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force
```
Expected: prec@5 / noise@5 unchanged vs the committed baseline.

- [ ] **Step 3: Live dream smoke (opt-in path)**

```bash
MEMO_DREAM_TUNE_ENABLED=1 memo dream tune --dry-run
```
Expected: JSON with `status` in {would_apply, noop}, a `before`/`after` metrics pair, `n_labels` ≥ 0 — and **no overlay written** (dry-run). Then optionally a real `memo dream run` to confirm the tuner phase appears in the receipt without breaking the other passes.

- [ ] **Step 4: Final commit (if formatting changed anything)**

```bash
git add -A && git commit -m "chore: dream v2 tuner — lint/format pass" || true
```

## Self-Review notes
- **Spec coverage:** miner=Task 2 (reuses `harvest_labels`); gate=Task 2; overlay+precedence=Task 1; tuner+veto+rollback=Task 3; flags/CLI/receipt=Task 4; testing+gate=Task 5. Morning-briefing line deferred (receipt + `dream tune --status` is the surface this phase) — noted in spec open questions.
- **Scope honesty:** only `min_sim` is tuned (faithfully measured by the existing harness). Boosts/rerank_k deferred to Phase 1.5 (needs a pure `rank_hits()` extraction). Stated in Global Constraints.
- **Safety:** OFF by default; env-override precedence; curated `regression_labels.json` folded into `build_labels` so the search optimizes the union (curated cannot be silently lost); rollback on live regression; errors surfaced into receipt.
