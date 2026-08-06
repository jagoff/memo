# Recall Admission by Outcome Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop paying full-body injection for memories the ledger shows are never used, without ever making a memory permanently invisible.

**Architecture:** A nightly dream pass joins `grounding.log` to per-memory surfaced/grounded counts and writes a smoothed utility prior. `rank_hits` — already pure and already shared by the daemon and the eval harness — gains one stage that applies the prior as a bounded demote multiplier, one exploration slot for cold memories, and a marginal floor that demotes weak tail hits from full injection to a cheap nudge. All default off, A/B'd by the existing nightly flag-graduation gate.

**Tech Stack:** `grounding.log` (JSONL ledger), `memo.dashboard_logs`, `recall_logic.rank_hits`, `dream_flags` gates, `memo eval recall`.

Spec: `docs/SPECS/2026-08-06-recall-outcome-admission-design.md`.

## Global Constraints

- Every ranking change lands **inside `rank_hits`** (`recall_logic.py:1214`) and nowhere else. That function is what keeps the daemon path (`_recall_logic`) and the eval harness (`eval_recall.py:802`) from diverging; a change made outside it is unmeasurable by construction.
- Default OFF. Every new `*_ENABLED` flag must declare a `GateSpec` in `dream_flags.GATES` — `test_dream_flags.py` enforces completeness, so a dark flag without a gate cannot merge.
- An unobserved memory must produce a multiplier of **exactly 1.0**. Anything else silently penalises a young corpus.
- Never a hard drop from the candidate list. Demote only.
- `grounding.log` keys `recall_id` as an **8-char prefix**. Match prefixes the way `grounding.match_cited` already does; do not invent a second rule.
- The eval gate measures **corpus state, not the diff**. If `memo eval recall` blocks a push, confirm the cause is this change before reaching for `--no-verify` — on 2026-08-06 a nightly `negative_capture` run moved avoid@k from 1.0 to 0.5 with no ranking code involved.
- Shared working tree: stage explicit paths only.

---

### Task 1: The utility prior

**Files:**
- Create: `src/memo/recall_utility.py`
- Create: `tests/test_recall_utility.py`

**Interfaces:**
- Produces:
  - `SCHEMA: str` = `"memo.recall_utility.v1"`
  - `ALPHA: float` = `8.0` — smoothing strength
  - `compute(cfg) -> dict[str, Any]` — reads `grounding.log`, writes `state_dir/recall_utility.json`, returns the payload
  - `load(state_dir) -> dict[str, Any] | None` — the persisted payload, or `None` when absent/corrupt
  - `multiplier(u: float, prior: float, strength: float) -> float`
  - `utility_map(payload) -> dict[str, float]` — id-prefix → `u`, the shape `RankKnobs` consumes

- [ ] **Step 1: Write the failing tests**

`tests/test_recall_utility.py`:

```python
"""The prior. Its single most important property: a memory the ledger has never
seen must come out exactly neutral, so a young corpus is untouched."""

from __future__ import annotations

import json

from memo import recall_utility


def test_multiplier_is_exactly_neutral_at_the_prior() -> None:
    assert recall_utility.multiplier(0.5, 0.5, 0.15) == 1.0


def test_multiplier_demotes_below_the_prior() -> None:
    assert recall_utility.multiplier(0.0, 0.5, 0.15) < 1.0


def test_multiplier_promotes_above_the_prior() -> None:
    assert recall_utility.multiplier(1.0, 0.5, 0.15) > 1.0


def test_multiplier_is_clipped_both_ways() -> None:
    assert recall_utility.multiplier(0.0, 0.9, 0.15) >= 1 - 0.15
    assert recall_utility.multiplier(1.0, 0.1, 0.15) <= 1 + 0.15


def test_zero_strength_is_a_no_op() -> None:
    assert recall_utility.multiplier(0.0, 0.9, 0.0) == 1.0


def test_smoothing_keeps_a_single_hit_off_the_ceiling() -> None:
    payload = recall_utility._smoothed(grounded=1, surfaced=1, prior=0.5)
    assert payload < 0.9, "one observation should not read as a perfect memory"


def test_compute_writes_a_payload_and_survives_an_empty_ledger(tmp_cfg) -> None:
    out = recall_utility.compute(tmp_cfg)
    assert out["schema"] == recall_utility.SCHEMA
    assert out["memories"] == {}
    on_disk = json.loads(
        (tmp_cfg.state_dir / "recall_utility.json").read_text(encoding="utf-8")
    )
    assert on_disk["schema"] == recall_utility.SCHEMA


def test_compute_counts_surfaced_and_grounded(tmp_cfg) -> None:
    from memo.dashboard_logs import append_grounding_log

    for used in (0.9, 0.0, 0.0):
        append_grounding_log(
            tmp_cfg.state_dir,
            session_id="s1",
            turn=1,
            recall_id="aaaaaaaa",
            used_score=used,
            method="cited",
        )
    out = recall_utility.compute(tmp_cfg)
    row = out["memories"]["aaaaaaaa"]
    assert row["surfaced"] == 3
    assert row["grounded"] == 1


def test_load_returns_none_on_a_corrupt_file(tmp_cfg) -> None:
    (tmp_cfg.state_dir / "recall_utility.json").write_text("{not json", encoding="utf-8")
    assert recall_utility.load(tmp_cfg.state_dir) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync pytest tests/test_recall_utility.py -q`
Expected: `ModuleNotFoundError: No module named 'memo.recall_utility'`.

- [ ] **Step 3: Implement**

`src/memo/recall_utility.py`:

```python
"""Per-memory utility prior: how often a surfaced memory was actually used.

Measured 2026-08-06 on the live install: referenced_rate 0.011 (18 of 1702
surfaced memories were later fetched), grounded_rate 0.501, over 246k injected
tokens. Roughly half the injected block is ballast, paid on every prompt, and
nothing in the ranking knows which half.

`grounding.log` already answers this -- it is the (session_id, turn, recall_id)
recall-to-use ledger. This module joins it per memory, exactly as
`capture_weights.compute_type_citation_stats` does per type.

Smoothing matters more than it looks: with `u == prior` for an unobserved
memory, the ranking stage is provably neutral on a corpus the ledger has never
seen, so the feature is inert on a fresh install rather than wrong.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "memo.recall_utility.v1"
ALPHA = 8.0
# A row counts as grounded at or above this used_score. Mirrors capture_weights.
STRONG = 0.5

_log = logging.getLogger(__name__)


def _smoothed(*, grounded: int, surfaced: int, prior: float) -> float:
    """Beta-Bernoulli posterior mean. Returns `prior` exactly when surfaced=0."""
    return (grounded + ALPHA * prior) / (surfaced + ALPHA)


def multiplier(u: float, prior: float, strength: float) -> float:
    """Bounded demote/promote factor. Exactly 1.0 when `u == prior`."""
    if strength <= 0.0:
        return 1.0
    raw = 1.0 + strength * 2.0 * (u - prior)
    return max(1.0 - strength, min(1.0 + strength, raw))


def _path(state_dir: Path) -> Path:
    return state_dir / "recall_utility.json"


def compute(cfg: Any) -> dict[str, Any]:
    """Join grounding.log per memory and persist the prior.

    Raises on write failure so the dream caller records it in
    `receipt["errors"]` rather than swallowing it.
    """
    from memo.dashboard_logs import read_grounding_log

    rows = read_grounding_log(cfg.state_dir, limit=20000)
    surfaced: dict[str, int] = {}
    grounded: dict[str, int] = {}
    for r in rows:
        rid = str(r.get("recall_id") or "")
        if len(rid) < 8:
            continue
        rid = rid[:8]
        surfaced[rid] = surfaced.get(rid, 0) + 1
        try:
            used = float(r.get("used_score") or 0.0)
        except (TypeError, ValueError):
            used = 0.0
        if used >= STRONG:
            grounded[rid] = grounded.get(rid, 0) + 1

    total_s = sum(surfaced.values())
    total_g = sum(grounded.values())
    prior = (total_g / total_s) if total_s else 0.5

    memories = {
        rid: {
            "surfaced": n,
            "grounded": grounded.get(rid, 0),
            "u": round(_smoothed(grounded=grounded.get(rid, 0), surfaced=n, prior=prior), 4),
        }
        for rid, n in surfaced.items()
    }

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "computed": datetime.now(UTC).isoformat(timespec="seconds"),
        "prior": round(prior, 4),
        "memories": memories,
    }
    _path(cfg.state_dir).write_text(json.dumps(payload), encoding="utf-8")
    return payload


def load(state_dir: Path) -> dict[str, Any] | None:
    """The persisted prior, or None. A missing or corrupt file must never break
    recall -- the ranking stage simply does not run."""
    path = _path(state_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _log.debug("recall_utility: unreadable prior: %s", exc)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return None
    return payload


def utility_map(payload: dict[str, Any] | None) -> dict[str, float]:
    """id-prefix -> u, the shape RankKnobs consumes."""
    if not payload:
        return {}
    return {
        rid: float(row.get("u", 0.0))
        for rid, row in (payload.get("memories") or {}).items()
        if isinstance(row, dict)
    }
```

- [ ] **Step 4: Run the tests**

Run: `uv run --no-sync pytest tests/test_recall_utility.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/memo/recall_utility.py tests/test_recall_utility.py
git commit -m "feat(recall): compute a per-memory utility prior from grounding.log"
```

---

### Task 2: The nightly pass

**Files:**
- Modify: `src/memo/cli_dream_passes.py` (add `_run_recall_utility`, next to `_run_capture_weights` at line 546)
- Modify: `src/memo/cli_dream.py` (import it and call it in the pass chain, alongside the other maintenance passes)
- Create: `tests/test_dream_recall_utility.py`

**Interfaces:**
- Consumes: `recall_utility.compute` from Task 1.
- Produces: `_run_recall_utility(cfg: Config) -> dict` — the receipt fragment `{"memories": int, "prior": float}`.

- [ ] **Step 1: Write the failing test**

`tests/test_dream_recall_utility.py`:

```python
"""The pass records its result in the receipt, and a failure lands in
receipt["errors"] rather than vanishing."""

from __future__ import annotations

import pytest

from memo.cli_dream_passes import _run_recall_utility


def test_pass_returns_a_receipt_fragment(tmp_cfg) -> None:
    out = _run_recall_utility(tmp_cfg)
    assert out["memories"] == 0
    assert 0.0 <= out["prior"] <= 1.0


def test_pass_counts_the_ledger(tmp_cfg) -> None:
    from memo.dashboard_logs import append_grounding_log

    append_grounding_log(
        tmp_cfg.state_dir,
        session_id="s",
        turn=1,
        recall_id="bbbbbbbb",
        used_score=0.9,
        method="cited",
    )
    assert _run_recall_utility(tmp_cfg)["memories"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/test_dream_recall_utility.py -q`
Expected: `ImportError: cannot import name '_run_recall_utility'`.

- [ ] **Step 3: Implement the pass**

In `src/memo/cli_dream_passes.py`, next to `_run_capture_weights`:

```python
def _run_recall_utility(cfg: Config) -> dict:
    """Refresh the per-memory utility prior from grounding.log.

    Cheap and pure-stdlib: a JSONL read plus a dict join, no MLX and no store
    read. Raises on write failure so the dream caller records it in
    `receipt["errors"]` -- a prior that silently stopped refreshing would decay
    into a stale ranking input nobody notices.
    """
    from memo.recall_utility import compute

    payload = compute(cfg)
    return {"memories": len(payload["memories"]), "prior": payload["prior"]}
```

- [ ] **Step 4: Wire it into the chain**

In `src/memo/cli_dream.py`, add `_run_recall_utility` to the import block from `cli_dream_passes` (the one containing `_run_capture_weights`), and call it in the maintenance-pass sequence, recording its result in the receipt under the key `"recall_utility"` — follow the exact shape the neighbouring passes use (see the `_run_prewarm_queries` call at line 1592).

- [ ] **Step 5: Run the tests**

Run: `uv run --no-sync pytest tests/test_dream_recall_utility.py tests/test_dream_eval.py -q`
Expected: green.

- [ ] **Step 6: Verify the receipt**

Run: `uv run --no-sync memo dream run --dry-run` (or the nearest non-destructive invocation the command offers), then `uv run --no-sync memo dream status`
Expected: a `recall_utility` entry in the receipt.

- [ ] **Step 7: Commit**

```bash
git add src/memo/cli_dream_passes.py src/memo/cli_dream.py tests/test_dream_recall_utility.py
git commit -m "feat(dream): refresh the recall utility prior nightly"
```

---

### Task 3: The ranking stage

**Files:**
- Modify: `src/memo/recall_logic.py` — `RankKnobs` (line 1028), `knobs_from_flags` (line 1053), `rank_hits` (line 1214)
- Modify: `src/memo/flags_recall.py` (append four specs to `SPECS`)
- Create: `tests/test_rank_hits_utility.py`

**Interfaces:**
- Consumes: `recall_utility.multiplier`, `recall_utility.load`, `recall_utility.utility_map` from Task 1.
- Produces on `RankKnobs`:
  - `utility: dict[str, float] | None = None`
  - `utility_prior: float = 0.5`
  - `utility_strength: float = 0.0`
  - `explore_min_obs: int = 0`
  - `marginal_floor: float = 0.0`

Three behaviours, each independently disabled by its knob being `0`:

1. **Demote multiplier** — after the existing boosts, before the `min_sim` gate.
2. **Exploration slot** — the last position of the `top_k` prefix goes to the highest-scoring candidate the ledger has barely seen, chosen only among candidates that already passed the gate.
3. **Marginal floor** — a candidate in the `top_k` prefix scoring below `marginal_floor × score[0]` is moved after the prefix boundary. It is not dropped: it degrades from a full-body injection to a cheap id/title nudge. That is where the token saving comes from.

- [ ] **Step 1: Write the failing tests**

`tests/test_rank_hits_utility.py`:

```python
"""The ranking stage. Every assertion here is about a knob at its default being
a provable no-op, because that is what makes the feature safe to ship dark."""

from __future__ import annotations

from dataclasses import dataclass

from memo.recall_logic import RankKnobs, rank_hits


@dataclass
class _Hit:
    id: str
    score: float
    body: str = "x" * 200
    type: str = "note"
    tags: tuple[str, ...] = ()


def _hits() -> list[_Hit]:
    return [_Hit(id=f"{i:08d}" + "0" * 24, score=1.0 - i * 0.1) for i in range(5)]


def _knobs(**kw) -> RankKnobs:
    return RankKnobs(top_k=3, min_sim=0.0, min_body_chars=0, mode="vec", **kw)


def test_defaults_leave_ranking_identical() -> None:
    hits = _hits()
    assert [h.id for h in rank_hits(hits, _knobs())] == [h.id for h in hits]


def test_utility_map_without_strength_is_a_no_op() -> None:
    hits = _hits()
    knobs = _knobs(utility={hits[0].id[:8]: 0.0}, utility_prior=0.5, utility_strength=0.0)
    assert [h.id for h in rank_hits(hits, knobs)] == [h.id for h in hits]


def test_a_never_used_memory_is_demoted_below_its_neighbour() -> None:
    hits = [_Hit(id="a" * 32, score=1.00), _Hit(id="b" * 32, score=0.98)]
    knobs = _knobs(
        utility={"a" * 8: 0.0, "b" * 8: 0.5},
        utility_prior=0.5,
        utility_strength=0.15,
    )
    assert [h.id for h in rank_hits(hits, knobs)][0] == "b" * 32


def test_an_unobserved_memory_is_untouched() -> None:
    hits = [_Hit(id="a" * 32, score=1.00), _Hit(id="b" * 32, score=0.98)]
    knobs = _knobs(utility={"b" * 8: 0.5}, utility_prior=0.5, utility_strength=0.15)
    assert [h.id for h in rank_hits(hits, knobs)] == [h.id for h in hits]


def test_marginal_floor_moves_a_weak_hit_out_of_the_prefix() -> None:
    hits = [_Hit(id="a" * 32, score=1.0), _Hit(id="b" * 32, score=0.9), _Hit(id="c" * 32, score=0.2)]
    ranked = rank_hits(hits, _knobs(marginal_floor=0.6))
    assert [h.id for h in ranked[:2]] == ["a" * 32, "b" * 32]
    assert ranked[2].id == "c" * 32, "the weak hit must be demoted, never dropped"
    assert len(ranked) == 3


def test_marginal_floor_never_empties_the_prefix() -> None:
    hits = [_Hit(id="a" * 32, score=1.0), _Hit(id="b" * 32, score=0.01)]
    ranked = rank_hits(hits, _knobs(marginal_floor=0.99))
    assert ranked[0].id == "a" * 32


def test_exploration_slot_promotes_a_cold_candidate_into_the_prefix() -> None:
    hits = [
        _Hit(id="a" * 32, score=1.0),
        _Hit(id="b" * 32, score=0.9),
        _Hit(id="c" * 32, score=0.8),
        _Hit(id="d" * 32, score=0.7),
    ]
    knobs = _knobs(
        top_k=3,
        explore_min_obs=3,
        utility={"a" * 8: 0.5, "b" * 8: 0.5, "c" * 8: 0.5},
    )
    ranked = [h.id for h in rank_hits(hits, knobs)]
    assert "d" * 32 in ranked[:3], "the unobserved candidate never got a slot"
    assert ranked[0] == "a" * 32, "exploration must not touch the top slot"


def test_exploration_slot_off_by_default() -> None:
    hits = _hits()
    assert [h.id for h in rank_hits(hits, _knobs())] == [h.id for h in hits]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync pytest tests/test_rank_hits_utility.py -q`
Expected: `TypeError: RankKnobs.__init__() got an unexpected keyword argument 'utility'`.

The `_Hit` stub may need more attributes than `id`/`score`/`body` depending on which stages `rank_hits` runs. Run the first test alone, read the `AttributeError`, and add exactly the attributes it asks for — no more.

- [ ] **Step 3: Add the flags**

Append to the `SPECS` tuple in `src/memo/flags_recall.py`:

```python
    _spec(
        "MEMO_RECALL_UTILITY_ENABLED",
        "bool",
        False,
        "recall",
        "Apply the nightly per-memory utility prior (state_dir/recall_utility.json, "
        "computed from grounding.log) as a bounded demote multiplier in rank_hits. "
        "A memory surfaced 30 times and grounded 0 stops crowding the block; one "
        "the ledger has never seen is untouched (u == prior => multiplier exactly "
        "1.0). Demote only, never a drop. Default OFF; A/B'd nightly by the "
        "dream_flags recall gate.",
    ),
    _spec(
        "MEMO_RECALL_UTILITY_STRENGTH",
        "float",
        0.15,
        "recall",
        "Half-width of the utility multiplier: the factor is clipped to "
        "[1-strength, 1+strength]. Inert unless MEMO_RECALL_UTILITY_ENABLED.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_RECALL_UTILITY_MIN_OBSERVATIONS",
        "int",
        3,
        "recall",
        "A candidate with fewer than this many ledger observations is eligible for "
        "the exploration slot -- the last position of the top_k prefix, reserved "
        "for a cold candidate that ALREADY passed the min_sim gate. Without it, a "
        "memory never surfaced is never grounded and so is never surfaced: a pure "
        "exploit policy freezes the corpus. 0 = no exploration slot.",
        min_val=0,
    ),
    _spec(
        "MEMO_RECALL_MARGINAL_FLOOR",
        "float",
        0.0,
        "recall",
        "A top_k candidate scoring below floor x top_score is moved out of the "
        "injected prefix and into the nudge tail -- demoted from a full body to an "
        "id/title line, not dropped. This is where the token saving comes from on "
        "a prompt with one strong match and four weak ones. 0.0 = OFF.",
        min_val=0.0,
        max_val=1.0,
    ),
```

- [ ] **Step 4: Extend `RankKnobs` and `knobs_from_flags`**

Add the five fields to `RankKnobs` (after `cwd`, line 1050):

```python
    # Outcome admission (all inert at these defaults). `utility` maps an 8-char
    # id prefix to its smoothed grounded-rate; `utility_prior` is the corpus
    # mean, so an absent id yields a multiplier of exactly 1.0.
    utility: dict[str, float] | None = None
    utility_prior: float = 0.5
    utility_strength: float = 0.0
    explore_min_obs: int = 0
    marginal_floor: float = 0.0
```

In `knobs_from_flags`, resolve them from the flags, loading the prior only when enabled:

```python
    utility_map: dict[str, float] | None = None
    utility_prior = 0.5
    utility_strength = 0.0
    if flag_bool("MEMO_RECALL_UTILITY_ENABLED"):
        from memo import recall_utility

        payload = recall_utility.load(Config.from_env().state_dir)
        if payload is not None:
            utility_map = recall_utility.utility_map(payload)
            utility_prior = float(payload.get("prior", 0.5))
            utility_strength = flag_float("MEMO_RECALL_UTILITY_STRENGTH") or 0.0
```

Use whatever `Config`/`state_dir` accessor `knobs_from_flags` already has in scope — do NOT call `Config.from_env()` if a config is already available there, and never in a way that would fire on the eval path with the wrong state dir.

- [ ] **Step 5: Add the three stages to `rank_hits`**

After the altitude/code-proximity boosts and **before** the `min_sim` gate, add the multiplier:

```python
    if knobs.utility_strength > 0.0 and knobs.utility:
        from memo.recall_utility import multiplier as _utility_multiplier

        for h in raw:
            u = knobs.utility.get(getattr(h, "id", "")[:8])
            if u is None:
                continue  # unobserved: exactly neutral, no arithmetic at all
            h.score *= _utility_multiplier(u, knobs.utility_prior, knobs.utility_strength)
        if explain is not None:
            _explain_stage(explain, raw, "utility")
```

After the gate and the final ordering, add the floor and the exploration slot, in that order:

```python
    k = knobs.top_k
    if knobs.marginal_floor > 0.0 and len(ordered) > 1 and k >= 1:
        top = ordered[0].score
        prefix = [h for h in ordered[:k] if h.score >= knobs.marginal_floor * top]
        demoted = [h for h in ordered[:k] if h.score < knobs.marginal_floor * top]
        # The prefix never empties: the best hit always earns full injection.
        if not prefix:
            prefix, demoted = ordered[:1], demoted[1:]
        ordered = prefix + demoted + ordered[k:]

    if knobs.explore_min_obs > 0 and len(ordered) > k >= 1:
        seen = knobs.utility or {}
        cold = next(
            (h for h in ordered[k:] if seen.get(getattr(h, "id", "")[:8]) is None),
            None,
        )
        if cold is not None:
            # Reordering among ADMITTED candidates only -- never an injection of
            # something retrieval already rejected. The top slot is untouched.
            ordered.remove(cold)
            ordered.insert(max(1, k - 1), cold)
```

Adapt the variable name `ordered` to whatever `rank_hits` actually calls its post-gate list.

- [ ] **Step 6: Run the tests**

Run: `uv run --no-sync pytest tests/test_rank_hits_utility.py -q`
Expected: 8 passed.

- [ ] **Step 7: Prove the default path is untouched**

Run: `uv run --no-sync pytest tests/test_recall_hooks.py tests/test_recall_server.py tests/test_dream_tune_knobs.py -q`
Expected: green, with no test changed. All five knobs default to inert.

- [ ] **Step 8: Commit**

```bash
git add src/memo/recall_logic.py src/memo/flags_recall.py tests/test_rank_hits_utility.py
git commit -m "feat(recall): admit by outcome -- utility demote, exploration slot, marginal floor"
```

---

### Task 4: The graduation gate and the measurement

**Files:**
- Modify: `src/memo/dream_flags.py` (`GATES`)
- Create: `tests/test_recall_utility_gate.py`

**Interfaces:**
- Consumes: `MEMO_RECALL_UTILITY_ENABLED` from Task 3.

- [ ] **Step 1: Write the failing test**

`tests/test_recall_utility_gate.py`:

```python
"""A dark flag without a declared gate cannot merge -- test_dream_flags.py
enforces completeness. This asserts the gate says the right thing."""

from __future__ import annotations

from memo.dream_flags import GATES


def test_the_utility_flag_declares_a_recall_gate() -> None:
    gate = GATES["MEMO_RECALL_UTILITY_ENABLED"]
    assert gate.kind == "recall"
    assert gate.mode == "vec"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/test_recall_utility_gate.py tests/test_dream_flags.py -q`
Expected: both fail — `KeyError` here, and `test_dream_flags.py`'s completeness check flags the undeclared flag.

- [ ] **Step 3: Declare the gate**

In `src/memo/dream_flags.py`, add to the `GATES` construction, beside the other `recall` entries (line 105-108):

```python
        _g(
            "MEMO_RECALL_UTILITY_ENABLED",
            "recall",
            "outcome-based demote inside rank_hits; recall A/B via the eval "
            "flag_overrides seam -- the ON pin measures the prior at its "
            "nightly-computed strength",
            extra_flags=(("MEMO_RECALL_MARGINAL_FLOOR", "0.6"),),
        ),
```

- [ ] **Step 4: Run the tests**

Run: `uv run --no-sync pytest tests/test_recall_utility_gate.py tests/test_dream_flags.py -q`
Expected: green.

- [ ] **Step 5: Measure against the curated label set**

Seed the prior from the live ledger, then A/B:

```bash
env -u PYTHONPATH memo dream run --only recall-utility   # or run the pass directly
env -u PYTHONPATH memo eval recall --labels eval/regression_labels.json --k 5 --force
MEMO_RECALL_UTILITY_ENABLED=1 MEMO_RECALL_MARGINAL_FLOOR=0.6 \
  env -u PYTHONPATH memo eval recall --labels eval/regression_labels.json --k 5 --force
```

Record both runs. Acceptance: precision@5, noise@5 and avoid@k **not worse** with the flag on. A token reduction that costs precision is a failure.

`--only` may not be the real selector — check `memo dream run --help` and use whatever selects a single pass; if none exists, invoke `_run_recall_utility` through `python -c`.

- [ ] **Step 6: Record the token effect**

```bash
env -u PYTHONPATH memo usefulness
env -u PYTHONPATH memo tokens
```

Capture `referenced_rate`, `grounded_rate` and injected tokens before enabling, and again after a week of nightly runs. These are the success criteria; the eval gate is only the quality guard.

- [ ] **Step 7: Full check**

```bash
uv run --no-sync pytest -m "not slow and not conformance" -q
uv run --no-sync mypy src/memo/recall_utility.py src/memo/recall_logic.py src/memo/dream_flags.py
uv run --no-sync ruff check src/memo/recall_utility.py src/memo/recall_logic.py src/memo/dream_flags.py
```

- [ ] **Step 8: Commit**

```bash
git add src/memo/dream_flags.py tests/test_recall_utility_gate.py
git commit -m "feat(dream): declare the graduation gate for outcome-based admission"
```

---

## Self-review notes

- Spec coverage: nightly prior (Tasks 1-2), demote multiplier inside `rank_hits` (Task 3), exploration slot (Task 3), variable K (Task 3, as the marginal floor), four flags (Task 3), `GateSpec` (Task 4), the eval measurement and its acceptance rule (Task 4).
- The spec calls the third behaviour "variable K". Implemented as a demotion from the injected prefix to the nudge tail rather than a truncation: `rank_hits` returns one list that the caller splits at `top_k`, and truncating it would delete the "Also in your memory" nudges the spec does not touch. Same token saving — full bodies are the cost — without removing discovery.
- `ALPHA = 8.0` and `STRONG = 0.5` are stated as constants with reasons in the module docstring; `STRONG` deliberately mirrors `capture_weights` rather than introducing a second notion of "used".
- Unverified identifiers flagged in-step: the post-gate list variable name in `rank_hits` (Task 3 Step 5), the config accessor inside `knobs_from_flags` (Task 3 Step 4), the `_Hit` stub's required attributes (Task 3 Step 2), and the dream single-pass selector (Task 4 Step 5).
