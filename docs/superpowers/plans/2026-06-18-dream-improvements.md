# memo dream — 4 mejoras Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add signal-gather, date normalization, quality-floor prune, and orientation summary to `memo dream run`.

**Architecture:** Each improvement is a self-contained pass in `dream_run`. Improvements 1, 3, and 4 live entirely in `cli_dream.py`. Improvement 2 lives in `memory/prompts.py` (prompt) and `memory/consolidate_ops.py` (post-process). Each pass is opt-out via a new `--skip-*` flag.

**Tech Stack:** Python 3.12, Click, SQLite, Rich, pytest, existing `memo.transcript_miner.mine_transcripts`, `memo.lifecycle.LifecycleManager.archive_memoria`.

## Global Constraints

- Archive-first philosophy: quality-floor prune must call `lifecycle.archive_memoria()`, not `mem.delete()`
- All new flags use `_spec()` from `memo.flags_base` and go in `flags_misc.py`
- All new options added to `dream run` default to enabled (opt-out, not opt-in)
- `--dry-run` skips signal-gather, prune-floor mutations, and date-normalization saves
- Receipt dict is extended (new keys); existing keys unchanged — backwards-compatible
- Tests use `tmp_cfg` fixture or isolated `Config`; never read the real vault

---

### Task 1: Quality-floor prune — store method + flags

**Files:**
- Modify: `src/memo/store/signal_queries.py` (add `prune_floor_candidates` after `decay_roi`)
- Modify: `src/memo/flags_misc.py` (add 2 new `_spec` entries)
- Test: `tests/test_dream_prune_floor.py` (new file)

**Interfaces:**
- Produces: `_SignalQueriesMixin.prune_floor_candidates(roi_floor: float, min_age_days: int, exclude_types: set[str] | None = None) -> list[dict[str, Any]]`
  - Returns list of `{id: str, roi_score: float, days_old: int}`
- Produces: env flags `MEMO_DREAM_PRUNE_FLOOR` (float, default 0.15), `MEMO_DREAM_PRUNE_MIN_AGE_DAYS` (int, default 90)

- [ ] **Step 1: Write the failing test**

Create `tests/test_dream_prune_floor.py`:

```python
"""Tests for prune_floor_candidates store method."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memo.store.store import VecStore


def _make_store(tmp_path: Path) -> VecStore:
    import os
    os.environ.setdefault("MEMO_SKIP_MODEL_VERSION_CHECK", "1")
    return VecStore(tmp_path / "test.db", dims=4)


def _insert_memoria(store: VecStore, id_: str, type_: str, days_old: int, roi_score: float, access_count: int) -> None:
    with store._conn as cx:
        cx.execute(
            "INSERT OR REPLACE INTO meta (id, title, body, type, tags, path, updated) "
            "VALUES (?, ?, ?, ?, ?, ?, datetime('now', ? || ' days'))",
            (id_, f"title-{id_}", "body", type_, "[]", f"/fake/{id_}.md", f"-{days_old}"),
        )
        cx.execute(
            "INSERT OR REPLACE INTO memory_health (id, confidence, roi_score, updated_at) "
            "VALUES (?, 1.0, ?, datetime('now', ? || ' days'))",
            (id_, roi_score, f"-{days_old}"),
        )
        if access_count > 0:
            cx.execute(
                "INSERT OR REPLACE INTO access (id, access_count, last_accessed) "
                "VALUES (?, ?, datetime('now'))",
                (id_, access_count),
            )


def test_prune_floor_returns_low_roi_zero_access(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "aaa", "note", days_old=100, roi_score=0.10, access_count=0)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    ids = [c["id"] for c in candidates]
    assert "aaa" in ids


def test_prune_floor_excludes_accessed(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "bbb", "note", days_old=100, roi_score=0.10, access_count=3)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    ids = [c["id"] for c in candidates]
    assert "bbb" not in ids


def test_prune_floor_excludes_synthesis_and_reference(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "ccc", "synthesis", days_old=200, roi_score=0.05, access_count=0)
    _insert_memoria(store, "ddd", "reference", days_old=200, roi_score=0.05, access_count=0)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    ids = [c["id"] for c in candidates]
    assert "ccc" not in ids
    assert "ddd" not in ids


def test_prune_floor_excludes_too_recent(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "eee", "note", days_old=30, roi_score=0.10, access_count=0)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    ids = [c["id"] for c in candidates]
    assert "eee" not in ids


def test_prune_floor_excludes_above_floor(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "fff", "note", days_old=100, roi_score=0.50, access_count=0)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    ids = [c["id"] for c in candidates]
    assert "fff" not in ids


def test_prune_floor_result_has_required_keys(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "ggg", "note", days_old=100, roi_score=0.10, access_count=0)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    assert candidates
    c = candidates[0]
    assert "id" in c
    assert "roi_score" in c
    assert "days_old" in c
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --no-sync pytest tests/test_dream_prune_floor.py -v
```

Expected: `AttributeError: '_SignalQueriesMixin' object has no attribute 'prune_floor_candidates'`

- [ ] **Step 3: Add `prune_floor_candidates` to `signal_queries.py`**

Open `src/memo/store/signal_queries.py`. After the `decay_roi` method (line ~277), add:

```python
    def prune_floor_candidates(
        self,
        roi_floor: float = 0.15,
        min_age_days: int = 90,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return memorias below roi_floor, never accessed, and older than min_age_days.

        Always excludes 'synthesis' and 'reference' types. Returns list of
        {id, roi_score, days_old}.
        """
        excluded = (exclude_types or set()) | {"synthesis", "reference"}
        placeholders = ",".join("?" for _ in excluded)
        rows = self._conn.execute(
            f"""
            SELECT m.id,
                   COALESCE(h.roi_score, 1.0)                        AS roi_score,
                   CAST(julianday('now') - julianday(m.updated) AS INTEGER) AS days_old
              FROM meta m
              LEFT JOIN memory_health h ON h.id = m.id
              LEFT JOIN access a       ON a.id = m.id
             WHERE COALESCE(h.roi_score, 1.0) < ?
               AND COALESCE(a.access_count, 0) = 0
               AND m.updated < datetime('now', '-' || ? || ' days')
               AND m.type NOT IN ({placeholders})
            """,
            (roi_floor, min_age_days, *excluded),
        ).fetchall()
        return [{"id": r["id"], "roi_score": r["roi_score"], "days_old": r["days_old"]} for r in rows]
```

- [ ] **Step 4: Add flags to `flags_misc.py`**

Append before the closing `)` of `SPECS`:

```python
    # dream pipeline
    _spec(
        "MEMO_DREAM_PRUNE_FLOOR",
        "float",
        0.15,
        "dream",
        "ROI score floor for quality-floor prune in `memo dream run`. "
        "Memorias with roi_score below this threshold, zero access count, and age "
        "> MEMO_DREAM_PRUNE_MIN_AGE_DAYS are archived during the dream prune pass.",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec(
        "MEMO_DREAM_PRUNE_MIN_AGE_DAYS",
        "int",
        90,
        "dream",
        "Minimum age in days for the quality-floor prune pass in `memo dream run`. "
        "Only memorias older than this are considered for archival.",
        min_val=0,
    ),
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run --no-sync pytest tests/test_dream_prune_floor.py -v
```

Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/memo/store/signal_queries.py src/memo/flags_misc.py tests/test_dream_prune_floor.py
git commit -m "feat: add prune_floor_candidates to signal store + MEMO_DREAM_PRUNE_* flags"
```

---

### Task 2: Date normalization — helper + synthesis hook

**Files:**
- Modify: `src/memo/memory/prompts.py` (add instruction to `_SYNTHESIS_SYSTEM_PROMPT`)
- Modify: `src/memo/memory/consolidate_ops.py` (add `_normalize_relative_dates`, call before save)
- Test: `tests/test_date_normalization.py` (new file)

**Interfaces:**
- Consumes: nothing from Task 1
- Produces: module-level function `_normalize_relative_dates(text: str, ref_date: datetime.date) -> str` in `consolidate_ops.py` (used internally; not exported)

- [ ] **Step 1: Write the failing test**

Create `tests/test_date_normalization.py`:

```python
"""Unit tests for _normalize_relative_dates in consolidate_ops."""
from __future__ import annotations

import datetime

import pytest

from memo.memory.consolidate_ops import _normalize_relative_dates


REF = datetime.date(2026, 6, 18)


def test_ayer_replaced():
    result = _normalize_relative_dates("ayer decidimos usar sqlite", REF)
    assert "2026-06-17" in result
    assert "ayer" not in result.lower()


def test_hoy_replaced():
    result = _normalize_relative_dates("hoy cerramos el PR", REF)
    assert "2026-06-18" in result


def test_anteayer_replaced():
    result = _normalize_relative_dates("anteayer el build falló", REF)
    assert "2026-06-16" in result


def test_yesterday_english():
    result = _normalize_relative_dates("yesterday we merged the fix", REF)
    assert "2026-06-17" in result


def test_today_english():
    result = _normalize_relative_dates("today the deploy happened", REF)
    assert "2026-06-18" in result


def test_la_semana_pasada():
    result = _normalize_relative_dates("la semana pasada actualizamos deps", REF)
    assert "2026-06-11" in result


def test_last_week_english():
    result = _normalize_relative_dates("last week we upgraded deps", REF)
    assert "2026-06-11" in result


def test_el_mes_pasado():
    result = _normalize_relative_dates("el mes pasado migramos a uv", REF)
    assert "2026-05" in result


def test_last_month_english():
    result = _normalize_relative_dates("last month we migrated to uv", REF)
    assert "2026-05" in result


def test_hace_n_dias():
    result = _normalize_relative_dates("hace 3 días se rompió el test", REF)
    assert "2026-06-15" in result


def test_n_days_ago_english():
    result = _normalize_relative_dates("5 days ago the server went down", REF)
    assert "2026-06-13" in result


def test_no_match_returns_original():
    text = "nothing temporal here"
    assert _normalize_relative_dates(text, REF) == text


def test_never_raises_on_bad_input():
    # Should not raise even on weird input
    result = _normalize_relative_dates("hace abc días de algo", REF)
    assert isinstance(result, str)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --no-sync pytest tests/test_date_normalization.py -v
```

Expected: `ImportError: cannot import name '_normalize_relative_dates'`

- [ ] **Step 3: Add `_normalize_relative_dates` to `consolidate_ops.py`**

At the top of `src/memo/memory/consolidate_ops.py`, after the existing imports, add:

```python
import datetime as _dt
import re as _re
```

Then add the helper function before the `_ConsolidateOpsMixin` class definition:

```python
def _normalize_relative_dates(text: str, ref_date: _dt.date) -> str:
    """Replace relative temporal expressions with ISO dates anchored to ref_date.

    Never raises — returns original text on any error.
    Patterns covered (ES + EN): ayer/yesterday, hoy/today, anteayer,
    la semana pasada/last week, el mes pasado/last month,
    hace N días/N days ago.
    """
    try:
        result = text

        def _iso(d: _dt.date) -> str:
            return d.isoformat()

        # hace N días / N days ago  (before the simpler patterns to avoid partial match)
        def _sub_hace(m: _re.Match) -> str:
            try:
                n = int(m.group(1))
                return m.group(0).replace(m.group(1), "") \
                    .rstrip() + f" ({_iso(ref_date - _dt.timedelta(days=n))})"
            except Exception:
                return m.group(0)

        result = _re.sub(
            r"hace\s+(\d+)\s+d[ií]as?",
            lambda m: f"hace {m.group(1)} días ({_iso(ref_date - _dt.timedelta(days=int(m.group(1))))})",
            result,
            flags=_re.IGNORECASE,
        )
        result = _re.sub(
            r"(\d+)\s+days?\s+ago",
            lambda m: f"{m.group(1)} days ago ({_iso(ref_date - _dt.timedelta(days=int(m.group(1))))})",
            result,
            flags=_re.IGNORECASE,
        )

        # anteayer (before ayer to avoid partial)
        result = _re.sub(
            r"\banteayer\b",
            f"anteayer ({_iso(ref_date - _dt.timedelta(days=2))})",
            result,
            flags=_re.IGNORECASE,
        )

        # ayer / yesterday
        result = _re.sub(
            r"\bayer\b",
            f"ayer ({_iso(ref_date - _dt.timedelta(days=1))})",
            result,
            flags=_re.IGNORECASE,
        )
        result = _re.sub(
            r"\byesterday\b",
            f"yesterday ({_iso(ref_date - _dt.timedelta(days=1))})",
            result,
            flags=_re.IGNORECASE,
        )

        # hoy / today
        result = _re.sub(
            r"\bhoy\b",
            f"hoy ({_iso(ref_date)})",
            result,
            flags=_re.IGNORECASE,
        )
        result = _re.sub(
            r"\btoday\b",
            f"today ({_iso(ref_date)})",
            result,
            flags=_re.IGNORECASE,
        )

        # la semana pasada / last week
        week_start = ref_date - _dt.timedelta(days=7)
        result = _re.sub(
            r"\bla\s+semana\s+pasada\b",
            f"la semana pasada ({_iso(week_start)})",
            result,
            flags=_re.IGNORECASE,
        )
        result = _re.sub(
            r"\blast\s+week\b",
            f"last week ({_iso(week_start)})",
            result,
            flags=_re.IGNORECASE,
        )

        # el mes pasado / last month
        month_ref = (ref_date.replace(day=1) - _dt.timedelta(days=1))
        month_str = month_ref.strftime("%Y-%m")
        result = _re.sub(
            r"\bel\s+mes\s+pasado\b",
            f"el mes pasado ({month_str})",
            result,
            flags=_re.IGNORECASE,
        )
        result = _re.sub(
            r"\blast\s+month\b",
            f"last month ({month_str})",
            result,
            flags=_re.IGNORECASE,
        )

        return result
    except Exception:
        return text
```

- [ ] **Step 4: Call `_normalize_relative_dates` before save in `synthesize_cross_cluster`**

In `src/memo/memory/consolidate_ops.py`, find the block where `body` is assigned (after `body = (data.get("body") or "").strip()`). Add normalization immediately after:

```python
            body = (data.get("body") or "").strip()
            # normalize relative temporal references to ISO dates
            body = _normalize_relative_dates(body, _dt.date.today())
```

- [ ] **Step 5: Add date-norm instruction to `_SYNTHESIS_SYSTEM_PROMPT`**

In `src/memo/memory/prompts.py`, find `_SYNTHESIS_SYSTEM_PROMPT`. Add to the Rules section before "Output ONLY the JSON":

```
- If the insight body contains relative temporal references (ayer, hoy, la semana pasada, hace N días, yesterday, today, last week, N days ago, last month), convert them to absolute ISO dates in the body (e.g., "ayer" → "2026-06-17").
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run --no-sync pytest tests/test_date_normalization.py -v
```

Expected: 13 passed

- [ ] **Step 7: Commit**

```bash
git add src/memo/memory/consolidate_ops.py src/memo/memory/prompts.py tests/test_date_normalization.py
git commit -m "feat: normalize relative dates in dream synthesis output"
```

---

### Task 3: Orientation summary — read-only inventory before mutations

**Files:**
- Modify: `src/memo/cli_dream.py` (add `--skip-orientation` flag + orientation pass before step 1)
- Test: `tests/test_dream_orientation.py` (new file)

**Interfaces:**
- Consumes: nothing from prior tasks
- Produces: `receipt["orientation"]` dict with keys: `total`, `by_type` (dict str→int), `low_roi` (int), `stale_candidates` (int), `open_contradictions` (int), `unindexed_entities` (int)

- [ ] **Step 1: Write the failing test**

Create `tests/test_dream_orientation.py`:

```python
"""Tests for the orientation summary pass in dream run."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memo.cli_dream import _build_orientation


def _make_mock_mem():
    mem = MagicMock()
    # store._conn returns a sqlite connection mock
    conn = MagicMock()
    mem.store._conn = conn
    # contradict_store.list_open returns empty list by default
    mem.contradict_store.list_open.return_value = []
    # graph has a db path
    mem.graph.db.return_value = ":memory:"
    return mem, conn


def test_orientation_returns_required_keys():
    mem, conn = _make_mock_mem()

    def _fetchone_side(sql, *args, **kwargs):
        mock_row = MagicMock()
        mock_row.__getitem__ = MagicMock(return_value=0)
        return mock_row

    conn.execute.return_value.fetchone.side_effect = _fetchone_side
    conn.execute.return_value.fetchall.return_value = []

    result = _build_orientation(mem)
    assert "total" in result
    assert "by_type" in result
    assert "low_roi" in result
    assert "stale_candidates" in result
    assert "open_contradictions" in result
    assert "unindexed_entities" in result


def test_orientation_open_contradictions_counted():
    mem, conn = _make_mock_mem()
    conn.execute.return_value.fetchone.return_value = MagicMock(**{"__getitem__.return_value": 0})
    conn.execute.return_value.fetchall.return_value = []
    # 2 open contradiction pairs
    pair1 = MagicMock()
    pair2 = MagicMock()
    mem.contradict_store.list_open.return_value = [pair1, pair2]

    result = _build_orientation(mem)
    assert result["open_contradictions"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --no-sync pytest tests/test_dream_orientation.py -v
```

Expected: `ImportError: cannot import name '_build_orientation'`

- [ ] **Step 3: Add `_build_orientation` and `--skip-orientation` to `cli_dream.py`**

After the existing imports in `src/memo/cli_dream.py`, add:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memo.memory.facade import Memory
```

Add the helper function before `dream_cmd`:

```python
def _build_orientation(mem: "Memory") -> dict:
    """Read-only corpus inventory — runs before any mutation."""
    conn = mem.store._conn
    result: dict = {
        "total": 0,
        "by_type": {},
        "low_roi": 0,
        "stale_candidates": 0,
        "open_contradictions": 0,
        "unindexed_entities": 0,
    }
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta WHERE type != 'reference'"
        ).fetchone()
        result["total"] = int(row["n"]) if row else 0
    except Exception:
        pass

    try:
        rows = conn.execute(
            "SELECT type, COUNT(*) AS n FROM meta WHERE type != 'reference' GROUP BY type"
        ).fetchall()
        result["by_type"] = {r["type"]: int(r["n"]) for r in rows}
    except Exception:
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "LEFT JOIN memory_health h ON h.id = m.id "
            "WHERE COALESCE(h.roi_score, 1.0) < 0.3 AND m.type != 'reference'"
        ).fetchone()
        result["low_roi"] = int(row["n"]) if row else 0
    except Exception:
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "LEFT JOIN access a ON a.id = m.id "
            "WHERE m.updated < datetime('now', '-365 days') "
            "AND COALESCE(a.access_count, 0) = 0 "
            "AND m.type != 'reference'"
        ).fetchone()
        result["stale_candidates"] = int(row["n"]) if row else 0
    except Exception:
        pass

    try:
        pairs = mem.contradict_store.list_open()
        result["open_contradictions"] = len(pairs)
    except Exception:
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "WHERE m.type != 'reference' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM entity_memoria em "
            "  JOIN entities e ON e.id = em.entity_id "
            "  WHERE em.memoria_id = m.id"
            ")"
        ).fetchone()
        result["unindexed_entities"] = int(row["n"]) if row else 0
    except Exception:
        pass

    return result
```

Add `--skip-orientation` flag to `dream_run`:

```python
@click.option("--skip-orientation", is_flag=True, help="Skip the pre-mutation inventory panel.")
```

Update `dream_run` signature to include `skip_orientation: bool`.

Add the orientation pass at the start of `dream_run`, before step 0 (TTLs), after `mem = _get_memory(cfg)`:

```python
        # Orientation — read-only inventory before mutations
        orientation: dict = {}
        if not skip_orientation:
            progress.update(step, description="[dim]orientación — inventariando corpus...[/dim]")
            try:
                orientation = _build_orientation(mem)
                from rich.table import Table
                tbl = Table(show_header=False, box=None, padding=(0, 1))
                tbl.add_column("", style="dim")
                tbl.add_column("", justify="right")
                tbl.add_row("memorias totales", str(orientation["total"]))
                for t, n in sorted(orientation["by_type"].items()):
                    tbl.add_row(f"  {t}", str(n))
                tbl.add_row("roi < 0.3", str(orientation["low_roi"]))
                tbl.add_row("stale candidates (>365d)", str(orientation["stale_candidates"]))
                tbl.add_row("contradicciones abiertas", str(orientation["open_contradictions"]))
                tbl.add_row("sin entidades indexadas", str(orientation["unindexed_entities"]))
                from rich.panel import Panel
                console.print(Panel(tbl, title="[bold cyan]Inventario pre-dream[/bold cyan]", expand=False))
            except Exception as exc:
                receipt["errors"].append(f"orientation: {type(exc).__name__}: {exc}")
        receipt["orientation"] = orientation
```

Also update the `total_steps` calculation — orientation doesn't count as a mutable step (it's always fast and read-only).

- [ ] **Step 4: Run tests**

```bash
uv run --no-sync pytest tests/test_dream_orientation.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/memo/cli_dream.py tests/test_dream_orientation.py
git commit -m "feat: add orientation inventory panel to dream run"
```

---

### Task 4: Signal gather — wire mine_transcripts into dream pipeline

**Files:**
- Modify: `src/memo/cli_dream.py` (add `--skip-signal-gather` + Phase 0 signal-gather pass)
- Test: `tests/test_dream_signal_gather.py` (new file)

**Interfaces:**
- Consumes: `memo.transcript_miner.mine_transcripts(since_days=int, file_limit=int) -> dict` — returns `{status, files_processed, saved: list, candidates, skipped_dup, ...}`
- Consumes: `state_dir/dream/.last_run_ts` — float timestamp of last dream run
- Produces: `receipt["signal_gathered"]` dict with keys: `files_processed` (int), `memorias_saved` (int), `skipped_dup` (int)

- [ ] **Step 1: Write the failing test**

Create `tests/test_dream_signal_gather.py`:

```python
"""Tests for signal-gather phase in dream run."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memo.cli_dream import _run_signal_gather


def test_signal_gather_returns_summary_keys(tmp_path: Path):
    fake_result = {
        "status": "ok",
        "files_processed": 3,
        "saved": ["id1", "id2"],
        "skipped_dup": 1,
        "candidates": 5,
    }
    with patch("memo.cli_dream.mine_transcripts", return_value=fake_result):
        result = _run_signal_gather(since_days=7, file_limit=20)
    assert result["files_processed"] == 3
    assert result["memorias_saved"] == 2
    assert result["skipped_dup"] == 1


def test_signal_gather_no_files_returns_zeros(tmp_path: Path):
    with patch("memo.cli_dream.mine_transcripts", return_value={"status": "no_files"}):
        result = _run_signal_gather(since_days=7, file_limit=20)
    assert result["memorias_saved"] == 0
    assert result["files_processed"] == 0


def test_signal_gather_exception_returns_zeros(tmp_path: Path):
    with patch("memo.cli_dream.mine_transcripts", side_effect=Exception("boom")):
        result = _run_signal_gather(since_days=7, file_limit=20)
    assert result["memorias_saved"] == 0
    assert "error" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --no-sync pytest tests/test_dream_signal_gather.py -v
```

Expected: `ImportError: cannot import name '_run_signal_gather'`

- [ ] **Step 3: Add import and `_run_signal_gather` helper to `cli_dream.py`**

Add to imports at top of `src/memo/cli_dream.py`:

```python
from memo.transcript_miner import mine_transcripts
```

Add helper before `dream_cmd`:

```python
def _run_signal_gather(since_days: int, file_limit: int = 20) -> dict:
    """Run transcript mining and return a compact summary.

    Never raises — exceptions are captured in the returned dict.
    """
    try:
        res = mine_transcripts(since_days=since_days, file_limit=file_limit)
        return {
            "files_processed": res.get("files_processed", 0),
            "memorias_saved": len(res.get("saved") or []),
            "skipped_dup": res.get("skipped_dup", 0),
        }
    except Exception as exc:
        return {"files_processed": 0, "memorias_saved": 0, "skipped_dup": 0, "error": str(exc)}
```

- [ ] **Step 4: Wire signal-gather into `dream_run`**

Add `--skip-signal-gather` to `dream_run` options:

```python
@click.option("--skip-signal-gather", is_flag=True, help="Skip transcript mining phase.")
```

Update `dream_run` signature to include `skip_signal_gather: bool`.

After the orientation pass (and before TTL enforcement), add:

```python
        # Phase 0 — Signal gather: mine new transcripts since last dream run
        receipt["signal_gathered"] = {"files_processed": 0, "memorias_saved": 0, "skipped_dup": 0}
        if not skip_signal_gather and not dry_run:
            progress.update(step, description="[0/7] signal gather — minando transcripts...")
            try:
                ts_file = _state_path(cfg) / ".last_run_ts"
                try:
                    last_ts = float(ts_file.read_text().strip())
                    since_days = max(1, int((time.time() - last_ts) / 86400) + 1)
                except Exception:
                    since_days = 7
                sg = _run_signal_gather(since_days=since_days, file_limit=20)
                receipt["signal_gathered"] = sg
                progress.update(
                    step,
                    description=(
                        f"[0/7] signal gather [green]✓[/green]  "
                        f"{sg['files_processed']} files, {sg['memorias_saved']} saved"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"signal_gather: {type(exc).__name__}: {exc}")
                progress.update(step, description="[0/7] signal gather [yellow]warn[/yellow]")
        else:
            progress.update(step, description="[0/7] signal gather [dim]skip[/dim]")
```

Update the summary print at the end of `dream_run` to include signal-gather output:

```python
    sg = receipt.get("signal_gathered", {})
    if sg.get("files_processed") or sg.get("memorias_saved"):
        console.print(
            f"  signal gather: {sg['files_processed']} files, "
            f"{sg['memorias_saved']} saved, {sg.get('skipped_dup', 0)} dup skipped"
        )
```

- [ ] **Step 5: Run tests**

```bash
uv run --no-sync pytest tests/test_dream_signal_gather.py -v
```

Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/memo/cli_dream.py tests/test_dream_signal_gather.py
git commit -m "feat: add signal-gather phase 0 to dream run (mine transcripts since last run)"
```

---

### Task 5: Quality-floor prune — wire into dream pipeline

**Files:**
- Modify: `src/memo/cli_dream.py` (add `--skip-prune-floor` + prune pass after ROI decay)
- Test: `tests/test_dream_prune_floor.py` (extend with integration test)

**Interfaces:**
- Consumes: `store.prune_floor_candidates(roi_floor, min_age_days)` from Task 1
- Consumes: `lifecycle.archive_memoria(id) -> bool` from `memo.lifecycle.LifecycleManager`
- Consumes: `flag_float("MEMO_DREAM_PRUNE_FLOOR")`, `flag_int("MEMO_DREAM_PRUNE_MIN_AGE_DAYS")` from Task 1
- Produces: `receipt["pruned_floor"]` list of `{id, roi_score, days_old}`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_dream_prune_floor.py`:

```python
from unittest.mock import MagicMock, patch


def test_prune_floor_in_dream_pipeline_archives_candidates(tmp_path: Path) -> None:
    """Integration: dream pipeline calls archive_memoria for each candidate."""
    store = _make_store(tmp_path)
    _insert_memoria(store, "zzz", "note", days_old=100, roi_score=0.10, access_count=0)

    archived = []
    mem = MagicMock()
    mem.store = store
    mem.lifecycle.archive_memoria.side_effect = lambda id_: archived.append(id_) or True

    from memo.cli_dream import _run_prune_floor
    result = _run_prune_floor(mem, roi_floor=0.15, min_age_days=90, dry_run=False)
    assert any(r["id"] == "zzz" for r in result)
    assert "zzz" in archived


def test_prune_floor_dry_run_does_not_archive(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "yyy", "note", days_old=100, roi_score=0.10, access_count=0)

    mem = MagicMock()
    mem.store = store

    from memo.cli_dream import _run_prune_floor
    result = _run_prune_floor(mem, roi_floor=0.15, min_age_days=90, dry_run=True)
    assert any(r["id"] == "yyy" for r in result)
    mem.lifecycle.archive_memoria.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --no-sync pytest tests/test_dream_prune_floor.py::test_prune_floor_in_dream_pipeline_archives_candidates -v
```

Expected: `ImportError: cannot import name '_run_prune_floor'`

- [ ] **Step 3: Add `_run_prune_floor` helper and `--skip-prune-floor` to `cli_dream.py`**

Add helper before `dream_cmd`:

```python
def _run_prune_floor(
    mem: "Memory",
    roi_floor: float,
    min_age_days: int,
    dry_run: bool,
) -> list[dict]:
    """Archive memorias below roi_floor with zero access and age >= min_age_days.

    Returns list of {id, roi_score, days_old} candidates (even in dry-run).
    """
    candidates = mem.store.prune_floor_candidates(roi_floor=roi_floor, min_age_days=min_age_days)
    if not dry_run:
        for c in candidates:
            try:
                mem.lifecycle.archive_memoria(c["id"])
            except Exception as exc:
                _log.warning("prune_floor: archive failed for %s: %s", c["id"], exc)
    return candidates
```

Add at top of file after existing imports:

```python
import logging as _logging
_log = _logging.getLogger(__name__)
```

Add `--skip-prune-floor` to `dream_run` options:

```python
@click.option("--skip-prune-floor", is_flag=True, help="Skip the quality-floor prune pass.")
```

Update `dream_run` signature to include `skip_prune_floor: bool`.

After the ROI decay block (step 6), add:

```python
        # 7. Quality-floor prune -----------------------------------------
        receipt["pruned_floor"] = []
        if not skip_prune_floor:
            progress.update(
                step,
                description="[7/7] prune floor — buscando memorias bajo el piso...",
                total=None,
                completed=0,
            )
            try:
                from memo.flags import flag_float, flag_int
                roi_floor = flag_float("MEMO_DREAM_PRUNE_FLOOR") or 0.15
                min_age = flag_int("MEMO_DREAM_PRUNE_MIN_AGE_DAYS") or 90
                pruned = _run_prune_floor(mem, roi_floor=roi_floor, min_age_days=min_age, dry_run=dry_run)
                receipt["pruned_floor"] = pruned
                progress.update(
                    step,
                    description=(
                        f"[7/7] prune floor [green]✓[/green]  {len(pruned)} archivadas"
                    ),
                )
            except Exception as exc:
                progress.update(step, description="[7/7] prune floor [yellow]warn[/yellow]")
                receipt["errors"].append(f"prune_floor: {type(exc).__name__}: {exc}")
            progress.advance(overall)
        else:
            progress.update(step, description="[7/7] prune floor [dim]skip[/dim]")
```

Update `total_steps` from 6 to 7 and update the `skipped` calculation to account for `skip_prune_floor`.

Add to the summary print section:

```python
    console.print(f"  quality-floor pruned:      {len(receipt['pruned_floor'])}")
```

- [ ] **Step 4: Run tests**

```bash
uv run --no-sync pytest tests/test_dream_prune_floor.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/memo/cli_dream.py tests/test_dream_prune_floor.py
git commit -m "feat: wire quality-floor prune pass into dream run pipeline"
```

---

### Task 6: Full suite + smoke test + push

**Files:**
- No new files — verification only

- [ ] **Step 1: Run full test suite**

```bash
uv run --no-sync pytest tests/ -q --tb=short
```

Expected: all pass (no regressions). If failures exist, fix before proceeding.

- [ ] **Step 2: Smoke test dream run with --dry-run**

```bash
uv run --no-sync memo dream run --dry-run
```

Expected output includes:
- `Inventario pre-dream` panel with corpus stats
- Steps 0-7 with `[dim]skip[/dim]` or `[green]✓[/green]` markers
- No errors in output

- [ ] **Step 3: Verify dream --help shows new flags**

```bash
uv run --no-sync memo dream run --help
```

Expected: `--skip-orientation`, `--skip-signal-gather`, `--skip-prune-floor` all listed.

- [ ] **Step 4: Push to master**

```bash
git push origin master
```
