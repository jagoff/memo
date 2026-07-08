# Trust + Adoption Doctor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only `memo usefulness doctor` diagnostic that reports whether memo is consulted, attributed, grounded, and using trustworthy memories.

**Architecture:** Add a focused `memo.usefulness_doctor` module that derives a structured report from existing dashboard logs, `consult_breakdown()`, `recall_health()`, and store signal rows. Wire it into the existing `memo usefulness` CLI as a command group with a `doctor` subcommand while keeping the current `memo usefulness` output stable.

**Tech Stack:** Python 3.13, Click, stdlib JSON/SQLite via existing `Config`/`Memory`/`VecStore`, pytest, ruff, mypy.

## Global Constraints

- The doctor is read-only: no memory mutation, no ranking changes, no recall-hook changes.
- No MLX imports or LLM calls.
- No new persistence schema.
- Do not touch the real vault or default state dir in tests; use isolated `Config` or `CliRunner` env values.
- Preserve existing `memo usefulness --json` shape.
- Keep `docs/superpowers/` staging explicit because it is ignored locally.
- Do not stage unrelated local changes in `src/memo/dream_*` or `tests/test_dream_health.py`.

---

## File Structure

- Create `src/memo/usefulness_doctor.py`
  - Owns all report-building logic.
  - Public entrypoint: `build_report(cfg: Config, *, limit: int = 500) -> dict[str, Any]`.
  - Public formatter: `format_text_report(report: dict[str, Any]) -> str`.
  - Private helpers analyze adoption, grounding, support-count sparsity, and grounded low-trust memories.
- Modify `src/memo/cli_usefulness.py`
  - Convert the current command into a Click group with `invoke_without_command=True` so `memo usefulness` keeps working.
  - Add `memo usefulness doctor [--json] [--limit N]`.
- Create `tests/test_usefulness_doctor.py`
  - Unit tests for report generation and text formatting.
- Modify `tests/test_usefulness.py`
  - Add CLI regression tests that current `memo usefulness --json` remains stable and the new doctor command is registered.

---

### Task 1: Core Adoption Doctor

**Files:**
- Create: `src/memo/usefulness_doctor.py`
- Create: `tests/test_usefulness_doctor.py`

**Interfaces:**
- Consumes: `memo.dashboard.consult_breakdown(state_dir, limit=...)`, `memo.dashboard.recall_health(state_dir, limit=...)`, `memo.dashboard.read_recall_log(state_dir, limit=...)`, `memo.dashboard.consumer_label(row)`, `memo.config.Config`.
- Produces:
  - `build_report(cfg: Config, *, limit: int = 500) -> dict[str, Any]`
  - `format_text_report(report: dict[str, Any]) -> str`

- [ ] **Step 1: Write failing adoption tests**

Create `tests/test_usefulness_doctor.py` with this content:

```python
"""Trust + adoption doctor report derivation."""

from __future__ import annotations

from pathlib import Path

from memo.config import Config
from memo.dashboard import append_recall_log
from memo.usefulness_doctor import build_report, format_text_report


def _cfg(tmp_path: Path) -> Config:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    cfg = Config(data_dir=data_dir, state_dir=state_dir)
    cfg.ensure_dirs()
    return cfg


def test_doctor_reports_silent_consumers(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    append_recall_log(
        cfg.state_dir,
        prompt="what did we decide about memo",
        hits=[{"id": "a" * 8, "score": 0.91, "title": "Memo decision"}],
        via="daemon",
        client="codex",
    )

    report = build_report(cfg, limit=100)

    assert report["verdict"] == "degraded"
    silent = [i for i in report["adoption"] if i["id"] == "silent_consumers"]
    assert silent
    assert "memflow" in silent[0]["evidence"]["silent"]
    assert any("source" in a["command"] for a in report["actions"])


def test_doctor_reports_unattributed_mcp_consults(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    append_recall_log(
        cfg.state_dir,
        prompt="search from an mcp client",
        hits=[{"id": "b" * 8, "score": 0.82, "title": "Search result"}],
        via="mcp:search",
    )

    report = build_report(cfg, limit=100)

    item = next(i for i in report["adoption"] if i["id"] == "unattributed_consults")
    assert item["severity"] == "warning"
    assert item["evidence"]["count"] == 1
    assert item["action"] == 'Pass source="<client>" on memo read tool calls.'


def test_doctor_text_report_is_action_oriented(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    append_recall_log(
        cfg.state_dir,
        prompt="what did we decide about retrieval",
        hits=[{"id": "c" * 8, "score": 0.86, "title": "Retrieval decision"}],
        via="daemon",
        client="codex",
    )

    text = format_text_report(build_report(cfg, limit=100))

    assert "memo trust + adoption doctor" in text
    assert "verdict:" in text
    assert "adoption" in text
    assert "action:" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --no-sync pytest tests/test_usefulness_doctor.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'memo.usefulness_doctor'`.

- [ ] **Step 3: Implement the minimal adoption report**

Create `src/memo/usefulness_doctor.py`:

```python
"""Read-only trust + adoption diagnostics for `memo usefulness doctor`."""

from __future__ import annotations

from typing import Any

from memo.config import Config
from memo.dashboard import (
    consult_breakdown,
    consumer_label,
    read_recall_log,
    recall_health,
)

DiagnosticItem = dict[str, Any]
ActionItem = dict[str, Any]


def _item(
    *,
    id: str,
    severity: str,
    status: str,
    message: str,
    evidence: dict[str, Any] | None = None,
    action: str,
) -> DiagnosticItem:
    return {
        "id": id,
        "severity": severity,
        "status": status,
        "message": message,
        "evidence": evidence or {},
        "action": action,
    }


def _action(id: str, command: str, reason: str) -> ActionItem:
    return {"id": id, "command": command, "reason": reason}


def _adoption_items(cfg: Config, *, limit: int) -> tuple[list[DiagnosticItem], list[ActionItem]]:
    breakdown = consult_breakdown(cfg.state_dir, limit=limit)
    health = recall_health(cfg.state_dir, limit=limit)
    rows = read_recall_log(cfg.state_dir, limit=limit)
    items: list[DiagnosticItem] = []
    actions: list[ActionItem] = []

    silent = list(breakdown.get("silent") or [])
    consumers = list(breakdown.get("consumers") or [])
    if not consumers:
        items.append(
            _item(
                id="no_consults",
                severity="critical",
                status="silent",
                message="No memo consults were recorded in the sampled window.",
                evidence={"sampled": breakdown.get("sampled", 0)},
                action="Run a memo-enabled agent session or check MCP/hook installation.",
            )
        )
        actions.append(
            _action(
                "check_install",
                "memo doctor --strict-runtime",
                "No consult telemetry exists.",
            )
        )
    elif silent:
        items.append(
            _item(
                id="silent_consumers",
                severity="warning",
                status="degraded",
                message="Expected memo consumers have zero recent consults.",
                evidence={"silent": silent},
                action='Configure those clients to call memo and pass source="<client>".',
            )
        )
        actions.append(
            _action(
                "source_attribution",
                'Pass source="<client>" on memo read tool calls.',
                "Silent or unattributed consumers cannot prove adoption.",
            )
        )

    unattributed = [
        r
        for r in rows
        if consumer_label(r) in {"mcp:unknown", "unknown"}
        or ((r.get("via") or "").startswith("mcp:") and not (r.get("source") or r.get("client")))
    ]
    if unattributed:
        items.append(
            _item(
                id="unattributed_consults",
                severity="warning",
                status="degraded",
                message="Some memo consults lack explicit source attribution.",
                evidence={"count": len(unattributed), "sample": unattributed[:3]},
                action='Pass source="<client>" on memo read tool calls.',
            )
        )
        actions.append(
            _action(
                "add_source",
                'Pass source="<client>" on memo read tool calls.',
                "Unattributed consults collapse into generic consumers.",
            )
        )

    grounded_rate = health.get("grounded_rate")
    if consumers and grounded_rate is None:
        items.append(
            _item(
                id="grounding_unmeasured",
                severity="warning",
                status="unmeasured",
                message="Memo is consulted, but grounding has not been measured yet.",
                evidence={
                    "sampled": health.get("sampled"),
                    "measured_turns": health.get("measured_turns"),
                    "surfaced_turns": health.get("surfaced_turns"),
                },
                action="Let Stop-hook grounding accumulate or inspect grounding diagnostics.",
            )
        )
    elif isinstance(grounded_rate, (int, float)) and float(grounded_rate) < 0.1:
        items.append(
            _item(
                id="low_grounding",
                severity="warning",
                status="degraded",
                message="Memo is consulted, but few surfaced memories are grounded.",
                evidence={"grounded_rate": grounded_rate},
                action="Run memo debug-recall on recent low-quality prompts.",
            )
        )

    return items, actions


def _derive_verdict(adoption: list[DiagnosticItem], trust: list[DiagnosticItem]) -> str:
    items = adoption + trust
    if any(i["status"] == "silent" for i in items):
        return "silent"
    if any(i["severity"] == "critical" for i in items):
        return "untrusted"
    if any(i["severity"] == "warning" for i in items):
        return "degraded"
    if not items:
        return "healthy"
    return "healthy"


def build_report(cfg: Config, *, limit: int = 500) -> dict[str, Any]:
    """Build a read-only trust + adoption report."""
    adoption, actions = _adoption_items(cfg, limit=limit)
    trust: list[DiagnosticItem] = []
    verdict = _derive_verdict(adoption, trust)
    return {
        "verdict": verdict,
        "adoption": adoption,
        "trust": trust,
        "actions": actions,
        "summary": {
            "limit": int(limit),
            "adoption_items": len(adoption),
            "trust_items": len(trust),
        },
    }


def format_text_report(report: dict[str, Any]) -> str:
    """Render a compact human report."""
    lines = [
        "memo trust + adoption doctor",
        "",
        f"verdict: {report.get('verdict', 'unknown')}",
        "",
        "adoption",
    ]
    adoption = list(report.get("adoption") or [])
    if not adoption:
        lines.append("  - healthy: consumers are attributed and grounding is not degraded")
    for item in adoption:
        lines.append(f"  - {item['id']}: {item['status']} ({item['severity']})")
        lines.append(f"    {item['message']}")
        lines.append(f"    action: {item['action']}")

    lines.extend(["", "trust"])
    trust = list(report.get("trust") or [])
    if not trust:
        lines.append("  - no trust warnings in sampled data")
    for item in trust:
        lines.append(f"  - {item['id']}: {item['status']} ({item['severity']})")
        lines.append(f"    {item['message']}")
        lines.append(f"    action: {item['action']}")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run:

```bash
uv run --no-sync pytest tests/test_usefulness_doctor.py -v
```

Expected: PASS.

- [ ] **Step 5: Lint the new files**

Run:

```bash
uv run --no-sync ruff check src/memo/usefulness_doctor.py tests/test_usefulness_doctor.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/memo/usefulness_doctor.py tests/test_usefulness_doctor.py
git commit -m "feat(usefulness): add trust adoption doctor core"
```

---

### Task 2: Corpus Trust Checks

**Files:**
- Modify: `src/memo/usefulness_doctor.py`
- Modify: `tests/test_usefulness_doctor.py`

**Interfaces:**
- Consumes:
  - `Memory(cfg).store.dump_signal() -> dict[str, list[dict[str, Any]]]`
  - `Memory(cfg).store.get_batch(ids: list[str]) -> list[dict[str, Any]]`
  - `memo.dashboard.read_grounding_log(state_dir, limit=...)`
- Produces:
  - `trust` report items for support-count starvation and grounded low-trust memories.
  - `summary["memory_health_rows"]`, `summary["support_count_positive"]`, `summary["grounded_memory_ids"]`.

- [ ] **Step 1: Add failing trust tests**

Append to `tests/test_usefulness_doctor.py`:

```python
from memo.dashboard import append_grounding_log
from memo.memory import Memory


def test_doctor_reports_support_count_starvation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    mem = Memory(cfg)
    ids: list[str] = []
    for n in range(25):
        rec = mem.save(content=f"fact {n}", title=f"Fact {n}", defer_embed=True)
        ids.append(rec.id)
    mem.store.set_confidence_batch([(id_, 1.0) for id_ in ids])

    report = build_report(cfg, limit=100)

    item = next(i for i in report["trust"] if i["id"] == "support_count_starvation")
    assert item["severity"] == "warning"
    assert item["evidence"]["memory_health_rows"] == 25
    assert item["evidence"]["support_count_positive"] == 0


def test_doctor_reports_invalidated_grounded_memory(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    mem = Memory(cfg)
    rec = mem.save(
        content="usamos webpack",
        title="Bundler",
        tags=["_invalidated"],
        extra={"invalidated_reason": "migramos a vite"},
        defer_embed=True,
    )
    append_grounding_log(
        cfg.state_dir,
        session_id="s1",
        turn=1,
        recall_id=rec.id[:8],
        used_score=0.9,
        method="test",
    )

    report = build_report(cfg, limit=100)

    item = next(i for i in report["trust"] if i["id"] == "untrusted_memories_grounded")
    assert item["severity"] == "critical"
    assert item["evidence"]["count"] == 1
    assert item["evidence"]["memories"][0]["id"] == rec.id[:8]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --no-sync pytest tests/test_usefulness_doctor.py -v -k "support_count or invalidated"
```

Expected: FAIL because the `trust` checks are not implemented.

- [ ] **Step 3: Add trust readers and checks**

Modify `src/memo/usefulness_doctor.py`.

Add imports:

```python
from memo.dashboard import read_grounding_log
from memo.memory import Memory
```

Add helper functions above `build_report()`:

```python
def _health_rows(mem: Memory) -> list[dict[str, Any]]:
    try:
        return list((mem.store.dump_signal().get("memory_health") or []))
    except Exception:
        return []


def _grounded_ids(cfg: Config, *, limit: int) -> set[str]:
    ids: set[str] = set()
    for row in read_grounding_log(cfg.state_dir, limit=limit):
        score = row.get("used_score")
        rid = row.get("recall_id")
        if rid and isinstance(score, (int, float)) and float(score) >= 0.8:
            ids.add(str(rid))
    return ids


def _trust_items(cfg: Config, *, limit: int) -> tuple[list[DiagnosticItem], list[ActionItem], dict[str, Any]]:
    items: list[DiagnosticItem] = []
    actions: list[ActionItem] = []
    summary: dict[str, Any] = {
        "memory_health_rows": 0,
        "support_count_positive": 0,
        "grounded_memory_ids": 0,
    }
    try:
        mem = Memory(cfg)
    except Exception as exc:
        items.append(
            _item(
                id="store_unavailable",
                severity="warning",
                status="unknown",
                message="Memory store could not be opened for trust checks.",
                evidence={"error": str(exc)[:200]},
                action="Run memo doctor.",
            )
        )
        actions.append(_action("doctor", "memo doctor", "Store trust checks failed."))
        return items, actions, summary

    health_rows = _health_rows(mem)
    support_positive = sum(1 for r in health_rows if int(r.get("support_count") or 0) > 0)
    summary["memory_health_rows"] = len(health_rows)
    summary["support_count_positive"] = support_positive
    if len(health_rows) >= 20 and support_positive == 0:
        items.append(
            _item(
                id="support_count_starvation",
                severity="warning",
                status="degraded",
                message="No memory_health rows have support_count > 0.",
                evidence={
                    "memory_health_rows": len(health_rows),
                    "support_count_positive": support_positive,
                },
                action="Verify corroboration bump sites and signal export/import.",
            )
        )
        actions.append(
            _action(
                "verify_support_count",
                "uv run --no-sync pytest tests/test_support_count.py -v",
                "support_count is not accumulating.",
            )
        )

    grounded_prefixes = _grounded_ids(cfg, limit=limit)
    summary["grounded_memory_ids"] = len(grounded_prefixes)
    if grounded_prefixes:
        full_ids = mem.store.all_ids()
        resolved = [i for i in full_ids if i[:8] in grounded_prefixes]
        rows = mem.store.get_batch(resolved)
        health = mem.store.get_health_batch(resolved)
        bad: list[dict[str, Any]] = []
        for row in rows:
            rid = str(row["id"])
            tags = list(row.get("tags") or [])
            extra = row.get("extra") or {}
            conf = float((health.get(rid) or {}).get("confidence", 1.0))
            reasons: list[str] = []
            if "_invalidated" in tags or extra.get("invalidated_reason"):
                reasons.append("invalidated")
            if extra.get("superseded_by"):
                reasons.append("superseded")
            if conf < 0.5:
                reasons.append("low_confidence")
            if reasons:
                bad.append(
                    {
                        "id": rid[:8],
                        "title": row.get("title"),
                        "confidence": conf,
                        "reasons": reasons,
                    }
                )
        if bad:
            severity = "critical" if any("invalidated" in b["reasons"] or "superseded" in b["reasons"] for b in bad) else "warning"
            items.append(
                _item(
                    id="untrusted_memories_grounded",
                    severity=severity,
                    status="untrusted",
                    message="Grounded recall used memories with low-trust markers.",
                    evidence={"count": len(bad), "memories": bad[:10]},
                    action="Update stale memories, run contradiction triage, or undo incorrect invalidations.",
                )
            )
            actions.append(
                _action(
                    "triage_untrusted",
                    "memo contradict triage",
                    "Grounded memories include invalidated, superseded, or low-confidence records.",
                )
            )

    return items, actions, summary
```

Modify `build_report()` so it calls `_trust_items()`:

```python
def build_report(cfg: Config, *, limit: int = 500) -> dict[str, Any]:
    """Build a read-only trust + adoption report."""
    adoption, adoption_actions = _adoption_items(cfg, limit=limit)
    trust, trust_actions, trust_summary = _trust_items(cfg, limit=limit)
    verdict = _derive_verdict(adoption, trust)
    return {
        "verdict": verdict,
        "adoption": adoption,
        "trust": trust,
        "actions": adoption_actions + trust_actions,
        "summary": {
            "limit": int(limit),
            "adoption_items": len(adoption),
            "trust_items": len(trust),
            **trust_summary,
        },
    }
```

- [ ] **Step 4: Run trust tests**

Run:

```bash
uv run --no-sync pytest tests/test_usefulness_doctor.py -v
```

Expected: PASS.

- [ ] **Step 5: Run related signal tests**

Run:

```bash
uv run --no-sync pytest tests/test_usefulness_doctor.py tests/test_support_count.py tests/test_cli_invalidate.py -v
```

Expected: PASS.

- [ ] **Step 6: Lint Task 2 files**

Run:

```bash
uv run --no-sync ruff check src/memo/usefulness_doctor.py tests/test_usefulness_doctor.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/memo/usefulness_doctor.py tests/test_usefulness_doctor.py
git commit -m "feat(usefulness): report corpus trust diagnostics"
```

---

### Task 3: CLI Wiring

**Files:**
- Modify: `src/memo/cli_usefulness.py`
- Modify: `tests/test_usefulness.py`

**Interfaces:**
- Consumes:
  - `build_report(cfg: Config, *, limit: int = 500) -> dict[str, Any]`
  - `format_text_report(report: dict[str, Any]) -> str`
- Produces:
  - `memo usefulness` unchanged for users.
  - `memo usefulness --json` unchanged top-level keys: `recall_hook`, `by_consumer`.
  - `memo usefulness doctor [--json] [--limit N]`.

- [ ] **Step 1: Add failing CLI tests**

Append to `tests/test_usefulness.py`:

```python
import json

from click.testing import CliRunner

from memo.cli import cli


def _cli_env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_usefulness_json_shape_remains_unchanged(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["usefulness", "--json"], env=_cli_env(tmp_path))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert sorted(payload) == ["by_consumer", "recall_hook"]


def test_usefulness_doctor_command_outputs_text(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["usefulness", "doctor"], env=_cli_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "memo trust + adoption doctor" in result.output
    assert "verdict:" in result.output


def test_usefulness_doctor_json_shape(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["usefulness", "doctor", "--json"], env=_cli_env(tmp_path))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert sorted(payload) == ["actions", "adoption", "summary", "trust", "verdict"]
```

- [ ] **Step 2: Run CLI tests to verify doctor command fails**

Run:

```bash
uv run --no-sync pytest tests/test_usefulness.py -v -k "doctor or json_shape"
```

Expected: FAIL because `memo usefulness doctor` is not registered.

- [ ] **Step 3: Convert `memo usefulness` into a group**

Modify `src/memo/cli_usefulness.py`.

Replace the current command decorator and function body with this group setup while preserving the existing report logic in `_render_usefulness()`:

```python
def _render_usefulness(*, limit: int = 500, as_json: bool = False) -> None:
    cfg = Config.from_env()
    state_dir = cfg.state_dir
    health = recall_health(state_dir, limit=limit)
    breakdown = consult_breakdown(state_dir, limit=limit)

    if as_json:
        click.echo(
            _json.dumps(
                {"recall_hook": health, "by_consumer": breakdown}, ensure_ascii=False, indent=2
            )
        )
        return

    consumers = breakdown["consumers"]
    silent = breakdown["silent"]

    if not consumers:
        click.echo("No consults recorded yet — memo has not been read.")
        click.echo(
            "(The recall-hook logs Claude Code consults; MCP tools log when "
            "callers pass `source=`.)"
        )
        return

    click.echo(f"memo usefulness — {breakdown['sampled']} consults sampled\n")
    click.echo(
        f"  {'consumer':<16} {'consults':>8} {'fired':>6} {'bail':>5} "
        f"{'hit%':>6} {'strong%':>8} {'grnd%':>6} {'top':>6}  last"
    )
    click.echo("  " + "-" * 80)
    for c in consumers:
        hit = f"{c['hit_rate'] * 100:.0f}" if c["hit_rate"] is not None else "—"
        strong = (
            f"{c['strong_hit_rate'] * 100:.0f}" if c.get("strong_hit_rate") is not None else "—"
        )
        grnd = f"{c['grounded_rate'] * 100:.0f}" if c.get("grounded_rate") is not None else "—"
        top = f"{c['median_top_score']:.2f}" if c["median_top_score"] is not None else "—"
        click.echo(
            f"  {c['consumer']:<16} {c['consults']:>8} {c['fired']:>6} "
            f"{c['bailed']:>5} {hit:>6} {strong:>8} {grnd:>6} {top:>6}  {_age(c['last_seen'])}"
        )

    if silent:
        click.echo(f"\n⚠ Expected consumers with ZERO consults: {', '.join(silent)}")
        click.echo("  These layers are NOT reading memo as source-of-truth.")

    ref_rate = health.get("referenced_rate")
    if ref_rate is not None:
        click.echo(
            f"\nreferenced_rate={ref_rate} "
            f"({health.get('referenced')}/{health.get('surfaced')} surfaced memories "
            f"later fetched — lower bound on 'used')."
        )

    g_rate = health.get("grounded_rate")
    if g_rate is not None:
        click.echo(
            f"grounded_rate={g_rate} "
            f"({health.get('grounded')}/{health.get('grounded_surfaced')} surfaced memories "
            f"used in the answer — outcome-based, not just shown)."
        )

    reask = reask_stats(state_dir, limit=limit)
    if reask.get("considered"):
        click.echo(
            f"reask_avoided={reask['reask_avoided']}/{reask['considered']} "
            f"(grounded recalls the user did NOT have to ask again — see `memo roi`)."
        )

    total = breakdown["sampled"]
    n = len(consumers)
    click.echo(
        f"\nmemo consulted {total}× across {n} consumer(s); recall-hook "
        f"hit_rate={health.get('hit_rate')} strong_hit_rate={health.get('strong_hit_rate')}."
    )


@click.group(name="usefulness", invoke_without_command=True)
@click.option("--limit", default=500, show_default=True, help="Consult-log rows to sample.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.pass_context
def usefulness(ctx: click.Context, *, limit: int = 500, as_json: bool = False) -> None:
    """Report how useful memo is: who consults it, hit rate, and silent gaps."""
    if ctx.invoked_subcommand is not None:
        return
    _render_usefulness(limit=limit, as_json=as_json)
```

Then add the doctor subcommand below it:

```python
@usefulness.command(name="doctor")
@click.option("--limit", default=500, show_default=True, help="Rows/signals to sample.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def usefulness_doctor(limit: int = 500, as_json: bool = False) -> None:
    """Diagnose memo adoption and trust signals."""
    from memo.usefulness_doctor import build_report, format_text_report

    cfg = Config.from_env()
    report = build_report(cfg, limit=limit)
    if as_json:
        click.echo(_json.dumps(report, ensure_ascii=False, indent=2))
        return
    click.echo(format_text_report(report))
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
uv run --no-sync pytest tests/test_usefulness.py tests/test_usefulness_doctor.py -v
```

Expected: PASS.

- [ ] **Step 5: Smoke command help**

Run:

```bash
uv run --no-sync memo usefulness doctor --help
```

Expected: exit 0 and output includes `Diagnose memo adoption and trust signals`.

- [ ] **Step 6: Lint CLI files**

Run:

```bash
uv run --no-sync ruff check src/memo/cli_usefulness.py tests/test_usefulness.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/memo/cli_usefulness.py tests/test_usefulness.py
git commit -m "feat(cli): add usefulness doctor command"
```

---

### Task 4: Error Handling, JSON Contract, and Full Verification

**Files:**
- Modify: `src/memo/usefulness_doctor.py`
- Modify: `tests/test_usefulness_doctor.py`

**Interfaces:**
- Consumes: the Task 1-3 interfaces.
- Produces:
  - stable top-level JSON keys under partial failure
  - `summary["errors"]` and `summary["malformed_rows"]`
  - graceful verdicts: `silent`, `unknown`, `degraded`, `untrusted`, `healthy`

- [ ] **Step 1: Add failing partial-failure tests**

Append to `tests/test_usefulness_doctor.py`:

```python
import json

from memo.dashboard_logs import recall_log_path


def test_doctor_skips_malformed_recall_rows(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    path = recall_log_path(cfg.state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"via": "mcp:search"}\nnot-json\n', encoding="utf-8")

    report = build_report(cfg, limit=100)

    assert sorted(report) == ["actions", "adoption", "summary", "trust", "verdict"]
    assert report["summary"]["malformed_rows"] == 1


def test_doctor_json_is_serializable_under_empty_state(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    payload = json.dumps(build_report(cfg, limit=100), ensure_ascii=False)

    assert '"verdict"' in payload
    assert '"actions"' in payload
```

- [ ] **Step 2: Run new tests to verify malformed count fails**

Run:

```bash
uv run --no-sync pytest tests/test_usefulness_doctor.py -v -k "malformed or serializable"
```

Expected: FAIL because `malformed_rows` is not counted yet.

- [ ] **Step 3: Add malformed-row counting and stable summaries**

Modify `src/memo/usefulness_doctor.py`.

Add imports:

```python
import json
```

Add helper:

```python
def _malformed_jsonl_rows(path: Any) -> int:
    try:
        p = path
        if not p.is_file():
            return 0
        count = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                json.loads(line)
            except json.JSONDecodeError:
                count += 1
        return count
    except OSError:
        return 0
```

Modify imports from `memo.dashboard` to include:

```python
recall_log_path,
```

Modify `build_report()` summary:

```python
    malformed_rows = _malformed_jsonl_rows(recall_log_path(cfg.state_dir))
    return {
        "verdict": verdict,
        "adoption": adoption,
        "trust": trust,
        "actions": adoption_actions + trust_actions,
        "summary": {
            "limit": int(limit),
            "adoption_items": len(adoption),
            "trust_items": len(trust),
            "malformed_rows": malformed_rows,
            "errors": [],
            **trust_summary,
        },
    }
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --no-sync pytest tests/test_usefulness_doctor.py tests/test_usefulness.py tests/test_dashboard.py -v
```

Expected: PASS.

- [ ] **Step 5: Run lint and type checks**

Run:

```bash
uv run --no-sync ruff check src/memo/usefulness_doctor.py src/memo/cli_usefulness.py tests/test_usefulness_doctor.py tests/test_usefulness.py
uv run --no-sync mypy src/memo
```

Expected: both PASS.

- [ ] **Step 6: Run command smoke tests**

Run:

```bash
uv run --no-sync memo usefulness doctor --json
uv run --no-sync memo usefulness doctor
```

Expected: both exit 0. The JSON command emits keys `verdict`, `adoption`, `trust`, `actions`, and `summary`. The text command starts with `memo trust + adoption doctor`.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/memo/usefulness_doctor.py tests/test_usefulness_doctor.py
git commit -m "test(usefulness): harden doctor report contract"
```

---

## Final Verification

Run:

```bash
uv run --no-sync pytest tests/test_usefulness_doctor.py tests/test_usefulness.py tests/test_dashboard.py tests/test_support_count.py tests/test_cli_invalidate.py -v
uv run --no-sync ruff check src/memo/usefulness_doctor.py src/memo/cli_usefulness.py tests/test_usefulness_doctor.py tests/test_usefulness.py
uv run --no-sync mypy src/memo
uv run --no-sync memo usefulness doctor --json
```

Expected:

- pytest passes
- ruff passes
- mypy passes
- doctor JSON exits 0 with stable top-level keys

`memo eval recall` is not required for this plan because it does not modify ranking or the recall-hook path.

## Self-Review

- Spec coverage: adoption, attribution, grounding, support-count starvation, low-trust grounded memories, text output, JSON output, and graceful failures are covered by Tasks 1-4.
- Scope: the plan is one subsystem, a read-only diagnostic over existing signals. It does not attempt to complete all Wave 2 lifecycle work.
- Type consistency: the plan uses one public module entrypoint, `build_report(cfg: Config, *, limit: int = 500) -> dict[str, Any]`, and one formatter, `format_text_report(report: dict[str, Any]) -> str`, consistently across tests and CLI.
- Hot path: no task touches `recall_logic.py`, `cli_recall_hook.py`, embedder, reranker, or MLX imports.
