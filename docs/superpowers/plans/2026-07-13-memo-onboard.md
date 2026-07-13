# `memo onboard` (wizard Day-0) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `memo onboard` — wizard de 4 pasos que colapsa el time-to-value a ~5 minutos: (1) recall hook + shims, (2) backfill de transcripts ya en disco, (3) punteros de import, (4) "3 cosas que ya sé de vos".

**Architecture:** Un solo comando click (`cli_onboard.py`) que ORQUESTA piezas ya shipped — no reimplementa nada: `wire_recall_hook` (idempotente, `cli_hooks.py:161`), `install_shims_cmd` vía `ctx.invoke`, `mine_transcripts` (`transcript_miner.py:209` — ya resumable, con dry-run de estimación y redact default-ON), y un lector de memorias recientes por disco (markdown = source of truth, cero MLX). Guard de interactividad idéntico a `memo sync setup` (`cli_sync.py:533`); `--yes` habilita el modo headless.

**Tech Stack:** Python 3.11+, click, rich, pytest (CliRunner).

**Spec:** `docs/superpowers/specs/2026-07-13-roadmap-gamechangers-design.md` §3 (memo onboard). El MCPB Node bootstrap del §3 es un plan APARTE (no entra acá).

## Global Constraints

- Flag nuevo: `MEMO_ONBOARD_BACKFILL_DAYS` (int, default **90**, min 1, max 3650) en `src/memo/flags_capture.py`.
- El wizard NO llama al LLM ni al embedder directamente; el único paso pesado es `mine_transcripts`, que ya gestiona su propio helper-LLM y redact (`MEMO_REDACT_SECRETS` default-ON en `flags_behavior.py`).
- `MEMO_NONINTERACTIVE` / stdin-no-tty ⇒ sin prompts jamás (patrón `cli_sync.py:533-536`): sin `--yes` imprime guía y sale 0; con `--yes` corre todo sin preguntar.
- Working tree compartido: `git add` SOLO paths explícitos, nunca `-A`. Lint solo archivos propios.
- Tests: CliRunner siempre con `env={"MEMO_NONINTERACTIVE": "1", "MEMO_DATA_DIR": ..., "MEMO_STATE_DIR": ..., "MEMO_VAULT_PATH": ..., "MEMO_EMBEDDER_VIA_DAEMON": "0", "MEMO_SKIP_MODEL_VERSION_CHECK": "1"}` (patrón `tests/test_cli_mcp_surface_smoke.py:11-31`); nunca tocar el vault real ni `~/.claude` real (stubear `wire_recall_hook`).
- Comandos: `uv run --no-sync pytest tests/test_cli_onboard.py -v` · `uv run --no-sync ruff check src/memo/cli_onboard.py tests/test_cli_onboard.py` · `uv run --no-sync mypy src/memo/cli_onboard.py`.
- Código/comentarios en inglés; strings de UX del wizard en español (precedente: `sync setup`).

## File Structure

| Archivo | Responsabilidad |
|---|---|
| Create `src/memo/cli_onboard.py` | El wizard completo: guard de interactividad, 4 pasos como helpers testeables, resumen |
| Modify `src/memo/flags_capture.py` | FlagSpec `MEMO_ONBOARD_BACKFILL_DAYS` |
| Modify `src/memo/cli.py` | import + `cli.add_command(onboard)` |
| Test `tests/test_cli_onboard.py` | Unit de helpers + CliRunner del wizard (deps stubbeadas) |

---

### Task 1: Flag de ventana de backfill

**Files:**
- Modify: `src/memo/flags_capture.py` (dentro de la tupla `SPECS`)
- Test: `tests/test_cli_onboard.py` (nuevo)

**Interfaces:**
- Produces: `MEMO_ONBOARD_BACKFILL_DAYS` (int, default 90) resoluble vía `flag_int(...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_onboard.py
"""Tests for the `memo onboard` Day-0 wizard."""
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }


def test_onboard_backfill_days_flag_registered():
    from memo.flags import REGISTRY

    spec = REGISTRY["MEMO_ONBOARD_BACKFILL_DAYS"]
    assert spec.default == 90
```

- [ ] **Step 2: Run to verify FAIL**

Run: `uv run --no-sync pytest tests/test_cli_onboard.py::test_onboard_backfill_days_flag_registered -v`
Expected: FAIL con `KeyError: 'MEMO_ONBOARD_BACKFILL_DAYS'`.

- [ ] **Step 3: Implement**

En `src/memo/flags_capture.py`, dentro de `SPECS` (patrón verbatim de `MEMO_CRUSHER_ROWS_KEEP_RATIO`, `flags_capture.py:32-41`):

```python
    _spec(
        "MEMO_ONBOARD_BACKFILL_DAYS",
        "int",
        90,
        "capture",
        "Day-0 backfill window for `memo onboard`: how many days of transcript "
        "history to mine on first run.",
        min_val=1,
        max_val=3650,
    ),
```

- [ ] **Step 4: Run to verify PASS**

Run: `uv run --no-sync pytest tests/test_cli_onboard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memo/flags_capture.py tests/test_cli_onboard.py
git commit -m "feat(onboard): MEMO_ONBOARD_BACKFILL_DAYS flag (default 90)"
```

---

### Task 2: Helper `_recent_memories` ("3 cosas que ya sé de vos")

**Files:**
- Create: `src/memo/cli_onboard.py` (módulo con el helper; el comando llega en Task 3)
- Test: `tests/test_cli_onboard.py` (extender)

**Interfaces:**
- Produces: `_recent_memories(memory_dir: Path, n: int = 3) -> list[dict[str, str]]` — las n memorias más nuevas por mtime, excluyendo buckets `_*` (`_profile`, `_chronicle`); cada item `{"title": ..., "file": ...}` con title = primer heading `# ` o el stem del archivo. Cero MLX, cero Memory.

- [ ] **Step 1: Write the failing test**

```python
import os
import time


def test_recent_memories_orders_by_mtime_and_skips_buckets(tmp_path):
    from memo.cli_onboard import _recent_memories

    root = tmp_path / "mem"
    root.mkdir()
    for i, name in enumerate(["old", "mid", "new"]):
        p = root / f"{name}.md"
        p.write_text(f"---\nid: {'a' * 32}\n---\n# titulo {name}\n", encoding="utf-8")
        os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))
    bucket = root / "_profile"
    bucket.mkdir()
    (bucket / "profile.md").write_text("# not a memory\n", encoding="utf-8")

    out = _recent_memories(root, n=2)
    assert [m["title"] for m in out] == ["titulo new", "titulo mid"]


def test_recent_memories_empty_dir(tmp_path):
    from memo.cli_onboard import _recent_memories

    assert _recent_memories(tmp_path / "nope") == []
```

- [ ] **Step 2: Run to verify FAIL**

Run: `uv run --no-sync pytest tests/test_cli_onboard.py -k recent_memories -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'memo.cli_onboard'`.

- [ ] **Step 3: Implement**

```python
# src/memo/cli_onboard.py
"""`memo onboard` — Day-0 wizard: recall hook + transcript backfill + first briefing.

Orchestrates already-shipped pieces (wire_recall_hook, install_shims_cmd,
mine_transcripts); owns no heavy logic of its own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from memo.config import Config
from memo.flags import flag_bool, flag_int
from memo.runtime.shims import install_shims_cmd

console = Console()

DEFAULT_BACKFILL_DAYS = 90


def _recent_memories(memory_dir: Path, n: int = 3) -> list[dict[str, str]]:
    """Newest saved memories by mtime — the '3 cosas que ya sé de vos'.

    Disk-only on purpose: markdown is the source of truth and this must not
    cold-load MLX inside a first-run wizard."""
    if not memory_dir.exists():
        return []
    files = [
        p
        for p in memory_dir.rglob("*.md")
        if not any(part.startswith("_") for part in p.relative_to(memory_dir).parts)
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, str]] = []
    for p in files[:n]:
        head = p.read_text(encoding="utf-8", errors="ignore")[:1000]
        title = next(
            (ln.lstrip("# ").strip() for ln in head.splitlines() if ln.startswith("# ")),
            p.stem,
        )
        out.append({"title": title, "file": p.name})
    return out
```

- [ ] **Step 4: Run to verify PASS**

Run: `uv run --no-sync pytest tests/test_cli_onboard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memo/cli_onboard.py tests/test_cli_onboard.py
git commit -m "feat(onboard): _recent_memories disk reader (no MLX in the wizard)"
```

---

### Task 3: Comando `memo onboard` — guard, pasos y registro

**Files:**
- Modify: `src/memo/cli_onboard.py` (agregar helpers de paso + comando)
- Modify: `src/memo/cli.py` — import junto a los demás (`cli.py:30-72`) + `cli.add_command(onboard)` (`cli.py:348-486`)
- Test: `tests/test_cli_onboard.py` (extender)

**Interfaces:**
- Consumes:
  - `wire_recall_hook(claude_dir) -> {"action": "added"|"updated"|"already", "command": ...}` y `_claude_dir()` de `memo.cli_hooks` (`cli_hooks.py:161-204`).
  - `install_shims_cmd` (click command, `runtime/shims.py:229`) vía `ctx.invoke` (usa sus defaults).
  - `mine_transcripts(root=None, *, since_days, file_limit=None, dry_run, debug=False, progress_cb=None) -> dict` (`transcript_miner.py:209`) — summary keys: `status, root, files_total, files_processed, files_skipped, candidates, saved, skipped_dup, dry_run`.
  - `flag_bool("MEMO_NONINTERACTIVE")`, `flag_int("MEMO_ONBOARD_BACKFILL_DAYS")`.
- Produces: comando top-level `memo onboard [--yes] [--days N] [--dry-run] [--json]`; helpers `_step_hook() -> dict` y `_step_backfill(days, *, dry_run) -> dict` (imports diferidos adentro para que los tests los stubbeen en su módulo de origen).

- [ ] **Step 1: Write the failing tests**

```python
def _stub_shims(monkeypatch):
    import click as _click

    @_click.command(name="install-shims-stub")
    def _noop() -> None:
        pass

    monkeypatch.setattr("memo.cli_onboard.install_shims_cmd", _noop)


def _fake_memories(tmp_path, n=3):
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (data / f"m{i}.md").write_text(
            f"---\nid: {'a' * 32}\n---\n# aprendizaje {i}\n", encoding="utf-8"
        )


def test_onboard_noninteractive_without_yes_prints_guidance(tmp_path, monkeypatch):
    from memo.cli import cli

    calls = []
    monkeypatch.setattr("memo.cli_hooks.wire_recall_hook", lambda *a, **k: calls.append(1) or {})
    result = CliRunner().invoke(cli, ["onboard"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "--yes" in result.output
    assert calls == []  # ningún paso corrió


def test_onboard_yes_runs_all_steps(tmp_path, monkeypatch):
    from memo.cli import cli

    _stub_shims(monkeypatch)
    _fake_memories(tmp_path)
    monkeypatch.setattr(
        "memo.cli_hooks.wire_recall_hook",
        lambda *a, **k: {"action": "added", "command": "memo recall-hook"},
    )
    mined = []

    def _fake_mine(root=None, **kw):
        mined.append(kw)
        return {"status": "ok", "files_total": 5, "candidates": 12, "saved": 7, "skipped_dup": 2}

    monkeypatch.setattr("memo.transcript_miner.mine_transcripts", _fake_mine)

    result = CliRunner().invoke(cli, ["onboard", "--yes"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert len(mined) == 1 and mined[0]["dry_run"] is False and mined[0]["since_days"] == 90
    assert "7 memorias" in result.output
    assert "aprendizaje 2" in result.output  # las 3 cosas que ya sé de vos
    assert "memo import whatsapp" in result.output


def test_onboard_days_override_and_dry_run(tmp_path, monkeypatch):
    from memo.cli import cli

    _stub_shims(monkeypatch)
    monkeypatch.setattr(
        "memo.cli_hooks.wire_recall_hook", lambda *a, **k: {"action": "already", "command": "x"}
    )
    mined = []

    def _fake_mine(root=None, **kw):
        mined.append(kw)
        return {"status": "ok", "files_total": 2, "candidates": 3, "saved": 0,
                "skipped_dup": 0, "dry_run": True}

    monkeypatch.setattr("memo.transcript_miner.mine_transcripts", _fake_mine)

    result = CliRunner().invoke(
        cli, ["onboard", "--yes", "--days", "7", "--dry-run"], env=_env(tmp_path)
    )
    assert result.exit_code == 0, result.output
    assert len(mined) == 1 and mined[0]["dry_run"] is True and mined[0]["since_days"] == 7


def test_onboard_json_summary(tmp_path, monkeypatch):
    import json as _json

    from memo.cli import cli

    _stub_shims(monkeypatch)
    monkeypatch.setattr(
        "memo.cli_hooks.wire_recall_hook", lambda *a, **k: {"action": "added", "command": "x"}
    )
    monkeypatch.setattr(
        "memo.transcript_miner.mine_transcripts",
        lambda root=None, **kw: {"status": "ok", "files_total": 0, "candidates": 0,
                                 "saved": 0, "skipped_dup": 0},
    )
    result = CliRunner().invoke(cli, ["onboard", "--yes", "--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    payload = _json.loads(result.output[result.output.index("{"):])
    assert payload["hook"]["action"] == "added"
    assert payload["backfill"]["status"] == "ok"
    assert isinstance(payload["memories"], list)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `uv run --no-sync pytest tests/test_cli_onboard.py -v`
Expected: los 4 tests nuevos FAIL — `No such command 'onboard'` (exit_code 2).

- [ ] **Step 3: Implement — helpers + comando + registro**

Agregar a `src/memo/cli_onboard.py`:

```python
def _step_hook() -> dict[str, Any]:
    # Deferred so tests stub it at its home module (memo.cli_hooks).
    from memo.cli_hooks import _claude_dir, wire_recall_hook

    return wire_recall_hook(_claude_dir())


def _step_backfill(days: int, *, dry_run: bool) -> dict[str, Any]:
    # Deferred: transcript_miner drags capture deps; stubs also target its home.
    from memo.transcript_miner import mine_transcripts

    return mine_transcripts(since_days=days, dry_run=dry_run)


@click.command(name="onboard")
@click.option("--yes", is_flag=True, help="Correr todos los pasos sin preguntar.")
@click.option(
    "--days", type=int, default=None,
    help="Ventana de backfill en días (default: MEMO_ONBOARD_BACKFILL_DAYS=90).",
)
@click.option("--dry-run", is_flag=True, help="Estimar el backfill sin guardar nada.")
@click.option("--json", "as_json", is_flag=True, help="Emitir resumen JSON al final.")
@click.pass_context
def onboard(ctx: click.Context, yes: bool, days: int | None, dry_run: bool, as_json: bool) -> None:
    """Wizard Day-0: hook de recall + backfill de historial + primer briefing."""
    cfg = Config.from_env()
    interactive = not (flag_bool("MEMO_NONINTERACTIVE") or not sys.stdin.isatty())
    if not yes and not interactive:
        # Never prompt from hooks / pipes — mirror of `memo sync setup`.
        console.print(
            "Corré [cyan]memo onboard[/cyan] en una terminal interactiva, "
            "o [cyan]memo onboard --yes[/cyan] para el modo automático."
        )
        return

    summary: dict[str, Any] = {}

    # 1/4 — recall hook + shims (both idempotent)
    if yes or click.confirm("1/4 · ¿Instalar el recall hook de Claude Code?", default=True):
        summary["hook"] = _step_hook()
        console.print(f"[green]✓[/green] hook: {summary['hook'].get('action')}")
        ctx.invoke(install_shims_cmd)

    # 2/4 — transcript backfill (Day-0 corpus from history already on disk)
    window = days if days is not None else (flag_int("MEMO_ONBOARD_BACKFILL_DAYS") or DEFAULT_BACKFILL_DAYS)
    if dry_run:
        summary["backfill"] = _step_backfill(window, dry_run=True)
        console.print(
            f"Backfill (dry-run): {summary['backfill'].get('files_total', 0)} transcripts, "
            f"~{summary['backfill'].get('candidates', 0)} candidatos ({window} días)."
        )
    elif yes:
        summary["backfill"] = _step_backfill(window, dry_run=False)
        console.print(
            f"[green]✓[/green] backfill: {summary['backfill'].get('saved', 0)} memorias nuevas "
            f"({summary['backfill'].get('skipped_dup', 0)} duplicados salteados)"
        )
    else:
        estimate = _step_backfill(window, dry_run=True)
        console.print(
            f"Backfill: {estimate.get('files_total', 0)} transcripts, "
            f"~{estimate.get('candidates', 0)} candidatos ({window} días)."
        )
        if click.confirm(f"2/4 · ¿Minar {window} días de historial ahora?", default=True):
            summary["backfill"] = _step_backfill(window, dry_run=False)
            console.print(
                f"[green]✓[/green] backfill: {summary['backfill'].get('saved', 0)} memorias nuevas"
            )
        else:
            summary["backfill"] = {"status": "skipped"}

    # 3/4 — other sources (pointers only; each has its own command)
    console.print(
        "3/4 · Otras fuentes: [cyan]memo import whatsapp[/cyan] · "
        "[cyan]memo import json[/cyan] · [cyan]memo import csv[/cyan]"
    )

    # 4/4 — first briefing: newest memories, straight from disk
    known = _recent_memories(Path(cfg.memory_dir))
    summary["memories"] = known
    if known:
        console.print("\n4/4 · [bold]3 cosas que ya sé de vos:[/bold]")
        for m in known:
            console.print(f"  · {m['title']}")
    else:
        console.print("4/4 · Todavía no hay memorias — van a aparecer solas mientras trabajás.")
    console.print("\nListo. Reiniciá la sesión de Claude Code para que el hook arranque.")

    if as_json:
        click.echo(json.dumps(summary, indent=2, ensure_ascii=False))
```

En `src/memo/cli.py` — import junto a los demás (zona `cli.py:30-72`):

```python
from memo.cli_onboard import onboard
```

y registro (zona `cli.py:348-486`):

```python
cli.add_command(onboard)
```

- [ ] **Step 4: Run to verify PASS**

Run: `uv run --no-sync pytest tests/test_cli_onboard.py tests/test_cli_mcp_surface_smoke.py -v`
Expected: PASS (el smoke de superficie recoge `onboard --help` automáticamente).

- [ ] **Step 5: Commit**

```bash
git add src/memo/cli_onboard.py src/memo/cli.py tests/test_cli_onboard.py
git commit -m "feat(onboard): Day-0 wizard — hook + backfill + import pointers + first briefing"
```

---

### Task 4: Gate final

**Files:** ninguno nuevo — verificación.

- [ ] **Step 1: Suite dirigida + lint + types**

```bash
uv run --no-sync pytest tests/test_cli_onboard.py -v
uv run --no-sync ruff check src/memo/cli_onboard.py tests/test_cli_onboard.py
uv run --no-sync mypy src/memo/cli_onboard.py
```
Expected: verde.

- [ ] **Step 2: Suite completa (merge-drift check)**

Run: `uv run --no-sync pytest tests/ -x -q`
Expected: verde (flakes de snapshot TUI pre-existentes excluidos).

- [ ] **Step 3: Smoke manual del camino real (SIN --yes, interactivo)**

```bash
uv run --no-sync memo onboard --dry-run
```
Expected: guard noninteractive NO dispara (TTY real); muestra estimación de backfill sin guardar nada; ningún traceback. Cancelable en cada paso.

- [ ] **Step 4: Commit final si hubo fixes**

```bash
git add <solo archivos tocados por fixes>
git commit -m "test(onboard): green full suite"
```

## Medición post-ship (del spec — observabilidad ya existente, sin código nuevo)

- Funnel Day-0: memorias creadas en día 0 + tiempo-hasta-primer-grounded-recall — ambos observables en `grounding.log` / token ledger.
- Si el corpus backfilleado mete ruido: correr `memo eval recall --gate`; si noise@K sube, bajar `MEMO_ONBOARD_BACKFILL_DAYS` o filtrar por tipo antes de shippear release.
