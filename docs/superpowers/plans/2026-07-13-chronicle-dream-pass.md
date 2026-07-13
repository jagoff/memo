# Chronicle (diario de ingeniería nocturno) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dream pass nocturno que escribe un diario markdown humano por día (`<memory_dir>/_chronicle/YYYY-MM-DD.md`) desde datos que memo ya tiene (episodios, memorias nuevas, grounding, receipt), con provenance obligatorio por id y filtro anti-fabricación.

**Architecture:** Clon estructural de `dream_profile.py` (dream pass → markdown en bucket `_` bajo `memory_dir`, default-OFF, nunca lanza excepciones, receipt registra errores). Colección de hechos determinística (sin LLM) → un solo call LLM local con prompt "cite o descartá" → filtro post-hoc que borra bullets sin id citado válido → escritura atómica. Se escribe bajo `cfg.memory_dir` (no `vault_path` directo): cuando `memories_in_vault` está ON resuelve al vault, cuando está OFF resuelve a `data_dir` — esto implementa el fallback sin vault que exige el spec, gratis. Archivos sin key `id:` en frontmatter → reindex los ignora (mismo mecanismo que `_profile`).

**Tech Stack:** Python 3.11+, click, rich, MLX local vía `chat_with_timeout` (import diferido), pytest.

**Spec:** `docs/superpowers/specs/2026-07-13-roadmap-gamechangers-design.md` §2 (Chronicle).

## Global Constraints

- Flags nuevos nacen **default-OFF**: `MEMO_DREAM_CHRONICLE_ENABLED` (bool, False), `MEMO_CHRONICLE_WEEKLY` (bool, False). Registrar en `src/memo/flags_misc.py` con `_spec(...)`.
- `run_chronicle_pass` **nunca lanza** — retorna `status="error"` y el caller lo registra en `receipt["errors"]` (contrato idéntico a `run_profile_pass`, ver `src/memo/cli_dream_passes.py:312`).
- Imports de LLM **siempre diferidos** (dentro de funciones): `from memo.memory.record import chat_with_timeout`. Nunca a nivel módulo.
- Nada de este plan entra al recall hook (budget 5s intocado).
- Working tree compartido: `git add` SOLO con paths explícitos, nunca `-A`/`-a`. Lint solo archivos propios.
- Tests: nunca `Config.from_env()` sin controlar env; usar `_Cfg` fake (patrón `tests/test_dream_profile.py:8-14`) o `tmp_cfg`; nunca tocar el vault real. CliRunner siempre con `env={"MEMO_NONINTERACTIVE": "1", "MEMO_DATA_DIR": ..., "MEMO_STATE_DIR": ..., "MEMO_VAULT_PATH": ..., "MEMO_EMBEDDER_VIA_DAEMON": "0", "MEMO_SKIP_MODEL_VERSION_CHECK": "1"}`.
- Comandos de test: `uv run --no-sync pytest tests/<file>.py -v`. Lint: `uv run --no-sync ruff check <solo tus archivos>`. Types: `uv run --no-sync mypy src/memo/dream_chronicle.py src/memo/cli_chronicle.py`.
- Strings de código/comentarios en inglés (convención repo); output de CLI puede ser español (precedente: `cli_sync.py` setup).

## File Structure

| Archivo | Responsabilidad |
|---|---|
| Create `src/memo/dream_chronicle.py` | Todo el pase: paths, colección de hechos, filtro de provenance, LLM narrate, render, weekly rollup, `run_chronicle_pass` |
| Create `src/memo/cli_chronicle.py` | `memo chronicle` — lector on-demand de crónicas ya escritas |
| Modify `src/memo/flags_misc.py` | 2 FlagSpecs nuevos |
| Modify `src/memo/cli_dream.py` | Bloque gateado en el pipeline nocturno + subcomando `memo dream chronicle` |
| Modify `src/memo/cli.py` | Registrar `chronicle_cmd` |
| Test `tests/test_dream_chronicle.py` | Unit del pase completo (LLM stubbeado) |
| Test `tests/test_cli_chronicle.py` | CLI lector |

---

### Task 1: Flags

**Files:**
- Modify: `src/memo/flags_misc.py` (agregar al final de la tupla `SPECS`, antes del cierre)
- Test: `tests/test_dream_chronicle.py` (nuevo)

**Interfaces:**
- Produces: flags `MEMO_DREAM_CHRONICLE_ENABLED` (bool, default False) y `MEMO_CHRONICLE_WEEKLY` (bool, default False) resolubles vía `flag_bool(...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dream_chronicle.py
"""Tests for the nightly chronicle dream pass."""
from __future__ import annotations


def test_chronicle_flags_registered_default_off():
    from memo.flags import REGISTRY

    for name in ("MEMO_DREAM_CHRONICLE_ENABLED", "MEMO_CHRONICLE_WEEKLY"):
        assert name in REGISTRY
        assert REGISTRY[name].default is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py::test_chronicle_flags_registered_default_off -v`
Expected: FAIL con `AssertionError` (o KeyError) — flags no registrados.

- [ ] **Step 3: Implement — agregar los 2 specs**

En `src/memo/flags_misc.py`, dentro de la tupla `SPECS`, siguiendo el patrón exacto de los `_spec` existentes (p.ej. `flags_misc.py:290` `MEMO_PROMPT_CACHE`):

```python
    _spec(
        "MEMO_DREAM_CHRONICLE_ENABLED",
        "bool",
        False,
        "misc",
        "Nightly chronicle dream pass: write a human engineering diary for the "
        "day under <memory_dir>/_chronicle/, with per-id provenance. Default off.",
    ),
    _spec(
        "MEMO_CHRONICLE_WEEKLY",
        "bool",
        False,
        "misc",
        "Also regenerate the ISO-week rollup file (week-YYYY-Www.md) after each "
        "nightly chronicle write. Default off.",
    ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memo/flags_misc.py tests/test_dream_chronicle.py
git commit -m "feat(chronicle): register MEMO_DREAM_CHRONICLE_ENABLED + MEMO_CHRONICLE_WEEKLY flags (default off)"
```

---

### Task 2: Paths + day helper + módulo base

**Files:**
- Create: `src/memo/dream_chronicle.py`
- Test: `tests/test_dream_chronicle.py` (extender)

**Interfaces:**
- Produces: `CHRONICLE_BUCKET = "_chronicle"`, `chronicle_dir(cfg) -> Path`, `chronicle_path(cfg, day: str) -> Path`, `default_day(now: datetime | None = None) -> str`, `_atomic_write(path: Path, content: str) -> None`.
- Consumes: `cfg.memory_dir` (property de Config, `src/memo/config.py:430` — resuelve a vault o data_dir).

- [ ] **Step 1: Write the failing tests**

Agregar a `tests/test_dream_chronicle.py`:

```python
from datetime import datetime
from pathlib import Path


class _Cfg:
    """Minimal cfg fake — same shape test_dream_profile.py uses."""

    def __init__(self, tmp_path):
        self.memory_dir = tmp_path / "memories"
        self.state_dir = tmp_path / "state"
        self.helper_model = "stub-model"


def _mk_cfg(tmp_path):
    cfg = _Cfg(tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_chronicle_path_lives_in_underscore_bucket(tmp_path):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    p = dc.chronicle_path(cfg, "2026-07-13")
    assert p == Path(cfg.memory_dir) / "_chronicle" / "2026-07-13.md"


def test_default_day_is_previous_day_before_6am():
    from memo import dream_chronicle as dc

    # dream corre 03:00 — la crónica es del día que acaba de terminar
    assert dc.default_day(datetime(2026, 7, 14, 3, 0)) == "2026-07-13"
    assert dc.default_day(datetime(2026, 7, 14, 15, 0)) == "2026-07-14"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'memo.dream_chronicle'`.

- [ ] **Step 3: Create the module**

```python
# src/memo/dream_chronicle.py
"""Nightly chronicle — a human engineering diary distilled from memo's own logs.

Structural clone of dream_profile.py: dream pass -> markdown under a `_` bucket
in ``memory_dir`` (vault when memories_in_vault is on, data_dir otherwise).
Files carry no ``id:`` frontmatter key, so reindex never ingests them.
Gated by ``MEMO_DREAM_CHRONICLE_ENABLED`` (default off).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

CHRONICLE_BUCKET = "_chronicle"


def chronicle_dir(cfg: Any) -> Path:
    """Where chronicle documents live: ``memory_dir/_chronicle/``."""
    return Path(cfg.memory_dir) / CHRONICLE_BUCKET


def chronicle_path(cfg: Any, day: str) -> Path:
    return chronicle_dir(cfg) / f"{day}.md"


def default_day(now: datetime | None = None) -> str:
    """The day being chronicled. A 03:00 nightly run chronicles *yesterday*."""
    return ((now or datetime.now()) - timedelta(hours=6)).date().isoformat()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/memo/dream_chronicle.py tests/test_dream_chronicle.py
git commit -m "feat(chronicle): module skeleton — bucket paths + chronicled-day helper"
```

---

### Task 3: Filtro de provenance (anti-fabricación)

**Files:**
- Modify: `src/memo/dream_chronicle.py`
- Test: `tests/test_dream_chronicle.py` (extender)

**Interfaces:**
- Produces: `filter_cited(text: str, allowed: set[str]) -> tuple[str, float]` — descarta bullets sin id citado válido; retorna `(texto_filtrado, cited_ratio)`. Regla: un bullet sobrevive solo si cita ≥1 id y TODOS sus ids citados están en `allowed` (un id inventado mata el bullet completo). Líneas no-bullet (headings, blancos) pasan siempre. `cited_ratio = bullets_conservados / bullets_totales` (1.0 si no hay bullets).

- [ ] **Step 1: Write the failing tests**

```python
def test_filter_cited_drops_uncited_and_fabricated_bullets():
    from memo import dream_chronicle as dc

    text = (
        "## Trabajo\n"
        "- fixed the sync race [aaaaaaaa]\n"
        "- invented claim with no citation\n"
        "- claim citing unknown id [ffffffff]\n"
        "- two real ids [aaaaaaaa] [bbbbbbbb]\n"
    )
    out, ratio = dc.filter_cited(text, {"aaaaaaaa", "bbbbbbbb"})
    assert "sync race" in out
    assert "two real ids" in out
    assert "invented claim" not in out
    assert "unknown id" not in out
    assert "## Trabajo" in out  # headings siempre pasan
    assert ratio == 0.5  # 2 de 4 bullets sobrevivieron


def test_filter_cited_no_bullets_is_ratio_one():
    from memo import dream_chronicle as dc

    out, ratio = dc.filter_cited("just prose\n", {"aaaaaaaa"})
    assert out == "just prose\n" or out == "just prose"
    assert ratio == 1.0
```

- [ ] **Step 2: Run to verify FAIL**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py -k filter_cited -v`
Expected: FAIL con `AttributeError: ... has no attribute 'filter_cited'`.

- [ ] **Step 3: Implement**

Agregar a `src/memo/dream_chronicle.py`:

```python
_BULLET_RE = re.compile(r"^\s*[-*]\s+")
_ID_RE = re.compile(r"\[([0-9a-f]{8})\]")


def filter_cited(text: str, allowed: set[str]) -> tuple[str, float]:
    """Drop bullet lines whose citations are missing or not in ``allowed``.

    A bullet survives only if it cites at least one id AND every id it cites
    is allowed — one fabricated id kills the whole bullet. Non-bullet lines
    always pass. Returns (filtered_text, kept_bullets / total_bullets).
    """
    kept = total = 0
    out: list[str] = []
    for line in text.splitlines():
        if not _BULLET_RE.match(line):
            out.append(line)
            continue
        total += 1
        ids = set(_ID_RE.findall(line))
        if ids and ids <= allowed:
            kept += 1
            out.append(line)
    ratio = (kept / total) if total else 1.0
    return "\n".join(out), ratio
```

- [ ] **Step 4: Run to verify PASS**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memo/dream_chronicle.py tests/test_dream_chronicle.py
git commit -m "feat(chronicle): provenance filter — uncited/fabricated bullets are dropped"
```

---

### Task 4: Colección de hechos del día (sin LLM)

**Files:**
- Modify: `src/memo/dream_chronicle.py`
- Test: `tests/test_dream_chronicle.py` (extender)

**Interfaces:**
- Consumes:
  - `memo.resume._index.open_store(cfg)` → store con `.recent(limit) -> list[dict]` (campos: `agent, session_id, cwd, updated_at, summary, turn_count`) o `None` si episódico deshabilitado (patrón verificado en `dream_consolidate.py:148-154`).
  - `memo.dashboard_logs.read_grounding_log(state_dir, limit=4000) -> list[dict]` y `read_recall_log(state_dir, limit=...) -> list[dict]`.
  - `memo.token_ledger.grounded_by_day(rows) -> dict[str, int]` y `consults_by_day_client(rows) -> dict[str, dict[str, int]]`.
  - Receipt previo: `cfg.state_dir / "dream" / "last.json"`.
- Produces:
  - `collect_facts(cfg, day: str) -> dict` con keys `episodes` (list[dict]), `new_memories` (list[dict: id/type/title]), `grounded` (int), `consults` (dict[str,int]), `receipt_events` (dict[str,int]).
  - `fact_lines(facts) -> tuple[list[str], set[str]]` — bullets citables (cada uno termina en `[id8]`) + set de ids permitidos.
  - `_memories_created_on(cfg, day, cap=50) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

```python
def _write_memory_md(root: Path, mid: str, day: str, title: str, mtype: str = "decision"):
    p = root / f"{title.replace(' ', '-')}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"---\nid: {mid}\ntype: {mtype}\ncreated: {day}T10:00:00\n---\n# {title}\nbody\n",
        encoding="utf-8",
    )
    return p


def test_memories_created_on_filters_by_day_and_skips_buckets(tmp_path):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    cfg.memory_dir.mkdir(parents=True)
    _write_memory_md(cfg.memory_dir, "a" * 32, "2026-07-13", "hit today")
    _write_memory_md(cfg.memory_dir, "b" * 32, "2026-07-12", "old one")
    # bucket files (_profile/_chronicle) are never memories
    _write_memory_md(cfg.memory_dir / "_profile", "c" * 32, "2026-07-13", "profile doc")

    out = dc._memories_created_on(cfg, "2026-07-13")
    assert [m["id"][:8] for m in out] == ["aaaaaaaa"]
    assert out[0]["title"] == "hit today"
    assert out[0]["type"] == "decision"


def test_collect_facts_and_fact_lines(tmp_path, monkeypatch):
    import json as _json

    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    cfg.memory_dir.mkdir(parents=True)
    day = "2026-07-13"
    _write_memory_md(cfg.memory_dir, "a" * 32, day, "decided X")

    class _FakeStore:
        def recent(self, limit=200):
            return [
                {"agent": "claude-code", "session_id": "d" * 32, "cwd": "/x",
                 "updated_at": f"{day}T20:00:00", "summary": "fixed sync race", "turn_count": 12},
                {"agent": "claude-code", "session_id": "e" * 32, "cwd": "/x",
                 "updated_at": "2026-07-11T20:00:00", "summary": "other day", "turn_count": 3},
            ]

    monkeypatch.setattr("memo.resume._index.open_store", lambda cfg: _FakeStore())

    # receipt previo con actividad
    d = cfg.state_dir / "dream"
    d.mkdir(parents=True)
    (d / "last.json").write_text(_json.dumps({"superseded": ["x"], "merged": [], "errors": []}))

    facts = dc.collect_facts(cfg, day)
    assert len(facts["episodes"]) == 1
    assert facts["episodes"][0]["summary"] == "fixed sync race"
    assert [m["id"][:8] for m in facts["new_memories"]] == ["aaaaaaaa"]
    assert facts["receipt_events"] == {"superseded": 1}
    assert facts["grounded"] == 0  # no grounding.log en tmp

    lines, allowed = dc.fact_lines(facts)
    assert allowed == {"dddddddd", "aaaaaaaa"}
    assert any("[dddddddd]" in ln for ln in lines)
    assert any("[aaaaaaaa]" in ln for ln in lines)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py -k "memories_created or collect_facts" -v`
Expected: FAIL con `AttributeError` (funciones no existen).

- [ ] **Step 3: Implement**

Agregar a `src/memo/dream_chronicle.py`:

```python
_FM_ID_RE = re.compile(r"^id:\s*([0-9a-f]{8,})", re.MULTILINE)
_FM_TYPE_RE = re.compile(r"^type:\s*(\S+)", re.MULTILINE)
_DATE_KEYS = ("created:", "created_at:", "updated:", "date:")
_RECEIPT_KEYS = ("superseded", "merged", "archived_stale", "synthesized")


def _memories_created_on(cfg: Any, day: str, cap: int = 50) -> list[dict[str, str]]:
    """Memories whose frontmatter date starts with ``day``. Disk scan — markdown
    is the source of truth; `_`-prefixed buckets (_profile/_chronicle) are not
    memories."""
    root = Path(cfg.memory_dir)
    if not root.exists():
        return []
    out: list[dict[str, str]] = []
    for p in sorted(root.rglob("*.md")):
        if any(part.startswith("_") for part in p.relative_to(root).parts):
            continue
        head = p.read_text(encoding="utf-8", errors="ignore")[:2000]
        m_id = _FM_ID_RE.search(head)
        if m_id is None:
            continue  # no id: -> not a memory record
        date_lines = [
            ln.strip() for ln in head.splitlines()[:15] if ln.strip().startswith(_DATE_KEYS)
        ]
        if not any(day in ln for ln in date_lines):
            continue
        m_type = _FM_TYPE_RE.search(head)
        title = next(
            (ln.lstrip("# ").strip() for ln in head.splitlines() if ln.startswith("# ")),
            p.stem,
        )
        out.append({"id": m_id.group(1), "type": m_type.group(1) if m_type else "note", "title": title})
        if len(out) >= cap:
            break
    return out


def collect_facts(cfg: Any, day: str) -> dict[str, Any]:
    """Deterministic day facts — no LLM. Each sub-source is best-effort."""
    from memo.dashboard_logs import read_grounding_log, read_recall_log
    from memo.resume._index import open_store
    from memo.token_ledger import consults_by_day_client, grounded_by_day

    episodes: list[dict[str, Any]] = []
    store = open_store(cfg)
    if store is not None:
        episodes = [
            e for e in store.recent(limit=200)
            if str(e.get("updated_at") or "").startswith(day)
        ]

    grounded = grounded_by_day(read_grounding_log(cfg.state_dir)).get(day, 0)
    consults = consults_by_day_client(read_recall_log(cfg.state_dir, limit=4000)).get(day, {})

    receipt_events: dict[str, int] = {}
    last = Path(cfg.state_dir) / "dream" / "last.json"
    if last.exists():
        try:
            data = json.loads(last.read_text(encoding="utf-8"))
            for key in _RECEIPT_KEYS:
                v = data.get(key)
                if isinstance(v, list) and v:
                    receipt_events[key] = len(v)
        except (ValueError, OSError):
            pass

    return {
        "episodes": episodes,
        "new_memories": _memories_created_on(cfg, day),
        "grounded": grounded,
        "consults": consults,
        "receipt_events": receipt_events,
    }


def fact_lines(facts: dict[str, Any]) -> tuple[list[str], set[str]]:
    """Citable bullets fed to the LLM. Every line ends with its [id8];
    the returned set is the provenance whitelist for filter_cited()."""
    lines: list[str] = []
    allowed: set[str] = set()
    for e in facts["episodes"]:
        sid = str(e.get("session_id") or "")[:8]
        if not sid:
            continue
        allowed.add(sid)
        lines.append(
            f"- session {sid} ({e.get('agent', '?')}, {e.get('turn_count', 0)} turns): "
            f"{str(e.get('summary') or '')[:200]} [{sid}]"
        )
    for m in facts["new_memories"]:
        mid = str(m["id"])[:8]
        allowed.add(mid)
        lines.append(f"- new {m['type']} memory: {str(m['title'])[:120]} [{mid}]")
    return lines, allowed
```

- [ ] **Step 4: Run to verify PASS**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py -v`
Expected: PASS (todos).

> Nota de verificación: antes de dar la task por cerrada, abrí UNA memoria real (`ls ~/.local/share/memo/*.md | head -1` o el memory_dir configurado) y confirmá que el key de fecha del frontmatter está cubierto por `_DATE_KEYS`. Si el repo usa otro nombre (p.ej. `ts:`), agregalo a la tupla.

- [ ] **Step 5: Commit**

```bash
git add src/memo/dream_chronicle.py tests/test_dream_chronicle.py
git commit -m "feat(chronicle): deterministic day-facts collection + citable fact lines"
```

---

### Task 5: LLM narrate + render + run_chronicle_pass

**Files:**
- Modify: `src/memo/dream_chronicle.py`
- Test: `tests/test_dream_chronicle.py` (extender)

**Interfaces:**
- Consumes: `chat_with_timeout(chat, timeout, model, messages, options)` de `memo.memory.record` (patrón verbatim de `dream_consolidate._llm_synthesize`, `dream_consolidate.py:101-133`); `mem._ensure_chat()`; `mem.cfg.helper_model`.
- Produces:
  - `_llm_narrate(mem, day: str, lines: list[str]) -> str | None` (None = LLM no disponible/timeout/QUIET_DAY).
  - `render_chronicle(day: str, narrative: str, facts: dict) -> str` — markdown SIN key `id:` en frontmatter.
  - `run_chronicle_pass(cfg, mem, *, day: str | None = None, weekly: bool = False, dry_run: bool = False) -> dict` — nunca lanza; `status` ∈ {`done`, `skipped`, `llm_unavailable`, `error`}; con `done` incluye `path` y `cited_ratio`.

- [ ] **Step 1: Write the failing tests**

```python
class _Mem:
    """Minimal Memory fake — _llm_narrate is monkeypatched so it's never used."""

    def __init__(self, cfg):
        self.cfg = cfg


def test_run_chronicle_pass_writes_file_with_provenance(tmp_path, monkeypatch):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    cfg.memory_dir.mkdir(parents=True)
    day = "2026-07-13"
    _write_memory_md(cfg.memory_dir, "a" * 32, day, "decided X")

    monkeypatch.setattr("memo.resume._index.open_store", lambda cfg: None)
    monkeypatch.setattr(
        dc, "_llm_narrate",
        lambda mem, d, lines: "- decided X hoy [aaaaaaaa]\n- invented stuff\n",
    )
    res = dc.run_chronicle_pass(cfg, _Mem(cfg), day=day)
    assert res["status"] == "done"
    assert res["cited_ratio"] == 0.5
    doc = dc.chronicle_path(cfg, day).read_text(encoding="utf-8")
    assert "[aaaaaaaa]" in doc
    assert "invented stuff" not in doc  # provenance filter aplicado
    assert "id:" not in doc.split("---")[1]  # frontmatter sin id: -> reindex lo ignora


def test_run_chronicle_pass_skips_on_no_signal(tmp_path, monkeypatch):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    monkeypatch.setattr("memo.resume._index.open_store", lambda cfg: None)
    res = dc.run_chronicle_pass(cfg, _Mem(cfg), day="2026-07-13")
    assert res["status"] == "skipped"
    assert not dc.chronicle_path(cfg, "2026-07-13").exists()


def test_run_chronicle_pass_llm_unavailable(tmp_path, monkeypatch):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    cfg.memory_dir.mkdir(parents=True)
    _write_memory_md(cfg.memory_dir, "a" * 32, "2026-07-13", "decided X")
    monkeypatch.setattr("memo.resume._index.open_store", lambda cfg: None)
    monkeypatch.setattr(dc, "_llm_narrate", lambda *a, **k: None)
    res = dc.run_chronicle_pass(cfg, _Mem(cfg), day="2026-07-13")
    assert res["status"] == "llm_unavailable"
    assert not dc.chronicle_path(cfg, "2026-07-13").exists()


def test_run_chronicle_pass_dry_run_writes_nothing(tmp_path, monkeypatch):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    cfg.memory_dir.mkdir(parents=True)
    _write_memory_md(cfg.memory_dir, "a" * 32, "2026-07-13", "decided X")
    monkeypatch.setattr("memo.resume._index.open_store", lambda cfg: None)
    monkeypatch.setattr(dc, "_llm_narrate", lambda *a, **k: "- decided X [aaaaaaaa]")
    res = dc.run_chronicle_pass(cfg, _Mem(cfg), day="2026-07-13", dry_run=True)
    assert res["status"] == "done"
    assert "path" not in res
    assert not dc.chronicle_path(cfg, "2026-07-13").exists()


def test_run_chronicle_pass_never_raises(tmp_path, monkeypatch):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    monkeypatch.setattr(dc, "collect_facts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = dc.run_chronicle_pass(cfg, _Mem(cfg), day="2026-07-13")
    assert res["status"] == "error"
    assert "boom" in res["error"]
```

- [ ] **Step 2: Run to verify FAIL**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py -k run_chronicle -v`
Expected: FAIL con `AttributeError: ... 'run_chronicle_pass'`.

- [ ] **Step 3: Implement**

Agregar a `src/memo/dream_chronicle.py`:

```python
_SYS = (
    "You write ONE short engineering-diary entry in Spanish (150-400 words, "
    "markdown bullets under short headings like '## Trabajo' / '## Decisiones') "
    "from the day's facts below. State ONLY what the facts state — never invent. "
    "EVERY bullet MUST end with the bracketed 8-char id(s) of the fact(s) it "
    "came from, e.g. [a1b2c3d4]. If the facts are too thin for a diary, reply "
    "exactly: QUIET_DAY"
)


def _llm_narrate(mem: Any, day: str, lines: list[str]) -> str | None:
    from memo.memory.record import chat_with_timeout

    out = chat_with_timeout(
        mem._ensure_chat(),
        timeout=60,
        model=mem.cfg.helper_model,
        messages=[
            {"role": "system", "content": _SYS},
            {"role": "user", "content": f"Facts for {day}:\n" + "\n".join(lines)},
        ],
        options={"temperature": 0.0, "max_tokens": 700, "thinking": False},
    )
    if out is None:
        return None
    text = ((out.get("message") or {}).get("content") or "").strip()
    if not text or text == "QUIET_DAY":
        return None
    return text


def render_chronicle(day: str, narrative: str, facts: dict[str, Any]) -> str:
    """Full document. Frontmatter has NO ``id:`` key -> reindex ignores it."""
    consults = facts.get("consults") or {}
    events = facts.get("receipt_events") or {}
    nums = [
        f"- episodios: {len(facts.get('episodes') or [])}",
        f"- memorias nuevas: {len(facts.get('new_memories') or [])}",
        f"- recalls grounded: {facts.get('grounded', 0)}",
    ]
    if consults:
        nums.append("- consults: " + ", ".join(f"{k} {v}" for k, v in sorted(consults.items())))
    if events:
        nums.append("- mantenimiento: " + ", ".join(f"{k} {v}" for k, v in sorted(events.items())))
    return (
        "---\n"
        "kind: chronicle\n"
        f"day: {day}\n"
        "generated_by: memo dream chronicle\n"
        "---\n"
        f"# Crónica — {day}\n\n"
        f"{narrative}\n\n"
        "## Números del día\n" + "\n".join(nums) + "\n"
    )


def run_chronicle_pass(
    cfg: Any,
    mem: Any,
    *,
    day: str | None = None,
    weekly: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One nightly chronicle pass. Never raises — the cli_dream caller records
    a returned ``status="error"`` in ``receipt["errors"]``."""
    res: dict[str, Any] = {"status": "noop", "day": ""}
    try:
        d = day or default_day()
        res["day"] = d
        facts = collect_facts(cfg, d)
        lines, allowed = fact_lines(facts)
        if not lines:
            res["status"] = "skipped"
            return res
        narrative = _llm_narrate(mem, d, lines)
        if narrative is None:
            res["status"] = "llm_unavailable"
            return res
        filtered, ratio = filter_cited(narrative, allowed)
        res["cited_ratio"] = round(ratio, 3)
        if not dry_run:
            path = chronicle_path(cfg, d)
            _atomic_write(path, render_chronicle(d, filtered, facts))
            res["path"] = str(path)
            if weekly:
                wk = write_weekly(cfg, d)
                if wk is not None:
                    res["weekly_path"] = str(wk)
        res["status"] = "done"
    except Exception as exc:  # surfaced into the receipt, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res
```

Nota: `write_weekly` se implementa en Task 6 — para que este paso compile, agregá temporalmente:

```python
def write_weekly(cfg: Any, day: str) -> Path | None:
    return None
```

- [ ] **Step 4: Run to verify PASS**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py -v`
Expected: PASS (todos).

- [ ] **Step 5: Commit**

```bash
git add src/memo/dream_chronicle.py tests/test_dream_chronicle.py
git commit -m "feat(chronicle): LLM narrate + provenance-filtered render + run_chronicle_pass"
```

---

### Task 6: Weekly rollup

**Files:**
- Modify: `src/memo/dream_chronicle.py` (reemplazar el stub `write_weekly`)
- Test: `tests/test_dream_chronicle.py` (extender)

**Interfaces:**
- Produces: `write_weekly(cfg, day: str) -> Path | None` — concatena los archivos diarios de la ISO-week de `day` en `week-<year>-W<ww>.md` (determinístico, sin LLM). `None` si no hay diarios esa semana.

- [ ] **Step 1: Write the failing test**

```python
def test_write_weekly_concatenates_days_of_iso_week(tmp_path):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    # 2026-07-13 es lunes; 2026-07-14 martes -> misma ISO week 29
    for d, body in (("2026-07-13", "lunes body"), ("2026-07-14", "martes body")):
        dc._atomic_write(dc.chronicle_path(cfg, d), f"# Crónica — {d}\n{body}\n")
    wk = dc.write_weekly(cfg, "2026-07-14")
    assert wk is not None and wk.name == "week-2026-W29.md"
    text = wk.read_text(encoding="utf-8")
    assert "lunes body" in text and "martes body" in text
    assert text.index("lunes body") < text.index("martes body")  # orden cronológico


def test_write_weekly_none_when_empty(tmp_path):
    from memo import dream_chronicle as dc

    assert dc.write_weekly(_mk_cfg(tmp_path), "2026-07-14") is None
```

- [ ] **Step 2: Run to verify FAIL**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py -k weekly -v`
Expected: FAIL (stub retorna None siempre → primer test falla).

- [ ] **Step 3: Implement (reemplaza el stub)**

```python
def write_weekly(cfg: Any, day: str) -> Path | None:
    """Deterministic ISO-week rollup — concatenation, no LLM."""
    from datetime import date

    y, w, _ = date.fromisoformat(day).isocalendar()
    root = chronicle_dir(cfg)
    if not root.exists():
        return None
    days: list[Path] = []
    for p in sorted(root.glob("*.md")):
        try:
            py, pw, _ = date.fromisoformat(p.stem).isocalendar()
        except ValueError:
            continue  # week-*.md and anything not a day file
        if (py, pw) == (y, w):
            days.append(p)
    if not days:
        return None
    parts = [p.read_text(encoding="utf-8") for p in days]
    out = root / f"week-{y}-W{w:02d}.md"
    _atomic_write(out, f"# Semana {y}-W{w:02d}\n\n" + "\n\n---\n\n".join(parts))
    return out
```

- [ ] **Step 4: Run to verify PASS**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py -v`
Expected: PASS (todos).

- [ ] **Step 5: Commit**

```bash
git add src/memo/dream_chronicle.py tests/test_dream_chronicle.py
git commit -m "feat(chronicle): deterministic ISO-week rollup"
```

---

### Task 7: Wiring en el pipeline nocturno + subcomando `memo dream chronicle`

**Files:**
- Modify: `src/memo/cli_dream.py` — (a) bloque gateado en el pipeline `dream run`, inmediatamente DESPUÉS del bloque `MEMO_DREAM_PROFILE_ENABLED` (~línea 561); (b) subcomando standalone al final, junto a `consolidate-episodes` (~línea 1285).
- Test: `tests/test_dream_chronicle.py` (extender)

**Interfaces:**
- Consumes: `run_chronicle_pass` (Task 5), `flag_bool`, `receipt["errors"]`, `_get_memory(cfg)` y `console`/`json`/`click` ya presentes en `cli_dream.py`.
- Produces: key `receipt["chronicle"]` en el receipt nocturno; comando `memo dream chronicle [--day] [--dry-run] [--json]`.

- [ ] **Step 1: Write the failing test (subcomando vía CliRunner, pase stubbeado)**

```python
def test_dream_chronicle_subcommand_json(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from memo import dream_chronicle as dc
    from memo.cli import cli

    monkeypatch.setattr(
        dc, "run_chronicle_pass",
        lambda cfg, mem, **kw: {"status": "done", "day": "2026-07-13", "path": "/x.md", "cited_ratio": 1.0},
    )
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }
    result = CliRunner().invoke(cli, ["dream", "chronicle", "--json"], env=env)
    assert result.exit_code == 0, result.output
    assert '"status": "done"' in result.output
```

- [ ] **Step 2: Run to verify FAIL**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py::test_dream_chronicle_subcommand_json -v`
Expected: FAIL — `Error: No such command 'chronicle'` en el output (exit_code 2).

- [ ] **Step 3: Implement — (a) bloque en pipeline**

En `src/memo/cli_dream.py`, inmediatamente después del bloque `if flag_bool("MEMO_DREAM_PROFILE_ENABLED"):` (que termina ~línea 561), agregar (mismo patrón verbatim que profile/consolidate):

```python
        # Chronicle — nightly engineering diary ------------------------------
        if flag_bool("MEMO_DREAM_CHRONICLE_ENABLED"):
            progress.update(step, description="[chronicle] writing diary...")
            try:
                from memo import dream_chronicle

                receipt["chronicle"] = dream_chronicle.run_chronicle_pass(
                    cfg,
                    mem,
                    weekly=flag_bool("MEMO_CHRONICLE_WEEKLY"),
                    dry_run=dry_run,
                )
                if receipt["chronicle"].get("status") == "error":
                    receipt["errors"].append(f"chronicle: {receipt['chronicle'].get('error')}")
                _ch = receipt["chronicle"]
                progress.update(
                    step,
                    description=f"[chronicle] [green]✓[/green]  {_ch.get('status')}",
                )
            except Exception as exc:
                receipt["errors"].append(f"chronicle: {type(exc).__name__}: {exc}")
                progress.update(step, description="[chronicle] [yellow]warn[/yellow]")
```

**(b) subcomando standalone** — al final de `cli_dream.py`, junto al subcomando `consolidate-episodes` (patrón verbatim de `cli_dream.py:1285-1300`):

```python
@dream_cmd.command(name="chronicle")
@click.option("--day", "day", default=None, help="Day to chronicle (YYYY-MM-DD, default: last finished day).")
@click.option("--dry-run", is_flag=True, help="Compute + narrate, don't write.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def dream_chronicle_cmd(day: str | None, dry_run: bool, as_json: bool) -> None:
    """Write the engineering diary for one day (see MEMO_DREAM_CHRONICLE_ENABLED)."""
    from memo import dream_chronicle

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    res = dream_chronicle.run_chronicle_pass(
        cfg, mem, day=day, weekly=flag_bool("MEMO_CHRONICLE_WEEKLY"), dry_run=dry_run
    )
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]chronicle:[/bold] {res.get('status')} {res.get('path', '')}")
```

- [ ] **Step 4: Run to verify PASS + no regresiones dream**

Run: `uv run --no-sync pytest tests/test_dream_chronicle.py tests/test_cli_dream_status.py -v`
Expected: PASS (todos, incluidos los pre-existentes de dream).

- [ ] **Step 5: Smoke manual (flag off por default → no-op)**

Run: `uv run --no-sync memo dream chronicle --dry-run --json`
Expected: JSON con `"status"` ∈ {`skipped`, `llm_unavailable`, `done`} según la máquina — sin traceback. (El pipeline nocturno queda inerte hasta `memo config set dream.chronicle_enabled true` / flag ON — no lo enciendas en este plan.)

- [ ] **Step 6: Commit**

```bash
git add src/memo/cli_dream.py tests/test_dream_chronicle.py
git commit -m "feat(chronicle): wire nightly pass (flag-gated) + memo dream chronicle subcommand"
```

---

### Task 8: `memo chronicle` — lector on-demand

**Files:**
- Create: `src/memo/cli_chronicle.py`
- Modify: `src/memo/cli.py` — import + `cli.add_command(chronicle_cmd)` (patrón: `cli.py:54` import / `cli.py:358` add_command)
- Test: `tests/test_cli_chronicle.py` (nuevo)

**Interfaces:**
- Consumes: `chronicle_dir`, `chronicle_path`, `default_day` (Task 2).
- Produces: comando top-level `memo chronicle [--date YYYY-MM-DD] [--week]` — imprime la crónica pedida (default: la más reciente); `--week` imprime el rollup semanal más reciente.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_chronicle.py
"""Tests for the `memo chronicle` reader command."""
from __future__ import annotations

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


def test_chronicle_reads_latest(tmp_path):
    from memo.cli import cli

    chron = tmp_path / "data" / "_chronicle"
    chron.mkdir(parents=True)
    (chron / "2026-07-12.md").write_text("# Crónica — 2026-07-12\nviejo\n", encoding="utf-8")
    (chron / "2026-07-13.md").write_text("# Crónica — 2026-07-13\nnuevo\n", encoding="utf-8")

    result = CliRunner().invoke(cli, ["chronicle"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "nuevo" in result.output and "viejo" not in result.output


def test_chronicle_date_and_missing(tmp_path):
    from memo.cli import cli

    result = CliRunner().invoke(cli, ["chronicle", "--date", "2026-01-01"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "no hay crónica" in result.output.lower()
```

- [ ] **Step 2: Run to verify FAIL**

Run: `uv run --no-sync pytest tests/test_cli_chronicle.py -v`
Expected: FAIL — `No such command 'chronicle'`.

- [ ] **Step 3: Implement**

```python
# src/memo/cli_chronicle.py
"""`memo chronicle` — read the nightly engineering diary (see dream_chronicle)."""
from __future__ import annotations

import click
from rich.console import Console
from rich.markdown import Markdown

from memo.config import Config
from memo.dream_chronicle import chronicle_dir, chronicle_path

console = Console()


@click.command(name="chronicle")
@click.option("--date", "date", default=None, help="Day to show (YYYY-MM-DD, default: latest).")
@click.option("--week", "week", is_flag=True, help="Show the latest weekly rollup instead.")
def chronicle_cmd(date: str | None, week: bool) -> None:
    """Show the engineering diary memo wrote for a day (or week)."""
    cfg = Config.from_env()
    root = chronicle_dir(cfg)
    if date is not None:
        target = chronicle_path(cfg, date)
    else:
        pattern = "week-*.md" if week else "[0-9]*.md"
        candidates = sorted(root.glob(pattern)) if root.exists() else []
        target = candidates[-1] if candidates else None
    if target is None or not target.exists():
        console.print(
            "No hay crónica todavía — encendé el pase nocturno con "
            "[cyan]MEMO_DREAM_CHRONICLE_ENABLED[/cyan] o corré [cyan]memo dream chronicle[/cyan]."
        )
        return
    console.print(Markdown(target.read_text(encoding="utf-8")))
```

En `src/memo/cli.py`, junto a los demás imports de comandos (zona `cli.py:30-72`):

```python
from memo.cli_chronicle import chronicle_cmd
```

y junto a los `add_command` (zona `cli.py:348-486`):

```python
cli.add_command(chronicle_cmd)
```

- [ ] **Step 4: Run to verify PASS**

Run: `uv run --no-sync pytest tests/test_cli_chronicle.py tests/test_cli_mcp_surface_smoke.py -v`
Expected: PASS (el smoke de superficie CLI recoge el comando nuevo automáticamente vía `--help`).

- [ ] **Step 5: Commit**

```bash
git add src/memo/cli_chronicle.py src/memo/cli.py tests/test_cli_chronicle.py
git commit -m "feat(chronicle): memo chronicle reader command"
```

---

### Task 9: Gate final de la feature

**Files:** ninguno nuevo — verificación.

- [ ] **Step 1: Suite dirigida + lint + types de archivos propios**

```bash
uv run --no-sync pytest tests/test_dream_chronicle.py tests/test_cli_chronicle.py tests/test_cli_dream_status.py tests/test_dream_profile.py -v
uv run --no-sync ruff check src/memo/dream_chronicle.py src/memo/cli_chronicle.py tests/test_dream_chronicle.py tests/test_cli_chronicle.py
uv run --no-sync mypy src/memo/dream_chronicle.py src/memo/cli_chronicle.py
```
Expected: todo verde.

- [ ] **Step 2: Suite completa (merge-drift check — trampa conocida del repo)**

Run: `uv run --no-sync pytest tests/ -x -q`
Expected: verde (los 3 flakes de snapshot TUI pre-existentes no cuentan como regresión).

- [ ] **Step 3: Validación de flags**

Run: `uv run --no-sync memo config validate`
Expected: sin errores por los flags nuevos.

- [ ] **Step 4: Commit final si hubo fixes**

```bash
git add <solo archivos tocados por fixes>
git commit -m "test(chronicle): green full suite"
```

## Gate humano post-implementación (del spec — NO es parte de este plan)

Encender `MEMO_DREAM_CHRONICLE_ENABLED` vía `memo config set` y **leer las crónicas 2 semanas** antes de anunciar/documentar la feature. Métrica proxy automatizada: `cited_ratio` en el receipt (target ≈1.0). Arrancar leyendo el rollup `--week`.
