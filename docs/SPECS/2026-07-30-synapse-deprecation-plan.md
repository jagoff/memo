# Synapse Deprecation (Big Bang) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** memo adopta sus 3 daemons (recall, nightly, vault-ingest) como agentes nativos `com.memo.*`, y synapse + consciousness-contracts se desinstalan y archivan por completo en una pasada.

**Architecture:** El port a memo son 3 módulos nuevos (`ops_gc.py` lógica pura, `ingest_exclude.py` tombstones, `vault_ingest.py` orquestación) + un grupo CLI `memo ops` (`cli_ops.py`) + templates launchd en `memo/launchd/`. Nada de synapse se importa; los jobs nightly que memo ya tiene nativos (`contradict scan`, `consolidate`) se llaman directo desde el script. Después: swap de daemons, teardown del fleet `com.synapse.*`, archivo de repos.

**Tech Stack:** Python 3 + click (convención `cli_*.py` de memo), pytest, launchd, uv tool (runtime aislado `mlx-memo`).

**Spec:** `docs/SPECS/2026-07-30-synapse-deprecation-design.md`

## Global Constraints

- Working tree: worktree `~/repos/memo-spec` (branch `docs/synapse-deprecation-spec`); NO tocar `~/repos/memo` (branch de otra sesión).
- Tests: `cd ~/repos/memo-spec && uv run --no-sync pytest tests/<file> -q` (convención memo).
- El binario instalado es `~/.local/bin/memo` → uv tool `mlx-memo`; el código nuevo NO existe ahí hasta `uv tool install --from ~/repos/memo-spec mlx-memo --force --reinstall`.
- Los ports NO deben importar synapse ni consciousness_contracts (guard `_RETIRED_IMPORTS` en `definitive.py` lo verifica).
- Se descartan del vault-ingest de synapse: señal memflow, gbrain import, overview rebuild (extras synapse-side; se reporta la pérdida al final).
- Rollback: nada se borra — plists y `~/.synapse/` van a `~/.memo-daemon-backups/<ts>-synapse-final/`.

---

### Task 1: `ops_gc.py` — lógica pura de GC

**Files:**
- Create: `src/memo/ops_gc.py`
- Test: `tests/test_ops_gc.py`

**Interfaces:**
- Produces: `find_vault_orphans(records: list[dict], *, path_exists=os.path.exists) -> list[dict]`; `find_exact_duplicates(records: list[dict]) -> list[dict]` (registros stale a borrar; conserva el más nuevo por `updated`/`created`).
- Records = shape de `Record.to_dict()` (`id`, `body`, `updated`, `created`, `extra.{source,abs_path}`).

- [ ] **Step 1: test que falla**

```python
"""Tests for pure GC logic (ported from synapse ops on deprecation)."""
from memo.ops_gc import find_exact_duplicates, find_vault_orphans


def _rec(id_, body="x", updated="2026-01-02", source="", abs_path=None):
    extra = {"source": source}
    if abs_path is not None:
        extra["abs_path"] = abs_path
    return {"id": id_, "body": body, "updated": updated, "created": "2026-01-01", "extra": extra}


def test_orphans_only_vault_ingest_missing_path():
    recs = [
        _rec("a", source="vault-ingest:notes", abs_path="/nope/gone.md"),
        _rec("b", source="vault-ingest:notes", abs_path="/exists.md"),
        _rec("c", source="chat", abs_path="/nope/gone.md"),
        _rec("d", source="vault-ingest:notes"),
    ]
    got = find_vault_orphans(recs, path_exists=lambda p: p == "/exists.md")
    assert [r["id"] for r in got] == ["a"]


def test_exact_duplicates_keep_newest():
    recs = [
        _rec("old", body="same", updated="2026-01-01"),
        _rec("new", body="same", updated="2026-02-01"),
        _rec("uniq", body="other"),
        _rec("empty1", body="  "),
        _rec("empty2", body="  "),
    ]
    stale = find_exact_duplicates(recs)
    assert [r["id"] for r in stale] == ["old"]
```

- [ ] **Step 2:** `uv run --no-sync pytest tests/test_ops_gc.py -q` → FAIL (ModuleNotFoundError)
- [ ] **Step 3: implementación** — port literal de `synapse/ops.py:515,595` como funciones puras:

```python
"""Pure GC logic over memo record dicts (`Record.to_dict()` shape).

Ported from synapse `ops.gc_vault_orphans` / `ops.gc_memo_duplicates` when the
synapse control plane was deprecated (2026-07-30). Pure functions; cli_ops
lists records and performs deletions.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Any


def find_vault_orphans(
    records: list[dict[str, Any]],
    *,
    path_exists: Callable[[str], bool] = os.path.exists,
) -> list[dict[str, Any]]:
    """Vault-ingested records whose source file no longer exists on disk."""
    orphans: list[dict[str, Any]] = []
    for r in records:
        extra = r.get("extra") or {}
        abs_path = extra.get("abs_path")
        source = str(extra.get("source") or "")
        if "vault-ingest" in source and abs_path and not path_exists(abs_path):
            orphans.append(r)
    return orphans


def find_exact_duplicates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records whose body is an exact duplicate of a newer record (the stale ones)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        body = str(r.get("body") or "")
        if not body.strip():
            continue
        groups.setdefault(hashlib.sha256(body.encode()).hexdigest(), []).append(r)
    stale: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda r: str(r.get("updated") or r.get("created") or ""), reverse=True)
        stale.extend(members[1:])
    return stale
```

- [ ] **Step 4:** pytest → PASS
- [ ] **Step 5:** `git add … && git commit -m "feat(ops): pure GC logic ported from synapse"`

---

### Task 2: `ingest_exclude.py` — tombstones

**Files:**
- Create: `src/memo/ingest_exclude.py`
- Test: `tests/test_ingest_exclude.py`

**Interfaces:**
- Produces: `IngestExcludeStore(state_dir: Path | None = None)` → `.globs(label) -> list[str]`, `.add(vault_label=, rel_path=) -> bool`, `.remove(vault_label=, rel_path=) -> bool`, `.all_labels() -> list[str]`. Estado en `<state_dir>/ingest_excludes/<label>.txt`; default `Config.from_env().state_dir`.
- NO se portan `source_vault_rel` / `filter_tombstoned_sources` (chat synapse, muere).

- [ ] **Step 1: test que falla**

```python
from memo.ingest_exclude import IngestExcludeStore


def test_add_globs_remove_roundtrip(tmp_path):
    store = IngestExcludeStore(state_dir=tmp_path)
    assert store.globs("notes") == []
    assert store.add(vault_label="notes", rel_path="a/b.md") is True
    assert store.add(vault_label="notes", rel_path="a/b.md") is False  # idempotent
    store.add(vault_label="notes", rel_path="c.md")
    assert store.globs("notes") == ["a/b.md", "c.md"]
    assert store.all_labels() == ["notes"]
    assert store.remove(vault_label="notes", rel_path="a/b.md") is True
    assert store.globs("notes") == ["c.md"]
    assert store.remove(vault_label="notes", rel_path="zzz") is False


def test_label_sanitized(tmp_path):
    store = IngestExcludeStore(state_dir=tmp_path)
    store.add(vault_label="Mi Vault!", rel_path="x.md")
    assert (tmp_path / "ingest_excludes" / "mi-vault.txt").exists()
```

- [ ] **Step 2:** pytest → FAIL
- [ ] **Step 3:** port de `synapse/ingest_exclude.py` (clase completa + `_safe_label`), con `default_state_dir()` reemplazado por `Config.from_env().state_dir` (import lazy en `__init__` para no pagar Config en import) y logger `logging.getLogger("memo.ingest_exclude")`. Mantener dedupe on read, append-on-add, rewrite-on-remove, skip de comentarios `#`.
- [ ] **Step 4:** pytest → PASS
- [ ] **Step 5:** commit `feat(ops): vault-ingest tombstone store ported from synapse`

---

### Task 3: `vault_ingest.py` — orquestación

**Files:**
- Create: `src/memo/vault_ingest.py`
- Test: `tests/test_vault_ingest.py`

**Interfaces:**
- Consumes: `IngestExcludeStore` (Task 2).
- Produces: `vault_paths() -> list[Path]` (env `MEMO_VAULT_PATHS` coma-separado override; default las 2 vaults iCloud existentes en disco); `vault_label(p: Path) -> str` (`Notes`→`notes`, `obsidian-work`→`work`); `build_ingest_command(memo_bin, path, label, excludes) -> list[str]`; `run_vault_ingest(*, memo_bin=None) -> dict` (`{"ok": True, "vaults": [{vault,path,returncode,excludes}]}`).

- [ ] **Step 1: test que falla**

```python
from pathlib import Path

from memo.vault_ingest import build_ingest_command, vault_label, vault_paths, _FIXED_VAULT_EXCLUDES


def test_vault_label():
    assert vault_label(Path("/x/Notes")) == "notes"
    assert vault_label(Path("/x/obsidian-work")) == "work"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMO_VAULT_PATHS", f"{tmp_path}/v1, {tmp_path}/v2")
    assert vault_paths() == [tmp_path / "v1", tmp_path / "v2"]


def test_build_ingest_command():
    cmd = build_ingest_command("/bin/memo", Path("/v/Notes"), "notes", ["a/**", "b.md"])
    assert cmd == [
        "/bin/memo", "ingest", "/v/Notes", "--name", "notes", "--prune",
        "--exclude", "a/**", "--exclude", "b.md",
    ]


def test_fixed_excludes_present():
    assert "04-Archive/**" in _FIXED_VAULT_EXCLUDES
```

- [ ] **Step 2:** pytest → FAIL
- [ ] **Step 3:** implementación — port de `synapse/ops.py:749-806+` con renombres de env (`SYNAPSE_VAULT_PATHS`→`MEMO_VAULT_PATHS`): `_DEFAULT_VAULT_PATHS` = las 2 vaults iCloud (`.../iCloud~md~obsidian/Documents/Notes` y `.../obsidian-work`), `_FIXED_VAULT_EXCLUDES` = las mismas 5 entradas (`Obsidian/Whatsapp/**`, `Obsidian/AI/**`, `04-Archive/**`, `Archive/**`, `archive/**`). `run_vault_ingest`: por vault `git pull --ff-only` best-effort si hay `.git`, luego subprocess `build_ingest_command(...)` con excludes = fijas + `IngestExcludeStore().globs(label)`; `memo_bin` default `shutil.which("memo") or "memo"`. SIN gbrain / señal memflow / overview rebuild.
- [ ] **Step 4:** pytest → PASS
- [ ] **Step 5:** commit `feat(ops): vault re-ingest orchestration ported from synapse`

---

### Task 4: `cli_ops.py` — grupo `memo ops`

**Files:**
- Create: `src/memo/cli_ops.py`
- Modify: `src/memo/cli.py` (registrar `ops_group` donde se registran los demás grupos)
- Test: `tests/test_cli_ops.py`

**Interfaces:**
- Consumes: Tasks 1-3; `from memo.cli_common import get_memory`; `from memo.contracts import ActorIdentity`.
- Produces comandos: `memo ops gc-vault-orphans [--dry-run] [--json]`, `memo ops gc-memo-duplicates [--dry-run] [--json]`, `memo ops vault-ingest [--json]`, `memo ops exclude list|add|remove`.
- Deletes: `mem.delete(id, actor=ActorIdentity(actor_id="memo-ops", actor_kind="system"))`. Listado: `[r.to_dict() for r in mem.list(limit=999999)]`.
- Output JSON compatible con el de synapse: gc-orphans `{scanned, orphans, deleted, dry_run}`, gc-dupes `{scanned, dup_groups, deleted, dry_run}` (en dupes, `deleted` cuenta también en dry-run, fidelidad al original).

- [ ] **Step 1: test que falla** — CliRunner sobre un memo vacío (`MEMO_DATA_DIR`/`MEMO_STATE_DIR` a tmp): `memo ops gc-vault-orphans --dry-run --json` sale 0 con `{"scanned": 0, ...}`; `memo ops exclude add notes a.md` + `exclude list --json` roundtrip.
- [ ] **Step 2:** FAIL
- [ ] **Step 3:** implementación click group, mirando `cli_dedupe.py` como referencia de estilo/registración; registrar en `cli.py`.
- [ ] **Step 4:** PASS + `uv run --no-sync pytest tests/ -q -k "ops or ingest_exclude or vault_ingest"` verde
- [ ] **Step 5:** commit `feat(cli): memo ops group (gc, vault-ingest, excludes)`

---

### Task 5: templates launchd + nightly script

**Files:**
- Create: `launchd/com.memo.recall-daemon.plist`, `launchd/com.memo.nightly.plist`, `launchd/com.memo.vault-ingest.plist`, `launchd/memo-nightly.sh`

Convención de `com.memo.dream.plist`: placeholders `__HOME__` / `__MEMO_BIN__` (+ `__CODEGRAPH_BIN__` en nightly.sh), comentario de instalación arriba. Valores de schedule/env/WatchPaths: **copiar de los plists vivos** `~/Library/LaunchAgents/com.synapse.{memo-recall-daemon,memo-nightly,vault-ingest}.plist` (schedule nightly, ThrottleInterval y WatchPaths de vault-ingest, env vars MEMO_* y KeepAlive del recall daemon), cambiando solo label, paths de log (`__HOME__/Library/Logs/memo/*.log`) y comandos:
- recall-daemon: `__MEMO_BIN__ recall-daemon _serve`, KeepAlive true.
- nightly: `/bin/sh __HOME__/.local/share/memo/bin/memo-nightly.sh`.
- vault-ingest: `__MEMO_BIN__ ops vault-ingest --json` (ya no hay shim shell con PYTHONPATH).

`memo-nightly.sh` (template; codegraph sync SIN synapse):

```sh
#!/bin/sh
# memo nightly maintenance — one wake, one MLX load. Template: replace
# __MEMO_BIN__ / __CODEGRAPH_BIN__; install to ~/.local/share/memo/bin/.
set -u
log() { echo "[memo-nightly $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
log "start codegraph-sync"
for r in memo memflow; do
  "__CODEGRAPH_BIN__" sync "$HOME/repos/$r" --quiet || "__CODEGRAPH_BIN__" unlock "$HOME/repos/$r"
done || log "codegraph-sync FAILED (exit $?)"
log "start contradict-scan"
"__MEMO_BIN__" contradict scan --max-memorias 500 --min-days-apart 3 --since "$(date -v-30d +%Y-%m-%d)" || log "contradict-scan FAILED (exit $?)"
log "start gc-memo-duplicates"
"__MEMO_BIN__" ops gc-memo-duplicates --json || log "gc-memo-duplicates FAILED (exit $?)"
log "start gc-vault-orphans"
"__MEMO_BIN__" ops gc-vault-orphans --json || log "gc-vault-orphans FAILED (exit $?)"
log "start memo-consolidate"
"__MEMO_BIN__" consolidate apply --auto-threshold 0.95 --max-clusters 15 || log "memo-consolidate FAILED (exit $?)"
log "done"
```

- [ ] Escribir los 4 archivos; commit `feat(launchd): native memo agents (recall, nightly, vault-ingest)`

---

### Task 6: PR + runtime reinstall

- [ ] `git push` + `gh pr create` (branch `docs/synapse-deprecation-spec` → master) con spec+plan+port; merge cuando checks verdes (`gh pr merge --auto --squash` o merge directo si la policy lo permite).
- [ ] Reinstalar runtime aislado desde el repo local: `uv tool install --from ~/repos/memo-spec mlx-memo --force --reinstall` → verificar `~/.local/bin/memo ops gc-vault-orphans --dry-run --json` responde.

---

### Task 7: snapshot + migración + swap de daemons

- [ ] Snapshot: `TS=$(date +%Y%m%dT%H%M%S); D=~/.memo-daemon-backups/$TS-synapse-final; mkdir -p $D; cp ~/Library/LaunchAgents/com.synapse.*.plist $D/; cp -R ~/.synapse $D/dot-synapse`
- [ ] Tag: `cd ~/repos/synapse && git tag deprecation-final && git push origin deprecation-final` (best-effort si hay remote)
- [ ] Migrar tombstones: `cp -R ~/.synapse/ingest_excludes <memo-state-dir>/` (state dir real: `cd ~/repos/memo-spec && uv run --no-sync python -c "from memo.config import Config; print(Config.from_env().state_dir)"`)
- [ ] Render de templates (sed `__HOME__`/`__MEMO_BIN__`/`__CODEGRAPH_BIN__`) → `~/Library/LaunchAgents/` + `~/.local/share/memo/bin/memo-nightly.sh` (chmod 755); `mkdir -p ~/Library/Logs/memo`
- [ ] Swap recall: `launchctl bootout gui/$(id -u)/com.synapse.memo-recall-daemon` → `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.memo.recall-daemon.plist` → verificar socket vivo y respuesta (`launchctl list | grep com.memo.recall`, log sin crash-loop, y una búsqueda memo que use el daemon)
- [ ] Bootstrap `com.memo.nightly` y `com.memo.vault-ingest`; corrida manual verde: `launchctl kickstart gui/$(id -u)/com.memo.vault-ingest` + `sh ~/.local/share/memo/bin/memo-nightly.sh` (o kickstart) revisando logs

---

### Task 8: teardown synapse

- [ ] `cd ~/repos/synapse && for s in dashboard runtime-loop eval-nightly memo-daemon memo-recall-daemon vault-ingest whatsapp-ingest morning-digest gc-vault-orphans gc-memo-duplicates contradict-scan memo-consolidate memo-nightly dream-synthesis watcher; do PYTHONPATH=src ~/repos/memflow/.venv/bin/python -m synapse.cli ops uninstall $s; done` (+ `dashboard-relay` si está en el registro; si no, bootout+rm manual del plist)
- [ ] Verificar `launchctl list | grep com.synapse` vacío; matar plists remanentes a mano (bootout + rm)
- [ ] Git hooks: revisar `.git/hooks/post-commit` (marker "Synapse git awareness hook") en `~/repos/{memo,memflow,synapse,consciousness-contracts}` y borrar los instalados por synapse
- [ ] MCP: `grep -l synapse ~/repos/*/.mcp.json ~/.mcp.json ~/.claude/settings*.json` → sacar entradas synapse
- [ ] `mv ~/.synapse $D/dot-synapse-live` (el estado ya está copiado; mover = nada queda activo)
- [ ] Archivar: `mkdir -p ~/repos/_archived && mv ~/repos/synapse ~/repos/consciousness-contracts ~/repos/_archived/`; `gh repo archive <owner>/<repo> --yes` para ambos si tienen remote

---

### Task 9: gates finales + docs + memoria

- [ ] Gates: `launchctl list | grep -E 'com\.(synapse|memo)\.'` → solo `com.memo.*` exit 0 · recall hook < 5s (correr el comando del hook de `~/.claude/settings.json` con `time`) · `memo doctor` verde · logs de nightly y vault-ingest sin errores · memflow MCP :18766 responde
- [ ] Actualizar `~/CLAUDE.md`: trinity → memo+memflow, tabla de venvs sin synapse, sección launchd reescrita (fleet `com.memo.*` + fuente launchd/ de memo), MCP sin synapse
- [ ] `memo save` del resultado (deprecación completada, qué murió, dónde está el backup) + actualizar memorias que referencien synapse como vivo
- [ ] Reporte final al usuario: qué murió sin reemplazo (dashboard, watcher, chat federado, whatsapp-ingest, morning-digest, gbrain import post-ingest, señal memflow post-ingest), dónde está el rollback

---

## Self-review

- Cobertura spec: decisiones tabla → Tasks 1-5 (port), 7 (adopción/swap), 8 (teardown+archivo), 9 (gates/docs). ✓
- Sin placeholders: código real en Tasks 1-3; Task 4-5 referencian archivos de convención concretos y valores copiables de plists vivos (fuente exacta indicada). ✓
- Consistencia de tipos: `find_vault_orphans`/`find_exact_duplicates` (T1) consumidos en T4; `IngestExcludeStore` (T2) en T3/T4; nombres estables. ✓
