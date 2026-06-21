# Guía de daemons del sistema (memo / synapse / memflow)

> Qué corre en segundo plano alrededor de memo, quién es dueño de cada cosa, para
> qué sirve, su costo de memoria, y cómo sanear/revertir. Auditado el 2026-06-20.

## 0. TL;DR — el "sprawl de daemons de memo" en realidad era un bug de synapse

Lo que parecían "muchos daemons de memo descontrolados" eran **~13 agentes launchd que instala SYNAPSE** (no memo), la mayoría **crasheando en loop** por un solo bug: el instalador de synapse (`_detect_python` en `synapse/ops.py`) elegía `/Users/USER/repos/memo/.venv` como intérprete, pero **ese venv no tiene `synapse` ni `consciousness_contracts`** → `ModuleNotFoundError` en cada arranque. Solo memo aporta 1 agente nativo (`com.memo.dream`) + el recall-daemon.

**Causa raíz + fix (2026-06-20):** `_detect_python()` ahora **valida que el venv pueda importar `consciousness_contracts`** antes de elegirlo → prefiere `/Users/USER/repos/memflow/.venv` (el dev-venv completo de esta máquina). Aplicado en fuente (`synapse/ops.py`) y en los 8 plists vivos. Resultado: dashboard:8765, watcher y runtime-loop pasaron de **exit 1 (crash-loop) → running limpio**.

## 1. Inventario completo de daemons

Dueño: **M**=memo nativo · **S**=synapse · **F**=memflow. Tipo: **R**=residente (KeepAlive) · **P**=periódico (StartCalendarInterval/Interval).

| Agente launchd | Dueño | Tipo | Comando | Schedule | Para qué sirve |
|---|---|---|---|---|---|
| `com.synapse.memo-recall-daemon` | S→M | R | `memo` recall socket (embedder 4B/2560d warm) | KeepAlive | Mantiene el embedder caliente sobre `recall.sock` para que el recall-hook entre en el presupuesto de 5s. **El consumidor de RAM grande (~2.6 GB) y es esperado.** |
| `com.synapse.dashboard` | S | R | `synapse … dashboard-serve :8765` | KeepAlive | Dashboard de observabilidad + chat federado de synapse (el que usás). **Estaba roto, ahora HTTP 200.** |
| `com.synapse.dashboard-relay` | S | R | `relay_8766.py` | KeepAlive | Relay del dashboard (:8766). |
| `com.synapse.watcher` | S | R | `synapse.watcher` | KeepAlive | Reactive Consciousness Engine: dispara pases de runtime ante cambios de estado. **Estaba roto, ahora running.** |
| `com.synapse.runtime-loop` | S | P (60s) | `synapse runtime loop` | cada 60s | Snapshot periódico health/replay/eval/trust. **Estaba roto, ahora running.** Cadencia agresiva (spawn/min) — candidato a subir el intervalo. |
| `com.memo.dream` | **M** | P | `memo dream run` | diario 03:00 | Análisis "dream" de memo: insights proactivos cross-cluster. Único agente **nativo de memo**. |
| `com.synapse.memo-consolidate` | S→M | P | `memo consolidate apply --auto-threshold 0.95` | Dom 03:00 | Consolida/dedup memorias de memo (semanal). |
| `com.synapse.gc-memo-duplicates` | S | P | `synapse ops gc-memo-duplicates` | Dom 02:00 | GC de duplicados en memo. |
| `com.synapse.gc-vault-orphans` | S | P | `synapse ops gc-vault-orphans` | diario **03:40** ⬅ staggered | GC de huérfanos del vault. **Movido de 03:00 → 03:40** para no solapar el cold-load MLX con `memo-dream` 03:00. |
| `com.synapse.contradict-scan` | S | P | `synapse ops contradict-scan` | Mié 02:00 | Escanea contradicciones en memo. |
| `com.synapse.dream-synthesis` | F | P | memflow dream analyze+synthesize | diario 02:15 | Dream de **memflow** (no memo). Distinto de `memo-dream`. |
| `com.synapse.morning-digest` | S | P | `synapse.morning_digest` | diario 08:00 | Digest diario: conflictos, insights, health → Memflow. |
| `com.synapse.vault-ingest` | S | R/Watch | `vault-ingest.sh` (git pull + `memo ingest`) | WatchPaths | Ingesta de los 2 vaults Obsidian a la superficie *memory* de memo. |
| `com.synapse.whatsapp-ingest` | S→M | P | `memo import whatsapp --all-chats` | diario | Ingesta de WhatsApp a memo. |

**memo-mcp NO es un daemon:** es un MCP **stdio**, hijo de cada sesión de Claude Code (ver `docs/` / `.mcp.json`). Muere con la sesión; no aparece en launchd.

## 2. Memoria — cómo no consumir de más

El costo real **no** es el número de daemons residentes (la mayoría son livianos), sino los **cold-loads de MLX** de los jobs periódicos: `memo dream`, `consolidate`, `gc-*`, `contradict-scan` y `dream-synthesis` cargan embedder 4B (~2.5 GB) y a veces el LLM 30B por unos minutos y salen.

- **Residente esperado:** `memo-recall-daemon` ~2.6 GB (embedder 4B warm). Es el precio del recall-hook < 5s. No es leak.
- **Pico evitado:** antes había `memo-dream` + `gc-vault-orphans` (+ `consolidate` los domingos) **simultáneos a las 03:00** → 2-3 cold-loads MLX en paralelo. Staggered: `gc-vault-orphans` → 03:40. `gc-memo-duplicates` (Dom 02:00) y `contradict-scan` (Mié 02:00) ya están aislados por día de la semana.
- **Reducción de churn:** los 3 crash-loopers (dashboard/watcher/runtime-loop) reiniciaban sin parar bajo KeepAlive → CPU/log desperdiciados en loop. Al arreglarse, dejan de reiniciar.

**Verificar memoria:** `/bin/ps -axo pid,rss,comm | sort -k2 -rn | head` (usar `/bin/ps`, no el alias `procs`). Buscar `*/.venv/bin/python3` con RSS alto.

## 3. El fix de raíz (durable)

`synapse/src/synapse/ops.py` → `_detect_python()` (~línea 88). Antes elegía el primer venv que **existiera**; ahora elige el primero que **pueda importar `consciousness_contracts`** (probe `python -c "import consciousness_contracts"`), priorizando `/Users/USER/repos/memflow/.venv`. Un solo helper alimenta todos los builders de plist, así que arregla la flota entera.

Para regenerar todos los plists desde la fuente arreglada:
```bash
cd /Users/USER/repos/synapse && PYTHONPATH=src /Users/USER/repos/memflow/.venv/bin/python -m synapse.cli ops install --all
```

## 4. Saneamiento aplicado (2026-06-20) — todo reversible

1. **Fix venv** en `_detect_python` (fuente) + en 8 plists vivos: `memo/.venv → memflow/.venv` (contradict-scan, dashboard, dream-synthesis, gc-memo-duplicates, gc-vault-orphans, morning-digest, runtime-loop, watcher).
2. **Stagger** `gc-vault-orphans` 03:00 → 03:40.
3. **Reload** de los 8 (`launchctl bootout`+`bootstrap`). dashboard/watcher/runtime-loop: exit 1 → running (PID asignado, exit 0). dashboard:8765 → HTTP 200.

**Backups:** todos los plists originales en `~/.memo-daemon-backups/<timestamp>/`.

**Revertir un agente:**
```bash
cp ~/.memo-daemon-backups/<TS>/com.synapse.<x>.plist ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u)/com.synapse.<x>; launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.synapse.<x>.plist
```

## 5. Consolidación del maintenance nocturno (APLICADO 2026-06-20)

**4 agentes → 1.** `memo-consolidate` + `gc-memo-duplicates` + `gc-vault-orphans` + `contradict-scan` se fusionaron en un único `com.synapse.memo-nightly` (diario 03:00) que corre los 4 pasos **en secuencia** vía wrapper `~/.synapse/bin/memo-nightly.sh` → un solo cold-load MLX a la vez en vez de 4 solapados. Durable en fuente: `build_memo_nightly_plist`/`build_memo_nightly_script` + entrada `memo-nightly` en `SERVICE_TO_LABEL` (`synapse/ops.py`). Los 4 builders/labels viejos quedan (instalables individualmente con `synapse ops install <svc>`); los plists vivos se removieron (backups en `~/.memo-daemon-backups/`).

```bash
# logs del run nocturno
tail -f ~/.synapse/state/logs/memo-nightly.log
# revertir a los 4 agentes separados
cd ~/repos/synapse && for s in memo-consolidate gc-memo-duplicates gc-vault-orphans contradict-scan; do PYTHONPATH=src ~/repos/memflow/.venv/bin/python -m synapse.cli ops install $s; done
PYTHONPATH=src ~/repos/memflow/.venv/bin/python -m synapse.cli ops uninstall memo-nightly  # o launchctl bootout
```

### Resueltos (2026-06-20)
- **`runtime-loop` 60s → 300s.** `build_runtime_loop_plist(start_interval_seconds=300)` en `synapse/ops.py` + plist vivo regenerado (`StartInterval=300`). Menos spawns de intérprete por hora.
- **Dream NO está duplicado.** `com.memo.dream` = `memo dream run` → pipeline de memo (temporal/contradicción/evolución) sobre el corpus de **memo**. `com.synapse.dream-synthesis` = `memflow.dream.analyze` sobre `~/repos/memflow` → corpus de **memflow**. Mismo nombre, **stores soberanos distintos**; mergear violaría la invariante de la trinity (Memo=memoria semántica / Memflow=estado vivo). **Se quedan ambos.**

## 6. Verificación
```bash
launchctl list | grep -E "com.(synapse|fer)" | grep -v "voicememo"   # 2ª col = last exit (0 = ok, 1 = roto)
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8765/      # dashboard synapse
PYTHONPATH=/Users/USER/repos/synapse/src /Users/USER/repos/memflow/.venv/bin/python -c "from synapse.ops import _detect_python; print(_detect_python())"
```
