# Deprecación big bang de synapse

**Fecha:** 2026-07-30
**Estado:** aprobado (brainstorming 2026-07-30)
**Alcance:** una sola pasada — memo adopta sus daemons, synapse se desinstala y archiva por completo, `consciousness-contracts` se archiva con él.

## Contexto

La trinity (memo + memflow + synapse) ya está desacoplada a nivel código:

- memflow es standalone desde `3b39a295fb` (`MEMFLOW_MEMO_BRIDGE` default OFF); cero referencias a synapse en `memflow/src`.
- memo no importa synapse: `definitive.py` lo lista en `_RETIRED_IMPORTS`; `resume/` inlineó lo que usaba. Quedan solo nombres de campos de metadata (`synapse_trace_id`, …) y strings de tests — se conservan tal cual.
- `consciousness_contracts` solo lo importa synapse (15 archivos; el hit en memo es el guard de retirados). Muere con synapse.

Lo que sigue vivo es infraestructura: el fleet launchd `com.synapse.*` que synapse instala y posee (fuente: `synapse/src/synapse/ops.py`), el dashboard :8765 y `~/.synapse/` (estado + scripts generados).

## Decisiones

| Pieza | Destino |
|---|---|
| `memo-recall-daemon` | **memo la adopta.** El plist ya ejecuta `/Users/fer/.local/bin/memo recall-daemon _serve` — solo cambia label/dueño a `com.memo.recall-daemon`. |
| `memo-nightly` + `vault-ingest` | **memo los adopta**, portando la poca lógica synapse-side (ver Port). |
| `whatsapp-ingest` + `morning-digest` | **Mueren.** Sin reemplazo. |
| dashboard + relay + watcher + runtime-loop + dream-synthesis + chat federado | **Mueren hoy.** Un dashboard/watcher memflow-nativo es una spec futura de memflow, fuera de este alcance. |
| `consciousness-contracts` | **Se archiva** junto con synapse (ningún otro repo lo importa). |
| `~/.synapse/` | **Se mueve al backup** (no se borra). |
| Repos `synapse` y `consciousness-contracts` | **Archive:** GitHub archive si hay remote + mover el checkout local a `~/repos/_archived/`. |

Secuenciación: **big bang** (elegida sobre strangler). Mitigación: snapshot completo previo + verify gates + rollback documentado.

## Port a memo (único trabajo de código)

Los 4 jobs nightly synapse-side son wrappers finos que shell-out al binario `memo`:

1. **`gc-vault-orphans`** (`ops.py:515`): borra registros cuyo `extra.abs_path` ya no existe, solo `source=vault-ingest`. Lógica pura sobre `memo list --json`.
2. **`gc-memo-duplicates`** (`ops.py:595`): borra duplicados exactos por SHA-256 del body, conserva el más nuevo. Lógica pura sobre `memo list --json`.
3. **`contradict-scan`** (`ops.py:681`): wrapper de `memo contradict scan --max-memorias 500 --min-days-apart 3 --since <30d>` + `memo contradict list`. Casi solo re-wiring.
4. **`vault-ingest`** (`ops.py:806`): por vault, `git pull --ff-only` best-effort + `memo ingest <root> --name <label> --prune` con excludes de sistema + tombstones de `IngestExcludeStore`.

Se portan a memo como subcomandos CLI nativos (grupo `memo ops` o equivalente según convención del repo), con tests. `IngestExcludeStore` se porta a memo y su archivo de estado migra de `~/.synapse/` a `~/.local/share/memo/` (migración de datos incluida: si el archivo viejo existe, copiarlo).

`memo-nightly.sh` se regenera memo-nativo, mismo orden, sin `PYTHONPATH=synapse`:
codegraph sync (igual que hoy) → `memo contradict scan` → gc-dupes → gc-orphans → `memo consolidate apply --auto-threshold 0.95 --max-clusters 15`.

Plists nuevos con templates en `memo/launchd/` (mismo patrón `__HOME__`/`__MEMO_BIN__` que `com.memo.dream`): `com.memo.recall-daemon`, `com.memo.nightly`, `com.memo.vault-ingest`.

## Orden de ejecución

1. **Snapshot:** copiar `~/Library/LaunchAgents/com.synapse.*.plist` + `~/.synapse/` completo a `~/.memo-daemon-backups/<timestamp>-synapse-final/`. Tag `deprecation-final` en el repo synapse.
2. **Port + tests en memo** (branch → PR; master protegido).
3. **Swap recall-daemon:** `launchctl bootout` del viejo → bootstrap `com.memo.recall-daemon` → verificar `recall.sock` responde y el hook queda bajo el budget de 5s. Ventana de segundos.
4. **Instalar** `com.memo.nightly` + `com.memo.vault-ingest`; corrida manual verde de cada uno.
5. **Teardown:** `synapse ops uninstall <svc>` para todo `SERVICE_TO_LABEL` (usar su propio tooling mientras existe). Desinstalar los git-hooks de awareness que synapse instaló en los repos. Sacar synapse de todo `.mcp.json` donde figure.
6. **Archivo:** mover `~/.synapse/` al backup; archivar repos (GitHub + `~/repos/_archived/`).
7. **Docs/memoria:** actualizar `~/CLAUDE.md` (trinity → memo+memflow, venvs, fleet) y guardar el resultado en memo.

## Criterios de verificación (gates)

- `launchctl list` sin ningún `com.synapse.*`; `com.memo.{recall-daemon,nightly,vault-ingest}` con exit 0.
- Sesión Claude nueva: recall hook responde < 5s, sin errores de hooks.
- `memo doctor` verde.
- Corrida manual de nightly y vault-ingest verde de punta a punta.
- memflow MCP (:18766) intacto.
- `grep -r "synapse" ~/.claude/settings*.json ~/.mcp.json <repos>/.mcp.json` sin referencias activas.

## Rollback

Re-bootstrap desde los plists del snapshot (`launchctl bootstrap gui/$UID <plist>`); `~/.synapse/` restaurable desde el backup; el repo synapse queda intacto en `_archived` con tag `deprecation-final`. Nada se borra de forma irreversible.

## Fuera de alcance

- Dashboard/watcher memflow-nativos (spec futura en memflow).
- Reemplazo de whatsapp-ingest / morning-digest / chat federado.
- Limpieza de campos de metadata `synapse_*` en memo (`contracts.py`) — son datos históricos, se quedan.
