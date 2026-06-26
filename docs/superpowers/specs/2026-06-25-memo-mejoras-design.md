# memo — 5 mejoras de alto valor (design)

**Fecha:** 2026-06-25
**Estado:** propuesta aprobada (set), pendiente de planes de implementación por ítem
**Alcance:** repo `memo` (`~/repos/memo`), runtime instalado y configs MCP del entorno

## Contexto

Se analizó el proyecto sobre el código real (no solo memoria de sesión): markers
`TODO/DEPRECATED/not-wired`, tamaño de archivos, superficie de tools del MCP,
disciplina de regresión de retrieval, y los dos gotchas de runtime resueltos hoy
(versión reusada ocultando un build viejo; configs MCP hardcodeando la ruta
interna del venv pipx). De ahí salen 5 mejoras, una por eje de valor, cada una
**independiente y verificable**. Cada ítem se implementa con su propio plan
(spec → plan → implementación); este documento es el menú priorizado.

Convención del repo respetada: archivos < 800 líneas; markdown es source of
truth; toda regresión de retrieval se arregla de forma sistémica y se mide
contra el set de labels, nunca por-pregunta; versión sincronizada en 4 archivos
(`pyproject.toml`, `.claude-plugin/plugin.json`, `server.json`, `CHANGELOG.md`).

## Orden recomendado

`M2 → M1 → M4 → M3 → M5` — arranca por lo de menor riesgo que previene el dolor
de hoy y baja el costo diario; deja M5 (el más pesado) al final.

---

## M1 · Perfil `agent` real del MCP + deduplicar memo MCP

**Eje:** costo / UX del MCP.

**Problema.** `memo doctor` reporta `MEMO_MCP_PROFILE=agent ~118 tools
(~35k tokens/conexión)`, cuando el perfil `agent` debería exponer ~5 tools
(`surface.py`, `MEMO_MCP_PROFILE`). El recorte no está surtiendo efecto en el
runtime. Además hay **dos** memo MCP corriendo en paralelo (uno en
`~/.claude.json`, otro vía el plugin `plugin:memo`), duplicando el costo de
tokens por sesión. En opencode esto ya causó un prefill de ~51s al exponer 100+
tools.

**Cambio.**
1. Verificar y corregir `surface.py` para que `agent` realmente limite el
   set de tools registradas en `build_server()` al núcleo previsto.
2. Colapsar a **un solo** memo MCP en el entorno. Canónico recomendado: el
   **plugin** `plugin:memo` (ya invoca `memo-mcp` vía PATH → shim estable
   `~/.local/bin`, así que sobrevive cambios de runtime); remover la entrada
   `memo` manual de `~/.claude.json`. Esto también elimina la duplicación de
   tools.

**Valor.** Reduce el overhead por sesión de ~70k a ~2.4k tokens y elimina la
clase de prefill multi-segundo en clientes con presupuesto de tools.

**Verificación.** Test en `surface.py`/`server.py` que cuenta las tools
surfaced por perfil y asegura `agent ≤ N` (N = tamaño del set agent). Smoke:
`MEMO_MCP_PROFILE=agent memo-mcp` lista solo el núcleo. Confirmar en `doctor`
que la cuenta cae.

**Esfuerzo:** S-M. **Riesgo:** bajo (el gate de superficie ya existe; el resto
es config).

---

## M2 · Guardrails de release/runtime en `memo doctor`

**Eje:** seguridad de release.

**Problema.** Hoy aparecieron dos fallas sin guardia:
- la versión `1.0.12` se reusó entre dos contenidos distintos → el build
  instalado quedó viejo con el mismo número y nadie lo detectó;
- 4 configs MCP (claude, devin, opencode, mcp-gateway) apuntaban a la ruta
  **interna** del venv pipx; al desinstalar pipx quedaron colgadas.

**Cambio.**
1. Check `instalado ≠ source`: comparar versión + content-hash (o build-stamp
   embebido en el wheel) del paquete instalado contra el HEAD del repo; advertir
   en `doctor` cuando divergen con la misma versión.
2. Validación de paths MCP: `doctor` escanea los configs MCP conocidos y marca
   todo `command` que apunte a un binario inexistente o a una ruta interna de
   venv (pipx/uv) en vez del shim estable `~/.local/bin/<bin>` o el nombre en
   PATH.
3. Helper `memo release --bump <level>`: sincroniza atómicamente los 4 archivos
   de versión + agrega esqueleto de entrada en `CHANGELOG.md`, evitando el drift
   manual.

**Valor.** Previene exactamente las dos fallas de hoy; convierte el "parece
actualizado pero no" en una advertencia accionable.

**Verificación.** Tests que disparan cada check con un mismatch sintético: build
viejo con misma versión → warning; config con ruta muerta → warning; `--bump`
deja los 4 archivos en la misma versión y el CHANGELOG con la sección nueva.

**Esfuerzo:** M. **Riesgo:** bajo (todo aditivo en doctor + un helper).

---

## M3 · Split de god-files vía `/demonolith-split`

**Eje:** salud de código.

**Problema.** 4 archivos superan el límite de 800 líneas del propio repo:
`cli_dream.py` (915), `memory/search_ops.py` (866), `session.py` (856),
`memory/ask_ops.py` (801).

**Cambio.** Usar el workflow existente `/demonolith-split` para partir cada uno
en submódulos cohesivos, empezando por `cli_dream.py`. Behavior-preserving: sin
cambios en la API pública ni en el comportamiento observable.

**Valor.** Mantenibilidad, edits más confiables (archivos que entran en
contexto), y se cumple la regla de tamaño del repo.

**Verificación.** `pytest` verde antes y después, `mypy` limpio, `ruff` limpio,
y diff que demuestra que solo se movió código (sin cambios de lógica). Hacer un
archivo por PR para mantener el diff revisable.

**Esfuerzo:** M (por archivo). **Riesgo:** bajo-medio (re-exports para no romper
imports existentes).

---

## M4 · Purga de código muerto / flags `DEPRECATED`

**Eje:** dead-code.

**Problema.** Varios code-paths y flags marcados muertos o no cableados inflan la
superficie: GLiNER (entity extraction, no wired), cache modes
`smart/prefetch/aggressive` (`MEMO_CACHE_MODE`, dead — solo `off` está cableado),
`sharing.py` (host placeholder, no live service), `cli_federation` (no wired),
`time_machine` (stub), encryption (no cableada al save/search por defecto).

**Cambio.** Remover los code-paths muertos. Para flags ya públicos, mantener el
**nombre** del flag como aceptado-pero-no-op durante una release (para no romper
`memo config validate` de instalaciones que lo tengan seteado), documentar la
deprecación en CHANGELOG, y remover el nombre en una release posterior.

**Valor.** Menos superficie de config y menos confusión sobre qué está vivo;
menos para mantener y testear.

**Verificación.** `pytest` + `config validate` verdes; `grep` confirma 0
referencias vivas a los símbolos removidos; CHANGELOG lista las deprecaciones.

**Esfuerzo:** S-M. **Riesgo:** bajo (se remueve lo que ya no se ejecuta;
los flags quedan no-op un ciclo).

---

## M5 · Endurecer el gate de retrieval + cerrar 1 pieza Sleep-time

**Eje:** calidad de retrieval.

**Problema.** Existe disciplina de regresión (`eval/regression_labels.json`,
`memo eval recall`) pero el baseline es mínimo (prec@5=0.2 para prompts de
respuesta única, noise@5=0.0). Varias piezas de Sleep-time Compute están a medias
(flags `MEMO_DREAM_*` off o parciales: co-recall graph edges, query-prediction
pre-synthesis, eviction, verbose compression, pre-warm).

**Cambio.**
1. Crecer el set de labels (cada incidente de recall agrega un prompt etiquetado:
   `expect_ids` + `noise_*`).
2. Cablear `memo eval recall` como **gate real** (precision↑ / noise↓ sobre todo
   el set) en el flujo de commit/CI local — falla si la precisión cae o el ruido
   sube.
3. Cerrar **una** pieza de alto valor medida contra el set. Recomendado:
   **co-recall graph edges** (`MEMO_GRAPH_CO_RECALL`) — registra pares de
   memorias co-recuperadas para mejorar ranking relacional; alternativa:
   query-prediction pre-synthesis.

**Valor.** Calidad de recall medible y sistémica (no por-pregunta); cierra deuda
de Sleep-time con evidencia.

**Verificación.** El gate corre en CI local y bloquea regresiones; se reporta el
delta prec@5 / noise@5 antes/después de la pieza elegida; la pieza tiene su
propio test unitario.

**Esfuerzo:** M-L. **Riesgo:** medio (toca el hot-path de recall; mitigado por el
gate y por mantener la pieza detrás de su flag).

---

## Notas de implementación

- Cada mejora es un plan independiente: implementar de a uno, verde antes de
  pasar al siguiente, commit/push por ítem.
- M1 y M2 tocan también el entorno (configs MCP fuera del repo); esos cambios van
  documentados pero no se commitean al repo `memo`.
- M3 y M5 son los que más se benefician de la disciplina de tests-verdes-antes-y-
  después; M5 además exige medir contra el set de labels.
