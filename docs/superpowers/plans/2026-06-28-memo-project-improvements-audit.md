# Memo Project Improvement Audit Plan

**Fecha:** 2026-06-28
**Alcance:** repo completo `memo` en `/Users/fer/repos/memo`
**Tipo:** auditoria tecnica + plan de mejoras, sin cambios de codigo de producto

## Evidencia Ejecutada

- Codegraph indexado: 426 archivos, 7.673 nodos, 13.536 edges.
- Codigo Python en `src/memo`: 62.250 lineas.
- Tests Python en `tests`: 29.705 lineas.
- `uv run --no-sync ruff check src/ tests/`: PASS.
- `uv run --no-sync mypy src/memo`: PASS, 250 source files.
- `uv run --no-sync memo config validate`: PASS, 1 flag seteado y valido.
- `uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --maxfail=1 -q`: PASS, 1644 passed, 29 skipped, 15.95s.
- `uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force`: interrumpido tras mas de 90s sin progreso visible; dejo ademas un `KeyboardInterrupt` durante `GraphStore.__del__`.
- Worktree ya venia sucio con version 2.3.5 en `.claude-plugin/plugin.json`, `CHANGELOG.md`, `pyproject.toml`, `server.json`; no se modifico ni revirtio.

## Lectura General

El proyecto no esta roto: lint, tipos y suite rapida pasan. El problema de mayor valor no es "arreglar tests", sino reducir complejidad acumulada en las superficies que mas cambian: MCP/CLI, retrieval, capture, flags/config, runtime/update, y lifecycle de recursos. La base tiene buena cobertura y muchos fixes recientes, pero varias decisiones todavia viven como convencion en comentarios o docs, no como contratos ejecutables.

## Plan Prioritario, Linea Por Linea

### P0 - Cerrar el estado de release antes de mas refactors

**Riesgo:** el repo esta en medio de una version 2.3.5 no commiteada. Cualquier auditoria que ignore eso puede mezclar release pendiente con mejoras estructurales.

- `pyproject.toml:8`: version actual del paquete esta en `2.3.5`.
- `.claude-plugin/plugin.json:4`: version sincronizada a `2.3.5`.
- `server.json:10` y `server.json:15`: version del MCP/package sincronizada a `2.3.5`.
- `CHANGELOG.md:10`: se agrego `2.3.5 - 2026-06-28`.
- `src/memo/cli_release.py:85-125`: `plan_release_edits()` ya sincroniza los cuatro archivos.
- `src/memo/cli_release.py:115`: el helper aun inserta `TODO: describe changes`; debe haber un guard que impida finalizar release con ese placeholder.

**Mejora:** agregar `memo release check` que valide version sincronizada, seccion de changelog presente, cero `TODO` en la seccion nueva, tag `vX.Y.Z` ausente/presente segun modo, y formula/docs relacionados actualizados.

**Verificacion:** test unitario sobre `plan_release_edits()` + test CLI de `release check`; luego `ruff`, `mypy`, `pytest -m "not slow"`.

### P1 - Hacer que el gate de retrieval sea usable en iteracion diaria

**Riesgo:** el eval formal existe, pero una corrida fresca con el corpus real tardo mas de 90s sin salida visible. Como gate humano/pre-commit eso es demasiado opaco.

- `eval/regression_labels.json:1-243`: hay un corpus comprometido de prompts.
- `eval/regression_labels.json:10-16`: solo el primer prompt tiene `expect_ids` real.
- `eval/regression_labels.json:18-133`: muchos prompts relevantes tienen `expect_ids: []`, entonces caen al heuristico de terminos.
- `src/memo/eval_recall.py:162-168`: cada eval compara 4 configs por defecto.
- `src/memo/eval_recall.py:234-243`: por cada prompt/config se llama `mem.search()`.
- `src/memo/cli_eval.py:107-110`: `--gate` y `--update-baseline` fuerzan corrida fresca.
- `src/memo/cli_eval.py:131-142`: el cache existe, pero no aplica para gate.

**Mejora:** dividir el eval en dos niveles:

1. `memo eval recall --quick --gate`: un solo config, labels con `expect_ids`, progreso por prompt y timeout por prompt.
2. `memo eval recall --matrix`: la comparativa cara de 4 configs para tuning manual.

**Verificacion:** fixture pequena sin MLX real para CI; test de timeout/progreso; prueba manual del matrix en Mac.

### P2 - Convertir retrieval en pipeline testeable por etapas

**Riesgo:** `Memory.search()` concentra demasiada politica en un metodo largo: candidate generation, RRF, feedback, health, rerank, decay, cache, entity boost, penalties, graph expansion, access logging y body resolution.

- `src/memo/memory/search_ops.py:32-48`: firma publica de `search()` ya acepta muchos controles.
- `src/memo/memory/search_ops.py:121-151`: rama vec/hybrid y fallback de embedder.
- `src/memo/memory/search_ops.py:152-240`: adaptive pool, BM25, exact, graph candidates y RRF.
- `src/memo/memory/search_ops.py:252-288`: materializacion de records y bodies desde FTS.
- `src/memo/memory/search_ops.py:289-321`: forgotten filter y source feedback.
- `src/memo/memory/search_ops.py:322-362`: health pre-rerank, skip rerank, rerank.
- `src/memo/memory/search_ops.py:363-390`: recency decay y cache read-through.
- `src/memo/memory/search_ops.py:391-437`: entity boost, contradiction penalty, graph expansion, retrieval boost, co-recall boost.
- `src/memo/memory/search_ops.py:438-459`: access logging, co-recall writes y body re-resolution.

**Mejora:** crear `SearchPlan`/`SearchContext` y funciones puras por etapa:

- `generate_candidates()`
- `fuse_candidates()`
- `materialize_candidates()`
- `apply_rank_modifiers()`
- `finalize_hits()`

El objetivo no es cambiar ranking, sino hacer cada etapa observable y testeable sin abrir modelos.

**Verificacion:** golden tests de `search_with_trace()` antes/despues; precision/noise igual en eval quick; perf no peor en p50.

### P3 - Partir capture.py por responsabilidades

**Riesgo:** `capture.py` es el archivo mas grande del repo y mezcla parser de transcript, prompt LLM, quality gates, dedup semantico, estado global y watermarks.

- `src/memo/capture.py:1-44`: docstring describe toda la pipeline en un solo modulo.
- `src/memo/capture.py:61-108`: trigger patterns.
- `src/memo/capture.py:111-134`: prompt del extractor.
- `src/memo/capture.py:137-159`: estado global `last-capture.json`.
- `src/memo/capture.py:162-279`: parsing de transcript y armado de exchanges.
- `src/memo/capture.py:282-349`: extraccion de texto y actividad de tools.
- `src/memo/capture.py:352-397`: prefilter y quality gate.
- `src/memo/capture.py:403-462`: llamada LLM + parse JSON.
- `src/memo/capture.py:465-595`: dedup + save.
- `src/memo/capture.py:598-668`: Stop-hook capture.
- `src/memo/capture.py:671-847`: incremental capture + watermarks.

**Mejora:** split conservador:

- `capture_transcript.py`: parseo y tool evidence.
- `capture_quality.py`: prefilter, quality, dedup thresholds.
- `capture_extract.py`: prompt y parsing LLM.
- `capture_state.py`: state file, watermark, flock.
- `capture.py`: facade publica compatible.

**Verificacion:** mover tests existentes sin cambiar expectativas; agregar test de compatibilidad importando `memo.capture` como hoy.

### P4 - Registrar superficie MCP/CLI por perfil, no registrar todo y remover despues

**Riesgo:** el MCP crea muchas herramientas para luego borrar parte segun perfil. Eso mantiene acoplamiento con nombres y aumenta riesgo de que una nueva tool quede visible por accidente.

- `src/memo/server.py:136-166`: registra modulos avanzados si perfil full.
- `src/memo/server.py:173-177`: siempre registra core/history/idle/resources.
- `src/memo/server.py:179-182`: remueve tools despues de registrarlas.
- `src/memo/surface.py:45-60`: set `AGENT_MCP_TOOLS`.
- `src/memo/surface.py:62-90`: set `CORE_MCP_TOOLS`.
- `src/memo/surface.py:121-125`: calcula removals para perfil agent.
- `src/memo/cli.py:124-180`: secciones manuales de help.
- `src/memo/cli.py:259-378`: 120 lineas de `cli.add_command(...)`.

**Mejora:** un registry declarativo con metadata por comando/tool: `name`, `profile`, `stability`, `module`, `register`. `build_server()` debe registrar solo lo incluido por perfil. `cli.py` debe iterar el mismo concepto para comandos.

**Verificacion:** tests de conteo y allowlist por perfil; `memo doctor` debe reportar el conteo real, no estimado.

### P5 - Alinear docs de "experimental" con la realidad publica

**Riesgo:** README vende features como core, mientras `experimental_index.md` marca varias como experimentales o no cableadas. Eso confunde a usuarios y a agentes.

- `README.md:122-164`: contradiccion, sintesis, sync, graph y multimodal aparecen como features de producto.
- `src/memo/experimental_index.md:23-99`: multimodal, collaborative, contradict, chunker, crossref, contextual, lifecycle, navigation, proactive, sync y versioning figuran como experimentales.
- `src/memo/experimental_index.md:45-51`: dice que `chunker.py` no esta wired into reindex.
- `src/memo/memory/maintain_ops.py:207-221`: ya hay camino de chunk ingest por flag.
- `src/memo/config.py:225-233`: `single_db` se documenta como feature real.

**Mejora:** definir tres estados en docs y surface metadata:

- `stable`: contrato soportado.
- `advanced`: soportado, pero opt-in/costoso.
- `experimental`: puede cambiar y no aparece por defecto.

**Verificacion:** test que compara `experimental_index.md`/registry para evitar drift de documentacion.

### P6 - Consolidar ownership de flags y eliminar lecturas raw evitables

**Riesgo:** el repo tiene una regla explicita de no leer `MEMO_*` con `os.environ.get()` fuera de config/flags, pero hay excepciones dispersas. Algunas son justificadas por tri-state, pero deberian estar modeladas.

- `src/memo/flags.py:1-21`: define el contrato de registry central.
- `src/memo/flags.py:73-115`: accesores tipados.
- `src/memo/flags.py:124-148`: unknown vars excluye config-owned vars.
- `src/memo/flags_misc.py:5-674`: un solo archivo contiene demasiados dominios.
- `src/memo/memory/facade.py:97-103`: raw `MEMO_EMBEDDER_VIA_DAEMON` para tri-state.
- `src/memo/store/queries.py:623`: raw `MEMO_SOFT_DELETE`.
- `src/memo/runtime/autoupdate.py:213-217`: raw `MEMO_AUTO_UPDATE` para default-on tri-state.
- `src/memo/config.py:481-488`, `517-529`, `569-573`: raw env en config, aceptable por ownership pero repetitivo.
- `src/memo/config.py:55` y `src/memo/config.py:410`: `SYSTEM_DIR` import-time vs lectura runtime.

**Mejora:** agregar APIs:

- `flag_raw(name) -> str | None`
- `flag_is_set(name) -> bool`
- `flag_bool_default_on(name) -> bool`
- `ConfigEnv` helper para env-to-field.

Luego mover `MEMO_SOFT_DELETE` a `flag_bool`, documentar solo las excepciones de config.

**Verificacion:** test de arquitectura que falle ante `os.environ.get("MEMO_` fuera de allowlist.

### P7 - Encapsular accessos directos a conexiones sqlite

**Riesgo:** las capas superiores entran al `_conn` del store. Eso rompe la frontera que el propio AGENTS define: storage entrypoint es `VecStore`, writes por `_tx()`.

- `src/memo/memory/write_ops.py:379-382`: `save()` consulta `self.store._conn` para `topic_key`.
- `src/memo/memory/write_ops.py:789-792`: `_read_body()` consulta FTS via `self.store._conn`.
- `src/memo/store/queries.py:257-270`: ya existe `get_fts_bodies()`.
- `src/memo/store/queries.py:389-411`: ya existe `get_by_path()`/`get_by_path_ci()`.
- `src/memo/store/queries.py:485-518`: `list_recent()` recalcula PRAGMA `table_info(meta)` en cada llamada.

**Mejora:** agregar metodos `VecStore.find_by_topic_key()`, `VecStore.get_fts_body_by_path()`, y cachear columnas disponibles en init/migrations. Las capas memory no deberian tocar `_conn`.

**Verificacion:** test `test_architecture_boundaries.py` para prohibir `store._conn` fuera de `src/memo/store`.

### P8 - Endurecer lifecycle de recursos y destructores

**Riesgo:** al interrumpir `memo eval recall` aparecio `Exception ignored while calling deallocator ... GraphStore.__del__ ... KeyboardInterrupt`. `contextlib.suppress(Exception)` no cubre `KeyboardInterrupt`.

- `src/memo/graph.py:367-373`: `close()` + `__del__()`.
- `src/memo/history.py:276`: otro `__del__`.
- `src/memo/store/connection.py:31`: otro `__del__`.
- `src/memo/store/store.py:150`: otro `__del__`.
- `src/memo/crossref.py:265`: otro `__del__`.
- `src/memo/contradict.py:185`: otro `__del__`.
- `src/memo/versioning.py:199`: otro `__del__`.
- `src/memo/memory/facade.py:440-480`: `Memory.close()` ya existe y es el lugar correcto para teardown explicito.

**Mejora:** eliminar `__del__` donde sea posible, usar ownership explicito (`Memory.close()`, context managers) y `weakref.finalize` solo donde haga falta. Si queda destructor, debe ser minimo y no emitir advertencias bajo `KeyboardInterrupt`.

**Verificacion:** test que crea/cierra stores; smoke que interrumpe eval/command y no deja warning de destructor.

### P9 - Runtime/update: cerrar huecos de path, shim y scanner

**Riesgo:** la instalacion y auto-update ya tienen fixes recientes, pero siguen teniendo convenciones dispersas y scanners incompletos.

- `src/memo/runtime/autoupdate.py:196-263`: auto-update en startup.
- `src/memo/runtime/autoupdate.py:253-259`: `Popen([sys.executable, "-m", "memo.cli", "update", ...])`; correcto para mismo runtime, pero requiere evidencia en doctor/log.
- `src/memo/runtime/update.py:176-184`: prewarm usa `shutil.which("memo") or sys.executable`, que puede elegir otro shim/runtime si PATH cambia.
- `src/memo/runtime/update.py:234-319`: hay duplicacion entre `--to-tag` y update normal por tag.
- `src/memo/runtime/mcp_config.py:14-20`: scanner revisa Claude, Devin, OpenCode y mcp-gateway.
- `src/memo/runtime/install.py:146-153` y `204-210`: Codex se instala, pero no aparece en `KNOWN_MCP_CONFIGS`.
- `src/memo/cli_doctor.py:69-77`: doctor reporta paths MCP fragiles.

**Mejora:** refactor de update en una funcion `install_spec(method, spec)`; scanner MCP debe incluir Codex y Windsurf, y exponer JSON estable. `doctor --strict-runtime` debe mostrar runtime usado por auto-update y prewarm.

**Verificacion:** tests de runtime con paths fake para pipx/uv/codex/windsurf; test de log de auto-update sin red real.

### P10 - Tratar `memo eval recall` como producto, no como script interno

**Riesgo:** el eval es clave para no parchear queries una por una, pero hoy tiene fricciones de UX.

- `src/memo/cli_eval.py:64-88`: opciones actuales.
- `src/memo/cli_eval.py:193-207`: detalle solo se imprime al final.
- `src/memo/eval_recall.py:288-299`: tabla final simple.
- `src/memo/eval_recall.py:302-340`: recomendacion tiene presupuesto de hook, pero no timeout real.
- `src/memo/eval_recall.py:390-400`: fingerprint usa count + mtime; barato, pero no captura cambios de flags/config que afectan ranking.

**Mejora:** progreso incremental, `--timeout-per-prompt`, `--config`, cache key con flags de ranking activos, y salida parcial recuperable si se interrumpe.

**Verificacion:** tests sin modelos reales + manual matrix en Mac.

### P11 - Ratchet de typing/cobertura por modulos calientes

**Riesgo:** el suite pasa, pero el coverage floor global es bajo para un producto que depende de hooks y runtime local.

- `pyproject.toml:86-96`: mypy strict solo cubre `memo.config`, `memo.flags`, `memo.util`.
- `pyproject.toml:98-103`: phase-2 strict cubre `memo.recall_server`, `memo.memory.search_ops`, `memo.memory.write_ops`.
- `pyproject.toml:108-111`: coverage floor global `fail_under = 58`.
- `.github/workflows/test.yml:36-43`: Linux CI corre ruff, mypy, pytest con coverage.
- `.github/workflows/macos-smoke.yml:36-46`: macOS smoke cubre doctor y dos tests reales MLX.

**Mejora:** ratchet por dominios:

- subir strict typing para `capture`, `runtime.update`, `runtime.autoupdate`, `surface`, `server`.
- coverage floor por carpeta o por modulo caliente, no solo global.
- agregar smoke real de `memo eval recall --quick` en macOS sin descargar todo el mundo.

**Verificacion:** CI Linux + macOS smoke.

### P12 - Documentacion/versiones: eliminar drift visible

**Riesgo:** la documentacion principal y artefactos de distribucion pueden quedar atrasados aunque el release helper sincronice cuatro archivos.

- `README.md:7`: headline dice `memo 2.0` mientras package esta en `2.3.5`. Puede ser marca, pero se lee como version.
- `docs/homebrew/mlx-memo.rb:19-20`: formula de referencia sigue apuntando a `mlx_memo-2.3.3.tar.gz` con sha de 2.3.3.
- `docs/reference.md:1-20`: reference manual es front door tecnico.
- `src/memo/cli_release.py:85-125`: release helper no toca README ni Homebrew reference formula.

**Mejora:** decidir si `memo 2.0` es marca o version. Si es marca, cambiar a "memo" y mover version real a badge/metadata. Agregar al release check validacion opcional de formula de referencia.

**Verificacion:** test o script `memo release check --docs` que detecte drift de formula/docs.

### P13 - Politica de errores: menos `except Exception` silencioso en core estable

**Riesgo:** hay muchos fallbacks correctos en hooks, pero tambien se normalizo el patron de absorber excepciones. En core estable, un fallback debe dejar evidencia.

- `src/memo/util.py:66`: ya documenta reemplazo de silent `except Exception: pass` por logging observable.
- `src/memo/memory/facade.py:429-431`: probe legacy paths absorbe todo, justificado pero sin trace.
- `src/memo/capture.py:39-43`: failure modes "All swallowed silently" salvo debug.
- `src/memo/capture.py:584-587`: save failed solo imprime en debug.
- `src/memo/memory/write_ops.py:601-629`: varios `contextlib.suppress` en recovery, razonable pero auditable.
- `src/memo/cli_doctor.py:257-269`: doctor ya convierte adoption failures en salida visible.

**Mejora:** clasificar errores por dominio:

- hook hot path: swallow permitido, pero escribir log bounded.
- core CRUD/storage: usar `MemoError` o propagar.
- recovery paths: registrar warning con id/path.
- debug-only print no alcanza para perdida de captura.

**Verificacion:** tests que simulan fallas y verifican log/resultado.

## Orden Recomendado

1. P0 release check: pequeno, evita publicar inconsistencia.
2. P1/P10 eval quick: desbloquea medicion barata para retrieval.
3. P2 retrieval pipeline: mayor impacto tecnico, pero hacerlo con eval quick ya disponible.
4. P4 surface registry: reduce tokens y evita exposiciones accidentales.
5. P6/P7 flags + store boundaries: mejora arquitectura sin cambiar UX.
6. P3 capture split: mejora mantenibilidad del archivo mas grande.
7. P8 destructors: bug de teardown observado durante esta auditoria.
8. P9 runtime/update scanner: endurece instalacion multi-cliente.
9. P5/P12 docs/stability: alinear contrato publico.
10. P11/P13 ratchets: consolidar que la deuda no vuelva.

## No Hacer Todavia

- No reescribir ranking antes de tener `eval recall --quick`.
- No mover todos los comandos CLI a la vez; migrar por dominios.
- No borrar experimental code sin decidir estado `stable/advanced/experimental`.
- No tocar los cambios locales de version 2.3.5 salvo que el objetivo sea terminar ese release.

