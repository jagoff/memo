# Checklist vivo — absorción completa de Memflow en Memo

Última actualización: 2026-07-30
Branch: `feat/memflow-absorption`
Worktree: `/Users/fer/repos/memo/.worktrees/memflow-absorption`

## Ejecución en tiempo real

Último checkpoint: 2026-07-30 10:08 America/Argentina/Cordoba

### En curso

- [x] Revisión independiente de especificación, APIs y tests de Plan 01
  Tareas 2–6 — `T2 PASS`, `T3 FAIL`, `T4 FAIL`, `T5 PASS`, `T6 FAIL`.
- [ ] Revisor continuo read-only de corrección y durabilidad para cada lote
  nuevo — agente `review_plan01_security`.
- [ ] Revisión independiente cross-repo de Plan 03 Tareas 1–4 — agente
  `review_plan03`; preliminar: T1/T2/T3 `FAIL`, T4 `HIGH`.
- [ ] Preparar TDD de Plan 01 Tarea 7: selector fail-closed v1/v2, fresh
  install v2 y activation stamp firmado — agente principal.
- [ ] Corregir `P01-T03 HIGH`: una aplicación incremental con backfill
  multi-origen puede divergir del rebuild por orden global. Agregar regresión
  `newest(a) -> older(b)` y re-reducción transaccional determinista.
- [x] Corrección técnica `P01-T02 HIGH`: TOCTOU entre verificación del ledger v1 y la
  relectura usada por manifest/anchor. Capturar una sola instantánea
  descriptor-relative con `O_NOFOLLOW` y derivar de ella bytes, eventos, heads
  y manifest. Focused `70 passed`; Ruff y mypy limpios; re-revisión pendiente.
- [ ] Resolver `P01-T07 BLOCKER`: `MacOSKeychainProvider` es un placeholder
  fail-closed; el fresh install v2 productivo no puede crear la clave,
  roster y epoch-0. Implementar un provider productivo no exportable antes de
  habilitar activación.
- [ ] Re-revisar `P01-T04 MEDIUM`: implementación técnica lista; fsync del
  staging antes del rename, crash pre-publish y retry verificados.
- [ ] Corregir `P01-T06 HIGH`: adapters públicos deben propagar timestamp
  `None` y reutilizar el resultado canónico para retry idempotente después de
  crash — agente `review_plan01_spec` reasignado como implementador.
- [ ] Corregir `P01-T06 MEDIUM`: preservar get por prefijo/ambigüedad y
  artifacts locales en reads canónicos locales, sin federarlos — mismo agente.
- [ ] Completar el gate diferido `P01-T03 MEDIUM`: stale/missing epoch por
  MCP/CLI/daemon/Memory/direct store queda bloqueante de T7.
- [ ] Corregir `P03-T01`: manifest fail-closed ante symlinks y receipts stale;
  validar snapshot bytes/target/mode, cobertura/fixtures y transform allowlist.
- [ ] Corregir `P03-T02/T03`: cubrir todos los writers CLI/auxiliares y
  reverificar criptográficamente el control OID durante drain.
- [ ] Corregir `P03-T04 HIGH`: construir desde snapshot/commit inmutable o
  revalidar el repo después del build antes de publicar runtime.
- [ ] Reemplazar expiries fijas de tests P03 que ya vencieron; matriz actual
  `221 passed, 3 failed` por tiempo de fixture.

### Completado en esta ejecución

- [x] Alcance fijado: absorber uso vivo de 90 días y dependencias
  indispensables; eliminar capacidades sin uso junto con Memflow.
- [x] Estrategia fijada: revisores y subagentes paralelos con ownership no
  superpuesto.
- [x] P03-T03 drenaje observable entregado en Memflow `a3a6070912`.
- [x] Evidencia P03-T03 registrada en Memo `e4de56db`.
- [x] Memflow productivo verificado sin mutaciones: PID `2046`, binario y
  listener `127.0.0.1:18766` sin cambios.
- [x] Gate fresco de Plan 01 foundation:
  `187 passed in 24.08s`.
- [x] Skills de implementación activas: `memo`, `python-testing` y
  `python-patterns`.
- [x] Codegraph consultado antes de preparar Tarea 7; el índice pertenece al
  checkout principal y se contrastó con el worktree mediante lectura local.
- [x] Brief de `P01-T07` congelado con BASE `e4de56db`, selector cerrado,
  fresh-install transaccional, activation binding y RED contracts.
- [x] Revisión de especificación P01-T2–T6: `107 passed`; fallas semánticas
  adversariales registradas aunque la suite cubierta esté verde.
- [x] `P01-T04` RED confirmado por falta de fsync pre-rename; GREEN:
  regresión `1 passed`, migración+definitive `33 passed`, Ruff y mypy limpios.
- [x] `P01-T02` GREEN: snapshot descriptor-relative único, identidad exacta
  del descriptor, detección de reemplazo same-size y un solo consumo en
  plan/anclaje; `70 passed`, Ruff y mypy limpios.

### Próximos gates

- [ ] Consolidar hallazgos de los tres revisores sin duplicados.
- [ ] Cerrar `P01-T03 HIGH` y obtener re-revisión PASS del revisor continuo.
- [ ] Cerrar `P01-T02 HIGH` y obtener re-revisión PASS del revisor continuo.
- [ ] Cerrar el provider productivo de claves y obtener revisión de seguridad
  antes de escribir cualquier activation stamp.
- [ ] Obtener re-revisión PASS de `P01-T04 MEDIUM`.
- [ ] Cerrar ambos hallazgos `P01-T06` y obtener re-revisión PASS.
- [ ] Asignar correcciones BLOCKER/HIGH/MEDIUM a agentes con archivos
  separados.
- [ ] Obtener PASS de re-revisión para cada tarea corregida.
- [ ] Implementar y aceptar Plan 01 Tarea 7.
- [ ] Ejecutar suite completa, Ruff, mypy y frozen-v1.
- [ ] Congelar manifest firmado con evidencia real de ambas Macs.
- [ ] Implementar Plan 02 Tareas 1–11.
- [ ] Completar Plan 03 Tareas 5–6.
- [ ] Ejecutar Plan 04: migración del estado activo.
- [ ] Ejecutar Plan 05: cutover atómico, reinicios, VERIFIED y retiro.
- [ ] Eliminar Memflow sólo después de la auditoría de independencia final.

## Estado global

- [x] Diseño de producto y arquitectura aprobados.
- [x] Cinco planes de implementación escritos y auditados.
- [x] Worktree aislado creado.
- [x] Baseline verde: `5809 passed, 18 skipped`.
- [ ] Plan 01 — Operational Ledger v2: **Tarea 1 aceptada; Tareas 2–6
  implementadas y verdes, pendientes de revisión independiente**.
- [ ] Plan 02 — Runtime nativo de coordinación viva.
- [ ] Plan 03 — Readiness para corte.
- [ ] Plan 04 — Migración del estado activo.
- [ ] Plan 05 — Corte atómico y retiro de Memflow.
- [ ] Despliegue productivo.
- [ ] Verificación de uso vivo exclusivamente vía Memo.
- [ ] Baja definitiva de Memflow.

Progreso aceptado: **1/35 tareas**.

## Gate obligatorio para cada tarea

Cada tarea debe completar todos estos pasos antes de marcarse terminada:

- [ ] Consultar Memo con `source="codex"`.
- [ ] Leer brief y contratos normativos.
- [ ] Consultar Codegraph antes de editar.
- [ ] Escribir pruebas primero.
- [ ] Ejecutar y registrar RED real.
- [ ] Implementar sólo el alcance asignado.
- [ ] Ejecutar focused pytest.
- [ ] Ejecutar ruff.
- [ ] Ejecutar mypy.
- [ ] Ejecutar suite de regresión proporcional al riesgo.
- [ ] Revisar diff y archivos congelados.
- [ ] Commit con paths explícitos; nunca `git add -A`.
- [ ] Crear paquete de revisión desde el BASE registrado.
- [ ] Obtener revisión independiente de especificación y calidad.
- [ ] Corregir todos los hallazgos `BLOCKER/HIGH/MEDIUM`.
- [ ] Obtener `PASS` de re-revisión.
- [ ] Actualizar ledger, este checklist y Memo.

## Plan 01 — Operational Ledger v2

Estado: **1/7 aceptadas**.

### Tarea 1 — Congelar v1 y definir contratos puros v2

- [x] Brief generado.
- [x] BASE registrado: `d9ed37a6`.
- [x] RED inicial: 8 errores de colección esperados.
- [x] Implementación inicial: `f03d7418`.
- [x] Revisión 1: FAIL; autoridad epoch/roster/signing incompleta.
- [x] Hardening ronda 1: `00aeb750`.
- [x] Revisión 2: FAIL; rollback coordinado, capability y payloads v1.
- [x] Pin externo y recovery ronda 2: `2b545cd3`.
- [x] Revisión 3: FAIL; namespace caller-controlled y capability.
- [x] Composición sellada ronda 3: `c0bd6861`.
- [x] Revisión 4: FAIL; capability todavía forjable.
- [x] Implementador fresco asignado.
- [x] RED capability: `7 failed, 17 deselected`.
- [x] Capability autenticada: `54760a8a`.
- [x] Revisión de ronda 4: FAIL; lifecycle ligado a `id(fence)`.
- [x] RED lifecycle causal: `1 failed, 19 deselected`.
- [x] Self-review RED de inmutabilidad: `1 failed, 20 deselected`.
- [x] Nonce CSPRNG inmutable por instancia: `0edbc2ed`.
- [x] Focused de la tarea: `119 passed`.
- [x] Focused del brief: `30 passed`.
- [x] Ruff: limpio.
- [x] Mypy: limpio.
- [x] Non-slow: `5928 passed, 18 skipped, 7 deselected`.
- [x] Compatibilidad v1 y archivos congelados: intactos.
- [x] Compatibilidad pública y roster pinneado: `c36f3566`.
- [x] Refresh latest/historical y revocación live: `54b48b9e`.
- [x] Paquete final generado para `d9ed37a6..54b48b9e`.
- [x] Revisión independiente final: `PASS`.
- [x] Auditoría de contrato público: `PASS`.
- [x] Sin `BLOCKER/HIGH/MEDIUM` abiertos.
- [x] Tarea marcada completa en el ledger.

### Tarea 2 — Anchors, append, verificación y bundles locales v2

- [x] Generar brief y registrar BASE `8e85662b`.
- [x] Asignar implementador fresco `plan01_task02_ledger_impl`.
- [x] RED de ledger ausente: error de colección `ModuleNotFoundError`.
- [x] RED de lock epoch continuo: `1 failed`.
- [x] Implementar `OperationLedgerV2`.
- [x] Agregar `EpochFence.verified(context)` para mantener el lock durante
  append, fsync y actualización del head.
- [x] Focused v2 + compatibilidad v1 + epoch: `56 passed`.
- [x] Ruff: limpio.
- [x] Mypy: limpio.
- [x] Non-slow completo: `5965 passed, 18 skipped`.
- [x] Commit explícito: `5a80c74d`.
- [x] Paquete de revisión generado para `8e85662b..5a80c74d`.
- [x] Revisión independiente de contrato y calidad: `FAIL`, con
  4 `HIGH` y 2 `MEDIUM` reproducidos.
- [x] Auditoría independiente de seguridad y durabilidad: `FAIL`, con
  1 `BLOCKER`, 5 `HIGH` y 1 `MEDIUM` reproducidos.
- [x] Auditoría independiente de API y fortaleza de tests: `FAIL`, con
  3 `HIGH` y 2 `MEDIUM` reproducidos.
- [x] Hallazgos consolidados sin duplicados: 1 `BLOCKER`, 7 `HIGH` y
  3 `MEDIUM`.
- [x] Blueprint de hardening de provenance, autoridad, recovery y filesystem.
- [x] Brief TDD generado con BASE `0b1c859d`.
- [x] Implementador único asignado: `task02_hardening_impl`.
- [x] Escribir y registrar los 18 escenarios RED de hardening.
- [x] RED global: `36 failed, 76 passed` antes de tocar producción.
- [x] Implementar provenance autocontenida y snapshot de autoridad.
- [x] Implementar I/O descriptor-relative, transacciones y recovery durable.
- [x] Task 1 + Task 2: `186 passed`.
- [x] Frozen-v1: `3 passed`.
- [x] Ruff: PASS en 10 paths.
- [x] Mypy: PASS en 6 módulos.
- [x] Non-slow completo: `5995 passed, 18 skipped`.
- [x] Commit técnico de hardening con paths explícitos: `815307ac`.
- [x] Paquetes de revisión hardening y Task 2 completo generados.
- [x] Re-revisión final sobre `815307ac`: `FAIL`, con 3 `HIGH` y
  2 `MEDIUM` consolidados.
- [x] Brief TDD de hardening ronda 2 generado con BASE `123cd8f6`.
- [x] RED de markers atómicos, finalización, anchor history y root único:
  `9 failed, 2 passed`.
- [x] Implementar y verificar hardening ronda 2 en `ed454393`.
- [x] Corregir los 3 `HIGH`, 2 `MEDIUM` y quality guards de la ronda.
- [x] Task 1 + Task 2: `200 passed`.
- [x] Frozen v1: `3 passed` y sin diff.
- [x] Ruff y mypy: limpios.
- [x] Non-slow completo: `6009 passed, 18 skipped`.
- [x] Reporte de implementación generado.
- [x] Corrección técnica del `HIGH` TOCTOU: una captura verificada alimenta
  eventos, heads y manifest; rechaza drift de identidad, metadata o paths.
- [x] Regresiones de mezcla verify/reread y reemplazo same-size.
- [x] Focused ledger+migration: `70 passed`; Ruff y mypy limpios.
- [ ] Revisión independiente y `PASS`.

### Tarea 3 — Vistas SQLite transaccionales e idempotencia

- [x] Generar brief y registrar BASE `c9127f39`.
- [x] RED de transacción, replay, crash window e idempotency conflict.
- [x] Implementar schema, reducers, catch-up, rebuild y commit de nueve pasos.
- [x] Probar bypass directo, integración ledger/fence real, epoch
  stale/future, control OID, contexto ausente y actor incorrecto.
- [x] Focused `22 passed`, acumulado `256 passed`, ruff y mypy limpios.
- [x] Non-slow: `6032 passed, 18 skipped`.
- [x] Commit técnico: `daf3bf36`.
- [ ] Activar propagación autenticada y probar bypass en CLI/MCP/daemon al
  seleccionar v2 en Tarea 7.
- [ ] Revisión independiente y `PASS`.

### Tarea 4 — Migración genesis v1 determinista y gate de paridad

- [x] Generar brief y registrar BASE `42f458c0`.
- [x] RED real: `ModuleNotFoundError` para `memo.operation_migration`.
- [x] RED de corrupción, source drift, determinismo e idempotencia.
- [x] Implementar plan/apply/verify sin tocar v1.
- [x] Probar paridad exacta, reapertura con plan exacto y ausencia de
  activation stamp prematuro.
- [x] Focused `10 passed`; matriz requerida `100 passed`; acumulado `276
  passed`.
- [x] Ruff y mypy limpios; frozen-v1 sin diff.
- [x] Non-slow: `6043 passed, 18 skipped, 7 deselected`.
- [x] Commit técnico: `78764d74`.
- [x] Reporte de implementación generado.
- [ ] Revisión independiente y `PASS`.

### Tarea 5 — Outbox durable exactly-once

- [x] Generar brief y registrar BASE `20e16ba4`.
- [x] Consultar Memo y Codegraph antes de editar producción.
- [x] RED real: 2 errores de colección por `memo.durable_outbox` ausente.
- [x] RED de crash antes/después de save, retry y collision.
- [x] Implementar identidad estable, reconciliación y provenance.
- [x] Probar rebuild entre requested/completed y cero duplicados.
- [x] Focused `490 passed`; matriz requerida `500 passed`; acumulado `676
  passed`.
- [x] Ruff y mypy limpios; frozen-v1 sin diff.
- [x] Non-slow: `6079 passed, 18 skipped`.
- [x] Commit técnico: `24f7a406`.
- [x] Reporte de implementación generado.
- [ ] Revisión independiente y `PASS`.

### Tarea 6 — Sesiones canónicas sobre ledger v2

- [x] Generar brief y registrar BASE `6b68a260`.
- [x] Consultar Memo y Codegraph antes de editar producción.
- [x] RED de lifecycle monotónico, merge y artifacts locales.
- [x] Implementar servicio de sesiones y cache JSON derivado.
- [x] Reemplazar nombres públicos `mem_session_*` por `memo_session_*`.
- [x] Focused `586 passed`; acumulado Plan 01 `817 passed`; Ruff y mypy
  limpios.
- [x] Commit técnico: `ecf2b951`.
- [x] Reporte de implementación generado.
- [ ] Revisión independiente y `PASS`.

### Tarea 7 — Activar facade operacional v2 tras paridad completa

- [x] Generar brief y registrar BASE `e4de56db`.
- [ ] RED de activation stamp, fresh install y partial install.
- [ ] Implementar selector de backend v1/v2 fail-closed.
- [ ] Integrar facade, federation y errores MCP tipados.
- [ ] Ejecutar gates completos del Plan 01.
- [ ] Confirmar cero mutación de v1.
- [ ] Commit, revisión final del plan y `PASS`.

## Plan 02 — Runtime nativo de coordinación viva

Estado: **0/11 aceptadas**.

- [ ] Tarea 1 — Congelar fixtures de paridad seleccionados de Memflow.
- [ ] Tarea 2 — Coordinación, handoffs y tasks nativos.
- [ ] Tarea 3 — Delivery, ACK, retries y cursors.
- [ ] Tarea 4 — Presence, heartbeat y conflictos de workspace.
- [ ] Tarea 5 — Extender sesiones y composición de continuidad.
- [ ] Tarea 6 — Bridge de terminal controlado.
- [ ] Tarea 7 — Sync operacional firmado sobre Git.
- [ ] Tarea 8 — Retención acotada y compactación firmada de prefijos.
- [ ] Tarea 9 — Writer control, lifecycle del daemon y health.
- [ ] Tarea 10 — Exponer únicamente APIs nativas de Memo.
- [ ] Tarea 11 — Probar el runtime distribuido end-to-end.

Gate del plan:

- [ ] Todas las capacidades vivas seleccionadas tienen paridad probada.
- [ ] No hay imports runtime de Memflow.
- [ ] Todos los writes pasan por ledger/epoch/idempotencia de Memo.
- [ ] Pruebas multi-peer, crash, retry, offline y reconnect verdes.
- [ ] Revisión final independiente del plan.

## Plan 03 — Readiness para corte

Estado: **0/6 aceptadas; Tareas 1–4 implementadas y verdes, pendientes de
revisión independiente**.

- [ ] Tarea 1 — Snapshot seguro, capability manifest e inventario.
  Implementación técnica: `f6ca3fff`…`7858d540`; focused `32 passed`;
  suite completa `6138 passed, 18 skipped`. Falta revisión independiente y
  `PASS`.
- [ ] Tarea 2 — Fencing de requests Memflow en todo boundary de mutación.
  Implementación técnica: `22299647` + `2c643863`; focused `34 passed`;
  paridad `211 passed`; suite completa `1888 passed`. Falta revisión
  independiente y `PASS`.
- [ ] Tarea 3 — Drain observable y startup refusal. Implementación técnica:
  `a3a6070912`; matriz requerida `224 passed`; suite completa `1906 passed`;
  Ruff, mypy, instalador y diff limpios. Falta revisión independiente y
  `PASS`.
- [ ] Tarea 4 — Aislar Synapse del runtime Memflow. Implementación técnica:
  `933445fd`; focused `18 passed`; suite completa `2326 passed, 5 skipped`;
  build real e idempotente verificado. Falta revisión independiente y `PASS`.
- [ ] Tarea 5 — Reemplazar el contrato Memflow de Synapse por registry Memo.
- [ ] Tarea 6 — Stagear configuración de consumidores y readiness report.

Gate del plan:

- [ ] Inventario de consumidores completo.
- [ ] Todas las mutaciones antiguas pueden cercarse.
- [ ] Drain medible y sin writes ocultos.
- [ ] Cada consumidor tiene configuración Memo preparada.
- [ ] Readiness report firmado y verde.
- [ ] Revisión final independiente del plan.

## Plan 04 — Migración del estado activo

Estado: **0/5 aceptadas**.

- [ ] Tarea 1 — Escanear y probar las tres fuentes de migración.
- [ ] Tarea 2 — Importar conocimiento durable faltante vía políticas Memo.
- [ ] Tarea 3 — Traducir y aplicar estado activo determinísticamente en staging.
- [ ] Tarea 4 — Crear y probar rollback bundle pre-epoch.
- [ ] Tarea 5 — Verificar y ensayar migración entre dos peers aislados.

Gate del plan:

- [ ] Sólo se migra estado activo/vivo aprobado.
- [ ] Traducción determinista e idempotente.
- [ ] Paridad de estado y provenance verificadas.
- [ ] Rollback pre-epoch probado.
- [ ] Ensayo multi-peer completo.
- [ ] Revisión final independiente del plan.

## Plan 05 — Corte atómico y retiro de Memflow

Estado: **0/6 aceptadas**.

- [ ] Tarea 1 — Votes firmados, estado monotónico y control CAS.
- [ ] Tarea 2 — Controller fail-closed y staging descartable.
- [ ] Tarea 3 — Switching atómico de clientes y prueba rollback pre-epoch.
- [ ] Tarea 4 — Verificación global y auditoría de independencia permanente.
- [ ] Tarea 5 — Ejecutar el activation epoch ensayado.
- [ ] Tarea 6 — Retirar Memflow y remover maquinaria temporal.

Gate final de producto:

- [ ] Todos los clientes cambian atómicamente a Memo.
- [ ] No quedan writers ni readers vivos sobre Memflow.
- [ ] Memo procesa coordinación viva, delivery, presence, sessions y sync.
- [ ] Verificación global post-epoch verde.
- [ ] Ventana de observación sin fallback ni dependencia oculta.
- [ ] Uso vivo exclusivamente nativo de Memo.
- [ ] Memflow detenido.
- [ ] Launch agents, MCP, hooks, config y credenciales Memflow retirados.
- [ ] Código temporal de migración removido.
- [ ] Auditoría permanente de independencia verde.
- [ ] Despliegue productivo documentado.
- [ ] Memo actualizado como única fuente de verdad del resultado.

## Matriz de ejecución en tiempo real

Esta tabla se actualiza al iniciar y cerrar cada tarea. `—` significa que el
paso todavía no comenzó, no que esté aprobado.

| ID | Tarea | Owner | BASE | RED | GREEN | Commit | Review | Deploy |
|---|---|---|---|---|---|---|---|---|
| P01-T01 | Contratos y v1 congelado | múltiples implementadores especializados | `d9ed37a6` | 7 rondas registradas | `134`; full `5943` | `f03d7418`…`54b48b9e` | **PASS final** | — |
| P01-T02 | Anchors, append y bundles | implementadores + revisores especializados | `8e85662b`; hardening `0b1c859d`; ronda 2 `123cd8f6` | inicial + hardening `36 failed, 76 passed`; ronda 2 pendiente | contracts `186`; full `5995` | `5a80c74d` + `815307ac` | **FAIL** ronda 2: 3H/2M | — |
| P01-T03 | Vistas SQLite e idempotencia | Codex | `c9127f39` | import RED | focused `22`; acumulado `256`; full `6032` | `daf3bf36` | pendiente | — |
| P01-T04 | Migración genesis y paridad | Codex | `42f458c0` | import RED | focused `10`; matriz `100`; acumulado `276`; full `6043` | `78764d74` | pendiente | — |
| P01-T05 | Outbox durable exactly-once | Codex | `20e16ba4` | import RED | focused `490`; matriz `500`; acumulado `676`; full `6079` | `24f7a406` | pendiente | — |
| P01-T06 | Sesiones canónicas | Codex | `6b68a260` | — | — | — | pendiente | — |
| P01-T07 | Activación facade v2 | — | — | — | — | — | — | — |
| P02-T01 | Fixtures de paridad Memflow | — | — | — | — | — | — | — |
| P02-T02 | Coordinación, handoffs y tasks | — | — | — | — | — | — | — |
| P02-T03 | Delivery, ACK, retry y cursors | — | — | — | — | — | — | — |
| P02-T04 | Presence, heartbeat y conflictos | — | — | — | — | — | — | — |
| P02-T05 | Continuidad de sesiones | — | — | — | — | — | — | — |
| P02-T06 | Terminal bridge controlado | — | — | — | — | — | — | — |
| P02-T07 | Sync operacional firmado | — | — | — | — | — | — | — |
| P02-T08 | Retención y compactación | — | — | — | — | — | — | — |
| P02-T09 | Writer, daemon y health | — | — | — | — | — | — | — |
| P02-T10 | APIs exclusivamente Memo | — | — | — | — | — | — | — |
| P02-T11 | Runtime distribuido E2E | — | — | — | — | — | — | — |
| P03-T01 | Snapshot, manifest e inventario | Codex | `90a7144e` | import RED | focused `32`; matriz `60`; full `6138` | `f6ca3fff`…`7858d540` | pendiente | — |
| P03-T02 | Fencing de requests Memflow | Codex | Memflow `5426e8e5` | import RED | focused `34`; paridad `211`; full `1888` | `22299647` + `2c643863` | pendiente | — |
| P03-T03 | Drain y startup refusal | Codex | Memflow `2c643863` | pendiente | en curso | — | — | — |
| P03-T04 | Aislamiento de Synapse | Codex | Synapse `45c146d5` | import RED | focused `20`; full `2326` | `933445fd` | pendiente | — |
| P03-T05 | Registry backend de Memo | — | — | — | — | — | — | — |
| P03-T06 | Configuración y readiness | — | — | — | — | — | — | — |
| P04-T01 | Probar inputs de migración | — | — | — | — | — | — | — |
| P04-T02 | Import durable por política | — | — | — | — | — | — | — |
| P04-T03 | Traducción y staging apply | — | — | — | — | — | — | — |
| P04-T04 | Rollback bundle pre-epoch | — | — | — | — | — | — | — |
| P04-T05 | Ensayo entre dos peers | — | — | — | — | — | — | — |
| P05-T01 | Votes, monotonicidad y CAS | — | — | — | — | — | — | — |
| P05-T02 | Controller fail-closed | — | — | — | — | — | — | — |
| P05-T03 | Switching atómico | — | — | — | — | — | — | — |
| P05-T04 | Verificación e independencia | — | — | — | — | — | — | — |
| P05-T05 | Activation epoch | — | — | — | — | — | — | — |
| P05-T06 | Retiro total de Memflow | — | — | — | — | — | — | — |
