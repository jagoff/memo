# Checklist vivo — absorción completa de Memflow en Memo

Última actualización: 2026-07-29  
Branch: `feat/memflow-absorption`  
Worktree: `/Users/fer/repos/memo/.worktrees/memflow-absorption`

## Estado global

- [x] Diseño de producto y arquitectura aprobados.
- [x] Cinco planes de implementación escritos y auditados.
- [x] Worktree aislado creado.
- [x] Baseline verde: `5809 passed, 18 skipped`.
- [ ] Plan 01 — Operational Ledger v2: **Tarea 1 aceptada; Tarea 2 con
  re-revisión `FAIL` y segunda ronda de hardening en preparación**.
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
- [ ] RED de markers atómicos, finalización, anchor history y root único.
- [ ] Implementar y verificar hardening ronda 2.
- [ ] Corregir todo hallazgo `BLOCKER/HIGH/MEDIUM`.
- [ ] Revisión independiente y `PASS`.

### Tarea 3 — Vistas SQLite transaccionales e idempotencia

- [ ] Generar brief y registrar BASE.
- [ ] RED de transacción, replay, crash window e idempotency conflict.
- [ ] Implementar schema, reducers, catch-up, rebuild y commit de nueve pasos.
- [ ] Probar bypass de epoch en CLI, MCP, daemon y librería.
- [ ] Focused pytest, ruff y mypy.
- [ ] Commit, revisión independiente y `PASS`.

### Tarea 4 — Migración genesis v1 determinista y gate de paridad

- [ ] Generar brief y registrar BASE.
- [ ] RED de corrupción, source drift, determinismo e idempotencia.
- [ ] Implementar plan/apply/verify sin tocar v1.
- [ ] Probar paridad exacta y ausencia de activation stamp prematuro.
- [ ] Focused pytest, ruff y mypy.
- [ ] Commit, revisión independiente y `PASS`.

### Tarea 5 — Outbox durable exactly-once

- [ ] Generar brief y registrar BASE.
- [ ] RED de crash antes/después de save, retry y collision.
- [ ] Implementar identidad estable, reconciliación y provenance.
- [ ] Probar rebuild entre requested/completed y cero duplicados.
- [ ] Focused pytest, ruff y mypy.
- [ ] Commit, revisión independiente y `PASS`.

### Tarea 6 — Sesiones canónicas sobre ledger v2

- [ ] Generar brief y registrar BASE.
- [ ] RED de lifecycle monotónico, merge y artifacts locales.
- [ ] Implementar servicio de sesiones y cache JSON derivado.
- [ ] Reemplazar nombres públicos `mem_session_*` por `memo_session_*`.
- [ ] Focused pytest, ruff y mypy.
- [ ] Commit, revisión independiente y `PASS`.

### Tarea 7 — Activar facade operacional v2 tras paridad completa

- [ ] Generar brief y registrar BASE.
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

Estado: **0/6 aceptadas**.

- [ ] Tarea 1 — Snapshot seguro, capability manifest e inventario.
- [ ] Tarea 2 — Fencing de requests Memflow en todo boundary de mutación.
- [ ] Tarea 3 — Drain observable y startup refusal.
- [ ] Tarea 4 — Aislar Synapse del runtime Memflow.
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
| P01-T03 | Vistas SQLite e idempotencia | — | — | — | — | — | — | — |
| P01-T04 | Migración genesis y paridad | — | — | — | — | — | — | — |
| P01-T05 | Outbox durable exactly-once | — | — | — | — | — | — | — |
| P01-T06 | Sesiones canónicas | — | — | — | — | — | — | — |
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
| P03-T01 | Snapshot, manifest e inventario | — | — | — | — | — | — | — |
| P03-T02 | Fencing de requests Memflow | — | — | — | — | — | — | — |
| P03-T03 | Drain y startup refusal | — | — | — | — | — | — | — |
| P03-T04 | Aislamiento de Synapse | — | — | — | — | — | — | — |
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
