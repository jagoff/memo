# Checklist vivo — absorción completa de Memflow en Memo

Última actualización: 2026-07-29  
Branch: `feat/memflow-absorption`  
Worktree: `/Users/fer/repos/memo/.worktrees/memflow-absorption`

## Estado global

- [x] Diseño de producto y arquitectura aprobados.
- [x] Cinco planes de implementación escritos y auditados.
- [x] Worktree aislado creado.
- [x] Baseline verde: `5809 passed, 18 skipped`.
- [ ] Plan 01 — Operational Ledger v2: **Tarea 1 en revisión final**.
- [ ] Plan 02 — Runtime nativo de coordinación viva.
- [ ] Plan 03 — Readiness para corte.
- [ ] Plan 04 — Migración del estado activo.
- [ ] Plan 05 — Corte atómico y retiro de Memflow.
- [ ] Despliegue productivo.
- [ ] Verificación de uso vivo exclusivamente vía Memo.
- [ ] Baja definitiva de Memflow.

Progreso aceptado: **0/35 tareas**. La Tarea 1 tiene implementación y gates
verdes, pero no se contabiliza como terminada hasta recibir `PASS` del revisor
independiente.

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

Estado: **0/7 aceptadas**.

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
- [x] Focused de la tarea: `117 passed`.
- [x] Focused del brief: `30 passed`.
- [x] Ruff: limpio.
- [x] Mypy: limpio.
- [x] Non-slow: `5926 passed, 18 skipped, 7 deselected`.
- [x] Compatibilidad v1 y archivos congelados: intactos.
- [ ] Generar paquete de revisión final para `d9ed37a6..54760a8a`.
- [ ] Revisión independiente final.
- [ ] Corregir hallazgos residuales, si existen.
- [ ] Obtener `PASS`.
- [ ] Marcar la tarea completa en el ledger.

### Tarea 2 — Anchors, append, verificación y bundles locales v2

- [ ] Generar brief y registrar BASE.
- [ ] RED de anchors, gaps, forks, tampering, repair e import idempotente.
- [ ] Implementar `OperationLedgerV2`.
- [ ] Focused pytest, ruff y mypy.
- [ ] Commit explícito.
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
