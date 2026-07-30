# Checklist vivo — absorción completa de Memflow en Memo

Última actualización: 2026-07-30
Branch: `feat/memflow-absorption`
Worktree: `/Users/fer/repos/memo/.worktrees/memflow-absorption`

## Ejecución en tiempo real

Último checkpoint: 2026-07-30 14:54 America/Argentina/Cordoba

### En curso

- [x] Revisión independiente de especificación, APIs y tests de Plan 01
  Tareas 2–6 — `T2 PASS`, `T3 PASS`, `T4 PASS`, `T5 PASS`, `T6 PASS`.
- [ ] Revisor continuo read-only de corrección y durabilidad para cada lote
  nuevo — agente `constant_code_reviewer`; `P01-T04 FAIL HIGH` cerrado,
  `P03-T04` aceptado y `P03-T03 FAIL BLOCKER` confirmado; Secure Enclave
  `23a64ccb` continúa `FAIL HIGH`.
- [x] Auditoría read-only cross-repo de Plan 03 T3/T4: ambos lotes corregidos
  siguen presentes sin commit y no se solapan con Secure Enclave.
- [ ] Corregir el prerrequisito productivo de Plan 01 Tarea 7: helper
  precompilado/estable sin `swiftc` en runtime y ACL/upgrade de Keychain.
  `23a64ccb` tuvo `236 passed / 1 skipped`, pero revisión independiente
  encontró 2 HIGH. Fix de helper precompilado pasó disponibilidad, pero
  upgrade Keychain entre helpers ad-hoc falló de forma reproducible;
  `DONE_WITH_CONCERNS`, sin commit.
- [x] Corregir `P01-T03 HIGH`: una aplicación incremental con backfill
  multi-origen puede divergir del rebuild por orden global. Agregar regresión
  `newest(a) -> older(b)` y re-reducción transaccional determinista. Revisión
  continua final: `PASS`.
- [x] Corrección técnica `P01-T02 HIGH`: TOCTOU entre verificación del ledger
  v1 y la relectura usada por manifest/anchor. Snapshot descriptor-relative
  único, locks estables antes/durante creación y devices vacíos ligados.
  Revisión final: `PASS`, `80 passed`.
- [x] Resolver el blocker del helper de `P01-T07`: `54acf447` fija namespace
  productivo v2, helper precompilado/cache `helpers-v1/<sha256>`, bindings
  canónicos y validación fail-closed sin Swift runtime/fallback. Revisión
  independiente PASS; aún no se habilita ninguna activación productiva.
- [x] Re-review `P01-T04` commit `6e54600f`: PASS/ADDRESSED desde archive
  exacto; cadena root→boundary→parent→target con FD/no-follow/dev-ino,
  re-resolve antes de éxito, `21 passed`, sin nuevos BLOCKER/HIGH/MEDIUM.
- [x] Corrección técnica `P01-T06 HIGH/MEDIUM`: replay del evento/resultado
  canónico por idempotency key, validación de identidad explícita y derivación
  de key default bajo el lock de sesión. Revisión final: `PASS`, `141 passed`.
- [ ] Completar el gate diferido `P01-T03 MEDIUM`: stale/missing epoch por
  MCP/CLI/daemon/Memory/direct store queda bloqueante de T7.
- [x] Corregir `P03-T01`: manifest fail-closed ante symlinks y receipts stale;
  snapshot post-publication reverified, receipts v2 firmados con buckets
  horarios y agregados por evento, y transforms/fixtures canónicos con
  registry obligatorio y digest en operation-map. Commits `d14755bc`..`86d87e91`,
  `42` pruebas focalizadas, Ruff/mypy limpios y revisiones independientes PASS.
- [x] Corregir `P03-T02/T03`: cubrir todos los writers CLI/auxiliares y
  autoridad descriptor-relative. Auditoría confirmó writers sin fence en
  chat delete, dream, homeostasis, autopilot, kernel, user_signal, lookup,
  quality_feedback y cursor de extract. P03-T02 quedó fenced en todos los
  writers auditados; commits `c0ae3a0779`..`3cb254dc3`, `131` pruebas,
  Ruff/mypy/diff-check limpios y revisión independiente PASS.
- [x] Cerrar `P03-T04`: `f32e789` construye desde objetos Git exactos,
  valida submódulos recursivos, digest completo, mount kernel-read-only,
  attestation externa y cleanup de mounts; `8600800` exige provenance
  privada para cada attestation. Re-review final `ADDRESSED/PASS`, focused
  `45 passed`, Ruff/mypy/diff-check limpios, sin activar servicios.
- [x] Corregir `P03-T03 BLOCKER`: aunque el control ya proviene de un commit
  Git exacto, `GIT_DIR` heredado puede desviar el gate fresh a otro state root,
  ocultar el ledger real (`inflight=1`) y devolver falsamente `clean=True`.
  `45b22d6c64` fijó entorno/ejecutable/root; follow-up `8ebc663091` liga
  `.git`/git-dir. `7193be2a0` retiene descriptores, cierra split-brain de lock,
  ABA de Git y cleanup/cache. `79 focused` + `312 integrated` verdes; revisión
  independiente PASS.
- [ ] Reemplazar expiries fijas de tests P03 que ya vencieron; matriz actual
  `221 passed, 3 failed` por tiempo de fixture.

### Completado en esta ejecución

- [x] Alcance fijado: absorber uso vivo de 90 días y dependencias
  indispensables; eliminar capacidades sin uso junto con Memflow.
- [x] Estrategia fijada: revisores y subagentes paralelos con ownership no
  superpuesto.
- [x] Auditoría read-only P03-T01/T02 completada: `T01` tiene 1 `BLOCKER` +
  3 `HIGH`; `T02` tiene 1 `BLOCKER` + 1 `HIGH`, con paths y briefs mínimos
  congelados para el próximo fix-loop.
- [x] P03-T03 drenaje observable inicial entregado en Memflow `a3a6070912`;
  el control Git-autoritativo exacto permanece abierto.
- [x] Evidencia P03-T03 registrada en Memo `e4de56db`.
- [x] Memflow productivo verificado sin mutaciones: PID `1961`, binario y
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
- [x] `P01-T02` follow-up: lock estable por path absoluto antes del lock por
  identidad para write y admission, incluso si se crean ancestros; device
  vacío ligado al manifest. Matriz `79 passed` + atomic focused `7 passed`,
  Ruff y mypy limpios.
- [x] `P01-T02` re-revisión final de `0b2e3c9d` + `e032a775` + `396d13a8`:
  `FINAL PASS`, matriz final `80 passed`, sin hallazgos materiales.
- [x] `P01-T06` GREEN técnico: timestamps canónicos estables en retry y
  lecturas con artifacts locales; `118 passed`, Ruff y mypy limpios.
- [x] `P01-T06` follow-up: recupera evento/result canónicos por idempotency key
  antes de reconstruir derivados locales; explicit-input drift sigue en
  conflicto. Segundo re-review detectó la key default aún no recuperada;
  follow-up final deriva la key bajo lock. Matriz `134 passed`, Ruff/mypy.
- [x] `P01-T06` re-revisión final de `3fc4bf88` + `f79427bf` + `e84da0b3`:
  `PASS`; gate ampliado `141 passed`, sin hallazgos materiales.
- [x] `P01-T03` re-revisión final: `PASS`; backfill global y upgrade v1
  fail-closed cerrados, `20 + 55 passed`, Ruff y mypy limpios.
- [x] `P01-T04` segunda corrección técnica: rename exclusivo sin clobber bajo
  parent FD retenido, identidad pre/post publish y retry tras crash; migración
  + definitive `17 passed`, Ruff y mypy limpios.

### Próximos gates

- [ ] Consolidar hallazgos de los tres revisores sin duplicados.
- [x] Cerrar `P01-T03 HIGH` y obtener re-revisión PASS del revisor continuo.
- [x] Cerrar `P01-T02 HIGH` y obtener re-revisión PASS del revisor continuo.
- [ ] Cerrar el provider productivo de claves y obtener revisión de seguridad
  antes de escribir cualquier activation stamp.
- [x] Corregir el nuevo `P01-T04 HIGH` de namespace desplazado y obtener
  re-revisión `PASS`.
- [x] Cerrar ambos hallazgos `P01-T06` y obtener re-revisión PASS.
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
- [ ] Plan 01 — Operational Ledger v2: **Tareas 1–6 aceptadas; fix de helper
  T7 aceptado, activación productiva aún pendiente de los gates de corte**.
- [ ] Plan 02 — Runtime nativo de coordinación viva.
- [ ] Plan 03 — Readiness para corte.
- [ ] Plan 04 — Migración del estado activo.
- [ ] Plan 05 — Corte atómico y retiro de Memflow.
- [ ] Despliegue productivo.
- [ ] Verificación de uso vivo exclusivamente vía Memo.
- [ ] Baja definitiva de Memflow.

Progreso aceptado: **7/35 tareas**.

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

Estado: **6/7 aceptadas**.

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
- [x] Re-revisión de `0b2e3c9d`: `FAIL MEDIUM`; el primer append podía crear
  ancestros y adquirir una identidad de lock distinta.
- [x] Follow-up: doble exclusión estable-path + descriptor-identity para
  write/admission y binding de devices persistentes sin segmentos/head.
- [x] Atomic I/O + ledger + migration: `79 passed`; atomic final `7 passed`;
  Ruff y mypy limpios.
- [x] Revisión independiente final y `PASS`.

### Tarea 3 — Vistas SQLite transaccionales e idempotencia

- [x] Generar brief y registrar BASE `c9127f39`.
- [x] RED de transacción, replay, crash window e idempotency conflict.
- [x] Implementar schema, reducers, catch-up, rebuild y commit de nueve pasos.
- [x] Probar bypass directo, integración ledger/fence real, epoch
  stale/future, control OID, contexto ausente y actor incorrecto.
- [x] Focused `22 passed`, acumulado `256 passed`, ruff y mypy limpios.
- [x] Non-slow: `6032 passed, 18 skipped`.
- [x] Commit técnico: `daf3bf36`.
- [x] Corregir backfill multi-origen mediante re-reducción transaccional
  determinista desde eventos canónicos persistidos.
- [x] Upgrade v1 marca `rebuild_required` y bloquea reads/apply/catch-up hasta
  rebuild explícito exitoso, preservando artifacts locales.
- [x] Re-revisión continua final: `PASS`; `20 + 55 passed`, Ruff/mypy limpios.
- [ ] Activar propagación autenticada y probar bypass en CLI/MCP/daemon al
  seleccionar v2 en Tarea 7.
- [x] Revisión independiente y `PASS`.

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
- [x] Primera corrección de fsync nominal: `217adc28`; re-revisión `FAIL`
  `MEDIUM` por reemplazo de staging/parent entre fsync y rename.
- [x] Segunda corrección: parent FD retenido, renameat exclusivo, identidad
  `(dev, ino)` verificada y fsync del mismo parent descriptor.
- [x] Regresiones de reemplazo de staging y crash post-rename/pre-parent-fsync.
- [x] Migración + definitive: `17 passed`; Ruff y mypy limpios.
- [x] Revisión independiente final: `PASS`; publish root-anchored
  `6e54600f`, `21 passed`, sin `BLOCKER/HIGH/MEDIUM`.

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
- [x] Revisión independiente: `PASS`, sin defecto material; Memo
  `99210fd486024444b3715ef395a24ff0`. Verificación actual: `40 passed`.

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
- [x] Corrección técnica de retry: adapters envían timestamps `None` y el
  cache derivado reutiliza el timestamp retornado por el servicio canónico.
- [x] Lecturas públicas canónicas preservan prefijos y artifacts locales sin
  incorporarlos al evento portable.
- [x] Focused adapters/sesiones: `118 passed`; Ruff y mypy limpios.
- [x] Re-review de `3fc4bf88`: `FAIL MEDIUM`; un HEAD/transcript nuevo tras
  crash cambiaba el request hash con la misma idempotency key.
- [x] Follow-up: replay del checkpoint original desde ledger+idempotency,
  validando identidad/session/project/workspace/source/timestamp explícitos.
- [x] Segundo re-review: `FAIL MEDIUM`; faltaba replay cuando la idempotency key
  era derivada por Memo.
- [x] Follow-up final: `turn_count` y key explícita/default se fijan bajo el
  session lock antes del replay/commit.
- [x] Matriz sesiones/adapters: `134 passed`; Ruff y mypy limpios.
- [x] Revisión independiente final: `PASS`; gate ampliado `141 passed`.

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

Estado: **4/6 aceptadas; Tareas 1–4 aceptadas; Tareas 5–6 siguen abiertas**.

- [x] Tarea 1 — Snapshot seguro, capability manifest e inventario.
  Receipts v2, cobertura horaria firmada, registry/fixtures ejecutables y
  operation-map enlazado; `42 focused`, Ruff/mypy limpios, revisión PASS.
- [x] Tarea 2 — Fencing de requests Memflow en todo boundary de mutación.
  Writers auditados fenced; `131 focused`, Ruff/mypy/diff-check limpios,
  revisión PASS.
- [x] Tarea 3 — Drain observable y startup refusal. Descriptor authority,
  lock/ABA y Git trampoline endurecidos; `79 focused`, `312 integrated`,
  Ruff/mypy limpios, revisión PASS.
- [x] Tarea 4 — Aislar Synapse del runtime Memflow. Implementación técnica:
  `933445fd` + hardening `f32e789` + provenance `8600800`; focused final
  `45 passed`; Ruff, mypy y diff-check limpios. Re-review independiente:
  `ADDRESSED/PASS`. Sin activación de servicio.
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
| P01-T02 | Anchors, append y bundles | implementadores + revisores especializados | `8e85662b`; hardening `0b1c859d`; ronda 2 `123cd8f6` | inicial + hardening `36 failed, 76 passed` | final `80` | `5a80c74d`…`396d13a8` | **PASS final** | — |
| P01-T03 | Vistas SQLite e idempotencia | Codex | `c9127f39` | import RED + backfill multi-origen | final `20 + 55` | `daf3bf36` + `62ea3066` | **PASS final** | — |
| P01-T04 | Migración genesis y paridad | Codex | `42f458c0` | import RED + swaps de namespace | final `21` | `78764d74`…`6e54600f` | **PASS final** | — |
| P01-T05 | Outbox durable exactly-once | Codex | `20e16ba4` | import RED | matriz `500`; actual `40` | `24f7a406` | **PASS final** | — |
| P01-T06 | Sesiones canónicas | Codex | `6b68a260` | retry/crash replay | final `141` | `ecf2b951`…`e84da0b3` | **PASS final** | — |
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
| P03-T01 | Snapshot, manifest e inventario | Codex + agentes especializados | `90a7144e` | receipts stale, cobertura declarativa, transforms arbitrarios | `42` focused; Ruff/mypy/diff-check | `d14755bc` + `e5320f96` + `14cf56f4` + `a4dc2ae8` + `ee6a0c7f` + `6b7b83a9` + `f331798f` + `1c0e8af5` + `4187276a` + `785155cc` + `f968c923` + `647fc1d1` + `e3152027` + `eac18e20` + `c7c39f05` + `1b2ef7c2` + `ded0ca3c` + `86d87e91` | **PASS final** | revisiones independientes PASS; Synapse registry threaded |
| P03-T02 | Fencing de requests Memflow | Codex + agente especializado | Memflow `5426e8e5` | writers directos y dominios/compatibilidad | `131` focused; Ruff/mypy/diff-check | `c0ae3a0779` + `4b3f99348` + `ab252b7bc` + `303c6a4ad` + `b15cf9570` + `3cb254dc3` | **PASS final** | revisión independiente PASS |
| P03-T03 | Drain y startup refusal | Codex + agentes especializados | Memflow `2c643863` | Git env/root, `.git` symlink/swap, descriptor ABA | focused `79`; integrated `312` | `a3a6070912` + `45b22d6c64` + `8ebc663091` + `7193be2a0` | **PASS final** | revisión independiente PASS |
| P03-T04 | Aislamiento de Synapse | Codex + agentes especializados | Synapse `45c146d5` | regresiones Git-object, attestation, submódulos y mounts | focused `45`; re-review subset `14` | `933445fd` + `f32e789` + `8600800` | **PASS final** | — |
| P03-T05 | Registry backend de Memo | Synapse | `f45ca47`, `2f78569`, `e4162e3`, `7461ec5`, `7592813`, `6228d0b`, `f1e16ea`, `a7569e8`, `eeb2c88`, `e0cc14b`, `8b2e05a` | memo-only registry, routing, CLI/MCP guards, dashboard legacy routes 410 | 121 focused + runtime review | final independent review | **PASS** | historical parity oracle tests remain separate |
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
