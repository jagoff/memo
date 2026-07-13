# MCPB Node bootstrap (Fase B2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Variante Node del `.mcpb` (`memo-node.mcpb`): un `bootstrap.js` de CERO deps npm que (1) verifica/instala `uv`, (2) instala `mlx-memo==<pin del manifest>` vía `uv tool install` si falta, (3) spawnea `memo-mcp` real con stdio heredado. Desbloquea la prioridad Node del Desktop Extensions Directory (submission ya enviada) y elimina el requisito "uv preinstalado" del bundle actual.

**Architecture:** El `.mcpb` actual (`packaging/mcpb/`, `release_mcpb.py:19-39`, ZIP determinístico de 3 miembros) queda INTACTO. Se agrega `packaging/mcpb-node/` (manifest propio con `server.type: "node"` + `bootstrap.js` + icon compartido) y `build_mcpb_node()` en `release_mcpb.py`. **Decisión v1 (YAGNI):** bootstrap NO habla MCP — instala primero (sin responder nada; Claude Desktop espera) y después spawnea el child, que responde `initialize` él mismo. El shim JSON-RPC con replay (la parte fiddly que el spec señaló) queda para v2 SOLO si el smoke manual en Desktop muestra timeout de primer arranque. Mitigación v1: el install es idempotente — si el primer connect muere por timeout, el segundo encuentra todo instalado y arranca al instante; el `long_description` lo documenta. El modelo (~600MB) NO bloquea el arranque (MLX carga lazy en el primer embed — verificado: `memo-mcp` arranca sin modelo).

**Tech Stack:** Node ≥18 (APIs: `child_process.spawnSync/spawn`, `fs`, `path` — nada externo), Python/pytest para los tests de empaquetado y sincronía de versiones. Primer `.js` productivo del repo: el gate de sintaxis es `node --check` desde pytest (skip si no hay node).

**Spec:** `docs/superpowers/specs/2026-07-13-roadmap-gamechangers-design.md` §3 (bootstrap Node como apuesta barata secundaria — sin expectativa de mover la aguja sola). Datos verificados: `release_mcpb.py:9-39` (`MCPB_MEMBERS`, `_ZIP_EPOCH`, `build_mcpb`), `packaging/mcpb/manifest.json` (manifest_version 0.3, `mcp_config: {command: "uvx", args: ["--from", "mlx-memo>=3.4.0", "memo-mcp"]}`, privacy_policies/author/tools_generated obligatorios), stub actual `packaging/mcpb/server/main.py`, entrypoint `memo-mcp = memo.server:main` (`pyproject.toml:122`, stdio default, cero env obligatorias), auto-update default OFF y tag-based (no interfiere con pin), CI `macos-smoke.yml` (brew python@3.13 + uv; NO hay job que buildee mcpb), bump de release actualizó `packaging/mcpb/manifest.json` hoy (3.4.0).

## Global Constraints

- `bootstrap.js`: cero dependencias npm; Node stdlib only; TODO output diagnóstico a **stderr** (stdout es el canal MCP del child — una línea nuestra en stdout corrompe el protocolo).
- Pin de versión: el bootstrap instala `mlx-memo==<version>` leyendo `version` del `manifest.json` adyacente (`path.join(__dirname, "manifest.json")`) — una sola fuente por bundle. La sincronía pyproject ↔ ambos manifests se ENFORCEA por test (Task 4) y por el bump de release (Task 5).
- El `.mcpb` Python existente no se toca; `build_mcpb` queda idéntico.
- Manifest nuevo: conserva TODOS los campos que la validación MCPB exige (privacy_policies, author.email, tools_generated — memoria [3b4165ca]).
- Working tree compartido: `git add` paths explícitos. Tests aislados (tmp dirs; el test de `node --check` hace `pytest.mark.skipif(shutil.which("node") is None, ...)`).
- Sin flags MEMO_* nuevos: esto es empaquetado, no comportamiento del server.

## File Structure

| Archivo | Responsabilidad |
|---|---|
| Create `packaging/mcpb-node/bootstrap.js` | ensure-uv → ensure-mlx-memo(pin) → spawn memo-mcp stdio inherit |
| Create `packaging/mcpb-node/manifest.json` | manifest 0.3 con server.type node + mcp_config node |
| Modify `src/memo/release_mcpb.py` | `MCPB_NODE_MEMBERS` + `build_mcpb_node(repo, output)` (mismo ZIP determinístico) |
| Modify `<módulo del release bump>` (Task 5 lo localiza: el que hoy editó packaging/mcpb/manifest.json) | sumar el manifest node a los archivos bumpeados |
| Test `tests/test_release_mcpb_node.py` | build determinístico, miembros, `node --check`, sincronía de versiones |

---

### Task 1: bootstrap.js

**Files:** Create `packaging/mcpb-node/bootstrap.js` · Test `tests/test_release_mcpb_node.py` (nuevo — arranca con el check de sintaxis)

**Interfaces — Produces (estructura del script):**
```js
#!/usr/bin/env node
"use strict";
// Zero-dep bootstrap: stderr for diagnostics, stdout belongs to the child (MCP).
const { spawnSync, spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");

function log(msg) { process.stderr.write(`[memo-bootstrap] ${msg}\n`); }
function readPin() {
  const m = JSON.parse(fs.readFileSync(path.join(__dirname, "manifest.json"), "utf8"));
  return m.version; // manifest version IS the pin
}
function which(cmd) { /* spawnSync(process.platform === "win32" ? "where" : "which", [cmd]) → path|null */ }
function uvBin() { /* which("uv") || ~/.local/bin/uv si existe (instalación fresca fuera de PATH) */ }
function ensureUv() { /* si falta: spawnSync("sh", ["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"]) con stderr inherit; re-resolver ~/.local/bin/uv; throw con mensaje accionable si sigue faltando */ }
function memoMcpBin() { /* which("memo-mcp") || ~/.local/bin/memo-mcp */ }
function ensureMemo(uv, pin) { /* si memo-mcp falta O `uv tool list` no muestra mlx-memo==pin: spawnSync(uv, ["tool", "install", "--force", `mlx-memo==${pin}`], stderr inherit) */ }
function main() {
  const pin = readPin();
  const uv = uvBin() || ensureUv();
  ensureMemo(uv, pin);
  const bin = memoMcpBin();
  if (!bin) { log("memo-mcp not found after install — see https://github.com/jagoff/memo#readme"); process.exit(1); }
  const child = spawn(bin, [], { stdio: "inherit" });  // stdout/stdin = MCP passthrough
  child.on("exit", (code, sig) => process.exit(code ?? (sig ? 1 : 0)));
  for (const s of ["SIGINT", "SIGTERM"]) process.on(s, () => child.kill(s));
}
main();
```
Los cuerpos `/* ... */` los escribe el implementer con spawnSync + checks de exit code; CERO lógica MCP; fast path (todo ya instalado) debe llegar al `spawn` en <100ms.

- [ ] Test primero (`tests/test_release_mcpb_node.py`): `test_bootstrap_js_syntax` — `subprocess.run(["node", "--check", str(repo/"packaging/mcpb-node/bootstrap.js")])` exit 0, con `skipif` sin node; `test_bootstrap_js_is_zero_dep` — el texto no contiene `require(` de nada fuera de `node:` prefijos; `test_bootstrap_reads_pin_from_manifest` — grep del patrón `manifest.json` + `version` en el fuente (estático; el flujo real se prueba en el smoke manual de Task 6). RED → escribir bootstrap.js → GREEN. Commit `feat(mcpb-node): zero-dep bootstrap.js (ensure uv + pinned mlx-memo + stdio passthrough)` staging solo esos 2 archivos.

### Task 2: manifest node

**Files:** Create `packaging/mcpb-node/manifest.json` · Test (extender)

- [ ] Copiar `packaging/mcpb/manifest.json` cambiando SOLO: `"server": {"type": "node", "entry_point": "bootstrap.js", "mcp_config": {"command": "node", "args": ["${__dirname}/bootstrap.js"]}}` (verificar la sintaxis exacta de la variable de sustitución contra la spec MCPB/DXT — si el bundle Python no la usa, buscar un ejemplo oficial; si la spec no soporta `${__dirname}`, usar el path relativo que la spec prescriba y documentarlo), `long_description` ajustada (primer arranque puede tardar minutos instalando; si el primer connect falla, reintentar — es idempotente; ya NO requiere uv preinstalado), y `"name": "memo-node"`? — NO: mismo `"name": "memo"` y `display_name` con sufijo "(Node)" solo si el Directory exige nombres únicos; decisión del implementer con disclosure leyendo la spec. Conservar author/privacy_policies/tools_generated/compatibility intactos. Test: `test_node_manifest_required_fields` (privacy_policies, author.email, tools_generated presentes; server.type == "node") + `test_manifest_versions_in_sync` — `pyproject.toml [project].version == packaging/mcpb/manifest.json version == packaging/mcpb-node/manifest.json version`. Commit `feat(mcpb-node): node manifest (0.3)`.

### Task 3: build_mcpb_node

**Files:** Modify `src/memo/release_mcpb.py` · Test (extender)

- [ ] `MCPB_NODE_MEMBERS = ("icon.png", "manifest.json", "bootstrap.js")` — icon: NO duplicar el binario; el builder lo lee de `packaging/mcpb/icon.png` cuando no exista en `mcpb-node/` (parámetro de fallback o copia explícita — decisión del implementer, la simple gana). `build_mcpb_node(repo, output=None) -> Path` → `packaging/memo-node.mcpb`, MISMO mecanismo determinístico (`_ZIP_EPOCH`, `ZIP_DEFLATED`, `os.replace`). Refactor DRY permitido: extraer `_build_zip(source_pairs, destination)` compartido si queda más chico — sin cambiar el output byte-a-byte del bundle Python (test lo protege). Tests: build produce ZIP con exactamente los 3 miembros; dos builds seguidos = bytes idénticos (determinismo); `build_mcpb` (Python) sigue produciendo los mismos miembros de antes. Commit `feat(mcpb-node): deterministic node bundle builder`.

### Task 4: gate de sincronía de versiones

**Files:** Test (extender `tests/test_release_mcpb_node.py`)

- [ ] Test único fuerte: `test_pin_chain_in_sync` — pyproject version == ambos manifests == (si el manifest Python pinnea `mlx-memo>=X` en args, X == version). Este test es el que rompe cuando alguien bumpea sin tocar el manifest node → el error dice exactamente qué archivo actualizar. Commit `test(mcpb-node): version pin-chain sync gate`.

### Task 5: integración con el release bump

**Files:** Modify el módulo que implementa `memo release bump` (localizarlo: `grep -rn "packaging/mcpb/manifest.json" src/memo/` — es el que lo editó en el bump a 3.4.0) · Test del mismo módulo si existe

- [ ] Sumar `packaging/mcpb-node/manifest.json` a la lista de archivos bumpeados (mismo campo `version`; si además reescribe el `>=X` del mcp_config Python, replicar la lógica que ya use). Test: el test existente del bump (si hay) extendido, o dry-run `memo release bump --dry-run patch` mostrando el archivo nuevo en la lista. Commit `feat(mcpb-node): release bump updates node manifest`.

### Task 6: gate final

- [ ] Suites: `uv run --no-sync pytest tests/test_release_mcpb_node.py -v` + el test file existente de release/mcpb si hay (grep `release_mcpb` en tests/). Lint propio. Full suite `-q` (pre-existentes conocidos excluidos). Build real: `uv run --no-sync python -c "from pathlib import Path; from memo.release_mcpb import build_mcpb_node; print(build_mcpb_node(Path('.')))"` → verificar ZIP con `unzip -l packaging/memo-node.mcpb`. **Smoke manual documentado (para Fer, fuera del gate automatizado):** instalar `memo-node.mcpb` en Claude Desktop en una cuenta/máquina sin uv → medir si el primer connect sobrevive el install (~1-2 min) o requiere el segundo intento; si requiere shim v2, abrir ticket con lo observado. Commit solo si hubo fixes.

## Medición post-ship (del spec)

La apuesta es barata y secundaria: el éxito NO se mide en adopción propia sino en desbloquear la review del Directory (prioriza Node). Registro: estado de la submission + si el smoke manual v1 sobrevive el primer arranque sin shim. El funnel Day-0 real se mide con `memo onboard` (ya shipped), no con este bundle.
