#!/usr/bin/env node
"use strict";
// Zero-dep bootstrap: stderr for diagnostics, stdout belongs to the child (MCP).
const { spawnSync, spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");

function log(msg) { process.stderr.write(`[memo-bootstrap] ${msg}\n`); }

function readPin() {
  const manifestPath = path.join(__dirname, "manifest.json");
  let raw;
  try {
    raw = fs.readFileSync(manifestPath, "utf8");
  } catch (err) {
    throw new Error(`manifest.json not found next to bootstrap.js (${manifestPath}): ${err.message}`);
  }
  let m;
  try {
    m = JSON.parse(raw);
  } catch (err) {
    throw new Error(`manifest.json is not valid JSON (${manifestPath}): ${err.message}`);
  }
  if (!m.version) {
    throw new Error(`manifest.json is missing a "version" field (${manifestPath})`);
  }
  return m.version; // manifest version IS the pin
}

function which(cmd) {
  const finder = process.platform === "win32" ? "where" : "which";
  const result = spawnSync(finder, [cmd], { encoding: "utf8" });
  if (result.status !== 0 || !result.stdout) return null;
  const first = result.stdout.split(/\r?\n/).find((line) => line.trim().length > 0);
  return first ? first.trim() : null;
}

function localBinPath(name) {
  const candidate = path.join(os.homedir(), ".local", "bin", name);
  return fs.existsSync(candidate) ? candidate : null;
}

function uvBin() {
  // fresh installs land in ~/.local/bin, often outside the PATH of GUI apps
  return which("uv") || localBinPath("uv");
}

function ensureUv() {
  const existing = uvBin();
  if (existing) return existing;

  log("uv not found — installing via astral.sh installer");
  const install = spawnSync("sh", ["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], {
    stdio: ["ignore", "ignore", "inherit"],
  });
  if (install.status !== 0) {
    throw new Error(`uv installer failed (exit ${install.status}) — install uv manually: https://docs.astral.sh/uv/`);
  }

  const found = uvBin();
  if (!found) {
    throw new Error("uv installed but still not found on PATH or ~/.local/bin — install uv manually: https://docs.astral.sh/uv/");
  }
  return found;
}

function memoMcpBin() {
  return which("memo-mcp") || localBinPath("memo-mcp");
}

function isPinInstalled(uv, pin) {
  const list = spawnSync(uv, ["tool", "list"], { encoding: "utf8" });
  if (list.status !== 0 || !list.stdout) return false;
  return list.stdout.includes(`mlx-memo==${pin}`);
}

function ensureMemo(uv, pin) {
  if (memoMcpBin() && isPinInstalled(uv, pin)) return; // fast path: nothing to do

  log(`installing mlx-memo==${pin} via uv tool install`);
  const install = spawnSync(uv, ["tool", "install", "--force", `mlx-memo==${pin}`], {
    stdio: ["ignore", "ignore", "inherit"],
  });
  if (install.status !== 0) {
    throw new Error(`uv tool install mlx-memo==${pin} failed (exit ${install.status})`);
  }
}

function main() {
  let pin, uv;
  try {
    pin = readPin();
    uv = uvBin() || ensureUv();
    ensureMemo(uv, pin);
  } catch (err) {
    log(err.message);
    process.exit(1);
  }

  const bin = memoMcpBin();
  if (!bin) {
    log("memo-mcp not found after install — see https://github.com/jagoff/memo#readme");
    process.exit(1);
  }

  const child = spawn(bin, [], { stdio: "inherit" }); // stdout/stdin = MCP passthrough
  child.on("exit", (code, sig) => process.exit(code ?? (sig ? 1 : 0)));
  for (const s of ["SIGINT", "SIGTERM"]) process.on(s, () => child.kill(s));
}

if (require.main === module) main();

module.exports = { readPin, which, uvBin, ensureUv, memoMcpBin, isPinInstalled, ensureMemo, main };
