#!/usr/bin/env node
"use strict";
// Zero-dep bootstrap: stderr for diagnostics, stdout belongs to the child (MCP).
const { spawn } = require("node:child_process");
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
  const extensions = process.platform === "win32"
    ? (process.env.PATHEXT || ".EXE;.CMD;.BAT;.COM").split(";")
    : [""];
  for (const directory of (process.env.PATH || "").split(path.delimiter)) {
    if (!directory) continue;
    for (const extension of extensions) {
      const candidate = path.join(directory, cmd + extension);
      try {
        if (fs.statSync(candidate).isFile()) {
          fs.accessSync(candidate, process.platform === "win32" ? fs.constants.F_OK : fs.constants.X_OK);
          return candidate;
        }
      } catch (_) {
        // Missing, non-executable, or inaccessible: continue searching PATH.
      }
    }
  }
  return null;
}

function uvBin() {
  const onPath = which("uv");
  if (onPath) return onPath;
  // GUI apps often omit ~/.local/bin even after the official uv installer.
  const local = path.join(os.homedir(), ".local", "bin", "uv");
  try {
    if (!fs.statSync(local).isFile()) return null;
    fs.accessSync(local, process.platform === "win32" ? fs.constants.F_OK : fs.constants.X_OK);
    return local;
  } catch (_) {
    return null;
  }
}

function ensureUv() {
  const existing = uvBin();
  if (existing) return existing;
  throw new Error(
    "uv not found on PATH or ~/.local/bin. Automatic remote shell execution is disabled; "
    + "install uv from https://docs.astral.sh/uv/getting-started/installation/ and retry."
  );
}

function main() {
  let pin, uv;
  try {
    pin = readPin();
    uv = ensureUv();
  } catch (err) {
    log(err.message);
    process.exit(1);
  }

  // Execute the command from the exact package pin. Never hand control to a
  // stale or malicious `memo-mcp` that happens to appear earlier on PATH.
  const child = spawn(
    uv,
    ["tool", "run", "--from", `mlx-memo==${pin}`, "memo-mcp"],
    { stdio: "inherit" },
  ); // stdout/stdin = MCP passthrough
  child.on("error", (err) => {
    log(`failed to launch mlx-memo==${pin}: ${err.message}`);
    process.exit(1);
  });
  child.on("exit", (code, sig) => process.exit(code ?? (sig ? 1 : 0)));
  for (const s of ["SIGINT", "SIGTERM"]) process.on(s, () => child.kill(s));
}

if (require.main === module) main();

module.exports = { readPin, which, uvBin, ensureUv, main };
