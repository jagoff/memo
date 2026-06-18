#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pluginRoot = "/Users/fer/.understand-anything/repo/understand-anything-plugin";
const skillDir = join(pluginRoot, "skills/understand");
const core = await import(pathToFileURL(join(pluginRoot, "packages/core/dist/index.js")).href);
const { sanitizeGraph, autoFixGraph, validateGraph } = core;

const repos = process.argv.slice(2);
if (repos.length === 0) {
  console.error("Usage: node generate-local-graphs.mjs <repo> [repo...]");
  process.exit(1);
}

function run(cmd, args, cwd, opts = {}) {
  const result = spawnSync(cmd, args, {
    cwd,
    encoding: "utf8",
    maxBuffer: 1024 * 1024 * 128,
    ...opts,
  });
  if (result.status !== 0) {
    throw new Error(
      `${cmd} ${args.join(" ")} failed in ${cwd}\nSTDOUT:\n${result.stdout}\nSTDERR:\n${result.stderr}`,
    );
  }
  return result;
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function writeJson(path, data) {
  writeFileSync(path, JSON.stringify(data, null, 2) + "\n", "utf8");
}

function gitHash(repo) {
  const res = spawnSync("git", ["rev-parse", "HEAD"], { cwd: repo, encoding: "utf8" });
  return res.status === 0 ? res.stdout.trim() : "";
}

function projectDescription(repo) {
  for (const name of ["README.md", "README.rst", "readme.md"]) {
    const path = join(repo, name);
    if (!existsSync(path)) continue;
    const text = readFileSync(path, "utf8")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"))
      .slice(0, 2)
      .join(" ");
    if (text) return text.slice(0, 280);
  }
  return `Knowledge graph for ${basename(repo)}.`;
}

function projectName(repo) {
  const pyproject = join(repo, "pyproject.toml");
  if (existsSync(pyproject)) {
    const match = readFileSync(pyproject, "utf8").match(/^\s*name\s*=\s*["']([^"']+)["']/m);
    if (match) return match[1];
  }
  const pkg = join(repo, "package.json");
  if (existsSync(pkg)) {
    try {
      return JSON.parse(readFileSync(pkg, "utf8")).name ?? basename(repo);
    } catch {}
  }
  return basename(repo);
}

function complexity(lines) {
  if (lines > 500) return "complex";
  if (lines > 150) return "moderate";
  return "simple";
}

function fileNodeType(file) {
  const p = file.path.toLowerCase();
  if (file.fileCategory === "docs" || file.language === "markdown") return "document";
  if (file.fileCategory === "config") return "config";
  if (p.includes(".github/workflows/") || p.endsWith("makefile")) return "pipeline";
  if (["dockerfile", "docker-compose", "kubernetes"].includes(file.language)) return "service";
  if (file.language === "terraform") return "resource";
  if (["graphql", "protobuf", "prisma"].includes(file.language)) return "schema";
  if (file.language === "sql") return "table";
  return "file";
}

function categoryTags(file, result) {
  const tags = [file.language, file.fileCategory].filter(Boolean);
  if (result?.functions?.length) tags.push("functions");
  if (result?.classes?.length) tags.push("classes");
  if (result?.endpoints?.length) tags.push("endpoints");
  return [...new Set(tags)];
}

function node(id, type, name, filePath, summary, tags, extra = {}) {
  return {
    id,
    type,
    name,
    filePath,
    summary,
    tags,
    complexity: extra.complexity ?? "simple",
    ...extra,
  };
}

function edge(source, target, type, weight = 0.5) {
  return { source, target, type, direction: "forward", weight };
}

function layerName(file) {
  const p = file.path.toLowerCase();
  if (p.startsWith("tests/") || p.includes("/tests/") || p.includes("test_")) return "Tests";
  if (p.startsWith("docs/") || file.fileCategory === "docs") return "Documentation";
  if (p.startsWith(".github/") || p.includes("docker") || p.includes("install") || file.fileCategory === "infra") return "Operations";
  if (p.includes("/cli") || basename(file.path).startsWith("cli")) return "CLI";
  if (p.includes("/server") || p.includes("/api") || p.includes("mcp")) return "Server";
  if (p.includes("/memory") || p.includes("/store") || p.includes("/retrieval")) return "Core Memory";
  if (p.includes("/hooks") || p.includes("hook")) return "Hooks";
  if (p.includes("/ui") || p.includes("/components") || p.includes("/app")) return "UI";
  if (file.fileCategory === "config") return "Configuration";
  return "Core";
}

function layerDescription(name) {
  return {
    "Core": "Main implementation files and shared application logic.",
    "Core Memory": "Memory, retrieval, storage, and persistence code.",
    "CLI": "Command-line entrypoints and command wiring.",
    "Server": "Server, MCP, API, and protocol-facing code.",
    "Hooks": "Agent hooks, lifecycle integrations, and recall triggers.",
    "Tests": "Automated tests and fixtures.",
    "Documentation": "Project docs, usage notes, and design references.",
    "Operations": "Install, CI, packaging, deployment, and runtime operations.",
    "UI": "Frontend or interface components.",
    "Configuration": "Project configuration and metadata.",
  }[name] ?? `${name} files.`;
}

function makeLayers(files, fileIdByPath) {
  const map = new Map();
  for (const file of files) {
    const id = fileIdByPath.get(file.path);
    if (!id) continue;
    const name = layerName(file);
    if (!map.has(name)) map.set(name, []);
    map.get(name).push(id);
  }
  return [...map.entries()].map(([name, nodeIds]) => ({
    id: `layer:${name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "")}`,
    name,
    description: layerDescription(name),
    nodeIds,
  }));
}

function makeTour(layers) {
  return layers.slice(0, 10).map((layer, index) => ({
    order: index + 1,
    title: layer.name,
    description: layer.description,
    nodeIds: layer.nodeIds.slice(0, 12),
  }));
}

function buildGraph(repo, scan, importMap, structure) {
  const results = new Map(structure.results.map((r) => [r.path, r]));
  const nodes = [];
  const edges = [];
  const nodeIds = new Set();
  const edgeKeys = new Set();
  const fileIdByPath = new Map();

  function addNode(n) {
    if (nodeIds.has(n.id)) return;
    nodeIds.add(n.id);
    nodes.push(n);
  }

  function addEdge(e) {
    if (!nodeIds.has(e.source) || !nodeIds.has(e.target)) return;
    const key = `${e.source}|${e.target}|${e.type}`;
    if (edgeKeys.has(key)) return;
    edgeKeys.add(key);
    edges.push(e);
  }

  for (const file of scan.files) {
    const result = results.get(file.path);
    const type = fileNodeType(file);
    const id = `${type}:${file.path}`;
    fileIdByPath.set(file.path, id);
    addNode(node(
      id,
      type,
      basename(file.path),
      file.path,
      `${file.path} (${file.language}, ${file.sizeLines} lines).`,
      categoryTags(file, result),
      { complexity: complexity(file.sizeLines) },
    ));

    for (const fn of result?.functions ?? []) {
      const fnId = `function:${file.path}:${fn.name}`;
      addNode(node(fnId, "function", fn.name, file.path, `Function ${fn.name} in ${file.path}.`, ["function"], {
        complexity: complexity((fn.endLine ?? fn.startLine ?? 0) - (fn.startLine ?? 0)),
        lineRange: [fn.startLine, fn.endLine],
      }));
      addEdge(edge(id, fnId, "contains", 1));
    }

    for (const cls of result?.classes ?? []) {
      const clsId = `class:${file.path}:${cls.name}`;
      addNode(node(clsId, "class", cls.name, file.path, `Class ${cls.name} in ${file.path}.`, ["class"], {
        complexity: complexity((cls.endLine ?? cls.startLine ?? 0) - (cls.startLine ?? 0)),
        lineRange: [cls.startLine, cls.endLine],
      }));
      addEdge(edge(id, clsId, "contains", 1));
    }

    for (const endpoint of result?.endpoints ?? []) {
      const name = `${endpoint.method ?? ""} ${endpoint.path}`.trim();
      const epId = `endpoint:${file.path}:${name}`;
      addNode(node(epId, "endpoint", name, file.path, `Endpoint ${name}.`, ["endpoint"], {
        lineRange: [endpoint.startLine, endpoint.endLine],
      }));
      addEdge(edge(id, epId, "contains", 1));
    }
  }

  for (const [from, targets] of Object.entries(importMap)) {
    const fromId = fileIdByPath.get(from);
    if (!fromId) continue;
    for (const target of targets) {
      const toId = fileIdByPath.get(target);
      if (toId) addEdge(edge(fromId, toId, "imports", 0.7));
    }
  }

  for (const result of structure.results) {
    for (const call of result.callGraph ?? []) {
      const caller = `function:${result.path}:${call.caller}`;
      const callee = `function:${result.path}:${call.callee}`;
      if (caller !== callee) addEdge(edge(caller, callee, "calls", 0.8));
    }
  }

  const layers = makeLayers(scan.files, fileIdByPath);
  const graph = {
    version: "1.0.0",
    project: {
      name: projectName(repo),
      languages: Object.keys(scan.stats?.byLanguage ?? {}).sort(),
      frameworks: [],
      description: projectDescription(repo),
      analyzedAt: new Date().toISOString(),
      gitCommitHash: gitHash(repo),
    },
    nodes,
    edges,
    layers,
    tour: makeTour(layers),
  };

  const fixed = autoFixGraph(sanitizeGraph(graph)).data;
  const validation = validateGraph(fixed);
  return { graph: fixed, validation };
}

for (const repo of repos) {
  const root = repo.startsWith("/") ? repo : join(process.cwd(), repo);
  const outDir = join(root, ".understand-anything");
  const inter = join(outDir, "intermediate");
  const tmp = join(outDir, "tmp");
  mkdirSync(inter, { recursive: true });
  mkdirSync(tmp, { recursive: true });

  const scanPath = join(inter, "scan-result.json");
  const importInputPath = join(tmp, "import-input.json");
  const importPath = join(inter, "import-map.json");
  const structureInputPath = join(tmp, "structure-input.json");
  const structurePath = join(inter, "structure-result.json");

  console.error(`[Phase 1/7] Scanning ${root}...`);
  run("node", [join(skillDir, "scan-project.mjs"), root, scanPath], root);
  const scan = readJson(scanPath);

  console.error(`[Phase 2/7] Resolving imports for ${root}...`);
  writeJson(importInputPath, { projectRoot: root, files: scan.files });
  run("node", [join(skillDir, "extract-import-map.mjs"), importInputPath, importPath], root);
  const importMap = readJson(importPath).importMap ?? {};

  console.error(`[Phase 3/7] Extracting structure for ${root} (${scan.files.length} files)...`);
  writeJson(structureInputPath, {
    projectRoot: root,
    batchFiles: scan.files,
    batchImportData: importMap,
  });
  run("node", [join(skillDir, "extract-structure.mjs"), structureInputPath, structurePath], root);
  const structure = readJson(structurePath);

  console.error(`[Phase 4/7] Building graph for ${root}...`);
  const { graph, validation } = buildGraph(root, scan, importMap, structure);
  writeJson(join(inter, "assembled-graph.json"), graph);
  writeJson(join(inter, "review.json"), {
    issues: validation.valid ? [] : validation.issues,
    warnings: [],
    stats: {
      totalNodes: graph.nodes.length,
      totalEdges: graph.edges.length,
      totalLayers: graph.layers.length,
      tourSteps: graph.tour.length,
    },
  });
  writeJson(join(outDir, "knowledge-graph.json"), graph);
  writeJson(join(outDir, "meta.json"), {
    lastAnalyzedAt: graph.project.analyzedAt,
    gitCommitHash: graph.project.gitCommitHash,
    version: graph.version,
    analyzedFiles: scan.files.length,
    generatedBy: relative(root, join(here, "generate-local-graphs.mjs")),
    mode: "local-structural",
  });

  console.log(JSON.stringify({
    repo: root,
    graph: join(outDir, "knowledge-graph.json"),
    files: scan.files.length,
    nodes: graph.nodes.length,
    edges: graph.edges.length,
    layers: graph.layers.map((l) => l.name),
    valid: validation.valid,
    issues: validation.valid ? 0 : validation.issues.length,
  }));
}
