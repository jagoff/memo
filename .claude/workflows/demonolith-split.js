export const meta = {
  name: 'demonolith-split',
  description: 'Map a god-file and propose a clean in-repo package split (re-export shim preserves imports). Plan only — does not edit.',
  whenToUse: 'When a single file is too large and you want a concrete, low-churn split into a package before refactoring.',
  phases: [
    { title: 'Map', detail: 'inventory symbols + dependencies' },
    { title: 'Cluster', detail: 'multi-angle split proposals + judge' },
    { title: 'Plan', detail: 'synthesize the chosen file-by-file split' },
  ],
}

// Repo-agnostic: runs from whatever repo cwd the workflow is invoked in.
// `args` = target file path (relative to repo root). If empty, the Map agent finds the largest source file.
const target = typeof args === 'string' && args.trim() ? args.trim() : null

const SYMBOL_MAP = {
  type: 'object',
  required: ['file', 'lineCount', 'symbols'],
  properties: {
    file: { type: 'string' },
    lineCount: { type: 'integer' },
    externalImports: { type: 'array', items: { type: 'string' } },
    publicApi: {
      type: 'array',
      description: 'symbols imported by OTHER modules (must survive the split)',
      items: { type: 'string' },
    },
    symbols: {
      type: 'array',
      items: {
        type: 'object',
        required: ['name', 'kind', 'lines'],
        properties: {
          name: { type: 'string' },
          kind: { type: 'string', enum: ['class', 'function', 'constant', 'other'] },
          lines: { type: 'string', description: 'start-end' },
          dependsOn: { type: 'array', items: { type: 'string' }, description: 'other in-file symbols it uses' },
          responsibility: { type: 'string' },
        },
      },
    },
  },
}

const SPLIT_PROPOSAL = {
  type: 'object',
  required: ['lens', 'modules'],
  properties: {
    lens: { type: 'string' },
    modules: {
      type: 'array',
      items: {
        type: 'object',
        required: ['module', 'symbols'],
        properties: {
          module: { type: 'string', description: 'new module path within the package' },
          symbols: { type: 'array', items: { type: 'string' } },
          rationale: { type: 'string' },
        },
      },
    },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const SPLIT_PLAN = {
  type: 'object',
  required: ['package', 'modules', 'shim', 'testMoves'],
  properties: {
    package: { type: 'string', description: 'new package dir, e.g. src/memo/capture/' },
    modules: {
      type: 'array',
      items: {
        type: 'object',
        required: ['module', 'symbols'],
        properties: {
          module: { type: 'string' },
          symbols: { type: 'array', items: { type: 'string' } },
          imports: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    shim: { type: 'string', description: 'how the original path re-exports to keep existing imports working' },
    testMoves: { type: 'array', items: { type: 'string' } },
    order: { type: 'array', items: { type: 'string' }, description: 'safe step-by-step execution order' },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const findCmd =
  `largest source file: run \`git ls-files '*.py' | xargs wc -l | sort -rn | sed -n '2p'\` ` +
  `(skip the TOTAL line) and use that path`

const map = await agent(
  `Map a god-file in the CURRENT repo (cwd is the repo root).\n` +
    (target
      ? `Target file: ${target}\n`
      : `No target given — pick the ${findCmd}.\n`) +
    `Read the file. Inventory every top-level symbol with its line range, kind, what other ` +
    `in-file symbols it depends on, and a one-line responsibility. List external imports and, ` +
    `by grepping the repo, the public API surface (symbols other modules import from this file). ` +
    `Do not edit anything.`,
  { phase: 'Map', schema: SYMBOL_MAP }
)

const angles = ['by-responsibility', 'by-dependency-cohesion', 'by-public-API-stability']
const proposals = await parallel(
  angles.map((lens) => () =>
    agent(
      `Propose a package split for ${map.file} using a ${lens} lens. ` +
        `Group the symbols into cohesive modules. Keep a re-export shim so existing imports survive. ` +
        `Avoid circular imports.\nSYMBOL MAP:\n${JSON.stringify(map)}`,
      { label: `cluster:${lens}`, phase: 'Cluster', schema: SPLIT_PROPOSAL }
    )
  )
)

const plan = await agent(
  `Score these ${angles.length} split proposals for cohesion, churn (lines moved), and ` +
    `import-stability, then pick the best and graft good ideas from the others into a single ` +
    `concrete file-by-file plan for ${map.file}. Include the re-export shim, which tests move where, ` +
    `and a safe execution order. This is a PLAN — do not edit files.\n` +
    `PROPOSALS:\n${JSON.stringify(proposals.filter(Boolean))}`,
  { phase: 'Plan', schema: SPLIT_PLAN }
)

return { target: map.file, lineCount: map.lineCount, plan }
