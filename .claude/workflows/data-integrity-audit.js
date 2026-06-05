export const meta = {
  name: 'data-integrity-audit',
  description: 'Read-only health sweep over memo + memflow data: dup/near-empty chunks, stale entries, sync/heartbeat anomalies. Reports fixes, never mutates.',
  whenToUse: 'Periodically, or when you suspect index drift, duplicate chunks, or sync/heartbeat dups.',
  phases: [
    { title: 'Probe', detail: 'run read-only/dry-run diagnostics' },
    { title: 'Synthesize', detail: 'consolidated report with exact fix commands' },
  ],
}

// Every probe is read-only or --dry-run. NO command here mutates data.
const PROBE_CMDS = [
  { repo: 'memo', label: 'memo:doctor', cmd: 'uv run --no-sync memo doctor --json' },
  { repo: 'memo', label: 'memo:doctor-db', cmd: 'uv run --no-sync memo doctor --db' },
  { repo: 'memo', label: 'memo:lint', cmd: 'uv run --no-sync memo lint --json' },
  { repo: 'memo', label: 'memo:dedupe', cmd: 'uv run --no-sync memo dedupe --dry-run --json' },
  { repo: 'memo', label: 'memo:cross-dedup', cmd: 'uv run --no-sync memo cross-dedup' },
  { repo: 'memo', label: 'memo:audit-script', cmd: 'uv run --no-sync python scripts/audit-data-integrity.py' },
  { repo: 'memflow', label: 'memflow:dream', cmd: 'uv run --no-sync memflow dream status --json' },
  { repo: 'memflow', label: 'memflow:homeostasis', cmd: 'uv run --no-sync memflow homeostasis status --json' },
  { repo: 'memflow', label: 'memflow:sync', cmd: 'uv run --no-sync memflow sync-once --no-op 2>/dev/null || uv run --no-sync memflow status' },
  { repo: 'memflow', label: 'memflow:daemon', cmd: 'uv run --no-sync memflow daemon health' },
]

const PROBE_RESULT = {
  type: 'object',
  required: ['label', 'ran'],
  properties: {
    label: { type: 'string' },
    ran: { type: 'boolean', description: 'false if the command was unavailable/errored' },
    anomalies: {
      type: 'array',
      items: {
        type: 'object',
        required: ['kind', 'count'],
        properties: {
          kind: { type: 'string', description: 'e.g. exact-dup-chunks, near-empty-chunks, stale, heartbeat-dups, orphan-rows, sync-noop' },
          count: { type: 'integer' },
          detail: { type: 'string' },
        },
      },
    },
    raw: { type: 'string', description: 'key lines of output (trimmed)' },
  },
}

const REPORT = {
  type: 'object',
  required: ['issues'],
  properties: {
    summary: { type: 'string' },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        required: ['title', 'severity', 'fixCommand'],
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          count: { type: 'integer' },
          source: { type: 'string', description: 'which probe surfaced it' },
          fixCommand: { type: 'string', description: 'exact command to remediate (dry-run first where destructive)' },
        },
      },
    },
  },
}

const probes = await parallel(
  PROBE_CMDS.map((p) => () =>
    agent(
      `cd /Users/fer/repos/${p.repo}\nRun this READ-ONLY diagnostic and report what it found:\n${p.cmd}\n\n` +
        `If the command is unavailable or errors, set ran=false and put the error in raw. ` +
        `Extract anomaly counts (duplicate chunks, near-empty/noise rows, stale entries, ` +
        `orphan rows, heartbeat dups, sync no-ops). Do NOT run any --apply/--fix/mutating command.`,
      { label: p.label, phase: 'Probe', schema: PROBE_RESULT }
    )
  )
)

const report = await agent(
  `Consolidate these probe results into a prioritized data-integrity report for the memo+memflow data layer. ` +
    `For each real issue give title, severity, count, source probe, and the EXACT remediation command ` +
    `(e.g. \`memo dedupe --apply\`, \`memo doctor --gc --fix\`, \`memflow dream apply <id>\`). ` +
    `Never propose a destructive command without a dry-run first. This report does not mutate anything.\n` +
    `PROBES:\n${JSON.stringify(probes.filter(Boolean))}`,
  { phase: 'Synthesize', schema: REPORT }
)

return report
