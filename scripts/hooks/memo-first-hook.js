#!/usr/bin/env node
const { execSync } = require('child_process');

const MEMO_ROOT = process.env.MEMO_ROOT || '/Users/fer/repos/memo';
const HOOK_ID = process.env.HOOK_ID || '';

if (HOOK_ID !== 'memo-first') {
  process.exit(0);
}

try {
  const result = execSync(
    `${MEMO_ROOT}/.venv/bin/python -m memo --noninteractive unified-briefing`,
    { cwd: MEMO_ROOT, encoding: 'utf8', timeout: 30000 }
  );
  if (result.trim()) {
    console.log('\n📋 MEMO BRIEFING:\n' + result.trim());
  }
} catch (e) {
  // memo no disponible, silently skip
}