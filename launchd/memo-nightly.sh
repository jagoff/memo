#!/bin/sh
# memo nightly maintenance — one wake, one MLX load. Sequential:
# codegraph sync → contradict scan → gc dupes → gc orphans → consolidate.
# Template: replace __MEMO_BIN__ (`command -v memo`) and __CODEGRAPH_BIN__
# (`command -v codegraph`; delete that block if you don't use codegraph),
# then install to ~/.local/share/memo/bin/memo-nightly.sh (chmod 755).
# Driven by the com.memo.nightly LaunchAgent (see com.memo.nightly.plist).
# Adopted from the synapse-era memo-nightly.sh on its deprecation (2026-07-30).
set -u

log() { echo "[memo-nightly $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "start codegraph-sync"
for r in memo memflow; do
  "__CODEGRAPH_BIN__" sync "$HOME/repos/$r" --quiet || "__CODEGRAPH_BIN__" unlock "$HOME/repos/$r"
done || log "codegraph-sync FAILED (exit $?)"

log "start contradict-scan"
"__MEMO_BIN__" contradict scan --max-memories 500 --min-days-apart 3 --since "$(date -v-30d +%Y-%m-%d)" || log "contradict-scan FAILED (exit $?)"

log "start gc-memo-duplicates"
"__MEMO_BIN__" ops gc-memo-duplicates --json || log "gc-memo-duplicates FAILED (exit $?)"

log "start gc-vault-orphans"
"__MEMO_BIN__" ops gc-vault-orphans --json || log "gc-vault-orphans FAILED (exit $?)"

log "start memo-consolidate"
"__MEMO_BIN__" consolidate apply --auto-threshold 0.95 --max-clusters 15 || log "memo-consolidate FAILED (exit $?)"

log "done"
