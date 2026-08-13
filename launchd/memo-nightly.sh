#!/bin/sh
# memo nightly maintenance — one wake, one MLX load. Sequential:
# codegraph sync → contradict scan → contradict resolve → gc dupes →
# gc orphans → gc emitted ledgers → consolidate.
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

# The scan only DETECTS: it stores pairs and prints "run memo contradict
# triage". Without this step nothing acts on them, so every night adds
# contradictions nobody settles and the operational snapshot grows forever.
# `--max-pairs 0` skips a second scan — this acts on what the pass above
# just found. Archive-only and reversible (`memo maintain undo`); the
# duplicate and staleness passes are left to the steps below.
log "start contradict-resolve"
"__MEMO_BIN__" maintain --max-pairs 0 --skip-consolidate --skip-stale --skip-synthesize \
  || log "contradict-resolve FAILED (exit $?)"

log "start gc-memo-duplicates"
"__MEMO_BIN__" ops gc-memo-duplicates --json || log "gc-memo-duplicates FAILED (exit $?)"

log "start gc-vault-orphans"
"__MEMO_BIN__" ops gc-vault-orphans --json || log "gc-vault-orphans FAILED (exit $?)"

log "start gc-emitted-ledgers"
"__MEMO_BIN__" ops gc-emitted-ledgers --json || log "gc-emitted-ledgers FAILED (exit $?)"

log "start memo-consolidate"
"__MEMO_BIN__" consolidate apply --force --auto-threshold 0.95 --max-clusters 15 || log "memo-consolidate FAILED (exit $?)"

log "done"
