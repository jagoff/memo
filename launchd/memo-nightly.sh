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

# Due-guard. The LaunchAgent asks for 03:00, but macOS does not wake for a
# StartCalendarInterval and does not reliably replay a slot it slept through:
# measured on mac-black 2026-08-13, the machine was in darkwake at 03:02,
# `launchctl print` reported `runs = 0`, and the newest nightly.log entry was
# three days old. The plist therefore also carries an hourly StartInterval, and
# this guard is what keeps that from running the whole pass every hour: the
# pass runs at most once per MEMO_NIGHTLY_MIN_INTERVAL_H (default 20 — under
# 24 so a slightly-late catch-up run never pushes the next night out).
# `--force` (or MEMO_NIGHTLY_FORCE=1) runs it regardless.
_stamp="${HOME}/.local/share/memo/.last_nightly_ts"
_min_h="${MEMO_NIGHTLY_MIN_INTERVAL_H:-20}"
if [ "${1:-}" != "--force" ] && [ "${MEMO_NIGHTLY_FORCE:-0}" != "1" ] && [ -f "$_stamp" ]; then
  _age_s=$(( $(date +%s) - $(cat "$_stamp" 2>/dev/null || echo 0) ))
  if [ "$_age_s" -lt $(( _min_h * 3600 )) ]; then
    exit 0
  fi
fi
mkdir -p "$(dirname "$_stamp")" && date +%s > "$_stamp"

# Only repos that still exist: memflow was archived to ~/repos/_archived on the
# 2026-07-30 trinity deprecation, and syncing it printed
# "✗ CodeGraph not initialized in .../memflow" into nightly.err every night.
log "start codegraph-sync"
for r in memo; do
  [ -d "$HOME/repos/$r" ] || { log "codegraph-sync: skipping missing $HOME/repos/$r"; continue; }
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

# `--yes` is REQUIRED here, not belt-and-braces: `--force` alone still raises
# an interactive `click.confirm(..., abort=True)` (cli_consolidate.py), and a
# LaunchAgent has no stdin — so this line aborted with exit 1 every single
# night and consolidation never ran. Observed on mac-black through
# 2026-08-13: "This will merge memories and archive the originals. Continue?
# [y/N]:" followed by "memo-consolidate FAILED (exit 1)" in nightly.log.
log "start memo-consolidate"
"__MEMO_BIN__" consolidate apply --force --yes --auto-threshold 0.95 --max-clusters 15 || log "memo-consolidate FAILED (exit $?)"

log "done"
