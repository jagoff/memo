#!/bin/sh
# cg_impact_gate.sh — pre-refactor gate over the codegraph index.
#
# Usage: scripts/cg_impact_gate.sh <symbol> [depth]
#
# Runs query -> callers -> impact for <symbol> and ABORTS if the symbol does
# not exist in the graph: `codegraph callers`/`impact` silently substitute the
# best fuzzy match for unknown names (upstream #1473), so an unchecked typo
# reports the blast radius of a DIFFERENT symbol. Exit 0 = report printed;
# exit 1 = symbol not found (do not trust any callers/impact output for it);
# exit 2 = usage / missing tooling.
set -u

SYMBOL="${1:-}"
DEPTH="${2:-2}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Worktrees don't carry .codegraph/ — fall back to the main checkout's index.
if [ ! -d "$REPO_ROOT/.codegraph" ]; then
    MAIN_GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
    [ -n "$MAIN_GIT_DIR" ] && [ -d "$(dirname "$MAIN_GIT_DIR")/.codegraph" ] \
        && REPO_ROOT="$(dirname "$MAIN_GIT_DIR")"
fi

[ -n "$SYMBOL" ] || { echo "usage: $0 <symbol> [depth]" >&2; exit 2; }

CG_BIN="$HOME/.nvm/versions/node/v24.14.0/bin/codegraph"
[ -x "$CG_BIN" ] || CG_BIN="$(command -v codegraph 2>/dev/null || true)"
[ -n "$CG_BIN" ] || { echo "cg_impact_gate: codegraph CLI not found" >&2; exit 2; }

# Existence check: exact-name match in the query JSON. `query` returns fuzzy
# candidates too, so filter for the exact symbol name.
MATCHES="$("$CG_BIN" query "$SYMBOL" -p "$REPO_ROOT" -j 2>/dev/null \
  | python3 -c '
import json, sys
target = sys.argv[1]
try:
    rows = json.load(sys.stdin)
except Exception:
    rows = []
nodes = [r.get("node", r) for r in rows]
exact = [n for n in nodes if n.get("name") == target]
print(len(exact))
for n in exact[:10]:
    kind = n.get("kind", "?")
    path = n.get("filePath", "?")
    line = n.get("startLine", "?")
    print("  %-10s %s:%s" % (kind, path, line))
' "$SYMBOL")"

COUNT="$(printf '%s\n' "$MATCHES" | head -1)"
case "$COUNT" in
    ''|*[!0-9]*)
        echo "ABORT: could not parse codegraph query output for '$SYMBOL' — not proceeding to callers/impact." >&2
        exit 2
        ;;
esac
if [ "$COUNT" = "0" ]; then
    echo "ABORT: symbol '$SYMBOL' not found in the codegraph index." >&2
    echo "callers/impact would silently report a fuzzy-matched OTHER symbol (#1473)." >&2
    echo "Check the spelling with: codegraph query '$SYMBOL' -p '$REPO_ROOT'" >&2
    exit 1
fi

echo "== symbol: $SYMBOL ($COUNT exact match(es)) =="
printf '%s\n' "$MATCHES" | tail -n +2
if [ "$COUNT" -gt 1 ]; then
    echo "NOTE: overloaded name — pin the right one with 'codegraph node' (file/line) before trusting the report."
fi

echo
echo "== callers (each one must appear in your diff or be justified) =="
"$CG_BIN" callers "$SYMBOL" -p "$REPO_ROOT" -l 50

echo
echo "== impact (depth $DEPTH) =="
"$CG_BIN" impact "$SYMBOL" -p "$REPO_ROOT" -d "$DEPTH"

echo
echo "Rename success criterion: after the change, 'codegraph callers $SYMBOL' returns 0 for the OLD name."
