#!/bin/sh
# cg_affected_tests.sh — select the test files impacted by a diff, via SQL over
# the codegraph index. Fail-SAFE: any anomaly falls back to the FULL suite.
#
# Usage:
#   scripts/cg_affected_tests.sh [base-ref]         # list affected test files
#   scripts/cg_affected_tests.sh [base-ref] --run   # run pytest on them
#
# base-ref defaults to origin/master. The diff considered is base-ref...HEAD
# plus uncommitted and untracked changes.
#
# Why SQL and not `codegraph affected`: `affected` is broken for this repo's
# src-layout (returns 0 tests despite ~2k test->src import edges in the graph).
# Why not parse `codegraph node --symbols-only`: it truncates dependents with
# "+N more", which would silently run a SUBSET while looking complete.
#
# Selection is a reverse-dependency TRANSITIVE CLOSURE over the graph's edges,
# depth-capped at 3 hops (covers test -> facade -> mixin -> store layering;
# over-selecting is safe, under-selecting is not). Cycles terminate via the
# recursive CTE's UNION dedup on (file_path, depth). If the closure selects
# more than 400 test files, that IS the suite -> FULL_SUITE.
#
# Fallback (exit 3 + "FULL_SUITE" on stdout) triggers when:
#   - the codegraph DB is missing or the sqlite query fails
#   - the diff touches conftest.py, pyproject.toml, or any non-Python file
#     under src/ (build/config changes have unmodelled blast radius)
#   - a changed src file is absent from the nodes table (index doesn't know
#     it: new file or stale index)
set -u

BASE_REF="${1:-origin/master}"
RUN=0
[ "${2:-}" = "--run" ] && RUN=1
[ "${1:-}" = "--run" ] && { BASE_REF="origin/master"; RUN=1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$REPO_ROOT/.codegraph/codegraph.db"
if [ ! -f "$DB" ]; then
    MAIN_GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
    [ -n "$MAIN_GIT_DIR" ] && DB="$(dirname "$MAIN_GIT_DIR")/.codegraph/codegraph.db"
fi

full_suite() {
    echo "FULL_SUITE"
    [ "$RUN" = "1" ] && cd "$REPO_ROOT" && exec uv run --no-sync pytest tests/ -q
    exit 3
}

[ -f "$DB" ] || { echo "note: codegraph DB not found" >&2; full_suite; }

# Freshen the index so the graph reflects the working tree (cheap, ~0.3s).
CG_BIN="$HOME/.nvm/versions/node/v24.14.0/bin/codegraph"
[ -x "$CG_BIN" ] || CG_BIN="$(command -v codegraph 2>/dev/null || true)"
[ -n "$CG_BIN" ] && "$CG_BIN" sync -q "$(dirname "$(dirname "$DB")")" 2>/dev/null || true

CHANGED="$( (git -C "$REPO_ROOT" diff --name-only "$BASE_REF"...HEAD 2>/dev/null; \
             git -C "$REPO_ROOT" diff --name-only HEAD 2>/dev/null; \
             git -C "$REPO_ROOT" ls-files --others --exclude-standard 2>/dev/null) | sort -u)"
[ -n "$CHANGED" ] || { echo "no changes vs $BASE_REF" >&2; exit 0; }

AFFECTED="$(CG_CHANGED="$CHANGED" python3 - "$DB" "$REPO_ROOT" <<'PYEOF'
import os, sqlite3, sys

db_path, repo_root = sys.argv[1], sys.argv[2]
changed = [ln.strip() for ln in os.environ.get("CG_CHANGED", "").splitlines() if ln.strip()]

MAX_DEPTH = 3        # test -> facade -> mixin -> store; over-selecting is safe
MAX_SELECTED = 400   # >400 selected test files IS the suite (519 total)

src_files, test_files = [], []
for path in changed:
    if path.endswith("conftest.py") or path == "pyproject.toml":
        print("FULL_SUITE"); sys.exit(0)
    if path.startswith("tests/") and path.endswith(".py"):
        test_files.append(path)
    elif path.startswith("src/") and path.endswith(".py"):
        src_files.append(path)
    elif path.startswith("src/"):
        print("FULL_SUITE"); sys.exit(0)
    # docs/scripts/CI files outside src+tests don't select tests

if not src_files and not test_files:
    sys.exit(0)  # nothing test-relevant changed

affected = set(test_files)
if src_files:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        for src in src_files:
            known = con.execute(
                "SELECT 1 FROM nodes WHERE file_path = ? LIMIT 1", (src,)
            ).fetchone()
            if not known:
                print("FULL_SUITE"); sys.exit(0)  # index doesn't know the file
            # Reverse-dependency transitive closure, depth-capped. UNION (not
            # UNION ALL) dedups (file_path, depth) rows, so cycles terminate.
            rows = con.execute(
                """
                WITH RECURSIVE dep(file_path, depth) AS (
                    SELECT ?, 0
                    UNION
                    SELECT ns.file_path, dep.depth + 1
                    FROM dep
                    JOIN nodes nt ON nt.file_path = dep.file_path
                    JOIN edges e ON e.target = nt.id
                    JOIN nodes ns ON ns.id = e.source
                    WHERE dep.depth < ?
                      AND ns.file_path <> dep.file_path
                )
                SELECT DISTINCT file_path FROM dep
                WHERE file_path LIKE 'tests/%'
                  AND file_path LIKE '%.py'
                """,
                (src, MAX_DEPTH),
            ).fetchall()
            affected.update(r[0] for r in rows)
            if len(affected) > MAX_SELECTED:
                print("FULL_SUITE"); sys.exit(0)  # selection this big IS the suite
    except sqlite3.Error:
        print("FULL_SUITE"); sys.exit(0)

for path in sorted(affected):
    print(path)
PYEOF
)"

case "$AFFECTED" in
    *FULL_SUITE*) full_suite ;;
    '') echo "no test-relevant changes vs $BASE_REF" >&2; exit 0 ;;
esac

printf '%s\n' "$AFFECTED"
if [ "$RUN" = "1" ]; then
    cd "$REPO_ROOT" && exec uv run --no-sync pytest $(printf '%s ' $AFFECTED) -q
fi
