#!/bin/bash
# memo statusline for Claude Code.
# Reads the Claude Code statusline JSON on stdin and prints a compact one-liner:
#   <dir basename> · <git branch> · <model> · [MEMO <version>]
#
# Dependency-free beyond bash + standard tools (jq is used if present, with a
# pure-shell fallback). The memo version comes from a filesystem glob over the
# installed dist-info dir — no python launch — so it stays fast and a reinstall
# / upgrade updates the badge by itself.

INPUT=$(cat)

# ── tiny JSON field reader (jq when available, else a crude grep fallback) ─────
_json() {
  # $1 = jq filter, $2 = grep-fallback key (last "key":"value" string match)
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$INPUT" | jq -r "$1 // empty" 2>/dev/null
  else
    printf '%s' "$INPUT" \
      | grep -o "\"$2\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" \
      | tail -1 | sed 's/.*:[[:space:]]*"//; s/"$//'
  fi
}

# ── Directory (basename of the workspace dir) ─────────────────────────────────
CWD=$(_json '.workspace.current_dir // .cwd' 'current_dir')
[ -z "$CWD" ] && CWD=$(_json '.cwd' 'cwd')
DIR_OUT=""
[ -n "$CWD" ] && DIR_OUT="${CWD##*/}"

# ── Git branch (only when inside a repo) ──────────────────────────────────────
BRANCH=""
if [ -n "$CWD" ] && [ -d "$CWD" ]; then
  BRANCH=$(git -C "$CWD" --no-optional-locks symbolic-ref --short HEAD 2>/dev/null \
           || git -C "$CWD" --no-optional-locks rev-parse --short HEAD 2>/dev/null)
fi

# ── Model display name ────────────────────────────────────────────────────────
MODEL=$(_json '.model.display_name' 'display_name')

# ── Memo version badge (filesystem glob over the installed dist-info dir) ─────
# Sanitized to [0-9A-Za-z.+-] so nothing can inject ANSI/OSC escapes.
MEMO_VER=""
for _b in "$HOME/.local/pipx/venvs/mlx-memo" "$HOME/.local/share/uv/tools/mlx-memo"; do
  for _d in "$_b"/lib/python*/site-packages/mlx_memo-*.dist-info; do
    [ -d "$_d" ] || continue
    MEMO_VER="${_d##*/mlx_memo-}"; MEMO_VER="${MEMO_VER%.dist-info}"
    break 2
  done
done
if [ -z "$MEMO_VER" ]; then
  MEMO_VER_FILE="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/.memo-version"
  [ -f "$MEMO_VER_FILE" ] && [ ! -L "$MEMO_VER_FILE" ] && \
    MEMO_VER=$(head -c 16 "$MEMO_VER_FILE" 2>/dev/null)
fi
MEMO_VER=$(printf '%s' "$MEMO_VER" | tr -cd '0-9A-Za-z.+-')
MEMO_BADGE=""
[ -n "$MEMO_VER" ] && MEMO_BADGE="[MEMO $MEMO_VER]"

# ── Compose: join non-empty parts with " · " ──────────────────────────────────
OUT=""
for part in "$DIR_OUT" "$BRANCH" "$MODEL" "$MEMO_BADGE"; do
  [ -n "$part" ] || continue
  if [ -z "$OUT" ]; then OUT="$part"; else OUT="$OUT · $part"; fi
done
printf '%s' "$OUT"
