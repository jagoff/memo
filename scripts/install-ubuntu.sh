#!/usr/bin/env bash
# memo — Ubuntu/Linux installer (standalone CPU backend, no MLX).
#
# Installs memo as an isolated tool via uv (preferred) or pipx. The Linux
# platform dependency pulls sentence-transformers. Semantic search/recall/save work;
# the MLX-only reranker + LLM verbs (ask/synthesize/dream) are unavailable.
# See docs/ubuntu.md.
#
# Usage:
#   scripts/install-ubuntu.sh              # install mlx-memo from PyPI
#   scripts/install-ubuntu.sh --from-source  # install this checkout (editable-free)
set -euo pipefail

SPEC="mlx-memo"
FROM_SOURCE=0
for arg in "$@"; do
  case "$arg" in
    --from-source) FROM_SOURCE=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$FROM_SOURCE" == "1" ]]; then
  repo_root="$(cd "$(dirname "$0")/.." && pwd)"
  SPEC="${repo_root}"
fi

# pipx needs a compatible interpreter; uv gets the requested managed 3.13.
py_ok() { command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,13) else 1)'; }

install_with_uv() {
  echo "→ uv tool install --python 3.13 '$SPEC' (official PyTorch CPU index)"
  uv tool install --python 3.13 \
    --find-links https://download.pytorch.org/whl/cpu/torch/ "$SPEC"
}
install_with_pipx() {
  echo "→ pipx install '$SPEC' (official PyTorch CPU wheels)"
  pipx install "$SPEC" \
    --pip-args="--find-links https://download.pytorch.org/whl/cpu/torch/"
}

if command -v uv >/dev/null 2>&1; then
  install_with_uv
elif command -v pipx >/dev/null 2>&1; then
  if ! py_ok; then
    echo "error: pipx needs an available Python >= 3.13 for memo." >&2
    echo "Install uv for a managed interpreter, then rerun:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
  install_with_pipx
else
  cat >&2 <<'EOF'
error: neither `uv` nor `pipx` found.

Install one, then re-run:
  # uv (recommended — manages Python too):
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # or pipx (needs system Python >= 3.13):
  sudo apt-get install -y pipx && pipx ensurepath
EOF
  exit 1
fi

echo
echo "✓ memo installed (CPU backend). Next:"
echo "    memo doctor"
echo "    memo config validate"
echo "    memo save 'hello from ubuntu' --type note && memo search 'ubuntu'"
echo
echo "First search downloads the embedding model (~1.2 GB). See docs/ubuntu.md."
