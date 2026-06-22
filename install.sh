#!/usr/bin/env bash
set -euo pipefail

APP_NAME="mlx-memo"
OLD_APP_NAME="memo-mcp"
PYPI_SPEC="mlx-memo"
GIT_SPEC="git+https://github.com/jagoff/memo.git"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=13

say() {
  printf '[memo install] %s\n' "$*"
}

warn() {
  printf '[memo install] warning: %s\n' "$*" >&2
}

die() {
  printf '[memo install] error: %s\n' "$*" >&2
  exit 1
}

is_macos_arm64() {
  [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]
}

python_ok() {
  "$1" -c "import sys; raise SystemExit(sys.version_info < (${MIN_PYTHON_MAJOR}, ${MIN_PYTHON_MINOR}))" >/dev/null 2>&1
}

find_python() {
  local candidate
  for candidate in "${PYTHON:-}" python3.14 python3.13 python3; do
    [[ -n "$candidate" ]] || continue
    if command -v "$candidate" >/dev/null 2>&1; then
      local resolved
      resolved="$(command -v "$candidate")"
      if python_ok "$resolved"; then
        printf '%s\n' "$resolved"
        return 0
      fi
    fi
  done
  return 1
}

run_pipx() {
  if [[ -n "${PIPX_BIN:-}" ]]; then
    "$PIPX_BIN" "$@"
  else
    "$PYTHON_BIN" -m pipx "$@"
  fi
}

ensure_pipx() {
  if command -v pipx >/dev/null 2>&1; then
    PIPX_BIN="$(command -v pipx)"
    return 0
  fi

  if "$PYTHON_BIN" -m pipx --version >/dev/null 2>&1; then
    PIPX_BIN=""
    return 0
  fi

  if [[ "${MEMO_INSTALL_NO_BOOTSTRAP_PIPX:-0}" == "1" ]]; then
    die "pipx is not installed. Install pipx first, or unset MEMO_INSTALL_NO_BOOTSTRAP_PIPX."
  fi

  if command -v brew >/dev/null 2>&1; then
    say "pipx not found; installing pipx with Homebrew"
    brew install pipx
    PIPX_BIN="$(command -v pipx)"
    return 0
  fi

  say "pipx not found; installing pipx with ${PYTHON_BIN} -m pip --user"
  "$PYTHON_BIN" -m pip install --user pipx
  PIPX_BIN=""
}

install_spec() {
  if [[ -n "${MEMO_INSTALL_SPEC:-}" ]]; then
    printf '%s\n' "$MEMO_INSTALL_SPEC"
  elif [[ -n "${MEMO_VERSION:-}" ]]; then
    printf '%s==%s\n' "$PYPI_SPEC" "$MEMO_VERSION"
  elif [[ "${MEMO_INSTALL_FROM_PYPI:-0}" == "1" ]]; then
    printf '%s\n' "$PYPI_SPEC"
  else
    printf '%s\n' "$GIT_SPEC"
  fi
}

clean_old_pipx_package() {
  if run_pipx list --short 2>/dev/null | awk '{print $1}' | grep -qx "$OLD_APP_NAME"; then
    say "removing old pipx package: $OLD_APP_NAME"
    run_pipx uninstall "$OLD_APP_NAME"
  fi
}

ensure_default_dirs() {
  mkdir -p "${MEMO_DATA_DIR:-$HOME/Documents/memo}"
  mkdir -p "${MEMO_STATE_DIR:-$HOME/.local/share/memo}"
}

# consciousness-contracts is a sibling repo, not on PyPI — a `pipx install
# git+...memo.git` cannot pull it. memo runs fine without it (shared embed cache
# + URI parsing fall back), but `install-mcp` and the shared cache need it.
# Inject best-effort from a local checkout; never fail the install if it's absent.
inject_contracts() {
  local cc="${MEMO_CONTRACTS_PATH:-$HOME/repos/consciousness-contracts}"
  if [[ ! -f "$cc/pyproject.toml" ]]; then
    say "consciousness-contracts not found at $cc — skipping (memo runs with fallbacks; set MEMO_CONTRACTS_PATH to enable)"
    return 0
  fi
  say "injecting consciousness-contracts from $cc (enables install-mcp + shared cache)"
  if run_pipx inject "$APP_NAME" -e "$cc" >/dev/null 2>&1; then
    say "consciousness-contracts injected"
  else
    warn "consciousness-contracts inject failed (non-fatal; memo runs with fallbacks)."
    warn "Re-run manually: pipx inject $APP_NAME -e $cc"
  fi
}

resolve_memo_bin() {
  if command -v memo >/dev/null 2>&1; then
    command -v memo
    return 0
  fi
  if [[ -x "$HOME/.local/bin/memo" ]]; then
    printf '%s\n' "$HOME/.local/bin/memo"
    return 0
  fi
  return 1
}

# Ask Y/n on a TTY. Returns 0 for yes, 1 for no.
# If stdin is not a TTY (e.g. `curl | bash`), uses default ($2, "Y" or "n").
ask_yes_no() {
  local prompt="$1" default="${2:-Y}" reply
  if [[ ! -t 0 ]]; then
    [[ "$default" == "Y" || "$default" == "y" ]] && return 0 || return 1
  fi
  read -rp "$prompt " reply || reply=""
  reply="${reply:-$default}"
  [[ "$reply" =~ ^[Yy]$ ]]
}

# Decide whether to download MLX models during install.
# Models are part of memo's structure (embedder + reranker + chat are required
# for retrieval and ambient recall), so the default answer is yes. The user can
# decline interactively or force a value via MEMO_INSTALL_DOWNLOAD_MODELS.
should_download_models() {
  case "${MEMO_INSTALL_DOWNLOAD_MODELS:-auto}" in
    yes|true|1|Y|y) return 0 ;;
    no|false|0|N|n) return 1 ;;
    auto|"")
      # Skip prompt if models are already cached
      if [[ -d "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-Embedding-0.6B-4bit-DWQ" ]]; then
        say "models already cached, skipping download"
        return 1
      fi
      ask_yes_no "[memo install] Download MLX models now (~7 GB, required for retrieval)? [Y/n]" Y
      ;;
    *)
      warn "unknown MEMO_INSTALL_DOWNLOAD_MODELS='${MEMO_INSTALL_DOWNLOAD_MODELS}', defaulting to yes"
      return 0
      ;;
  esac
}

main() {
  if [[ "${MEMO_INSTALL_SKIP_PLATFORM_CHECK:-0}" != "1" ]] && ! is_macos_arm64; then
    die "memo requires macOS on Apple Silicon (Darwin arm64). Set MEMO_INSTALL_SKIP_PLATFORM_CHECK=1 to bypass."
  fi

  PYTHON_BIN="$(find_python)" || die "Python >= 3.13 is required. Install python@3.13 or python@3.14 first."
  export PYTHON_BIN
  say "using Python: $PYTHON_BIN"

  ensure_pipx
  clean_old_pipx_package

  local spec
  spec="$(install_spec)"
  say "installing $APP_NAME with pipx spec: $spec"
  run_pipx install --force "$spec"
  run_pipx ensurepath >/dev/null 2>&1 || true

  inject_contracts
  ensure_default_dirs

  local memo_bin
  memo_bin="$(resolve_memo_bin)" || die "memo was installed but no memo binary was found in PATH or ~/.local/bin"

  say "installed: $("$memo_bin" --version)"

  if should_download_models; then
    say "downloading MLX models (~7 GB, first install may take 5–15 min)…"
    say "(embedder + reranker load now; chat models download in background)"
    if MEMO_NONINTERACTIVE=1 "$memo_bin" prewarm --download-all; then
      say "models ready"
    else
      warn "model download did not complete — models will download on first use."
      warn "Re-run: MEMO_NONINTERACTIVE=1 memo prewarm --download-all"
    fi
  else
    say "skipping MLX model download (models will load lazily on first use)."
    say "Run later: MEMO_NONINTERACTIVE=1 memo prewarm --download-all"
  fi

  say "runtime check:"
  MEMO_NONINTERACTIVE=1 "$memo_bin" doctor --strict-runtime

  if [[ "${MEMO_INSTALL_SKIP_AGENT_CONFIG:-0}" != "1" ]]; then
    say "configuring MCP clients: Claude Code, Codex, OpenCode, Windsurf"
    if MEMO_NONINTERACTIVE=1 "$memo_bin" install-slash \
      --client claude-code \
      --client codex \
      --client opencode \
      --client windsurf \
      --best-effort; then
      say "agent clients configured"
    else
      warn "agent client configuration did not complete."
      warn "Re-run after installing clients: memo install-slash --client claude-code --client codex --client opencode --client windsurf"
    fi
  else
    say "skipping agent client configuration (MEMO_INSTALL_SKIP_AGENT_CONFIG=1)"
  fi

  # Cross-Mac corpus: clone the git-synced memo-sync repo and point this
  # install at it (opt-in, mirrors memflow's MEMFLOW_DATA_REMOTE cutover).
  # Idempotent — re-running reuses the existing clone.
  if [[ -n "${MEMO_SYNC_REMOTE:-}" ]]; then
    local sync_dest="${MEMO_SYNC_DEST:-$HOME/repos/memo-sync}"
    say "wiring git-synced corpus: $MEMO_SYNC_REMOTE → $sync_dest"
    if MEMO_NONINTERACTIVE=1 "$memo_bin" sync bootstrap "$MEMO_SYNC_REMOTE" --dest "$sync_dest"; then
      say "corpus wired; SessionStart pull / Stop push hooks keep it in sync"
    else
      warn "sync bootstrap failed — private repo needs git creds (SSH key/PAT) on this Mac."
      warn "Re-run after auth: memo sync bootstrap $MEMO_SYNC_REMOTE --dest $sync_dest"
    fi
  else
    say "no shared corpus wired (memo starts empty). To join an existing corpus on"
    say "  a fresh Mac, re-run with: MEMO_SYNC_REMOTE=<git-url> ./install.sh"
  fi

  say "MCP registration command (manual fallback):"
  MEMO_NONINTERACTIVE=1 "$memo_bin" mcp-command
}

main "$@"
