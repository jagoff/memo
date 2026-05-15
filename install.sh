#!/usr/bin/env bash
set -euo pipefail

APP_NAME="mlx-memo"
OLD_APP_NAME="memo-mcp"
DEFAULT_SPEC="mlx-memo"
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
  elif [[ "${MEMO_INSTALL_FROM_GIT:-0}" == "1" ]]; then
    printf '%s\n' "$GIT_SPEC"
  elif [[ -n "${MEMO_VERSION:-}" ]]; then
    printf '%s==%s\n' "$DEFAULT_SPEC" "$MEMO_VERSION"
  else
    printf '%s\n' "$DEFAULT_SPEC"
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

  ensure_default_dirs

  local memo_bin
  memo_bin="$(resolve_memo_bin)" || die "memo was installed but no memo binary was found in PATH or ~/.local/bin"

  say "installed: $("$memo_bin" --version)"
  say "runtime check:"
  MEMO_NONINTERACTIVE=1 "$memo_bin" doctor --strict-runtime

  say "MCP registration command:"
  MEMO_NONINTERACTIVE=1 "$memo_bin" mcp-command
}

main "$@"
