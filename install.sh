#!/usr/bin/env bash
set -euo pipefail

APP_NAME="mlx-memo"
OLD_APP_NAME="memo-mcp"
PYPI_SPEC="mlx-memo"
DEFAULT_VERSION="4.8.0"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=13

# Installer backend: "uv" (preferred) or "pipx" (fallback).
# Detected at main() start — set here so helper functions can reference it.
USE_UV=0
UV_BIN=""

# ── UI / feedback ────────────────────────────────────────────────────────────
# Colors + spinners are enabled only on an interactive, color-capable TTY.
# `curl | bash` (no TTY), NO_COLOR, or TERM=dumb fall back to plain log lines.
STEP=0
TOTAL=8
USE_FANCY=false
BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""; RESET=""

setup_ui() {
  if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-dumb}" != "dumb" ]]; then
    USE_FANCY=true
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
    YELLOW=$'\033[33m'; BLUE=$'\033[34m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
    trap 'tput cnorm 2>/dev/null || true' EXIT
  fi
}

banner() {
  printf '\n%s%s  memo%s\n' "$BOLD" "$CYAN" "$RESET"
  printf '%s  local-first semantic memory · installer%s\n' "$DIM" "$RESET"
  printf '%s  ────────────────────────────────────────%s\n' "$DIM" "$RESET"
}

phase() {
  STEP=$((STEP + 1))
  printf '\n%s%s▶ [%d/%d] %s%s\n' "$BOLD" "$BLUE" "$STEP" "$TOTAL" "$1" "$RESET"
}

say() { printf '  %s·%s %s\n' "$DIM" "$RESET" "$*"; }
ok()  { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$*"; }

warn() { printf '  %s!%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }

die() {
  printf '  %s✗ error:%s %s\n' "$RED" "$RESET" "$*" >&2
  exit 1
}

# Run a command behind an animated spinner, capturing its output. On success the
# line collapses to a ✓; on failure the captured log is printed so the error is
# visible. Falls back to a plain log line + streamed output when not on a TTY.
# Usage: spin "message" cmd args...   (cmd may be a function or `env VAR=v prog`)
spin() {
  local msg="$1"; shift
  if [[ "$USE_FANCY" != true ]]; then
    say "$msg"
    "$@"
    return $?
  fi
  local log; log="$(mktemp)"
  "$@" >"$log" 2>&1 &
  local pid=$! frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏' i=0 start=$SECONDS
  tput civis 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null; do
    printf '\r  %s%s%s %s %s(%ds)%s' \
      "$CYAN" "${frames:i++%${#frames}:1}" "$RESET" "$msg" "$DIM" "$((SECONDS - start))" "$RESET"
    sleep 0.1
  done
  local rc=0; wait "$pid" || rc=$?
  tput cnorm 2>/dev/null || true
  if [[ $rc -eq 0 ]]; then
    printf '\r\033[K  %s✓%s %s %s(%ds)%s\n' \
      "$GREEN" "$RESET" "$msg" "$DIM" "$((SECONDS - start))" "$RESET"
  else
    printf '\r\033[K  %s✗%s %s\n' "$RED" "$RESET" "$msg"
    sed 's/^/      /' "$log" >&2
  fi
  rm -f "$log"
  return $rc
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
  # uv manages its own Python runtimes — check if uv can provide one.
  local uv_bin
  uv_bin="$(command -v uv 2>/dev/null || echo "${HOME}/.local/bin/uv")"
  if [[ -x "$uv_bin" ]]; then
    local uv_py
    uv_py="$("$uv_bin" python find ">=${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}" 2>/dev/null)" || true
    if [[ -n "$uv_py" ]] && python_ok "$uv_py"; then
      printf '%s\n' "$uv_py"
      return 0
    fi
  fi
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
    ok "pipx found: $PIPX_BIN"
    return 0
  fi

  if "$PYTHON_BIN" -m pipx --version >/dev/null 2>&1; then
    PIPX_BIN=""
    ok "pipx available via $PYTHON_BIN -m pipx"
    return 0
  fi

  if [[ "${MEMO_INSTALL_NO_BOOTSTRAP_PIPX:-0}" == "1" ]]; then
    die "pipx is not installed. Install pipx first, or unset MEMO_INSTALL_NO_BOOTSTRAP_PIPX."
  fi

  if command -v brew >/dev/null 2>&1; then
    spin "pipx not found — installing with Homebrew" brew install pipx \
      || die "Homebrew pipx install failed (see log above)"
    PIPX_BIN="$(command -v pipx)"
    return 0
  fi

  spin "pipx not found — installing with ${PYTHON_BIN} -m pip --user" \
    "$PYTHON_BIN" -m pip install --user pipx \
    || die "pip pipx install failed (see log above)"
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
    printf '%s==%s\n' "$PYPI_SPEC" "$DEFAULT_VERSION"
  fi
}

clean_old_pipx_package() {
  if run_pipx list --short 2>/dev/null | awk '{print $1}' | grep -qx "$OLD_APP_NAME"; then
    spin "removing old pipx package: $OLD_APP_NAME" run_pipx uninstall "$OLD_APP_NAME" \
      || warn "could not remove old package $OLD_APP_NAME (non-fatal)"
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

# Detect which agent CLIs/apps are present on PATH so the install can report
# coverage. `memo install-slash --client all` is still the source of truth (it
# skips absent clients); this is purely a human-readable summary. Sets the global
# DETECTED_AGENTS to a comma-separated list (empty when none found).
DETECTED_AGENTS=""
detect_agents() {
  local agent found=()
  for agent in claude codex opencode devin blackbox cursor gemini; do
    if command -v "$agent" >/dev/null 2>&1; then
      found+=("$agent")
    fi
  done
  if [[ ${#found[@]} -gt 0 ]]; then
    local joined; printf -v joined '%s, ' "${found[@]}"
    DETECTED_AGENTS="${joined%, }"
  else
    DETECTED_AGENTS=""
  fi
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

# Informative (not a question) explanation of why memo needs to download models,
# what they are, and roughly how long it takes. Shown right before the download
# begins so the user understands the wait and the disk usage.
explain_models() {
  printf '  %smemo needs local MLX models to work — without them, retrieval and%s\n' "$DIM" "$RESET"
  printf '  %sambient recall cannot run. Downloads land in ~/.cache/huggingface:%s\n' "$DIM" "$RESET"
  printf '      %s•%s embedder   Qwen3-Embedding-0.6B-4bit   ~0.4 GB  %s(required · now)%s\n' "$CYAN" "$RESET" "$DIM" "$RESET"
  printf '      %s•%s reranker   Qwen3-Reranker-0.6B          ~0.6 GB  %s(required · now)%s\n' "$CYAN" "$RESET" "$DIM" "$RESET"
  # shellcheck disable=SC2016  # backticks are intentional user-facing Markdown
  printf '      %s•%s chat LLM   7B + 4B helper               ~6   GB  %s(for `memo ask` · background)%s\n' "$CYAN" "$RESET" "$DIM" "$RESET"
  printf '  %sRequired models (~1 GB) download now (~3 min); chat models continue%s\n' "$DIM" "$RESET"
  printf '  %sin the background so the install finishes without waiting on them.%s\n' "$DIM" "$RESET"
  printf '  %sNote: large weight files can sit at 0%% a while as they transfer —%s\n' "$DIM" "$RESET"
  printf '  %sthat is normal, not a hang.%s\n' "$DIM" "$RESET"
}

# Hugging Face throttles unauthenticated downloads (~5 MB/s), which is the main
# reason the model download feels stuck. Nudge the user toward a token; it is
# inherited from the environment by the prewarm subprocesses when set.
hf_token_note() {
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    warn "no HF_TOKEN set — Hugging Face throttles unauthenticated downloads (~5 MB/s)."
    warn "Export HF_TOKEN=<token> before re-running to download large models faster."
  fi
}

# Download the chat LLM models (7B + 4B helper, ~6 GB) in a detached background
# process so the install completes immediately. memo's retrieval runs without
# them; they're only needed for `memo ask`, which also lazy-loads on first use.
# Opt out with MEMO_INSTALL_DOWNLOAD_CHAT=0.
start_chat_download_bg() {
  case "${MEMO_INSTALL_DOWNLOAD_CHAT:-auto}" in
    no|false|0|n|N)
      say "chat models: skipped (MEMO_INSTALL_DOWNLOAD_CHAT=0) — they load on first \`memo ask\`."
      return 0
      ;;
  esac
  local log="${MEMO_STATE_DIR:-$HOME/.local/share/memo}/chat-download.log"
  mkdir -p "$(dirname "$log")"
  nohup env MEMO_NONINTERACTIVE=1 "$1" prewarm --download-all >"$log" 2>&1 </dev/null &
  disown 2>/dev/null || true
  ok "chat models (~6 GB) downloading in the background → $log"
  say "\`memo ask\` works once it finishes · follow with: tail -f $log"
}

# Decide whether to download MLX models during install. The embedder + reranker
# are required for retrieval and ambient recall, so the default is yes; the chat
# models (handled separately, in the background) are only for `memo ask`. The
# user can decline interactively or force a value via MEMO_INSTALL_DOWNLOAD_MODELS.
should_download_models() {
  case "${MEMO_INSTALL_DOWNLOAD_MODELS:-auto}" in
    yes|true|1|Y|y) return 0 ;;
    no|false|0|N|n) return 1 ;;
    auto|"")
      # Skip prompt if the required embedder is already cached
      if [[ -d "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-Embedding-0.6B-4bit-DWQ" ]]; then
        ok "required models already cached — skipping download"
        return 1
      fi
      ask_yes_no "  ${YELLOW}?${RESET} Download required MLX models now (~1 GB; chat models download in the background)? [Y/n]" Y
      ;;
    *)
      warn "unknown MEMO_INSTALL_DOWNLOAD_MODELS='${MEMO_INSTALL_DOWNLOAD_MODELS}', defaulting to yes"
      return 0
      ;;
  esac
}

main() {
  setup_ui
  banner

  phase "Checking environment"
  if [[ "${MEMO_INSTALL_SKIP_PLATFORM_CHECK:-0}" != "1" ]] && ! is_macos_arm64; then
    die "memo requires macOS on Apple Silicon (Darwin arm64). Set MEMO_INSTALL_SKIP_PLATFORM_CHECK=1 to bypass."
  fi
  ok "platform: $(uname -s) $(uname -m)"

  # Prefer uv when available — it manages its own Python so no system Python
  # >= 3.13 is required, and it's not subject to PEP 668 externally-managed errors.
  UV_BIN="$(command -v uv 2>/dev/null || echo "")"
  [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]] && UV_BIN="$HOME/.local/bin/uv"
  if [[ -n "$UV_BIN" && -x "$UV_BIN" ]]; then
    USE_UV=1
    ok "uv found: $UV_BIN (using uv tool install)"
  else
    USE_UV=0
    PYTHON_BIN="$(find_python)" || die "Python >= 3.13 is required. Install python@3.13 or python@3.14 first."
    export PYTHON_BIN
    ok "Python: $PYTHON_BIN"
    phase "Preparing pipx"
    ensure_pipx
  fi

  phase "Installing $APP_NAME"
  local spec
  spec="$(install_spec)"
  if [[ "$USE_UV" == "1" ]]; then
    spin "installing from $spec" "$UV_BIN" tool install "$spec" --force \
      || die "uv tool install failed (see log above)"
    # Ensure ~/.local/bin is on PATH for this session.
    export PATH="$HOME/.local/bin:$PATH"
  else
    spin "installing from $spec" run_pipx install "$spec" --force \
      || die "pipx install failed (see log above)"
    run_pipx ensurepath >/dev/null 2>&1 || true
    # Remove the historical package only after the replacement is healthy.
    clean_old_pipx_package
  fi

  local memo_bin
  memo_bin="$(resolve_memo_bin)" || die "memo was installed but no memo binary was found in PATH or ~/.local/bin"
  ok "installed: $("$memo_bin" --version)"

  ensure_default_dirs

  phase "MLX models (required for retrieval)"
  if should_download_models; then
    explain_models
    hf_token_note
    # The mlx-community model repos are Xet-backed; Xet's chunked transfer is
    # slow and bursty on some links. hf_transfer (parallel multi-connection HTTP)
    # saturates the link, but only on the classic LFS path — so disable Xet for
    # the install-time download. Scoped to this process (export here is inherited
    # by the prewarm children); runtime sessions keep Xet's dedup. Opt out with
    # MEMO_INSTALL_FAST_DOWNLOAD=0.
    if [[ "${MEMO_INSTALL_FAST_DOWNLOAD:-1}" != "0" ]]; then
      local hf_ok=0
      if [[ "$USE_UV" == "1" ]]; then
        local uv_venv="$HOME/.local/share/uv/tools/$APP_NAME"
        "$UV_BIN" pip install --python "$uv_venv" hf_transfer >/dev/null 2>&1 && hf_ok=1 || true
      else
        run_pipx inject "$APP_NAME" hf_transfer >/dev/null 2>&1 && hf_ok=1 || true
      fi
      if [[ "$hf_ok" == "1" ]]; then
        export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1
        ok "fast downloads enabled (hf_transfer, parallel HTTP)"
      else
        warn "hf_transfer install failed — downloads use the default backend (slower)."
      fi
    fi
    printf '\n'
    # Required models (embedder + reranker, ~1 GB) — synchronous so retrieval
    # works the moment the install finishes. Streamed (not spinner-wrapped) so
    # Hugging Face's live progress bars show real percentages.
    if env MEMO_NONINTERACTIVE=1 "$memo_bin" prewarm; then
      ok "required models ready (embedder + reranker)"
    else
      warn "required model download did not complete — they load lazily on first use."
      warn "Re-run: MEMO_NONINTERACTIVE=1 memo prewarm"
    fi
    # Chat models (~6 GB, only for `memo ask`) — detached background download.
    start_chat_download_bg "$memo_bin"
  else
    say "skipping MLX model download (models will load lazily on first use)."
    say "Run later: MEMO_NONINTERACTIVE=1 memo prewarm --download-all"
  fi

  phase "Runtime check"
  # Informational, not a gate: doctor reports non-fatal issues too (data_dir not
  # yet bootstrapped, chat models still downloading in the background, a stale
  # recall socket). Aborting here would skip agent-client config and corpus
  # wiring for problems the user can resolve later, so report and continue.
  if env MEMO_NONINTERACTIVE=1 "$memo_bin" doctor --strict-runtime; then
    ok "runtime healthy"
  else
    warn "runtime check reported issues (above) — install continues; resolve with \`memo doctor\`."
  fi

  phase "Configuring agent clients"
  if [[ "${MEMO_INSTALL_SKIP_AGENT_CONFIG:-0}" != "1" ]]; then
    # Probe which agent CLIs/apps are on PATH so the user sees coverage before
    # wiring. install-slash --client all is the source of truth (it skips absent
    # clients); detection is only a report and never gates the wiring.
    detect_agents
    if [[ -n "$DETECTED_AGENTS" ]]; then
      say "detected agents: $DETECTED_AGENTS"
    else
      say "no agent CLIs detected on PATH — wiring anyway (install-slash skips absent clients)"
    fi
    # Configure every supported client (claude-code, codex, devin, opencode,
    # devin-desktop, blackbox) so the MCP lands in all tools present on the machine.
    # `devin` covers the devin CLI and Devin Desktop, which share
    # ~/.devin/mcp.json. --best-effort skips clients that aren't installed.
    if spin "wiring MCP into all available clients (best-effort)" \
      env MEMO_NONINTERACTIVE=1 "$memo_bin" install-slash \
      --client all \
      --best-effort; then
      if [[ -n "$DETECTED_AGENTS" ]]; then
        ok "agent clients configured (detected: $DETECTED_AGENTS)"
      else
        ok "agent clients configured"
      fi
    else
      warn "agent client configuration did not complete."
      warn "Re-run after installing clients: memo install-slash --client all --best-effort"
    fi
  else
    say "skipping agent client configuration (MEMO_INSTALL_SKIP_AGENT_CONFIG=1)"
  fi

  phase "Statusline badge"
  # Install the memo version-badge statusline into Claude Code (no-clobber: a
  # user with an existing statusLine keeps it). Best-effort — never fail the
  # install over a cosmetic badge.
  if spin "installing the [MEMO <version>] statusline badge (no-clobber)" \
    env MEMO_NONINTERACTIVE=1 "$memo_bin" install-statusline; then
    ok "statusline badge installed"
  else
    warn "statusline install did not complete (non-fatal)."
    warn "Re-run manually: memo install-statusline"
  fi
  say "only Claude Code exposes a native statusline; other agents (Codex/OpenCode/"
  say "  Devin/Gemini/Blackbox) have no status bar — memo runs there as an MCP server"
  say "  (visible in the agent's tool list), just without a persistent badge."

  phase "Shared corpus"
  # Cross-Mac corpus: clone the git-synced memo-sync repo and point this
  # install at it when explicitly requested.
  # Idempotent — re-running reuses the existing clone.
  if [[ -n "${MEMO_SYNC_REMOTE:-}" ]]; then
    local sync_dest="${MEMO_SYNC_DEST:-$HOME/repos/memo-sync}"
    if spin "wiring git-synced corpus: $MEMO_SYNC_REMOTE → $sync_dest" \
      env MEMO_NONINTERACTIVE=1 "$memo_bin" sync bootstrap "$MEMO_SYNC_REMOTE" --dest "$sync_dest"; then
      ok "corpus wired; SessionStart pull / Stop push hooks keep it in sync"
    else
      warn "sync bootstrap failed — private repo needs git creds (SSH key/PAT) on this Mac."
      warn "Re-run after auth: memo sync bootstrap $MEMO_SYNC_REMOTE --dest $sync_dest"
    fi
  else
    # No remote requested this run — but a prior install may already have wired
    # one (data_dir inside a git clone). Detect that so we don't tell a working
    # install it "starts empty".
    local sync_state sync_remote sync_url sync_dest
    sync_state="$(env MEMO_NONINTERACTIVE=1 "$memo_bin" sync status 2>/dev/null || true)"
    if printf '%s' "$sync_state" | grep -q 'remote:'; then
      sync_remote="$(printf '%s\n' "$sync_state" | sed -n 's/.*remote:[[:space:]]*//p' | head -1)"
      ok "shared corpus already wired: ${sync_remote:-git clone} — SessionStart pull / Stop push keep it in sync"
    elif [[ -t 0 ]] && ask_yes_no "  ${YELLOW}?${RESET} Connect a git repo for cross-device memory sync? [y/N]" n; then
      read -rp "  git remote URL (e.g. git@github.com:you/memo-sync.git): " sync_url || sync_url=""
      if [[ -n "$sync_url" ]]; then
        sync_dest="${MEMO_SYNC_DEST:-$HOME/repos/memo-sync}"
        if spin "wiring corpus: $sync_url" \
          env MEMO_NONINTERACTIVE=1 "$memo_bin" sync bootstrap "$sync_url" --dest "$sync_dest"; then
          ok "corpus wired; SessionStart pull / Stop push hooks keep it in sync"
        else
          warn "sync bootstrap failed — private repo needs git creds (SSH key/PAT) on this Mac."
          warn "Re-run after auth: memo sync bootstrap $sync_url --dest $sync_dest"
        fi
      else
        say "no URL entered — skipping cross-device sync (memo stays local)."
      fi
    elif [[ -t 0 ]]; then
      say "skipping cross-device sync (memo stays local). Re-run with"
      say "  MEMO_SYNC_REMOTE=<url> or \`memo sync bootstrap <url>\` to enable later."
    else
      say "no shared corpus wired (memo starts empty). To join an existing corpus on"
      say "  a fresh Mac, re-run with: MEMO_SYNC_REMOTE=<git-url> ./install.sh"
    fi
  fi

  # Final recap — derive sync state fresh so it reflects whatever this run wired
  # (env path, interactive prompt, or a pre-existing clone).
  local recap_state recap_sync
  recap_state="$(env MEMO_NONINTERACTIVE=1 "$memo_bin" sync status 2>/dev/null || true)"
  if printf '%s' "$recap_state" | grep -q 'remote:'; then
    recap_sync="git: $(printf '%s\n' "$recap_state" | sed -n 's/.*remote:[[:space:]]*//p' | head -1)"
  else
    recap_sync="local-only"
  fi

  printf '\n%s%s  ✓ memo is ready%s\n' "$BOLD" "$GREEN" "$RESET"
  say "version:  $("$memo_bin" --version)"
  say "agents:   ${DETECTED_AGENTS:-configured via install-slash} · tagline on Claude Code (others: MCP-only)"
  say "sync:     $recap_sync"
  printf '  %sMCP registration command (manual fallback):%s\n' "$DIM" "$RESET"
  env MEMO_NONINTERACTIVE=1 "$memo_bin" mcp-command
}

main "$@"
