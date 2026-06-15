# Installing memo on another Mac

Use this checklist for a fresh Apple Silicon Mac or a second machine that should
run the same local memo memory.

## 1. Prepare the Mac

Requirements:

- Apple Silicon Mac (M1 or newer).
- Python 3.13 or newer.
- Enough free disk for the model cache (~8 GB for the default profile).
- Optional agent CLIs/apps: Claude Code, Codex CLI, and Windsurf.

If Python is missing, install it first:

```bash
brew install python@3.13
```

## 2. Install memo

Recommended path:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash
```

The installer uses `pipx`, downloads the default MLX models, runs
`memo doctor --strict-runtime`, and configures these agent clients when
available:

- Claude Code via `claude plugin ...` and `claude mcp add-json`.
- Codex via the memo skill/plugin plus `codex mcp add`.
- OpenCode via `opencode mcp add`.
- Windsurf/Cascade by editing `~/.codeium/windsurf/mcp_config.json`.

If one of the CLIs is not installed yet, the installer warns and continues.
After installing that client, rerun:

```bash
memo install-slash --client claude-code --client codex --client opencode --client windsurf
```

To install the latest PyPI release instead of GitHub `master`:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | MEMO_INSTALL_FROM_PYPI=1 bash
```

To skip client configuration entirely:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | MEMO_INSTALL_SKIP_AGENT_CONFIG=1 bash
```

## 3. Move Your Data

There are two supported paths.

### Option A: Portable Backup

On the old Mac:

```bash
memo recall-daemon stop || true
memo backup --out ~/Desktop/memo-transfer.zip
```

Move `memo-transfer.zip` to the new Mac, then run:

```bash
memo restore ~/Desktop/memo-transfer.zip --reindex --yes
memo doctor --strict-runtime
```

The portable zip includes memoria Markdown files plus `memvec.db` and
`history.db`. `--reindex` rebuilds vectors on the new machine, which is safer
when model profiles or sqlite-vec versions changed.

### Option B: Synced Obsidian/iCloud/Git Folder

If your memorias already live in a synced folder, let the folder finish syncing
on the new Mac, then point memo at it:

```bash
memo init
memo reindex
memo doctor --strict-runtime
```

The `.md` files are the storage of record. `memvec.db` is rebuildable.
`history.db` is not rebuildable from Markdown alone; copy it too if you need
time-machine snapshots from before the move.

## 4. Preserve Model/Profile Env Vars

If the old Mac used a non-default profile, export the same env vars before
running `memo install-slash` so agent client configs get the same model
settings:

```bash
export MEMO_MODEL_PROFILE=quality
export MEMO_EMBEDDER_DIMS=2560
memo install-slash --client claude-code --client codex --client opencode --client windsurf
rm -f ~/.local/share/memo/memvec.db
memo reindex
```

The MCP registration stores current `MEMO_*` overrides in each client config.
That prevents shell sessions and GUI clients from using different embedding
dimensions.

## 5. Verify Clients

```bash
which -a memo
which -a memo-mcp
memo doctor --strict-runtime
memo mcp-command --client claude-code
memo mcp-command --client codex
memo mcp-command --client opencode
memo mcp-command --client windsurf
```

Open a new Claude Code, Codex, or OpenCode session after installing. In Windsurf, refresh
Cascade MCP servers after `~/.codeium/windsurf/mcp_config.json` changes.
Windsurf documents that config path and refresh flow in its MCP docs:
<https://docs.windsurf.com/windsurf/cascade/mcp>.

## 6. Useful Paths

| Path | Purpose |
| --- | --- |
| `~/Documents/memo` | Default Markdown memoria folder |
| `~/.config/memo/config.toml` | Stores the selected `data_dir` / `vault_path` |
| `~/.local/share/memo/memvec.db` | sqlite-vec index; safe to rebuild |
| `~/.local/share/memo/history.db` | Audit log used by time-machine |
| `~/.local/share/memo/graph.db` | Entity graph sidecar |
| `~/.local/share/memo/contradictions.db` | Contradiction radar sidecar |
| `~/.codeium/windsurf/mcp_config.json` | Windsurf/Cascade MCP config |
