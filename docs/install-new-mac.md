# Installing memo on another Mac

Use this checklist for a fresh Apple Silicon Mac or a second machine that should
run the same local memo memory.

> Installing on **Linux / Ubuntu** instead? memo runs there standalone via a CPU
> `sentence-transformers` backend (using the official PyTorch CPU index). See
> [ubuntu.md](ubuntu.md) for that path and its trade-offs.

## 1. Prepare the Mac

Requirements:

- Apple Silicon Mac (M1 or newer).
- Python 3.13 or newer.
- Enough free disk for the model cache (~8 GB for the default profile).
- Optional agent CLIs/apps: Claude Code, Codex CLI, and Devin Desktop.

If Python is missing, install it first:

```bash
brew install python@3.13
```

## 2. Install memo

Recommended path:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/v4.12.2/install.sh | bash
```

The installer uses `pipx`, downloads the default MLX models, runs
`memo doctor --strict-runtime`, and configures these agent clients when
available:

- Claude Code via `claude plugin ...` and `claude mcp add-json`.
- Codex via the memo skill/plugin plus `codex mcp add`.
- OpenCode via `opencode mcp add`.
- Devin Desktop by editing `~/.devin/mcp.json`.

If one of the CLIs is not installed yet, the installer warns and continues.
After installing that client, rerun:

```bash
memo install-slash --client claude-code --client codex --client opencode --client devin-desktop
```

Open the terminal configuration center to verify storage, hardware profile,
hooks, recall, privacy, and capture settings before moving data:

```bash
memo config
```

This is a terminal-native TUI and does not start a browser or web service. A new
machine gets a four-step wizard; existing Markdown values are loaded with their
effective source and can be reviewed before any file is written.

To explicitly track the latest PyPI release instead of the installer's pinned release:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/v4.12.2/install.sh | MEMO_INSTALL_FROM_PYPI=1 bash
```

To skip client configuration entirely:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/v4.12.2/install.sh | MEMO_INSTALL_SKIP_AGENT_CONFIG=1 bash
```

## 3. Move Your Data

There are three supported paths.

### Option A: Git Cross-Mac Sync

Use this when the old Mac already pushed a `memo-sync` repo with
`memo sync init`.

```bash
memo sync bootstrap git@github.com:yourname/memo-sync.git
memo doctor --strict-runtime
memo sync status --check-remote
```

`memo sync bootstrap` clones or reuses `~/repos/memo-sync`, points
`~/.config/memo/config.toml` at the synced `memories/` directory, rebuilds the
local sqlite index from Markdown, and imports signal snapshots. If an existing
config is repointed, memo first writes a sibling rollback copy named like
`config.toml.pre-sync-bootstrap.bak`.

After bootstrap, Claude Code hooks pull on `SessionStart`, push on `Stop`, and
run debounced `memo sync auto` during long sessions. Re-run agent wiring if you
changed model/profile env vars before bootstrap:

```bash
memo install-slash --client claude-code --client codex --client opencode --client devin-desktop
```

### Option B: Portable Backup

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

The portable zip includes memory Markdown files plus `memvec.db` and
`history.db`. `--reindex` rebuilds vectors on the new machine, which is safer
when model profiles or sqlite-vec versions changed.

### Option C: Synced Obsidian/iCloud Folder

If your memories already live in a synced folder, let the folder finish syncing
on the new Mac, then point memo at it:

```bash
memo config
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
memo install-slash --client claude-code --client codex --client opencode --client devin-desktop
memo reindex --rebuild
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
memo mcp-command --client devin-desktop
```

Open a new Claude Code, Codex, or OpenCode session after installing. Restart
Devin Desktop after `~/.devin/mcp.json` changes.

## 6. Useful Paths

| Path | Purpose |
| --- | --- |
| `~/Documents/memo` | Default Markdown memory folder |
| `~/.config/memo/memo-config.md` | Config index and human notes |
| `~/.config/memo/config/*-config.md` | Human-editable domain settings |
| `~/.config/memo/.transactions/` | TUI transaction manifests and rollback backups |
| `~/.config/memo/config.toml` | Legacy fallback and migration source |
| `~/.local/share/memo/memvec.db` | sqlite-vec index; safe to rebuild |
| `~/.local/share/memo/history.db` | Audit log used by time-machine |
| `~/.local/share/memo/graph.db` | Entity graph sidecar |
| `~/.local/share/memo/contradictions.db` | Legacy contradiction import source (read-only compatibility window) |
| `~/.devin/mcp.json` | Devin Desktop MCP config |
