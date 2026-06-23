# memo — reference manual

The [README](../README.md) is the front door. This is the full manual: install
detail, per-client MCP setup, the complete tool and CLI surface, every `MEMO_*`
knob, design notes, and how memo compares to other agent-memory projects.

- [Install detail](#install-detail)
- [MCP setup](#mcp-setup)
- [MCP tools](#mcp-tools)
- [Ambient memory](#ambient-memory)
- [Surfaces](#surfaces) — session briefing, semantic map, time-machine
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Design and comparison](#design-and-comparison)

---

## Install detail

Recommended install: keep memo isolated as its own tool. Do **not** vendor it
inside another project's `.venv`; the MLX runtime, model cache, MCP server,
sqlite state, and CLI should move together as one subsystem.

```bash
# One-line installer (pipx under the hood, installs GitHub master,
# and configures Claude Code + Codex + OpenCode + Windsurf when available)
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash
# or install the latest published PyPI release explicitly
pipx install mlx-memo
# or
uv tool install mlx-memo
# or via the Homebrew tap
brew tap jagoff/memo && brew install mlx-memo
```

Any of those expose two binaries: `memo` (CLI) and `memo-mcp` (MCP server). For
MCP clients, prefer an isolated tool install so memo's MLX dependencies, sqlite
state, and `memo-mcp` runtime stay independent from whichever repo is active in
your shell.

> The PyPI distribution is **`mlx-memo`** as of 0.5.0. Earlier versions shipped
> as `memo-mcp`; the binary names haven't changed, so existing MCP configs keep
> working. The one-line installer installs GitHub `master` by default so it can
> deploy repo changes before the next PyPI release exists.

If you are developing this repo and want the real system install to use your
checkout:

```bash
pipx install --force /path/to/memo
memo doctor --strict-runtime
memo --version
```

### Installer knobs

```bash
# Install the latest published PyPI release instead of GitHub master.
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | MEMO_INSTALL_FROM_PYPI=1 bash

# Pin a published PyPI version.
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | MEMO_VERSION=0.6.0 bash

# Install from an explicit pipx spec (local checkout, git ref, wheel, etc.).
MEMO_INSTALL_SPEC=/path/to/memo ./install.sh

# Skip agent-client configuration during install.
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | MEMO_INSTALL_SKIP_AGENT_CONFIG=1 bash

# Force-skip the MLX model download (models load lazily on first use).
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | MEMO_INSTALL_DOWNLOAD_MODELS=no bash

# Force-yes the MLX model download (skip the interactive confirmation).
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | MEMO_INSTALL_DOWNLOAD_MODELS=yes bash
```

**Model download** is part of memo's structure (embedder + reranker + chat
models are required for retrieval and ambient recall). On an interactive
terminal the installer asks for confirmation (default `Y`); on a piped install
the default is also yes. Re-run the download manually any time:

```bash
# Download all default-profile models (~7 GB, shows progress, safe to re-run)
MEMO_NONINTERACTIVE=1 memo prewarm --download-all

# Or download individual models with the HF CLI
hf download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
hf download mku64/Qwen3-Reranker-0.6B-mlx-8Bit
hf download mlx-community/Qwen2.5-3B-Instruct-4bit
hf download mlx-community/Qwen2.5-7B-Instruct-4bit

# Optional quality profile.
hf download mlx-community/Qwen3-Embedding-4B-4bit-DWQ
hf download mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510
```

### Stack

| Component | Choice | Why |
|---|---|---|
| LLM (chat) | [`Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) + [`3B helper`](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit) via [`mlx-lm`](https://github.com/ml-explore/mlx-lm) | Two-tier; 7B for `ask()` synthesis, 3B for cheap helpers. Both 4-bit fit comfortably. |
| Embedder | [`Qwen3-Embedding-0.6B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) by default; [`Qwen3-Embedding-4B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ) in `quality` | 1024-dim default, 2560-dim quality. Choose via `MEMO_MODEL_PROFILE`. |
| Reranker | [`mku64/Qwen3-Reranker-0.6B-mlx-8Bit`](https://huggingface.co/mku64/Qwen3-Reranker-0.6B-mlx-8Bit) | Cross-encoder over top-30 from vec+BM25, then alpha-fusion. |
| Vector store | [`sqlite-vec`](https://github.com/asg017/sqlite-vec) | One file, no daemon, embedded. Reset = `rm memvec.db`. |
| Source of truth | Markdown files under `MEMO_DATA_DIR` with YAML frontmatter | Human-editable; sync via iCloud/git/Syncthing. |
| MCP transport | [`fastmcp`](https://github.com/jlowin/fastmcp) | Stdio out of the box. |

### Installing on another Mac

For a fresh Apple Silicon Mac, run the one-line installer first, then bring over
the corpus:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash
memo doctor --strict-runtime
memo install-slash --client claude-code --client codex --client opencode --client windsurf
```

To move existing data:

```bash
# On the old Mac: portable zip with .md memorias + memvec.db + history.db.
memo backup --out ~/Desktop/memo-transfer.zip

# On the new Mac, after installing memo:
memo restore ~/Desktop/memo-transfer.zip --reindex --yes
memo doctor --strict-runtime
```

If your memorias already live in an iCloud/Syncthing/Git-synced Obsidian folder,
point the new Mac at that same folder instead of copying the zip:

```bash
memo init
memo reindex
```

`MEMO_DATA_DIR` holds the human-readable `.md` source of truth; `MEMO_STATE_DIR`
(default `~/.local/share/memo`) holds rebuildable indexes plus sidecars such as
`history.db` — keep `history.db` if you want time-machine snapshots to survive
the move. Full checklist: [install-new-mac.md](install-new-mac.md).

### Verify no old install is being used

```bash
which -a memo
which -a memo-mcp
pipx list --short
memo doctor --strict-runtime
```

A healthy isolated install prints a single `memo` path, resolves `memo` and
`memo-mcp` from the same environment, and passes `memo doctor --strict-runtime`.

---

## MCP setup

After installing `mlx-memo`, register the MCP with your client. The `memo` CLI
prints commands pinned to the resolved `memo-mcp` executable so clients don't
accidentally start a copy from a project `.venv`:

```bash
memo install-slash
```

`install-slash` configures Claude Code, Codex, Windsurf, and Devin where each
supports it, and forwards current `MEMO_*` model/storage env vars into each MCP
client config. This matters with the 2560-dim quality embedder: GUI clients
often don't inherit your shell env, and a 1024/2560 mismatch breaks semantic
search until the config is updated or `memvec.db` is rebuilt.

Released wheels include the Claude/Codex/Devin agent assets, so a normal
`pipx` / `uv tool` / Homebrew install is enough. When developing from a local
checkout, pass `--repo /path/to/memo` to test uncommitted plugin changes.

Tools surface inside the agent as `mcp__memo__memo_*`. Agent installs default to
a five-tool surface (`briefing`, `search`, `ask`, `get`, `save`) so
administrative schemas don't consume model context — set `MEMO_MCP_PROFILE=core`
(25 tools) or `full` (everything) only for a dedicated administrative client.

### Claude Code

```bash
memo mcp-command --client claude-code
# then run the printed command, e.g.
claude mcp add-json -s user memo '{"type":"stdio","command":"/Users/you/.local/pipx/venvs/mlx-memo/bin/memo-mcp","args":[],"env":{"MEMO_NONINTERACTIVE":"1"}}'
```

Or hand-edit `~/.claude.json`:

```jsonc
{
  "mcpServers": {
    "memo": {
      "type": "stdio",
      "command": "/path/to/memo-mcp",
      "args": [],
      "env": { "MEMO_NONINTERACTIVE": "1" }
    }
  }
}
```

Restart Claude Code. If it starts the wrong server, run
`memo doctor --strict-runtime` — it warns when `memo`/`memo-mcp` resolve from a
project-local venv or from different environments.

### Codex CLI

```bash
memo mcp-command --client codex
# then run the printed command, e.g.
codex mcp add memo --env MEMO_NONINTERACTIVE=1 --env MEMO_MCP_PROFILE=agent -- /Users/you/.local/pipx/venvs/mlx-memo/bin/memo-mcp
codex mcp list
memo install-slash --client codex   # also installs the memo skill
```

Current Codex CLI builds list only built-in slash commands in the TUI
dispatcher. The installer writes the exact `memo` skill to
`$CODEX_HOME/skills/memo/SKILL.md`; Codex loads it as a model-visible skill and
routes to the `memo` MCP server, but `/memo` won't appear in that TUI menu until
Codex exposes custom skills there.

### Devin for Terminal

```bash
memo mcp-command --client devin
# then run the printed command, e.g.
devin mcp add -s user -e MEMO_NONINTERACTIVE=1 memo -- /Users/you/.local/pipx/venvs/mlx-memo/bin/memo-mcp
devin mcp list
```

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```jsonc
{
  "mcpServers": {
    "memo": {
      "command": "/path/to/memo-mcp",
      "env": { "MEMO_NONINTERACTIVE": "1" }
    }
  }
}
```

### Windsurf / Cascade

Windsurf stores Cascade MCP servers in `~/.codeium/windsurf/mcp_config.json`.
memo can write that file directly (`memo install-slash --client windsurf`) or
print the JSON block for manual editing (`memo mcp-command --client windsurf`).
It preserves any existing `mcpServers` and only replaces the `memo` entry. Set
`WINDSURF_MCP_CONFIG` for a non-standard config path.

### Cursor / Cline / Continue

Each has its own MCP config UI but the contract is the same: register a stdio
server pointing at the `memo-mcp` binary. Print a portable `mcpServers` block
with `memo mcp-command --client json`.

### The `/memo` slash command

`/memo` ships only for CLIs that can expose an exact custom `/memo`; the backend
is always the same isolated `memo-mcp` server.

```bash
# Claude Code — registers the /memo skill, MCP server, and ambient hooks together
memo install-slash --client claude-code
# or manually:
claude plugin marketplace add jagoff/memo
claude plugin install memo@memo -s user

# Codex — installs a user skill + the Codex plugin
memo install-slash --client codex

# Devin — installs the /memo router skill under ~/.config/devin/skills/
memo install-slash --client devin
```

Restart the client (or open a new session) after installing from the CLI so the
slash-command registry reloads. The skill routes user input to the right MCP
tool:

| Input | Action |
|---|---|
| `/memo <query>` | semantic search (k=5, snippet body) |
| `/memo` | smart capture — distills the turn's insight and saves it |
| `/memo list [n]` | recent memories |
| `/memo save <text>` | save with auto-derived type/tags |
| `/memo get <id\|prefix>` | full record (prefix ≥4 chars) |
| `/memo update <id\|prefix> [flags] [body]` | patch metadata or body |
| `/memo delete <id\|prefix>` | delete (asks confirmation) |
| `/memo ask <question>` | RAG synthesis with citations |
| `/memo stats` | totals + paths + models |
| `/memo reindex` | absorb edits made directly in Obsidian |
| `/memo history [op] [id]` | audit log of save/update/delete |
| `/memo consolidate [threshold]` | cluster near-duplicates + merge proposals |
| `/memo mapa [--output FILE]` | generate 2D semantic canvas HTML |
| `/memo doctor [--gc] [--fix]` | self-check + orphan detect |

---

## MCP tools

| Tool | What it does |
|---|---|
| `memo_save(content, title?, type?, tags?)` | Persist a new memory; returns the full record. |
| `memo_search(query, limit?, type?, body_chars=280, mode="hybrid")` | Top-k. `hybrid` (default) fuses vec + bm25 via RRF, then optionally re-ranks. `vec` is semantic only; `bm25` is keyword (FTS5 unicode61, diacritic-stripping for Spanish). |
| `memo_list(limit?, type?)` | Recent by `updated` desc. |
| `memo_get(id)` | Full record. Accepts a unique prefix ≥4 chars (git-style); returns `{"error": "ambiguous", "matches": [...]}` on collision. |
| `memo_update(id, title?, type?, tags?, content?)` | Patches fields; re-embeds only if body changed. |
| `memo_reindex()` | Re-scan vault, re-embed entries whose `body_hash` diverged. |
| `memo_delete(id)` | Removes from vec + disk. |
| `memo_ask(question)` | RAG synthesis; cites memorias by id. |
| `memo_chat_ask(question, history?, context?)` | Chat-shaped RAG envelope (`memo.chat_ask.v2`) with answer, citations, retrieval trace, and synthesis status. |
| `memo_stats()` | Counts, paths, active models. |
| `memo_history(limit?, record_id?)` | Recent save/update/delete events, optionally filtered to one record. |
| `memo_record_diff(id, limit?)` | Chronological audit trail for one record with field-level diffs. |
| `memo_consolidate()`, `memo_extract_entities()`, `memo_entities()` | Corpus maintenance — see CHANGELOG. |

---

## Ambient memory

Install the bundled Claude Code plugin and memo silently consults your past on
every prompt and injects the most relevant memorias as `additionalContext` —
**the agent sees them before answering**, no manual invocation. With the plugin,
six hooks plug in automatically:

| Event | Command | Mode | Budget | Purpose |
|---|---|---|---|---|
| `SessionStart` (startup/clear) | `memo prewarm` | async | 30 s | Pre-loads MLX embedder + reranker; writes warm-signal file |
| `SessionStart` (startup/clear) | `memo recall-daemon start` | async | 5 s | Starts the recall daemon (keeps embedder in RAM; <200 ms recall) |
| `SessionStart` (startup/resume) | `memo briefing` | sync | 5 s | Session-briefing panel: open loops, memory of the day, last session |
| `UserPromptSubmit` | `memo recall-hook` | sync | 8 s | Queries the recall daemon (fast path) or falls back to BM25 when cold |
| `Stop` | `memo capture-stop` | async | 30 s | Extracts insights from the finished exchange via helper LLM |
| `Stop` | `memo session checkpoint` | async | 5 s | Snapshots session state for crash recovery |

**Note:** idle capture and other hooks require Claude Code's hook system
(`hooks/hooks.json`). Other agents (OpenCode, Windsurf, …) using MCP only get
the `memo_*` tools — add a similar idle trigger via the agent's native hook
system or use `memo session idle-maintenance --mode capture`. All hooks run 100%
local; your prompts never leave the machine.

### Recall daemon

The recall daemon is the hot-path optimization that makes ambient recall feel
instant. Without it, each `UserPromptSubmit` spawns a fresh Python process that
re-imports MLX from disk (~1–2 s even when cached). With it, a single long-lived
process keeps the embedder in RAM and answers socket requests in **<200 ms**.

```
SessionStart
  └─ memo recall-daemon start (async)
       └─ loads Memory + embedder
       └─ listens on ~/.local/share/memo/recall.sock

UserPromptSubmit
  └─ memo recall-hook
       ├─ daemon running? → socket request → <200 ms → additionalContext
       └─ daemon not ready? → BM25 fallback → ~100 ms → additionalContext
```

```bash
memo recall-daemon start    # start in background (also auto-started by the hook)
memo recall-daemon stop     # send SIGTERM + cleanup
memo recall-daemon status   # pid, socket path, warm/cold state
```

Logs: `~/Library/Logs/memo/recall-daemon.log`. The daemon restarts on the next
session start if macOS killed it under memory pressure.

### Recall tuning

| Env var | Default | Purpose |
|---|---|---|
| `MEMO_RECALL_DISABLE` | unset | Set to `1` to skip recall entirely |
| `MEMO_RECALL_TOP_K` | `3` | Max memorias to inject |
| `MEMO_RECALL_MIN_SIM` | `0.6` | Cosine similarity floor |
| `MEMO_RECALL_MIN_PROMPT_CHARS` | `12` | Skip very short prompts |
| `MEMO_RECALL_BODY_CHARS` | `240` | Snippet length per memoria |
| `MEMO_RECALL_SKIP_SLASH` | `1` | Skip recall on `/` prompts |
| `MEMO_RECALL_TOKEN_BUDGET` | `0` | When > 0, pack memorias greedily until ~N tokens; truncate tail to fit |
| `MEMO_RECALL_PROJECT_BOOST` | `0.15` | Additive score boost for memorias whose tags match the current project tag |
| `MEMO_RECALL_MIN_BODY_CHARS` | `40` | Filter out stub memorias (empty or near-empty bodies) |
| `MEMO_RECALL_FORCE_MODE` | unset | Set to `1` to disable the warm-signal cold-start check |
| `MEMO_RECALL_DEBUG` | unset | Print failure reasons to stderr |

The `MIN_SIM=0.6` floor is empirically tuned: on a 223-doc corpus, a relevant
query returns hits at 0.71–0.74 while an off-topic query ("how to bake apple
pie") returns 0 hits at 0.6 (3 noise hits at 0.51–0.56 cut by the floor). Tune
lower (0.5) on sparse corpora, higher (0.7) for high-precision only.

### Capture tuning

| Env var | Default | Purpose |
|---|---|---|
| `MEMO_CAPTURE_CONTEXT_TURNS` | `3` | Recent exchanges fed to the helper LLM (catches multi-turn decisions) |
| `MEMO_CAPTURE_COOLDOWN_MIN` | `0` | Min minutes between captures in the same session |
| `MEMO_CAPTURE_MIN_WORDS` | `15` | Minimum word count for an extracted insight (0 disables) |
| `MEMO_CAPTURE_DEBUG` | unset | Print extraction results to stderr |

The capture pipeline applies a **quality gate** before saving: insights are
discarded if too short (< `MEMO_CAPTURE_MIN_WORDS`) or if they start with
session-narrative openers like "the user…", "we discussed…", "i helped…". Only
specific, durable knowledge passes through, which keeps recall precision high
over time.

### Hook observability — `memo hook-log`

Every `recall-hook` invocation is appended to a JSONL ring buffer at
`~/.local/share/memo/recall.log` (auto-rotated at ~200 KB):

```
2026-05-16 14:32:01  vec     daemon   3 hits   187 ms   "how can we improve all of this?"
2026-05-16 14:31:44  bm25    subproc  1 hit    94 ms    "resolve todo"
2026-05-16 14:28:12  vec     daemon   0 hits   203 ms   "what does prewarm do"
```

Each row shows timestamp · search mode (`vec`/`bm25`) · path
(`daemon`/`subprocess`) · hit count · latency · prompt snippet.

```bash
memo hook-log              # last 20 entries
memo hook-log --limit 100
memo hook-log --follow     # stream live (Ctrl+C to stop)
```

### Backfill from past Claude Code conversations

`memo mine-history` walks `~/.claude/projects/<hash>/*.jsonl`, runs the same
prefilter + helper-LLM extract + embedding-dedup pipeline as the live capture
hook, and saves what's new (resumable per file):

```bash
memo mine-history --since 30 --limit 20     # last 30 days, 20 newest sessions
memo mine-history --dry-run --debug         # cost estimation, no writes
```

### Auto-reindex on edit

Editing a memoria directly in Obsidian normally needs a manual `memo reindex`.
`memo watch` (foreground) or `memo install-watcher` (background launchd job)
debounces FS events and runs `Memory.reindex()` automatically. Logs land in
`~/Library/Logs/memo/`.

### Project-scoped recall

`memo save` auto-attaches a `project:<repo>` tag derived from the git toplevel of
your cwd (or `MEMO_PROJECT_TAG`). The recall hook reads `cwd` from the hook
payload and boosts memorias whose tags match by `MEMO_RECALL_PROJECT_BOOST`
(default `0.15`). Opt out per-call with `memo save --no-project-tag`; disable
globally with `MEMO_AUTO_PROJECT_TAG=0`.

---

## Surfaces

### Session briefing — `memo briefing`

`memo briefing` is the `SessionStart` hook entrypoint. Every new session it
emits an `additionalContext` panel with three blocks:

1. **Last session in this project** — summary of the most recent session in the
   current `cwd`, with a one-line `claude --resume <session_id>` for instant
   crash recovery.
2. **Open loops** — the N memories most recently updated (default 7-day window),
   numbered for interactive selection. Say *"give me loop 2"* and the agent
   retrieves it.
3. **Memory of the day** — one memory picked deterministically by a SHA-256 hash
   of today's date, biased toward the least-recently-touched entries so the
   corpus rotates over time.

```markdown
## Briefing

**Last session in this project** (12m ago): reviewing the project…
`claude --resume be72126f-3bcb-4faa-9a0f-dd97b8caa296`

### Open loops (last 7 days)

1. `91fc486c` **note** · memo diff as a real change surface — today [memory, versioning]
2. `5da4cdc1` **note** · Smarter recall hook — today [memory, recall]
…

### Memory of the day
`064031dd` **fact** · sqlite-vec L2 normalisation invariant — 3 days ago
> The embedder must L2-normalise before storing…

_Continue with: `give me loop N` · `/memo get <id>` · `/memo ask <question>`_
```

| Env var | Default | Purpose |
|---|---|---|
| `MEMO_BRIEFING_DISABLE` | unset | Set to `1` to skip the panel |
| `MEMO_BRIEFING_LOOPS_N` | `5` | Number of open loops to show |
| `MEMO_BRIEFING_LOOPS_DAYS` | `7` | Recency window for open loops |
| `MEMO_BRIEFING_DEBUG` | unset | Print failures to stderr |

### Semantic map — `memo mapa`

`memo mapa` reads all embeddings in `memvec.db`, projects them to 2D via **UMAP**
(if `umap-learn` is installed) or **PCA** (numpy fallback), and renders a
self-contained interactive HTML file.

```bash
memo mapa                                      # generate + open in browser
memo mapa --output ~/Desktop/mapa.html --no-open
memo mapa --limit 200                          # most recent 200
memo mapa --no-animate                         # skip the timeline animation
```

The HTML colours points by type, shows title/tags/date on hover, opens full
metadata on click, supports a search filter, and animates corpus growth over
time. For better cluster topology on 50+ entries, install `umap-learn`
(`pipx runpip mlx-memo install umap-learn`); without it, PCA is used.

### Time-machine

memo is the only agent-memory product that lets you rewind the corpus to any
past date. `history.db` is an append-only audit log of every save/update/delete;
a snapshot at any `T` is rebuilt by replaying events in reverse from "now". See
[time-machine.svg](time-machine.svg) for the algorithm.

```bash
memo as-of ask "MLX vs Ollama" --date 2026-02-01   # what did I think 3 months ago?
memo diff --from 2026-03-01 --to 2026-04-30        # what changed between releases?
memo as-of search "auth middleware" --date 2026-03-15
memo as-of list --date 2026-03-01                  # memorias that existed then
```

Use cases: debugging agent regressions, reproducible AI behaviour (serve a past
snapshot as an alternate MCP), personal audit, and compliance ("what did the
model know when it took action X?").

---

## CLI reference

```bash
# ── Core CRUD ──────────────────────────────────────────────────────────────
memo save 'body markdown' --title 'X' -t mlx -t local
memo search 'query' --limit 5
memo list --limit 20 --type decision
memo get <id>
memo update <id> --title 'X2' -t mlx -t local --type decision
memo update <id> --content -      # read replacement body from stdin
memo delete <id> --yes
memo reindex                      # absorb edits made directly in Obsidian
memo stats
memo ask 'what changed in the embedder this month?'

# ── History & audit ────────────────────────────────────────────────────────
memo historia <id>                # chronological audit trail for one record with field diffs
memo history                      # recent save/update/delete events across all records

# ── Ambient memory commands (also run by hooks) ────────────────────────────
memo briefing                     # preview the SessionStart panel in the terminal
memo recall-hook                  # UserPromptSubmit hook (reads JSON from stdin)
memo prewarm                      # pre-load MLX models (SessionStart hook)
memo capture-stop                 # extract insights from last exchange (Stop hook)
memo session checkpoint           # snapshot current session state (Stop hook)
memo session recent --limit 5     # list recent sessions

# ── Semantic map ───────────────────────────────────────────────────────────
memo mapa                         # generate + open in browser (UMAP or PCA → Plotly HTML)
memo mapa --output ~/Desktop/mapa.html --no-open
memo mapa --limit 200 --no-animate

# ── Setup & maintenance ────────────────────────────────────────────────────
memo doctor                       # self-check
memo doctor --gc                  # report orphans (store ↔ disk)
memo doctor --gc --fix            # drop orphan store rows (.md never auto-deleted)
memo install-slash                # configure Claude Code, Codex, Windsurf, Devin
memo mcp-command --client windsurf # print Windsurf mcp_config.json block
memo init                         # re-run first-run picker
memo migrate-vault <new-path>     # move memorias to a different folder
memo backup --out memo.zip        # backup .md files + index

# ── Time-machine ───────────────────────────────────────────────────────────
memo as-of search 'query' --date 2026-03-01
memo as-of ask 'question' --date 2026-03-01
memo as-of list --date 2026-03-01
memo diff --from 2026-03-01 --to 2026-04-30

# ── Knowledge graph ────────────────────────────────────────────────────────
memo entities                     # top entities across the corpus
memo entity <name>                # memorias that mention a specific entity
memo extract-entities --all       # populate the entity graph (Qwen 3B, batch)
memo consolidate                  # cluster near-duplicates + merge proposals

# ── Backfill & watching ────────────────────────────────────────────────────
memo mine-history --since 30      # backfill memorias from past Claude Code chats
memo watch                        # foreground file-watcher: auto-reindex on .md edit
memo install-watcher              # background watcher via launchd plist
memo uninstall-watcher            # remove the launchd watcher job

# ── Recall daemon ──────────────────────────────────────────────────────────
memo recall-daemon start          # start the persistent recall daemon
memo recall-daemon stop
memo recall-daemon status

# ── Observability ──────────────────────────────────────────────────────────
memo hook-log                     # last 20 recall-hook entries: mode, via, hits, latency
memo hook-log --limit 50
memo hook-log --follow            # stream new entries as they arrive

# ── Updates ────────────────────────────────────────────────────────────────
memo self-update                  # upgrade via pipx/uv + re-warm models
memo self-update --check          # check PyPI for a newer version without installing

# ── Live dashboard ─────────────────────────────────────────────────────────
memo tui                          # live terminal dashboard (Ctrl+C exits)
```

### Live dashboard — `memo tui`

![memo tui dashboard](tui-dashboard.png)

Six panels, refresh every second: **corpus** (totals, project tags, top types),
**runtime** (MLX warm/cold flags, vault size, watcher state), **recent saves**,
**recent recalls** (mode + path per row, live daemon status), **top tags**, and
**activity** (14-day saves/recalls sparklines). It reads read-only from
`history.db`, the JSONL recall log, the daemon PID file, and the warm-signal
file. Quit with `q`, `ESC`, or `Ctrl+C`.

### Updating — `memo self-update`

`memo self-update` detects the active install method (checks `pipx list` then
`uv tool list`), runs the appropriate upgrade, and re-warms models with
`memo prewarm --download-all`. `memo self-update --check` compares installed vs
latest PyPI without installing.

---

## Configuration

All env vars are optional; defaults aim at a fresh Apple Silicon Mac. On first
run in an interactive shell, an arrow-key picker asks where memorias should live
and persists the choice to `~/.config/memo/config.toml`. Re-run it with
`memo init`. Hooks get `MEMO_NONINTERACTIVE=1` so they never trigger the picker.

Resolution precedence (highest first): explicit kwargs → `MEMO_*` env vars →
`~/.config/memo/config.toml` → legacy `MEMO_VAULT_PATH` + `MEMO_MEMORY_SUBDIR` →
hardcoded defaults.

**Storage & paths**

| Env var | Default | What |
|---|---|---|
| `MEMO_DATA_DIR` | `~/Documents/memo` | Where memoria `.md` files live |
| `MEMO_VAULT_PATH` | `(unset)` | Optional Obsidian vault for `memo ingest` |
| `MEMO_STATE_DIR` | `~/.local/share/memo` | sqlite-vec DB + state |
| `MEMO_CONFIG_FILE` | `~/.config/memo/config.toml` | Override config-file path |
| `MEMO_NONINTERACTIVE` | unset | Set to `1` in hooks to skip the first-run picker |

**Models**

| Env var | Default | What |
|---|---|---|
| `MEMO_MODEL_PROFILE` | `balanced` | Model bundle: `light`, `balanced`, or `quality` |
| `MEMO_LLM_MODEL` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Chat tier |
| `MEMO_HELPER_MODEL` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Helper tier |
| `MEMO_EMBEDDER_MODEL` | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | Embedder |
| `MEMO_EMBEDDER_DIMS` | `1024` | Embedding dim — must match the embedder |
| `MEMO_RERANKER_ENABLED` | `1` in `balanced`/`quality` | Enable cross-encoder rerank for hybrid search |
| `MEMO_RERANKER_MODEL` | `mku64/Qwen3-Reranker-0.6B-mlx-8Bit` | MLX reranker model |
| `MEMO_RERANK_INPUT_K` | `5` | Hybrid candidates sent to the reranker |
| `MEMO_RERANK_FUSION_ALPHA` | `0.7` | Weight of reranker score vs RRF position bonus |

**Search**

| Env var | Default | What |
|---|---|---|
| `MEMO_MAX_CONTENT_CHARS` | `64000` | Truncate body before embed |
| `MEMO_SEARCH_DEFAULT_LIMIT` | `10` | Default `--limit` for search |
| `MEMO_SEARCH_DECAY_HALFLIFE` | `0` | When > 0, blend recency into scores. Half-life in days (`exp(-days/N)`) |
| `MEMO_SEARCH_DECAY_ALPHA` | `0.15` | Weight of decay signal vs raw similarity |

**Tagging**

| Env var | Default | What |
|---|---|---|
| `MEMO_AUTO_PROJECT_TAG` | `1` | Auto-add `project:<repo>` tag from git toplevel on save |
| `MEMO_PROJECT_TAG` | unset | Explicit project tag (overrides git-toplevel detection) |

Recall, capture, and briefing knobs are listed under
[Ambient memory](#ambient-memory).

**Model profiles**

- `light`: 0.6B embedder, Qwen2.5 chat/helper, no reranker. Best for low-latency hooks.
- `balanced`: 0.6B embedder + 0.6B reranker + Qwen2.5 chat/helper. Default for most users.
- `quality`: 4B embedder (2560 dims) + 0.6B reranker + Qwen3 4B chat. Requires
  `rm ~/.local/share/memo/memvec.db && memo reindex` when switching from
  1024-dim profiles.

If models are still downloading, you can save without MLX and keep keyword
search available:

```bash
memo save "text to remember" --title "Short title" --defer-embed
memo search "text" --mode bm25
memo reindex     # later, once the embedder is cached
```

### Upgrading the embedder

The default 0.6B is fast (~50 ms/embed) and small (~600 MB) but recall on diffuse
queries can be noisy. For the 200–2000 memorias range, swap to a larger variant.

| Model | Dims | Disk | Recall | Per-embed |
|---|---|---|---|---|
| [`Qwen3-Embedding-0.6B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) *(default)* | 1024 | ~600 MB | OK | ~50 ms |
| [`Qwen3-Embedding-4B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ) | 2560 | ~3 GB | better | ~200 ms |
| [`Qwen3-Embedding-8B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-8B-4bit-DWQ) | 4096 | ~5 GB | best | ~400 ms |

```bash
hf download mlx-community/Qwen3-Embedding-4B-4bit-DWQ   # 1) pre-download
export MEMO_MODEL_PROFILE=quality                       # 2) point memo at it
memo backup --out memo-pre-4b.zip                       # 3) backup before re-embed
rm ~/.local/share/memo/memvec.db && memo reindex        # 4) wipe + rebuild
memo doctor --strict-runtime
```

The dim mismatch is a hard error — `MEMO_EMBEDDER_DIMS` must match the new
model's hidden size, and `memo doctor` validates it at load.

---

## Design and comparison

### Design notes

- **One sqlite file, no Qdrant.** `sqlite-vec` outperforms a small Qdrant
  snapshot at the corpus size memo targets (a few thousand entries, single
  writer). One file makes reset trivial: `rm memvec.db`.
- **Embed `title + body` together.** Titles carry the highest-density retrieval
  signal; prepending also protects the title from head-truncation on long
  bodies. Pure retag/type changes skip the embedder.
- **`.md` is the storage of record.** Edit in Obsidian; the next `memo reindex`
  picks it up via `body_hash` mismatch.
- **Head-truncate long inputs + append EOS.** The embedder caps at 512 tokens; we
  head-truncate and explicitly append `<|im_end|>` so Qwen3-Embedding's
  last-token pool lands on the EOS hidden state it was fine-tuned for.
- **Asymmetric retrieval.** Queries get an `Instruct: …\nQuery: …` prefix;
  documents go raw. Without the prefix, cosine collapses toward 0.
- **Cosine distance metric.** The vec0 schema declares `distance_metric=cosine`,
  so `score = 1 − distance` is interpretable in [0, 1].
- **No Ollama dependency, anywhere.** `pyproject.toml` doesn't declare it;
  `doctor` doesn't probe `:11434`.

### How memo compares

memo's neighbours diverge on the things that matter day-to-day: where the model
runs, where the data lives, how recall is wired, and whether you can read your
own memory in plain text.

| | **memo** | [`mem0`](https://github.com/mem0ai/mem0) | [`letta`](https://github.com/letta-ai/letta) | [`cognee`](https://github.com/topoteretes/cognee) | [`supermemory`](https://github.com/supermemoryai/supermemory) | MCP [`memory` ref](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) |
|---|---|---|---|---|---|---|
| **Runtime** | MLX, in-process | Cloud API or Ollama | Postgres + LLM API | Cloud or Ollama | Cloud SaaS | Node, in-process |
| **Network in hot path** | **0** | yes or `:11434` | yes (LLM API) | yes (LLM API) | always | yes (LLM API) |
| **Vector store** | sqlite-vec (one file) | Qdrant / pgvector | Postgres + pgvector | LanceDB / Qdrant | hosted | in-memory JSON |
| **External daemons** | **none** (recall daemon optional) | Ollama + Qdrant | Postgres | Postgres / vector DB | none (SaaS) | none |
| **Storage of record** | **markdown files** | DB blob | DB rows | DB rows + graph | hosted DB | JSON entity graph |
| **Human-readable / editable** | ✅ Obsidian/vim | ❌ | ❌ | ❌ | ❌ | partial (JSON) |
| **Hybrid retrieval + reranker** | ✅ vec + BM25 + RRF + cross-encoder | vec | vec | vec + graph | vec | entity-based |
| **Ambient recall (zero invoke)** | ✅ hooks + daemon (<200 ms) | ❌ | n/a | ❌ | ❌ | ❌ |
| **Time-machine (past snapshots)** | ✅ `memo as-of …` | ❌ | ❌ | ❌ | ❌ | ❌ |
| **License** | MIT | Apache-2.0 | Apache-2.0 | Apache-2.0 | proprietary | MIT |

> Projects move fast — cells reflect the public state of each repo at the time of
> writing. PR a correction if any is stale.

**The differentiators in plain terms:**

1. **Time-machine** — every other store serves *current* state only. memo rebuilds
   any past corpus state from its audit log. No competitor can retrofit this
   without an audit log they don't have.
2. **100% local hot path, no Ollama** — LLM, embedder, and reranker run in-process
   via MLX. No `:11434` round-trip, no Docker, no provider key.
3. **Markdown is the storage of record** — plain `.md` you can edit, sync, and
   `grep`; the sqlite index is rebuildable.
4. **Ambient recall + session awareness** as a turnkey hook bundle — the agent
   sees the right memorias before it answers, and the corpus grows on its own.
5. **MCP is a primary interface** — same stdio contract for every client on day
   one, with a deliberately tiny default tool surface.

**When *not* to pick memo:** you're not on Apple Silicon (MLX is load-bearing);
you need a hosted multi-tenant service (`supermemory`/`mem0` cloud); you want an
explicit core/archival agent runtime (`letta`); or you want a knowledge-graph +
ontology layer (`cognee`).

### Experimental modules

These ship in the package but are **not** covered by CI, not exposed via MCP
tools, and may change without notice. They stay inside memo's pillar — local
semantic storage, retrieval, and corpus-level utilities; coordination,
federation, and orchestration belong outside memo's surface.

| Module | What it does |
|---|---|
| `multimodal` | Cross-modal semantic search over images, audio, and text |
| `collaborative` | Shared knowledge graph across multiple users |
| `sharing` | Per-memoria sharing links and permission grants |
| `encryption` | AES-256-GCM file-level primitives (gated OFF; `MEMO_ENCRYPTION_ENABLED=1`) |
| `contradict` | Contradiction and staleness radar with triage workflow |
| `chunker` | Heading-aware sub-document chunking for long memories |
| `crossref` | Obsidian `[[wikilink]]` backlink index and multi-hop traversal |
| `contextual` | Conversation-history-aware recall boosting |
| `navigation` | BFS path finding and community detection on the entity graph |
| `sync` | Multi-device sync and compressed backups |
| `versioning` | Per-memoria version history and unified-diff rollback |

The current inventory of broader corpus/workflow experiments lives in
[`src/memo/experimental_index.md`](../src/memo/experimental_index.md).

### Consciousness-stack integration

memo is one of three sovereign systems: **Synapse** (federator), **[Memflow](https://github.com/jagoff/memflow)**
(cross-Mac operational continuity), and memo (semantic corpus). Integration is
opt-in everywhere — single-Mac users without Synapse or Memflow see zero
behaviour change.

| Surface | Doc | Default | Opt-in knob |
|---|---|---|---|
| Synapse adapter — `MemoSynapseBackend`, provenance, freeze-write | [synapse-adapter.md](synapse-adapter.md) | OFF | `MEMO_RESPECT_SYNAPSE_FREEZE=1` |
| Embedder daemon — shared MLX sidecar protocol | [embedder-daemon.md](embedder-daemon.md) | ON via `SessionStart` | — |
| Contradict loop — synapse pulls via `memo contradict list --json` | [contradict-loop.md](contradict-loop.md) | ON (synapse-side) | — |
| Receipts — operational breadcrumbs to memflow | [receipts.md](receipts.md) | OFF | `MEMO_EMIT_RECEIPTS=1` |
| Briefing — synapse `present_state` in the session panel | [briefing.md](briefing.md) | ON when `synapse` is on PATH | `MEMO_BRIEFING_SYNAPSE_DISABLE=1` |
