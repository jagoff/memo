# Privacy and Network Policy

memo stores memories, embeddings, caches, and history locally. Normal
`memo-mcp` startup is offline by default: it makes no outbound network request
and does not modify Claude, Codex, or other agent configuration.

## Startup opt-ins

All four startup behaviors default to `0`/false:

- `MEMO_UPDATE_CHECK_ENABLED=1` permits a throttled remote tag check and records
  a local update notification.
- `MEMO_AUTO_UPDATE=1` permits that check and may install a newer tagged memo
  release in the background for the next startup.
- `MEMO_STATUSLINE_SELFHEAL=1` permits memo to repair its statusline entry in
  local Claude settings.
- `MEMO_HOOK_SELFHEAL=1` permits memo to repair its local recall-hook entries.

The two self-heal operations modify local configuration but do not require the
network. Opted-in background errors never prevent the MCP server from starting.

## Explicit network operations

The offline default does not claim that every command is offline. Network access
is expected when the user explicitly requests:

- `memo update` or installation from a remote package/git source;
- `memo sync` against a configured git remote;
- model downloads during installation or the first use of a model not already
  present in the local Hugging Face cache; or
- benchmark downloads for evaluation commands that use external corpora.

Local search, recall, save, history, briefing, and normal MCP startup do not send
prompts or memories to a hosted memo service. A separately configured model,
sync remote, HTTP integration, or external helper has its own privacy boundary.

Encrypted credential storage is a separate, explicit opt-in
(`MEMO_SECRET_STORAGE_ENABLED=1`). Secret values never become searchable memory
or markdown and memo's git sync excludes legacy `secrets/` markers. The local
master key and ciphertext database are permission-restricted, but a process
running as your OS user can still request plaintext; protect access to your
account and use `memo secret get/export` only when disclosure is intended.

## Optional verbatim transcript index

Total Recall is off by default. `MEMO_VERBATIM_INDEX=1` enables its nightly
index pass; `memo verbatim index` is the explicit one-shot equivalent. It reads
local Claude transcript JSONL, keeps only timestamped user/assistant turns,
applies memo's known-secret redactor, and writes a lexical FTS5 sidecar under
`MEMO_STATE_DIR`. It uses no embeddings, never becomes Markdown, never enters
ambient recall, and is not part of git sync.

The default retention/backfill window is 90 days
(`MEMO_VERBATIM_MAX_DAYS=90`). Rows without a valid timestamp are rejected so
they cannot bypass pruning. The state directory is restricted to mode `0700`
and the database, SQLite sidecars, and watermark to `0600`; CLI and MCP searches
are explicitly invoked and capped at 100 results.

Known-secret redaction is defense in depth, not a guarantee that arbitrary
sensitive prose can be recognized. Enabling this feature creates a second local
copy of transcript text that any process running as your OS user can query.
Leave it disabled if that duplication is not acceptable, and delete
`MEMO_STATE_DIR/verbatim.db` plus `verbatim-index.json` to remove the derived
index.
