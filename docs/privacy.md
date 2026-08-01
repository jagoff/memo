# Privacy and Network Policy

memo stores memories, embeddings, caches, and history locally. Every memory
operation — search, recall, save, history, briefing — is offline by default and
sends nothing to a hosted service.

Normal `memo-mcp` startup is fully offline. Remote update checks and background
auto-update are both opt-in. When `MEMO_AUTO_UPDATE=1`, startup makes a
throttled `git ls-remote` to the memo repo to see whether a newer tagged release
exists and, if so, installs it in the background for the next start. It sends no
memory content or paths—only the normal network metadata of a git request.
Startup also does not modify Claude, Codex, or other agent configuration unless
a self-heal opt-in below is enabled.

## Startup behaviors

All startup behaviors below default to `0`/false:

- `MEMO_UPDATE_CHECK_ENABLED=1` permits a throttled remote tag check and records
  a local update notification (this is the check-only half of auto-update; it is
  implied when `MEMO_AUTO_UPDATE` is on).
- `MEMO_AUTO_UPDATE=1` opts in to the throttled remote tag check and background
  install. Leaving it unset performs neither operation.
- `MEMO_UPDATE_ENDPOINT=<url>` (empty by default) routes the tag check above
  through an HTTP endpoint instead of `git ls-remote`. It sends three anonymous
  fields — `id=sha256(device_id)[:16]` (hashed on-device, raw id never sent),
  `v` (version), `os` (OS name) — letting the operator count active installs. No
  memory content, paths, or IP. Only fires when a tag check is already enabled;
  unset → no such request. Falls back to `git ls-remote` on any HTTP failure.
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

`memo map` and `memo dashboard` are offline browser views. Both use small
Canvas/SVG renderers bundled in memo's generated HTML and make no third-party
request. The local renderer keeps filtering, hover, click details, timelines,
and bar charts, but intentionally omits Plotly's advanced pan/zoom/export
controls.

Local search, recall, save, history, briefing, and normal MCP startup do not send
prompts or memories to a hosted memo service. A separately configured model,
sync remote, HTTP integration, or external helper has its own privacy boundary.

Encrypted credential storage is a separate, explicit opt-in
(`MEMO_SECRET_STORAGE_ENABLED=1`). Secret values never become searchable memory
or markdown and memo's git sync excludes legacy `secrets/` markers. The local
master key and ciphertext database are permission-restricted, but a process
running as your OS user can still request plaintext; protect access to your
account and use `memo secret get/export` only when disclosure is intended.

## Mandatory memory persistence boundary

Every normal memory write and update strips `<private>...</private>` spans and
masks known AWS, GitHub, OpenAI, Anthropic, Slack, GCP, and PEM private-key
patterns before title derivation, identity hashing, embedding, Markdown,
SQLite/FTS, history, receipts, or logs. Reindex applies the same sanitizer to
the derived searchable representation without rewriting hand-edited Markdown.
Long high-entropy token detection remains opt-in with
`MEMO_REDACT_ENTROPY=1` because it has a higher false-positive rate; pure hex
hashes and IDs are exempt.

`MEMO_REDACT_SECRETS=0` and `MEMO_PRIVATE_MARKERS=0` now disable only the early
capture/ingest preprocessing passes. They cannot disable the final persistence
boundary. Known-pattern masking is defense in depth, not recognition of every
form of sensitive prose; use `<private>` markers for prose that must not become
memory.

Memory identity is namespaced as `project:<slug>`, `_global` for explicit
global saves, or `_unscoped` when project detection was requested but produced
no project. Repeated exact evidence corroborates one canonical record; a
same-topic changed body versions that record. Save responses add ephemeral
`action` and `index_pending` fields. The legacy `normalized_hash` remains
unchanged for compatibility, while schema v5 derives a separate normalized
content hash in the rebuildable SQLite index. Migration never rewrites
Markdown; `memo reindex --rebuild` repairs derived state while preserving user
signals.

`memo doctor --db` reports identity collisions, legacy/ambiguous rows, and
Markdown files containing known secret patterns or private markers. It returns
counts only, never excerpts or matched values, and performs no repair or merge.
Relation/review/installer preflight is intentionally outside this phase.

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
