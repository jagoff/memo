# Privacy Policy

_Last updated: 2026-07-11_

**memo** (`mlx-memo` on PyPI) is a **local-first** memory server for AI agents.
It is designed so that your data never leaves your machine.

## What data memo handles

memo stores the memories you (or your AI agent) explicitly save — decisions,
facts, preferences, notes — plus a derived search index and local usage logs
(which memories were recalled and whether they were useful). It also reads the
conversation turns your agent passes to it in order to capture durable insights.

## Where it is stored

All data is stored **locally on your own device**:

- **Markdown files** — the canonical source of truth, in your configured
  `data_dir` or your Obsidian vault. Human-readable, yours to edit or delete.
- **A rebuildable SQLite index** (`memvec.db`) and local log files
  (recall / grounding logs) in your state directory.

## What is sent off your device

Memory operations themselves are offline. Embeddings and any LLM steps run
**in-process** — MLX on Apple Silicon, CPU `sentence-transformers` elsewhere.
memo makes **no calls to a cloud model API**, uses **no provider API keys**, and
collects **no telemetry or analytics** about prompts or memory content.

The one default-on network exception is **auto-update**. On `memo-mcp` startup,
memo makes a throttled `git ls-remote` tag probe against the public GitHub
repository and installs a newer tagged release in the background when one is
available. The probe sends no memory content, queries, file paths, or memo
identity. Set `MEMO_AUTO_UPDATE=0` to opt out and keep startup fully offline.

Memory leaves your machine only if **you** explicitly configure a git
`memo-sync` remote. In that case your memory markdown is pushed to **that remote,
which you own and control** — memo neither hosts nor has access to it.

### Startup network and self-heal controls

- `MEMO_AUTO_UPDATE` defaults to `1`. Set it to `0` to disable the remote tag
  probe and background install.
- `MEMO_UPDATE_CHECK_ENABLED=1` enables the check-only path and records a local
  update notification. It is implied while auto-update is enabled.
- `MEMO_STATUSLINE_SELFHEAL=1` and `MEMO_HOOK_SELFHEAL=1` permit local repairs
  to Claude settings. Both are off by default and require no network.

If **you** set `MEMO_UPDATE_ENDPOINT` (empty by default) while an update check is
enabled, the check GETs that endpoint instead of using `git ls-remote`. This
lets the endpoint operator count active installs. The **entire** payload is
three anonymous fields:

- `id` — `sha256(device_id)[:16]`, a one-way hash computed on your machine; the
  raw device id never leaves it and the hash is not reversible to identity.
- `v` — your memo version. `os` — your OS name (e.g. `Darwin`).

No memory content, file paths, or queries are included. This endpoint path is
**opt-in**: leave `MEMO_UPDATE_ENDPOINT` unset and memo uses only the public git
tag probe described above.

Explicit commands may also use the network when you request `memo update`,
`memo sync` against a configured remote, model downloads, or benchmark
downloads.

The explicit `memo map` and `memo dashboard` browser views use Canvas/SVG
renderers contained in their generated HTML. They make no third-party request
and remain usable offline. The local renderer keeps filtering, hover, click
details, timelines, and bar charts; it intentionally omits advanced
pan/zoom/export controls to avoid a large browser dependency.

## Third-party sharing

memo shares your memory data with **no third parties**. No server operated by
the author receives, stores, or processes your memory **content** — ever. The
only optional author-operated endpoint is the update-check heartbeat above,
which sends a hashed install id, version, and OS name (never content) only when
you configure it.

## Retention & deletion

You control retention entirely. Delete a memory's markdown file (or run
`memo delete`) to remove it; the index re-derives from disk. Uninstalling memo
and deleting its `data_dir` / state directory removes all stored data.

## Contact

Questions or concerns: open an issue at
<https://github.com/jagoff/memo/issues> or email fernandoferrari@gmail.com.
