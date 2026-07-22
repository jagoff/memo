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

**Nothing, by default.** Embeddings and any LLM steps run **in-process** — MLX on
Apple Silicon, CPU `sentence-transformers` elsewhere. memo makes **no calls to any
cloud API**, uses **no API keys**, and collects **no telemetry or analytics**.

The **only** way memory leaves your machine is if **you** explicitly configure a
git `memo-sync` remote. In that case your memory markdown is pushed to **that
remote, which you own and control** — memo neither hosts nor has access to it.

### Optional update-check heartbeat (off by default)

memo can check for a newer release on startup. By default that check uses
`git ls-remote` against the public GitHub repo and sends **nothing** about you.

If **you** set `MEMO_UPDATE_ENDPOINT` (empty by default) *and* enable update
checks (`MEMO_UPDATE_CHECK_ENABLED` or `MEMO_AUTO_UPDATE`, both off by default),
the check instead GETs that endpoint, which lets the operator count active
installs. The **entire** payload is three anonymous fields:

- `id` — `sha256(device_id)[:16]`, a one-way hash computed on your machine; the
  raw device id never leaves it and the hash is not reversible to identity.
- `v` — your memo version. `os` — your OS name (e.g. `Darwin`).

No memory content, file paths, queries, or IP are collected. This is **opt-in**:
leave `MEMO_UPDATE_ENDPOINT` unset and memo never contacts any such endpoint.

The explicit `memo map` and `memo dashboard` browser views use Canvas/SVG
renderers contained in their generated HTML. They make no third-party request
and remain usable offline. The local renderer keeps filtering, hover, click
details, timelines, and bar charts; it intentionally omits advanced
pan/zoom/export controls to avoid a large browser dependency.

## Third-party sharing

memo shares your data with **no third parties**. No server operated by the author
receives, stores, or processes your memory **content** — ever. The only optional
exception is the opt-in update-check heartbeat above, which sends a hashed
install id, version, and OS name (never content) and only when you configure it.

## Retention & deletion

You control retention entirely. Delete a memory's markdown file (or run
`memo delete`) to remove it; the index re-derives from disk. Uninstalling memo
and deleting its `data_dir` / state directory removes all stored data.

## Contact

Questions or concerns: open an issue at
<https://github.com/jagoff/memo/issues> or email fernandoferrari@gmail.com.
