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

**Nothing, unless you opt in.** Embeddings and any LLM steps run **in-process** —
MLX on Apple Silicon, CPU `sentence-transformers` elsewhere. memo makes **no calls
to any cloud API** for your data, uses **no API keys**, and never transmits memory
content, file paths, or queries.

The only way your memory **content** leaves your machine is if **you** explicitly
configure a git `memo-sync` remote. In that case your memory markdown is pushed to
**that remote, which you own and control** — memo neither hosts nor has access to it.

### Anonymous usage heartbeat (opt-in — you're asked once at setup)

To gauge how many people use memo, memo can send a tiny anonymous heartbeat on
startup (throttled to ~once every 6h). It is **off by default**: the first-run
setup (`memo init`) asks you a single yes/no question, and only a **yes** turns it
on. Nothing is sent unless you opt in. The **entire** payload is three anonymous
fields:

- `id` — `sha256(device_id)[:16]`, a one-way hash computed on your machine; the
  raw device id never leaves it and the hash is not reversible to identity.
- `v` — your memo version. `os` — your OS name (e.g. `Darwin`).

No memory content, file paths, queries, or IP are collected. Change your mind
anytime:

- `memo config set update.check_enabled false` — turn the heartbeat off.
- `memo config set update.check_enabled true` — turn it on later.

(If you enable update checks but want no heartbeat, set `MEMO_UPDATE_ENDPOINT=off`
— checks then use `git ls-remote` against public GitHub.)

The explicit `memo map` and `memo dashboard` browser views use Canvas/SVG
renderers contained in their generated HTML. They make no third-party request
and remain usable offline. The local renderer keeps filtering, hover, click
details, timelines, and bar charts; it intentionally omits advanced
pan/zoom/export controls to avoid a large browser dependency.

## Third-party sharing

memo shares your data with **no third parties**. No server operated by the author
receives, stores, or processes your memory **content** — ever. The only exception
is the anonymous usage heartbeat above (a hashed install id, version, and OS name —
never content), and only if you opted in when memo asked at setup.

## Retention & deletion

You control retention entirely. Delete a memory's markdown file (or run
`memo delete`) to remove it; the index re-derives from disk. Uninstalling memo
and deleting its `data_dir` / state directory removes all stored data.

## Contact

Questions or concerns: open an issue at
<https://github.com/jagoff/memo/issues> or email fernandoferrari@gmail.com.
