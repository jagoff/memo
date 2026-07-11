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

## Third-party sharing

memo shares your data with **no third parties**. There is no server operated by
the author that receives, stores, or processes your content.

## Retention & deletion

You control retention entirely. Delete a memory's markdown file (or run
`memo delete`) to remove it; the index re-derives from disk. Uninstalling memo
and deleting its `data_dir` / state directory removes all stored data.

## Contact

Questions or concerns: open an issue at
<https://github.com/jagoff/memo/issues> or email fernandoferrari@gmail.com.
