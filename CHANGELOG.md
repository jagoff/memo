# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-07

First public release. Distribution name on PyPI is `memo-mcp`
(`memo` was already taken).

### Added

- Public PyPI distribution as [`memo-mcp`](https://pypi.org/project/memo-mcp/).
- Claude Code plugin format (`.claude-plugin/plugin.json`) — single-step install
  via `/plugin install memo@jagoff/memo`.
- `.mcp.json` bundled in repo root so MCP-aware clients can auto-register.
- `skills/memo/SKILL.md` bundled in repo — slash-command UX layer for Claude
  Code CLI users (optional).
- LICENSE file (MIT).
- README polish: install, quickstart, architecture diagram, comparison vs
  `mem-vault` / `mem0` / `engram`.
- Migration script `scripts/migrate-from-mem-vault.py` (already shipped in
  0.1.0 development; documented in README for this release).

### Changed

- Bumped Development Status from `3 - Alpha` to `4 - Beta` in pyproject classifiers.
- Author name expanded from "Fer" to "Fernando Ferrari" for PyPI metadata clarity.

## [0.1.0] - 2026-04-28

Initial development release (private — not on PyPI).

### Added

- MLX-native memory MCP for Apple Silicon. Stack: `mlx-lm` + `mlx`
  (Qwen2.5-7B/3B-Instruct-4bit + Qwen3-Embedding-0.6B-4bit-DWQ),
  `sqlite-vec` for vectors, markdown files in Obsidian vault for storage.
- Tools exposed: `memory_save`, `memory_search`, `memory_list`, `memory_get`,
  `memory_update`, `memory_delete`, `memory_stats`, `memory_reindex`,
  `memory_ask` (RAG over memorias with inline citations), `memory_consolidate`
  (cluster + LLM merge proposals), graph queries (entity extraction).
- CLI (`memo`) with subcommands matching MCP tools.
- MCP server entry point (`memo-mcp`) using FastMCP framework.
- History tracking via SQLite for memory edits + accesses.
