# Distribution

This project has two separate distribution contracts:

1. The `mlx-memo` package and `memo-mcp` server must be installable from the
   common package and MCP registries.
2. The exact `/memo` slash command must be advertised only for CLIs that can
   actually expose that command.

## Supported Stores

| Surface | Status | Install path | Repo assets |
| --- | --- | --- | --- |
| PyPI | Live package, release workflow added | `pipx install mlx-memo` or `uv tool install mlx-memo` | `pyproject.toml`, `.github/workflows/publish.yml` |
| Homebrew tap | Public tap exists | `brew tap jagoff/memo && brew install mlx-memo` | `docs/homebrew/mlx-memo.rb`, `jagoff/homebrew-memo` |
| Official MCP Registry | Manifest ready, release workflow added | Registry clients discover `io.github.jagoff/memo` | `server.json`, README `mcp-name`, `.github/workflows/publish.yml` |
| Claude Code marketplace | Repo marketplace ready | `claude plugin marketplace add jagoff/memo` then `claude plugin install memo@memo -s user` | `.claude-plugin/marketplace.json`, `.claude-plugin/plugin.json`, `commands/memo.md`, `skills/memo/SKILL.md`, `hooks/hooks.json` |
| Codex plugin marketplace + user skill | Repo marketplace ready for MCP metadata; installer also copies exact `memo` skill to `$CODEX_HOME/skills/memo/SKILL.md`. Codex CLI 0.130.0 does not expose custom skills in the TUI slash menu. | `memo install-slash --client codex`; plugin-only: `codex plugin marketplace add /path/to/memo`, then install `memo@memo` | `.agents/plugins/marketplace.json`, `plugins/memo/.codex-plugin/plugin.json`, `plugins/memo/skills/memo/SKILL.md`, `plugins/memo/.mcp.json`, `skills/memo/SKILL.md` |
| OpenCode | Direct MCP install supported | `memo install-slash --client opencode` or `memo mcp-command --client opencode` | `src/memo/runtime/install.py`, README MCP setup |
| Windsurf / Cascade | Direct MCP config install supported | `memo install-slash --client windsurf` writes `~/.codeium/windsurf/mcp_config.json` | `src/memo/cli.py`, README MCP setup |
| Devin | User skill install supported | `memo install-slash --client devin` | `skills/memo/SKILL.md` |

## Explicit Non-Goals

Do not publish a Docker MCP Catalog or hosted Smithery entry until there is a
tested non-MLX runtime. memo's production path is Apple Silicon + MLX + local
stdio. A Linux container would not have the same Metal-backed model runtime and
would either fail at first model call or silently ship a weaker product.

Smithery can list local stdio servers, but its current one-click hosted paths
are a poor fit for a private, local-first memory server. Publish there only when
the listing can point users back to the local `memo-mcp` binary without
containerizing the MLX runtime.

## Release Checklist

1. Bump `pyproject.toml`, `server.json`, `.claude-plugin/plugin.json`, and
   `plugins/memo/.codex-plugin/plugin.json` to the same version.
2. Build the wheel and confirm it includes `memo/agent_assets/commands`,
   `memo/agent_assets/skills`, `memo/agent_assets/plugins`, and
   `memo/agent_assets/.claude-plugin`.
3. Tag and publish a GitHub Release. The `publish` workflow builds and pushes
   PyPI, then publishes `server.json` to the official MCP Registry with GitHub
   OIDC.
4. After PyPI shows the new sdist, update `docs/homebrew/mlx-memo.rb` and
   mirror it to `jagoff/homebrew-memo/Formula/mlx-memo.rb`.
5. Verify the stores:
   - `python3 -m pip index versions mlx-memo`
   - `brew update && brew info jagoff/memo/mlx-memo`
   - `curl https://registry.modelcontextprotocol.io/v0.1/servers/io.github.jagoff/memo`
    - `claude plugin marketplace add jagoff/memo`
    - `memo install-slash --client codex --dry-run --repo /path/to/memo`
    - `memo install-slash --client opencode --dry-run`
    - `WINDSURF_MCP_CONFIG=$(mktemp) memo install-slash --client windsurf`

The live PyPI package is currently `0.6.0`; do not update Homebrew to `0.7.0`
until the `0.7.0` sdist exists on PyPI.
