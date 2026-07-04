# Changelog

## 0.1.0

- Initial release.
- Contributes the `memo` MCP server to VS Code agent mode via a
  `mcpServerDefinitionProvider` (stdio transport).
- Launches memo with `uvx --from mlx-memo memo-mcp` by default, or an installed
  `memo-mcp` binary when `memo.useInstalledBinary` is enabled.
- Settings for `MEMO_DATA_DIR` (`memo.dataDir`) and Obsidian vault storage
  (`memo.vaultPath`).
