import * as vscode from 'vscode';

const PROVIDER_ID = 'memo';
// `uvx` fetches the `mlx-memo` package from PyPI and runs its `memo-mcp`
// console script — nothing to install ahead of time beyond `uv`.
const UVX_ARGS = ['--from', 'mlx-memo', 'memo-mcp'];

export function activate(context: vscode.ExtensionContext): void {
  const didChange = new vscode.EventEmitter<void>();
  context.subscriptions.push(didChange);

  context.subscriptions.push(
    vscode.lm.registerMcpServerDefinitionProvider(PROVIDER_ID, {
      onDidChangeMcpServerDefinitions: didChange.event,
      provideMcpServerDefinitions: () => {
        const cfg = vscode.workspace.getConfiguration('memo');
        const useBinary = cfg.get<boolean>('useInstalledBinary', false);
        const command = useBinary ? 'memo-mcp' : 'uvx';
        const args = useBinary ? [] : UVX_ARGS;

        const env: Record<string, string> = {};
        const dataDir = (cfg.get<string>('dataDir', '') || '').trim();
        const vaultPath = (cfg.get<string>('vaultPath', '') || '').trim();
        if (dataDir) {
          env.MEMO_DATA_DIR = dataDir;
        }
        if (vaultPath) {
          env.MEMO_VAULT_PATH = vaultPath;
          env.MEMO_MEMORIES_IN_VAULT = '1';
        }

        return [new vscode.McpStdioServerDefinition('memo', command, args, env)];
      },
    }),
  );

  // Re-resolve the server definition (VS Code re-reads it and offers to refresh
  // tools) whenever the user changes any memo.* setting.
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('memo')) {
        didChange.fire();
      }
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('memo.setup', showSetup),
  );
}

async function showSetup(): Promise<void> {
  const pick = await vscode.window.showInformationMessage(
    'memo is registered as an MCP server for agent mode — it runs locally, no cloud, no API keys. ' +
      'By default it launches with `uvx --from mlx-memo memo-mcp`, which needs `uv` installed.',
    'Install uv',
    'memo docs',
  );
  if (pick === 'Install uv') {
    await vscode.env.openExternal(
      vscode.Uri.parse('https://docs.astral.sh/uv/getting-started/installation/'),
    );
  } else if (pick === 'memo docs') {
    await vscode.env.openExternal(vscode.Uri.parse('https://github.com/jagoff/memo'));
  }
}

export function deactivate(): void {
  // Registrations are disposed via context.subscriptions.
}
