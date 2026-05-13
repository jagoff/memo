import { usePluginData, type PluginWidgetProps } from "@paperclipai/plugin-sdk/ui";

type Stats = {
  total?: number;
  vault_path?: string;
  state_dir?: string;
  embedder?: string;
  llm?: string;
  error?: string;
};

export function DashboardWidget(_props: PluginWidgetProps) {
  const { data, loading, error } = usePluginData<Stats>("stats");

  if (loading) return <div>Loading memo stats…</div>;
  if (error) return <div style={{ color: "crimson" }}>memo error: {error.message}</div>;
  if (data?.error) {
    return (
      <div style={{ color: "crimson" }}>
        <strong>Memo not reachable</strong>
        <div style={{ fontSize: "0.85em" }}>{data.error}</div>
      </div>
    );
  }

  return (
    <div style={{ display: "grid", gap: "0.35rem", fontSize: "0.9em" }}>
      <strong>Memo (local MCP memory)</strong>
      <div>
        <span style={{ opacity: 0.7 }}>Total:</span> {data?.total ?? "—"}
      </div>
      {data?.vault_path && (
        <div style={{ wordBreak: "break-all" }}>
          <span style={{ opacity: 0.7 }}>Vault:</span> {data.vault_path}
        </div>
      )}
      {data?.embedder && (
        <div>
          <span style={{ opacity: 0.7 }}>Embedder:</span> {data.embedder}
        </div>
      )}
    </div>
  );
}
