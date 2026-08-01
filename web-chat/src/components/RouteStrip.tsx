import type { RouteDecision } from "../types";

function toneForTarget(target?: string): "ok" | "warn" | "info" | "accent" {
  if (!target) return "info";
  const t = target.toLowerCase();
  if (t === "memflow") return "info";
  if (t === "memo") return "accent";
  if (t === "both") return "ok";
  return "warn";
}

function pct(c?: number): string {
  if (typeof c !== "number") return "—";
  return `${Math.round(c * 100)}%`;
}

const BREAKDOWN_COLORS: Record<string, string> = {
  memflow: "#3b82f6",
  memo: "#a855f7",
  both: "#22c55e",
};

function pickBreakdown(
  raw: Record<string, unknown> | undefined,
): { key: string; value: number }[] {
  if (!raw || typeof raw !== "object") return [];
  const entries: { key: string; value: number }[] = [];
  for (const [k, v] of Object.entries(raw)) {
    const n = typeof v === "number" ? v : Number(v);
    if (Number.isFinite(n) && n > 0) entries.push({ key: k, value: n });
  }
  // Stable display order: memflow → memo → both → others
  const rank: Record<string, number> = { memflow: 0, memo: 1, both: 2 };
  entries.sort((a, b) => (rank[a.key] ?? 99) - (rank[b.key] ?? 99));
  return entries;
}

function formatSignals(
  raw: Record<string, string[] | undefined> | undefined,
): string {
  if (!raw) return "";
  const lines: string[] = [];
  for (const [k, v] of Object.entries(raw)) {
    if (Array.isArray(v) && v.length > 0) {
      lines.push(`${k}: ${v.join(", ")}`);
    }
  }
  return lines.join("\n");
}

function ScoreBreakdown({ route }: { route: RouteDecision }) {
  const breakdown = pickBreakdown(
    route.score_breakdown as Record<string, unknown> | undefined,
  );
  if (breakdown.length === 0) return null;
  const total = breakdown.reduce((acc, e) => acc + e.value, 0);
  if (total <= 0) return null;
  const signals = formatSignals(route.matched_signals);
  return (
    <span
      className="route-breakdown"
      title={signals || "route score breakdown"}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        marginLeft: "6px",
      }}
    >
      <span
        aria-label="route score breakdown"
        style={{
          display: "inline-flex",
          width: "64px",
          height: "8px",
          borderRadius: "4px",
          overflow: "hidden",
          background: "rgba(255,255,255,0.08)",
        }}
      >
        {breakdown.map((e) => (
          <span
            key={e.key}
            style={{
              width: `${(e.value / total) * 100}%`,
              background: BREAKDOWN_COLORS[e.key] ?? "#94a3b8",
            }}
            title={`${e.key} ${(e.value).toFixed(3)}`}
          />
        ))}
      </span>
    </span>
  );
}

export function RouteStrip({
  route,
  packetStatus,
  traceId,
}: {
  route: RouteDecision | null;
  packetStatus: string;
  traceId: string;
}) {
  // The search-only / multi-source chat path carries no route decision
  // (packet is skipped), which rendered an empty "routed to — intent —
  // packet skipped" strip with placeholder dashes. Hide it entirely: there's
  // nothing meaningful to show, and the trace id surfaces in the meta row.
  if (!route) return null;

  const target = route?.target ?? "—";
  const intent = route?.intent ?? route?.kind ?? "—";
  const reason = route?.reason ?? "";
  const statusTone: "ok" | "warn" | "bad" =
    packetStatus === "ready" ? "ok" : packetStatus === "partial" ? "warn" : "bad";
  return (
    <div className="route-strip">
      <span className="label">routed to</span>
      <span className={`pill ${toneForTarget(target)}`}>{target}</span>
      {route && <ScoreBreakdown route={route} />}
      <span className="label">intent</span>
      <span className="pill">{intent}</span>
      {route?.confidence !== undefined && (
        <>
          <span className="label">confidence</span>
          <span className="pill">{pct(route.confidence)}</span>
        </>
      )}
      <span className="label">packet</span>
      <span className={`pill ${statusTone}`}>{packetStatus}</span>
      {reason && <span className="label" title={reason}>· {reason}</span>}
      {traceId && <span className="label" style={{ marginLeft: "auto", opacity: 0.7 }}>{traceId}</span>}
    </div>
  );
}
