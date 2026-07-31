import type { EvidenceRef } from "../types";

function toneForSource(source?: string): "info" | "accent" {
  return (source ?? "").toLowerCase() === "memo" ? "accent" : "info";
}

function formatDate(iso?: string): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    return d.toLocaleString("es-AR", {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

export function EvidenceCard({ e, rank }: { e: EvidenceRef; rank: number }) {
  const meta = (e.metadata ?? {}) as Record<string, unknown>;
  const raw = (meta.raw ?? {}) as Record<string, unknown>;
  const tags = Array.isArray(raw.tags) ? (raw.tags as string[]) : [];
  const created = typeof raw.created === "string" ? (raw.created as string) : "";
  const path = typeof raw.path === "string" ? (raw.path as string) : "";
  const section = typeof meta.section === "string" ? (meta.section as string) : "";

  return (
    <article className="result">
      <header className="result-head">
        <span className="result-rank">{rank}</span>
        <div className="result-title-wrap">
          {e.title && <h3 className="result-title">{e.title}</h3>}
          <div className="result-source-row">
            <span className={`pill ${toneForSource(e.source)}`}>{e.source ?? "?"}</span>
            {typeof e.score === "number" && (
              <span className="result-score" title="match score">
                {(e.score * 100).toFixed(0)}% match
              </span>
            )}
            {section && <span className="result-section">{section}</span>}
            {created && <span className="result-date">{formatDate(created)}</span>}
          </div>
        </div>
      </header>

      {e.snippet && <div className="result-body">{e.snippet}</div>}

      {tags.length > 0 && (
        <div className="result-tags">
          {tags.slice(0, 8).map((t) => (
            <span className="tag" key={t}>#{t}</span>
          ))}
        </div>
      )}

      <footer className="result-foot">
        <code className="result-uri" title={e.uri}>{e.uri ?? "—"}</code>
        {path && <code className="result-path" title={path}>{path}</code>}
      </footer>
    </article>
  );
}
