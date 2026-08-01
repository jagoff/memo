import { useState } from "react";
import { ApiError, captureInsight } from "../api";
import type { CaptureResult, InsightCandidate } from "../types";

type Status = "idle" | "saving" | "saved" | "error";

const PREVIEW_LIMIT = 200;

const cardStyle: React.CSSProperties = {
  marginTop: 14,
  padding: "12px 14px",
  borderRadius: 10,
  border: "1px solid rgba(192, 132, 252, 0.45)",
  background: "rgba(192, 132, 252, 0.06)",
  display: "flex",
  flexDirection: "column",
  gap: 8,
};

const headerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  flexWrap: "wrap",
};

const headerTitleStyle: React.CSSProperties = {
  fontSize: 12,
  fontWeight: 600,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  color: "var(--accent)",
};

const scoreBadgeStyle: React.CSSProperties = {
  fontFamily: "var(--mono)",
  fontSize: 10.5,
  padding: "2px 8px",
  borderRadius: 999,
  border: "1px solid rgba(192, 132, 252, 0.45)",
  background: "rgba(192, 132, 252, 0.12)",
  color: "var(--accent)",
  fontWeight: 600,
};

const typeBadgeStyle: React.CSSProperties = {
  fontFamily: "var(--mono)",
  fontSize: 10.5,
  padding: "2px 8px",
  borderRadius: 999,
  border: "1px solid var(--border)",
  background: "var(--bg-elev)",
  color: "var(--fg-dim)",
  fontWeight: 500,
};

const previewTitleStyle: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 600,
  color: "var(--fg)",
  letterSpacing: "-0.01em",
};

const previewBodyStyle: React.CSSProperties = {
  fontSize: 13,
  lineHeight: 1.55,
  color: "var(--fg-dim)",
  whiteSpace: "pre-wrap",
  wordWrap: "break-word",
};

const toggleStyle: React.CSSProperties = {
  background: "transparent",
  border: 0,
  color: "var(--accent)",
  cursor: "pointer",
  font: "inherit",
  fontSize: 12,
  padding: 0,
  marginLeft: 6,
  textDecoration: "underline dashed",
};

const tagsRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 6,
};

const actionsStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  flexWrap: "wrap",
};

const buttonStyleBase: React.CSSProperties = {
  background: "var(--accent-strong)",
  color: "#170a32",
  border: 0,
  padding: "6px 14px",
  borderRadius: 8,
  cursor: "pointer",
  fontWeight: 600,
  fontSize: 13,
  fontFamily: "inherit",
};

const buttonDisabledStyle: React.CSSProperties = {
  ...buttonStyleBase,
  background: "var(--border-strong)",
  color: "var(--fg-muted)",
  cursor: "not-allowed",
};

const buttonSavedStyle: React.CSSProperties = {
  ...buttonStyleBase,
  background: "rgba(52, 211, 153, 0.18)",
  color: "var(--ok)",
  border: "1px solid rgba(52, 211, 153, 0.45)",
  cursor: "default",
};

const savedMetaStyle: React.CSSProperties = {
  fontFamily: "var(--mono)",
  fontSize: 11,
  color: "var(--ok)",
};

const errorStyle: React.CSSProperties = {
  fontFamily: "var(--mono)",
  fontSize: 11.5,
  color: "var(--bad)",
};

const rationaleStyle: React.CSSProperties = {
  fontSize: 11.5,
  color: "var(--fg-muted)",
  fontStyle: "italic",
};

export function InsightProposal({
  candidate,
  onCaptured,
}: {
  candidate: InsightCandidate;
  onCaptured?: (result: CaptureResult) => void;
}) {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<CaptureResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const body = candidate.body ?? "";
  const isLong = body.length > PREVIEW_LIMIT;
  const visibleBody = expanded || !isLong ? body : `${body.slice(0, PREVIEW_LIMIT).trimEnd()}…`;

  const handleCapture = async () => {
    if (status === "saving" || status === "saved") return;
    setStatus("saving");
    setError(null);
    try {
      const r = await captureInsight(candidate);
      setResult(r);
      if (r.ok) {
        setStatus("saved");
        onCaptured?.(r);
      } else {
        setStatus("error");
        setError(r.error ?? "captura no completada");
      }
    } catch (err) {
      setStatus("error");
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

  const scoreLabel = `${Math.round(candidate.score)}/100`;
  const buttonLabel =
    status === "saving"
      ? "guardando…"
      : status === "saved"
      ? "✓ guardado"
      : "Capturar a memo";

  const buttonStyle =
    status === "saving"
      ? buttonDisabledStyle
      : status === "saved"
      ? buttonSavedStyle
      : buttonStyleBase;

  return (
    <div style={cardStyle} className="insight-proposal">
      <div style={headerStyle}>
        <span style={{ fontSize: 16 }}>💎</span>
        <span style={headerTitleStyle}>Capturable insight</span>
        <span style={scoreBadgeStyle} title="Score 0-100">{scoreLabel}</span>
        <span style={typeBadgeStyle} title="Tipo sugerido">{candidate.suggested_type}</span>
      </div>

      <div style={previewTitleStyle}>{candidate.title}</div>

      {body && (
        <div style={previewBodyStyle}>
          {visibleBody}
          {isLong && (
            <button
              type="button"
              style={toggleStyle}
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? "ver menos" : "ver completo"}
            </button>
          )}
        </div>
      )}

      {candidate.tags && candidate.tags.length > 0 && (
        <div style={tagsRowStyle}>
          {candidate.tags.map((t, i) => (
            <span key={`${t}-${i}`} className="tag">#{t}</span>
          ))}
        </div>
      )}

      {candidate.rationale && (
        <div style={rationaleStyle} title="Por qué Synapse propone capturar esto">
          {candidate.rationale}
        </div>
      )}

      <div style={actionsStyle}>
        <button
          type="button"
          style={buttonStyle}
          onClick={handleCapture}
          disabled={status === "saving" || status === "saved"}
        >
          {buttonLabel}
        </button>
        {status === "saved" && result && (result.memoria_id || result.uri) && (
          <span style={savedMetaStyle}>
            {result.memoria_id ? `id: ${result.memoria_id}` : ""}
            {result.uri ? (result.memoria_id ? " · " : "") + result.uri : ""}
          </span>
        )}
      </div>

      {status === "error" && error && (
        <div style={errorStyle}>Error: {error}</div>
      )}
    </div>
  );
}
