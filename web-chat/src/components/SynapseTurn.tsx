import type { AskResponse, FeedbackRating } from "../types";
import { CiteText, buildCiteNumbers } from "./CiteText";
import { SourcesPanel } from "./SourcesPanel";
import { RouteStrip } from "./RouteStrip";
import { InsightProposal } from "./InsightProposal";
import { FeedbackButtons } from "./FeedbackButtons";

function UnderstandingBadges({ u, sources }: { u: AskResponse["understanding"]; sources?: AskResponse["sources"] }) {
  if (!u) return null;
  const badges: { label: string; tone: string; title?: string }[] = [];

  // Intent — always show; maps to readable label
  const intentLabels: Record<string, string> = {
    decide: "decisión",
    find: "búsqueda",
    summarize: "resumen",
    compare: "comparar",
    navigate: "navegar",
    whatsapp: "whatsapp",
  };
  const hasWaTranscript = (sources ?? []).some((s) =>
    String(s.title ?? "").startsWith("WhatsApp ·")
  );
  const intent = hasWaTranscript ? "whatsapp" : (u.intent ?? "find");
  badges.push({
    label: intentLabels[intent] ?? intent,
    tone: intent === "whatsapp" ? "wa" : "info",
  });

  // Type filter
  if (u.record_type_filter) {
    badges.push({ label: `tipo: ${u.record_type_filter}`, tone: "accent" });
  }

  // Recency
  if (u.recency_days) {
    const recLabel = u.recency_days === 1 ? "hoy" : u.recency_days === 7 ? "esta semana" : `últimos ${u.recency_days}d`;
    badges.push({ label: recLabel, tone: "warn" });
  }

  // Entities (named-CAPS phrases from query)
  if (u.entities_in_query && u.entities_in_query.length > 0) {
    badges.push({
      label: `entidades: ${u.entities_in_query.slice(0, 2).join(", ")}`,
      tone: "ok",
      title: u.entities_in_query.join(", "),
    });
  }

  // Keyword count — always show as last anchor so the row never collapses
  const kwCount = u.keyword_tokens?.length ?? 0;
  if (kwCount > 0) {
    badges.push({
      label: `${kwCount} kw`,
      tone: "neutral",
      title: u.keyword_tokens?.join(", "),
    });
  }

  return (
    <div className="understanding-row" title="Cómo Synapse entendió tu query">
      <span className="label">entendí</span>
      {badges.map((b, i) => (
        <span key={i} className={`pill ${b.tone}`} title={b.title}>{b.label}</span>
      ))}
    </div>
  );
}

export function SynapseTurn({
  response,
  historical = false,
  feedback,
  feedbackPending,
  feedbackError,
  onFeedback,
}: {
  response: AskResponse;
  historical?: boolean;
  feedback?: FeedbackRating;
  feedbackPending?: boolean;
  feedbackError?: string;
  onFeedback?: (rating: FeedbackRating) => void;
}) {
  const conflicts = response.reality_conflicts ?? [];
  const sources = response.sources ?? [];
  const answer = response.answer ?? "";
  const status = response.synthesis_status;
  const fallbackLabel =
    response.synthesis_source === "synapse.federated_packet"
      ? "Memo no respondió ahora · Synapse mostró evidencia federada"
      : "Memo no pudo completar la síntesis · Synapse mostró evidencia federada";

  const sourceIds = sources
    .map((s) => (s.id_short ?? s.id ?? "").toLowerCase())
    .filter((s) => s.length > 0);
  const citeNumbers = buildCiteNumbers(answer, sourceIds);

  return (
    <div className="bubble synapse-bubble">
      {!historical && (
        <>
          <RouteStrip
            route={response.route_decision}
            packetStatus={response.packet_status}
            traceId={response.trace_id}
          />
          <UnderstandingBadges u={response.understanding} sources={sources} />
        </>
      )}

      {(status === "ok" || status === "fallback") && answer && (
        <>
          {status === "fallback" && (
            <div className="synthesis-badge fallback" title={response.synthesis_error}>
              ⚠ {fallbackLabel}
            </div>
          )}
          <CiteText text={answer} citeNumbers={citeNumbers} />
        </>
      )}

      {(status === "error" || status === "unavailable") && (
        <div className="answer-error">
          <strong>No pude sintetizar respuesta.</strong>
          {response.synthesis_error && (
            <div className="answer-error-detail">{response.synthesis_error}</div>
          )}
          <div className="answer-error-hint">
            Mostrando fuentes encontradas igual:
          </div>
        </div>
      )}

      {!historical && onFeedback && response.trace_id && (
        <FeedbackButtons
          current={feedback}
          pending={feedbackPending}
          error={feedbackError}
          onRate={onFeedback}
        />
      )}

      {conflicts.length > 0 && (
        <div className="conflict-block">
          <h4>⚠ Conflictos detectados ({conflicts.length})</h4>
          {conflicts.map((c, i) => (
            <div key={`c-${i}`} className="conflict-card">
              <div className="conflict-kind">
                {c.kind ?? "?"}
                {c.freeze_write && <span className="pill bad" style={{ marginLeft: 8 }}>freeze_write</span>}
              </div>
              {c.summary && <div className="conflict-summary">{c.summary}</div>}
              <code className="result-uri">{c.conflict_id ?? "?"}</code>
            </div>
          ))}
        </div>
      )}

      <SourcesPanel
        sources={sources}
        citeNumbers={citeNumbers}
        openByDefault={
          status !== "ok" ||
          response.synthesis_source === "synapse.multi_source"
        }
        userQuery={response.query}
      />

      {!historical && response.notes && response.notes.length > 0 && (
        <ul className="notes">
          {response.notes.map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      )}

      {!historical && response.errors && Object.keys(response.errors).length > 0 && (
        <ul className="notes">
          {Object.entries(response.errors).map(([k, v]) => <li key={k}>{k}: {v}</li>)}
        </ul>
      )}

      {!historical && response.insight_proposal && (
        <InsightProposal candidate={response.insight_proposal} />
      )}
    </div>
  );
}
