import type { FeedbackRating } from "../types";

export type FeedbackButtonsProps = {
  current?: FeedbackRating;
  pending?: boolean;
  error?: string;
  onRate: (rating: FeedbackRating) => void;
};

export function FeedbackButtons({ current, pending, error, onRate }: FeedbackButtonsProps) {
  return (
    <div className="feedback-row" title="Tu rating alimenta el corpus de eval-chat">
      <span className="label">¿útil?</span>
      <button
        type="button"
        className={`feedback-btn${current === "up" ? " active" : ""}`}
        aria-label="Marcar como útil"
        aria-pressed={current === "up"}
        disabled={pending || current !== undefined}
        onClick={() => onRate("up")}
      >
        👍
      </button>
      <button
        type="button"
        className={`feedback-btn${current === "down" ? " active" : ""}`}
        aria-label="Marcar como no útil"
        aria-pressed={current === "down"}
        disabled={pending || current !== undefined}
        onClick={() => onRate("down")}
      >
        👎
      </button>
      {pending && <span className="feedback-status">guardando…</span>}
      {!pending && current && <span className="feedback-status">gracias</span>}
      {error && <span className="feedback-status error">error: {error}</span>}
    </div>
  );
}
