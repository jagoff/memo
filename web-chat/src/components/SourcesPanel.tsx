import { useState } from "react";
import type { AskSource } from "../types";
import { deleteMemory, submitSourceFeedback, type SourceFeedbackRating } from "../api";
import { ConfirmModal } from "./ConfirmModal";
import { MarkdownSnippet } from "./MarkdownSnippet";

type PendingDelete = { rawId: string; sid: string; title: string };

function toneForSource(src?: string): "info" | "accent" | "warn" {
  const s = (src ?? "").toLowerCase();
  if (s === "memory" || s === "memo") return "accent";
  if (s === "repo" || s === "vault" || s === "memflow") return "info";
  return "warn";
}

function pct(score?: number): string {
  if (typeof score !== "number") return "";
  return `${(score * 100).toFixed(0)}%`;
}

// Path may be "file.md:81-148" — strip line range for opening in Obsidian.
function stripLineRange(path: string): string {
  const idx = path.lastIndexOf(":");
  if (idx <= 0) return path;
  const tail = path.slice(idx + 1);
  if (/^\d+(-\d+)?$/.test(tail)) return path.slice(0, idx);
  return path;
}

function obsidianOpenUrl(repo: string, path: string): string | null {
  if (!repo || !path) return null;
  const cleanPath = stripLineRange(path);
  return `obsidian://open?vault=${encodeURIComponent(repo)}&file=${encodeURIComponent(cleanPath)}`;
}

function obsidianSearchUrl(query: string): string {
  return `obsidian://search?query=${encodeURIComponent(query)}`;
}

function formatScore(score: number | null | undefined): string {
  if (typeof score !== "number" || Number.isNaN(score)) return "";
  // Final/rerank scores can exceed 1.0 (composed with base). Show 2 decimals.
  return score >= 0 && score <= 1
    ? `${(score * 100).toFixed(0)}%`
    : score.toFixed(2);
}

export function SourcesPanel({
  sources,
  citeNumbers,
  openByDefault,
  userQuery,
}: {
  sources: AskSource[];
  citeNumbers: Map<string, number>;
  openByDefault: boolean;
  userQuery?: string;
}) {
  const [showDebug, setShowDebug] = useState(false);
  const [voted, setVoted] = useState<Record<string, SourceFeedbackRating>>({});
  const [voteError, setVoteError] = useState<string | null>(null);
  const [deletedIds, setDeletedIds] = useState<Set<string>>(new Set());
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<PendingDelete | null>(null);
  if (sources.length === 0) return null;

  function openDeleteConfirm(rawSourceId: string, title: string): void {
    if (!rawSourceId) return;
    const sid = rawSourceId.startsWith("memo:")
      ? rawSourceId.slice("memo:".length)
      : rawSourceId;
    setDeleteError(null);
    setPendingDelete({ rawId: rawSourceId, sid, title });
  }

  async function confirmDelete(): Promise<void> {
    const pending = pendingDelete;
    if (!pending) return;
    setDeletingIds((prev) => new Set(prev).add(pending.rawId));
    setDeleteError(null);
    try {
      const receipt = await deleteMemory({ memoryId: pending.sid });
      setDeletedIds((prev) => new Set(prev).add(pending.rawId));
      setPendingDelete(null);
      // Surface to devtools so user can confirm in console. The server reports
      // whether the .md was moved into 04-Archive (data-driven copy).
      const msg = receipt.archived
        ? "[chat] memoria borrada + .md movido a 04-Archive (recuperable)"
        : "[chat] memoria borrada (vault .md conservado)";
      console.info(msg, receipt);
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeletingIds((prev) => {
        const next = new Set(prev);
        next.delete(pending.rawId);
        return next;
      });
    }
  }

  async function castVote(
    rawSourceId: string,
    rating: SourceFeedbackRating,
  ): Promise<void> {
    if (!userQuery || !rawSourceId) return;
    // Strip the `memo:` namespace the federator adds when emitting
    // source_ids — memo.feedback_record works on the raw meta.id.
    const sid = rawSourceId.startsWith("memo:")
      ? rawSourceId.slice("memo:".length)
      : rawSourceId;
    // Optimistic UI — colour the button immediately so the click feels
    // instant. The server round-trip (subprocess + SQLite write) takes
    // ~700ms-1s; without optimistic state the user sees a perceptible
    // lag between click and visual confirmation.
    const previous = voted[rawSourceId];
    setVoted((prev) => ({ ...prev, [rawSourceId]: rating }));
    setVoteError(null);
    try {
      await submitSourceFeedback({
        sourceId: sid,
        query: userQuery,
        rating,
      });
    } catch (err) {
      // Roll back to the previous state on failure so the UI doesn't
      // lie about server-confirmed feedback.
      setVoted((prev) => {
        const next = { ...prev };
        if (previous) {
          next[rawSourceId] = previous;
        } else {
          delete next[rawSourceId];
        }
        return next;
      });
      setVoteError(err instanceof Error ? err.message : String(err));
    }
  }

  const hasRerank = sources.some(
    (s) => typeof s.rerank_score === "number" || typeof s.final_score === "number",
  );

  const enriched = sources
    .filter((s) => {
      const rawId = String(s.id ?? s.id_short ?? "");
      return !deletedIds.has(rawId);
    })
    .map((s) => {
      const idShort = (s.id_short ?? s.id ?? "").toLowerCase();
      const n = citeNumbers.get(idShort) ?? null;
      return { ...s, _idShort: idShort, _n: n };
    });

  enriched.sort((a, b) => {
    if (a._n == null && b._n == null) return (b.score ?? 0) - (a.score ?? 0);
    if (a._n == null) return 1;
    if (b._n == null) return -1;
    return a._n - b._n;
  });

  return (
    <details className="sources" open={openByDefault}>
      <summary>
        <span className="sources-summary-label">Fuentes</span>
        <span className="sources-summary-count">{sources.length}</span>
        {hasRerank && (
          <button
            type="button"
            className="sources-debug-toggle"
            onClick={(e) => {
              e.preventDefault();
              setShowDebug((v) => !v);
            }}
            style={{
              marginLeft: "8px",
              fontSize: "0.75rem",
              opacity: 0.7,
              background: "transparent",
              border: "1px solid currentColor",
              borderRadius: "4px",
              padding: "1px 6px",
              cursor: "pointer",
            }}
            title="Mostrar rerank/final score por fuente"
          >
            {showDebug ? "ocultar debug" : "debug"}
          </button>
        )}
      </summary>
      {voteError && (
        <div className="sources-vote-error" style={{ color: "var(--warn, #c66)", fontSize: "0.8rem", padding: "4px 8px" }}>
          {voteError}
        </div>
      )}
      {deleteError && (
        <div className="sources-delete-error" style={{ color: "var(--warn, #c66)", fontSize: "0.8rem", padding: "4px 8px" }}>
          No se pudo borrar la memoria: {deleteError}
        </div>
      )}
      <div className="sources-list">
        {enriched.map((s, idx) => {
          const repo = s.obsidian_vault ?? s.repo_name ?? "";
          const obsUrl = obsidianOpenUrl(repo, s.path ?? "");
          const titleText = s.title ?? s.id_short ?? "—";
          const titleNode = obsUrl ? (
            <a
              className="source-title-link"
              href={obsUrl}
              title={`Abrir en Obsidian (${repo})`}
            >
              {titleText}
            </a>
          ) : (
            <span className="source-title">{titleText}</span>
          );
          return (
            <article
              key={`${s.id ?? idx}`}
              id={s._n != null ? `source-${s._n}` : undefined}
              className="source-card"
            >
              <header className="source-card-head">
                {s._n != null && <span className="source-n">[{s._n}]</span>}
                {titleNode}
                <span className={`pill ${toneForSource(s.source)}`}>{s.source ?? "?"}</span>
                {pct(s.score) && <span className="source-score">{pct(s.score)}</span>}
                {userQuery && (s.id ?? s.id_short) && (
                  <span
                    className="source-feedback"
                    style={{ marginLeft: "6px", display: "inline-flex", gap: "4px" }}
                  >
                    <button
                      type="button"
                      title="👍 priorizá esta fuente para queries similares"
                      onClick={() => castVote(String(s.id ?? s.id_short ?? ""), "up")}
                      style={{
                        background: voted[String(s.id ?? s.id_short ?? "")] === "up" ? "rgba(80,200,120,0.25)" : "transparent",
                        border: "1px solid currentColor",
                        borderRadius: "4px",
                        padding: "0 5px",
                        cursor: "pointer",
                        fontSize: "0.8rem",
                        opacity: voted[String(s.id ?? s.id_short ?? "")] === "up" ? 1 : 0.65,
                      }}
                    >
                      👍
                    </button>
                    <button
                      type="button"
                      title="👎 ocultá esta fuente para queries similares"
                      onClick={() => castVote(String(s.id ?? s.id_short ?? ""), "down")}
                      style={{
                        background: voted[String(s.id ?? s.id_short ?? "")] === "down" ? "rgba(220,90,90,0.25)" : "transparent",
                        border: "1px solid currentColor",
                        borderRadius: "4px",
                        padding: "0 5px",
                        cursor: "pointer",
                        fontSize: "0.8rem",
                        opacity: voted[String(s.id ?? s.id_short ?? "")] === "down" ? 1 : 0.65,
                      }}
                    >
                      👎
                    </button>
                    {(() => {
                      const kind = String(s.source ?? "").toLowerCase();
                      const supportsDelete = kind === "memo" || kind === "memory";
                      if (!supportsDelete) return null;
                      const rawId = String(s.id ?? s.id_short ?? "");
                      const isDeleting = deletingIds.has(rawId);
                      return (
                        <button
                          type="button"
                          title="🗑️ Borrar de memo + mover .md a 04-Archive (no vuelve; recuperable)"
                          onClick={() => openDeleteConfirm(rawId, String(s.title ?? rawId))}
                          disabled={isDeleting}
                          style={{
                            background: "transparent",
                            border: "1px solid currentColor",
                            borderRadius: "4px",
                            padding: "0 5px",
                            cursor: isDeleting ? "wait" : "pointer",
                            fontSize: "0.8rem",
                            opacity: isDeleting ? 0.4 : 0.65,
                          }}
                        >
                          {isDeleting ? "…" : "🗑️"}
                        </button>
                      );
                    })()}
                  </span>
                )}
                {showDebug && (
                  <span
                    className="source-debug"
                    style={{
                      marginLeft: "auto",
                      fontSize: "0.7rem",
                      opacity: 0.7,
                      fontFamily: "monospace",
                    }}
                    title="Synapse rerank/final/rrf scores"
                  >
                    {typeof s.rerank_score === "number" && (
                      <span style={{ marginRight: "6px" }}>
                        rerank {formatScore(s.rerank_score)}
                      </span>
                    )}
                    {typeof s.final_score === "number" && (
                      <span style={{ marginRight: "6px" }}>
                        final {formatScore(s.final_score)}
                      </span>
                    )}
                    {typeof s.rrf === "number" && (
                      <span>orig #{s.rrf}</span>
                    )}
                  </span>
                )}
              </header>
              {s.snippet && (
                <div className="source-snippet">
                  <MarkdownSnippet text={s.snippet} />
                </div>
              )}
              <footer className="source-foot">
                {s.path ? (
                  obsUrl ? (
                    <a className="source-path-link" href={obsUrl} title="Abrir en Obsidian">
                      <code>{s.path}</code>
                    </a>
                  ) : (
                    <a
                      className="source-path-link"
                      href={obsidianSearchUrl(stripLineRange(s.path))}
                      title="Buscar en Obsidian"
                    >
                      <code>{s.path}</code>
                    </a>
                  )
                ) : (
                  <code title={s.id}>{s.id_short ?? s.id ?? ""}</code>
                )}
                {repo && <span className="source-repo">{repo}</span>}
                {s.type && <span className="source-type">{s.type}</span>}
              </footer>
            </article>
          );
        })}
      </div>
      <ConfirmModal
        open={pendingDelete !== null}
        variant="danger"
        title="Borrar esta memoria"
        message={
          pendingDelete ? (
            <>
              <p style={{ margin: 0 }}>
                Vas a borrar <strong>{pendingDelete.title}</strong>{" "}
                <code className="mono">({pendingDelete.sid.slice(0, 8)})</code>.
              </p>
              <p style={{ marginTop: 10, marginBottom: 0 }}>
                Se borra la entrada en memo y la nota se mueve a{" "}
                <code className="mono">04-Archive/</code> (carpeta excluida del
                índice), así que no vuelve a aparecer. El archivo{" "}
                <code className="mono">.md</code> no se destruye — queda
                recuperable en el archivo.
              </p>
              <p style={{ marginTop: 10, marginBottom: 0, opacity: 0.85 }}>
                Reversible: mové el <code className="mono">.md</code> de vuelta
                desde <code className="mono">04-Archive/</code> y re-ingestá.
              </p>
            </>
          ) : (
            ""
          )
        }
        confirmLabel="Borrar"
        cancelLabel="Cancelar"
        busy={pendingDelete ? deletingIds.has(pendingDelete.rawId) : false}
        onConfirm={() => {
          void confirmDelete();
        }}
        onCancel={() => setPendingDelete(null)}
      />
    </details>
  );
}
