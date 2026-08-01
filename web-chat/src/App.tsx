import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  ask,
  askStream,
  deleteAllSessions,
  deleteSession,
  fetchSessions,
  fetchSuggestions,
  loadSession,
  submitFeedback,
  type StreamEvent,
} from "./api";
import type {
  AskResponse,
  ChatHistoryTurn,
  ChatTurn,
  FeedbackRating,
  SessionListItem,
  SuggestionChip,
} from "./types";
import { Composer } from "./components/Composer";
import { ConfirmModal } from "./components/ConfirmModal";
import { SynapseTurn } from "./components/SynapseTurn";

type PendingSessionDelete = { sessionId: string; label: string };

const USE_STREAM = true;
const SIDEBAR_STORAGE_KEY = "synapse.chat.sidebarCollapsed";

function emptyResponse(query: string): AskResponse {
  return {
    schema: "synapse.web.ask.v4",
    generated_at: new Date().toISOString(),
    query,
    trace_id: "",
    chat_session_id: "",
    packet_status: "",
    federation_status: "",
    route_decision: null,
    answer: "",
    sources: [],
    citations: [],
    synthesis_status: "ok",
    synthesis_source: "",
    synthesis_error: "",
    understanding: {},
    pipeline_trace: [],
    total_ms: 0,
    reality_conflicts: [],
    notes: [],
    semantic: {},
    errors: {},
  };
}

const FALLBACK_SUGGESTIONS: SuggestionChip[] = [
  { label: "memorias recientes", query: "qué se capturó recientemente", kind: "static" },
  { label: "decisiones de la semana", query: "decisiones de esta semana", kind: "static" },
  { label: "conflictos abiertos", query: "conflictos abiertos en synapse", kind: "static" },
  { label: "estado del sistema", query: "estado del sistema y backends", kind: "static" },
];

function nowIso(): string {
  return new Date().toISOString();
}

function randId(): string {
  return Math.random().toString(36).slice(2, 10);
}

const STAGE_LABELS: Record<string, string> = {
  memo_retrieval: "🔍 Buscando memorias",
  rerank: "🔄 Reordenando (segundo plano)",
  memo_synthesis: "🧠 Generando con memo",
  self_synthesis: "🧠 Sintetizando",
  multi_source_synthesis: "🧠 Fusionando notas relacionadas",
  streaming: "✍️ Generando respuesta",
};

function stageLabel(name: string, phase: "start" | "done", ms?: number): string {
  const base = STAGE_LABELS[name] ?? name;
  if (phase === "done") {
    const elapsed = typeof ms === "number" ? ` (${(ms / 1000).toFixed(1)}s)` : "";
    return `${base} ✓${elapsed}`;
  }
  return `${base}…`;
}

function dedupSessions(sessions: SessionListItem[], limit: number): SessionListItem[] {
  const seen = new Set<string>();
  const out: SessionListItem[] = [];
  for (const s of sessions) {
    const key = (s.label ?? "").trim().toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(s);
    if (out.length >= limit) break;
  }
  return out;
}

function initialSidebarCollapsed(): boolean {
  if (typeof window === "undefined") return false;
  try {
    const stored = window.localStorage.getItem(SIDEBAR_STORAGE_KEY);
    if (stored === "true") return true;
    if (stored === "false") return false;
  } catch {
    // Storage can be unavailable in restricted browser contexts.
  }
  return window.matchMedia("(max-width: 720px)").matches;
}

export default function App() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [pending, setPending] = useState(false);
  const [hasInput, setHasInput] = useState(false);
  const [suggestions, setSuggestions] = useState<SuggestionChip[]>(FALLBACK_SUGGESTIONS);
  const [recentSessions, setRecentSessions] = useState<SessionListItem[]>([]);
  const [sessionsLoaded, setSessionsLoaded] = useState(false);
  const [loadingSessionId, setLoadingSessionId] = useState<string>("");
  const [pendingSessionDelete, setPendingSessionDelete] = useState<PendingSessionDelete | null>(null);
  const [deletingSessionId, setDeletingSessionId] = useState<string>("");
  const [sessionDeleteError, setSessionDeleteError] = useState<string | null>(null);
  const [pendingDeleteAll, setPendingDeleteAll] = useState(false);
  const [deletingAll, setDeletingAll] = useState(false);
  const [lastAnchorId, setLastAnchorId] = useState<string | null>(null);
  const [chatSessionId, setChatSessionId] = useState<string>("");
  const [currentStage, setCurrentStage] = useState<string>("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(initialSidebarCollapsed);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Load dynamic chips from memo on mount; refresh after every assistant turn.
  useEffect(() => {
    let cancelled = false;
    fetchSuggestions(6).then((chips) => {
      if (cancelled) return;
      if (chips.length > 0) setSuggestions(chips);
    });
    return () => {
      cancelled = true;
    };
  }, [turns.length]);

  // Recent chat sessions: load on mount and after the local turn count
  // changes (so new sessions appear once the assistant turn completes).
  useEffect(() => {
    let cancelled = false;
    fetchSessions(10).then((sessions) => {
      if (cancelled) return;
      setRecentSessions(sessions);
      setSessionsLoaded(true);
    });
    return () => {
      cancelled = true;
    };
  }, [turns.length]);

  const started = turns.length > 0 || pending || hasInput;
  const sidebarSessions = dedupSessions(recentSessions, 10);

  useEffect(() => {
    try {
      window.localStorage.setItem(SIDEBAR_STORAGE_KEY, String(sidebarCollapsed));
    } catch {
      // Persistence is a convenience; the sidebar still works without storage.
    }
  }, [sidebarCollapsed]);

  // Anchor the last user question to the top of the viewport when sent.
  // The question stays put while the response renders below.
  useEffect(() => {
    if (!lastAnchorId) return;
    requestAnimationFrame(() => {
      const el = document.getElementById(`turn-${lastAnchorId}`);
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }, [lastAnchorId]);

  const openSession = useCallback(
    async (id: string) => {
      if (!id || pending || loadingSessionId) return;
      setLoadingSessionId(id);
      try {
        const history = await loadSession(id);
        if (!history) return;
        const loaded: ChatTurn[] = [];
        for (const t of history.turns) {
          if (t.role === "user") {
            loaded.push({
              kind: "user",
              id: randId(),
              at: t.at ?? nowIso(),
              text: t.text,
            });
          } else {
            loaded.push({
              kind: "synapse",
              id: randId(),
              at: t.at ?? nowIso(),
              response: {
                ...emptyResponse(""),
                chat_session_id: id,
                answer: t.text,
                synthesis_source: "memflow.session_replay",
              },
              latencyMs: 0,
              historical: true,
            });
          }
        }
        setTurns(loaded);
        setChatSessionId(id);
        const lastUser = [...loaded].reverse().find((t) => t.kind === "user");
        if (lastUser) setLastAnchorId(lastUser.id);
      } finally {
        setLoadingSessionId("");
      }
    },
    [pending, loadingSessionId],
  );

  const startNewChat = useCallback(() => {
    if (pending) return;
    setTurns([]);
    setHasInput(false);
    setLastAnchorId(null);
    setChatSessionId("");
    setCurrentStage("");
  }, [pending]);

  const send = useCallback(
    async (text: string) => {
      if (!text.trim() || pending) return;
      const userTurn: ChatTurn = { kind: "user", id: randId(), at: nowIso(), text };
      const history: ChatHistoryTurn[] = turns
        .map((t): ChatHistoryTurn | null => {
          if (t.kind === "user") return { role: "user", text: t.text };
          if (t.kind === "synapse") {
            const answer = t.response.answer ?? "";
            if (!answer) return null;
            return { role: "assistant", text: answer };
          }
          return null;
        })
        .filter((t): t is ChatHistoryTurn => t !== null);
      setTurns((prev) => [...prev, userTurn]);
      setLastAnchorId(userTurn.id);
      setPending(true);
      const startedAt = performance.now();

      if (USE_STREAM) {
        const synapseTurnId = randId();
        const placeholder: ChatTurn = {
          kind: "synapse",
          id: synapseTurnId,
          at: nowIso(),
          response: { ...emptyResponse(text), answer: "" },
          latencyMs: 0,
        };
        setTurns((prev) => [...prev, placeholder]);
        const updateTurn = (updater: (r: AskResponse) => AskResponse) => {
          setTurns((prev) =>
            prev.map((t) =>
              t.kind === "synapse" && t.id === synapseTurnId
                ? { ...t, response: updater(t.response), latencyMs: Math.round(performance.now() - startedAt) }
                : t,
            ),
          );
        };
        try {
          await askStream(text, history, 7, chatSessionId, (event: StreamEvent) => {
            if (event.type === "context") {
              if (event.chat_session_id && event.chat_session_id !== chatSessionId) {
                setChatSessionId(event.chat_session_id);
              }
              updateTurn((r) => ({
                ...r,
                trace_id: event.trace_id ?? r.trace_id,
                chat_session_id: event.chat_session_id ?? r.chat_session_id,
                sources: (event.sources as any) ?? r.sources,
                understanding: (event.understanding as any) ?? r.understanding,
                pipeline_trace: (event.pipeline_partial as any) ?? r.pipeline_trace,
                route_decision: (event.route_decision as any) ?? r.route_decision,
                packet_status: event.packet_status ?? r.packet_status,
                federation_status: event.federation_status ?? r.federation_status,
                semantic: event.semantic ?? r.semantic,
                errors: event.errors ?? r.errors,
              }));
            } else if (event.type === "stage") {
              setCurrentStage(stageLabel(event.name, event.phase, event.ms));
            } else if (event.type === "token") {
              updateTurn((r) => ({ ...r, answer: (r.answer ?? "") + event.text }));
            } else if (event.type === "insight_proposal") {
              updateTurn((r) => ({ ...r, insight_proposal: event.candidate }));
            } else if (event.type === "done") {
              setCurrentStage("");
              updateTurn((r) => ({
                ...r,
                answer: event.answer ?? r.answer,
                citations: (event.citations as any) ?? r.citations,
                synthesis_status: (event.synthesis_status as any) ?? r.synthesis_status,
                synthesis_source: event.synthesis_source ?? r.synthesis_source,
                synthesis_error: event.synthesis_error ?? r.synthesis_error,
                sources: (event.sources as any) ?? r.sources,
                pipeline_trace: (event.pipeline_trace as any) ?? r.pipeline_trace,
                total_ms: event.total_ms ?? r.total_ms,
                trace_id: event.trace_id ?? r.trace_id,
                chat_session_id: event.chat_session_id ?? r.chat_session_id,
                federation_status: event.federation_status ?? r.federation_status,
                semantic: event.semantic ?? r.semantic,
                errors: event.errors ?? r.errors,
              }));
            } else if (event.type === "error") {
              setCurrentStage("");
              setTurns((prev) => [
                ...prev.filter((t) => !(t.kind === "synapse" && t.id === synapseTurnId)),
                { kind: "error", id: randId(), at: nowIso(), message: event.message },
              ]);
            }
          });
        } catch (err) {
          const message = err instanceof ApiError ? err.message : String(err);
          setCurrentStage("");
          setTurns((prev) => [
            ...prev.filter((t) => !(t.kind === "synapse" && t.id === synapseTurnId)),
            { kind: "error", id: randId(), at: nowIso(), message },
          ]);
        } finally {
          setPending(false);
          setCurrentStage("");
        }
        return;
      }

      try {
        const response = await ask(text, history, 7, chatSessionId);
        const latencyMs = Math.round(performance.now() - startedAt);
        if (response.chat_session_id && response.chat_session_id !== chatSessionId) {
          setChatSessionId(response.chat_session_id);
        }
        setTurns((prev) => [
          ...prev,
          { kind: "synapse", id: randId(), at: nowIso(), response, latencyMs },
        ]);
      } catch (err) {
        const message = err instanceof ApiError ? err.message : String(err);
        setTurns((prev) => [...prev, { kind: "error", id: randId(), at: nowIso(), message }]);
      } finally {
        setPending(false);
      }
    },
    [pending, turns, chatSessionId],
  );

  const handleFeedback = useCallback(
    async (turnId: string, rating: FeedbackRating) => {
      const turn = turns.find((t) => t.kind === "synapse" && t.id === turnId);
      if (!turn || turn.kind !== "synapse") return;
      if (turn.feedback || turn.feedbackPending) return;
      const { response } = turn;
      setTurns((prev) =>
        prev.map((t) =>
          t.kind === "synapse" && t.id === turnId
            ? { ...t, feedbackPending: true, feedbackError: undefined }
            : t,
        ),
      );
      try {
        await submitFeedback({
          turnId,
          rating,
          query: response.query,
          answer: response.answer ?? "",
          sources: response.sources ?? [],
          traceId: response.trace_id,
          chatSessionId: response.chat_session_id ?? chatSessionId,
        });
        setTurns((prev) =>
          prev.map((t) =>
            t.kind === "synapse" && t.id === turnId
              ? { ...t, feedback: rating, feedbackPending: false }
              : t,
          ),
        );
      } catch (err) {
        const message = err instanceof ApiError ? err.message : String(err);
        setTurns((prev) =>
          prev.map((t) =>
            t.kind === "synapse" && t.id === turnId
              ? { ...t, feedbackPending: false, feedbackError: message }
              : t,
          ),
        );
      }
    },
    [turns, chatSessionId],
  );

  return (
    <div className={`app-shell ${sidebarCollapsed ? "sidebar-is-collapsed" : ""}`}>
      <aside className="sidebar" aria-label="Navegación de chat">
        <div className="sidebar-top">
          <a className="sidebar-brand" href="/chat" aria-label="Synapse Chat">
            <span className="sidebar-brand-mark" aria-hidden="true">S</span>
            <span className="sidebar-label">Synapse</span>
          </a>
          <button
            className="sidebar-icon-button"
            type="button"
            onClick={() => setSidebarCollapsed((value) => !value)}
            aria-label={sidebarCollapsed ? "Mostrar barra lateral" : "Ocultar barra lateral"}
            aria-expanded={!sidebarCollapsed}
            title={sidebarCollapsed ? "Mostrar barra lateral" : "Ocultar barra lateral"}
          >
            <span aria-hidden="true">{sidebarCollapsed ? "›" : "‹"}</span>
          </button>
        </div>

        <button
          className="sidebar-action"
          type="button"
          onClick={startNewChat}
          disabled={pending}
          title="Nuevo chat"
        >
          <span className="sidebar-icon" aria-hidden="true">+</span>
          <span className="sidebar-label">Nuevo chat</span>
        </button>

        <nav className="sidebar-nav" aria-label="Principal">
          <a href="/" title="Dashboard">
            <span className="sidebar-icon" aria-hidden="true">⌂</span>
            <span className="sidebar-label">Dashboard</span>
          </a>
          <a href="/chat" className="active" title="Chat">
            <span className="sidebar-icon" aria-hidden="true">●</span>
            <span className="sidebar-label">Chat</span>
          </a>
        </nav>

        <div className="sidebar-recents" aria-label="Chats recientes">
          <div className="sidebar-recents-head">
            <h3>Recientes</h3>
            {sessionsLoaded && sidebarSessions.length > 0 && (
              <button
                type="button"
                className="sidebar-recents-clear"
                title="Borrar todos los chats"
                aria-label="Borrar todos los chats"
                onClick={() => {
                  setSessionDeleteError(null);
                  setPendingDeleteAll(true);
                }}
                disabled={pending || deletingAll}
              >
                🗑️
              </button>
            )}
          </div>
          <div className="sidebar-session-list">
            {!sessionsLoaded && <div className="sidebar-empty">Cargando…</div>}
            {sessionsLoaded && sidebarSessions.length === 0 && (
              <div className="sidebar-empty">Sin chats recientes</div>
            )}
            {sidebarSessions.map((s) => {
              const isDeleting = deletingSessionId === s.session_id;
              return (
                <div className="sidebar-session-row" key={s.session_id}>
                  <button
                    type="button"
                    onClick={() => openSession(s.session_id)}
                    disabled={pending || !!loadingSessionId || isDeleting}
                    title={s.label || s.session_id}
                    className={s.session_id === chatSessionId ? "active session-open-btn" : "session-open-btn"}
                  >
                    <span className="session-label">
                      {loadingSessionId === s.session_id
                        ? "Cargando…"
                        : isDeleting
                          ? "Borrando…"
                          : s.label}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="session-delete-btn"
                    title="Borrar este chat"
                    aria-label={`Borrar chat ${s.label || s.session_id}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSessionDeleteError(null);
                      setPendingSessionDelete({
                        sessionId: s.session_id,
                        label: s.label || s.session_id,
                      });
                    }}
                    disabled={pending || isDeleting}
                  >
                    🗑️
                  </button>
                </div>
              );
            })}
          </div>
          {sessionDeleteError && (
            <div className="sidebar-delete-error">
              {sessionDeleteError}
            </div>
          )}
        </div>
      </aside>

      <ConfirmModal
        open={pendingSessionDelete !== null}
        variant="danger"
        title="Borrar este chat"
        message={
          pendingSessionDelete ? (
            <>
              <p style={{ margin: 0 }}>
                Vas a borrar <strong>{pendingSessionDelete.label}</strong>.
              </p>
              <p style={{ marginTop: 10, marginBottom: 0 }}>
                Esto elimina toda la conversación (preguntas y respuestas)
                del store local de memflow.
              </p>
              <p style={{ marginTop: 10, marginBottom: 0, opacity: 0.85 }}>
                Solo es reversible restaurando un backup de memflow.
              </p>
            </>
          ) : (
            ""
          )
        }
        confirmLabel="Borrar chat"
        cancelLabel="Cancelar"
        busy={
          pendingSessionDelete
            ? deletingSessionId === pendingSessionDelete.sessionId
            : false
        }
        onConfirm={() => {
          const pending = pendingSessionDelete;
          if (!pending) return;
          setDeletingSessionId(pending.sessionId);
          setSessionDeleteError(null);
          deleteSession({ sessionId: pending.sessionId })
            .then(() => {
              setRecentSessions((prev) =>
                prev.filter((row) => row.session_id !== pending.sessionId),
              );
              if (chatSessionId === pending.sessionId) {
                setChatSessionId("");
                setTurns([]);
              }
              setPendingSessionDelete(null);
            })
            .catch((err: unknown) => {
              setSessionDeleteError(
                err instanceof Error ? err.message : String(err),
              );
            })
            .finally(() => {
              setDeletingSessionId("");
            });
        }}
        onCancel={() => setPendingSessionDelete(null)}
      />

      <ConfirmModal
        open={pendingDeleteAll}
        variant="danger"
        title="Borrar todos los chats"
        message={
          <>
            <p style={{ margin: 0 }}>
              Vas a borrar <strong>{sidebarSessions.length}</strong> chat
              {sidebarSessions.length === 1 ? "" : "s"} recientes.
            </p>
            <p style={{ marginTop: 10, marginBottom: 0 }}>
              Se elimina toda la conversación (preguntas y respuestas) del
              store local de memflow para cada sesión listada.
            </p>
            <p style={{ marginTop: 10, marginBottom: 0, opacity: 0.85 }}>
              Solo es reversible restaurando un backup de memflow.
            </p>
          </>
        }
        confirmLabel="Borrar todo"
        cancelLabel="Cancelar"
        busy={deletingAll}
        onConfirm={() => {
          setDeletingAll(true);
          setSessionDeleteError(null);
          deleteAllSessions()
            .then(() => {
              setRecentSessions([]);
              setChatSessionId("");
              setTurns([]);
              setPendingDeleteAll(false);
            })
            .catch((err: unknown) => {
              setSessionDeleteError(
                err instanceof Error ? err.message : String(err),
              );
            })
            .finally(() => {
              setDeletingAll(false);
            });
        }}
        onCancel={() => setPendingDeleteAll(false)}
      />

      <main className="app">
        <header className="app-header">
          <div>
            <h1>Synapse · Chat</h1>
            <div className="subtitle">consulta federada — rutea a Memflow y Memo</div>
          </div>
        </header>

        <div className={`stage ${started ? "started" : "empty"}`}>
          <div className="hero">
            <h2>¿Qué querés saber?</h2>
            <p>Synapse decide a qué memoria ir y devuelve evidencia con la ruta tomada.</p>
          </div>

          <div className="chat-scroll" ref={scrollRef}>
            {turns.map((turn) => {
              if (turn.kind === "user") {
                return (
                  <div className="turn user" key={turn.id} id={`turn-${turn.id}`}>
                    <div className="bubble">{turn.text}</div>
                  </div>
                );
              }
              if (turn.kind === "error") {
                return (
                  <div className="turn error" key={turn.id} id={`turn-${turn.id}`}>
                    <div className="bubble">
                      <strong>Error:</strong> {turn.message}
                    </div>
                  </div>
                );
              }
              return (
                <div className="turn synapse" key={turn.id} id={`turn-${turn.id}`}>
                  <SynapseTurn
                    response={turn.response}
                    historical={turn.historical}
                    feedback={turn.feedback}
                    feedbackPending={turn.feedbackPending}
                    feedbackError={turn.feedbackError}
                    onFeedback={
                      turn.historical
                        ? undefined
                        : (rating) => handleFeedback(turn.id, rating)
                    }
                  />
                </div>
              );
            })}

            {pending && (
              <div className="turn synapse">
                <div className="bubble">
                  <div className="thinking" style={{ marginBottom: 10 }}>
                    <span className="dot" />
                    <span className="dot" />
                    <span className="dot" />
                    <span>{currentStage || "Synapse consultando memorias y sintetizando…"}</span>
                  </div>
                  <div className="skeleton">
                    <div className="skeleton-line" style={{ width: "92%" }} />
                    <div className="skeleton-line" style={{ width: "78%" }} />
                    <div className="skeleton-line" style={{ width: "85%" }} />
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="composer-wrap">
            <Composer onSubmit={send} disabled={pending} onActivity={setHasInput} />
            {!started && (
              <div className="suggestions">
                {suggestions.map((s, i) => (
                  <button
                    key={`${s.query}-${i}`}
                    onClick={() => send(s.query)}
                    disabled={pending}
                    title={s.query}
                    data-kind={s.kind ?? "static"}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            )}
            <div className="composer-hint">
              Enter envía · Shift+Enter nueva línea
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
