import type {
  AskResponse,
  AskSource,
  CaptureResult,
  ChatHistoryTurn,
  FeedbackRating,
  FeedbackReceipt,
  InsightCandidate,
  SessionHistory,
  SessionListItem,
  SuggestionChip,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function ask(
  query: string,
  history: ChatHistoryTurn[] = [],
  k = 7,
  chatSessionId = "",
): Promise<AskResponse> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      q: query,
      history,
      k,
      chat_session_id: chatSessionId,
    }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, `ask failed: ${res.status} ${body}`.trim());
  }
  return (await res.json()) as AskResponse;
}

export type StageEventName =
  | "memo_retrieval"
  | "rerank"
  | "memo_synthesis"
  | "self_synthesis"
  | "multi_source_synthesis"
  | "streaming";

export type StreamEvent =
  | { type: "context"; chat_session_id?: string; trace_id?: string; sources?: unknown[]; understanding?: unknown; pipeline_partial?: unknown[]; route_decision?: unknown; packet_status?: string; federation_status?: string; semantic?: Record<string, unknown>; errors?: Record<string, string> }
  | { type: "token"; text: string }
  | { type: "stage"; name: StageEventName; phase: "start" | "done"; ms?: number; mode?: string }
  | { type: "done"; answer?: string | null; citations?: unknown[]; synthesis_status?: string; synthesis_source?: string; synthesis_error?: string; sources?: unknown[]; pipeline_trace?: unknown[]; total_ms?: number; chat_session_id?: string; trace_id?: string; generated_at?: string; federation_status?: string; semantic?: Record<string, unknown>; errors?: Record<string, string> }
  | { type: "insight_proposal"; candidate: InsightCandidate }
  | { type: "error"; message: string };

export async function askStream(
  query: string,
  history: ChatHistoryTurn[] = [],
  k = 7,
  chatSessionId = "",
  onEvent?: (event: StreamEvent) => void,
): Promise<void> {
  const res = await fetch("/api/ask/stream", {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      q: query,
      history,
      k,
      chat_session_id: chatSessionId,
    }),
  });
  if (!res.ok || !res.body) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, `askStream failed: ${res.status} ${body}`.trim());
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        try {
          const event = JSON.parse(payload) as StreamEvent;
          onEvent?.(event);
        } catch {
          // skip malformed event
        }
      }
    }
  }
}

export type SubmitFeedbackArgs = {
  turnId: string;
  rating: FeedbackRating;
  query: string;
  answer: string;
  sources: AskSource[];
  traceId?: string;
  chatSessionId?: string;
  correctionText?: string;
};

export async function submitFeedback(args: SubmitFeedbackArgs): Promise<FeedbackReceipt> {
  const res = await fetch("/api/feedback", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({
      trace_id: args.traceId ?? "",
      chat_session_id: args.chatSessionId ?? "",
      turn_id: args.turnId,
      query: args.query,
      answer: args.answer,
      sources: args.sources,
      rating: args.rating,
      correction_text: args.correctionText ?? "",
    }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, `submitFeedback failed: ${res.status} ${body}`.trim());
  }
  return (await res.json()) as FeedbackReceipt;
}

export type SourceFeedbackRating = "up" | "down";

export type SourceFeedbackReceipt = {
  ok: boolean;
  feedback_id?: string;
  source_id?: string;
  query_text?: string;
  rating?: SourceFeedbackRating;
  error?: string;
};

export async function submitSourceFeedback(args: {
  sourceId: string;
  query: string;
  rating: SourceFeedbackRating;
}): Promise<SourceFeedbackReceipt> {
  const res = await fetch("/api/feedback/source", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({
      source_id: args.sourceId,
      query: args.query,
      rating: args.rating,
    }),
  });
  const data = (await res.json().catch(() => ({}))) as SourceFeedbackReceipt;
  if (!res.ok) {
    throw new ApiError(res.status, data.error ?? `submitSourceFeedback failed: ${res.status}`);
  }
  return data;
}

export type DeleteMemoryReceipt = {
  ok: boolean;
  memory_id?: string;
  deleted_file?: string | null;
  is_vault_source?: boolean;
  excluded_from_ingest?: string;
  excluded_vault?: string;
  vault?: string;
  chunk_count?: number;
  message?: string;
  archived?: boolean;
  archive_to?: string;
  archive_rel?: string;
  archive_skipped?: string;
  error?: string;
};

export async function deleteMemory(args: {
  memoryId: string;
  traceId?: string;
}): Promise<DeleteMemoryReceipt> {
  const res = await fetch("/api/memory/delete", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({
      memory_id: args.memoryId,
      trace_id: args.traceId ?? "",
    }),
  });
  const data = (await res.json().catch(() => ({}))) as DeleteMemoryReceipt;
  if (!res.ok) {
    throw new ApiError(res.status, data.error ?? `deleteMemory failed: ${res.status}`);
  }
  return data;
}

export type DeleteSessionReceipt = {
  ok: boolean;
  session_id?: string;
  deleted_dir?: string;
  deleted_files?: number;
  error?: string;
};

export async function deleteSession(args: {
  sessionId: string;
}): Promise<DeleteSessionReceipt> {
  const res = await fetch("/api/sessions/delete", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: args.sessionId }),
  });
  const data = (await res.json().catch(() => ({}))) as DeleteSessionReceipt;
  if (!res.ok) {
    throw new ApiError(res.status, data.error ?? `deleteSession failed: ${res.status}`);
  }
  return data;
}

export type DeleteAllSessionsReceipt = {
  ok: boolean;
  deleted_sessions?: number;
  deleted_files?: number;
  error?: string;
};

export async function deleteAllSessions(): Promise<DeleteAllSessionsReceipt> {
  const res = await fetch("/api/sessions/delete-all", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const data = (await res.json().catch(() => ({}))) as DeleteAllSessionsReceipt;
  if (!res.ok) {
    throw new ApiError(res.status, data.error ?? `deleteAllSessions failed: ${res.status}`);
  }
  return data;
}

export async function captureInsight(candidate: InsightCandidate): Promise<CaptureResult> {
  const res = await fetch("/api/insight/capture", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({ candidate }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, `captureInsight failed: ${res.status} ${body}`.trim());
  }
  return (await res.json()) as CaptureResult;
}

export async function fetchSuggestions(limit = 6): Promise<SuggestionChip[]> {
  try {
    const res = await fetch(`/api/suggestions?limit=${limit}`, {
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return [];
    const body = (await res.json()) as { chips?: SuggestionChip[] };
    return body.chips ?? [];
  } catch {
    return [];
  }
}

export async function fetchSessions(limit = 10): Promise<SessionListItem[]> {
  try {
    const res = await fetch(`/api/sessions?limit=${limit}`, {
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return [];
    const body = (await res.json()) as { sessions?: SessionListItem[] };
    return body.sessions ?? [];
  } catch {
    return [];
  }
}

export async function loadSession(id: string): Promise<SessionHistory | null> {
  try {
    const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`, {
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return null;
    return (await res.json()) as SessionHistory;
  } catch {
    return null;
  }
}
