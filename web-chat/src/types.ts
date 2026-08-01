export type EvidenceRef = {
  uri?: string;
  title?: string;
  snippet?: string;
  source?: string;
  score?: number;
  metadata?: Record<string, unknown>;
};

export type RouteDecision = {
  schema?: string;
  intent?: string;
  target?: string;
  reason?: string;
  trace_id?: string;
  read_backends?: string[];
  write_target?: string | null;
  kind?: string;
  confidence?: number;
  matched_signals?: Record<string, string[]>;
  mode?: "read" | "write";
  score_breakdown?: Record<string, unknown>;
};

export type RealityConflict = {
  conflict_id?: string;
  kind?: string;
  status?: string;
  freeze_write?: boolean;
  summary?: string;
};

export type AskSource = {
  source?: string;
  id?: string;
  id_short?: string;
  title?: string;
  type?: string;
  score?: number;
  snippet?: string;
  path?: string;
  repo_name?: string;
  obsidian_vault?: string;
  locator?: string;
  rrf?: number;
  boost?: number;
  rerank_score?: number | null;
  final_score?: number;
  sources_present?: string[];
  metadata?: Record<string, unknown>;
  rank_in_source?: number;
};

export type ChatHistoryTurn = {
  role: "user" | "assistant";
  text: string;
};

export type SuggestionChip = {
  label: string;
  query: string;
  kind?: string;
  type?: string;
};

export type AskCitation = {
  n: number;
  id: string;
  source: string;
  title: string;
  metadata?: Record<string, unknown>;
};

export type PipelineStage = {
  stage: string;
  ms: number;
  [key: string]: unknown;
};

export type RewrittenQuery = {
  schema?: string;
  original: string;
  expanded: string;
  applied: boolean;
  pronouns_resolved?: string[];
  entities_added?: string[];
  synonyms_added?: string[];
};

export type Understanding = {
  intent?: string;
  record_type_filter?: string | null;
  recency_days?: number | null;
  entities_in_query?: string[];
  expanded_terms?: string[];
  keyword_tokens?: string[];
  // Synapse pre-Memo rewrite (opt-in via SYNAPSE_QUERY_REWRITE). Only
  // present when at least one rewrite rule fired.
  query_rewrite?: RewrittenQuery;
  // Memo engine metadata exposed by the federator.
  owner?: string;
  engine_schema?: string;
  history_turns_used?: number;
};

export type AskResponse = {
  schema: string;
  generated_at: string;
  query: string;
  trace_id: string;
  chat_session_id?: string;
  packet_status: string;
  federation_status?: string;
  route_decision: RouteDecision | null;
  answer: string | null;
  sources: AskSource[];
  citations: AskCitation[];
  synthesis_status: "ok" | "fallback" | "unavailable" | "error";
  synthesis_source?: string;
  synthesis_error?: string;
  understanding: Understanding;
  pipeline_trace: PipelineStage[];
  total_ms: number;
  reality_conflicts: RealityConflict[];
  notes: string[];
  semantic?: Record<string, unknown>;
  errors?: Record<string, string>;
  insight_proposal?: InsightCandidate | null;
};

export type FeedbackRating = "up" | "down";

export type FeedbackReceipt = {
  ok: boolean;
  feedback_id?: string;
  rating?: FeedbackRating;
  ledger_entry_id?: string;
  corpus_row_id?: string;
  error?: string;
};

export type ChatTurn =
  | { kind: "user"; id: string; text: string; at: string }
  | {
      kind: "synapse";
      id: string;
      at: string;
      response: AskResponse;
      latencyMs: number;
      historical?: boolean;
      feedback?: FeedbackRating;
      feedbackPending?: boolean;
      feedbackError?: string;
    }
  | { kind: "error"; id: string; at: string; message: string };

export type InsightCandidate = {
  schema?: string;
  title: string;
  body: string;
  tags: string[];
  confidence: number;       // 0-1
  score: number;            // 0-100
  suggested_type: "decision" | "note" | "fact";
  evidence_uris: string[];
  rationale: string;
  chat_session_id?: string;
  chat_turn_id?: string;
  generated_at?: string;
};

export type CaptureResult = {
  schema?: string;
  ok: boolean;
  memoria_id?: string;
  uri?: string;
  error?: string;
};

export type SessionListItem = {
  session_id: string;
  first_ts?: string | null;
  last_ts?: string | null;
  turn_count: number;
  label: string;
};

export type SessionHistoryTurn = {
  role: "user" | "assistant";
  text: string;
  at?: string | null;
};

export type SessionHistory = {
  schema?: string;
  session_id: string;
  created_at?: string | null;
  updated_at?: string | null;
  turns: SessionHistoryTurn[];
};
