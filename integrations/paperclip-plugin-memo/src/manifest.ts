import type { PaperclipPluginManifestV1 } from "@paperclipai/plugin-sdk";

const PLUGIN_ID = "memo.paperclip-plugin-memo";
const PLUGIN_VERSION = "0.1.0";

const manifest: PaperclipPluginManifestV1 = {
  id: PLUGIN_ID,
  apiVersion: 1,
  version: PLUGIN_VERSION,
  displayName: "Memo — local MCP memory",
  description:
    "Bridges the local `memo` CLI (Apple-Silicon MLX-native MCP memory backed by an Obsidian vault) into Paperclip as agent tools. Lets agents recall, save, list, and ask over the operator's persistent memory store.",
  author: "Fernando Ferrari",
  categories: ["connector", "automation"],
  capabilities: [
    "agent.tools.register",
    "instance.settings.register",
    "ui.dashboardWidget.register",
  ],
  entrypoints: {
    worker: "./dist/worker.js",
    ui: "./dist/ui",
  },
  ui: {
    slots: [
      {
        type: "dashboardWidget",
        id: "memo-stats-widget",
        displayName: "Memo (local MCP memory)",
        exportName: "DashboardWidget",
      },
    ],
  },
  instanceConfigSchema: {
    type: "object",
    properties: {
      memoBinary: {
        type: "string",
        title: "memo binary path",
        description:
          "Absolute path to the `memo` executable. Defaults to `memo` on $PATH.",
        default: "memo",
      },
      defaultSearchLimit: {
        type: "number",
        title: "Default search limit",
        default: 5,
      },
      defaultSearchMode: {
        type: "string",
        title: "Default search mode",
        enum: ["hybrid", "vec", "bm25"],
        default: "hybrid",
      },
    },
  },
  tools: [
    {
      name: "memo_search",
      displayName: "Memo: search memories",
      description:
        "Top-k search over the operator's memo store. Hybrid (semantic + keyword) by default. Returns id, title, type, tags, score, and a body snippet.",
      parametersSchema: {
        type: "object",
        properties: {
          query: { type: "string", description: "Natural-language query." },
          limit: { type: "number", description: "Top-K. Default 5." },
          mode: {
            type: "string",
            enum: ["hybrid", "vec", "bm25"],
            description: "Search mode. Default hybrid.",
          },
          type: {
            type: "string",
            description:
              "Filter by record type (decision|fact|bug|feedback|preference|note|manual).",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "memo_save",
      displayName: "Memo: save memory",
      description:
        "Persist a new memory to the operator's vault. The body should be self-contained — title/type/tags are auto-derived if omitted.",
      parametersSchema: {
        type: "object",
        properties: {
          content: {
            type: "string",
            description: "Body of the memory. Markdown OK.",
          },
          title: { type: "string", description: "Optional short title." },
          type: {
            type: "string",
            enum: [
              "decision",
              "fact",
              "bug",
              "feedback",
              "preference",
              "note",
              "manual",
            ],
            description: "Record type. Default: note.",
          },
          tags: {
            type: "array",
            items: { type: "string" },
            description: "Optional tags. Lower-cased + de-duplicated by memo.",
          },
          autoDerive: {
            type: "boolean",
            description:
              "If true, ask the local Qwen2.5-3B helper to fill missing title/type/tags.",
          },
        },
        required: ["content"],
      },
    },
    {
      name: "memo_list",
      displayName: "Memo: list recent memories",
      description: "Most recent memories by `updated` desc.",
      parametersSchema: {
        type: "object",
        properties: {
          limit: { type: "number", description: "Default 20." },
          type: { type: "string", description: "Filter by record type." },
        },
      },
    },
    {
      name: "memo_get",
      displayName: "Memo: fetch memory by id",
      description: "Fetch one full memory by id (or git-style id prefix ≥4).",
      parametersSchema: {
        type: "object",
        properties: {
          id: { type: "string", description: "Memory id or prefix." },
        },
        required: ["id"],
      },
    },
    {
      name: "memo_ask",
      displayName: "Memo: chat RAG over memories",
      description:
        "Chat-shaped RAG over the memory archive. Returns the `memo.chat_ask.v2` envelope with answer, citations, retrieval trace, and synthesis status.",
      parametersSchema: {
        type: "object",
        properties: {
          question: { type: "string" },
          k: {
            type: "number",
            description: "Top-K memorias to feed the LLM. Default 7.",
          },
          type: { type: "string", description: "Restrict to one record type." },
          history: {
            type: "array",
            items: { type: "object" },
            description:
              "Optional chat history turns as {role,text} or {role,content}.",
          },
          context: {
            type: "object",
            description:
              "Optional caller context included in the retrieval question.",
          },
        },
        required: ["question"],
      },
    },
  ],
};

export default manifest;
