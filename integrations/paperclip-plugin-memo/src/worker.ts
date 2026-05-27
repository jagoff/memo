import { definePlugin, runWorker, type ToolResult } from "@paperclipai/plugin-sdk";
import { execFile } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const pExecFile = promisify(execFile);

interface MemoConfig {
  memoBinary: string;
  defaultSearchLimit: number;
  defaultSearchMode: "hybrid" | "vec" | "bm25";
}

const DEFAULTS: MemoConfig = {
  memoBinary: "memo",
  defaultSearchLimit: 5,
  defaultSearchMode: "hybrid",
};

async function resolveConfig(
  getConfig: () => Promise<Record<string, unknown>>,
): Promise<MemoConfig> {
  const raw = await getConfig().catch(() => ({}) as Record<string, unknown>);
  return {
    memoBinary:
      typeof raw.memoBinary === "string" && raw.memoBinary.trim()
        ? (raw.memoBinary as string)
        : DEFAULTS.memoBinary,
    defaultSearchLimit:
      typeof raw.defaultSearchLimit === "number" && raw.defaultSearchLimit > 0
        ? (raw.defaultSearchLimit as number)
        : DEFAULTS.defaultSearchLimit,
    defaultSearchMode:
      raw.defaultSearchMode === "vec" ||
      raw.defaultSearchMode === "bm25" ||
      raw.defaultSearchMode === "hybrid"
        ? (raw.defaultSearchMode as MemoConfig["defaultSearchMode"])
        : DEFAULTS.defaultSearchMode,
  };
}

async function runMemo(
  binary: string,
  args: string[],
  opts: { stdin?: string } = {},
): Promise<unknown> {
  // execFile + array args = no shell, no injection.
  const child = pExecFile(binary, args, {
    maxBuffer: 16 * 1024 * 1024,
    timeout: 60_000,
    encoding: "utf8",
    env: { ...process.env, NO_COLOR: "1", FORCE_COLOR: "0" },
  });
  if (opts.stdin !== undefined && child.child.stdin) {
    child.child.stdin.write(opts.stdin);
    child.child.stdin.end();
  }
  const { stdout } = await child;
  const trimmed = stdout.trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return { raw: trimmed };
  }
}

function asString(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}

function asNumber(v: unknown, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

function asStringArray(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

function asObject(v: unknown): Record<string, unknown> | null {
  if (v === null || typeof v !== "object" || Array.isArray(v)) return null;
  return v as Record<string, unknown>;
}

function asObjectArray(v: unknown): Record<string, unknown>[] {
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is Record<string, unknown> => asObject(x) !== null);
}

async function withJsonTempFiles<T>(
  files: Record<string, unknown>,
  fn: (paths: Record<string, string>) => Promise<T>,
): Promise<T> {
  const entries = Object.entries(files);
  if (entries.length === 0) return fn({});

  const dir = await mkdtemp(join(tmpdir(), "memo-paperclip-"));
  try {
    const paths: Record<string, string> = {};
    for (const [name, value] of entries) {
      const path = join(dir, `${name}.json`);
      await writeFile(path, JSON.stringify(value), "utf8");
      paths[name] = path;
    }
    return await fn(paths);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

const plugin = definePlugin({
  async setup(ctx) {
    ctx.logger.info("memo plugin: setup");

    ctx.data.register("stats", async () => {
      const cfg = await resolveConfig(() => ctx.config.get());
      try {
        const data = (await runMemo(cfg.memoBinary, ["stats", "--json"])) as Record<
          string,
          unknown
        > | null;
        return data ?? { error: "empty stats output" };
      } catch (err) {
        return { error: (err as Error).message };
      }
    });

    ctx.tools.register(
      "memo_search",
      {
        displayName: "Memo: search memories",
        description:
          "Top-k search over the operator's memo store. Returns id/title/type/tags/score/snippet.",
        parametersSchema: {
          type: "object",
          properties: {
            query: { type: "string" },
            limit: { type: "number" },
            mode: { type: "string", enum: ["hybrid", "vec", "bm25"] },
            type: { type: "string" },
          },
          required: ["query"],
        },
      },
      async (params): Promise<ToolResult> => {
        const cfg = await resolveConfig(() => ctx.config.get());
        const p = (params ?? {}) as Record<string, unknown>;
        const query = asString(p.query).trim();
        if (!query) return { error: "query is required" };
        const limit = asNumber(p.limit, cfg.defaultSearchLimit);
        const mode = (asString(p.mode) || cfg.defaultSearchMode) as MemoConfig["defaultSearchMode"];
        const args = ["search", query, "--json", "--limit", String(limit), "--mode", mode];
        const typeFilter = asString(p.type);
        if (typeFilter) args.push("--type", typeFilter);
        try {
          const data = await runMemo(cfg.memoBinary, args);
          return { content: `memo: ${query}`, data };
        } catch (err) {
          return { error: `memo search failed: ${(err as Error).message}` };
        }
      },
    );

    ctx.tools.register(
      "memo_save",
      {
        displayName: "Memo: save memory",
        description: "Persist a new memory to the operator's vault.",
        parametersSchema: {
          type: "object",
          properties: {
            content: { type: "string" },
            title: { type: "string" },
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
            },
            tags: { type: "array", items: { type: "string" } },
            autoDerive: { type: "boolean" },
          },
          required: ["content"],
        },
      },
      async (params): Promise<ToolResult> => {
        const cfg = await resolveConfig(() => ctx.config.get());
        const p = (params ?? {}) as Record<string, unknown>;
        const content = asString(p.content);
        if (!content) return { error: "content is required" };
        const args = ["save", "-", "--json"];
        const title = asString(p.title);
        if (title) args.push("--title", title);
        const type = asString(p.type);
        if (type) args.push("--type", type);
        for (const tag of asStringArray(p.tags)) args.push("--tag", tag);
        if (p.autoDerive === true) args.push("--auto-derive");
        try {
          const data = await runMemo(cfg.memoBinary, args, { stdin: content });
          return { content: "memory saved", data };
        } catch (err) {
          return { error: `memo save failed: ${(err as Error).message}` };
        }
      },
    );

    ctx.tools.register(
      "memo_list",
      {
        displayName: "Memo: list recent",
        description: "Most recent memories by `updated` desc.",
        parametersSchema: {
          type: "object",
          properties: {
            limit: { type: "number" },
            type: { type: "string" },
          },
        },
      },
      async (params): Promise<ToolResult> => {
        const cfg = await resolveConfig(() => ctx.config.get());
        const p = (params ?? {}) as Record<string, unknown>;
        const args = ["list", "--json", "--limit", String(asNumber(p.limit, 20))];
        const type = asString(p.type);
        if (type) args.push("--type", type);
        try {
          const data = await runMemo(cfg.memoBinary, args);
          return { content: "recent memories", data };
        } catch (err) {
          return { error: `memo list failed: ${(err as Error).message}` };
        }
      },
    );

    ctx.tools.register(
      "memo_get",
      {
        displayName: "Memo: fetch by id",
        description: "Fetch one full memory by id (prefix ≥4 chars OK).",
        parametersSchema: {
          type: "object",
          properties: { id: { type: "string" } },
          required: ["id"],
        },
      },
      async (params): Promise<ToolResult> => {
        const cfg = await resolveConfig(() => ctx.config.get());
        const p = (params ?? {}) as Record<string, unknown>;
        const id = asString(p.id).trim();
        if (!id) return { error: "id is required" };
        try {
          const data = await runMemo(cfg.memoBinary, ["get", id, "--json"]);
          return { content: `memo: ${id}`, data };
        } catch (err) {
          return { error: `memo get failed: ${(err as Error).message}` };
        }
      },
    );

    ctx.tools.register(
      "memo_ask",
      {
        displayName: "Memo: chat RAG ask",
        description:
          "Chat-shaped RAG over memorias. Returns the memo.chat_ask.v2 envelope.",
        parametersSchema: {
          type: "object",
          properties: {
            question: { type: "string" },
            k: { type: "number" },
            type: { type: "string" },
            history: {
              type: "array",
              items: { type: "object" },
              description: "Optional chat history turns as {role,text} or {role,content}.",
            },
            context: {
              type: "object",
              description: "Optional caller context to include in the retrieval question.",
            },
          },
          required: ["question"],
        },
      },
      async (params): Promise<ToolResult> => {
        const cfg = await resolveConfig(() => ctx.config.get());
        const p = (params ?? {}) as Record<string, unknown>;
        const question = asString(p.question).trim();
        if (!question) return { error: "question is required" };
        const args = ["chat-ask", question, "--json", "--k", String(asNumber(p.k, 7))];
        const type = asString(p.type);
        if (type) args.push("--type", type);
        const tempInputs: Record<string, unknown> = {};
        const history = asObjectArray(p.history);
        if (history.length > 0) tempInputs.history = history;
        const context = asObject(p.context);
        if (context !== null) tempInputs.context = context;
        try {
          const data = await withJsonTempFiles(tempInputs, async (paths) => {
            if (paths.history) args.push("--history-json", paths.history);
            if (paths.context) args.push("--context-json", paths.context);
            return await runMemo(cfg.memoBinary, args);
          });
          return { content: `memo ask: ${question}`, data };
        } catch (err) {
          return { error: `memo ask failed: ${(err as Error).message}` };
        }
      },
    );
  },

  async onHealth() {
    return { status: "ok", message: "memo plugin worker running" };
  },
});

export default plugin;
runWorker(plugin, import.meta.url);
