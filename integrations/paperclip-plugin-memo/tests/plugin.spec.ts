import { describe, expect, it } from "vitest";
import { createTestHarness } from "@paperclipai/plugin-sdk/testing";
import manifest from "../src/manifest.js";
import plugin from "../src/worker.js";

describe("memo paperclip plugin", () => {
  it("registers tools + stats data handler and shells out to the configured binary", async () => {
    const harness = createTestHarness({
      manifest,
      capabilities: [...manifest.capabilities],
    });
    harness.setConfig({ memoBinary: "/bin/echo", defaultSearchLimit: 3, defaultSearchMode: "hybrid" });

    await plugin.definition.setup(harness.ctx);

    // stats handler shells out and wraps the non-JSON echo output as { raw: ... }.
    const stats = (await harness.getData("stats")) as Record<string, unknown>;
    expect(stats).toHaveProperty("raw");
    expect(typeof stats.raw).toBe("string");
    expect(stats.raw).toContain("stats --json");

    // memo_search forwards the query + flags to the binary.
    const search = await harness.executeTool("memo_search", { query: "astor terapia" });
    expect(search).toHaveProperty("data");
    const searchData = (search as { data: { raw: string } }).data;
    expect(searchData.raw).toContain("search astor terapia --json --limit 3 --mode hybrid");

    // memo_save reads body via stdin (no body in argv -> just `save - --json`).
    const saved = await harness.executeTool("memo_save", { content: "remember X", title: "t" });
    expect(saved).toHaveProperty("data");
    const savedData = (saved as { data: { raw: string } }).data;
    expect(savedData.raw).toContain("save - --json --title t");
  });

  it("rejects empty arguments", async () => {
    const harness = createTestHarness({
      manifest,
      capabilities: [...manifest.capabilities],
    });
    await plugin.definition.setup(harness.ctx);

    const r = await harness.executeTool("memo_search", { query: "" });
    expect(r).toEqual({ error: "query is required" });

    const s = await harness.executeTool("memo_save", { content: "" });
    expect(s).toEqual({ error: "content is required" });

    const g = await harness.executeTool("memo_get", { id: "" });
    expect(g).toEqual({ error: "id is required" });
  });
});
