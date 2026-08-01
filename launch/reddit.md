# Reddit

## r/LocalLLaMA — best fit. Post ~2h after HN, same day.

**Title:** I built a fully local memory layer for coding agents — MLX embeddings, sqlite-vec, Markdown source of truth, no Ollama

**Body:** rewrite the HN comment with MORE technical detail and LESS narrative:
- Embedding model options (Qwen3-Embedding 0.6B / 4B / 8B, 4-bit) and the recall-quality tradeoff
- Why in-process MLX beat an Ollama round-trip — include real latency numbers
- The <200ms warm recall daemon architecture
- RRF fusion weights and how `MEMO_RECALL_MIN_SIM` was tuned
- Link the eval harness (`memo eval recall --gate`) — rigor earns credibility here

Include the full comparison table and a terminal screenshot, not just the GIF.
Write like a build log to a friend. This sub punishes anything that smells like marketing.

## r/ClaudeAI + r/mcp — workflow angle, not architecture

**Title:** Made Claude Code remember things between sessions (and cut ~14k tokens of MCP overhead while doing it)

Lead entirely with token economics and the session-start briefing. Before/after screenshots of a fresh session that already knows the stack conventions. Short post. Link at the bottom.
