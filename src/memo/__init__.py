"""memo — local MCP memory backed by Obsidian vault, MLX-native.

100% local stack — zero Ollama, zero cloud APIs:

- LLM: `mlx-lm` running Qwen2.5-Instruct quantized models on Apple
  Silicon Metal (in-process, no daemon).
- Embedder: `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` with
  last-token pooling + L2 normalize (1024-dim).
- Vector store: `sqlite-vec` (single file, no daemon, no Qdrant).
- Storage of record: markdown files under
  `<vault>/04-Archive/99-obsidian-system/99-AI/memory/`.
- MCP server: `fastmcp` with tools `memory_save`, `memory_search`,
  `memory_list`, `memory_get`, `memory_update`, `memory_delete`.

Public API:

    from memo import Memory, Config

    cfg = Config.from_env()
    mem = Memory(cfg)
    mem.save(content="...", title="...", tags=["x", "y"])
    hits = mem.search("query", limit=10)
"""

from memo.config import Config
from memo.memory import Memory, MemoryRecord

__version__ = "0.1.0"

__all__ = ["Config", "Memory", "MemoryRecord", "__version__"]
