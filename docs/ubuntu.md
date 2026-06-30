# memo on Ubuntu / Linux (CPU backend)

memo's primary runtime is **MLX**, which ships wheels for Apple Silicon only.
On Linux (Ubuntu and friends) and Intel macs, memo runs with a **CPU
`sentence-transformers` embedder backend** instead. This is a **standalone**
install: semantic search, recall, and save all work — but it is not a
vector-coherent peer of a Mac in the trinity (see *Trade-offs*).

## One-command install

```bash
pipx install "mlx-memo[cpu]"
# or, from a checkout:
scripts/install-ubuntu.sh
```

`[cpu]` pulls `sentence-transformers` (+ CPU torch). The first search downloads
the embedding model (`Qwen/Qwen3-Embedding-0.6B`, ~1.2 GB) to the HuggingFace
cache. Python ≥ 3.13 is required (`pipx` will use a managed interpreter if your
system Python is older — or install `uv` and use `uv tool install "mlx-memo[cpu]"`).

`uv` alternative:

```bash
uv tool install "mlx-memo[cpu]"
```

## What works / what doesn't

| Feature | Linux (CPU) |
|---|---|
| `memo save` / markdown store / git sync | ✅ |
| Semantic search + recall (`memo search`, recall hook, MCP `memo_search`) | ✅ (CPU embeds) |
| BM25 / FTS5 keyword search, Spanish folding | ✅ |
| Knowledge graph, temporal, consolidation, dedup | ✅ |
| **Reranker** (cross-encoder) | ❌ off by default (MLX-only); search still ranks via hybrid fusion |
| **LLM features** — `memo ask`, `synthesize`, `dream`, chat | ❌ require MLX; raise a clear error |
| Warm recall daemon (`recall.sock`) | ❌ MLX-only; in-process embed is used instead |

Search/recall are the load-bearing paths and they work. The LLM-backed
verbs degrade with an explicit message rather than a cryptic import error.

## How backend selection works

`MEMO_EMBEDDER_BACKEND` (config `embedder_backend`, default `auto`):

- `auto` — MLX when the runtime is importable (Apple Silicon), else the CPU
  backend. No configuration needed on Ubuntu.
- `mlx` — force MLX.
- `st` — force the CPU `sentence-transformers` backend (useful to test on a Mac).

The CPU model is `MEMO_ST_EMBEDDER_MODEL` (default `Qwen/Qwen3-Embedding-0.6B`,
**1024-dim** — same family and dimensionality as the default MLX quant, so the
vec0 schema and `embedder_dims=1024` are unchanged, and the asymmetric
query-instruction prefix is preserved).

> **Dims must match the model.** If your `embedder_dims` is not 1024 (e.g. you
> were on the 4B / 2560-dim profile), point `MEMO_ST_EMBEDDER_MODEL` at a model
> of that dimensionality (e.g. `Qwen/Qwen3-Embedding-4B`) or reset to the 1024
> profile and `memo reindex --rebuild`. STEmbedder fails fast with the exact
> instruction if they disagree.

GPU: set `device="cuda"` is not wired to an env flag yet; the default is CPU.

## Trade-offs (read before syncing across machines)

- **Standalone corpus.** MLX-4bit and ST-fp embeddings occupy slightly different
  regions of the vector space. An Ubuntu node and an Apple-Silicon node produce
  **incompatible vectors** for the same text, so cosine across them is
  meaningless. Do **not** point an Ubuntu memo at a Mac's `memo-sync` remote and
  expect coherent cross-machine recall. Keep the Linux box as its own store.
- The recorded `embedder_model` in the index metadata is the configured profile
  id; the vectors are produced by the ST model. Harmless for a standalone box.

## Verify

```bash
memo doctor
memo config validate
memo save "ubuntu smoke test" --type note
memo search "ubuntu smoke"
```
