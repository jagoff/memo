# memo in Docker (CPU backend, cross-platform)

Docker image runs memo's **CPU-only backend** — ideal for Linux, testing, or cross-platform deployment without native build. Includes embeddings (quantized, CPU-optimized); **no MLX**, so `ask`, `synthesize`, `dream`, and reranking are disabled with clear error messages.

> For full memo (reranking + LLM verbs), use native install on Apple Silicon Mac:
> ```bash
> curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash
> ```

## Quick start (10 seconds)

```bash
docker run --rm ghcr.io/jagoff/memo:latest memo doctor
```

## Persistent storage

```bash
docker volume create memo-data

# Save a memory
docker run --rm -v memo-data:/data ghcr.io/jagoff/memo:latest \
  memo save 'we use Postgres, not Mongo' --title 'db choice'

# Search
docker run --rm -v memo-data:/data ghcr.io/jagoff/memo:latest \
  memo search 'which database'

# List all
docker run --rm -v memo-data:/data ghcr.io/jagoff/memo:latest \
  memo list
```

## MCP server (stdio)

```bash
docker run -i --rm -v memo-data:/data ghcr.io/jagoff/memo:latest memo-mcp
```

Point your MCP client (Claude Code, IDE integration, etc.) to that command. The embedding model ships pre-embedded in the image — first startup has zero download overhead.

## Building locally

```bash
git clone https://github.com/jagoff/memo
cd memo
docker build -t memo:local .
docker run --rm memo:local memo doctor
```

See [Dockerfile](../Dockerfile) for build details (Python 3.13+, uv, CPU-optimized quantization).

## Environment variables

Pass `MEMO_*` flags via `docker run -e`:

```bash
docker run -e MEMO_DATA_DIR=/data \
  -e MEMO_SEARCH_MODE=hybrid \
  -v memo-data:/data \
  ghcr.io/jagoff/memo:latest \
  memo search 'query'
```

Common flags:
- `MEMO_DATA_DIR` — data directory (default: `/data`)
- `MEMO_SEARCH_MODE` — `vec`, `bm25`, or `hybrid` (default: `hybrid`)
- `MEMO_EMBEDDER_DIMS` — embedding dimension (CPU uses 384-dim; do not override)
- `MEMO_NONINTERACTIVE=1` — suppress prompts

See `memo config validate` for full registry.

## Advanced usage

### Custom config directory

```bash
docker run -v memo-data:/data \
  -v ~/.memo/config:/config \
  -e MEMO_STATE_DIR=/config \
  ghcr.io/jagoff/memo:latest \
  memo list
```

### Network MCP server (HTTP)

For remote clients, use Docker Compose or expose via `docker run`:

```bash
docker run -d --name memo-server \
  -v memo-data:/data \
  -p 8765:8765 \
  -e MEMO_MCP_PORT=8765 \
  ghcr.io/jagoff/memo:latest \
  memo-mcp --http 0.0.0.0:8765
```

Then connect clients to `http://localhost:8765`.

### HTTP auth

`memo http-api` requires bearer auth by default. Set `MEMO_HTTP_API_TOKEN`
and send `Authorization: Bearer <token>` on API requests. Binding to a
non-loopback host requires both a token and `--allow-non-loopback`.

Loopback-only development can use `--allow-no-auth`; this flag is rejected for
non-loopback binds.

### Compose example

```yaml
version: '3.9'
services:
  memo:
    image: ghcr.io/jagoff/memo:latest
    volumes:
      - memo-data:/data
    environment:
      MEMO_DATA_DIR: /data
      MEMO_SEARCH_MODE: hybrid
    command: memo-mcp
    ports:
      - "8765:8765"

volumes:
  memo-data:
```

```bash
docker compose up -d
```

## Feature matrix

| Feature | Docker (CPU) | Native (Apple Silicon) |
|---|---|---|
| Save, markdown store | ✅ | ✅ |
| Semantic search + recall | ✅ (CPU embeds) | ✅ (MLX) |
| BM25 / FTS5 | ✅ | ✅ |
| Reranking | ❌ (MLX-only) | ✅ |
| `ask`, `synthesize`, `dream` | ❌ (MLX-only) | ✅ |
| MCP server | ✅ (stdio/HTTP) | ✅ (stdio/HTTP) |
| Graph, contradictions | ✅ | ✅ |

## Performance notes

- **Embeddings:** CPU-quantized (384-dim) runs ~50-200ms per query depending on workload.
- **Search latency:** BM25 is instant; hybrid adds embedding cost (50-200ms).
- **Memory:** Typical container uses 300-500MB (embedder cached).
- **Scaling:** For high throughput, use `docker compose` with resource limits or orchestrate via Kubernetes.

## Troubleshooting

**"Connection refused" when connecting MCP client:**
- Ensure `-i` flag (stdio) or `-p 8765:8765` (HTTP) in `docker run`
- Check `docker logs <container>` for startup errors

**"No space left" after many saves:**
- Check volume size: `docker volume inspect memo-data`
- Prune old memories: `memo consolidate` (run once inside container)

**Embedding fails:**
- Verify model loaded: `memo doctor`
- Check CPU resource allocation to container (set `--cpus` if constrained)

**Search returns no results:**
- Verify memories were saved: `memo list`
- Check search mode: `memo config validate | grep SEARCH_MODE`
- Try `memo reindex` to rebuild the index

## Image tags

- `latest` — latest release (stable)
- `vX.Y.Z` — pinned release
- `dev` — bleeding edge (master branch, unstable)

Pull images from `ghcr.io/jagoff/memo:<tag>`.

## License

memo is licensed under MIT. See [LICENSE](../LICENSE).
