# memo en Docker (backend CPU, cross-platform)

La imagen Docker corre el **backend CPU** de memo — ideal para Linux o para
probar memo sin instalar nada. **No incluye MLX**, así que `search`, `recall`
y `save` funcionan; el **reranker** y los verbos LLM (`ask`, `synthesize`,
`dream`) son Apple-Silicon-only y quedan deshabilitados con un mensaje claro.

> Docker = puerta de entrada cross-platform. Para memo **completo**
> (rerank + ask + synthesize + dream) usá la instalación nativa en un Mac
> Apple Silicon:
> `curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash`

## Probarlo en 10 segundos

```bash
docker run --rm ghcr.io/jagoff/memo:latest memo doctor
```

## Uso con persistencia

```bash
docker volume create memo-data
docker run --rm -v memo-data:/data ghcr.io/jagoff/memo:latest \
  memo save 'we use Postgres, not Mongo' --title 'db choice'
docker run --rm -v memo-data:/data ghcr.io/jagoff/memo:latest \
  memo search 'which database'
```

## Como servidor MCP (stdio)

```bash
docker run -i --rm -v memo-data:/data ghcr.io/jagoff/memo:latest memo-mcp
```

Apuntá tu cliente MCP a ese comando (`docker run -i … memo-mcp`). El modelo de
embeddings ya viene pre-horneado en la imagen — el primer arranque no descarga
nada.

## Qué funciona / qué no

| Feature | Docker (CPU) |
|---|---|
| `save`, markdown store | ✅ |
| Semantic search + recall + MCP `memo_search` | ✅ (embeds CPU) |
| BM25 / FTS5 | ✅ |
| Reranker, `ask`, `synthesize`, `dream` | ❌ MLX-only (mensaje claro) |
