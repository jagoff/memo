# Embedder daemon (shared MLX sidecar)

`memo recall-daemon` is more than a recall server. It also doubles as a
**shared MLX embedder sidecar**: any peer process — `synapse`'s federator,
`memflow`'s daemon, another `memo` CLI invocation — can reuse the one warm
MLX model loaded in the daemon's RAM (~2GB for Qwen3-Embedding-0.6B +
reranker) instead of paying its own cold start (~2s) and memory cost.

This document covers the socket protocol, lifecycle, observability surface,
and the env knobs that tune all three.

## Lifecycle

```bash
memo recall-daemon start    # background, PID written to state_dir
memo recall-daemon status   # alive/dead + socket path
memo recall-daemon stop     # SIGTERM the PID, cleans socket+pid file
```

The daemon is started automatically by the `SessionStart` hook in
`hooks/hooks.json`. The PID file lives at `state_dir/recall-daemon.pid` and
the Unix socket at `state_dir/recall.sock`. Both are removed on clean
shutdown.

## Socket protocol

One JSON line in, one JSON line out, newline-delimited. The request is
dispatched on its `op` field. Legacy callers that omit `op` default to
`recall` for backward compatibility.

| Op | Request | Response |
|---|---|---|
| `recall` (default) | `{"prompt": "...", "cwd": "..."}` | `{}` or `{"hookSpecificOutput": {...}}` |
| `embed_query` | `{"op": "embed_query", "text": "..."}` | `{"vector": [...], "dim": N, "model": "..."}` |
| `embed_batch` | `{"op": "embed_batch", "texts": [...]}` | `{"vectors": [[...]], "dim": N, "model": "..."}` |
| `ping` | `{"op": "ping"}` | `{"ok": true, "model": "...", "dims": N, "started_at": <epoch>, "uptime_s": N}` |
| `stats` | `{"op": "stats"}` | per-op counters + p50/p95/p99 latency (see below) |
| error | any malformed request | `{"error": "<message>"}` |

`embed_query` adds the asymmetric query prefix. `embed_batch` does not — pass
documents for symmetric corpus embeddings.

## In-process client

`memo.embedder_client` is the socket-first, fallback-in-process client:

```python
from memo.embedder_client import embed_query, embed, ping, stats, status

vec  = embed_query("astor terapia ocupacional")
vecs = embed(["doc 1", "doc 2"])

if status() is not None:
    snap = stats()
    print(snap["ops"]["embed_query"]["p99_ms"])
```

If the daemon socket is missing/refused/timed out, the client lazily
loads `MLXEmbedder` in-process and continues. Set
`MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON=1` to raise instead.

## Stats snapshot

`stats` returns a snapshot like:

```json
{
  "started_at": 1736552000.0,
  "uptime_s": 4321,
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "dims": 1024,
  "ops": {
    "embed_query": {"count": 412, "errors": 0, "samples": 412, "p50_ms": 7.2, "p95_ms": 12.1, "p99_ms": 18.4},
    "recall":      {"count": 53,  "errors": 0, "samples": 53,  "p50_ms": 142.0, "p95_ms": 198.0, "p99_ms": 220.0},
    "ping":        {"count": 7,   "errors": 0, "samples": 7,   "p50_ms": 0.4,  "p95_ms": 0.7,  "p99_ms": 0.9}
  }
}
```

Latency samples are stored in a bounded reservoir (1024 most recent
per op) so memory stays flat under load. Errors include malformed
requests, unknown ops, and handler exceptions.

The daemon also persists this snapshot to
`state_dir/embed_daemon_stats.json` every
`MEMO_EMBEDDER_STATS_INTERVAL_S` seconds (default `60`). Peers that
cannot or do not want to open the socket — `synapse_doctor`, a static
health dashboard — can `cat` that file instead.

CLI surface:

```bash
memo embed-daemon status         # alive/dead + model + uptime
memo embed-daemon stats          # pretty table
memo embed-daemon stats --json   # raw snapshot for jq/scripts
```

## Environment knobs

| Variable | Default | Purpose |
|---|---|---|
| `MEMO_EMBEDDER_CLIENT_TIMEOUT` | `8.0` | socket read timeout (seconds) |
| `MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON` | unset | `1` disables the in-process fallback |
| `MEMO_EMBEDDER_STATS_INTERVAL_S` | `60` | snapshot persistence cadence (set `0` to disable) |
| `MEMO_RECALL_DEBUG` | unset | `1` prints daemon-side diagnostics to stderr |

## Notes

- The MLX model is loaded once in the daemon process. All embed/recall
  calls go through a single `threading.Lock` to serialise MLX forward
  passes, so concurrent peers do not race the same model.
- Auto-batching of `embed_query` requests across a short window is on
  the roadmap (`MEMO_EMBEDDER_BATCH_WINDOW_MS`) but requires a batched
  query-prefix path inside `MLXEmbedder`; today each request triggers
  its own forward pass.
- The daemon's `dims` value MUST match `MEMO_EMBEDDER_DIMS` in callers,
  or stored vectors will not align. The `ping` / `stats` responses
  expose both so peers can fail fast on a mismatch.
