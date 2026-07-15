# MCP client-sampling for synthesis tools — design

**Date:** 2026-07-15
**Status:** approved (brainstorm), pending implementation plan
**Scope:** spec 1 of 2 (spec 2, separate: heading-aware chunker wired into reindex)

## Problem

memo's MCP surface is a passive toolbox: every synthesis tool (`memo_ask`,
`memo_chat_ask`, `memo_reflect`, `memo_synthesize_run`, `memo_consolidate`)
answers with the local MLX helper LLM (small Qwen-class model). When the
caller is an MCP client backed by a frontier model, memo throws away the
better model sitting on the other side of the connection. MCP sampling
(`ctx.sample()`, FastMCP ≥ 3.x — repo pins `fastmcp>=0.5,<4`, 3.4.4
installed) lets the server delegate synthesis to the client's model.

Non-goals: ambient recall parity for MCP-only clients, push notifications,
chunker/crossref (spec 2), sampling for cheap background LLM calls (title
extraction, dedup classification, save-gate), sampling for `score_grounding`.

## Design

### Architecture — contextvar + adapter (approach A)

New module `src/memo/sampling.py`, mirroring the `_trace.py` pattern:

- `_sampler` contextvar; `sampling_scope(sampler)` context manager;
  `current_sampler()` accessor.
- `SamplingChat`: same duck-type interface as `MLXChat.chat(model, messages,
  ...)`. Calls the active sampler; on any exception (timeout, client refusal,
  missing capability) it delegates to a fallback `MLXChat` and sets a
  **sticky fallback** for the rest of the request (no retry storm).
- `make_bridge(ctx)`: builds the sync-callable sampler from a FastMCP
  `Context`. Internally `asyncio.run_coroutine_threadsafe(ctx.sample(...),
  loop).result(timeout)` — the synthesis code runs in a worker thread while
  `ctx.sample()` must run on the server event loop.

`Memory._ensure_chat()` (facade) changes: when `MEMO_SAMPLING_SYNTH_ENABLED`
is on and `current_sampler()` is set, return a per-call `SamplingChat`
wrapping the (lazily cached) MLX chat. The `SamplingChat` wrapper is **never
cached on `self`** — the facade is shared across requests/threads and caching
would leak one client's context into another request. The existing
`self._chat` cache remains MLX-only.

Dream / CLI / daemons: contextvar is empty → MLX exactly as today. Zero
behavior change outside MCP request scope.

### Tool surface

Tools that set `sampling_scope` (become `async def` with `ctx: Context`;
sync body runs via `anyio.to_thread.run_sync`):

1. `memo_ask` (server_core_search)
2. `memo_chat_ask` (server_core_search)
3. `memo_reflect` (server_reflect)
4. `memo_synthesize_run` (server_synthesis)
5. `memo_consolidate` (server_consolidate)

Each does a capability check first: if the client does not advertise
sampling support, the scope is not set (pure MLX path, no error).

Excluded on purpose:
- `score_grounding` — entailment scoring stays on MLX (cheap; avoids a
  second client round-trip per answer). Grounding still runs on sampled
  answers.
- All background/cheap LLM calls (titles, dedup classification, save-gate).

Per-request guard: `MEMO_SAMPLING_MAX_CALLS` (default 3). Multi-call flows
(`synthesize_run` cross-cluster, `consolidate`) sample up to the cap, then
fall back to MLX for the remainder.

Attribution: every synthesized response dict gains
`synthesizer: "client:<model>" | "mlx:<model>"` so users, eval, and debugging
can tell which path answered.

### Data flow

```
memo_ask (async, ctx: Context)
  ├─ capability check: client supports sampling? no → no scope (MLX)
  ├─ sampler = sampling.make_bridge(ctx)     # threadsafe bridge to loop
  └─ anyio.to_thread.run_sync(
        with sampling_scope(sampler): memory.ask(...)   # sync, unchanged
     )
```

`Memory.ask()` / `reflect()` / etc. keep their signatures — `_ensure_chat()`
resolves the backend alone. The bridge lives in one helper shared by all
five tools.

### Error handling

- Any sampler failure → `log.debug` + MLX fallback + sticky-per-request.
  **A tool call never fails because sampling broke.**
- Timeout budget: `MEMO_SAMPLING_TIMEOUT_S` per sample call.
- No partial answers: a sample either returns text or the whole call falls
  back to MLX.

### Flags (`flags_behavior.py`)

| Flag | Default | Meaning |
|---|---|---|
| `MEMO_SAMPLING_SYNTH_ENABLED` | OFF (dark) | master switch |
| `MEMO_SAMPLING_TIMEOUT_S` | 30 | per-sample timeout |
| `MEMO_SAMPLING_MAX_CALLS` | 3 | samples per MCP request, then MLX |
| `MEMO_SAMPLING_MAX_TOKENS` | (impl-chosen) | budget for the sample prompt |

`MEMO_SAMPLING_SYNTH_ENABLED` declares a `manual` gate (+ reason) in
`dream_flags.GATES` — completeness is CI-enforced by `test_dream_flags.py`;
the recall A/B and tuner gates do not measure synthesis quality.

Resolution order unchanged: env > markdown config > tuned overlay > default.

## Testing

- **Unit** — `sampling.py` scope semantics (isolation across threads and
  requests); `SamplingChat` fallback on error/timeout + sticky behavior;
  `_ensure_chat()` selection matrix (flag off → MLX; no sampler → MLX;
  flag+sampler → SamplingChat; wrapper never cached on the facade).
- **Integration** — FastMCP in-memory client with a `sampling_handler`:
  `memo_ask` returns the handler's answer with `synthesizer=client:*`;
  a client without the handler gets `synthesizer=mlx:*` (MLX stubbed).
  Call-cap verified on `memo_consolidate` / `memo_synthesize_run`.
- **Convention** — gate declared in `dream_flags.GATES`; explicit test that
  dream/CLI paths never observe a sampler; `mypy` + `ruff` green on `src/`
  and `tests/`.
- Synthesis-class regressions keep gating in synapse `eval-chat`
  (per the retrieval/synthesis split documented in CLAUDE.md).

## Risks

- **Client sampling support is uneven.** Mitigated: capability check + MLX
  fallback means the feature degrades to today's behavior.
- **Latency**: a client round-trip can exceed local MLX for short answers.
  Mitigated: timeout + dark flag (opt-in).
- **Thread/loop deadlock** if a sample is awaited from the loop thread
  itself. Mitigated: bridge is only handed to code already running in the
  worker thread (`anyio.to_thread`); unit test covers it.
