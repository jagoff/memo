# Token Savings — Proxy Context Compression

Date: 2026-08-18
Status: Approved design
Branch: (to be cut from `feat/emission-ledger`)
Supersedes: SP2 and SP4 of `2026-08-18-token-savings-sp1-live-measurement-design.md`

## Context

memo's stated purpose includes saving the user tokens. This design follows a
measurement of whether it actually does, taken on 2026-08-18 against the
installed 4.12.2 binary and this machine's live logs.

**What the measurement found:**

| # | Finding | Evidence |
|---|---|---|
| 1 | memo's own MCP tool schemas are its largest token cost | 43 tools ≈ **9.8k tok in every request** (wire-format Anthropic `tools` array; an earlier measurement of 41 tools, 46,562 B ≈ 11,640 tok counted the full MCP `Tool` protocol object incl. `outputSchema`, ~19% heavier than what's actually on the wire, and is corrected here) — 4.9% of a 200k window, paid whether or not a tool is called. Largest: `memo_save` 785, `memo_graph` 754, `memo_search` 678 |
| 2 | The recall injection is cheap and is not the problem | `context_cost.log`, n=1281: mean 889 chars ≈ **222 tok/turn** |
| 3 | 92% of output spend is the tool loop, where memo has no presence | memo's own meter: `1.25M tok answer` vs `15.15M tok tool-loops`. `hooks/hooks.json` registers SessionStart, UserPromptSubmit, Stop, PreCompact — **no `PreToolUse`, no `PostToolUse`** |
| 4 | `memo tokens` contradicts itself on one screen | Measured panel: `1,608 tok/turn cost` (memo costs). Estimated panel: `2.00M tokens saved`. The latter is `grounded × 350 + consults × 200` from `MEMO_ROI_TOKENS_PER_GROUNDED` / `_PER_CONSULT` — hardcoded constants, not measurement |
| 5 | The measured A/B has no control arm | `n=2245 grounded / 9 ungrounded`. Nine samples |
| 6 | memo is blind to its own input cost | `token_meter.py` reads only `output_tokens`; never `input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` |
| 7 | Measured, gate-passed levers ship off | `crusher_L1` reports `+44.4% (measured, gate-passed)` while `MEMO_CRUSHER_ENABLED` defaults `False`; `crush_cache/` holds 1 file, `offload/` 1 |

Findings 1 and 3 are the two large ones, and neither is addressed by any lever in
memo's current token surface. Finding 3 in particular says the spend lives on a
plane memo does not touch at all.

**Relation to the existing sub-project plan.** A prior session reviewed the same
nine external projects and decomposed the work into SP1–SP4. SP1
(live-distribution measurement of recall levers) is in flight on
`feat/emission-ledger` and is **unaffected** by this design — it measures the
recall-output and capture planes on real traffic, which stays useful. This design
supersedes SP2 (token-sink audit from session logs) and SP4 (cache-safe injection
hardening + tool-output crush) because a proxy observes provider-reported `usage`
directly, which is strictly better evidence than parsing session logs, and
because tool-output compression at the proxy subsumes a hook-based crush. SP3
(structure maps over codegraph) survives as a transform inside the proxy rather
than a standalone sub-project.

**Scope decisions taken by the user in this session**, each chosen over more
conservative alternatives that were presented:

1. Scope: the full tool-output plane, not just memo's own footprint.
2. Mechanism: a proxy that rewrites the request, not `PreToolUse` filters.
3. Planes: the entire payload, including system prompt, history, and tool
   schemas — plus pixel mode.
4. Rollout: everything on by default from day one; measure in production.

Decisions 3 and 4 carry risk that is stated plainly in "Risks accepted" below.
They are the user's call and this design implements them as chosen.

## Goal

Put memo in the request path as a local proxy that rewrites outbound payloads to
cost fewer tokens, recovers anything it cut on demand, and reports what it
actually saved using the provider's own `usage` numbers rather than an estimate.

## Non-Goals

- No change to memo's retrieval quality, ranking, or recall hook behavior.
- No routing to non-Claude models, and no model substitution.
- No compression of the response stream — responses relay untouched.
- No replacement of SP1's fixed-corpus gate (`eval/token_baseline.json`).
- No cloud component; everything runs on `127.0.0.1`.

## Section 1 — Architecture and deployment

```
Claude Code ──ANTHROPIC_BASE_URL=http://127.0.0.1:8768──> memo proxy ──> api.anthropic.com
                                                              │
                                                    rewrite request payload
                                                    relay response bytes untouched
                                                    record provider `usage`
```

`ANTHROPIC_BASE_URL` is Claude Code's documented LLM-gateway variable, and the
docs confirm a custom base URL is a supported connection type (the byte-level
stream watchdog explicitly covers "gateway connections, including a custom
`ANTHROPIC_BASE_URL`"). Three documented constraints are load-bearing here and
are requirements, not notes:

- **Forward `anthropic-beta` verbatim.** Setting only `ANTHROPIC_BASE_URL`
  without a gateway credential keeps the user's claude.ai subscription as the
  active credential, and gateways passing that traffic to Anthropic must forward
  the OAuth capability in `anthropic-beta`. Dropping or rewriting that header
  breaks subscription auth. All request headers are forwarded unmodified except
  `content-length`, which is recomputed.
- **Never buffer the response.** The byte-level watchdog aborts a stream after
  180s with no bytes on the direct API. The proxy relays SSE bytes as they
  arrive, including keep-alive pings, and never accumulates the body.
- **Configure via `settings.json`, not a shell export.** Shell exports reach only
  the shell that cold-started the background-agent supervisor. `memo ops install
  proxy` writes the `env` block in `~/.claude/settings.json`.

Port **8768**. 8765 is taken by `~/repos/rag/web/server.py` and 8767 by
`com.memo.chat` on this machine; the port is a flag, and `memo ops install proxy`
fails with a clear error if the chosen port is already listening.

Deployment mirrors `com.memo.chat` exactly: `memo ops install proxy [--port]` /
`memo ops uninstall proxy`, rendering a `com.memo.proxy` LaunchAgent with
`KeepAlive`, logging to `~/Library/Logs/memo/proxy.log`. Templates live in
`launchd/` with the existing `__HOME__` / `__MEMO_BIN__` placeholders. As with
the chat agent, installing before the code ships crash-loops under `KeepAlive`,
so install happens after release, against the isolated runtime binary.

Requires the existing `[http]` extra (FastAPI/uvicorn), already a dependency of
`memo chat serve`. A missing extra raises a `ClickException`, matching
`cli_chat.py`'s handling — never a bare `ImportError`.

## Section 2 — Zones and the cache rule

Rewriting the cached prefix is the single way this project can lose: a cache read
costs 0.1× a fresh input token, so a transform that saves 20% of prefix tokens
while forcing a re-cache every turn is a large net loss. `zones.py` owns this
rule and every transform declares which zone it may touch.

- **Stable prefix** — system prompt, tool definitions, and all turns older than
  the live window. A transform here must be **deterministic and session-stable**:
  given the same session, it produces byte-identical output on every turn. The
  prefix then changes once per session, paying one cache-creation and hitting
  cache thereafter.
- **Live zone** — the most recent turns and fresh tool results. Not yet cached,
  so transforms here are always safe and always profitable.

Session-stability is enforced mechanically, not by convention: `zones.py` hashes
the emitted prefix per session and a mismatch across turns within one session is
a test failure and a logged runtime warning. A transform that cannot guarantee
stability is restricted to the live zone.

## Section 3 — Modules

New subpackage `src/memo/proxy/`, following the repo's one-domain-per-file
convention:

| File | Responsibility |
|---|---|
| `server.py` | ASGI app: `POST /v1/messages`, catch-all passthrough for every other path, header forwarding, SSE relay |
| `zones.py` | Splits the payload into stable prefix and live zone; owns and enforces the cache rule |
| `plan.py` | Builds a `TransformPlan` — which edits apply, estimated saving, profitability gate |
| `ccr.py` | Content-addressed recovery: stores originals before cutting, reusing `crush_cache/` |
| `meter.py` | Per-request ledger keyed on provider `usage`; holdout assignment |
| `transforms/toolschemas.py` | Prunes MCP tool definitions (finding 1) |
| `transforms/toolresults.py` | Declarative per-command filters plus a generic fallback (finding 3) |
| `transforms/structmap.py` | Code file reads reduced to signatures and imports |
| `transforms/delta.py` | Re-read of an already-seen file reduced to its diff |
| `transforms/jsoncrush.py` | Wraps the existing L1 crusher (finding 7) |
| `transforms/pixel.py` | Renders dense blocks to PNG under a profitability gate |
| `filters/*.yaml` | The per-command filter catalog |

Every transform implements one interface — `plan(payload, zone) -> list[Edit]` —
so `plan.py` composes them without knowing what any of them does, and each is
unit-testable against a fixture payload with no proxy running.

## Section 4 — Transforms

Ordered by expected saving, each grounded in a measured number where one exists.

**1. Tool-schema pruning** — the measured ~9.8k tok/request (43 tools, wire-format
`tools` array; see the corrected finding #1 above). The proxy keeps in
the payload only tools called at least once in the current project's last
`MEMO_PROXY_TOOL_WINDOW_SESSIONS` sessions (default 20), plus a small
always-present `memo_tool_docs(name)` that hydrates a pruned schema on demand.
The retained set is computed from memo's existing consult/usage logs,
frozen at the first request of a session, and reused byte-identically for the
rest of it — satisfying the stable-prefix rule. A tool call naming a pruned tool
is not an error: the proxy hydrates it and the model retries, which is the same
discovery-then-hydrate shape as the GPL-licensed
`mcp-server-code-execution-mode` (idea only; implemented natively).

**2. Tool-result filtering** — the 92% plane. A declarative YAML catalog matches
on command and subcommand, then runs a pipeline of composable actions
(`keep_lines`, `truncate_lines`, `json_extract`, `aggregate`, `head`, `tail`,
`dedup`, `format_template`). Commands with no matching filter fall back to a
generic head+tail+elision-count transform, so coverage is total from day one and
the catalog is an optimization, not a prerequisite. Filters are ported from snip
(MIT — see Section 8).

**3. Structure map and delta** — a code file read is replaced by its signatures
and imports; a re-read of a file already in context is replaced by its diff
against the copy already there. memo already indexes this repo with codegraph, so
`structmap.py` queries that index rather than re-parsing.

**4. JSON crush** — reuses the L1 crusher that already measures `+44.4%`
gate-passed and ships off.

**5. Pixel mode** — dense blocks (minified JSON, long-line logs, old history)
render to grayscale PNG pages. Gated per block on estimated profitability: fires
only when `est_image_tokens < est_text_tokens * 0.8`. A block whose gate fails
passes through as text.

## Section 5 — Recovery

Nothing is cut without being recoverable. Before any lossy edit, `ccr.py` stores
the original keyed by `sha256(content)` in the existing `crush_cache/`, and the
replacement text carries the key. `memo retrieve` already exists and already
reads that cache; the proxy reuses it rather than adding a second recovery path.
The MCP surface exposes it as `memo_retrieve` so the model can pull an original
back mid-turn.

A recovery is the signal that a transform cut too deep. `meter.py` counts
retrievals per transform, and a transform whose retrieval rate exceeds
`MEMO_PROXY_RETRIEVE_ALARM_FRAC` (default 0.05) is reported in `memo tokens` as
over-cutting — a retrieved original costs its tokens twice.

## Section 6 — Measurement

This section fixes findings 4, 5, and 6, and is the part that makes the rest
verifiable rather than believed.

The proxy sees the provider's own `usage` on every response: `input_tokens`,
`output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`. That
is ground truth, and it is strictly better than anything memo can infer from a
transcript.

- **`token_meter.py` is extended to all four fields.** Today it reads
  `output_tokens` alone, which is why memo cannot see its own input cost. Cache
  creation and cache read are tracked separately, because the whole cache rule in
  Section 2 is unfalsifiable without them.
- **A real holdout replaces the n=9 control arm.** `MEMO_PROXY_HOLDOUT_FRAC`
  (default 0.05) of requests pass through uncompressed, assigned by a hash of the
  request so assignment is stable and unbiased. The holdout is the control arm
  and is what every saving claim is measured against.
- **The estimated panel is deleted.** The `grounded × 350 + consults × 200`
  arithmetic and the `MEMO_ROI_TOKENS_PER_*` flags that feed it are removed, not
  reworded. `memo tokens` reports one number per transform, measured, with a
  sample count and a confidence interval — and reports a cost as a cost.
- **Per-transform attribution.** Each request records which transforms applied
  and their estimated contribution, so `memo tokens --by-transform` shows which
  ones earn their place and which are noise.

`memo tokens` prints "no measured data yet" when the ledger is empty, matching
the existing empty-state convention, rather than printing a zero that reads as a
result.

## Section 7 — Rollout, flags, and failure

Per the user's decision, every transform is **on by default** at first install,
with the holdout measuring in parallel so "measure later" costs nothing in time.
Pixel mode's per-block profitability gate is part of the transform itself, not a
flag, so it is on but self-limiting.

Flags, registered in a new `flags_proxy.py` per the repo's per-domain convention:

| Flag | Default | Effect |
|---|---|---|
| `MEMO_PROXY_ENABLED` | `true` | Master switch; off = pure passthrough |
| `MEMO_PROXY_PORT` | `8768` | Listen port |
| `MEMO_PROXY_HOLDOUT_FRAC` | `0.05` | Uncompressed control arm |
| `MEMO_PROXY_TOOL_SCHEMAS` | `true` | Transform 1 |
| `MEMO_PROXY_TOOL_WINDOW_SESSIONS` | `20` | Sessions of usage history that decide which tool schemas survive |
| `MEMO_PROXY_TOOL_RESULTS` | `true` | Transform 2 |
| `MEMO_PROXY_STRUCTMAP` | `true` | Transform 3 |
| `MEMO_PROXY_JSONCRUSH` | `true` | Transform 4 |
| `MEMO_PROXY_PIXEL` | `true` | Transform 5 |
| `MEMO_PROXY_RETRIEVE_ALARM_FRAC` | `0.05` | Over-cutting alarm threshold |

`memo proxy off` is the one-command revert; it flips the master flag through the
Markdown config so it reaches the LaunchAgent, not just the current terminal.

**Failure is fail-open, everywhere.** Any exception raised while planning or
applying a transform is caught per-request, logged with the transform name, and
the **original body** is forwarded unmodified. A transform that throws is
disabled for the remainder of the session after three failures. If the upstream
is unreachable the proxy returns the upstream error verbatim rather than
synthesizing one. `memo doctor` gains a proxy check: agent loaded, port
listening, last request within expectations, and a warning when
`ANTHROPIC_BASE_URL` points at a proxy that is not running — the failure mode
that would otherwise look like a dead network.

The proxy handles the user's API credentials. They are forwarded and never
logged, never stored, and never included in a ledger row or an error message;
a test asserts that no header value reaches any log sink.

## Section 8 — Licensing

memo is MIT. Verified via the GitHub API and each project's LICENSE file on
2026-08-18:

| Project | License | Use |
|---|---|---|
| snip | MIT | **Code portable** with attribution — the filter catalog and pipeline actions |
| ccusage | MIT | **Code portable** with attribution — the four-field `usage` accounting |
| headroom | Apache-2.0 | **Code portable** with NOTICE — holdout and cache-alignment approach |
| entroly | Apache-2.0 | **Code portable** with NOTICE — recovery-handle and receipt shape |
| caveman | MIT shell, but `engine/`, `proxy/`, `cacheengine/`, `rewriter/`, `browse/`, `mcp/`, `shrink/`, `shared/platform/` are **BSL-1.1** | **Idea only** — the BSL directories are precisely the compression and proxy code, so none of it is copied |
| token-optimizer | Custom, required-notice, non-OSI | **Idea only** |
| jcodemunch-mcp | Dual-Use License v1.1 | **Idea only** |
| mcp-server-code-execution-mode | GPL-3.0 | **Idea only** — incompatible with MIT redistribution |
| awesome-llm-token-optimization | No license | **Idea only** (a link index) |

Ported files carry the upstream copyright and license header. A `NOTICE` file
records the Apache-2.0 attributions. This matches the rule the SP1 design already
set, sharpened with the per-directory BSL carve-out in caveman, which the earlier
review recorded only in general terms.

## Section 9 — Error handling

- Transform raises → caught per-request, original body forwarded, counted; three
  failures disable that transform for the session.
- Upstream unreachable or erroring → error relayed verbatim, never synthesized.
- Malformed request body (not JSON, unexpected shape) → forwarded untouched.
- `crush_cache/` unwritable → the lossy transform that needed it is skipped, not
  applied-without-recovery.
- Ledger file corrupt or missing → measurement degrades to a skip count; a
  measurement failure never fails a request.
- Port already bound at install → `memo ops install proxy` fails loudly with the
  conflicting process, rather than rendering a plist that crash-loops.

## Section 10 — Testing

- **Unit, per transform**: fixture payloads captured from real traffic; each
  transform tested in isolation without a running proxy.
- **Golden**: request-in → request-out byte-exact, so an unintended rewrite shows
  up as a diff.
- **Cache stability**: the same session across N turns must emit a
  byte-identical stable prefix. This is the test that protects the economics.
- **Fail-open**: a transform stubbed to raise must yield the original body
  unchanged.
- **Header fidelity**: `anthropic-beta` and auth headers arrive upstream
  verbatim; no header value appears in any log.
- **Streaming**: a fake upstream emitting slow SSE chunks must reach the client
  incrementally, asserting the proxy does not buffer.
- **E2E**: full request cycle against a fake upstream, asserting measured
  `usage` lands in the ledger and holdout assignment is stable.
- **Regression**: `memo eval recall --gate` and `memo eval behavior` continue to
  pass — the proxy must not perturb retrieval quality.
- Test isolation follows `tests/conftest.py`: `tmp_cfg`, `MEMO_NONINTERACTIVE=1`,
  isolated `MEMO_DATA_DIR` / `MEMO_STATE_DIR`, never the developer's real vault.

## Risks accepted

Stated once, and implemented as decided:

1. **The proxy is in the critical path of every model call.** A crash takes the
   session with it. Mitigated by fail-open, `KeepAlive`, and the `memo doctor`
   check — not eliminated.
2. **Aggressive compression on by default can degrade answers before the holdout
   notices.** The holdout measures cost, and `memo eval behavior` measures
   steering, but neither catches a mid-session quality loss in real time. This is
   the accepted cost of shipping on rather than gated.
3. **Pixel mode depends on the model reading rendered text reliably.** The
   profitability gate bounds the cost, not the comprehension risk.
4. **Prefix rewriting can invert the savings** if session-stability regresses.
   The cache-stability test is the guard, and it is the highest-value test here.
5. **memo takes on a role well beyond memory.** This is a deliberate expansion,
   chosen explicitly.
