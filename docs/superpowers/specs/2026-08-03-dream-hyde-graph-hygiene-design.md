# Activate HyDE-Tune + Graph-Hygiene Dream Passes

## Context

Prompted by "how much is memo actually using the LLM/MLX, and are there
improvement opportunities we're not taking advantage of?" An audit of
`src/memo/dream_flags.py`'s `GATES` registry (~55 entries) via
`memo dream graduate-flags --status` showed most dream self-improvement
passes are already `human_graduated` (ON): anticipate, bridges, chronicle,
code-drift, code-repair, communities, consolidate-episodes, distill,
hype-generator, tune, tune-boost, dynamic-mandate-sync, graph-code-trace,
graph-discovery, graph-projection, graph-reason, graph-signal,
negative-recall (×4), proactive, recall-code-refs, sampling-synth.

Three substantive, non-deprecated passes remain OFF. All three are
`kind="manual"` in `GATES` — the nightly flag-graduation pipeline
(`run_flag_graduation_pass`) never auto-flips a manual-kind flag; a human
must enable it explicitly:

- `MEMO_DREAM_HYDE_TUNE_ENABLED` (`flags_misc.py`) — meta-gate for a nightly
  A/B of the never-measured `MEMO_HYDE_ENABLED` against mined+curated recall
  labels. Applies `MEMO_HYDE_ENABLED=1` via the tuned overlay only when it
  wins precision without raising noise, passes the curated gate, stays
  within latency headroom, and the live recall mode is not hybrid.
  Reversible via `memo dream tune --rollback`. Never run once — always OFF.
- `MEMO_DREAM_ENTITY_CANON_ENABLED` (`flags_misc.py`) — MinHash+LSH blocking
  proposes near-duplicate entity-name pairs; the helper LLM confirms each
  candidate; confirmed pairs merge via `entity_aliases`. Capped at 30
  LLM-confirmed pairs/night (`MEMO_DREAM_ENTITY_CANON_MAX_PAIRS`).
- `MEMO_DREAM_EDGE_VERIFY_ENABLED` (`flags_behavior.py`) — memory↔memory
  graph edges earn confidence from grounded co-use evidence
  (`grounding.log`); edges that never accumulate evidence decay gently.
  Curation of edge confidence only — never touches recall ranking, never
  deletes.

The last dream receipt showed 245 entities with 13 duplicate merges found
by the (currently manual, ad hoc) dedup path — signal that entity-canon and
edge-verify have real noise to work on.

None of the three flags have a dotted markdown-config key registered in
`src/memo/tui/config/catalog.py` (`path_to_env()` doesn't list them), so
`memo config set <key> <value>` cannot reach them today. The mechanism
already used for every other human-graduated dream flag (anticipate,
communities, tune, tune-boost, consolidate-episodes — confirmed by
inspecting `~/Library/LaunchAgents/com.memo.dream.plist`'s
`EnvironmentVariables`) is the LaunchAgent's own env block, sourced from the
template at `~/repos/memo/launchd/com.memo.dream.plist` per the machine-level
convention in `~/CLAUDE.md` ("editá el template y re-renderizá, no el plist
a mano").

## Chosen approach

Add the three flags to the dream LaunchAgent's `EnvironmentVariables`,
scoped to the nightly job only — no source-code change, no new
markdown-config key, no change to any already-graduated flag.

1. Edit the committed template `~/repos/memo/launchd/com.memo.dream.plist`
   (it still carries `__HOME__`/`__MEMO_BIN__` placeholders — same pattern
   as every other agent in the fleet). Add three `<key>`/`<string>` entries
   to the existing `EnvironmentVariables` dict, alongside
   `MEMO_DREAM_COMMUNITIES_ENABLED` etc.:
   ```
   MEMO_DREAM_HYDE_TUNE_ENABLED = 1
   MEMO_DREAM_ENTITY_CANON_ENABLED = 1
   MEMO_DREAM_EDGE_VERIFY_ENABLED = 1
   ```
   Then re-render (substitute `__HOME__` → `$HOME`, `__MEMO_BIN__` →
   `command -v memo`) into `~/Library/LaunchAgents/com.memo.dream.plist`,
   matching the deployed copy's existing already-substituted values.
2. Reload the agent:
   ```bash
   launchctl bootout gui/$(id -u)/com.memo.dream
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.memo.dream.plist
   ```
3. No `memo config set` call — no dotted key exists for these flags.

Rejected alternatives:

- **Add markdown-config dotted keys first, then `memo config set`.** More
  correct long-term (keeps all dream-flag control in one place instead of
  split between plist and markdown), but it's a source change to
  `tui/config/catalog.py` — out of scope for a same-day activation of an
  already-built, already-gated feature. Tracked as a follow-up, not blocking.
- **Flip all ten remaining OFF-but-viable candidates at once**
  (`INDEX_REPAIR`, `VECTOR_HYGIENE`/`VECTOR_VIEWS`, `RETAG_GLOBAL`,
  `PROFILE`, `INCREMENTAL`, `STAGING`, `TUNE_STRICT_GATE`, `SHADOW`,
  `LEDGER`). Rejected: conflates unrelated risk profiles and makes a bad
  night's receipt hard to attribute to one cause. This design scopes to the
  three flags with the clearest LLM-quality payoff and the best-understood
  blast radius; the rest is a separate future batch.
- **Prewarm/backfill entity-canon and edge-verify against the existing
  245-entity graph in one manual run before enabling nightly.** Rejected as
  unnecessary ceremony — both passes are designed to run incrementally and
  bounded (30 pairs/night cap on entity-canon); the first few nightly runs
  already function as the backfill.

## Risk and monitoring

| Flag | Risk | Why | Rollback |
|---|---|---|---|
| `HYDE_TUNE` | Low | Self-measuring, self-applying, self-reverting; gated by the curated eval set; refuses to activate under live hybrid mode (hook-budget guard) | `memo dream tune --rollback`, or remove the plist line |
| `ENTITY_CANON` | Low-medium | Bounded to 30 LLM-confirmed pairs/night; merge is a real mutation (`entity_aliases`) with no automated undo | Manually reverse specific `entity_aliases` rows if a bad merge surfaces; remove the plist line to stop new merges |
| `EDGE_VERIFY` | Low | Confidence-only curation, floored, never deletes, never touches recall ranking | Remove the plist line; no state mutation to undo beyond confidence scores, which re-earn from evidence |

Monitoring: `memo dream status` after each of the next 2-3 nightly runs —
check `receipt["errors"]` is clean and inspect the entity-canon /
edge-verify pass results (pairs proposed/confirmed, edges adjusted).
`memo dream graduate-flags --status` does not track manual-kind flags'
outcomes automatically; the receipt is the source of truth for these three.

## Out of scope

- Every already-`human_graduated` flag — untouched.
- The seven secondary candidates listed above.
- Adding a markdown-config dotted key for these three flags (noted as a
  nice-to-have follow-up).
- The unrelated `contradict: FileNotFoundError` WARN seen in the last dream
  receipt (missing WhatsApp vault chunk) — pre-existing bug, separate fix.
- MCP-sampling-based LLM routing — investigated and discarded earlier in
  this brainstorm: Claude Code doesn't implement the MCP `sampling`
  capability (anthropics/claude-code#1785) and the MCP spec itself
  deprecated sampling as of SEP-2577 (spec version 2026-07-28) in favor of
  MRTR. Not worth building against.
