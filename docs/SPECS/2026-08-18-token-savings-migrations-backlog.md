# Token-Savings Migration Audit — 9 external repos (2026-08-18)

Audit of token-optimization tooling from nine external repos, mapped onto
memo's existing token-savings surface and the approved
[proxy context-compression plan](2026-08-18-token-savings-proxy-context-compression-plan.md).
License notes matter: **snip + ccusage are MIT** (code portable), **headroom /
entroly are Apache-2.0**, **jcodemunch is dual-use (free personal only)**,
**code-execution-mode is GPL-3.0**, **caveman's skill is MIT but its engine/proxy
is BSL-1.1**, **token-optimizer is non-OSI**, **awesome-llm-token-optimization has
no license**. Anything from the non-OSI/BSL/GPL/dual-use set is migrated as
*ideas only* (fresh implementation), never as copied code.

Status legend: `DONE` (implemented in this or an earlier session), `PLANNED`
(already specced in the proxy plan), `BACKLOG` (proposed here, not yet specced).

## What memo already ships (verified working 2026-08-18)

- `memo tokens` — ledger-backed TUI (today/month/historic) with a measured
  transcript panel (answer/tool/injected + net per turn). Verified live.
- `memo token-savings` — measured, gate-passed per-lever savings
  (crusher_L1 +44.4%). Verified live.
- `memo token-gate` — cost-per-grounded regression gate
  (baseline: `state_dir/eval/token_gate_baseline.json`). Reseeded 2026-08-18:
  baseline was stale (Jul 7) vs current usage; cost/grounded had *improved*
  (166.2 → 159.7), grounded-rate drift was usage, not code. Now PASS.
- `memo roi`, `memo eval tokens` (token_baseline + gate), recall gate
  (`pre-push` hook), statusline badge, recall-hook token budget (~160 tok cap).

## DONE this session — ccusage-style 4-field + per-model accounting

`token_meter.py` read only `output_tokens`; the design (spec §6) planned the
4-field fix. Implemented now:

- Session ledger (schema v2, v1 rows still readable) records `input_tok`
  (max per-call prompt footprint), `cache_read_tok` / `cache_creation_tok`
  (summed billed volumes), and `models` (output-token spend per model).
- Honest-gate: builds that stamp degenerate `input_tokens` (fixed 1–2 per
  call) report footprint as 0/unknown; cache splits and model tally still land
  (they are the real billed volumes).
- Streaming-row dedup mirrors `iter_prompt_turns` (message id), excluding
  `<synthetic>` generations from the prompt side.
- `memo tokens` measured panel now shows input footprint, cache-read /
  cache-written volumes, and the per-model breakdown; `--json` exposes all
  keys additively. Tests: 15 green in `tests/test_token_meter.py`.

## Per-repo audit

### juliusbrussee/caveman (MIT skill / BSL-1.1 engine)
- **Per-type content detection + keep-list compressors** (json/log/code/diff/
  search/text) — `PLANNED` (proxy plan Task 6 transform catalog; idea-only
  beyond the MIT skill surface).
- **BM25+recency budgeting to fit context** (`contextwindow.Pack`) — `BACKLOG`
  (mirror of the recall budget; memo's FTS5 lez leg already does this shape).
- **Pixel-mode with profitability gate** — `PLANNED` idea-only (spec §4.5).
- **`caveman learn`-style ranked token-sink report** — `BACKLOG`: upgrade
  `memo tokens` with ranked sinks + fix classes.
- **TOON re-encoding of uniform JSON arrays** — `BACKLOG` (plan Task 6
  `jsontoon` transform candidate).

### ccusage/ccusage (MIT) — LARGELY DONE
- 4-field usage (input/output/cache_creation/cache_read) — **DONE** (this
  session; design §6 alignment).
- Per-model breakdown — **DONE** (this session; `models` in ledger + CLI).
- Locked pricing snapshot (LiteLLM revision, CI-updated) — `BACKLOG`
  (feeds `memo roi` dollars once §6 lands).
- 5-hour billing blocks + per-project grouping + source federation over
  memo's consult logs — `BACKLOG` (cheap aggregations over existing ledgers).

### alexgreensh/token-optimizer (non-OSI — ideas only)
- Pre-compact decision checkpoints + restore — `BACKLOG` (restore
  decisions after compaction; fits history.db).
- Loop/spin detection (similarity of recent messages, no LLM) — `BACKLOG`
  (best home: `token_meter` Stop-hook pass, same transcript).
- Keep-Warm cache-TTL ping + tripwire auto-off — `BACKLOG` (proxy plan
  zone rule owner; fail-open matches memo's philosophy).
- Decision extraction → verbatim injection at compaction — `BACKLOG`.
- Shared-deadline multi-subcommand hook dispatcher — `BACKLOG`
  (recall-hook budget discipline already applied).

### edouard-claude/snip (MIT) — CORE PLANNED
- Filter catalog + pipeline actions (keep_lines/truncate/json_extract/
  aggregate/dedup/...) — `PLANNED` (proxy plan Tasks 3/8, MIT code portable
  with attribution).
- Runner prefixes (`uv run pytest` → pytest) — `BACKLOG` (cheap matcher
  addition).
- SHA-256 trust store for project filters + inline filter tests
  (`snip verify` — expected-in/out per filter) — `BACKLOG` (regression-safe
  catalog in CI).
- `discover` missed-savings sweep over agent history — `BACKLOG`.

### headroomlabs-ai/headroom (Apache-2.0)
- CCR reversible compression + `retrieve` — `PLANNED` (proxy plan Task 4).
- Live-zone/cache-aligned compression — `PLANNED` (plan Task 3 zone rule).
- ContentRouter per-type dispatch — `PLANNED` (Task 6).
- Failure-mining `learn` (contradicted/ignored recalls → corrections) —
  `BACKLOG` (pairs with idle capture; markdown-truth makes it cheap).
- Output holdout measurement — `PLANNED` (plan Task 5).

### elusznik/mcp-server-code-execution-mode (GPL-3.0 — ideas only)
- Two-stage schema hydration / `detail="full"` escalation — `BACKLOG`
  (terse briefing/search outputs by default).
- Atomic functional memory update (`update_memory(key, fn)`) —
  `BACKLOG` (cheaper than read-modify-write for the save tool).
- `runtime.capability_summary()`-style `memo_about` — `BACKLOG`.

### jgravelle/jcodemunch-mcp (dual-use — ideas only)
- MUNCH compact line-encoding (interned prefixes, typed scalars, ≥15%
  savings threshold, fail-open `json`) — `BACKLOG`; direct fit for the
  recall-hook hot path (plan Task 6 encoding layer).
- Budgeted multi-source assembly (`assemble_task_context`) — `BACKLOG`
  (briefing gains a token-budget parameter).
- `stop_rule.terminal` verdicts (when re-querying cannot change the answer)
  — `BACKLOG` (reduces repeated hook recalls).
- Calibrated freshness/confidence flags on hits — `BACKLOG` (memo has
  scores; add staleness annotations + "searched, not found" coverage).

### juyterman1000/entroly (Apache-2.0)
- Picks-first-shrinks-second selection under an explicit budget —
  `PLANNED` (plan Task 6 shape).
- Context Receipts (kept/omitted/why/hash per selection) — `BACKLOG`
  (readable as `memo context --receipt` or `_meta` on recall).
- `simulate` offline budget preview — `BACKLOG` (CLI dry-run of briefing
  budget over the live vault).
- WITNESS evidence grounding — `BACKLOG` (verifiable against markdown
  source; pairs with receipts).

### pleasedodisturb/awesome-llm-token-optimization (unlicensed — ideas)
- Lossless distillation rules (strip prose transitions/hedging; preserve
  numbers/entities/decisions/constraints/risks; dense bullets) —
  `BACKLOG` as a rubric for the crusher and briefing prompts.
- Cached Prefix Pattern (stable system+profile first ~2k tokens) —
  `PLANNED` (plan Task 3 zones; sizing heuristic input).
- Reranker-pruning stage on recall candidates — `BACKLOG`
  (OpenProvence-style; MLX cross-encoder already in place).
- Browser-tool efficiency data — N/A for memo (recall hooks already
  cap injection ~160 tok).

## Next moves (priority order, all BACKLOG unless stated)

1. Loop/spin detection in the Stop hook (token_meter transcript pass, zero
   LLM) — highest practical token lever before the proxy lands.
2. TOON/MUNCH compact encoding behind the plan's Task 6 transform layer.
3. Billing-block + per-model dollar views once the pricing snapshot lands
   (design §6 deletes the estimated roi panel first).
4. Context receipts (entroly) as `_meta` on recall + `memo context`.
5. Decision checkpoints/restore + distillation-rubric crusher prompt.