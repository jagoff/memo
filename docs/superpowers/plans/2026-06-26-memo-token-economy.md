# Memo Token Economy — Compact Recall, Trivial Gate, Compress-Context

**Branch:** master  
**Base commit:** 7fb044c1bcad547563af5b954fa7c7701a203e68  
**Goal:** Reduce tokens consumed in each agent session using three orthogonal techniques inspired by RTK (output compression), caveman (stylistic compression), and ponytail (reuse-first). Ship stats proving the savings.

---

## Global Constraints

- Python 3.11+, uv, ruff + mypy must stay green
- Tests: `uv run --no-sync pytest tests/`
- Flags go in the matching `flags_<group>.py` file; access via `flag_bool/int/float/str`
- Never `os.environ.get("MEMO_...")` inline
- Markdown is source of truth; sqlite is rebuildable index (no changes to storage)
- All new CLI commands registered in `src/memo/cli.py`
- `MEMO_RECALL_FORMAT=compact` must default `off` (`full` stays default) — opt-in only
- `MEMO_RECALL_TRIVIAL_BAIL` defaults `True` (opt-out, safe improvement)
- `memo compress-context` is rule-based only (no MLX LLM calls) — deterministic + testable
- No version bump in these tasks (separate final step)

---

## Task 1 — Compact Recall Format

**Files:** `src/memo/flags_recall.py`, `src/memo/recall_logic.py`, `src/memo/cli_recall_hook.py`, `tests/test_recall_hooks.py`

Add `MEMO_RECALL_FORMAT` flag (`str`, default `"full"`, options `"full"|"compact"`).

When `compact`, `render_recall_context` emits a much smaller block:

```
<memo-recall readonly>
[57d1d1cf] capture-daemon · client-agnostic launchd idle-monitor
[005b5101] hanging: embedder dim mismatch 2560 vs 1024 → ingest
[03404929] versioning: auto-patch bump + auto-update on mcp start
</memo-recall>
```

Rules for compact:
- No `## Memory` header, no directive line, no score tag, no tags line, no body, no `_Full: …_` footer
- One line per hit: `[id8] title · first 60 chars of body (if body exists, stripped to single line)`
- Wrapped in `<memo-recall readonly>` / `</memo-recall>` only
- Token budget still applies (drop tail hits if over budget)

Add `render_recall_compact(relevant, *, token_budget)` to `recall_logic.py`.

Wire in `cli_recall_hook.py`: read `MEMO_RECALL_FORMAT`, call the right renderer.

**Tests** (in `tests/test_recall_hooks.py`):
- compact format emits exactly one `[id8]` line per hit, no headers/tags/scores
- compact respects token_budget (drops tail hits)
- full format unchanged (regression)
- measure: compact block for 3 hits is ≤30% chars of full block for same hits

---

## Task 2 — Trivial Prompt Gate

**Files:** `src/memo/flags_recall.py`, `src/memo/cli_recall_hook.py`, `tests/test_recall_hooks.py`

Add `MEMO_RECALL_TRIVIAL_BAIL` flag (`bool`, default `True`, opt-out).

A prompt is **trivial** if ALL of:
- After stripping punctuation, it contains ≤ 3 words
- The normalized prompt matches any word in the hardcoded trivial set

Hardcoded trivial set (lowercase, strip punctuation before compare):
```
yes no ok sure yep nope continue go ahead proceed
sí si dale listo gracias thanks k cool perfect
```

If trivial → `_bail("trivial prompt")` (same path as existing char-gate bail).

The gate runs AFTER the existing char-length gate (at line 100), before any search.

**Tests** (append to `tests/test_recall_hooks.py`):
- "sí" → bail trivial
- "yes please" → bail trivial (2 words, "yes" in set)
- "yes, please implement the auth module" → NOT trivial (>3 words)
- "ok" → bail trivial
- gate disabled (`MEMO_RECALL_TRIVIAL_BAIL=0`) → no bail on "ok"

---

## Task 3 — `memo compress-context` Command

**Files:** `src/memo/cli_compress_context.py` (new), `src/memo/cli.py`, `tests/test_compress_context.py` (new)

New CLI command: `memo compress-context <path> [--dry-run] [--backup]`

Rule-based compression (NO LLM, deterministic):

1. **Remove decorative markdown:** strip `---` horizontal rules, blank lines between single-line items, trailing spaces
2. **Collapse verbose lists:** if a markdown list item is >120 chars, truncate at last word boundary before 120 chars and append `…`
3. **Compress blockquotes:** `> text` → keep only first 100 chars per blockquote line
4. **Remove comment-only lines:** lines matching `^<!--.*-->$`
5. **Collapse 2+ consecutive blank lines → 1**
6. **Strip trailing whitespace per line**

Options:
- `--dry-run`: print compressed output to stdout, don't write
- `--backup`: save original as `<path>.orig` before overwriting (default: no backup)

Output: `Compressed <path>: NNN → MMM chars (XX% reduction)`

**Tests** (`tests/test_compress_context.py`):
- Horizontal rules removed
- Long list items truncated at 120 chars
- Blockquotes truncated at 100 chars
- Multiple blank lines collapsed
- `--dry-run` doesn't write file
- `--backup` creates `.orig`
- Idempotent: compressing twice yields same result

---

## Task 4 — Token Savings Stats + README

**Files:** `src/memo/cli_search.py` (or `cli_misc.py`), `tests/test_server.py` or `tests/test_flags.py`, `README.md`

### 4a: `memo token-savings` CLI command

Reads `context_cost_log` (already populated by `append_context_cost_log` in dashboard) and prints:

```
memo token savings (last 7 days)

  Recall injections:    47 prompts
  Context chars:     18,432 avg per session
  Compact savings:    ~65%  (if MEMO_RECALL_FORMAT=compact)
  Trivial bails:         8  (prompts skipped)
  
  Estimated total:  ~12,000 tokens saved vs. model rederiving context
  
  Run: MEMO_RECALL_FORMAT=compact memo recall-hook to enable compact mode.
```

The command reads `context_cost_log` from `state_dir`. If no log exists, prints a helpful message pointing to `memo stats`.

The "Estimated total tokens saved" uses the existing ROI formula already in `memo roi`.

Implementation: add `token_savings` Click command to `src/memo/cli_search.py` (it's the closest domain) OR create `src/memo/cli_token_savings.py`.

### 4b: README update

Add a **"Token savings"** section (after the existing ROI bullet, before or inside the "Why it saves tokens" block). Include:

- Compact recall format: ~65% fewer tokens in injected recall block (160-token budget → ~56 tokens)
- Trivial prompt gate: skips recall on confirmation prompts (saves 1 full recall injection per ~4 prompts in typical sessions)
- compress-context: one-time CLAUDE.md compression → ~30-40% smaller context file

Include a small table:

```
| Technique              | Where                    | Typical saving |
|------------------------|--------------------------|----------------|
| Compact recall format  | MEMO_RECALL_FORMAT=compact | ~65% per injection |
| Trivial prompt gate    | on by default            | ~25% fewer injections |
| compress-context       | one-time CLAUDE.md shrink | ~30-40% context file |
```

**Tests:** `memo token-savings` command exists, runs without error on an empty log, outputs the header line.

---

## Final Steps (not tasks — controller does these)

1. Run full test suite green: `uv run --no-sync pytest tests/ --tb=short`
2. Bump version to 1.1.2 via `release-bumper` agent
3. Commit CHANGELOG entry
4. `git push && git tag v1.1.2 && git push --tags`
