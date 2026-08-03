# Dream HyDE-Tune + Graph-Hygiene Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn on three dormant, manual-kind memo dream self-improvement
passes (`MEMO_DREAM_HYDE_TUNE_ENABLED`, `MEMO_DREAM_ENTITY_CANON_ENABLED`,
`MEMO_DREAM_EDGE_VERIFY_ENABLED`) for the nightly `com.memo.dream`
LaunchAgent, verified both in isolation and end-to-end.

**Architecture:** Config-only change. Add the three flags to the repo's
committed LaunchAgent plist *template* (which still carries
`__HOME__`/`__MEMO_BIN__` placeholders, same as every other agent in the
fleet), re-render the deployed copy under `~/Library/LaunchAgents/`, and
reload the agent. No memo source code changes — the passes and their flag
gates already exist and ship in the installed `mlx-memo` build.

**Tech Stack:** macOS `launchd` (plist XML, `launchctl`, `plutil`,
`PlistBuddy`), the installed `memo` CLI (`~/.local/bin/memo`, uv-tool
isolated runtime — never the repo's `.venv`).

## Global Constraints

- Config/plist only — no edits under `src/memo/`. The three flags and their
  passes already exist and are already gated in `cli_dream.py`.
- Do not add a markdown-config dotted key for these flags (spec explicitly
  scoped this out as a follow-up) — the LaunchAgent's own
  `EnvironmentVariables` is the mechanism.
- Do not touch any other `EnvironmentVariables` entry already in the plist
  (`MEMO_DREAM_TUNE_ENABLED`, `MEMO_BELIEF_NWAY`, etc.) — additive only.
- Preserve the `__HOME__` / `__MEMO_BIN__` placeholders in the **repo**
  template file — only the rendered copy under `~/Library/LaunchAgents/`
  gets real paths substituted in.
- The repo working tree is shared across concurrent agent sessions
  (`~/repos/memo/CLAUDE.md`) — stage only `launchd/com.memo.dream.plist` by
  exact path when committing, never `git add -A`.
- `memo` must resolve from the isolated uv-tool runtime
  (`~/.local/bin/memo`), never a repo `.venv` — every verification command
  below assumes plain `memo ...`, not `uv run memo ...`.
- Before running any verification command in this plan, run
  `memo doctor 2>&1 | grep -i "outside the isolated runtime"`. If it warns,
  `unset PYTHONPATH` in that shell first — a `PYTHONPATH=src` (relative to
  a `~/repos/memo` cwd) silently shadows the isolated install with the
  working-tree source, so `memo --version` looks right while the code that
  actually runs is stale/local (documented gotcha, not specific to this
  change, but every command below depends on running the real installed
  build).

---

## File Structure

- **Modify (repo, tracked):** `launchd/com.memo.dream.plist` — add three
  `<key>`/`<string>` pairs to the existing `EnvironmentVariables` dict.
- **Modify (machine-local, untracked):**
  `~/Library/LaunchAgents/com.memo.dream.plist` — re-rendered from the
  template above; this is the file `launchd` actually loads.
- **No new files.**

---

### Task 1: Add the three flags to the plist template

**Files:**
- Modify: `launchd/com.memo.dream.plist:59-62`

**Interfaces:**
- Produces: an updated template, still containing the literal placeholder
  strings `__HOME__` and `__MEMO_BIN__`, with three new
  `EnvironmentVariables` entries (`MEMO_DREAM_HYDE_TUNE_ENABLED`,
  `MEMO_DREAM_ENTITY_CANON_ENABLED`, `MEMO_DREAM_EDGE_VERIFY_ENABLED`, all
  `"1"`). Task 3 renders this file into the deployed copy.

- [ ] **Step 1: Edit the template**

Insert the three new entries right after the existing
`MEMO_DREAM_CONSOLIDATE_EPISODES_ENABLED` pair, matching the file's existing
comment style:

```xml
    <key>MEMO_DREAM_ANTICIPATE_ENABLED</key>
    <string>1</string>
    <key>MEMO_DREAM_COMMUNITIES_ENABLED</key>
    <string>1</string>
    <key>MEMO_DREAM_CONSOLIDATE_EPISODES_ENABLED</key>
    <string>1</string>
    <!-- Dream v2 self-improvement passes (batch 2026-08-03): nightly HyDE A/B
         (self-applying/self-reverting via the tuned overlay, gated on the
         curated eval set), MinHash-blocked LLM entity-dedup (bounded 30
         pairs/night), and grounded co-use edge-confidence curation
         (never touches recall ranking, never deletes). -->
    <key>MEMO_DREAM_HYDE_TUNE_ENABLED</key>
    <string>1</string>
    <key>MEMO_DREAM_ENTITY_CANON_ENABLED</key>
    <string>1</string>
    <key>MEMO_DREAM_EDGE_VERIFY_ENABLED</key>
    <string>1</string>
```

- [ ] **Step 2: Verify the XML is well-formed**

Run: `plutil -lint launchd/com.memo.dream.plist`
Expected: `launchd/com.memo.dream.plist: OK`

- [ ] **Step 3: Verify the three keys read back correctly**

Run:
```bash
for k in MEMO_DREAM_HYDE_TUNE_ENABLED MEMO_DREAM_ENTITY_CANON_ENABLED MEMO_DREAM_EDGE_VERIFY_ENABLED; do
  /usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:$k" launchd/com.memo.dream.plist
done
```
Expected: three lines, each `1`.

- [ ] **Step 4: Verify the placeholders are untouched**

Run: `grep -c '__HOME__\|__MEMO_BIN__' launchd/com.memo.dream.plist`
Expected: `3` (one `__MEMO_BIN__` in `ProgramArguments`, two `__HOME__` in
the log paths — unchanged from before this edit).

- [ ] **Step 5: Commit**

```bash
git add launchd/com.memo.dream.plist
git commit -m "$(cat <<'EOF'
chore(launchd): activate HyDE-tune + graph-hygiene dream passes

Add MEMO_DREAM_HYDE_TUNE_ENABLED, MEMO_DREAM_ENTITY_CANON_ENABLED, and
MEMO_DREAM_EDGE_VERIFY_ENABLED to the nightly dream LaunchAgent's
EnvironmentVariables. All three are manual-kind gates in
dream_flags.GATES that the graduation pipeline never flips on its own.
See docs/superpowers/specs/2026-08-03-dream-hyde-graph-hygiene-design.md.
EOF
)"
```

---

### Task 2: Isolated functional smoke test (dry-run, before touching the live agent)

Proves the three passes actually run and produce the expected receipt shape
— cheap, no mutation, no LaunchAgent involvement yet. Runs against the real
installed `memo` and the real local corpus, in dry-run mode.

**Files:** none (verification only).

**Interfaces:**
- Consumes: nothing from Task 1's file changes directly (the flags are
  exported in-shell here, not yet read from the deployed plist — that
  wiring is proven in Task 4).
- Produces: confirmation that `entity_canon`, `edge_verify`, and
  `hyde_tuner` all appear in the pipeline receipt with `status` other than
  `"disabled"`/`"skipped"`/`"error"`, before spending effort deploying.

- [ ] **Step 1: Run the full pipeline in dry-run with only the three new flags set**

```bash
MEMO_DREAM_HYDE_TUNE_ENABLED=1 \
MEMO_DREAM_ENTITY_CANON_ENABLED=1 \
MEMO_DREAM_EDGE_VERIFY_ENABLED=1 \
memo dream run --dry-run --json > /tmp/dream-smoke.json
```

- [ ] **Step 2: Verify entity-canon ran**

Run: `python3 -c "import json; d=json.load(open('/tmp/dream-smoke.json')); print(d.get('entity_canon'))"`
Expected: a dict with `status` equal to `"done"` or `"skipped"` (skipped
only if the graph currently has zero entities — not expected here, the last
receipt showed 245) — must NOT be `"disabled"` (that would mean the flag
wasn't read) or `"error"`.

- [ ] **Step 3: Verify edge-verify ran**

Run: `python3 -c "import json; d=json.load(open('/tmp/dream-smoke.json')); print(d.get('edge_verify'))"`
Expected: a dict with `status` in `{"done", "noop", "skipped"}` — must NOT
be `"error"`.

- [ ] **Step 4: Verify hyde-tuner ran**

Run: `python3 -c "import json; d=json.load(open('/tmp/dream-smoke.json')); print(d.get('hyde_tuner'))"`
Expected: a dict with a `status` key present — must not be absent (absence
would mean `MEMO_DREAM_HYDE_TUNE_ENABLED` wasn't read as true).

- [ ] **Step 5: Confirm no NEW errors from these three passes**

Run: `python3 -c "import json; d=json.load(open('/tmp/dream-smoke.json')); print([e for e in d.get('errors', []) if any(p in e for p in ('entity_canon', 'edge_verify', 'hyde_tuner'))])"`
Expected: `[]` (empty list). The pre-existing unrelated
`contradict: FileNotFoundError` warning (missing WhatsApp vault chunk,
tracked separately, out of scope for this plan) may still appear in
`errors` for a different key — that is expected and not a regression.

If any of Steps 2-5 fail, stop and fix the plist/flag before proceeding to
Task 3 — do not deploy an unverified configuration.

---

### Task 3: Render and deploy the LaunchAgent, reload

**Files:**
- Modify (machine-local, untracked): `~/Library/LaunchAgents/com.memo.dream.plist`

**Interfaces:**
- Consumes: `launchd/com.memo.dream.plist` from Task 1 (the template with
  placeholders + the three new flags).
- Produces: a loaded `com.memo.dream` LaunchAgent whose live
  `EnvironmentVariables` include the three new flags with real paths
  substituted for `__HOME__`/`__MEMO_BIN__`. Task 4 exercises this loaded
  agent.

- [ ] **Step 1: Render the template into the deployed path**

```bash
MEMO_BIN="$(command -v memo)"
sed -e "s#__HOME__#${HOME}#g" -e "s#__MEMO_BIN__#${MEMO_BIN}#g" \
  launchd/com.memo.dream.plist > ~/Library/LaunchAgents/com.memo.dream.plist
```

- [ ] **Step 2: Verify the rendered plist is well-formed**

Run: `plutil -lint ~/Library/LaunchAgents/com.memo.dream.plist`
Expected: `~/Library/LaunchAgents/com.memo.dream.plist: OK`

- [ ] **Step 3: Verify no placeholders leaked through**

Run: `grep -c '__HOME__\|__MEMO_BIN__' ~/Library/LaunchAgents/com.memo.dream.plist`
Expected: `0`

- [ ] **Step 4: Verify the three flags are present with real substitution intact**

```bash
for k in MEMO_DREAM_HYDE_TUNE_ENABLED MEMO_DREAM_ENTITY_CANON_ENABLED MEMO_DREAM_EDGE_VERIFY_ENABLED; do
  /usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:$k" ~/Library/LaunchAgents/com.memo.dream.plist
done
/usr/libexec/PlistBuddy -c "Print :ProgramArguments:0" ~/Library/LaunchAgents/com.memo.dream.plist
```
Expected: three lines of `1`, then the last line prints the absolute path
to the installed `memo` binary (matching `command -v memo`), not the
literal string `__MEMO_BIN__`.

- [ ] **Step 5: Reload the agent**

```bash
launchctl bootout gui/$(id -u)/com.memo.dream 2>/dev/null
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.memo.dream.plist
```
(A non-zero exit / "No such process" from `bootout` is expected and
harmless if the agent wasn't currently loaded — the goal is that
`bootstrap` succeeds.)

- [ ] **Step 6: Verify the agent is loaded**

Run: `launchctl list | grep com.memo.dream`
Expected: one line starting with a PID-or-`-` column, an exit-status
column, then `com.memo.dream` — presence of the line proves it loaded; a
`-` PID just means it isn't running at this instant (expected, it's a
`StartCalendarInterval` job, not `KeepAlive`).

---

### Task 4: End-to-end live verification

Proves the deployed LaunchAgent's environment actually reaches the three
passes when `launchd` itself invokes `memo dream run` — the thing Task 2
could not prove (Task 2 exported the flags in an interactive shell, not via
`launchd`).

**Files:** none (verification only).

**Interfaces:**
- Consumes: the loaded agent from Task 3.
- Produces: a real (non-dry-run) dream receipt confirming the three passes
  ran under `launchd`'s own environment.

- [ ] **Step 1: Trigger an immediate run instead of waiting for 03:00**

```bash
launchctl kickstart -k gui/$(id -u)/com.memo.dream
```

- [ ] **Step 2: Wait for completion**

Run: `tail -f ~/Library/Logs/memo/dream.log`
Wait until the log stops updating for at least 30 seconds (the pipeline
includes real MLX chat calls for entity-canon and HyDE — this can take
several minutes on a first run). Press Ctrl-C once it looks idle.

- [ ] **Step 3: Check the receipt via the CLI**

Run: `memo dream status`
Expected: a `last dream run:` timestamp matching the time you just
triggered (not the prior `2026-08-02 21:55` run), and no new `warn:` lines
mentioning `entity_canon`, `edge_verify`, or `hyde_tuner`. The pre-existing
`contradict: FileNotFoundError` warning may still be present — expected,
out of scope.

- [ ] **Step 4: Confirm all three passes reported success in the run log**

Run: `grep -E '\[entity-canon\]|\[edge-verify\]|\[tune\] hyde A/B' ~/Library/Logs/memo/dream.log | tail -3`
Expected: three lines (one per pass), each containing a checkmark
(`✓`), matching the progress text emitted by `cli_dream.py`'s `run`
command (`"[entity-canon] ✓  N LLM calls vs M naive"`,
`"[edge-verify] ✓  N promoted, M decayed"`,
`"[tune] hyde A/B ✓  <status>"`). A line ending in `warn` instead of `✓`
means that pass raised an exception — check `memo dream status` for the
matching `errors` entry and stop.

If Step 3 or 4 shows a new error tied to one of the three passes: revert by
removing the three lines from `launchd/com.memo.dream.plist`, re-run Task 3
Steps 1, 2, 5, 6 to redeploy the reverted template, and stop — do not
proceed further until the failure is understood.

- [ ] **Step 5: No commit needed**

This task only exercises the already-committed Task 1 change and the
machine-local deployment from Task 3 — nothing new to commit.
