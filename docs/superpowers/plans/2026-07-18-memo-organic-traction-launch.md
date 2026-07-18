# Memo Organic Traction Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a free, founder-led launch that gives memo its best credible chance of reaching 100 total GitHub stars within seven days.

**Architecture:** Treat the campaign as five bounded units: evidence, channel copy, readiness, publication/response, and measurement. A single Markdown tracker is the control plane; sanitized media lives in `memo-web`, while copy and aggregate results live in `memo`. No cross-posting bot or new analytics service is introduced.

**Tech Stack:** Markdown, git/GitHub CLI, memo CLI/MCP, Vercel CLI and Web Analytics, Astro/Playwright, `ffmpeg`/`ffprobe`, macOS `sips`, Docker, and authenticated platform web interfaces.

## Global Constraints

- Source design: `docs/superpowers/specs/2026-07-18-memo-organic-traction-launch-design.md`.
- Use a local `launch/2026-07-memo` branch in both `memo` and `memo-web`, created through `superpowers:using-git-worktrees` at execution time.
- Budget is free and organic only; do not buy ads, placements, directory upgrades, or promotion.
- Primary KPI is 100 total GitHub stars by day seven; capture a fresh baseline immediately before launch.
- Core English posts share one launch day but use channel-native copy and staggered publication times.
- The Spanish wave runs about 48 hours later unless the +24-hour contingency advances it.
- Do not solicit coordinated votes, trade engagement, create duplicate accounts, evade moderation, or mass-message strangers.
- The initial post bundle requires one batch approval from the user; every ordinary reply requires separate user approval.
- Hacker News and DEV comments are human-authored by the user. The assistant supplies facts and an outline, not final comment prose.
- Use personal founder accounts. The user personally completes passwords, email links, CAPTCHA, 2FA, identity checks, and acceptance of platform terms.
- Store no credential, session cookie, recovery code, API token, private memory, or private account detail in git, memo, screenshots, or launch files.
- Use isolated synthetic memo data in every demo and screenshot.
- Product Hunt must confirm the personal account meets its current one-week age rule. If it does not, the full launch moves to the next Tuesday rather than launching partially.
- No product feature work is allowed solely for the campaign. A critical product defect pauses publication and is fixed through the normal memo workflow.
- Treat existing MCP Registry, Glama, mcpservers.org, and awesome-mcp-servers entries as refresh surfaces, not new submissions.
- Consult memo with `source="codex"` before campaign decisions. After significant units, call `memo_idle_capture`, then `memo_pop_notification`; always pop notifications before user-facing completion messages.

---

## File Map

### `memo`

- `docs/superpowers/specs/2026-07-18-memo-organic-traction-launch-design.md` — approved source of truth; change only to correct verified facts.
- `docs/superpowers/plans/2026-07-18-memo-organic-traction-launch.md` — this execution plan.
- `docs/diagram-loop.svg` — existing capture → Markdown/SQLite → cross-agent recall diagram reused in the technical article.
- `docs/diagram-recall.svg` — existing hybrid-retrieval diagram reused in the technical article.
- `docs/launch/2026-07/tracker.md` — account readiness, policy decisions, evidence, asset hashes, approvals, publication URLs, response queue, and gate status.
- `docs/launch/2026-07/copy.md` — the approved English and Spanish initial post bundle plus platform-specific reply constraints.
- `docs/launch/2026-07/metrics.csv` — aggregate checkpoint snapshots only.
- `docs/launch/2026-07/postmortem.md` — day-seven outcome and lessons.

### `memo-web`

- `public/launch/demo.mp4` — 30–45 second sanitized cross-client demonstration.
- `public/launch/demo-es.srt` — reviewed Spanish subtitle source for the demonstration.
- `public/launch/demo-es.mp4` — demonstration with Spanish subtitles burned in.
- `public/launch/social-card.png` — 1200×630 shared social image.
- `public/launch/product-hunt-thumbnail.png` — 240×240 Product Hunt thumbnail.
- `public/launch/product-hunt-gallery-01.png` — 1270×760 landing hero gallery image.
- `public/launch/product-hunt-gallery-02.png` — 1270×760 product-detail gallery image.
- `public/launch/tokens.png` — sanitized current `memo tokens` terminal capture.
- `public/launch/markdown-history.png` — sanitized synthetic Markdown/history capture.

No production TypeScript, Astro, Python, database, or analytics code is changed by this campaign.

---

### Task 1: Establish the launch control plane and account inventory

**Files:**
- Create: `docs/launch/2026-07/tracker.md`
- Create: `docs/launch/2026-07/metrics.csv`

**Interfaces:**
- Consumes: approved design, current authenticated browser sessions, current platform rules.
- Produces: `LaunchDate`, `AccountStatus`, `RuleDecision`, and status enums used by every later task.

- [ ] **Step 1: Create isolated launch worktrees**

Invoke `superpowers:using-git-worktrees` and create:

```text
/Users/fer/repos/memo/.worktrees/launch-2026-07
  branch: launch/2026-07-memo
  base: current local master (includes the approved spec and plan)

/Users/fer/repos/memo-web/.worktrees/launch-2026-07
  branch: launch/2026-07-memo
  base: origin/main
```

Run:

```bash
git -C /Users/fer/repos/memo/.worktrees/launch-2026-07 status --short --branch
git -C /Users/fer/repos/memo-web/.worktrees/launch-2026-07 status --short --branch
```

Expected: both commands name `launch/2026-07-memo` and show no uncommitted files.

- [ ] **Step 2: Create the tracker with explicit state machines**

Use `apply_patch` to create `docs/launch/2026-07/tracker.md` with this content:

```markdown
# memo launch tracker — July 2026

## Status vocabulary

- Account/policy/readiness: `not_checked`, `blocked_user`, `blocked_platform`, `ready`, `skipped_policy`, `failed`.
- Copy/asset approval: `pending`, `approved`, `changes_requested`.
- Publication: `not_scheduled`, `scheduled`, `published`, `skipped_policy`, `failed`.

## Campaign decision

- Readiness deadline: Monday 18:00 America/Argentina/Cordoba.
- Launch date: derived from Product Hunt eligibility and the readiness gate.
- Primary KPI: 100 total GitHub stars by day seven.
- Primary language: English.
- Spanish wave: launch +48 hours unless advanced by the +24-hour contingency.

## Accounts

| Channel | Required | Account/profile URL | Created date | Access status | Human verification | Evidence |
|---|---:|---|---|---|---|---|
| GitHub | yes | public founder profile | public date or `verified_not_recorded` | not_checked | not_checked | public profile URL |
| Vercel | yes | public production deployment | public date or `verified_not_recorded` | not_checked | not_checked | public deployment URL |
| Product Hunt | yes | personal maker account | public date or `>=7_days_verified` | not_checked | not_checked | public profile URL |
| Hacker News | yes | personal account | public date or `verified_not_recorded` | not_checked | not_checked | public profile URL |
| LinkedIn | yes | personal founder account | public date or `verified_not_recorded` | not_checked | not_checked | public profile URL |
| X | yes | personal founder account | public date or `verified_not_recorded` | not_checked | not_checked | public profile URL |
| Reddit | yes | personal account | public date or `verified_not_recorded` | not_checked | not_checked | public profile URL |
| DEV Community | yes | personal account | public date or `verified_not_recorded` | not_checked | not_checked | public profile URL |
| Peerlist | reserve | personal account | public date or `verified_not_recorded` | not_checked | not_checked | public profile URL |
| Indie Hackers | reserve | personal account | public date or `verified_not_recorded` | not_checked | not_checked | public profile URL |
| Hashnode | reserve | personal blog account | public date or `verified_not_recorded` | not_checked | not_checked | public blog/profile URL |

## Reddit rule decisions

| Community | Relevance | Rule URL | Account threshold | Self-promotion decision | Moderator contact state | Status |
|---|---|---|---|---|---|---|
| r/LocalLLaMA | local execution/privacy | record current URL | record current threshold | record explicit rule | `not_needed`, `sent_utc`, or `answered` | not_checked |
| r/selfhosted | user-controlled local service | record current URL | record current threshold | record explicit rule | `not_needed`, `sent_utc`, or `answered` | not_checked |
| r/opensource | MIT/open architecture | record current URL | record current threshold | record explicit rule | `not_needed`, `sent_utc`, or `answered` | not_checked |
| r/ObsidianMD | Markdown interoperability only | record current URL | record current threshold | record explicit rule | `not_needed`, `sent_utc`, or `answered` | not_checked |
| agent/MCP candidate | confirm community identity first | record current URL | record current threshold | record explicit rule | `not_needed`, `sent_utc`, or `answered` | not_checked |

## MCP discovery

| Surface | Expected existing state | Current evidence | Refresh needed | Status |
|---|---|---|---|---|
| Official MCP Registry | `io.github.jagoff/memo`, 3.7.0 latest | record API output | no unless stale | not_checked |
| Glama | existing `jagoff/memo` page | record URL and visible version/copy | only if stale | not_checked |
| mcpservers.org | existing `jagoff/memo` page | record URL and visible stats | likely stats refresh | not_checked |
| awesome-mcp-servers | existing Knowledge & Memory line | record raw README line | only if stale | not_checked |
| PulseMCP | verify presence | record search result | only through official free route | not_checked |

## Product readiness

| Check | Evidence | Status |
|---|---|---|
| memo public release/CI | command or run URL | not_checked |
| macOS isolated install | command log path | not_checked |
| Linux/CPU isolated smoke | command log path | not_checked |
| landing `/` and `/es/` | HTTP/browser evidence | not_checked |
| memo-web CI/build/E2E/a11y | command or run URL | not_checked |
| Vercel production/Analytics | deployment and dashboard evidence | not_checked |

## Verified claims

| Claim key | Exact value | Captured at UTC | Evidence command/source | Copy wording |
|---|---|---|---|---|
| tokens_historic | record command result | record timestamp | `memo tokens --json` | “memo's counter estimates … tokens of repeated work avoided” |
| tokens_month | record command result | record timestamp | `memo tokens --json` | “this month …” |
| github_stars | record command result | record timestamp | GitHub repository API | exact count |
| github_forks | record command result | record timestamp | GitHub repository API | exact count if used |
| registry_latest | record command result | record timestamp | MCP Registry API | exact version |

## Assets

| Asset | Required dimensions/duration | SHA-256 | Privacy review | Approval | Status |
|---|---|---|---|---|---|
| demo.mp4 | 30–45 seconds | record hash | pending | pending | not_checked |
| demo-es.srt | matches approved video timing | record hash | pending | pending | not_checked |
| demo-es.mp4 | 30–45 seconds, burned Spanish subtitles | record hash | pending | pending | not_checked |
| social-card.png | 1200×630 | record hash | pending | pending | not_checked |
| product-hunt-thumbnail.png | 240×240 | record hash | pending | pending | not_checked |
| product-hunt-gallery-01.png | 1270×760 | record hash | pending | pending | not_checked |
| product-hunt-gallery-02.png | 1270×760 | record hash | pending | pending | not_checked |
| tokens.png | readable, sanitized | record hash | pending | pending | not_checked |
| markdown-history.png | readable, synthetic | record hash | pending | pending | not_checked |

## Initial publication queue

| Order | Channel | Scheduled time | Destination | Copy approval | Draft status | Publication status | Published URL |
|---:|---|---|---|---|---|---|---|
| 1 | Product Hunt | 00:01 Pacific | landing | pending | not_scheduled | not_scheduled | none before publication |
| 2 | LinkedIn | 09:30 ART | landing | pending | not_scheduled | not_scheduled | none before publication |
| 3 | X | 10:15 ART | landing + GitHub | pending | not_scheduled | not_scheduled | none before publication |
| 4 | Show HN | 11:00 ART | GitHub | pending | not_scheduled | not_scheduled | none before publication |
| 5 | DEV | 12:30 ART | landing + GitHub | pending | not_scheduled | not_scheduled | none before publication |
| 6 | Reddit | 14:00–17:00 ART | GitHub | pending | not_scheduled | not_scheduled | none before publication |
| 7 | MCP discovery refresh | 17:00–19:00 ART | directory pages | pending | not_scheduled | not_scheduled | none before publication |

## Readiness gate

| Gate | Required state | Current state | Evidence |
|---|---|---|---|
| Product | all product checks `ready` | not_checked | record exact evidence |
| Claims | captured on final preparation day | not_checked | record exact evidence |
| Assets | all required assets `approved` | not_checked | record exact evidence |
| Accounts | all core accounts `ready` | not_checked | record exact evidence |
| Policy | every selected community `ready` or `skipped_policy` | not_checked | record exact evidence |
| Copy | every initial row `approved` | not_checked | record user approval message |
| Measurement | baseline row written | not_checked | record metrics commit |
| Human availability | user confirms launch window | not_checked | record confirmation |

## Response queue

| Channel | Comment permalink | Topic | Fact check | Draft | User approval | Reply status | Published reply URL |
|---|---|---|---|---|---|---|---|

## Checkpoint decisions

| Checkpoint | Star target | Actual stars | Diagnostic summary | Allowed action | Chosen action |
|---|---:|---:|---|---|---|
| +6h | calculate from baseline | record at checkpoint | record aggregates | improve owned hook/proof only | record action |
| +24h | calculate from baseline | record at checkpoint | record aggregates | Spanish advance OR one reserve surface | record action |
| D3 | calculate from baseline | record at checkpoint | record aggregates | one new technical lesson on owned channel | record action |
| D5 | calculate from baseline | record at checkpoint | record aggregates | measure/respond only | record action |
| D7 | 100 | record at checkpoint | final aggregate | postmortem | record action |
```

Expected: the tracker defines every status; it contains no passwords, email addresses, private account IDs, or blank policy assumptions.

- [ ] **Step 3: Create the aggregate metric schema**

Use `apply_patch` to create `docs/launch/2026-07/metrics.csv` with exactly this header:

```csv
captured_at_utc,checkpoint,github_stars,github_views_14d,github_unique_visitors_14d,github_clones_14d,github_unique_cloners_14d,github_top_referrers,vercel_visitors,vercel_pageviews,linkedin_impressions,x_impressions,reddit_impressions,dev_views,product_hunt_visits,meaningful_comments,issues,contributions,notes
```

Expected: one header row and no personal identifiers.

- [ ] **Step 4: Audit existing core and reserve accounts**

Open these authenticated surfaces one at a time and record only the public profile URL, a creation date when already public, and one status enum. For a non-public creation date, record only `verified_not_recorded`; for a non-public authenticated dashboard, record `browser_verified` without its URL:

```text
https://github.com/
https://vercel.com/
https://www.producthunt.com/
https://news.ycombinator.com/login
https://www.linkedin.com/feed/
https://x.com/home
https://www.reddit.com/
https://dev.to/dashboard
https://peerlist.io/
https://www.indiehackers.com/
https://hashnode.com/
```

For each missing account, open the official sign-up flow. Pause while the user enters credentials and completes terms, email, CAPTCHA, or 2FA. Mark `ready` only after a harmless draft/profile page is accessible.

- [ ] **Step 5: Audit current community rules**

For every Reddit candidate, inspect the live `about/rules` page while authenticated. Record the exact public rule URL, threshold, and a `ready` or `skipped_policy` decision. If the rules do not explicitly permit the proposed post, send one concise moderator question and record only `sent_utc`; do not persist a private modmail URL. Mark `blocked_platform` until the response arrives and never infer permission from silence.

- [ ] **Step 6: Derive the launch date**

Use this decision exactly:

```text
If Product Hunt account age >= 7 days AND every core account can publish AND all gates can pass by Monday 18:00 ART:
  launch on 2026-07-21.
Otherwise:
  launch on 2026-07-28.
```

Record the chosen date and evidence in `tracker.md`. If execution occurs after both candidate dates, choose the next Tuesday after every gate can pass and record why the original date moved.

- [ ] **Step 7: Validate and commit the control plane**

Run:

```bash
rg -n -i 'password|cookie|recovery code|api[_ -]?token|authorization:' docs/launch/2026-07 || true
git diff --check
git status --short
```

Expected: the scan finds only the prohibition text if present; `git diff --check` is silent; status lists only `tracker.md` and `metrics.csv`.

Commit:

```bash
git add docs/launch/2026-07/tracker.md docs/launch/2026-07/metrics.csv
git commit -m "docs: establish memo launch control plane"
```

---

### Task 2: Prove product and landing readiness

**Files:**
- Modify: `docs/launch/2026-07/tracker.md`

**Interfaces:**
- Consumes: tracker status vocabulary and selected launch date from Task 1.
- Produces: `ProductReadiness` evidence required by the final gate.

- [ ] **Step 1: Verify the memo release and required CI**

Run from the memo launch worktree:

```bash
git fetch origin
git diff --exit-code origin/master...HEAD -- . ':(exclude)docs/superpowers/**' ':(exclude)docs/launch/**'
gh run list --repo jagoff/memo --branch master --limit 10
uv run --no-sync ruff check src/ tests/
uv run --no-sync mypy src/memo
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing
```

Expected: the product-code diff is empty; the latest required GitHub runs are `completed/success`; Ruff and mypy print success; pytest exits 0.

- [ ] **Step 2: Run an isolated macOS installer smoke**

Run from the memo launch worktree:

```bash
export MEMO_REPO="$PWD"
export SMOKE_ROOT="$(mktemp -d)"
export SMOKE_HOME="$SMOKE_ROOT/home"
mkdir -p "$SMOKE_HOME/.local/bin" "$SMOKE_HOME/.local/share/uv/tools" "$SMOKE_HOME/.cache"
ln -s "$HOME/.cache/huggingface" "$SMOKE_HOME/.cache/huggingface"

HOME="$SMOKE_HOME" \
UV_TOOL_DIR="$SMOKE_HOME/.local/share/uv/tools" \
UV_TOOL_BIN_DIR="$SMOKE_HOME/.local/bin" \
UV_CACHE_DIR="$SMOKE_HOME/.cache/uv" \
XDG_CONFIG_HOME="$SMOKE_HOME/.config" \
MEMO_DATA_DIR="$SMOKE_ROOT/data" \
MEMO_STATE_DIR="$SMOKE_ROOT/state" \
MEMO_INSTALL_SPEC="$MEMO_REPO" \
MEMO_INSTALL_DOWNLOAD_MODELS=yes \
MEMO_INSTALL_DOWNLOAD_CHAT=0 \
MEMO_INSTALL_SKIP_AGENT_CONFIG=1 \
MEMO_MODEL_PROFILE=balanced \
MEMO_NONINTERACTIVE=1 \
./install.sh | tee "$SMOKE_ROOT/install.log"
```

Expected: installer reaches `✓ memo is ready`, uses a uv tool install below `SMOKE_HOME`, and does not edit real client configuration.

- [ ] **Step 3: Verify the isolated macOS runtime and retrieval**

Run:

```bash
export PATH="$SMOKE_HOME/.local/bin:$PATH"
export MEMO_DATA_DIR="$SMOKE_ROOT/data"
export MEMO_STATE_DIR="$SMOKE_ROOT/state"
export MEMO_MODEL_PROFILE=balanced
export MEMO_NONINTERACTIVE=1

mkdir -p "$MEMO_DATA_DIR" "$MEMO_STATE_DIR"
memo doctor --strict-runtime
memo doctor --json > "$SMOKE_ROOT/doctor.json"
jq -e '.runtime.mode == "uv tool" and (.runtime.warnings | length == 0)' "$SMOKE_ROOT/doctor.json"
memo save "Atlas uses SQLite for local metadata because the demo must work offline." --title "Atlas database decision"
memo search "Which database does Atlas use and why?" --json > "$SMOKE_ROOT/search.json"
jq -e 'length >= 1 and any(.[]; .title == "Atlas database decision")' "$SMOKE_ROOT/search.json"
```

Expected: strict runtime health and both `jq -e` commands exit 0; the isolated search returns the synthetic decision.

- [ ] **Step 4: Verify the Linux/CPU Docker path**

Run:

```bash
docker pull ghcr.io/jagoff/memo:latest
docker run --rm ghcr.io/jagoff/memo:latest sh -lc '
  mkdir -p /data &&
  memo doctor &&
  memo save "Atlas Linux smoke uses SQLite for offline metadata." --title "Atlas Linux decision" &&
  memo search "Atlas offline database" --json > /tmp/search.json &&
  python -c "import json; hits=json.load(open('/tmp/search.json')); assert any(h['title'] == 'Atlas Linux decision' for h in hits); print('OK: Linux CPU save/search')"
'
```

Expected: doctor exits 0 and the last line is `OK: Linux CPU save/search`.

- [ ] **Step 5: Verify memo-web source quality**

Run from the memo-web launch worktree:

```bash
corepack pnpm install --frozen-lockfile
corepack pnpm format:check
corepack pnpm lint
corepack pnpm check
corepack pnpm build
corepack pnpm test
corepack pnpm test:build
corepack pnpm test:e2e --project=chromium
corepack pnpm test:a11y --project=chromium
corepack pnpm test:visual --project=chromium
corepack pnpm budget
```

Expected: every command exits 0; the bundle budget remains below the repository limit.

- [ ] **Step 6: Verify production landing and Analytics**

Run the HTTP and deployment checks:

```bash
curl -fsS -o /dev/null -w '%{http_code}\n' https://memo-web-sigma.vercel.app/
curl -fsS -o /dev/null -w '%{http_code}\n' https://memo-web-sigma.vercel.app/es/
curl -fsS -o /dev/null -w '%{http_code}\n' https://memo-web-sigma.vercel.app/_vercel/insights/script.js
vercel inspect https://memo-web-sigma.vercel.app --scope fernandoferrarigmailcoms-projects
vercel api '/v1/query/web-analytics/visits/count?projectId=prj_kekgqEeptrfnACvTuBamWwsE5Tir' --scope fernandoferrarigmailcoms-projects | jq -e '.data.visitors >= 1 and .data.pageviews >= 1'
```

Then use the installed Playwright browser to verify the production page at desktop and mobile widths, CTA destination, social metadata, bilingual rendering, horizontal overflow, and a normal-browser Analytics request:

```bash
corepack pnpm exec node --input-type=module <<'JS'
import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true });
for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
  for (const path of ['/', '/es/']) {
    const page = await browser.newPage({ viewport });
    const analyticsRequests = [];
    page.on('request', request => {
      if (request.url().includes('/_vercel/insights/view')) analyticsRequests.push(request.url());
    });
    const response = await page.goto(`https://memo-web-sigma.vercel.app${path}`, { waitUntil: 'networkidle' });
    if (!response?.ok()) throw new Error(`${path} returned ${response?.status()}`);
    const state = await page.evaluate(() => ({
      title: document.title,
      github: [...document.querySelectorAll('a')].some(a => a.href === 'https://github.com/jagoff/memo'),
      installer: document.body.textContent?.includes('curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash'),
      ogTitle: document.querySelector('meta[property="og:title"]')?.getAttribute('content'),
      ogImage: document.querySelector('meta[property="og:image"]')?.getAttribute('content'),
      twitterCard: document.querySelector('meta[name="twitter:card"]')?.getAttribute('content'),
      overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
      lang: document.documentElement.lang,
    }));
    if (!state.title || !state.github || !state.installer || !state.ogTitle || !state.ogImage || state.twitterCard !== 'summary_large_image' || state.overflow) {
      throw new Error(`${path} ${JSON.stringify(viewport)} failed: ${JSON.stringify(state)}`);
    }
    if ((path === '/' && state.lang !== 'en') || (path === '/es/' && state.lang !== 'es')) throw new Error(`wrong lang for ${path}: ${state.lang}`);
    if (analyticsRequests.length === 0) throw new Error(`no Analytics request observed for ${path}`);
    await page.close();
  }
}
await browser.close();
console.log('OK: production desktop/mobile, metadata, CTAs, locales, and Analytics');
JS
```

Open GitHub's security overview and the current open issue list; classify any security, privacy, installation, or data-loss report before marking readiness:

```bash
gh issue list --repo jagoff/memo --state open --limit 100 --json number,title,labels,url
open https://github.com/jagoff/memo/security
```

Expected: all three HTTP checks print `200`; Vercel reports production `READY`; Analytics has at least one visitor and pageview; Playwright prints its `OK` line; no open report is launch-blocking.

- [ ] **Step 7: Record readiness evidence and commit**

Use `apply_patch` to replace every Task 2 `not_checked` product row in `tracker.md` with `ready`, including command timestamps, GitHub run URLs, `local_ephemeral_log_reviewed` for the macOS smoke (not its private temporary path), the Docker image digest, and the public Vercel deployment URL. If a command fails, mark `failed`, stop the launch clock, and use the normal debugging workflow before proceeding.

Run:

```bash
git diff --check
git add docs/launch/2026-07/tracker.md
git commit -m "docs: record memo launch readiness evidence"
```

---

### Task 3: Capture current, citable claims

**Files:**
- Modify: `docs/launch/2026-07/tracker.md`

**Interfaces:**
- Consumes: healthy memo runtime and authenticated GitHub/Vercel access from Task 2.
- Produces: exact `tokens_historic`, `tokens_month`, `github_stars`, `github_forks`, and `registry_latest` values used by copy and screenshots.

- [ ] **Step 1: Capture memo token evidence**

Run against the real user corpus, read-only:

```bash
uv run --no-sync memo tokens --json > /tmp/memo-launch-tokens.json
jq -e '{historic_tokens: .historic.tokens, month_tokens: .month.tokens, grounded_historic: .historic.grounded, captured_date: .today.date}' /tmp/memo-launch-tokens.json
```

Expected: all four fields exist and the numeric fields are non-negative. The approved wording is “memo's counter estimates … tokens of repeated work avoided”; never present the estimate as a measured billing total.

- [ ] **Step 2: Capture repository evidence**

Run:

```bash
gh api repos/jagoff/memo --jq '{stargazers_count, forks_count, open_issues_count, watchers_count, pushed_at, default_branch}'
```

Expected: `default_branch` is `master`; record the exact current values and UTC timestamp.

- [ ] **Step 3: Capture official MCP Registry evidence**

Run:

```bash
curl -fsS 'https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.jagoff%2Fmemo&limit=100' \
  | jq -e '[.servers[] | select(.server.name == "io.github.jagoff/memo" and ._meta["io.modelcontextprotocol.registry/official"].isLatest == true)] | length == 1'

curl -fsS 'https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.jagoff%2Fmemo&limit=100' \
  | jq -r '.servers[] | select(.server.name == "io.github.jagoff/memo" and ._meta["io.modelcontextprotocol.registry/official"].isLatest == true) | [.server.version, ._meta["io.modelcontextprotocol.registry/official"].status] | @tsv'
```

Expected: exactly one latest record and the output is `3.7.0<TAB>active` unless a newer release has legitimately superseded it.

- [ ] **Step 4: Verify existing directory coverage**

Run:

```bash
curl -fsS https://glama.ai/mcp/servers/jagoff/memo | rg -q 'MEMO MCP Server'
curl -fsS https://mcpservers.org/servers/jagoff/memo | rg -qi 'local-first'
curl -fsS https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md \
  | rg -n '\[jagoff/memo\]\(https://github.com/jagoff/memo\)'
```

Expected: both directory checks exit 0 and the awesome-list command prints exactly one memo line.

- [ ] **Step 5: Record claim values and commit**

Use `apply_patch` to replace every `record command result` and `record timestamp` cell in the tracker claim table with exact values and evidence. Update MCP discovery rows to `ready` when their current descriptions/links are accurate; use `blocked_platform` only when a refresh request is actually necessary.

Run:

```bash
git diff --check
git add docs/launch/2026-07/tracker.md
git commit -m "docs: capture verified memo launch claims"
```

---

### Task 4: Produce and verify the visual evidence pack

**Files:**
- Create: `public/launch/demo.mp4`
- Create: `public/launch/demo-es.srt`
- Create: `public/launch/demo-es.mp4`
- Create: `public/launch/social-card.png`
- Create: `public/launch/product-hunt-thumbnail.png`
- Create: `public/launch/product-hunt-gallery-01.png`
- Create: `public/launch/product-hunt-gallery-02.png`
- Create: `public/launch/tokens.png`
- Create: `public/launch/markdown-history.png`
- Modify: `docs/launch/2026-07/tracker.md`

**Interfaces:**
- Consumes: verified claims from Task 3 and the existing memo-web brand assets.
- Produces: sanitized, approved media paths and SHA-256 hashes consumed by copy and platform drafts.

- [ ] **Step 1: Create the branded static assets**

From the memo-web launch worktree, create `public/launch/` and run:

```bash
mkdir -p public/launch
ffmpeg -y -i public/og.jpg -vf 'scale=1200:630:force_original_aspect_ratio=increase,crop=1200:630' -frames:v 1 public/launch/social-card.png
sips -s format png --resampleHeightWidth 240 240 public/logo.svg --out public/launch/product-hunt-thumbnail.png >/dev/null
corepack pnpm exec playwright screenshot --viewport-size='1270,760' --wait-for-timeout=2500 https://memo-web-sigma.vercel.app/ public/launch/product-hunt-gallery-01.png
corepack pnpm exec playwright screenshot --viewport-size='1270,760' --wait-for-timeout=2500 'https://memo-web-sigma.vercel.app/#features' public/launch/product-hunt-gallery-02.png
```

Expected: four non-empty image files using memo's existing visual system.

From the memo launch worktree, verify that the two existing repository diagrams cover the capture/recall and retrieval explanations required by the campaign:

```bash
test -s docs/diagram-loop.svg
test -s docs/diagram-recall.svg
rg -q 'Stored on your Mac' docs/diagram-loop.svg
rg -q 'Vector search' docs/diagram-recall.svg
rg -q 'Keyword search' docs/diagram-recall.svg
```

Expected: every command exits 0. Reuse both diagrams in the DEV article; do not create a competing architecture graphic.

- [ ] **Step 2: Seed a synthetic demonstration corpus**

From the memo launch worktree, run:

```bash
export DEMO_ROOT="$(mktemp -d)"
export MEMO_DATA_DIR="$DEMO_ROOT/data"
export MEMO_STATE_DIR="$DEMO_ROOT/state"
export MEMO_NONINTERACTIVE=1
mkdir -p "$MEMO_DATA_DIR" "$MEMO_STATE_DIR"

uv run --no-sync memo save \
  'For the Atlas demo project, use SQLite for local metadata; avoid Postgres because the demo must work offline.' \
  --title 'Atlas database decision' --type decision

uv run --no-sync memo search 'What database did Atlas choose for local metadata and why?' --json \
  | jq -e 'any(.[]; .title == "Atlas database decision")'
```

Expected: the search assertion exits 0. The corpus contains only the invented Atlas decision.

- [ ] **Step 3: Record the 30–45 second demo**

Invoke the `ui-demo` skill. Record two clean terminal panes using the isolated environment from Step 2:

```text
Pane 1 title: Claude Code / MCP save
Visible action: save the Atlas database decision.

Pane 2 title: Codex / MCP recall
Visible action: query “What database did Atlas choose for local metadata and why?” and show the exact SQLite/offline answer.

On-screen captions:
0–7s: “A coding agent saves a durable decision.”
8–18s: “memo keeps readable Markdown and a rebuildable local index.”
19–32s: “Another MCP client recalls the same decision in a fresh session.”
33–42s: “One local-first memory for every coding agent.”
```

Export to `public/launch/demo.mp4`. Use real Claude Code and Codex MCP clients if both can be pointed at the isolated corpus without touching global configuration. If that isolation cannot be proven, record deterministic memo CLI/MCP calls and label the panes “MCP client A” and “MCP client B”; do not imply they are live agent sessions.

Use `apply_patch` to create `public/launch/demo-es.srt` with the reviewed Spanish captions and the same timing:

```srt
1
00:00:00,000 --> 00:00:07,000
Un agente guarda una decisión duradera.

2
00:00:08,000 --> 00:00:18,000
memo conserva Markdown legible y un índice local reconstruible.

3
00:00:19,000 --> 00:00:32,000
Otro cliente MCP recupera la misma decisión en una sesión nueva.

4
00:00:33,000 --> 00:00:42,000
Una memoria local-first para todos tus agentes de código.
```

Burn the subtitles into the Spanish video:

```bash
ffmpeg -y -i public/launch/demo.mp4 \
  -vf "subtitles=public/launch/demo-es.srt:force_style='FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,MarginV=32'" \
  -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 160k -movflags +faststart \
  public/launch/demo-es.mp4
```

Expected: Spanish captions are burned into the image, fit within the safe area, and do not cover terminal evidence.

- [ ] **Step 4: Capture sanitized token and Markdown/history stills**

Use the same recording workflow to capture:

```text
tokens.png:
- command: memo tokens
- crop: totals/chart only
- exclude: shell prompt username, filesystem path, notification center, browser UI

markdown-history.png:
- resolve the synthetic Atlas record with the commands below, show its Markdown file, then run `memo record-history "$ATLAS_ID"`
- exclude every real vault path and memory
```

Resolve the synthetic record without manually copying an ID:

```bash
uv run --no-sync memo search 'Atlas database decision' --json --source codex > "$DEMO_ROOT/atlas-search.json"
export ATLAS_ID="$(jq -er 'map(select(.title == "Atlas database decision")) | .[0].id' "$DEMO_ROOT/atlas-search.json")"
export ATLAS_PATH="$(jq -er 'map(select(.title == "Atlas database decision")) | .[0].path' "$DEMO_ROOT/atlas-search.json")"
sed -n '1,80p' "$MEMO_DATA_DIR/$ATLAS_PATH"
uv run --no-sync memo record-history "$ATLAS_ID"
```

Expected: `ATLAS_ID` and `ATLAS_PATH` resolve from the isolated corpus, the Markdown contains only the invented SQLite/offline decision, and history shows its save event.

Copy the final PNG files to `public/launch/tokens.png` and `public/launch/markdown-history.png`.

- [ ] **Step 5: Verify media dimensions, duration, and hashes**

Run from memo-web:

```bash
sips -g pixelWidth -g pixelHeight public/launch/social-card.png
sips -g pixelWidth -g pixelHeight public/launch/product-hunt-thumbnail.png
sips -g pixelWidth -g pixelHeight public/launch/product-hunt-gallery-01.png
sips -g pixelWidth -g pixelHeight public/launch/product-hunt-gallery-02.png
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 public/launch/demo.mp4
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 public/launch/demo-es.mp4
shasum -a 256 public/launch/*
```

Expected: dimensions are exactly 1200×630, 240×240, 1270×760, and 1270×760; both video durations are between 30 and 45 seconds; every asset has a SHA-256 hash.

- [ ] **Step 6: Perform frame-by-frame privacy review**

Run:

```bash
ffmpeg -i public/launch/demo.mp4 -vf fps=1 "$DEMO_ROOT/review-en-%03d.png"
ffmpeg -i public/launch/demo-es.mp4 -vf fps=1 "$DEMO_ROOT/review-es-%03d.png"
```

Inspect every review frame, the subtitle source, and all six still images. Reject any frame containing a real username, email, home path, token, unrelated repository, real memory, notification, or private account detail. Re-record rather than blur sensitive content.

- [ ] **Step 7: Obtain asset approval and commit both repositories**

Present the eight media assets plus the Spanish subtitle source to the user as one batch. After approval:

```bash
git add public/launch/
git commit -m "assets: add sanitized memo launch media"
```

Use `apply_patch` in memo to record exact hashes, `approved` privacy/approval states, and `ready` statuses in the tracker, then commit:

```bash
git add docs/launch/2026-07/tracker.md
git commit -m "docs: approve memo launch evidence pack"
```

---

### Task 5: Write and approve channel-native launch copy

**Files:**
- Create: `docs/launch/2026-07/copy.md`
- Modify: `docs/launch/2026-07/tracker.md`

**Interfaces:**
- Consumes: exact claim values from Task 3, approved media paths from Task 4, and the prior approved LinkedIn memory `4cd6f2203be04877ac79d48e5e5f0b3f`.
- Produces: one approved initial-copy bundle; later tasks may publish it but may not silently rewrite it.

- [ ] **Step 1: Write the fixed platform copy**

Use `apply_patch` to create `copy.md`. Include these exact fixed elements:

```markdown
# memo launch copy — approved bundle

## Shared facts and prohibitions

- Core line: “Every coding session starts from zero. memo gives every coding agent one persistent, local-first memory.”
- Landing: https://memo-web-sigma.vercel.app
- Spanish landing: https://memo-web-sigma.vercel.app/es/
- Repository: https://github.com/jagoff/memo
- Say “local-first” and “no mandatory cloud API”; never say optional git sync cannot move memory.
- Mention Apple Silicon and Linux together.
- No vote solicitation in Hacker News, Product Hunt, Reddit, or DEV.

## Show HN

- Title: `Show HN: memo – local-first semantic memory shared by coding agents`
- URL: `https://github.com/jagoff/memo`
- Human first-comment facts: repeated-session amnesia; MCP compatibility; Markdown source of truth; MLX/CPU execution; hybrid search; one-step install; request technical feedback.
- Final first comment and all replies: authored by the user, not copied from assistant prose.

## Product Hunt

- Name: `memo`
- Tagline: `One local-first memory for every coding agent`
- Description: `memo gives Claude Code, Codex, Cursor, Devin, and other MCP clients one persistent, searchable memory. It runs locally on Apple Silicon or Linux, stores canonical memories as Markdown, and needs no mandatory cloud API.`
- Topics to validate in the live picker: Developer Tools, Artificial Intelligence, Open Source, Productivity.
- Destination: `https://memo-web-sigma.vercel.app`

Maker comment:

`I built memo after getting tired of explaining the same decisions to my coding agent in every new session. memo gives MCP-aware agents one persistent semantic memory: Claude Code can save a decision and Codex or Cursor can retrieve it later. It runs locally with MLX on Apple Silicon or a CPU backend on Linux. Memories stay readable as plain Markdown, while SQLite provides a rebuildable hybrid-search index. The project is MIT licensed and available now. I would especially value feedback on installation, cross-agent workflows, and the local-first trade-offs.`

## X thread

1. `Every coding session starts from zero. I built memo so Claude Code, Codex, Cursor, Devin, and other MCP agents can share one persistent memory.`
2. `It runs locally: MLX on Apple Silicon or a CPU backend on Linux. No mandatory cloud API and no separate vector database.`
3. `Markdown is the source of truth. SQLite is a rebuildable hybrid index. If memo disappears tomorrow, the knowledge is still readable.`
4. `It remembers decisions by meaning, preserves their history, and can flag contradictions when a decision changes.`
5. `Try it: https://memo-web-sigma.vercel.app  Source: https://github.com/jagoff/memo  If it helps your workflow, a GitHub star helps other developers find it.`

## Reddit — r/LocalLLaMA candidate

- Title: `I built a local-first semantic memory layer for coding agents (MLX on Apple Silicon, CPU on Linux)`
- Body: `I kept losing the reasoning behind coding decisions every time I opened a new agent session, so I built memo: an MIT-licensed MCP server that gives Claude Code, Codex, Cursor, Devin, and other clients one searchable memory. It uses local embeddings, hybrid vector + BM25 retrieval, plain Markdown as the source of truth, and a rebuildable SQLite index. Apple Silicon uses MLX; Linux has a CPU backend. There is no mandatory hosted service. I would value technical feedback on retrieval quality and the local-first architecture: https://github.com/jagoff/memo`

## Reddit — r/selfhosted candidate

- Title: `memo: self-hosted, local-first memory shared across AI coding agents`
- Body: `memo is an MIT-licensed MCP memory server that runs on your own machine and lets multiple coding agents share durable decisions. Memories are plain Markdown, the SQLite index is rebuildable, and private-git sync is optional if you want the same corpus across machines. It supports Apple Silicon through MLX and Linux through a CPU backend; no mandatory cloud API or separate vector database is required. Repository and install instructions: https://github.com/jagoff/memo`

## Reddit — r/opensource candidate

- Title: `memo — MIT-licensed local-first semantic memory for AI coding agents`
- Body: `I built memo because coding agents repeatedly forgot decisions, failed approaches, and project conventions between sessions. It is an MCP server with hybrid semantic/keyword retrieval, temporal history, contradiction detection, and Markdown as the canonical store. Claude Code, Codex, Cursor, Devin, and other MCP clients can use the same memory. It runs on Apple Silicon or Linux and is open for technical feedback and contributions: https://github.com/jagoff/memo`

## Reddit — r/ObsidianMD candidate

- Title: `Using plain Markdown as the source of truth for an AI-agent memory system`
- Body: `I built memo as an MCP memory server, but the durable store is intentionally ordinary Markdown: files remain readable, versionable, and editable in an Obsidian vault, while SQLite is only a rebuildable search index. The interesting design constraint was letting hand-edited Markdown win on reindex without turning Obsidian into a required plugin. This is relevant only if architecture/tool posts are allowed here; repository: https://github.com/jagoff/memo`

## DEV article

- Title: `Building a local-first semantic memory shared by coding agents`
- Cover: `memo-web/public/launch/social-card.png`
- Tags: `ai`, `opensource`, `mcp`, `productivity`
- Disclosure: `Disclosure: I used an AI coding assistant to help structure and edit this article. I verified every technical claim against the memo repository and live commands.`
- Required sections, in order:
  1. The problem: new coding-agent sessions repeatedly lose decisions and rationale.
  2. The contract: every MCP-aware client reads and writes the same durable memory.
  3. Local execution: MLX on Apple Silicon and CPU embeddings on Linux, with no mandatory cloud API.
  4. Storage: Markdown is canonical; SQLite/FTS/vector data is rebuildable.
  5. Retrieval: vector and BM25 candidates are fused, optionally reranked, and injected on a bounded recall budget.
  6. Trust: history and contradiction detection prevent stale decisions from silently winning.
  7. Token economy: quote the exact verified estimate from the tracker and label it as memo's estimate.
  8. Try it: the one-line installer, repository, and landing links.
- The article must explain each section in original prose, include the existing `docs/diagram-loop.svg` and `docs/diagram-recall.svg` diagrams, and stay between 900 and 1,400 words.

## Spanish wave

- LinkedIn: use the approved Spanish source from memo memory `4cd6f2203be04877ac79d48e5e5f0b3f`, refreshing the token-estimate sentence from the tracker.
- X: translate the approved five-post English thread faithfully; retain Apple Silicon, Linux, Markdown, MCP, landing, and repository.
- Video: use the approved demo with Spanish burned-in captions.
```

- [ ] **Step 2: Restore and refresh the approved LinkedIn posts**

Read memo record `4cd6f2203be04877ac79d48e5e5f0b3f` with `source="codex"`. Copy its complete English and Spanish final posts into `copy.md`. Replace the old 506k/182k sentence with the exact current `tokens_historic` and `tokens_month` values from `tracker.md`, formatted with thousands separators and explicitly described as estimates from `memo tokens`.

Expected: neither LinkedIn version mentions 506k or “day 3”; both preserve the previously approved voice rules.

- [ ] **Step 3: Write the complete DEV article**

Expand the eight required sections in `copy.md` into 900–1,400 words. Use only repository-backed claims. Include exactly these code blocks:

```bash
curl -fsSL https://raw.githubusercontent.com/jagoff/memo/master/install.sh | bash
memo doctor --strict-runtime
```

```text
prompt -> vector search + BM25 -> fusion -> optional rerank -> bounded recall -> agent
                  Markdown source of truth <- rebuildable SQLite index
```

End with the disclosure, repository, landing, and a request for architectural feedback. Do not ask for DEV reactions or votes.

- [ ] **Step 4: Validate copy mechanically**

Run:

```bash
rg -n '506[,.]000|182[,.]000|never leave|Claude-only|please upvote|upvote this|vote for' docs/launch/2026-07/copy.md
python - <<'PY'
from pathlib import Path

text = Path('docs/launch/2026-07/copy.md').read_text()
description = 'memo gives Claude Code, Codex, Cursor, Devin, and other MCP clients one persistent, searchable memory. It runs locally on Apple Silicon or Linux, stores canonical memories as Markdown, and needs no mandatory cloud API.'
assert len(description) <= 260, len(description)
assert 'Disclosure:' in text
assert 'Apple Silicon' in text and 'Linux' in text
assert 'https://github.com/jagoff/memo' in text
print('OK: fixed copy constraints')
PY
```

Expected: `rg` returns no matches and Python prints `OK: fixed copy constraints`.

- [ ] **Step 5: Obtain one batch approval**

Present `copy.md`, both videos, all six images, and the Spanish subtitle source together. Apply requested changes, rerun Step 4, then record the user's approval message in the tracker. Change every initial queue `Copy approval` cell to `approved`; do not approve reserve posts that have not passed a live policy check.

- [ ] **Step 6: Commit approved copy**

Run:

```bash
git diff --check
git add docs/launch/2026-07/copy.md docs/launch/2026-07/tracker.md
git commit -m "docs: approve memo multichannel launch copy"
```

---

### Task 6: Prepare platform drafts without publishing

**Files:**
- Modify: `docs/launch/2026-07/tracker.md`

**Interfaces:**
- Consumes: approved copy/assets and ready accounts.
- Produces: scheduled/draft states and preview evidence; no public initial post except the scheduled Product Hunt record becoming live at launch time.

- [ ] **Step 1: Prepare and schedule Product Hunt**

From the eligible personal account, create the memo launch with the exact Product Hunt fields from `copy.md`, landing destination, 240×240 thumbnail, two gallery images, maker identity, and maker comment. Schedule it for 00:01 Pacific on the selected launch date. Record `browser_verified` and the `scheduled` state in the tracker. Record a preview URL only if it is public and contains no access token or private draft identifier.

Expected: Product Hunt shows the selected date and all assets; no vote request appears anywhere.

- [ ] **Step 2: Save the DEV article as a draft**

Create a DEV draft with the exact article, cover, four tags, disclosure, and links from `copy.md`. Preview desktop and mobile, verify code blocks and diagram, and record `browser_verified`. Do not persist a private draft URL or publish/schedule until the final readiness gate passes.

- [ ] **Step 3: Prepare owned-social drafts**

Create or stage the LinkedIn English post and X thread using the approved asset. If the platform supports scheduling, schedule them for 09:30 and 10:15 ART on the launch date; otherwise leave them in the platform's draft UI and record `ready` plus the intended time. Verify that the landing and GitHub URLs are clickable in preview.

- [ ] **Step 4: Prepare Show HN submission materials**

Record the exact title and GitHub URL in the tracker. Ask the user to write the human first comment from the factual outline in `copy.md` and store that user-authored comment in the tracker only after they provide it. Do not submit to HN yet.

- [ ] **Step 5: Prepare Reddit drafts only for permitted communities**

For every `ready` community, preview the matching approved title/body and record the intended post type and order. For every `skipped_policy` or unresolved community, keep publication status `skipped_policy`; do not prepare an evasive alternative subreddit.

- [ ] **Step 6: Verify MCP discovery and request only necessary refreshes**

Recheck Registry, Glama, mcpservers.org, awesome-mcp-servers, and PulseMCP. If a description/version/stat is stale and the surface exposes an official free refresh route, prepare that request and record its draft/status. Do not open a duplicate listing or awesome-list PR.

- [ ] **Step 7: Review every preview and commit draft status**

Present platform previews to the user as one final preflight. After approval, update queue rows to `scheduled` or `ready` and commit:

```bash
git add docs/launch/2026-07/tracker.md
git commit -m "docs: stage memo launch platform drafts"
```

---

### Task 7: Pass the final readiness gate and capture the baseline

**Files:**
- Modify: `docs/launch/2026-07/tracker.md`
- Modify: `docs/launch/2026-07/metrics.csv`

**Interfaces:**
- Consumes: Tasks 1–6 evidence and approvals.
- Produces: `Gate=ready`, launch baseline `B`, and exact 6h/24h/D3/D5/D7 targets.

- [ ] **Step 1: Re-run launch-critical health checks**

At or before Monday 18:00 ART, run:

```bash
gh run list --repo jagoff/memo --branch master --limit 10
gh run list --repo jagoff/memo-web --branch main --limit 10
curl -fsS -o /dev/null -w '%{http_code}\n' https://memo-web-sigma.vercel.app/
curl -fsS -o /dev/null -w '%{http_code}\n' https://memo-web-sigma.vercel.app/es/
curl -fsS -o /dev/null -w '%{http_code}\n' https://github.com/jagoff/memo
```

Expected: relevant CI runs are successful and all HTTP checks print `200`.

- [ ] **Step 2: Capture baseline GitHub and Vercel data**

Run:

```bash
gh api repos/jagoff/memo --jq '{stargazers_count, forks_count, open_issues_count}'
gh api repos/jagoff/memo/traffic/views --jq '{count, uniques}'
gh api repos/jagoff/memo/traffic/clones --jq '{count, uniques}'
gh api repos/jagoff/memo/traffic/popular/referrers --jq '.'
vercel api '/v1/query/web-analytics/visits/count?projectId=prj_kekgqEeptrfnACvTuBamWwsE5Tir' --scope fernandoferrarigmailcoms-projects | jq '.data'
```

Expected: all APIs return aggregate data without exposing individual stargazer identities.

- [ ] **Step 3: Calculate checkpoint targets**

Resolve the exact baseline star count `B` from GitHub and calculate the targets:

```bash
B="$(gh api repos/jagoff/memo --jq '.stargazers_count')"
B="$B" python - <<'PY'
import os

B = int(os.environ['B'])
reference = {'+6h': 17, '+24h': 32, 'D3': 62, 'D5': 82, 'D7': 100}
targets = {name: B + round((100 - B) * (target - 7) / 93) for name, target in reference.items()}
assert targets['D7'] == 100
print(targets)
PY
```

Expected: the command prints integer targets ending in `'D7': 100`. Record the live baseline and calculated values in the tracker.

- [ ] **Step 4: Append the baseline aggregate row**

Use `apply_patch` to append one `baseline` row to `metrics.csv` using exact UTC timestamp and API values. Use `0` only for platform metrics whose campaign post does not yet exist; use `unavailable` in `notes` rather than inventing a numeric value.

- [ ] **Step 5: Evaluate every hard gate**

Change a gate to `ready` only if its referenced rows are ready/approved and evidence is present. If any hard gate is not ready, change campaign state to `blocked_platform`, `blocked_user`, or `failed`, reschedule Product Hunt and all drafts to the next Tuesday, and stop before Task 8.

Expected: there is no partial launch state.

- [ ] **Step 6: Confirm user availability and publication authority**

Ask the user for one explicit confirmation that they will be available during the first six hours and that the approved, scheduled initial queue may be published. Record the confirmation in the tracker and set the final gate to `ready`.

- [ ] **Step 7: Commit the readiness receipt**

Run:

```bash
git diff --check
git add docs/launch/2026-07/tracker.md docs/launch/2026-07/metrics.csv
git commit -m "docs: pass memo launch readiness gate"
```

---

### Task 8: Execute the coordinated launch day

**Files:**
- Modify: `docs/launch/2026-07/tracker.md`

**Interfaces:**
- Consumes: `Gate=ready`, approved copy/assets, authenticated accounts, and selected launch date.
- Produces: public post URLs, publication status for every core channel, and draft PRs containing only sanitized campaign artifacts.

- [ ] **Step 1: Verify Product Hunt went live**

At 00:01 Pacific, open the scheduled Product Hunt record. Verify the landing URL, thumbnail, gallery, maker, description, and maker comment. Record the public URL and `published`; if it did not go live, diagnose the schedule before publishing any other channel.

- [ ] **Step 2: Publish LinkedIn at 09:30 ART**

Publish the approved English LinkedIn post and approved social/demo asset. Verify the public post contains the landing link and refreshed estimate. Record the permalink immediately.

- [ ] **Step 3: Publish X at 10:15 ART**

Publish the five approved posts as one thread. Verify ordering, media, landing, and GitHub links. Record the root permalink immediately.

- [ ] **Step 4: Submit Show HN at 11:00 ART**

Submit the exact approved title with the direct GitHub URL. The user publishes their human-authored first comment and later HN replies. Record the item URL; do not ask anyone to vote.

- [ ] **Step 5: Publish DEV at 12:30 ART**

Publish the approved article with disclosure, cover, tags, diagram, installer, landing, and repository. The user authors any DEV comments. Record the article URL.

- [ ] **Step 6: Publish permitted Reddit posts from 14:00–17:00 ART**

Publish one approved, tailored post at a time, leaving enough time to inspect moderation status before the next. Record each permalink or `skipped_policy` reason. If one is removed, stop that community path and do not repost.

- [ ] **Step 7: Refresh MCP discovery from 17:00–19:00 ART**

Verify existing entries again and submit only the prepared official free refresh request for a surface proven stale. Record the request URL/status. Do not create a duplicate listing or PR.

- [ ] **Step 8: Run publication link checks**

Build the link-check list from rows whose status is `published`, inspect it, and check every URL:

```bash
python - <<'PY' > /tmp/memo-launch-public-urls.txt
import re
from pathlib import Path

text = Path('docs/launch/2026-07/tracker.md').read_text()
urls: set[str] = set()
for line in text.splitlines():
    if re.search(r'\|\s*published\s*\|', line):
        urls.update(re.findall(r'https://[^\s|)>]+', line))
print('\n'.join(sorted(urls)))
PY

test -s /tmp/memo-launch-public-urls.txt
cat /tmp/memo-launch-public-urls.txt
while IFS= read -r url; do
  curl -L -sS -o /dev/null --max-time 20 -w '%{http_code} %{url_effective}\n' "$url"
done < /tmp/memo-launch-public-urls.txt
```

Expected: the list contains every published queue/reply URL and no private draft URL; directly fetchable pages return `200`. For login-gated/anti-bot pages, verify in a normal authenticated browser and record `browser_verified` rather than treating `403` as a broken post.

- [ ] **Step 9: Sanitize and push campaign branches as draft PRs**

Run in both worktrees:

```bash
rg -n -i 'password|session cookie|recovery code|api[_ -]?token|authorization:|/Users/fer|@gmail' docs/launch public/launch 2>/dev/null || true
git diff --check
git push -u origin launch/2026-07-memo
```

Expected: the scan finds no credential/private-path material. Open draft PRs:

```bash
gh pr create --repo jagoff/memo --base master --head launch/2026-07-memo --draft --title "docs: record memo organic launch" --body "Sanitized copy, tracker, aggregate metrics, and postmortem branch for the July 2026 memo launch."
gh pr create --repo jagoff/memo-web --base main --head launch/2026-07-memo --draft --title "assets: add memo launch media" --body "Sanitized launch media derived from the existing memo-web visual system."
```

Record both PR URLs. Leave them draft until D7 so metrics/postmortem updates stay on the same branches.

- [ ] **Step 10: Commit publication receipts and capture memory**

Run in memo:

```bash
git add docs/launch/2026-07/tracker.md
git commit -m "docs: record memo launch publication URLs"
git push
```

Call `memo_idle_capture`, then `memo_pop_notification`, and show any non-empty notification to the user.

---

### Task 9: Operate the first 24 hours and approved contingencies

**Files:**
- Modify: `docs/launch/2026-07/tracker.md`
- Modify: `docs/launch/2026-07/metrics.csv`

**Interfaces:**
- Consumes: public URLs and calculated checkpoint targets.
- Produces: +6h/+24h metrics, approved replies, and at most one allowed acceleration.

- [ ] **Step 1: Start recurring monitoring without blocking sleeps**

Use the product's recurring monitoring/wait mechanism for the +6h and +24h checkpoints. Do not run a shell `sleep`. At each checkpoint, execute the same GitHub/Vercel commands from Task 7 and collect native aggregate metrics from each platform dashboard.

- [ ] **Step 2: Process public questions through the approval queue**

For each high-value question, add its public permalink/topic to the tracker, verify facts against memo or the repository, draft one concise reply, and request user approval. Publish only the approved text. For HN and DEV, give the user facts/outline and record the permalink of their human-authored response.

- [ ] **Step 3: Record and evaluate +6h**

Append an exact `+6h` CSV row with `apply_patch`, compare actual stars with the calculated target, and record diagnostics.

If below target, choose only one owned-channel improvement: replace a weak thumbnail, tighten the first paragraph, or attach the strongest missing proof. Do not repost to communities or add a reserve platform.

- [ ] **Step 4: Record and evaluate +24h**

Append an exact `+24h` CSV row and compare with target.

If below target, choose exactly one acceleration based on the strongest relevant audience:

```text
A. Advance the already-approved Spanish wave into +24h to +36h.
OR
B. Publish on one rules-validated reserve surface: Peerlist, Indie Hackers, or Hashnode.
```

For Hashnode, republish the DEV article and set the DEV URL as canonical. Never activate both A and B.

- [ ] **Step 5: Commit and push the checkpoint receipt**

Run:

```bash
git add docs/launch/2026-07/tracker.md docs/launch/2026-07/metrics.csv
git commit -m "docs: record memo launch first-day metrics"
git push
```

Call `memo_idle_capture`, then `memo_pop_notification`.

---

### Task 10: Run the Spanish wave and day-three review

**Files:**
- Modify: `docs/launch/2026-07/tracker.md`
- Modify: `docs/launch/2026-07/metrics.csv`

**Interfaces:**
- Consumes: approved Spanish copy, approved subtitled demo, and the +24h contingency decision.
- Produces: Spanish publication URLs and the D3 checkpoint decision.

- [ ] **Step 1: Publish the Spanish wave at the chosen time**

If Task 9 advanced the wave, use that recorded +24h-to-+36h slot. Otherwise publish at launch +48 hours. Publish the approved Spanish LinkedIn post, X thread, and subtitled demo. Use Spanish communities only if their live rules were recorded as `ready`. Record every permalink.

- [ ] **Step 2: Process Spanish responses through the same approval queue**

Verify technical facts, draft replies in natural Spanish, and request user approval individually. Do not translate a hostile or irrelevant thread merely to increase comment count.

- [ ] **Step 3: Record D3 metrics**

Append the exact D3 aggregate row, compare stars with the calculated D3 target, and summarize referrers plus meaningful questions.

If below target, publish one genuinely new technical lesson from real launch questions on the best-performing owned channel. Do not repeat the announcement or add a second reserve platform.

- [ ] **Step 4: Commit and push D3 receipts**

Run:

```bash
git add docs/launch/2026-07/tracker.md docs/launch/2026-07/metrics.csv
git commit -m "docs: record memo Spanish wave and day-three metrics"
git push
```

Call `memo_idle_capture`, then `memo_pop_notification`.

---

### Task 11: Close D5/D7 measurement and publish the postmortem

**Files:**
- Modify: `docs/launch/2026-07/tracker.md`
- Modify: `docs/launch/2026-07/metrics.csv`
- Create: `docs/launch/2026-07/postmortem.md`

**Interfaces:**
- Consumes: all publication URLs, checkpoint rows, response outcomes, and directory statuses.
- Produces: final star result, durable lessons, merged sanitized campaign artifacts, and memo memories for future launches.

- [ ] **Step 1: Record D5 without adding a new channel**

At D5, collect the same aggregate metrics, append the D5 row, and continue approved responses. Do not add a new publication surface merely because the target is missed.

- [ ] **Step 2: Record the final D7 checkpoint**

At exactly seven days after the first Product Hunt publication, collect final GitHub, Vercel, and platform aggregates. Append the D7 row and record whether total GitHub stars are at least 100.

- [ ] **Step 3: Write the exact postmortem structure**

Use `apply_patch` to create `postmortem.md` with these completed sections and no empty headings:

```markdown
# memo organic launch postmortem — July 2026

## Outcome

- Launch date and seven-day close timestamp.
- GitHub star baseline, final total, absolute gain, and whether the 100-star target was met.

## Funnel

- Platform impressions.
- Landing visitors/pageviews/referrers.
- GitHub views/unique visitors/clones/referrers.
- Meaningful comments, issues, contributions, and directory refresh outcomes.

## Channel results

- Product Hunt.
- LinkedIn.
- X.
- Show HN.
- DEV.
- Each permitted Reddit community.
- Spanish wave.
- MCP discovery surfaces.

For each channel: public URL, reach, observed traffic, technical feedback, moderation outcome, and evidence-backed interpretation. Do not claim star attribution that aggregate data cannot prove.

## Contingencies used

- Trigger, chosen action, evidence, and result for every activated contingency.

## Product findings

- Installation defects, documentation gaps, repeated questions, and fixes or follow-up issues.

## What worked

- Evidence-supported lessons only.

## What did not work

- Evidence-supported lessons only.

## Next distribution experiment

- One recommended experiment, its target audience, success metric, and why it follows from this launch.
```

- [ ] **Step 4: Validate campaign files and mark PRs ready**

Run:

```bash
python - <<'PY'
from pathlib import Path

needles = ('TO' + 'DO', 'T' + 'BD', 'FIX' + 'ME', 'place' + 'holder')
hits = []
for path in Path('docs/launch/2026-07').glob('*'):
    if path.is_file():
        text = path.read_text(errors='replace').lower()
        hits.extend((str(path), needle) for needle in needles if needle.lower() in text)
assert not hits, hits
print('OK: no unfinished markers')
PY
rg -n -i 'password|session cookie|recovery code|api[_ -]?token|authorization:|/Users/fer|@gmail' docs/launch/2026-07 || true
git diff --check
```

Expected: Python prints `OK: no unfinished markers`; the privacy scan finds nothing; diff check is silent.

Commit and push:

```bash
git add docs/launch/2026-07/tracker.md docs/launch/2026-07/metrics.csv docs/launch/2026-07/postmortem.md
git commit -m "docs: close memo organic launch campaign"
git push
```

Resolve both draft PR URLs, mark them ready, wait for required CI, and merge through the protected default branches:

```bash
MEMO_PR="$(gh pr list --repo jagoff/memo --head launch/2026-07-memo --json url --jq '.[0].url')"
MEMO_WEB_PR="$(gh pr list --repo jagoff/memo-web --head launch/2026-07-memo --json url --jq '.[0].url')"
test -n "$MEMO_PR" && test -n "$MEMO_WEB_PR"

gh pr ready "$MEMO_PR"
gh pr checks "$MEMO_PR" --watch
gh pr merge "$MEMO_PR" --merge --delete-branch

gh pr ready "$MEMO_WEB_PR"
gh pr checks "$MEMO_WEB_PR" --watch
gh pr merge "$MEMO_WEB_PR" --merge --delete-branch
```

Expected: both PRs merge only after required checks pass.

- [ ] **Step 5: Save durable launch outcomes to memo**

Use `memo_save` with `project:memo`, `marketing`, `launch`, and `postmortem` tags to save:

```text
- final star baseline/total/gain
- best-performing channels and evidence
- policy/moderation outcomes
- repeated user questions and product gaps
- contingencies and measured effects
- one next distribution experiment
- public postmortem path and merged PR URLs
```

Then call `memo_idle_capture` and `memo_pop_notification`; show any non-empty notification.

- [ ] **Step 6: Deliver the final user report**

Report the final star outcome first, then link the public landing, GitHub repository, Vercel Analytics dashboard, merged postmortem, and every major public launch post. Clearly distinguish measured attribution from inference.
