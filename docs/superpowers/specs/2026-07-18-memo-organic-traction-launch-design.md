# memo organic traction launch — design

| Field | Value |
|---|---|
| Date | 2026-07-18 |
| Status | Approved design; implementation plan written |
| Primary objective | Reach 100 total GitHub stars within seven days of launch |
| Budget | Free and organic only |
| Primary language | English, followed by a Spanish wave about 48 hours later |

## Summary

memo will run a founder-led, same-day, multichannel launch reinforced by durable
MCP ecosystem discovery. The campaign sends high-intent technical audiences
directly to the GitHub repository and broader audiences to the bilingual landing
page. Every channel receives native copy rather than an identical cross-post.

The target is deliberately aggressive: the repository had 7 stars when the
design was approved, so the campaign must add 93 stars in seven days. This is a
goal, not a guarantee. The launch will be measured through GitHub, Vercel Web
Analytics, and each platform's native analytics without paid promotion or a new
tracking service.

Public destinations:

- Landing page: <https://memo-web-sigma.vercel.app>
- Spanish landing page: <https://memo-web-sigma.vercel.app/es/>
- Repository: <https://github.com/jagoff/memo>
- Vercel Analytics dashboard: <https://vercel.com/fernandoferrarigmailcoms-projects/memo-memory/analytics>

## Context and positioning

The campaign starts from one problem:

> Every coding session starts from zero. memo gives every coding agent one
> persistent, local-first memory.

Supporting proof points, in priority order:

1. One MCP memory works across Claude Code, Codex, Cursor, Devin, Cline, and
   other MCP-aware agents.
2. memo is local-first: it uses MLX on Apple Silicon and a CPU backend on Linux,
   with no mandatory cloud API.
3. Plain Markdown is canonical and human-readable; SQLite is a rebuildable
   search index.
4. Semantic recall, temporal history, and contradiction detection preserve the
   reasoning behind decisions rather than only storing snippets.
5. A compact MCP surface and bounded recall reduce repeated model work. Any
   numerical token claim must be regenerated and verified immediately before
   publication.
6. Private-git memory sync is optional and controlled by the user. Marketing
   copy must not imply that memory can never leave a device after sync is
   explicitly enabled.

Voice rules:

- Say “coding agent” rather than centering the story on Claude.
- Mention both Apple Silicon and Linux.
- Describe control and local-first operation precisely; do not overstate
  isolation.
- Mention Markdown/Obsidian once, in the no-lock-in proof.
- Use firsthand language and concrete demonstrations rather than launch hype.

## Scope

### In scope

- Preparing channel-specific English launch copy and a later Spanish wave.
- Producing a small visual proof package from safe, synthetic demo data.
- Validating or creating the required personal platform accounts.
- Publishing the approved launch bundle on the core channels during one day.
- Verifying and refreshing existing MCP directory entries, and submitting only
  missing entries or contributions.
- Monitoring public responses, drafting replies for approval, and recording
  launch metrics for seven days.
- Producing a short postmortem after day seven.

### Out of scope

- Paid ads, sponsored placements, or premium directory upgrades.
- Vote trading, coordinated upvotes, fake engagement, or duplicate accounts.
- Mass direct messages or unsolicited outreach.
- New memo product features solely for the campaign.
- A new analytics application, database, or hosted service.
- Storing platform passwords, session cookies, recovery codes, or API tokens in
  git, memo memories, screenshots, or launch documents.
- Guaranteeing the 100-star outcome.

## Campaign units and boundaries

The campaign has five small operating units:

1. **Evidence pack:** derives approved claims and safe visual assets from the
   repository and isolated demo data. It does not publish anything.
2. **Channel copy:** adapts the same factual story to each platform. It depends
   on the evidence pack and current platform rules.
3. **Readiness gate:** verifies the product, accounts, copy, assets, and tracking.
   It is the only unit allowed to declare the campaign ready.
4. **Publication and response queue:** publishes the approved initial bundle,
   captures public questions, and routes proposed replies through the user.
5. **Measurement loop:** snapshots platform, landing, and GitHub results and
   selects only the pre-approved contingency for the next checkpoint.

The operating flow is:

`project facts -> evidence pack -> native channel copy -> batch user approval -> publication -> metrics and questions -> approved responses/contingencies`

No custom cross-posting bot is needed. Platform submission remains manual or
browser-assisted because authentication, previews, moderation context, and
human approval are material parts of the workflow.

## Core channel architecture

| Channel | Native angle | Primary destination | Launch action |
|---|---|---|---|
| Show HN | Why memo was built, how it works, and what can be run now | GitHub repository | Submit a `Show HN:` post and stay available for technical discussion |
| Product Hunt | One local-first memory for every coding agent | Landing page | Launch from the user's eligible personal account |
| LinkedIn | Firsthand story: repeatedly explaining the same decisions | Landing page | Publish the approved founder post with refreshed metrics and demo |
| X | Short hook, visual demo, then a compact technical thread | Landing first, GitHub second | Publish one thread from the user's personal account |
| Reddit | Privacy, self-hosting, open source, or Markdown according to the community | GitHub repository | Publish separately tailored posts only where current rules permit |
| DEV Community | Technical architecture and lessons, not a promotional announcement | Landing and GitHub | Publish a reviewed article with the required AI-assistance disclosure |
| MCP ecosystem | Accurate metadata and long-lived discovery | GitHub/package | Refresh existing entries and submit only genuinely missing surfaces |

### Community selection

Reddit candidates are deliberately conditional:

- `r/LocalLLaMA` for local execution and privacy.
- `r/selfhosted` for user-controlled data and no mandatory hosted service.
- `r/opensource` for the MIT project, architecture, and contribution path.
- `r/ObsidianMD` only for Markdown interoperability, without presenting memo as
  an Obsidian plugin.
- Claude, Codex, and MCP communities only after validating that the community is
  about the relevant technology and that its current rules permit the post.

Before any Reddit submission, record the rule URL, allowed post type, account
age/karma requirement, and whether moderator approval is needed. If rules are
unclear, ask moderators; if permission is absent, skip the community.

### MCP discovery truth as of 2026-07-18

Several surfaces initially considered “new submissions” already contain memo:

- The official MCP Registry contains `io.github.jagoff/memo`; version `3.7.0` is
  active and marked latest.
- [Glama already has a memo page](https://glama.ai/mcp/servers/jagoff/memo).
- [mcpservers.org already has a memo page](https://mcpservers.org/servers/jagoff/memo),
  although displayed repository statistics can lag.

The Knowledge & Memory section of
[`punkpeye/awesome-mcp-servers`](https://github.com/punkpeye/awesome-mcp-servers)
also already contains `jagoff/memo`. Therefore the launch action on all of these
surfaces is to verify version, description, links, categories, and imagery, then
request a refresh only where stale. Do not create duplicate entries or pull
requests. PulseMCP is checked for an existing memo entry; an update/submission is
attempted only if memo is absent and an official free route exists.

## Platform constraints

- **Hacker News:** link to the runnable repository rather than the marketing
  landing page, use a `Show HN:` title, stay available to discuss it, and never
  solicit votes. HN asks users not to generate or edit comments with AI, so the
  assistant may provide facts and an outline but the user authors the final HN
  comment. See the [Show HN rules](https://news.ycombinator.com/showhn.html) and
  [HN guidelines](https://news.ycombinator.com/newsguidelines.html).
- **Product Hunt:** use a personal account. The account must meet the platform's
  current one-week age requirement before the launch can be scheduled. The
  listing needs a direct product URL, tagline, topics, a 240×240 thumbnail,
  gallery assets, description, maker, and first comment. See
  [Product Hunt's posting guide](https://help.producthunt.com/en/articles/479557-how-to-post-a-product).
- **DEV Community:** the technical article must be fact-checked, disclose AI
  assistance, and provide value beyond promotion. AI-generated comments are not
  used. See the
  [DEV editor guide](https://dev.to/p/editor_guide/) and
  [AI-assisted article guidelines](https://dev.to/guidelines-for-ai-assisted-articles-on-dev).
- **Reddit:** community rules override this design. Identical reposts, moderation
  evasion, and excessive promotion are prohibited by the campaign even if a
  community has no explicit rule against them.
- **All platforms:** ask readers to try memo, inspect the code, or give feedback.
  A “star it if it helps you” CTA is allowed only on the user's owned social
  posts. Never ask for coordinated platform votes.

## Evidence and asset package

Required assets:

1. A 30–45 second screen recording that saves a synthetic decision with one
   agent and retrieves it from another.
2. A fresh `memo tokens` screenshot whose visible data has been reviewed for
   accuracy and privacy.
3. A simple capture-to-recall diagram:
   `capture -> Markdown -> hybrid index -> bounded recall -> any MCP agent`.
4. A screenshot of a synthetic Markdown memory and its history.
5. A 1200×630 social image usable by LinkedIn, X, DEV, and Hashnode.
6. A 240×240 Product Hunt thumbnail and at least two gallery images.
7. English copy for every core channel and Spanish copy for the second wave.

Demo and screenshot rules:

- Use an isolated `MEMO_DATA_DIR` and `MEMO_STATE_DIR` with invented project
  names, decisions, paths, and people.
- Do not record the real vault, terminal history, environment variables,
  notifications, browser bookmarks, email addresses, or private repository
  names.
- Review every frame before approval.
- Reuse the landing's existing visual language and repository diagrams where
  possible instead of designing an unrelated brand system.
- Re-run live metrics on the final preparation day; old screenshots and the
  previously approved 506k-token figure are not authoritative.

## Working artifacts

Implementation creates a local `launch/2026-07-memo` branch in each repository
that receives campaign files, plus the following bundle:

```text
memo/docs/launch/2026-07/
├── copy.md          # approved initial posts and reply constraints
├── tracker.md       # accounts, rules, schedule, URLs, status, and approvals
├── metrics.csv      # timestamped public/owner-visible aggregate metrics
└── postmortem.md    # day-seven outcome and lessons

memo-web/public/launch/
├── demo.mp4
├── social-card.png
├── product-hunt-thumbnail.png
└── product-hunt-gallery-*.png
```

The working branch remains local/unpublished until the initial posts are live.
Only public copy, aggregate metrics, and sanitized assets enter these files. No
credential or private user data is recorded.

## Responsibilities and approval boundaries

| Activity | Assistant | User |
|---|---|---|
| Research current rules and account requirements | Performs and records | Reviews material exceptions |
| Draft initial posts and assets | Produces | Approves the complete batch before launch |
| Create missing account shell | Performs when the platform permits | Enters email/password and accepts platform terms |
| CAPTCHA, email link, 2FA, identity check | Waits | Completes directly |
| Publish approved initial posts | Performs where authenticated browser/API access permits | Takes over only when the platform requires a human-only action |
| Draft ordinary public replies | Drafts one at a time | Approves before publication |
| HN and DEV comments | Supplies facts/outline only | Authors the final comment |
| Moderation appeal or policy exception | Prepares factual context | Approves and sends |

Initial publication authorization is batch-based: once every row in
`tracker.md` has approved copy, asset, destination, and account status, the
assistant may execute that row at its scheduled time. Replies remain individually
gated.

## Readiness gate

The campaign is ready only when all mandatory checks pass by 18:00 ART on the
Monday before launch:

### Product

- `master` is the intended public release and required CI checks are green.
- The one-step macOS install and Linux/CPU path pass in isolated environments.
- The landing returns 200 for `/` and `/es/`, mobile and desktop layouts render,
  GitHub/install links work, and Vercel Analytics ingests a normal browser visit.
- No launch-blocking security, privacy, installation, or data-loss issue is open.

### Claims and assets

- Every numerical claim has a command, screenshot, or repository source captured
  on the final preparation day.
- All demo data is synthetic and the frame-by-frame privacy review passes.
- Every asset renders within the target platform's size and format limits.
- English copy is approved as a batch; Spanish copy can remain scheduled for the
  second wave but must be complete before launch day.

### Accounts and policy

- Required personal accounts exist, are logged in, and can publish.
- Product Hunt confirms the account is old enough and the launch is scheduled.
- Each selected subreddit has a recorded rule decision.
- The user is available for human verification and reply approvals during the
  first six hours.

### Measurement

- The GitHub star baseline and 14-day traffic baseline are captured.
- Vercel Analytics is live and its dashboard is accessible.
- The tracker contains native post URLs/IDs immediately after publication.

If any mandatory check fails, do not partially launch. Fix the issue and move to
the next Tuesday. The candidate dates are 2026-07-21 if all accounts and assets
are already eligible, otherwise 2026-07-28; a newly created Product Hunt account
makes 2026-07-28 the earliest candidate.

## Launch-day schedule

Product Hunt is scheduled for the start of its Pacific launch day; the platform's
displayed timezone is authoritative. Other times are America/Argentina/Cordoba.

| Time | Action |
|---|---|
| 00:01 Pacific | Product Hunt becomes live |
| 09:30 ART | Publish LinkedIn founder post |
| 10:15 ART | Publish X thread |
| 11:00 ART | Submit Show HN |
| 12:30 ART | Publish DEV technical article |
| 14:00–17:00 ART | Publish approved Reddit posts one at a time |
| 17:00–19:00 ART | Verify awesome-list/MCP entries and request refreshes only where stale |

The posts share a launch day but not a single timestamp. The stagger preserves
capacity to answer early questions and catch broken links before the next row.

At roughly +48 hours, publish the Spanish LinkedIn and X versions, the subtitled
demo, and only those Spanish-community posts that have passed the same rules
check. This is a new-language wave, not an identical repost into the same feeds.

## Response workflow

For each public question:

1. Record the permalink, channel, topic, and whether a response is needed.
2. Verify any technical fact against the repository or a fresh command.
3. Draft a concise reply in the channel's voice.
4. Present it to the user for approval.
5. Publish only the approved version and record the permalink.

Do not debate hostile low-signal comments, reveal private implementation context,
or promise roadmap items. Security reports move to GitHub's existing responsible
reporting path. Repeated questions become an owned-channel clarification or README
improvement only if they reveal a genuine documentation gap.

## Measurement design

### Primary KPI

Total GitHub stars at the end of day seven. Capture a fresh launch baseline `B`.
The approved checkpoints below assume `B = 7`. If `B` changes, recalculate each
checkpoint as `B + round((100 - B) * (T - 7) / 93)`, where `T` is that row's
displayed target. This preserves the approved progress curve and always ends at
100.

| Checkpoint | Target with `B = 7` |
|---|---:|
| +6 hours | 17 |
| +24 hours | 32 |
| Day 3 | 62 |
| Day 5 | 82 |
| Day 7 | 100 |

### Diagnostic metrics

- Platform impressions and engagements.
- Vercel visitors, pageviews, and referrers.
- GitHub repository views, unique visitors, clones, and top referrers.
- Meaningful technical comments, issues, contributions, and refreshed directory
  or awesome-list entries.

GitHub Traffic covers the previous 14 days and is available to repository users
with push access; traffic updates hourly/daily depending on the panel. See
[GitHub's traffic documentation](https://docs.github.com/en/repositories/viewing-activity-and-data-for-your-repository/viewing-traffic-to-a-repository)
and [traffic API](https://docs.github.com/en/rest/metrics/traffic). Star totals and,
where access permits, timestamps come from the
[GitHub starring API](https://docs.github.com/en/rest/activity/starring).
Vercel collection and privacy behavior follow the
[Web Analytics documentation](https://vercel.com/docs/analytics).

Record aggregate snapshots at baseline, +6h, +24h, D3, D5, and D7. Native
platform metrics are diagnostic: the campaign does not infer a star's source or
identity when the available aggregate data cannot establish it.

## Contingencies

- **Below 17 stars at +6h:** inspect impressions, landing/repository referrers,
  title, thumbnail, and first paragraph. Improve owned posts or add the strongest
  missing proof; do not duplicate community submissions.
- **Below 32 stars at +24h:** publish one genuinely new proof update on owned
  channels, then choose one acceleration: bring the Spanish wave forward into
  the +24h-to-+36h window, or activate one rules-validated reserve surface
  (Peerlist, an Indie Hackers product/build post, or a Hashnode canonical
  republish of the DEV article). Do not do both. Hashnode must point its
  canonical URL to the original article.
- **Below 62 stars on D3:** publish one technical lesson learned from real launch
  questions on the best-performing owned channel. Do not repeat the original
  announcement or add another reserve platform at this checkpoint.
- **Post removed or rejected:** do not repost or evade the decision. Ask
  moderators only when clarification is appropriate and record the result.
- **Critical product bug:** stop pending outbound posts, reproduce and fix the
  issue, rerun the readiness checks, then resume or move to the next Tuesday.
- **Analytics unavailable:** continue only if product and links are healthy;
  record a gap and recover metrics later from GitHub/platform-native sources.
- **HN or Product Hunt receives little traction:** do not request votes or
  resubmit during the campaign window.

Only one reserve surface is activated per checkpoint, preventing a low-signal
spray across marginal communities.

## Verification before publication

The implementation plan must include evidence for:

- Clean macOS and Linux/CPU installation in isolated state/data directories.
- `memo doctor --strict-runtime` or equivalent runtime health after install.
- Landing HTTP status, bilingual routes, responsive rendering, CTA destinations,
  social metadata, and normal-browser Analytics ingestion.
- Link validation for every post preview.
- Exact image/video dimensions, duration, subtitles, and absence of sensitive
  information.
- Product Hunt account eligibility and scheduled launch preview.
- Current rule capture for each selected subreddit and directory.
- Dry-run rendering or draft preview for every platform before batch approval.
- Baseline GitHub and Vercel metric snapshots.

No publication row may be marked ready merely because its copy exists.

## Deliverables

The campaign is complete when it has:

1. A fully approved evidence pack and channel copy bundle.
2. A passed readiness gate with recorded evidence.
3. Published or explicitly skipped every core channel with a reason.
4. Responded to approved high-value questions for seven days.
5. Recorded all metric checkpoints and contingencies.
6. Produced a day-seven postmortem stating the final star count, channel results,
   policy/moderation outcomes, defects found, and next recommended distribution
   experiment.

## References

- [Show HN](https://news.ycombinator.com/showhn.html)
- [Hacker News guidelines](https://news.ycombinator.com/newsguidelines.html)
- [Product Hunt posting guide](https://help.producthunt.com/en/articles/479557-how-to-post-a-product)
- [DEV editor guide](https://dev.to/p/editor_guide/)
- [DEV AI-assisted content guidelines](https://dev.to/guidelines-for-ai-assisted-articles-on-dev)
- [Official MCP Registry](https://github.com/modelcontextprotocol/registry)
- [Glama memo listing](https://glama.ai/mcp/servers/jagoff/memo)
- [mcpservers.org memo listing](https://mcpservers.org/servers/jagoff/memo)
- [awesome-mcp-servers contribution guide](https://github.com/punkpeye/awesome-mcp-servers/blob/main/CONTRIBUTING.md)
- [GitHub repository traffic](https://docs.github.com/en/repositories/viewing-activity-and-data-for-your-repository/viewing-traffic-to-a-repository)
- [Vercel Web Analytics](https://vercel.com/docs/analytics)
- [Hashnode publishing and canonical URL](https://docs.hashnode.com/blogs/editor/writing-a-blog-post)
