# memo telemetry

Two independent ways to answer *"how many people use memo actively?"* — neither
sends any user content, ever.

## Tier 1 — passive proxies (no code ships to users)

`scripts/telemetry_report.py` pulls signals that already exist publicly:

- GitHub: stars / forks / watchers + the **14-day traffic window** (clones,
  unique cloners, views, unique viewers, referrers).
- PyPI: recent / weekly / monthly download counts for `mlx-memo`.

```bash
uv run --no-sync python scripts/telemetry_report.py            # render + snapshot
uv run --no-sync python scripts/telemetry_report.py --json     # machine output
```

Two caveats the report makes loud instead of hiding:

1. **Clones are polluted.** memo installs & auto-updates over
   `git+https://github.com/jagoff/memo.git`, CI clones every run, and you dev
   across several Macs — so `clones` is *not* users. Use **unique viewers** and
   PyPI as the honest interest floor.
2. **GitHub traffic is a rolling 14-day window.** Older data is gone from the
   API. The script snapshots each run to `~/.memo/telemetry-snapshots.jsonl`, so
   run it on a schedule to build history and see deltas. A daily user cron
   (keeps it out of synapse's launchd fleet — this is maintainer tooling):

   ```cron
   0 9 * * * cd ~/repos/memo && uv run --no-sync python scripts/telemetry_report.py --json >> ~/.memo/telemetry-cron.log 2>&1
   ```

## Tier 2 — anonymous active-install heartbeat (this folder)

A Cloudflare Worker that *is* memo's update-version endpoint. When a memo
install checks for a newer release, it can resolve the latest tag over HTTP from
this Worker instead of `git ls-remote`. That GET does real work (returns the
latest tag) **and** records one deduped heartbeat per install — giving a real,
deduped **DAU / WAU / MAU** the clone counts can't.

### What crosses the wire

`GET /v1/latest?id=<hash>&v=<version>&os=<name>` — that's the whole payload:

| field | value | notes |
|-------|-------|-------|
| `id`  | `sha256(device_id)[:16]` | hashed **client-side** in memo; the raw device id never leaves the machine |
| `v`   | current memo version | for a version-distribution histogram |
| `os`  | `platform.system()` (e.g. `Darwin`) | coarse OS bucket |

No memory content, no file paths, no IP is stored. The Worker keeps one KV key
per install (`i:<id>`, 45-day TTL) with `{last, v, os}` in metadata.

### Honest coverage limit

The heartbeat only fires when the install has **update checks on**
(`MEMO_UPDATE_CHECK_ENABLED` or `MEMO_AUTO_UPDATE`) **and** `MEMO_UPDATE_ENDPOINT`
is set. Both update flags are **default OFF**. So this is a **lower bound** on
active users, not a full count. Widening it (shipping a default endpoint and/or
flipping `MEMO_UPDATE_CHECK_ENABLED` on by default) is a product stance change on
an offline-first tool — decide that deliberately, and disclose it in
`PRIVACY.md` before doing so.

### Deploy

```bash
cd telemetry
npx wrangler kv namespace create MEMO_TEL     # paste the id into wrangler.toml
npx wrangler secret put STATS_TOKEN           # long random string
npx wrangler deploy                           # → https://memo-update.<sub>.workers.dev
```

Then point installs at it (your call how broadly):

```bash
memo config set update.endpoint https://memo-update.<sub>.workers.dev/v1/latest
# or in the daemon plist EnvironmentVariables: MEMO_UPDATE_ENDPOINT=...
# and ensure update checks are on:
memo config set update.check_enabled true
```

### Read the numbers

```bash
curl "https://memo-update.<sub>.workers.dev/v1/stats?token=$STATS_TOKEN"
# → { active_installs: { dau, wau, mau }, versions: {...}, os: {...} }
```

### Client fallback

If the endpoint is unreachable, memo silently falls back to `git ls-remote`
(`resolve_latest_tag` in `src/memo/runtime/autoupdate.py`), so a Worker outage
never breaks the update check.
