/**
 * memo update endpoint + anonymous active-install heartbeat (Tier 2).
 *
 * Two routes:
 *   GET /v1/latest?id=<hash>&v=<ver>&os=<name>
 *       Returns {"latest": "vX.Y.Z"} — the real job, a functional version check
 *       that memo's auto-update consumes instead of `git ls-remote`. As a side
 *       effect it records ONE deduped heartbeat per install: KV key `i:<id>`
 *       with {last, v, os} in metadata (45-day TTL). The id is already a hash of
 *       the device id (memo hashes it client-side); no raw identity, no IP is
 *       stored. Missing id → no heartbeat, still returns the latest tag.
 *
 *   GET /v1/stats?token=<STATS_TOKEN>
 *       Maintainer-only. Buckets installs by recency into DAU / WAU / MAU and
 *       returns version + OS histograms. Uses KV list() metadata only — no
 *       per-key reads — so it stays one cheap pass.
 *
 * Bindings (wrangler.toml):
 *   - KV namespace   MEMO_TEL
 *   - secret         STATS_TOKEN   (wrangler secret put STATS_TOKEN)
 *   - var            GH_REPO       (e.g. "jagoff/memo")
 *
 * Scale note: KV list is fine into the low tens of thousands of installs. Past
 * that, move heartbeats to D1 (a rows table keyed by id) and aggregate in SQL.
 */

const HEARTBEAT_TTL_S = 45 * 24 * 3600; // 45 days
const LATEST_CACHE_S = 900; // 15 min
const SEMVER = /^v?\d+\.\d+\.\d+$/;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method !== "GET") return json({ error: "method" }, 405);
    if (url.pathname === "/v1/latest") return handleLatest(url, env);
    if (url.pathname === "/v1/stats") return handleStats(url, env);
    return json({ error: "not found" }, 404);
  },
};

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function today() {
  return new Date().toISOString().slice(0, 10); // YYYY-MM-DD (UTC)
}

// --- /v1/latest -----------------------------------------------------------

async function handleLatest(url, env) {
  // Record heartbeat (best-effort; never blocks the version answer).
  const id = sanitizeId(url.searchParams.get("id"));
  if (id) {
    const v = sanitizeVer(url.searchParams.get("v"));
    const os = sanitizeOs(url.searchParams.get("os"));
    try {
      await env.MEMO_TEL.put(`i:${id}`, "", {
        expirationTtl: HEARTBEAT_TTL_S,
        metadata: { last: today(), v, os },
      });
    } catch (_) {
      // swallow — telemetry must never break the version check
    }
  }
  const latest = await latestTag(env);
  return json({ latest });
}

function sanitizeId(raw) {
  if (!raw) return null;
  return /^[0-9a-f]{1,32}$/.test(raw) ? raw : null;
}
function sanitizeVer(raw) {
  return raw && SEMVER.test(raw) ? raw.replace(/^v/, "") : "unknown";
}
function sanitizeOs(raw) {
  return raw && /^[A-Za-z]{1,16}$/.test(raw) ? raw : "unknown";
}

async function latestTag(env) {
  const cached = await env.MEMO_TEL.get("cfg:latest", { type: "json" });
  const now = Math.floor(Date.now() / 1000);
  if (cached && now - cached.at < LATEST_CACHE_S) return cached.tag;

  const tag = await fetchHighestTag(env.GH_REPO || "jagoff/memo");
  if (tag) {
    await env.MEMO_TEL.put("cfg:latest", JSON.stringify({ tag, at: now }), {
      expirationTtl: LATEST_CACHE_S * 4,
    });
    return tag;
  }
  return cached ? cached.tag : null; // stale-but-present beats null
}

async function fetchHighestTag(repo) {
  try {
    const resp = await fetch(`https://api.github.com/repos/${repo}/tags?per_page=100`, {
      headers: { "User-Agent": "memo-telemetry-worker", accept: "application/vnd.github+json" },
      cf: { cacheTtl: LATEST_CACHE_S, cacheEverything: true },
    });
    if (!resp.ok) return null;
    const tags = await resp.json();
    let best = null;
    let bestParts = null;
    for (const t of tags) {
      const name = t && t.name;
      if (!name || !SEMVER.test(name) || name.includes("-") || name.includes("+")) continue;
      const parts = name.replace(/^v/, "").split(".").map(Number);
      if (!bestParts || cmp(parts, bestParts) > 0) {
        best = name;
        bestParts = parts;
      }
    }
    return best;
  } catch (_) {
    return null;
  }
}

function cmp(a, b) {
  for (let i = 0; i < 3; i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return 0;
}

// --- /v1/stats ------------------------------------------------------------

async function handleStats(url, env) {
  if (!env.STATS_TOKEN || url.searchParams.get("token") !== env.STATS_TOKEN) {
    return json({ error: "unauthorized" }, 401);
  }
  const now = new Date();
  const dayMs = 24 * 3600 * 1000;
  const cutoff = (n) => new Date(now.getTime() - n * dayMs).toISOString().slice(0, 10);
  const d1 = cutoff(1);
  const d7 = cutoff(7);
  const d30 = cutoff(30);

  let dau = 0;
  let wau = 0;
  let mau = 0;
  const versions = {};
  const oses = {};

  let cursor;
  do {
    const page = await env.MEMO_TEL.list({ prefix: "i:", cursor, limit: 1000 });
    for (const key of page.keys) {
      const m = key.metadata || {};
      const last = m.last || "";
      if (last >= d1) dau++;
      if (last >= d7) wau++;
      if (last >= d30) {
        mau++;
        versions[m.v || "unknown"] = (versions[m.v || "unknown"] || 0) + 1;
        oses[m.os || "unknown"] = (oses[m.os || "unknown"] || 0) + 1;
      }
    }
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);

  return json({
    generated_at: now.toISOString(),
    active_installs: { dau, wau, mau },
    note: "active = update-check heartbeat within N days; lower bound (opt-in gated)",
    versions,
    os: oses,
  });
}
