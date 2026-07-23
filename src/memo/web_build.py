"""Generate the memo Health Dashboard — a self-contained interactive HTML.

Reads memo state via the same Python APIs used by ``memo doctor`` / ``memo
profile status`` / ``memo map``: no extra services, no shell-outs. Emits a
single HTML file at ``web/health.html`` (or ``--output``) with:

  - 4 colour-coded pillar lights (red / yellow / green / blue)
  - 3-D scatter of all memory embeddings (PCA — UMAP optional)
  - Corpus growth bar chart
  - Type distribution donut
  - Recent saves / recalls table

Run through the source compatibility wrapper or the packaged module:
    python web/build.py [--output PATH] [--limit N] [--open]
    python -m memo.web_build [--output PATH] [--limit N] [--open]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import struct
import webbrowser
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SOURCE_REPO = Path(__file__).resolve().parents[2]
REPO_ROOT = _SOURCE_REPO if (_SOURCE_REPO / "pyproject.toml").is_file() else None

from memo.atomic_io import atomic_write_text  # noqa: E402
from memo.cli_diag import (  # noqa: E402
    _db_health_report,
    _profile_status_report,
)
from memo.cli_runtime import _runtime_install_report  # noqa: E402
from memo.config import Config  # noqa: E402
from memo.dashboard import (  # noqa: E402
    consult_breakdown,
    read_context_cost_log,
    read_daily_trend,
    read_recall_log,
    reask_stats,
    recall_health,
    verdict,
)
from memo.embedder_select import resolve_backend  # noqa: E402
from memo.errors import MemoError  # noqa: E402
from memo.html_security import (  # noqa: E402
    content_security_policy,
    html_safe_json,
    new_csp_nonce,
)

_log = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _iso_to_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        s = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        return None


def _decode_embedding(blob: bytes, dims: int) -> list[float]:
    """Decode a raw `vec` blob to floats, dtype-aware by length.

    int8 (MEMO_VEC_QUANTIZE=int8) is 1 B/dim, dequantized (÷127); float32 is
    4 B/dim. Length-based detection keeps this independent of the running
    config, since the blob's dtype is whatever it was indexed with.
    """
    if len(blob) == dims:
        return [x / 127.0 for x in struct.unpack(f"{dims}b", blob)]
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _read_vectors(db_path: Path, limit: int, dims: int) -> list[dict[str, Any]]:
    """Pull embeddings + metadata straight from sqlite-vec (no MLX load)."""
    try:
        from memo.sqlite_compat import import_sqlite_vec
    except ImportError:
        return []
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    try:
        conn.enable_load_extension(True)
        import_sqlite_vec().load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT vec.id, vec.embedding, meta.title, meta.type, "
            "       meta.tags, meta.created, meta.updated "
            "FROM vec JOIN meta ON meta.id = vec.id "
            "ORDER BY meta.updated DESC "
            "LIMIT ?",
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for row in rows:
        blob = row["embedding"]
        if blob is None:
            continue
        vec = _decode_embedding(blob, dims)
        if not vec:
            continue
        try:
            tags = json.loads(row["tags"] or "[]")
        except json.JSONDecodeError:
            tags = []
        out.append(
            {
                "id": row["id"],
                "vec": vec,
                "title": row["title"] or "—",
                "type": row["type"] or "note",
                "tags": [str(t) for t in tags][:6],
                "created": (row["created"] or "")[:10],
                "updated": (row["updated"] or "")[:10],
            }
        )
    return out


def _project_3d(vecs: list[list[float]]) -> tuple[list[float], list[float], list[float], str]:
    """3-D projection. Tries UMAP, falls back to PCA via numpy."""
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy required for build (pip install numpy)") from exc
    mat = np.array(vecs, dtype=np.float32)
    try:
        import umap  # type: ignore[import-not-found]

        n_neighbors = min(15, len(vecs) - 1)
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=0.12,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(mat)
        method = "UMAP"
    except ImportError:
        centered = mat - mat.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:3].T
        method = "PCA (install umap-learn for richer topology)"
    xs = coords[:, 0].tolist()
    ys = coords[:, 1].tolist()
    zs = coords[:, 2].tolist()
    return xs, ys, zs, method


def _body_hash_drift(cfg: Config) -> dict[str, int]:
    """Count memories whose .md body diverges from store body_hash.

    No re-embedding — just hashes on disk and compares.
    """
    out = {"checked": 0, "drifted": 0, "missing_file": 0, "untracked_md": 0}
    if not cfg.memory_dir.is_dir() or not cfg.db_path.is_file():
        return out
    try:
        import frontmatter  # type: ignore[import-not-found]
    except ImportError:
        return out
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT id, path, body_hash FROM meta").fetchall()
    finally:
        conn.close()
    indexed_ids: set[str] = set()
    for rec_id, rel, body_hash in rows:
        indexed_ids.add(rec_id)
        out["checked"] += 1
        md = cfg.memory_dir / rel
        if not md.is_file():
            out["missing_file"] += 1
            continue
        try:
            post = frontmatter.loads(md.read_text(encoding="utf-8"))
        except Exception:
            out["drifted"] += 1
            continue
        if _sha256_short(post.content or "") != body_hash:
            out["drifted"] += 1
    for md_path in cfg.memory_dir.rglob("*.md"):
        try:
            post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _log.debug("dashboard: could not parse %s: %s", md_path, exc)
            continue
        md_id = post.get("id")
        if isinstance(md_id, str) and md_id and md_id not in indexed_ids:
            out["untracked_md"] += 1
    return out


def _history_recent(cfg: Config, limit: int = 30) -> list[dict[str, Any]]:
    path = cfg.history_db
    if not path.is_file():
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT ts, op, record_id, title, type FROM events ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
    except Exception:
        return []
    return [{"ts": r[0], "op": r[1], "id": r[2][:8], "title": r[3], "type": r[4]} for r in rows]


def _growth_by_day(rows: list[dict[str, Any]], days: int = 30) -> list[dict[str, Any]]:
    """Saves per day for the last `days` calendar days."""
    today = datetime.now(UTC).date()
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["op"] != "save":
            continue
        dt = _iso_to_dt(r["ts"])
        if not dt:
            continue
        counts[dt.date().isoformat()] += 1
    out: list[dict[str, Any]] = []
    for i in range(days - 1, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        out.append({"date": d, "count": counts.get(d, 0)})
    return out


def _contradictions_stats(cfg: Config) -> dict[str, int] | None:
    path = cfg.contradictions_db
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = conn.execute("SELECT status, count(*) FROM pairs GROUP BY status").fetchall()
        conn.close()
    except Exception:
        return None
    return {str(r[0]): int(r[1]) for r in rows}


# ── pillars ──────────────────────────────────────────────────────────────
#
# Each pillar reports {label, status, summary, detail}. Status is one of
# "green" / "yellow" / "red" / "blue". Blue = informational (no health
# signal); the others are the traffic-light contract from the spec.


def _pillar_vector_db(doctor: dict[str, Any], drift: dict[str, int] | None) -> dict[str, Any]:
    memvec = next((d for d in doctor["db"] if d.get("label") == "memvec"), None)
    if not memvec or not memvec.get("exists"):
        return {
            "label": "Vector DB",
            "status": "red",
            "summary": "memvec.db missing",
            "detail": [
                "Run `memo reindex` to build the vector index.",
            ],
        }
    sqlite_vec_ok = next(
        (i for i in doctor["imports"] if i["label"] == "sqlite_vec"), {"ok": False}
    )["ok"]
    if not sqlite_vec_ok:
        return {
            "label": "Vector DB",
            "status": "red",
            "summary": "sqlite-vec extension cannot load",
            "detail": ["Install sqlite-vec or check the Python environment."],
        }
    if memvec.get("status") == "dimension_mismatch":
        return {
            "label": "Vector DB",
            "status": "red",
            "summary": (
                f"dim mismatch · vec={memvec.get('vec_dims')} "
                f"expected={memvec.get('expected_dims')}"
            ),
            "detail": [
                "Backup and rebuild the vector index:",
                "  memo backup --out memo-pre-repair.zip",
                f"  rm {memvec['path']}",
                "  memo reindex",
            ],
        }
    if memvec.get("integrity_check") and memvec["integrity_check"].lower() != "ok":
        return {
            "label": "Vector DB",
            "status": "red",
            "summary": f"integrity_check={memvec['integrity_check']}",
            "detail": ["sqlite reports DB corruption — restore from backup."],
        }
    if drift is not None and (drift["drifted"] or drift["untracked_md"]):
        return {
            "label": "Vector DB",
            "status": "yellow",
            "summary": (f"{drift['drifted']} drifted · {drift['untracked_md']} untracked"),
            "detail": [
                f"records:        {memvec.get('records')}",
                f"checked:        {drift['checked']}",
                f"body_hash drift: {drift['drifted']}",
                f"missing files:   {drift['missing_file']}",
                f"untracked .md:   {drift['untracked_md']}",
                "Run `memo reindex` to re-embed only changed entries.",
            ],
        }
    detail = [
        f"path: {memvec['path']}",
        f"size: {memvec.get('size_bytes', 0):,} bytes",
        f"integrity: {memvec.get('integrity_check', '?')}",
        f"latest update: {memvec.get('latest_memory_update', '—')}",
    ]
    if drift is None:
        # Cheap poll path skips the full-corpus body_hash scan (rglob + parse).
        detail.append("body_hash drift: not checked (live poll)")
    return {
        "label": "Vector DB",
        "status": "green",
        "summary": f"{memvec.get('records')} memories · {memvec.get('vec_dims')}D",
        "detail": detail,
    }


def _pillar_embedder(doctor: dict[str, Any]) -> dict[str, Any]:
    profile = doctor["profile"]
    imports = doctor["imports"]
    # Backend-aware: a CPU/ST install has a "sentence_transformers" probe, an
    # Apple-Silicon install an "mlx" probe. Keying off "mlx" alone showed every
    # healthy Linux/CPU install a false-RED "mlx not importable".
    st_probe = next((i for i in imports if i["label"] == "sentence_transformers"), None)
    if st_probe is not None:
        pillar_label = "Embedder (CPU)"
        backend_import = st_probe
        install_hint = "Install with `pip install sentence-transformers`."
    else:
        pillar_label = "Embedder (MLX)"
        backend_import = next(
            (i for i in imports if i["label"] == "mlx"), {"ok": False, "error": ""}
        )
        install_hint = "Install with `pip install mlx mlx-lm` on Apple Silicon."
    if not backend_import["ok"]:
        return {
            "label": pillar_label,
            "status": "red",
            # Surface the real import error instead of a swallowed generic string.
            "summary": backend_import.get("error") or "embedder runtime not importable",
            "detail": [install_hint],
        }
    embedder_model = profile["active"].get("embedder_model", "?")
    cached = any(m["cached"] and m["role"] == "embedder" for m in profile["models"])
    if not cached:
        return {
            "label": pillar_label,
            "status": "yellow",
            "summary": "model weights not in HF cache",
            "detail": [
                f"model: {embedder_model}",
                "Will download on first use. Run `memo prewarm` to fetch now.",
            ],
        }
    if not profile["ok"]:
        return {
            "label": pillar_label,
            "status": "red",
            "summary": profile["status"],
            "detail": [f"model: {embedder_model}", "See `memo profile status`."],
        }
    return {
        "label": pillar_label,
        "status": "green",
        "summary": f"{profile['profile']} · {profile['active']['embedder_dims']}D",
        "detail": [
            f"model:    {embedder_model}",
            f"llm:      {profile['active'].get('llm_model')}",
            f"helper:   {profile['active'].get('helper_model')}",
            f"reranker: {profile['active'].get('reranker_model') or 'disabled'}",
        ],
    }


def _pillar_recall(recall_log: list[dict[str, Any]]) -> dict[str, Any]:
    if not recall_log:
        return {
            "label": "Recall Hook",
            "status": "yellow",
            "summary": "no recall events logged",
            "detail": [
                "The UserPromptSubmit hook hasn't fired yet (or logging is",
                "off). Confirm the hook is installed and prompts arrive.",
            ],
        }
    latest = recall_log[0]
    latest_dt = _iso_to_dt(latest.get("ts", ""))
    age_h = None
    if latest_dt:
        age_h = (datetime.now(UTC) - latest_dt).total_seconds() / 3600
    latencies = [int(r["latency_ms"]) for r in recall_log if r.get("latency_ms")]
    avg_ms = int(sum(latencies) / len(latencies)) if latencies else None
    over_budget = sum(1 for ms in latencies if ms > 5000)
    if age_h is not None and age_h > 48:
        status = "yellow"
        summary = f"last event {int(age_h)}h ago"
    elif avg_ms and avg_ms > 5000:
        status = "red"
        summary = f"avg {avg_ms}ms over 5s budget"
    elif over_budget:
        status = "yellow"
        summary = f"{over_budget}/{len(latencies)} events over 5s"
    else:
        status = "green"
        summary = f"{len(recall_log)} events · avg {avg_ms or 0}ms"
    detail = [
        f"latest: {latest.get('ts', '—')}",
        f"avg latency: {avg_ms or 0} ms",
        f"events logged: {len(recall_log)}",
        f"over-budget (>5s): {over_budget}",
    ]
    return {
        "label": "Recall Hook",
        "status": status,
        "summary": summary,
        "detail": detail,
    }


def _pillar_corpus(
    doctor: dict[str, Any],
    rows: list[dict[str, Any]],
    history: list[dict[str, Any]],
    contradictions: dict[str, int] | None,
    *,
    vec_count: int | None = None,
) -> dict[str, Any]:
    memvec: dict[str, Any] = next((d for d in doctor["db"] if d.get("label") == "memvec"), {})
    records = memvec.get("records") or 0
    type_counts = Counter(r["type"] for r in rows)
    n_vecs = vec_count if vec_count is not None else len(rows)
    last_save = next((h for h in history if h["op"] == "save"), None)
    detail = [
        f"records: {records}",
        f"with vectors: {n_vecs}",
        f"types: {dict(type_counts.most_common(5))}",
    ]
    if last_save:
        detail.append(f"latest save: {last_save['ts']}")
    if contradictions:
        open_n = contradictions.get("open", 0)
        detail.append(f"contradictions open: {open_n}")
    detail.append(f"data_dir: {doctor['storage']['data_dir']['path']}")
    return {
        "label": "Corpus",
        "status": "blue",
        "summary": f"{records:,} memories",
        "detail": detail,
    }


# ── HTML ─────────────────────────────────────────────────────────────────

_TYPE_COLORS = {
    "decision": "#4f8ef7",
    "fact": "#34d399",
    "bug": "#f87171",
    "preference": "#a78bfa",
    "feedback": "#fb923c",
    "note": "#94a3b8",
    "manual": "#e2e8f0",
}


def _render_html(data: dict[str, Any], *, nonce: str | None = None) -> str:
    nonce = nonce or new_csp_nonce()
    payload = html_safe_json(data, ensure_ascii=False, default=str)
    return (
        _HTML_TEMPLATE.replace("__DATA_JSON__", payload)
        .replace("__CSP_NONCE__", nonce)
        .replace("__CSP_POLICY__", content_security_policy(nonce, allow_local_fetch=True))
    )


def _usefulness(cfg: Config) -> dict[str, Any]:
    """Simple, cheap reader-impact metrics — who reads memo and whether it helps.

    Mirrors what `memo usefulness` / `memo roi` print, surfaced as plain numbers
    so the dashboard can show them without re-deriving anything heavy."""
    state_dir = cfg.state_dir
    try:
        breakdown = consult_breakdown(state_dir, limit=500)
    except Exception:
        breakdown = {"sampled": 0, "consumers": []}
    try:
        reask = reask_stats(state_dir, limit=500)
    except Exception:
        reask = {}
    return {
        "sampled": breakdown.get("sampled", 0),
        "consumers": breakdown.get("consumers", []),
        "silent": breakdown.get("silent", []),
        "reask_avoided": reask.get("reask_avoided"),
        "reask_considered": reask.get("considered"),
    }


def _consult_trend(state_dir: Path, *, days: int = 14, limit: int = 1000) -> list[dict[str, Any]]:
    """Consults per calendar day (last ``days``), split activated vs total.

    Uses daily_trend.json (persistent accumulator) as primary source; falls back
    to reading recall.log for any day not yet in the file (covers today's fresh
    entries before the first flush).  Shows adoption over time."""
    # Persistent per-day counters (survives recall.log rotation)
    persisted = read_daily_trend(state_dir)

    # Merge today's live recall.log on top (in case daily_trend.json lags behind)
    live: dict[str, dict[str, int]] = defaultdict(lambda: {"consultas": 0, "activado": 0})
    for r in read_recall_log(state_dir, limit=limit):
        day = (r.get("ts") or "")[:10]
        if not day:
            continue
        live[day]["consultas"] += 1
        if r.get("via") in ("daemon", "subprocess"):
            live[day]["activado"] += 1

    # Build merged view by per-field MAX. daily_trend.json is the synchronous,
    # complete accumulator (incremented on every recall append); recall.log is
    # size-capped and trims to its last lines, so the live count can only
    # UNDER-report a busy day. Taking the max means a trimmed recall.log never
    # shrinks today's bar below the persisted total, while still recovering a day
    # that daily_trend missed (e.g. a failed trend write, or rows logged by an
    # older runtime before the accumulator existed).
    merged: dict[str, dict[str, int]] = {}
    for d, v in persisted.items():
        merged[d] = dict(v)
    for d, v in live.items():
        cur = merged.get(d)
        if cur is None:
            merged[d] = dict(v)
        else:
            merged[d] = {
                "consultas": max(int(cur.get("consultas", 0)), v["consultas"]),
                "activado": max(int(cur.get("activado", 0)), v["activado"]),
            }

    out: list[dict[str, Any]] = []
    for i in range(days - 1, -1, -1):
        d = (datetime.now(UTC).date() - timedelta(days=i)).isoformat()
        b = merged.get(d, {"consultas": 0, "activado": 0})
        out.append({"date": d, "consultas": b["consultas"], "activado": b["activado"]})
    return out


def _fmt_tokens_compact(tokens: float) -> str:
    tokens = int(tokens)
    if tokens < 1000:
        return str(tokens)
    if tokens < 1_000_000:
        return f"{tokens / 1000:.1f}k"
    return f"{tokens / 1_000_000:.2f}M"


def _token_savings(state_dir: Path, *, days: int = 14) -> dict[str, Any]:
    """Detailed token-savings breakdown for the dashboard graph.

    Two honest drivers, each a clearly-labeled estimate:
      - "hechos reutilizados" — a surfaced memory the answer actually used
        (grounding.log, used_score ≥ GROUNDED_SCORE, deduped by sid+turn+id) ×
        ``MEMO_ROI_TOKENS_PER_GROUNDED``: tokens the model didn't spend
        re-deriving the fact. This one is DATED, so it drives the daily series.
      - "repreguntas evitadas" — grounded recalls the user did not have to ask
        again × ``MEMO_ROI_TOKENS_PER_REASK``: a saved answer-regeneration
        round-trip. A session-level metric (no clean per-day bucket), shown only
        in the composition total, not the daily bars.
    """
    from memo.dashboard import GROUNDED_SCORE, read_grounding_log
    from memo.flags import flag_int

    tok_grounded = flag_int("MEMO_ROI_TOKENS_PER_GROUNDED") or 350
    tok_reask = flag_int("MEMO_ROI_TOKENS_PER_REASK") or 900

    seen: set[tuple[str, int, str]] = set()
    by_day: dict[str, int] = defaultdict(int)
    context_by_day: dict[str, int] = defaultdict(int)
    context_costs: dict[str, int] = defaultdict(int)
    answer_lens: list[int] = []
    for grounding in read_grounding_log(state_dir):
        if grounding.get("answer_len"):
            answer_lens.append(int(grounding["answer_len"]))
        sid = grounding.get("session_id")
        turn = grounding.get("turn")
        rid = grounding.get("recall_id")
        score = grounding.get("used_score")
        day = (grounding.get("ts") or "")[:10]
        if not (
            sid
            and isinstance(turn, int)
            and rid
            and day
            and isinstance(score, (int, float))
            and float(score) >= GROUNDED_SCORE
        ):
            continue
        key = (sid, turn, rid)
        if key in seen:
            continue
        seen.add(key)
        by_day[day] += 1

    for row in read_context_cost_log(state_dir):
        day = str(row.get("ts") or "")[:10]
        kind = str(row.get("kind") or "unknown")
        tokens = row.get("tokens_est")
        if not isinstance(tokens, (int, float)):
            chars = max(0, int(row.get("chars") or 0))
            tokens = (chars + 3) // 4
        token_count = max(0, int(tokens))
        context_costs[kind] += token_count
        if day:
            context_by_day[day] += token_count

    today = datetime.now(UTC).date()
    daily: list[dict[str, Any]] = []
    for i in range(days - 1, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        grounded_count = by_day.get(d, 0)
        gross = grounded_count * tok_grounded
        context_tokens = context_by_day.get(d, 0)
        daily.append(
            {
                "date": d,
                "grounded": grounded_count,
                "tokens": gross,
                "context_tokens": context_tokens,
                # "Ahorro" floors at 0: savings only count grounding-scored recalls
                # while context cost counts every injection, so thin measurement
                # coverage makes net artificially negative. A day with no measured
                # savings is "saved nothing" (0), not "cost you tokens".
                "net_tokens": max(0, gross - context_tokens),
            }
        )

    grounded_total = sum(by_day.values())
    # Historic total must survive grounding.log rotation (capped, ~12 days):
    # fold into the durable per-day ledger and take the all-time grounded count
    # from it, so this headline grows like `memo tokens` instead of plateauing
    # when old grounded rows scroll out of the log. Daily bars stay windowed.
    try:
        from memo import token_ledger

        token_ledger.roll_up(state_dir)
        durable = token_ledger.summarize(state_dir)["historic"]["grounded"]
        grounded_total = max(grounded_total, durable)
    except Exception:  # noqa: S110 - best-effort; fall back to the in-log total
        pass
    try:
        reask = reask_stats(state_dir, limit=500)
    except Exception:
        reask = {}
    reask_avoided = int(reask.get("reask_avoided") or 0)
    grounded_tokens = grounded_total * tok_grounded
    reask_tokens = reask_avoided * tok_reask
    context_tokens = sum(context_costs.values())
    today_key = today.isoformat()
    today_tokens = next((d["tokens"] for d in daily if d["date"] == today_key), 0)
    today_context_tokens = next((d["context_tokens"] for d in daily if d["date"] == today_key), 0)
    total = grounded_tokens + reask_tokens
    return {
        "daily": daily,
        "today_tokens": today_tokens,
        "grounded": grounded_total,
        "grounded_tokens": grounded_tokens,
        "reask_avoided": reask_avoided,
        "reask_tokens": reask_tokens,
        "context_costs": dict(sorted(context_costs.items())),
        "context_tokens": context_tokens,
        "today_context_tokens": today_context_tokens,
        "today_net": max(0, today_tokens - today_context_tokens),
        "total": total,
        "net": max(0, total - context_tokens),
        "tok_grounded": tok_grounded,
        "tok_reask": tok_reask,
        "avg_answer_tokens": (
            round(sum(answer_lens) / len(answer_lens) / 4) if answer_lens else None
        ),
    }


def _sync_health(cfg: Config) -> dict[str, Any]:
    """GitHub sync health for the dashboard — is durable knowledge reaching the
    shared repo so other sessions/Macs can read it? Cheap read; [] on error."""
    try:
        from memo.sync_git import sync_status

        st = sync_status(cfg)
        if not st.get("is_git_clone"):
            return {"state": "off", "label": "Sin sync (data_dir no es repo git)"}
        if st.get("pending"):
            return {"state": "bad", "label": "Commits varados — push falló", **st}
        if st.get("ahead") or st.get("dirty_files"):
            return {
                "state": "warn",
                "label": f"{st['ahead']} sin pushear · {st['dirty_files']} sin commitear",
                **st,
            }
        return {"state": "ok", "label": "Al día con GitHub", **st}
    except Exception:
        return {"state": "off", "label": "Sync no disponible"}


def _gaps(cfg: Config, *, top: int = 12) -> list[dict[str, Any]]:
    """Knowledge gaps — what memo could NOT answer (The Outcome Loop). Cheap
    read; degrades to [] if the outcome module/logs are unavailable."""
    try:
        from memo.outcome import detect_gaps

        return detect_gaps(cfg.state_dir)[:top]
    except Exception:
        return []


def _gerencial(cfg: Config) -> dict[str, Any]:
    """Management-grade rollup: the consult funnel, the value numbers, the
    per-tool reader breakdown, and the adoption trend — everything the gerencial
    dashboard needs in plain numbers, no re-derivation in the browser."""
    state_dir = cfg.state_dir
    try:
        health = recall_health(state_dir, limit=500)
    except Exception:
        health = {}
    try:
        from memo.cli_roi import compute_roi

        roi = compute_roi(state_dir, limit=500)
    except Exception:
        roi = {}

    fired = int(health.get("fired") or 0)
    sampled = int(health.get("sampled") or 0)
    daily = read_daily_trend(state_dir)
    consults_total = sum(int(row.get("consultas") or 0) for row in daily.values())
    activated_total = sum(int(row.get("activado") or 0) for row in daily.values())

    # The top summary is intentionally split by source:
    # - historical totals from the persistent daily accumulator
    # - recent sample from recall/log windows for quality metrics
    funnel = [
        {
            "key": "preguntas",
            "label": "Consultas totales",
            "value": consults_total,
            "sub": "acumulado histórico en daily_trend.json",
        },
        {
            "key": "muestra",
            "label": "Consultas analizadas",
            "value": sampled,
            "sub": "muestra reciente usada para hit/grounding",
        },
        {
            "key": "activadas",
            "label": "Activaciones históricas",
            "value": activated_total,
            "sub": "consultas totales que abrieron memo",
        },
    ]

    k_total = int(health.get("answers_knowledge_total") or 0)
    used_rate = health.get("answer_rate_knowledge") if k_total else health.get("answer_rate")
    used_total = k_total if k_total else int(health.get("answers_total") or 0)
    used_grounded = (
        int(health.get("answers_knowledge_grounded") or 0)
        if k_total
        else int(health.get("answers_grounded") or 0)
    )

    reask_value = roi.get("reask")
    reask: dict[str, Any] = reask_value if isinstance(reask_value, dict) else {}
    # Detailed token-savings (full grounding.log, daily series + composition).
    # The KPI reads from this too so the headline number == the panel total.
    token_detail = _token_savings(state_dir)
    return {
        "funnel": funnel,
        "consults": consults_total,
        "consults_total": consults_total,
        "consults_sampled": sampled,
        "activations_total": activated_total,
        "activations_sampled": fired,
        "coverage_rate": round(activated_total / consults_total, 3) if consults_total else None,
        "activation_rate_total": round(activated_total / consults_total, 3)
        if consults_total
        else None,
        "activation_rate_sampled": round(fired / sampled, 3) if sampled else None,
        "hit_rate": health.get("hit_rate"),
        "grounded_rate": health.get("grounded_rate"),
        "referenced_rate": health.get("referenced_rate"),
        "used_rate": used_rate,
        "used_total": used_total,
        "used_grounded": used_grounded,
        "measurement_coverage": health.get("measurement_coverage"),
        "measured_turns": health.get("measured_turns"),
        "surfaced_turns": health.get("surfaced_turns"),
        "grounding_age_hours": health.get("grounding_age_hours"),
        "time_saved_human": roi.get("time_saved_human"),
        "reask_avoided": reask.get("reask_avoided"),
        "tokens_saved_today": token_detail["today_tokens"],
        "tokens_saved_today_human": _fmt_tokens_compact(token_detail["today_tokens"]),
        "context_tokens_today": token_detail["today_context_tokens"],
        "tokens_net_today": token_detail["today_net"],
        "tokens_net_today_human": _fmt_tokens_compact(token_detail["today_net"]),
        "tokens_saved": token_detail["total"],
        "tokens_saved_human": _fmt_tokens_compact(token_detail["total"]),
        "tokens_net": token_detail["net"],
        "tokens_net_human": _fmt_tokens_compact(token_detail["net"]),
        "avg_answer_tokens": token_detail["avg_answer_tokens"],
        "token_detail": token_detail,
        "trend": _consult_trend(state_dir),
    }


def collect_data(
    cfg: Config, *, include_projection: bool = True, limit: int = 1500
) -> dict[str, Any]:
    """Gather every metric the dashboard renders, as a JSON-serializable dict.

    ``include_projection`` gates the only expensive step (reading all vectors +
    PCA): the static build wants it, but the live-refresh endpoint skips it so a
    poll stays cheap.
    """
    doctor: dict[str, Any] = {
        "schema": "memo.doctor.v1",
        "runtime": _runtime_install_report(),
        "storage": {
            "data_dir": {"path": str(cfg.data_dir), "ok": cfg.data_dir.is_dir()},
            "vault_path": {
                "path": str(cfg.vault_path) if cfg.vault_path else "",
                "ok": True if cfg.vault_path is None else cfg.vault_path.is_dir(),
                "set": cfg.vault_path is not None,
            },
        },
        "profile": _profile_status_report(cfg, include_db=True),
        "imports": _imports_probe(cfg),
        "db": _db_health_report(cfg),
    }

    recall_log = read_recall_log(cfg.state_dir, limit=200)
    recall_health_data = recall_health(cfg.state_dir, limit=500)
    history = _history_recent(cfg, limit=50)
    contradictions = _contradictions_stats(cfg)

    # Heavy step (read every vector + PCA) — only on a full build, not per poll.
    # _vec_count feeds the corpus pillar on the cheap poll path; initialise it
    # here so the `len(rows) if rows else _vec_count` fallback is always bound,
    # even on a full build whose corpus read came back empty.
    _vec_count = 0
    # body_hash drift also scans the whole corpus (rglob + double frontmatter
    # parse), so it too is a full-build-only step; the cheap poll leaves it None
    # and the Vector DB pillar renders "drift: not checked".
    drift: dict[str, int] | None = None
    if include_projection:
        drift = _body_hash_drift(cfg)
        rows = _read_vectors(cfg.db_path, limit=limit, dims=cfg.embedder_dims)
        if len(rows) >= 3:
            xs, ys, zs, method = _project_3d([r["vec"] for r in rows])
        else:
            xs, ys, zs, method = [], [], [], "n/a (need ≥ 3 memories with vectors)"
        projection = {
            "method": method,
            "xs": xs,
            "ys": ys,
            "zs": zs,
            "ids": [r["id"][:8] for r in rows],
            "titles": [r["title"] for r in rows],
            "types": [r["type"] for r in rows],
            "tags": [", ".join(r["tags"]) for r in rows],
            "created": [r["created"] for r in rows],
        }
        type_counts: dict[str, int] = dict(Counter(r["type"] for r in rows))
    else:
        rows = []
        projection = None
        type_counts = {}
        # Cheap vector count for corpus pillar (no blob reads, no PCA)
        _vec_count = 0
        if cfg.db_path.is_file():
            _conn = None
            try:
                import sqlite3 as _sqlite3

                from memo.sqlite_compat import import_sqlite_vec

                _conn = _sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
                _conn.enable_load_extension(True)
                import_sqlite_vec().load(_conn)
                _conn.enable_load_extension(False)
                _vec_count = _conn.execute("SELECT COUNT(*) FROM vec").fetchone()[0]
            except Exception:
                _vec_count = 0
            finally:
                if _conn is not None:
                    _conn.close()

    pillars = [
        _pillar_vector_db(doctor, drift),
        _pillar_embedder(doctor),
        _pillar_recall(recall_log),
        _pillar_corpus(
            doctor, rows, history, contradictions, vec_count=len(rows) if rows else _vec_count
        ),
    ]

    growth = _growth_by_day(history, days=30)

    data = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "memo_version": doctor["runtime"].get("memo_version"),
        "pillars": pillars,
        "projection": projection,
        "type_palette": _TYPE_COLORS,
        "type_counts": type_counts,
        "growth": growth,
        "history": history[:20],
        "recall_log": recall_log[:20],
        "recall_util": {
            "hit_rate": recall_health_data["hit_rate"],
            "fired": recall_health_data["fired"],
            "bailed": recall_health_data["bailed"],
            "sampled": recall_health_data["sampled"],
            "strong_hit_rate": recall_health_data["strong_hit_rate"],
            "median_top_score": recall_health_data["median_top_score"],
            "grounded_rate": recall_health_data["grounded_rate"],
            "grounded": recall_health_data["grounded"],
            "grounded_surfaced": recall_health_data["grounded_surfaced"],
            "answer_rate": recall_health_data["answer_rate"],
            "answers_total": recall_health_data["answers_total"],
            "answers_grounded": recall_health_data["answers_grounded"],
            "answer_rate_knowledge": recall_health_data["answer_rate_knowledge"],
            "answers_knowledge_total": recall_health_data["answers_knowledge_total"],
            "answers_knowledge_grounded": recall_health_data["answers_knowledge_grounded"],
            "surfaced_turns": recall_health_data["surfaced_turns"],
            "measured_turns": recall_health_data["measured_turns"],
            "measurement_coverage": recall_health_data["measurement_coverage"],
            "grounding_last_seen": recall_health_data["grounding_last_seen"],
            "grounding_age_hours": recall_health_data["grounding_age_hours"],
            "unmeasured_surfaced": recall_health_data["unmeasured_surfaced"],
            "referenced_rate": recall_health_data["referenced_rate"],
        },
        "bail_breakdown": recall_health_data["bail_breakdown"],
        "usefulness": _usefulness(cfg),
        "doctor_raw": doctor,
        "contradictions": contradictions or {},
        "verdict": verdict(cfg.state_dir, limit=500),
        "gerencial": _gerencial(cfg),
        "gaps": _gaps(cfg),
        "sync": _sync_health(cfg),
    }
    return data


def build(
    *,
    output: Path | None = None,
    limit: int = 1500,
    open_browser: bool = False,
) -> Path:
    cfg = Config.from_env()
    data = collect_data(cfg, include_projection=True, limit=limit)
    default_output = (
        REPO_ROOT / "web" / "health.html" if REPO_ROOT else cfg.state_dir / "health.html"
    )
    out_path = output if output else default_output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, _render_html(data), mode=0o600)
    if open_browser:
        webbrowser.open(out_path.as_uri())
    return out_path


def _imports_probe(cfg: Config) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        backend = resolve_backend(cfg)
    except MemoError:
        # Undecidable platform (auto on non-Linux/non-Apple-Silicon): keep the
        # MLX probe so its real ImportError reaches the pillar, not a swallow.
        backend = "mlx"
    probes: list[tuple[str, Callable[[], None]]] = [("sqlite_vec", _probe_sqlite_vec)]
    if backend == "st":
        probes.append(("sentence_transformers", _probe_sentence_transformers))
    else:
        probes.append(("mlx", _probe_mlx))
    for label, fn in probes:
        try:
            fn()
            out.append({"label": label, "ok": True, "error": ""})
        except Exception as exc:
            # Preserve the real error (e.g. "No module named 'mlx'") instead of a
            # generic "probe unavailable" that hides why the pillar is red.
            out.append({"label": label, "ok": False, "error": str(exc) or type(exc).__name__})
    return out


def _probe_sqlite_vec() -> None:
    from memo.sqlite_compat import import_sqlite_vec

    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        import_sqlite_vec().load(conn)
    finally:
        conn.close()


def _probe_mlx() -> None:
    import mlx.core  # noqa: F401
    import mlx_lm  # noqa: F401


def _probe_sentence_transformers() -> None:
    import sentence_transformers  # noqa: F401


# ── HTML / JS template ───────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>memo · ¿funciona como memory?</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<meta name="referrer" content="no-referrer" />
<meta http-equiv="Content-Security-Policy" content="__CSP_POLICY__" />
<style>
  :root {
    --bg: #0a0e16;
    --bg-soft: #0e1320;
    --panel: #131a28;
    --panel-soft: #1a2233;
    --border: #243049;
    --fg: #eef3fb;
    --fg-mute: #93a1b8;
    --fg-dim: #5d6b85;
    --green: #2ee6a6;
    --yellow: #fbbf24;
    --red:    #fb7185;
    --blue:   #5b9dff;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.25);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background:
      radial-gradient(1200px 600px at 80% -10%, #16203a 0%, var(--bg) 55%); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif; -webkit-font-smoothing: antialiased; }

  header { padding: 22px 28px 18px; display: flex; align-items: baseline;
           gap: 14px; flex-wrap: wrap; max-width: 1180px; margin: 0 auto; }
  header h1 { margin: 0; font-size: 17px; font-weight: 700; letter-spacing: .3px; }
  header h1 .sub { color: var(--fg-mute); font-weight: 400; }
  header .meta { color: var(--fg-dim); font-size: 12px; }
  .live-badge { margin-left: auto; font-size: 12px; color: var(--fg-mute);
                display: inline-flex; align-items: center; gap: 6px;
                background: var(--panel); border: 1px solid var(--border);
                padding: 5px 11px; border-radius: 999px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block;
         box-shadow: 0 0 8px currentColor; }
  .dot.green  { background: var(--green);  color: var(--green); }
  .dot.yellow { background: var(--yellow); color: var(--yellow); }
  .dot.red    { background: var(--red);    color: var(--red); }
  .dot.blue   { background: var(--blue);   color: var(--blue); }
  .dot.live { animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

  main { padding: 4px 28px 48px; display: grid; gap: 18px;
         max-width: 1180px; margin: 0 auto; }
  .panel { background: var(--panel); border: 1px solid var(--border);
           border-radius: 16px; padding: 20px 22px; box-shadow: var(--shadow); }
  .panel > h2 { margin: 0 0 4px; font-size: 12px; font-weight: 600;
                color: var(--fg-mute); text-transform: uppercase; letter-spacing: .8px; }
  .panel .hint { color: var(--fg-dim); font-size: 12.5px; margin: 0 0 16px; line-height: 1.5; }

  /* ── verdict hero ── */
  .verdict { position: relative; overflow: hidden; border-width: 1px;
             display: flex; align-items: center; gap: 26px; padding: 30px 30px; }
  .verdict .glyph { font-size: 56px; line-height: 1; filter: drop-shadow(0 2px 8px rgba(0,0,0,.4)); }
  .verdict .vmain { flex: 1; min-width: 0; }
  .verdict .vlabel { font-size: clamp(26px, 4vw, 40px); font-weight: 800;
                     letter-spacing: -.5px; line-height: 1.05; }
  .verdict .vexplain { color: var(--fg-mute); font-size: 15px; margin-top: 8px; line-height: 1.5; }
  .verdict .vstat { text-align: right; }
  .verdict .vstat b { display: block; font-size: 30px; font-weight: 800; letter-spacing: -1px; }
  .verdict .vstat span { color: var(--fg-dim); font-size: 11.5px; text-transform: uppercase; letter-spacing: .6px; }
  @media (max-width: 720px) { .verdict { flex-wrap: wrap; } .verdict .vstat { text-align: left; } }

  /* ── funnel ── */
  .funnel { display: grid; gap: 10px; }
  .fstage { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 16px;
            padding: 10px 0; border-bottom: 1px solid var(--border); }
  .fstage:last-child { border-bottom: none; }
  .fstage .fname { font-size: 13.5px; color: var(--fg); }
  .fstage .fname small { display: block; color: var(--fg-dim); font-size: 11px; margin-top: 1px; }
  .fstage .fval { font-weight: 800; font-size: 17px; min-width: 104px; text-align: right; }
  @media (max-width: 620px) { .fstage { grid-template-columns: 1fr; gap: 4px; } .fstage .fval { text-align: left; } }

  /* ── KPI cards ── */
  .kpis { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
  .kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
         padding: 18px 20px; box-shadow: var(--shadow); position: relative; overflow: hidden; }
  .kpi::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--accent, var(--blue)); }
  .kpi .knum { font-size: 2.5rem; font-weight: 800; line-height: 1; letter-spacing: -1.5px; }
  .kpi .kcap { color: var(--fg); font-size: 13px; font-weight: 600; margin-top: 10px; }
  .kpi .ksub { color: var(--fg-dim); font-size: 11.5px; margin-top: 3px; line-height: 1.4; }

  /* ── readers (per-tool) ── */
  .tools { display: grid; gap: 9px; }
  .tool { display: grid; grid-template-columns: 150px 1fr 130px; align-items: center; gap: 14px; }
  .tool .tname { font-size: 13.5px; font-weight: 600; display: flex; align-items: center; gap: 7px; }
  .tbar { height: 26px; background: var(--panel-soft); border-radius: 7px; overflow: hidden; }
  .tbar > div { height: 100%; border-radius: 7px; transition: width .6s cubic-bezier(.16,1,.3,1); }
  .tool .tstate { font-size: 11.5px; font-weight: 600; text-align: right; }
  .tool .tstate small { display:block; color: var(--fg-dim); font-weight: 500; }
  @media (max-width: 620px) { .tool { grid-template-columns: 1fr auto; } .tool .tbar { display:none; } }
  .badge-silent { margin-top: 16px; font-size: 12.5px; padding: 11px 14px; border-radius: 10px;
                  background: rgba(251,113,133,.08); border: 1px solid rgba(251,113,133,.25); color: #ffb3c0; line-height: 1.5; }
  .badge-silent.ok { background: rgba(46,230,166,.07); border-color: rgba(46,230,166,.22); color: #9af0d0; }

  /* ── token savings detail ── */
  .tok-top { display: grid; grid-template-columns: 220px 1fr; gap: 26px; align-items: center; }
  @media (max-width: 620px) { .tok-top { grid-template-columns: 1fr; gap: 16px; } }
  .tok-total .tnum { font-size: 3rem; font-weight: 800; letter-spacing: -1.5px; line-height: 1; color: var(--green); }
  .tok-total .tcap { color: var(--fg); font-size: 13px; font-weight: 600; margin-top: 8px; }
  .tok-total .tassump { color: var(--fg-dim); font-size: 11px; margin-top: 6px; line-height: 1.5; }
  .tok-compbar { height: 30px; background: var(--panel-soft); border-radius: 9px; overflow: hidden; display: flex; }
  .tok-compbar > div { height: 100%; transition: width .6s cubic-bezier(.16,1,.3,1); }
  #tok-seg-grounded { background: var(--green); }
  #tok-seg-reask { background: var(--blue); }
  .tok-legend { display: flex; gap: 22px; margin-top: 12px; flex-wrap: wrap; font-size: 13px; color: var(--fg-mute); }
  .tok-legend b { color: var(--fg); }
  .tok-legend small { color: var(--fg-dim); }
  .tok-legend .sw { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: middle; }
  .tok-subh { margin: 22px 0 6px; font-size: 12px; font-weight: 600; color: var(--fg-mute); }

  /* ── knowledge gaps ── */
  .gap-row { display: grid; grid-template-columns: 44px 1fr auto; align-items: center;
             gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--border); }
  .gap-row:last-child { border-bottom: none; }
  .gap-n { font-weight: 800; color: var(--yellow); font-size: 14px; text-align: right; }
  .gap-q { font-size: 13.5px; color: var(--fg); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .gap-why { font-size: 11.5px; color: var(--fg-dim); white-space: nowrap; }
  @media (max-width: 620px) { .gap-row { grid-template-columns: 36px 1fr; } .gap-why { display: none; } }

  footer { padding: 18px 28px 28px; color: var(--fg-dim); font-size: 11.5px;
           text-align: center; max-width: 1180px; margin: 0 auto; }
  code { background: var(--panel-soft); padding: 1px 6px; border-radius: 5px; font-size: 11px; }
  .native-chart { display: block; width: 100%; height: 100%; overflow: visible; }
</style>
</head>
<body>
<header>
  <h1>memo · <span class="sub">¿está funcionando como memory?</span></h1>
  <span class="meta" id="meta-stamp"></span>
  <span class="live-badge" id="live-badge"></span>
</header>

<main>
  <!-- VEREDICTO -->
  <section class="panel verdict" id="verdict-panel">
    <div class="glyph" id="verdict-glyph">⏳</div>
    <div class="vmain">
      <div class="vlabel" id="verdict-label">—</div>
      <div class="vexplain" id="verdict-explain">—</div>
    </div>
    <div class="vstat">
      <b id="verdict-consults">—</b>
      <span>consultas analizadas</span>
    </div>
  </section>

  <!-- EMBUDO -->
  <section class="panel">
    <h2>De cada pregunta, ¿cuánto aporta memo?</h2>
    <p class="hint">Por cada cosa que se le pregunta al asistente, memo decide si activarse, busca, y entrega información. Esto muestra cuántas llegan a cada paso.</p>
    <div class="funnel" id="funnel"></div>
  </section>

  <!-- KPIs -->
  <section class="kpis" id="kpis"></section>

  <!-- AHORRO DE TOKENS (detalle) -->
  <section class="panel">
    <h2>Ahorro de tokens — detalle</h2>
    <p class="hint">Tokens que el modelo NO tuvo que gastar porque la respuesta usó información que memo ya tenía, en vez de re-generarla o repreguntar. Estimación con supuestos explícitos.</p>
    <div class="tok-top">
      <div class="tok-total">
        <div class="tnum" id="tok-total">—</div>
        <div class="tcap">tokens ahorrados (acumulado)</div>
        <div class="tassump" id="tok-assump">—</div>
      </div>
      <div class="tok-comp">
        <div class="tok-compbar">
          <div id="tok-seg-grounded" title="hechos reutilizados"></div>
          <div id="tok-seg-reask" title="repreguntas evitadas"></div>
        </div>
        <div class="tok-legend">
          <span><span class="sw" style="background:var(--green)"></span><b id="tok-leg-grounded">—</b> hechos reutilizados <small id="tok-leg-grounded-n"></small></span>
          <span><span class="sw" style="background:var(--blue)"></span><b id="tok-leg-reask">—</b> repreguntas evitadas <small id="tok-leg-reask-n"></small></span>
        </div>
      </div>
    </div>
    <h3 class="tok-subh">Por día — tokens ahorrados por hechos reutilizados (14 días)</h3>
    <div id="token-trend" style="height: 220px;"></div>
  </section>

  <!-- QUIÉN USA MEMO -->
  <section class="panel">
    <h2>¿Quién usa memo? · muestra reciente</h2>
    <p class="hint">memo es la memory compartida de todas las herramientas (Claude Code, Codex, Devin…). Esta tabla muestra la ventana reciente que usa el dashboard para medir actividad; no es un total histórico.</p>
    <div class="tools" id="tools"></div>
    <div class="badge-silent" id="silent-callout"></div>
  </section>

  <!-- VACÍOS DE CONOCIMIENTO -->
  <section class="panel">
    <h2>¿Qué le falta saber a memo?</h2>
    <p class="hint">Preguntas de conocimiento que memo no pudo responder (no encontró nada, o lo que mostró no se usó). Capturar estos temas hace que memo deje de fallar ahí.</p>
    <div id="gaps-body"></div>
  </section>

  <!-- TENDENCIA -->
  <section class="panel">
    <h2>Uso en el tiempo · últimos 14 días</h2>
    <p class="hint">¿Se consulta memo cada vez más, o se está quedando en silencio?</p>
    <div id="trend" style="height: 240px;"></div>
  </section>
</main>

<footer>
  memo <span id="memo-version"></span> · <span id="sys-status">—</span> ·
  <span id="sync-status">—</span> ·
  en vivo con <code>memo dashboard</code>
</footer>

<script id="payload" type="application/json" nonce="__CSP_NONCE__">__DATA_JSON__</script>
<script nonce="__CSP_NONCE__">
(() => {
  const PAYLOAD = JSON.parse(document.getElementById("payload").textContent);
  const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
  const asPct = (x, d=0) => (x == null ? "—" : (x * 100).toFixed(d) + "%");
  const shortTs = v => (v ? String(v).slice(0, 16).replace("T", " ") : "—");
  const ago = v => {
    if (!v) return "nunca";
    const ms = Date.now() - new Date(v).getTime();
    if (isNaN(ms)) return "—";
    const m = Math.floor(ms / 60000);
    if (m < 1) return "recién"; if (m < 60) return "hace " + m + "m";
    const h = Math.floor(m / 60); if (h < 24) return "hace " + h + "h";
    return "hace " + Math.floor(h / 24) + "d";
  };

  const SVG_NS = "http://www.w3.org/2000/svg";
  const svgEl = (name, attrs={}, text=null) => {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text != null) node.textContent = String(text);
    return node;
  };
  function renderBarChart(targetId, labels, series, stacked=false) {
    const target = document.getElementById(targetId);
    if (!target) return;
    target.replaceChildren();
    if (!labels.length || !series.length) return;

    const width = 900, height = 230;
    const margin = { top: series.length > 1 ? 30 : 12, right: 12, bottom: 34, left: 48 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const values = [];
    if (stacked) {
      labels.forEach((_, index) => {
        let positive = 0, negative = 0;
        series.forEach(s => {
          const value = Number(s.values[index]) || 0;
          if (value >= 0) positive += value; else negative += value;
        });
        values.push(positive, negative);
      });
    } else {
      series.forEach(s => s.values.forEach(value => values.push(Number(value) || 0)));
    }
    const minValue = Math.min(0, ...values);
    const maxValue = Math.max(0, ...values);
    const span = maxValue - minValue || 1;
    const y = value => margin.top + ((maxValue - value) / span) * innerH;
    const svg = svgEl("svg", {
      class: "native-chart", viewBox: `0 0 ${width} ${height}`,
      role: "img", "aria-label": target.getAttribute("aria-label") || "Bar chart",
    });

    for (let tick = 0; tick <= 4; tick += 1) {
      const value = minValue + span * tick / 4;
      const yy = y(value);
      svg.appendChild(svgEl("line", {
        x1: margin.left, x2: width - margin.right, y1: yy, y2: yy,
        stroke: "#243049", "stroke-width": 1,
      }));
      svg.appendChild(svgEl("text", {
        x: margin.left - 7, y: yy + 4, fill: "#5d6b85", "font-size": 10,
        "text-anchor": "end",
      }, Math.round(value).toLocaleString("es")));
    }

    const slot = innerW / labels.length;
    const groupWidth = Math.max(1, slot * 0.72);
    labels.forEach((label, index) => {
      let positive = 0, negative = 0;
      series.forEach((s, seriesIndex) => {
        const value = Number(s.values[index]) || 0;
        const start = stacked ? (value >= 0 ? positive : negative) : 0;
        const end = start + value;
        if (stacked) {
          if (value >= 0) positive = end; else negative = end;
        }
        const barWidth = stacked ? groupWidth : groupWidth / series.length;
        const x = margin.left + slot * index + (slot - groupWidth) / 2
          + (stacked ? 0 : seriesIndex * barWidth);
        const rect = svgEl("rect", {
          x, y: Math.min(y(start), y(end)), width: Math.max(1, barWidth - 1),
          height: Math.max(1, Math.abs(y(start) - y(end))), fill: s.color, rx: 2,
        });
        rect.appendChild(svgEl("title", {}, `${label} · ${s.name}: ${value.toLocaleString("es")}`));
        svg.appendChild(rect);
      });
      const every = Math.max(1, Math.ceil(labels.length / 10));
      if (index % every === 0 || index === labels.length - 1) {
        svg.appendChild(svgEl("text", {
          x: margin.left + slot * (index + 0.5), y: height - 10, fill: "#5d6b85",
          "font-size": 10, "text-anchor": "middle",
        }, label));
      }
    });

    if (series.length > 1) {
      let legendX = margin.left;
      series.forEach(s => {
        svg.appendChild(svgEl("rect", { x: legendX, y: 7, width: 10, height: 10, rx: 2, fill: s.color }));
        svg.appendChild(svgEl("text", { x: legendX + 15, y: 16, fill: "#93a1b8", "font-size": 11 }, s.name));
        legendX += 26 + String(s.name).length * 7;
      });
    }
    target.appendChild(svg);
  }

  const VERDICT_COPY = {
    ok:     { glyph: "✅", color: "var(--green)",
              explain: "memo se consulta seguido y la información que entrega termina usándose en las respuestas. Está cumpliendo su rol de memory primaria." },
    weak:   { glyph: "⚠️", color: "var(--yellow)",
              explain: "Las herramientas leen memo, pero lo que entrega casi no termina usándose en las respuestas. Funciona como búsqueda, todavía no como memory de verdad." },
    unmeasured: { glyph: "⚠️", color: "var(--yellow)",
              explain: "Las herramientas leen memo, pero falta cobertura de grounding reciente para saber si lo entregado termina usándose. Hay que arreglar la medición antes de juzgar utilidad." },
    unused: { glyph: "❌", color: "var(--red)",
              explain: "Casi no hay consultas registradas. Sin uso no se puede saber si memo ayuda — hay que conectar y ejercitar las herramientas." },
  };

  function render(DATA) {
    const G = DATA.gerencial || {};
    const V = DATA.verdict || {};
    const uf = DATA.usefulness || {};

    document.getElementById("meta-stamp").textContent = "datos al " + shortTs(DATA.generated_at);
    document.getElementById("memo-version").textContent = DATA.memo_version || "?";

    // ── VEREDICTO ──
    const vc = VERDICT_COPY[V.status] || VERDICT_COPY.unused;
    const vPanel = document.getElementById("verdict-panel");
    vPanel.style.borderColor = vc.color;
    vPanel.style.background = "linear-gradient(120deg, color-mix(in srgb, " + vc.color + " 9%, var(--panel)) 0%, var(--panel) 60%)";
    document.getElementById("verdict-glyph").textContent = vc.glyph;
    const vl = document.getElementById("verdict-label");
    vl.textContent = (V.label || "❌ NO SE USA").replace(/^[^ ]+ /, "");
    vl.style.color = vc.color;
    document.getElementById("verdict-explain").textContent = vc.explain;
    const vd = Number(V.consults_sampled ?? V.consults ?? 0);
    const vt = Number(V.consults_total ?? V.consults ?? 0);
    document.getElementById("verdict-consults").textContent =
      `${vd.toLocaleString("es")} / ${vt.toLocaleString("es")}`;
    document.querySelector("#verdict-panel .vstat span").textContent = "muestra / total";

    // ── EMBUDO ──
    const funnel = G.funnel || [];
    const fEl = document.getElementById("funnel");
    fEl.innerHTML = "";
    funnel.forEach((s) => {
      const row = document.createElement("div");
      row.className = "fstage";
      row.innerHTML =
        `<div class="fname">${esc(s.label)}<small>${esc(s.sub || "")}</small></div>
         <div class="fval">${esc((s.value ?? 0).toLocaleString("es"))}</div>`;
      fEl.appendChild(row);
    });

    // ── KPIs ──
    const usedRate = G.used_rate;
    const usedColor = usedRate == null ? "var(--fg-mute)"
      : usedRate >= 0.6 ? "var(--green)" : usedRate >= 0.35 ? "var(--yellow)" : "var(--red)";
    const covColor = (G.coverage_rate ?? 0) >= 0.7 ? "var(--green)" : "var(--yellow)";
    const groundedColor = G.grounded_rate == null ? "var(--fg-mute)"
      : G.grounded_rate >= 0.1 ? "var(--green)" : "var(--yellow)";
    const fmtTok = n => n == null ? "—" : (Math.abs(n) < 1000 ? String(n)
      : Math.abs(n) < 1e6 ? (n/1000).toFixed(1) + "k" : (n/1e6).toFixed(2) + "M");
    const consultsSampled = G.consults_sampled ?? 0;
    const consultsTotal = G.consults_total ?? G.consults ?? 0;
    const kpis = [
      { num: consultsTotal.toLocaleString("es"), accent: "var(--blue)",
        cap: "Consultas totales", sub: "acumulado histórico" },
      { num: consultsSampled.toLocaleString("es"), accent: "var(--yellow)",
        cap: "Consultas analizadas", sub: "muestra reciente para calidad" },
      { num: asPct(G.activation_rate_total ?? G.coverage_rate), accent: covColor,
        cap: "Activación histórica", sub: "memo se abrió sobre el total" },
      { num: asPct(G.hit_rate), accent: "var(--green)", cap: "Hit rate de la muestra",
        sub: "de las veces que memo se activó" },
      { num: asPct(G.grounded_rate), accent: groundedColor, cap: "Hechos reutilizados",
        sub: "de lo que memo mostró" },
      { num: usedRate == null ? "sin datos" : asPct(usedRate), accent: usedColor,
        cap: "Sus datos se usaron en la respuesta",
        sub: G.used_total ? `${G.used_grounded}/${G.used_total} respuestas medidas` : "aún sin medir" },
      { num: asPct(G.measurement_coverage), accent: "var(--yellow)", cap: "Cobertura de medición",
        sub: G.surfaced_turns ? `${G.measured_turns || 0}/${G.surfaced_turns} turnos con grounding` : "sin turnos correlatables" },
      { num: G.tokens_net_today_human || "—", accent: (G.tokens_net_today || 0) >= 0 ? "var(--blue)" : "var(--red)", cap: "Ahorro neto de tokens hoy",
        sub: `${fmtTok(G.tokens_saved_today || 0)} evitados - ${fmtTok(G.context_tokens_today || 0)} inyectados` },
    ];
    const kEl = document.getElementById("kpis");
    kEl.innerHTML = "";
    for (const k of kpis) {
      const d = document.createElement("div");
      d.className = "kpi";
      d.style.setProperty("--accent", k.accent);
      d.innerHTML = `<div class="knum" style="color:${k.accent}">${esc(k.num)}</div>
                     <div class="kcap">${esc(k.cap)}</div><div class="ksub">${esc(k.sub)}</div>`;
      kEl.appendChild(d);
    }

    // ── AHORRO DE TOKENS (detalle) ──
    const td = G.token_detail || {};
    const gTok = td.grounded_tokens || 0, rTok = td.reask_tokens || 0, tTot = td.total || 0;
    // Headline = GROSS saved, which is exactly what the composition bar (green +
    // blue) sums to — so the big number is never smaller than one of its own
    // segments. The net (after subtracting injected-context cost) is stated
    // explicitly in the assumptions line so the bottom line stays honest.
    document.getElementById("tok-total").textContent = fmtTok(tTot);
    document.getElementById("tok-assump").textContent =
      `${td.tok_grounded || 0} tok/hecho · ${td.tok_reask || 0} tok/repregunta`
      + ` · -${fmtTok(td.context_tokens || 0)} contexto → neto ${fmtTok(td.net || 0)}`
      + (td.avg_answer_tokens ? ` · ~${td.avg_answer_tokens} tok/respuesta medido` : "");
    const segG = tTot > 0 ? (gTok / tTot * 100) : 0;
    document.getElementById("tok-seg-grounded").style.width = segG.toFixed(1) + "%";
    document.getElementById("tok-seg-reask").style.width = (100 - segG).toFixed(1) + "%";
    document.getElementById("tok-leg-grounded").textContent = fmtTok(gTok);
    document.getElementById("tok-leg-reask").textContent = fmtTok(rTok);
    document.getElementById("tok-leg-grounded-n").textContent = `(${td.grounded || 0} hechos)`;
    document.getElementById("tok-leg-reask-n").textContent = `(${td.reask_avoided || 0} evitadas)`;
    const tdaily = td.daily || [];
    renderBarChart(
      "token-trend",
      tdaily.map(d => d.date.slice(5)),
      [{ name: "tokens netos", values: tdaily.map(d => d.net_tokens), color: "#2ee6a6" }],
    );

    // ── QUIÉN USA MEMO ──
    const consumers = (uf.consumers || []).slice().sort((a,b) => b.consults - a.consults);
    const maxC = consumers.reduce((m,c) => Math.max(m, c.consults || 0), 1);
    const tEl = document.getElementById("tools");
    tEl.innerHTML = "";
    for (const c of consumers) {
      const helping = c.grounded_rate != null && c.grounded_rate >= 0.10;
      let color, state, sub;
      if (c.consults < 5) { color = "var(--yellow)"; state = "uso esporádico"; }
      else if (helping)   { color = "var(--green)";  state = "uso activo"; }
      else                { color = "var(--green)";  state = "uso activo"; }
      sub = (c.grounded_rate != null ? "usa " + asPct(c.grounded_rate) : "hit " + asPct(c.hit_rate)) + " · " + ago(c.last_seen);
      const pct = Math.max(c.consults / maxC * 100, 3);
      const row = document.createElement("div");
      row.className = "tool";
      row.innerHTML =
        `<div class="tname"><span class="dot" style="background:${color};color:${color}"></span>${esc(c.consumer)}</div>
         <div class="tbar"><div style="width:${pct}%;background:${color}"></div></div>
         <div class="tstate" style="color:${color}">${esc(c.consults.toLocaleString("es"))} consultas<small>${esc(sub)}</small></div>`;
      tEl.appendChild(row);
    }
    const silent = uf.silent || [];
    const sc = document.getElementById("silent-callout");
    if (silent.length) {
      sc.className = "badge-silent";
      sc.innerHTML = "<b>No están usando memo:</b> " + silent.map(esc).join(", ") +
        " — estas herramientas están conectadas pero no consultan la memory.";
    } else {
      sc.className = "badge-silent ok";
      sc.innerHTML = "";
    }

    // ── VACÍOS DE CONOCIMIENTO ──
    const gaps = DATA.gaps || [];
    const gapsBody = document.getElementById("gaps-body");
    if (gapsBody) {
      if (!gaps.length) {
        gapsBody.innerHTML = `<div class="badge-silent ok">✅ Sin vacíos detectados — memo respondió lo que se le preguntó.</div>`;
      } else {
        gapsBody.innerHTML = `<div class="tools">` + gaps.map(g => {
          const reason = (g.reasons || []).join(", ");
          const n = g.count || 1;
          return `<div class="gap-row">
            <span class="gap-n">${esc(n)}&times;</span>
            <span class="gap-q">${esc((g.prompt || "").slice(0, 90))}</span>
            <span class="gap-why">${esc(reason)}</span>
          </div>`;
        }).join("") + `</div>`;
      }
    }

    // ── TENDENCIA ──
    const tr = G.trend || [];
    renderBarChart("trend", tr.map(d => d.date.slice(5)), [
      { name: "consultadas", values: tr.map(d => d.activado), color: "#2ee6a6" },
      { name: "omitidas", values: tr.map(d => Math.max(0, (d.consultas||0) - (d.activado||0))), color: "#3a4a68" },
    ], true);

    // ── system status (footer) ──
    const pillars = DATA.pillars || [];
    const bad = pillars.filter(p => p.status === "red");
    const warn = pillars.filter(p => p.status === "yellow");
    const ss = document.getElementById("sys-status");
    if (bad.length)      { ss.textContent = "⚠ " + bad.length + " problema(s) de sistema"; ss.style.color = "var(--red)"; }
    else if (warn.length){ ss.textContent = "sistema con avisos"; ss.style.color = "var(--yellow)"; }
    else                 { ss.textContent = "sistema sano"; ss.style.color = "var(--green)"; }

    // ── GitHub sync chip ──
    const sync = DATA.sync || {};
    const syncEl = document.getElementById("sync-status");
    if (syncEl) {
      const sc = { ok: "var(--green)", warn: "var(--yellow)", bad: "var(--red)", off: "var(--fg-dim)" };
      syncEl.textContent = "GitHub: " + (sync.label || "—");
      syncEl.style.color = sc[sync.state] || "var(--fg-dim)";
    }
  }

  render(PAYLOAD);

  // ── live auto-refresh (http only; file:// is a static snapshot) ──
  const badge = document.getElementById("live-badge");
  const setBadge = (state, txt, live=false) =>
    badge.innerHTML = `<span class="dot ${state}${live ? " live" : ""}"></span>${txt}`;
  if (location.protocol.startsWith("http")) {
    const intervalS = PAYLOAD.refresh_interval_s || 5;
    setBadge("green", "en vivo", true);
    const poll = async () => {
      try {
        const r = await fetch("/api/data.json", { cache: "no-store" });
        if (!r.ok) throw new Error("HTTP " + r.status);
        render(await r.json());
        setBadge("green", "en vivo · " + new Date().toLocaleTimeString("es"), true);
      } catch (e) {
        setBadge("yellow", "reconectando…");
      }
    };
    setInterval(poll, intervalS * 1000);
    poll();
  } else {
    setBadge("blue", "snapshot · usá `memo dashboard` para tiempo real");
  }
})();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--output", type=Path, default=None, help="Output HTML path (default: web/health.html)"
    )
    ap.add_argument(
        "--limit", type=int, default=1500, help="Max memories to project (default: 1500)"
    )
    ap.add_argument(
        "--open", action="store_true", help="Open the generated HTML in the default browser"
    )
    args = ap.parse_args()
    out = build(output=args.output, limit=args.limit, open_browser=args.open)
    print(f"✓ Wrote {out}")


if __name__ == "__main__":
    main()
