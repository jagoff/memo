"""Generate the memo Health Dashboard — a self-contained interactive HTML.

Reads memo state via the same Python APIs used by ``memo doctor`` / ``memo
profile status`` / ``memo mapa``: no extra services, no shell-outs. Emits a
single HTML file at ``web/health.html`` (or ``--output``) with:

  - 4 colour-coded pillar lights (red / yellow / green / blue)
  - 3-D scatter of all memory embeddings (PCA — UMAP optional)
  - Corpus growth bar chart
  - Type distribution donut
  - Recent saves / recalls table

Run:
    python web/build.py [--output PATH] [--limit N] [--open]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import struct
import sys
import webbrowser
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from memo.cli_diag import (  # noqa: E402
    _db_health_report,
    _profile_status_report,
)
from memo.cli_runtime import _runtime_install_report  # noqa: E402
from memo.config import Config  # noqa: E402
from memo.dashboard import (  # noqa: E402
    consult_breakdown,
    read_recall_log,
    reask_stats,
    recall_health,
    verdict,
)
from memo.dashboard_panels import _fetch_memflow_utility  # noqa: E402

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


def _read_vectors(db_path: Path, limit: int) -> list[dict[str, Any]]:
    """Pull embeddings + metadata straight from sqlite-vec (no MLX load)."""
    try:
        import sqlite_vec
    except ImportError:
        return []
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT vec.id, vec.embedding, meta.title, meta.type, "
            "       meta.tags, meta.created, meta.updated "
            "FROM vec JOIN meta ON meta.id = vec.id "
            "ORDER BY meta.updated DESC "
            f"LIMIT {int(limit)}"
        ).fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for row in rows:
        blob = row["embedding"]
        if blob is None:
            continue
        n = len(blob) // 4
        vec = list(struct.unpack(f"<{n}f", blob))
        if not vec:
            continue
        try:
            tags = json.loads(row["tags"] or "[]")
        except json.JSONDecodeError:
            tags = []
        out.append({
            "id": row["id"],
            "vec": vec,
            "title": row["title"] or "—",
            "type": row["type"] or "note",
            "tags": [str(t) for t in tags][:6],
            "created": (row["created"] or "")[:10],
            "updated": (row["updated"] or "")[:10],
        })
    return out


def _project_3d(vecs: list[list[float]]) -> tuple[list[float], list[float], list[float], str]:
    """3-D projection. Tries UMAP, falls back to PCA via numpy."""
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "numpy required for build (pip install numpy)"
        ) from exc
    mat = np.array(vecs, dtype=np.float32)
    try:
        import umap  # type: ignore[import-not-found]

        n_neighbors = min(15, len(vecs) - 1)
        reducer = umap.UMAP(
            n_components=3, n_neighbors=n_neighbors,
            min_dist=0.12, metric="cosine", random_state=42,
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
    """Count memorias whose .md body diverges from store body_hash.

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
        except Exception:
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
            "SELECT ts, op, record_id, title, type FROM events "
            "ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
    except Exception:
        return []
    return [
        {"ts": r[0], "op": r[1], "id": r[2][:8], "title": r[3], "type": r[4]}
        for r in rows
    ]


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
        rows = conn.execute(
            "SELECT status, count(*) FROM pairs GROUP BY status"
        ).fetchall()
        conn.close()
    except Exception:
        return None
    return {str(r[0]): int(r[1]) for r in rows}


# ── pillars ──────────────────────────────────────────────────────────────
#
# Each pillar reports {label, status, summary, detail}. Status is one of
# "green" / "yellow" / "red" / "blue". Blue = informational (no health
# signal); the others are the traffic-light contract from the spec.

def _pillar_vector_db(doctor: dict[str, Any], drift: dict[str, int]) -> dict[str, Any]:
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
    if drift["drifted"] or drift["untracked_md"]:
        return {
            "label": "Vector DB",
            "status": "yellow",
            "summary": (
                f"{drift['drifted']} drifted · {drift['untracked_md']} untracked"
            ),
            "detail": [
                f"records:        {memvec.get('records')}",
                f"checked:        {drift['checked']}",
                f"body_hash drift: {drift['drifted']}",
                f"missing files:   {drift['missing_file']}",
                f"untracked .md:   {drift['untracked_md']}",
                "Run `memo reindex` to re-embed only changed entries.",
            ],
        }
    return {
        "label": "Vector DB",
        "status": "green",
        "summary": f"{memvec.get('records')} memorias · {memvec.get('vec_dims')}D",
        "detail": [
            f"path: {memvec['path']}",
            f"size: {memvec.get('size_bytes', 0):,} bytes",
            f"integrity: {memvec.get('integrity_check', '?')}",
            f"latest update: {memvec.get('latest_memory_update', '—')}",
        ],
    }


def _pillar_embedder(doctor: dict[str, Any]) -> dict[str, Any]:
    profile = doctor["profile"]
    mlx_ok = next(
        (i for i in doctor["imports"] if i["label"] == "mlx"), {"ok": False}
    )["ok"]
    if not mlx_ok:
        return {
            "label": "Embedder (MLX)",
            "status": "red",
            "summary": "mlx / mlx_lm not importable",
            "detail": ["Install with `pip install mlx mlx-lm` on Apple Silicon."],
        }
    embedder_model = profile["active"].get("embedder_model", "?")
    cached = any(
        m["cached"] and m["role"] == "embedder" for m in profile["models"]
    )
    if not cached:
        return {
            "label": "Embedder (MLX)",
            "status": "yellow",
            "summary": "model weights not in HF cache",
            "detail": [
                f"model: {embedder_model}",
                "Will download on first use. Run `memo prewarm` to fetch now.",
            ],
        }
    if not profile["ok"]:
        return {
            "label": "Embedder (MLX)",
            "status": "red",
            "summary": profile["status"],
            "detail": [f"model: {embedder_model}", "See `memo profile status`."],
        }
    return {
        "label": "Embedder (MLX)",
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
) -> dict[str, Any]:
    memvec = next((d for d in doctor["db"] if d.get("label") == "memvec"), {})
    records = memvec.get("records") or 0
    type_counts = Counter(r["type"] for r in rows)
    last_save = next((h for h in history if h["op"] == "save"), None)
    detail = [
        f"records: {records}",
        f"with vectors: {len(rows)}",
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
        "summary": f"{records:,} memorias",
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


def _render_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return _HTML_TEMPLATE.replace("__DATA_JSON__", payload)


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


def collect_data(cfg: Config, *, include_projection: bool = True, limit: int = 1500) -> dict[str, Any]:
    """Gather every metric the dashboard renders, as a JSON-serializable dict.

    ``include_projection`` gates the only expensive step (reading all vectors +
    PCA): the static build wants it, but the live-refresh endpoint skips it so a
    poll stays cheap.
    """
    doctor = {
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
        "imports": _imports_probe(),
        "db": _db_health_report(cfg),
    }

    drift = _body_hash_drift(cfg)
    recall_log = read_recall_log(cfg.state_dir, limit=200)
    recall_health_data = recall_health(cfg.state_dir, limit=500)
    history = _history_recent(cfg, limit=50)
    contradictions = _contradictions_stats(cfg)

    # Heavy step (read every vector + PCA) — only on a full build, not per poll.
    if include_projection:
        rows = _read_vectors(cfg.db_path, limit=limit)
        if len(rows) >= 3:
            xs, ys, zs, method = _project_3d([r["vec"] for r in rows])
        else:
            xs, ys, zs, method = [], [], [], "n/a (need ≥ 3 memorias with vectors)"
        projection = {
            "method": method,
            "xs": xs, "ys": ys, "zs": zs,
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

    pillars = [
        _pillar_vector_db(doctor, drift),
        _pillar_embedder(doctor),
        _pillar_recall(recall_log),
        _pillar_corpus(doctor, rows, history, contradictions),
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
            "referenced_rate": recall_health_data["referenced_rate"],
        },
        "bail_breakdown": recall_health_data["bail_breakdown"],
        "usefulness": _usefulness(cfg),
        "doctor_raw": doctor,
        "contradictions": contradictions or {},
        "verdict": verdict(cfg.state_dir, limit=500),
        "memflow_util": _fetch_memflow_utility(),
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
    out_path = output if output else REPO_ROOT / "web" / "health.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_html(data), encoding="utf-8")
    if open_browser:
        webbrowser.open(out_path.as_uri())
    return out_path


def _imports_probe() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, fn in (
        ("sqlite_vec", _probe_sqlite_vec),
        ("mlx", _probe_mlx),
    ):
        try:
            fn()
            out.append({"label": label, "ok": True, "error": ""})
        except Exception as exc:
            out.append({"label": label, "ok": False, "error": str(exc)})
    return out


def _probe_sqlite_vec() -> None:
    import sqlite_vec
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    finally:
        conn.close()


def _probe_mlx() -> None:
    import mlx.core  # noqa: F401
    import mlx_lm  # noqa: F401


# ── HTML / JS template ───────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Memo · Health Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131a26;
    --panel-soft: #1a2231;
    --border: #243049;
    --fg: #e6edf7;
    --fg-mute: #94a3b8;
    --green: #34d399;
    --yellow: #fbbf24;
    --red:    #f87171;
    --blue:   #4f8ef7;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
               font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                            "Helvetica Neue", Arial, sans-serif; }
  header { padding: 18px 24px; border-bottom: 1px solid var(--border);
           display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: 0.5px; }
  header .meta { color: var(--fg-mute); font-size: 12px; }
  header .overall { margin-left: auto; display: flex; gap: 6px; align-items: center; }
  .dot { width: 10px; height: 10px; border-radius: 50%;
         box-shadow: 0 0 8px currentColor; display: inline-block; }
  .dot.green  { background: var(--green); color: var(--green); }
  .dot.yellow { background: var(--yellow); color: var(--yellow); }
  .dot.red    { background: var(--red); color: var(--red); }
  .dot.blue   { background: var(--blue); color: var(--blue); }

  main { padding: 18px 24px; display: grid; gap: 18px; }
  .pillars { display: grid; gap: 14px;
             grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
  .card { background: var(--panel); border: 1px solid var(--border);
          border-radius: 10px; padding: 14px 16px; }
  .card.green  { border-left: 4px solid var(--green); }
  .card.yellow { border-left: 4px solid var(--yellow); }
  .card.red    { border-left: 4px solid var(--red); }
  .card.blue   { border-left: 4px solid var(--blue); }
  .card .label { font-size: 12px; color: var(--fg-mute); text-transform: uppercase;
                 letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px; }
  .card .summary { font-size: 20px; font-weight: 600; margin-top: 6px; }
  .card details { margin-top: 10px; }
  .card details summary { cursor: pointer; color: var(--fg-mute); font-size: 12px;
                          list-style: none; }
  .card details summary::-webkit-details-marker { display: none; }
  .card details summary::before { content: "▸ "; }
  .card details[open] summary::before { content: "▾ "; }
  .card details pre { background: var(--panel-soft); border-radius: 6px;
                      padding: 10px 12px; font-size: 12px; line-height: 1.5;
                      white-space: pre-wrap; word-break: break-word;
                      margin: 8px 0 0; color: var(--fg); }

  .grid { display: grid; gap: 18px;
          grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); }
  @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
  .panel { background: var(--panel); border: 1px solid var(--border);
           border-radius: 10px; padding: 14px 16px; }
  .panel h2 { margin: 0 0 10px; font-size: 13px; font-weight: 600;
              color: var(--fg-mute); text-transform: uppercase;
              letter-spacing: 0.5px; }
  .row2 { display: grid; gap: 18px; grid-template-columns: 1fr 1fr; }
  @media (max-width: 800px) { .row2 { grid-template-columns: 1fr; } }

  .statcards { display: grid; gap: 12px;
               grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
  .statcard { background: var(--panel); border: 1px solid var(--border);
              border-radius: 10px; padding: 14px 16px; }
  .statcard .num { font-size: 1.9rem; font-weight: 800; line-height: 1.1;
                   letter-spacing: -1px; }
  .statcard .cap { color: var(--fg-mute); font-size: 0.74rem; margin-top: 4px;
                   text-transform: uppercase; letter-spacing: 0.5px; }
  .live-badge { font-size: 11px; color: var(--fg-mute); display: inline-flex;
                align-items: center; gap: 5px; }
  .live-badge .dot { width: 7px; height: 7px; }

  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--fg-mute); font-weight: 500; }
  tr:last-child td { border-bottom: none; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 999px;
          font-size: 11px; background: var(--panel-soft); color: var(--fg-mute); }
  .op-save   { color: var(--green); }
  .op-update { color: var(--yellow); }
  .op-delete { color: var(--red); }

  footer { padding: 12px 24px; color: var(--fg-mute); font-size: 11px;
           text-align: center; border-top: 1px solid var(--border); }
  code { background: var(--panel-soft); padding: 1px 5px; border-radius: 4px;
         font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>memo · health</h1>
  <span class="meta" id="meta-stamp"></span>
  <span class="live-badge" id="live-badge"></span>
  <span class="overall" id="overall-lights"></span>
</header>

<main>
  <section class="pillars" id="pillars"></section>

  <div class="row2">
    <div class="panel" id="verdict-panel" style="border-width:2px;">
      <h2>¿Funciona memo?</h2>
      <div id="verdict-label" style="font-size:1.8rem;font-weight:800;line-height:1.2;">—</div>
      <div id="verdict-sub" style="color:var(--fg-mute);font-size:0.85rem;margin:0.3rem 0 0.8rem;">—</div>
      <div style="font-size:0.82rem;line-height:1.8;">
        <div><span style="color:var(--fg-mute);">lee:&nbsp;</span><span id="verdict-readers" style="color:var(--green);">—</span></div>
        <div><span style="color:var(--fg-mute);">NO lee:&nbsp;</span><span id="verdict-silent" style="color:var(--red);font-weight:600;">—</span></div>
      </div>
    </div>
    <div class="panel" id="memflow-panel">
      <h2>memflow utility</h2>
      <div id="memflow-estado" style="font-size:1.1rem;font-weight:700;margin-bottom:0.6rem;">—</div>
      <table><tbody>
        <tr><td style="color:var(--fg-mute);">reads 7d</td><td id="mf-reads" style="text-align:right;font-weight:600;">—</td></tr>
        <tr><td style="color:var(--fg-mute);">memory_used</td><td id="mf-used" style="text-align:right;font-weight:600;">—</td></tr>
        <tr><td style="color:var(--fg-mute);">re_explain</td><td id="mf-reexplain" style="text-align:right;font-weight:600;">—</td></tr>
      </tbody></table>
    </div>
  </div>

  <section class="statcards" id="statcards"></section>

  <div class="panel" id="utility-panel" style="border-width:2px;">
    <h2>Utilidad — respuestas CON memo vs SIN memo</h2>
    <div style="display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap;">
      <div id="utility-pct" style="font-size:3.2rem;font-weight:800;line-height:1;letter-spacing:-1px;">—</div>
      <div id="utility-sub" style="color:var(--fg-mute);font-size:0.9rem;">—</div>
    </div>
    <div style="background:var(--border);border-radius:6px;height:16px;overflow:hidden;display:flex;margin-top:0.7rem;">
      <div id="utility-seg-con" style="height:100%;background:var(--green);width:0%;transition:width .4s;" title="respondió con memo"></div>
      <div id="utility-seg-sin" style="height:100%;background:#f8717155;width:0%;transition:width .4s;" title="respondió sin memo"></div>
    </div>
    <div style="font-size:0.78rem;color:var(--fg-mute);margin-top:0.45rem;">
      <span id="utility-leg-con" style="color:var(--green);font-weight:600;">—</span> con memo ·
      <span id="utility-leg-sin" style="font-weight:600;">—</span> sin memo
    </div>
  </div>

  <div class="panel">
    <h2>Quién lee memo</h2>
    <table id="readers-table">
      <thead><tr><th>capa</th><th>consultas</th><th>hit%</th><th>grounded%</th><th>última</th></tr></thead>
      <tbody></tbody>
    </table>
    <div id="readers-silent" style="margin-top:0.6rem;font-size:0.8rem;color:var(--fg-mute);"></div>
  </div>

  <div class="grid">
    <div class="panel">
      <h2>3-D memory map <span id="map-method" class="pill"></span></h2>
      <div id="map3d" style="height: 520px;"></div>
    </div>
    <div class="panel">
      <h2>Type distribution</h2>
      <div id="types" style="height: 240px;"></div>
      <h2 style="margin-top:14px">Saves · last 30 days</h2>
      <div id="growth" style="height: 220px;"></div>
    </div>
  </div>

  <div class="row2" style="grid-template-columns:1fr;">
    <div class="panel">
      <h2>Utilización de consultas</h2>
      <div style="display:flex;align-items:center;gap:2.5rem;padding:1.2rem 0 0.6rem;">
        <div id="util-pct" style="font-size:6rem;font-weight:800;line-height:1;color:var(--fg);min-width:8rem;letter-spacing:-2px;">—</div>
        <div style="flex:1;">
          <div id="util-sub" style="margin-bottom:1rem;color:var(--fg-mute);font-size:0.9rem;line-height:1.6;">—</div>
          <div style="background:var(--border);border-radius:6px;height:18px;overflow:hidden;display:flex;">
            <div id="util-seg-hits"  style="height:100%;background:var(--green);width:0%;transition:width 0.4s;" title="con hits"></div>
            <div id="util-seg-miss"  style="height:100%;background:#f8717166;width:0%;transition:width 0.4s;" title="sin hits"></div>
            <div id="util-seg-bail"  style="height:100%;background:#94a3b833;width:0%;transition:width 0.4s;" title="omitidas"></div>
          </div>
          <div style="display:flex;gap:1.2rem;margin-top:0.5rem;font-size:0.78rem;color:var(--fg-mute);">
            <span><span style="display:inline-block;width:10px;height:10px;background:var(--green);border-radius:2px;margin-right:4px;vertical-align:middle;"></span><span id="util-leg-hits">—</span> con hits</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:#f8717166;border-radius:2px;margin-right:4px;vertical-align:middle;"></span><span id="util-leg-miss">—</span> sin hits</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:#94a3b833;border-radius:2px;margin-right:4px;vertical-align:middle;"></span><span id="util-leg-bail">—</span> omitidas</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="row2">
    <div class="panel">
      <h2>Recent history</h2>
      <table id="history-table">
        <thead><tr><th>when</th><th>op</th><th>id</th><th>title</th><th>type</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="panel">
      <h2>Recent recalls</h2>
      <table id="recall-table">
        <thead><tr><th>when</th><th>prompt</th><th>hits</th><th>ms</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</main>

<footer>
  Generated by <code>python web/build.py</code> ·
  v<span id="memo-version"></span> ·
  refresh by re-running the script.
</footer>

<script id="payload" type="application/json">__DATA_JSON__</script>
<script>
(() => {
  const PAYLOAD = JSON.parse(document.getElementById("payload").textContent);

  function escapeHTML(s) {
    return String(s).replace(/[&<>"]/g, c => ({
      "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"
    })[c]);
  }
  const asPct = x => (x == null ? "—" : Math.round(x * 100) + "%");
  const shortTs = v => (v ? String(v).slice(0, 16).replace("T", " ") : "—");
  function dbRecords(DATA) {
    const db = (DATA.doctor_raw && DATA.doctor_raw.db) || [];
    const mv = db.find(d => d.label === "memvec");
    return mv && typeof mv.records === "number" ? mv.records : null;
  }

  // Single idempotent render — safe to call repeatedly with fresh data.
  function render(DATA) {
    const PAL = DATA.type_palette || {};
    document.getElementById("meta-stamp").textContent = "generated " + DATA.generated_at;
    document.getElementById("memo-version").textContent = DATA.memo_version || "?";

    // -- pillars + overall lights ----------------------------------------
    const order = ["red", "yellow", "green", "blue"];
    const pillars = (DATA.pillars || []).slice()
      .sort((a,b) => order.indexOf(a.status) - order.indexOf(b.status));
    const pillarsEl = document.getElementById("pillars");
    pillarsEl.innerHTML = "";
    for (const p of pillars) {
      const card = document.createElement("div");
      card.className = "card " + p.status;
      card.innerHTML = `
        <div class="label"><span class="dot ${p.status}"></span>${p.label}</div>
        <div class="summary">${p.summary}</div>
        <details><summary>details</summary><pre></pre></details>
      `;
      card.querySelector("pre").textContent = (p.detail || []).join("\n");
      pillarsEl.appendChild(card);
    }
    const lights = document.getElementById("overall-lights");
    lights.innerHTML = "";
    for (const p of pillars) {
      const d = document.createElement("span");
      d.className = "dot " + p.status;
      d.title = p.label + " — " + p.summary;
      lights.appendChild(d);
    }

    // -- stat cards (simple at-a-glance metrics) -------------------------
    const ru = DATA.recall_util || {};
    const uf = DATA.usefulness || {};
    const contra = DATA.contradictions || {};
    const cards = [
      { num: dbRecords(DATA) ?? "—", cap: "memorias" },
      { num: asPct(ru.hit_rate), cap: "recall hit" },
      { num: asPct(ru.grounded_rate), cap: "grounded" },
      { num: asPct(ru.referenced_rate), cap: "referenced" },
      { num: uf.reask_avoided ?? "—", cap: "repreguntas evitadas" },
      { num: (uf.consumers || []).length, cap: "lectores activos" },
      { num: contra.open ?? 0, cap: "contradicciones" },
      { num: uf.sampled ?? "—", cap: "consultas (muestra)" },
    ];
    const scEl = document.getElementById("statcards");
    scEl.innerHTML = "";
    for (const c of cards) {
      const d = document.createElement("div");
      d.className = "statcard";
      d.innerHTML = `<div class="num">${c.num}</div><div class="cap">${c.cap}</div>`;
      scEl.appendChild(d);
    }

    // -- utility: answers WITH memo vs WITHOUT (grounded outcome) --------
    const grounded = ru.grounded || 0;
    const groundedSurfaced = ru.grounded_surfaced || 0;
    const withoutMemo = Math.max(0, groundedSurfaced - grounded);
    const utilRate = groundedSurfaced > 0 ? grounded / groundedSurfaced : null;
    const utilColor = utilRate == null ? "var(--fg-mute)"
      : utilRate >= 0.5 ? "var(--green)" : utilRate >= 0.2 ? "var(--yellow)" : "var(--red)";
    const utilPanel = document.getElementById("utility-panel");
    if (utilPanel) utilPanel.style.borderColor = utilColor;
    const utilPct = document.getElementById("utility-pct");
    utilPct.textContent = utilRate == null ? "sin datos" : asPct(utilRate);
    utilPct.style.color = utilColor;
    document.getElementById("utility-sub").textContent = utilRate == null
      ? "todavía no hay respuestas medidas"
      : "de las respuestas que usaron una memoria surgida, esta fracción la aprovechó";
    if (groundedSurfaced > 0) {
      document.getElementById("utility-seg-con").style.width = (grounded / groundedSurfaced * 100).toFixed(1) + "%";
      document.getElementById("utility-seg-sin").style.width = (withoutMemo / groundedSurfaced * 100).toFixed(1) + "%";
    }
    document.getElementById("utility-leg-con").textContent = grounded;
    document.getElementById("utility-leg-sin").textContent = withoutMemo;

    // -- who reads memo (per-consumer) -----------------------------------
    const rdBody = document.querySelector("#readers-table tbody");
    rdBody.innerHTML = "";
    for (const c of (uf.consumers || [])) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${escapeHTML(c.consumer)}</td>
                      <td>${c.consults}</td>
                      <td>${asPct(c.hit_rate)}</td>
                      <td>${asPct(c.grounded_rate)}</td>
                      <td>${shortTs(c.last_seen)}</td>`;
      rdBody.appendChild(tr);
    }
    const silentLayers = uf.silent || [];
    document.getElementById("readers-silent").textContent = silentLayers.length
      ? "NO leen memo: " + silentLayers.join(", ")
      : "todas las capas esperadas leen memo ✅";

    // -- 3-D scatter (only present on a full build / first load) ----------
    if (DATA.projection) {
      const proj = DATA.projection;
      document.getElementById("map-method").textContent = proj.method;
      const colors = proj.types.map(t => PAL[t] || "#94a3b8");
      const hover = proj.ids.map((id, i) =>
        `${proj.titles[i]}<br><b>${proj.types[i]}</b> · ${id}<br>${proj.tags[i] || ""}<br>${proj.created[i]}`
      );
      Plotly.react("map3d", [{
        type: "scatter3d", mode: "markers",
        x: proj.xs, y: proj.ys, z: proj.zs,
        marker: { size: 4, color: colors, opacity: 0.85, line: { width: 0 } },
        text: hover, hoverinfo: "text",
      }], {
        paper_bgcolor: "transparent", plot_bgcolor: "transparent",
        margin: { l: 0, r: 0, t: 0, b: 0 },
        font: { color: "#e6edf7", size: 11 },
        scene: {
          xaxis: { showgrid: true, gridcolor: "#243049", zeroline: false, title: "" },
          yaxis: { showgrid: true, gridcolor: "#243049", zeroline: false, title: "" },
          zaxis: { showgrid: true, gridcolor: "#243049", zeroline: false, title: "" },
          bgcolor: "transparent",
        },
      }, { displaylogo: false, responsive: true });
    }

    // -- type donut (only when type_counts present) ----------------------
    const tc = DATA.type_counts || {};
    if (Object.keys(tc).length) {
      Plotly.react("types", [{
        type: "pie", hole: 0.55,
        labels: Object.keys(tc),
        values: Object.values(tc),
        marker: { colors: Object.keys(tc).map(k => PAL[k] || "#94a3b8") },
        textinfo: "label+percent", textfont: { color: "#0b0f17" },
      }], {
        paper_bgcolor: "transparent", plot_bgcolor: "transparent",
        font: { color: "#e6edf7", size: 11 }, showlegend: false,
        margin: { l: 0, r: 0, t: 0, b: 0 },
      }, { displaylogo: false, responsive: true });
    }

    // -- growth bar ------------------------------------------------------
    if (DATA.growth) {
      Plotly.react("growth", [{
        type: "bar",
        x: DATA.growth.map(g => g.date),
        y: DATA.growth.map(g => g.count),
        marker: { color: "#4f8ef7" },
      }], {
        paper_bgcolor: "transparent", plot_bgcolor: "transparent",
        font: { color: "#e6edf7", size: 11 },
        margin: { l: 30, r: 10, t: 4, b: 30 },
        xaxis: { tickangle: -45, automargin: true, color: "#94a3b8" },
        yaxis: { gridcolor: "#243049", color: "#94a3b8" },
      }, { displaylogo: false, responsive: true });
    }

    // -- recall utilization ----------------------------------------------
    if (ru && ru.fired > 0) {
      const hits   = Math.round((ru.hit_rate || 0) * ru.fired);
      const misses = ru.fired - hits;
      const total  = ru.fired + ru.bailed;
      const p      = Math.round((ru.hit_rate || 0) * 100);
      const color  = p >= 70 ? 'var(--green)' : p >= 40 ? 'var(--yellow)' : 'var(--red)';
      document.getElementById('util-pct').style.color = color;
      document.getElementById('util-pct').textContent = p + '%';
      document.getElementById('util-seg-hits').style.width = (hits   / total * 100).toFixed(1) + '%';
      document.getElementById('util-seg-miss').style.width = (misses / total * 100).toFixed(1) + '%';
      document.getElementById('util-seg-bail').style.width = (ru.bailed / total * 100).toFixed(1) + '%';
      document.getElementById('util-leg-hits').textContent = hits;
      document.getElementById('util-leg-miss').textContent = misses;
      document.getElementById('util-leg-bail').textContent = ru.bailed;
      const strong = ru.strong_hit_rate != null ? Math.round(ru.strong_hit_rate * 100) + '% strong' : '';
      const score  = ru.median_top_score != null ? 'score mediano ' + ru.median_top_score : '';
      document.getElementById('util-sub').textContent = [strong, score, total + ' totales'].filter(Boolean).join(' · ');
    } else {
      document.getElementById('util-pct').textContent = 'sin datos';
      document.getElementById('util-sub').textContent = ru && ru.bailed > 0
        ? ru.bailed + ' omitidas (prompts cortos / slash commands)'
        : 'recall.log vacío';
    }

    // -- verdict (¿funciona memo?) ---------------------------------------
    const V = DATA.verdict || {};
    const vColors = { ok: 'var(--green)', weak: 'var(--yellow)', unused: 'var(--red)' };
    const vColor = vColors[V.status] || 'var(--red)';
    const vPanel = document.getElementById('verdict-panel');
    if (vPanel) vPanel.style.borderColor = vColor;
    document.getElementById('verdict-label').textContent = V.label || '❌ NO SE USA';
    document.getElementById('verdict-label').style.color = vColor;
    const gr = V.grounded_rate == null ? '—' : Math.round(V.grounded_rate * 100) + '%';
    document.getElementById('verdict-sub').textContent = `grounded ${gr} · ${V.consults || 0} consultas`;
    const per = V.per_consumer || [];
    const readers = per.filter(x => x.reads).map(x => x.name + ' ✅');
    const silent  = per.filter(x => !x.reads).map(x => x.name + ' ❌');
    document.getElementById('verdict-readers').textContent = readers.length ? readers.join('  ') : '—';
    document.getElementById('verdict-silent').textContent  = silent.length  ? silent.join('  ')  : 'ninguno';

    // -- memflow utility -------------------------------------------------
    const MF = DATA.memflow_util || {};
    const cons = MF.consumption || {}, outc = MF.outcome || {};
    const reads = cons.total_read_calls || 0;
    const reexp = outc.re_explain || 0;
    const used  = outc.memory_used_rate;
    let mfEstado, mfColor;
    if (!MF.consumption) { mfEstado = 'no disponible'; mfColor = 'var(--fg-mute)'; }
    else if (reads < 5)  { mfEstado = '❌ casi no se lee'; mfColor = 'var(--red)'; }
    else if (used == null) { mfEstado = '⚠️ sin outcomes aún'; mfColor = 'var(--yellow)'; }
    else if (used >= 0.10 && reexp < 10) { mfEstado = '✅ útil'; mfColor = 'var(--green)'; }
    else { mfEstado = '⚠️ poco útil'; mfColor = 'var(--yellow)'; }
    const mfEl = document.getElementById('memflow-estado');
    mfEl.textContent = mfEstado; mfEl.style.color = mfColor;
    const mfp = document.getElementById('memflow-panel');
    if (mfp) mfp.style.borderColor = mfColor;
    document.getElementById('mf-reads').textContent = reads;
    document.getElementById('mf-used').textContent = used == null ? '—' : Math.round(used * 100) + '%';
    const reEl = document.getElementById('mf-reexplain');
    reEl.textContent = reexp; reEl.style.color = reexp >= 10 ? 'var(--red)' : '';

    // -- tables (idempotent) ---------------------------------------------
    const histBody = document.querySelector("#history-table tbody");
    histBody.innerHTML = "";
    for (const h of (DATA.history || [])) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${h.ts}</td><td class="op-${h.op}">${h.op}</td>
                      <td><code>${h.id}</code></td><td>${escapeHTML(h.title || "")}</td>
                      <td><span class="pill">${h.type || ""}</span></td>`;
      histBody.appendChild(tr);
    }
    const recBody = document.querySelector("#recall-table tbody");
    recBody.innerHTML = "";
    for (const r of (DATA.recall_log || [])) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${r.ts || ""}</td>
                      <td>${escapeHTML((r.prompt || "").slice(0, 80))}</td>
                      <td>${(r.hits || []).length}</td>
                      <td>${r.latency_ms ?? ""}</td>`;
      recBody.appendChild(tr);
    }
  }

  render(PAYLOAD);

  // -- live auto-refresh (only when served over http; file:// is static) -
  const badge = document.getElementById("live-badge");
  const setBadge = (state, txt) => {
    badge.innerHTML = `<span class="dot ${state}"></span>${txt}`;
  };
  if (location.protocol.startsWith("http")) {
    const intervalS = PAYLOAD.refresh_interval_s || 5;
    setBadge("green", "en vivo · " + intervalS + "s");
    const poll = async () => {
      try {
        const r = await fetch("/api/data.json", { cache: "no-store" });
        if (!r.ok) throw new Error("HTTP " + r.status);
        render(await r.json());
        setBadge("green", "actualizado " + new Date().toLocaleTimeString());
      } catch (e) {
        setBadge("yellow", "sin conexión · reintentando");
      }
    };
    setInterval(poll, intervalS * 1000);
  } else {
    setBadge("blue", "snapshot estático · usar `memo dashboard` para tiempo real");
  }
})();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None,
                    help="Output HTML path (default: web/health.html)")
    ap.add_argument("--limit", type=int, default=1500,
                    help="Max memorias to project (default: 1500)")
    ap.add_argument("--open", action="store_true",
                    help="Open the generated HTML in the default browser")
    args = ap.parse_args()
    out = build(output=args.output, limit=args.limit, open_browser=args.open)
    print(f"✓ Wrote {out}")


if __name__ == "__main__":
    main()
