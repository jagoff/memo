"""`memo map` — interactive 2D semantic map of the corpus.

Extracted from cli.py (3a/2b god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(map_cmd)`. Self-contained: reads
embeddings straight from sqlite-vec (no MLX load) and renders local Canvas HTML.
"""

from __future__ import annotations

import click

from memo.cli_common import console
from memo.config import Config
from memo.html_security import (
    content_security_policy,
    html_safe_json,
    new_csp_nonce,
)


def _decode_embedding(blob: bytes, dims: int) -> list[float]:
    """Decode a raw `vec` blob to floats, dtype-aware by length.

    int8 (MEMO_VEC_QUANTIZE=int8) is 1 B/dim, dequantized (÷127); float32 is
    4 B/dim. Length-based detection keeps this independent of the running
    config, since the blob's dtype is whatever it was indexed with.
    """
    import struct

    if len(blob) == dims:
        return [x / 127.0 for x in struct.unpack(f"{dims}b", blob)]
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _render_map_html(data: dict[str, object], *, nonce: str | None = None) -> str:
    """Render corpus data without allowing it to escape into executable HTML."""
    nonce = nonce or new_csp_nonce()
    return (
        _MAPA_HTML_TEMPLATE.replace("__DATA_JSON__", html_safe_json(data, ensure_ascii=True))
        .replace("__CSP_NONCE__", nonce)
        .replace("__CSP_POLICY__", content_security_policy(nonce, allow_local_fetch=False))
    )


@click.command(name="map")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output HTML path. Default: ~/.local/share/memo/map.html",
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Open in default browser after generating.",
)
@click.option(
    "--limit",
    default=500,
    show_default=True,
    help="Maximum number of memories to include.",
)
@click.option(
    "--animate/--no-animate",
    default=True,
    help="Include timeline animation slider.",
)
def map_cmd(output: str | None, open_browser: bool, limit: int, animate: bool) -> None:
    """Generate an interactive 2D semantic map of the memory corpus.

    Projects all memory embeddings (stored in memvec.db) to 2D space using
    UMAP when available, falling back to PCA via numpy. Renders a
    self-contained HTML file — hover for preview, click to copy ID.

    Requirements:
      Mandatory: numpy (already a transitive dep via mlx/scipy)
      Optional:  umap-learn (pip install umap-learn) for better topology.
                 Without it the map uses PCA (fast but loses cluster structure).
    """
    import json as _json
    import sqlite3 as _sqlite3
    import webbrowser as _wb
    from pathlib import Path as _Path

    cfg = Config.from_env()
    db_path = cfg.state_dir / "memvec.db"
    if not db_path.is_file():
        console.print(f"[red]DB not found:[/red] {db_path}. Run `memo reindex` first.")
        raise SystemExit(1)

    # ── Read embeddings + metadata directly from SQLite ───────────────────
    # We bypass VecStore to avoid loading MLX. sqlite-vec stores
    # FLOAT[N] columns as raw 4-byte little-endian blobs.
    console.print("[dim]Reading corpus from DB…[/dim]")
    try:
        from memo.sqlite_compat import import_sqlite_vec

        conn = _sqlite3.connect(str(db_path), timeout=10.0)
        conn.enable_load_extension(True)
        import_sqlite_vec().load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = _sqlite3.Row
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise SystemExit(1) from exc

    try:
        rows = conn.execute(
            "SELECT vec.id, vec.embedding, "
            "       meta.title, meta.type, meta.tags, "
            "       meta.created, meta.updated "
            "FROM vec JOIN meta ON meta.id = vec.id "
            "ORDER BY meta.updated DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception as exc:
        console.print(f"[red]Query failed:[/red] {exc}")
        raise SystemExit(1) from exc
    finally:
        conn.close()

    if len(rows) < 3:
        console.print(
            f"[yellow]Not enough memories to map ({len(rows)} found, need ≥ 3).[/yellow]\n"
            "Save some memories first with `/memo save` or `memo capture-stop`."
        )
        raise SystemExit(0)

    ids: list[str] = []
    titles: list[str] = []
    types: list[str] = []
    tags_list: list[str] = []
    created_list: list[str] = []
    updated_list: list[str] = []
    raw_vecs: list[list[float]] = []

    for row in rows:
        blob = row["embedding"]
        if blob is None:
            continue
        vec = _decode_embedding(blob, cfg.embedder_dims)
        if not vec:
            continue
        ids.append(row["id"])
        titles.append(row["title"] or "—")
        types.append(row["type"] or "note")
        try:
            tags = _json.loads(row["tags"] or "[]")
            tags_list.append(", ".join(str(t) for t in tags[:4]) if tags else "")
        except Exception:
            tags_list.append("")
        created_list.append((row["created"] or "")[:10])
        updated_list.append((row["updated"] or "")[:10])
        raw_vecs.append(vec)

    n_pts = len(raw_vecs)
    if n_pts < 3:
        console.print(f"[yellow]Only {n_pts} memories have vectors. Run `memo reindex`.[/yellow]")
        raise SystemExit(0)

    # ── 2D projection ─────────────────────────────────────────────────────
    console.print(f"[dim]Projecting {n_pts} memories to 2D…[/dim]")
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        console.print("[red]numpy is required for the map.[/red] Install: pip install numpy")
        raise SystemExit(1) from exc

    mat = np.array(raw_vecs, dtype=np.float32)

    xs: list[float]
    ys: list[float]
    method_name: str

    try:
        import umap  # type: ignore[import-not-found]

        n_neighbors = min(15, n_pts - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(mat)
        xs = coords[:, 0].tolist()
        ys = coords[:, 1].tolist()
        method_name = "UMAP"
    except ImportError:
        # PCA fallback — fast, preserves global variance, loses local clusters.
        mat_centered = mat - mat.mean(axis=0)
        _, _, vt = np.linalg.svd(mat_centered, full_matrices=False)
        coords = mat_centered @ vt[:2].T
        xs = coords[:, 0].tolist()
        ys = coords[:, 1].tolist()
        method_name = "PCA (install umap-learn for better topology)"

    console.print(f"[green]✓[/green] Projected via {method_name}")

    # ── Build timeline frames for animation ───────────────────────────────
    # Sort unique dates; each frame shows memories up to that date.
    if animate:
        dates_sorted = sorted(set(created_list))
        frames_data: list[dict] = []
        for d in dates_sorted:
            mask = [c <= d for c in created_list]
            frames_data.append(
                {
                    "name": d,
                    "x": [xs[i] for i, m in enumerate(mask) if m],
                    "y": [ys[i] for i, m in enumerate(mask) if m],
                    "ids": [ids[i][:8] for i, m in enumerate(mask) if m],
                    "titles": [titles[i] for i, m in enumerate(mask) if m],
                    "types": [types[i] for i, m in enumerate(mask) if m],
                    "tags": [tags_list[i] for i, m in enumerate(mask) if m],
                }
            )
    else:
        frames_data = []

    # ── Type → colour mapping ─────────────────────────────────────────────
    TYPE_COLORS = {
        "decision": "#4f8ef7",
        "fact": "#34d399",
        "bug": "#f87171",
        "preference": "#a78bfa",
        "feedback": "#fb923c",
        "note": "#94a3b8",
        "manual": "#e2e8f0",
    }

    # ── Emit self-contained HTML ──────────────────────────────────────────
    out_path = _Path(output) if output else cfg.state_dir / "map.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = _render_map_html(
        {
            "xs": xs,
            "ys": ys,
            "ids": [i[:8] for i in ids],
            "titles": titles,
            "types": types,
            "tags": tags_list,
            "created": created_list,
            "updated": updated_list,
            "frames": frames_data,
            "method": method_name,
            "n": n_pts,
            "type_colors": TYPE_COLORS,
        }
    )
    out_path.write_text(html, encoding="utf-8")

    console.print(f"[green]✓[/green] Map saved → [bold]{out_path}[/bold]")
    console.print(
        f"[dim]{n_pts} memories · {method_name}[/dim]"
        + (" [dim]· animation enabled[/dim]" if animate and frames_data else "")
    )

    if open_browser:
        _wb.open(out_path.as_uri())


_MAPA_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>The Map — memo</title>
<meta name="referrer" content="no-referrer">
<meta http-equiv="Content-Security-Policy" content="__CSP_POLICY__">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f172a; color: #e2e8f0; font-family: ui-monospace, 'Cascadia Code', monospace; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  #header { padding: 12px 20px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #1e293b; flex-shrink: 0; }
  #header h1 { font-size: 1rem; font-weight: 600; color: #f1f5f9; letter-spacing: 0.05em; }
  #header .meta { font-size: 0.75rem; color: #64748b; }
  #search-box { margin-left: auto; background: #1e293b; border: 1px solid #334155; color: #e2e8f0; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; outline: none; width: 220px; }
  #search-box:focus { border-color: #4f8ef7; }
  #plot { flex: 1; width: 100%; position: relative; min-height: 0; }
  #plot-canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
  #plot-tooltip { position: absolute; display: none; pointer-events: none; z-index: 5;
    max-width: 280px; padding: 7px 9px; border: 1px solid #334155; border-radius: 6px;
    background: #1e293b; color: #e2e8f0; white-space: pre-line; font-size: 0.73rem; }
  #timeline-wrap { position: absolute; left: 50%; bottom: 18px; transform: translateX(-50%);
    display: none; width: min(640px, 72vw); padding: 8px 12px; border: 1px solid #334155;
    border-radius: 8px; background: rgba(30,41,59,.94); color: #94a3b8; font-size: .7rem; }
  #timeline { width: 100%; accent-color: #4f8ef7; }
  #sidebar { position: fixed; right: 0; top: 0; bottom: 0; width: 320px; background: #1e293b; border-left: 1px solid #334155; padding: 20px; transform: translateX(100%); transition: transform 0.2s ease; overflow-y: auto; z-index: 10; }
  #sidebar.open { transform: translateX(0); }
  #sidebar-close { float: right; cursor: pointer; color: #64748b; font-size: 1.2rem; line-height: 1; margin-bottom: 16px; }
  #sidebar-close:hover { color: #e2e8f0; }
  #sidebar h2 { font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
  #sidebar-title { font-size: 1rem; font-weight: 600; color: #f1f5f9; margin-bottom: 6px; line-height: 1.4; }
  #sidebar-meta { font-size: 0.75rem; color: #64748b; margin-bottom: 12px; }
  #sidebar-id { font-size: 0.75rem; background: #0f172a; padding: 6px 10px; border-radius: 4px; color: #a78bfa; cursor: pointer; display: inline-block; margin-bottom: 12px; }
  #sidebar-id:hover { background: #1e293b; }
  #sidebar-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; }
  .tag-chip { font-size: 0.7rem; background: #0f172a; color: #94a3b8; padding: 2px 8px; border-radius: 10px; }
  #legend { position: fixed; bottom: 20px; left: 20px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; font-size: 0.73rem; }
  #legend h3 { color: #64748b; margin-bottom: 6px; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .legend-item { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  #toast { position: fixed; bottom: 20px; right: 20px; background: #1e293b; border: 1px solid #4f8ef7; color: #e2e8f0; padding: 8px 14px; border-radius: 6px; font-size: 0.8rem; opacity: 0; pointer-events: none; transition: opacity 0.3s; z-index: 20; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<div id="header">
  <h1>The Map</h1>
  <span class="meta" id="meta-label"></span>
  <input id="search-box" type="search" placeholder="Filter memories…" />
</div>
<div id="plot">
  <canvas id="plot-canvas"></canvas>
  <div id="plot-tooltip"></div>
  <label id="timeline-wrap"><span id="timeline-label"></span><input id="timeline" type="range" min="0" step="1" /></label>
</div>
<div id="sidebar">
  <span id="sidebar-close">✕</span>
  <h2>Memory</h2>
  <div id="sidebar-title"></div>
  <div id="sidebar-meta"></div>
  <div id="sidebar-id" title="Click to copy ID"></div>
  <div id="sidebar-tags"></div>
</div>
<div id="legend"><h3>Types</h3><div id="legend-items"></div></div>
<div id="toast">ID copied</div>

<script nonce="__CSP_NONCE__">
const DATA = __DATA_JSON__;
let currentId = null;
let screenPoints = [];
let query = '';
let cutoff = null;
const plot = document.getElementById('plot');
const canvas = document.getElementById('plot-canvas');
const tooltip = document.getElementById('plot-tooltip');
const ctx = canvas.getContext('2d');

// Build legend
const li = document.getElementById('legend-items');
Object.entries(DATA.type_colors).forEach(([type, color]) => {
  const item = document.createElement('div');
  item.className = 'legend-item';
  const dot = document.createElement('span');
  dot.className = 'legend-dot';
  dot.style.backgroundColor = color;
  const label = document.createElement('span');
  label.textContent = type;
  item.append(dot, label);
  li.appendChild(item);
});
document.getElementById('meta-label').textContent =
  `${DATA.n} memories · ${DATA.method}`;

const matches = i => !query || [DATA.titles[i], DATA.tags[i], DATA.types[i]]
  .some(value => String(value || '').toLowerCase().includes(query));
const isVisible = i => cutoff === null || String(DATA.created[i] || '') <= cutoff;

function drawPlot() {
  const box = plot.getBoundingClientRect();
  const width = Math.max(1, box.width);
  const height = Math.max(1, box.height);
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const indices = DATA.xs.map((_, i) => i).filter(isVisible);
  if (!indices.length) { screenPoints = []; return; }
  const xs = indices.map(i => Number(DATA.xs[i]) || 0);
  const ys = indices.map(i => Number(DATA.ys[i]) || 0);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = maxX - minX || 1, spanY = maxY - minY || 1;
  const pad = 28;

  screenPoints = indices.map(i => ({
    i,
    x: pad + ((Number(DATA.xs[i]) - minX) / spanX) * Math.max(1, width - pad * 2),
    y: height - pad - ((Number(DATA.ys[i]) - minY) / spanY) * Math.max(1, height - pad * 2),
    active: matches(i),
  }));
  for (const point of screenPoints) {
    ctx.globalAlpha = point.active ? 0.88 : 0.08;
    ctx.fillStyle = DATA.type_colors[DATA.types[point.i]] || '#94a3b8';
    ctx.beginPath();
    ctx.arc(point.x, point.y, point.active ? 4.5 : 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 0.7;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function pointAt(event) {
  const box = canvas.getBoundingClientRect();
  const x = event.clientX - box.left, y = event.clientY - box.top;
  let best = null, bestDistance = 11;
  for (const point of screenPoints) {
    if (!point.active) continue;
    const distance = Math.hypot(point.x - x, point.y - y);
    if (distance < bestDistance) { best = point; bestDistance = distance; }
  }
  return best;
}

function openSidebar(idx) {
  currentId = DATA.ids[idx];
  document.getElementById('sidebar-title').textContent = DATA.titles[idx];
  document.getElementById('sidebar-meta').textContent =
    `${DATA.types[idx]} · created ${DATA.created[idx]} · updated ${DATA.updated[idx]}`;
  document.getElementById('sidebar-id').textContent = `/memo get ${currentId}`;
  const tagsEl = document.getElementById('sidebar-tags');
  tagsEl.innerHTML = '';
  (DATA.tags[idx] || '').split(',').filter(Boolean).forEach(t => {
    const chip = document.createElement('span');
    chip.className = 'tag-chip';
    chip.textContent = t.trim();
    tagsEl.appendChild(chip);
  });
  document.getElementById('sidebar').classList.add('open');
}

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
}

async function copyId() {
  if (!currentId) return;
  let copied = false;
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(currentId);
      copied = true;
    }
  } catch (_) {
    copied = false;
  }
  if (!copied) {
    const fallback = document.createElement('textarea');
    fallback.value = currentId;
    fallback.setAttribute('readonly', '');
    fallback.style.position = 'fixed';
    fallback.style.opacity = '0';
    document.body.appendChild(fallback);
    fallback.select();
    copied = document.execCommand('copy');
    fallback.remove();
  }
  if (copied) {
    const t = document.getElementById('toast');
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 1800);
  }
}

canvas.addEventListener('mousemove', event => {
  const point = pointAt(event);
  if (!point) { tooltip.style.display = 'none'; return; }
  const i = point.i;
  tooltip.textContent = `${DATA.titles[i]}\n${DATA.types[i]}`
    + (DATA.tags[i] ? `\n${DATA.tags[i]}` : '') + `\n${DATA.created[i]}`;
  tooltip.style.display = 'block';
  tooltip.style.left = Math.max(8, Math.min(event.offsetX + 12, plot.clientWidth - 292)) + 'px';
  tooltip.style.top = Math.max(8, event.offsetY - 28) + 'px';
});
canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
canvas.addEventListener('click', event => {
  const point = pointAt(event);
  if (point) openSidebar(point.i);
});
document.getElementById('sidebar-close').addEventListener('click', closeSidebar);
document.getElementById('sidebar-id').addEventListener('click', copyId);

document.getElementById('search-box').addEventListener('input', function() {
  query = this.value.toLowerCase().trim();
  drawPlot();
});

if (DATA.frames && DATA.frames.length > 1) {
  const slider = document.getElementById('timeline');
  const wrap = document.getElementById('timeline-wrap');
  const label = document.getElementById('timeline-label');
  slider.max = String(DATA.frames.length - 1);
  slider.value = slider.max;
  wrap.style.display = 'block';
  const selectFrame = () => {
    cutoff = DATA.frames[Number(slider.value)].name;
    label.textContent = `Up to: ${cutoff}`;
    drawPlot();
  };
  slider.addEventListener('input', selectFrame);
  selectFrame();
} else {
  drawPlot();
}
window.addEventListener('resize', drawPlot);
</script>
</body>
</html>"""
