"""`memo mapa` — interactive 2D semantic map of the corpus.

Extracted from cli.py (3a/2b god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(mapa_cmd)`. Self-contained: reads
embeddings straight from sqlite-vec (no MLX load) and renders a Plotly HTML.
"""

from __future__ import annotations

import click

from memo.cli_common import console
from memo.config import Config


@click.command(name="mapa")
@click.option(
    "--output", "-o", default=None,
    help="Output HTML path. Default: ~/.local/share/memo/mapa.html",
)
@click.option(
    "--open/--no-open", "open_browser", default=True,
    help="Open in default browser after generating.",
)
@click.option(
    "--limit", default=500, show_default=True,
    help="Maximum number of memories to include.",
)
@click.option(
    "--animate/--no-animate", default=True,
    help="Include timeline animation slider.",
)
def mapa_cmd(output: str | None, open_browser: bool, limit: int, animate: bool) -> None:
    """Generate an interactive 2D semantic map of the memory corpus.

    Projects all memory embeddings (stored in memvec.db) to 2D space using
    UMAP when available, falling back to PCA via numpy. Renders a
    self-contained HTML file with Plotly — hover for preview, click to copy ID.

    Requirements:
      Mandatory: numpy (already a transitive dep via mlx/scipy)
      Optional:  umap-learn (pip install umap-learn) for better topology.
                 Without it the map uses PCA (fast but loses cluster structure).
    """
    import json as _json
    import sqlite3 as _sqlite3
    import struct
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
        import sqlite_vec as _sv  # type: ignore[import-untyped]

        conn = _sqlite3.connect(str(db_path), timeout=10.0)
        conn.enable_load_extension(True)
        _sv.load(conn)
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
            "ORDER BY meta.updated DESC "
            f"LIMIT {int(limit)}"
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
        n = len(blob) // 4
        vec = list(struct.unpack(f"<{n}f", blob))
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
        console.print("[red]numpy is required for mapa.[/red] Install: pip install numpy")
        raise SystemExit(1) from exc

    mat = np.array(raw_vecs, dtype=np.float32)

    xs: list[float]
    ys: list[float]
    method_name: str

    try:
        import umap  # type: ignore[import-not-found]

        n_neighbors = min(15, n_pts - 1)
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors,
            min_dist=0.1, metric="cosine", random_state=42,
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
            frames_data.append({
                "name": d,
                "x": [xs[i] for i, m in enumerate(mask) if m],
                "y": [ys[i] for i, m in enumerate(mask) if m],
                "ids": [ids[i][:8] for i, m in enumerate(mask) if m],
                "titles": [titles[i] for i, m in enumerate(mask) if m],
                "types": [types[i] for i, m in enumerate(mask) if m],
                "tags": [tags_list[i] for i, m in enumerate(mask) if m],
            })
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
    out_path = _Path(output) if output else cfg.state_dir / "mapa.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Encode data as JSON to embed in the HTML script block
    data_json = _json.dumps({
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
    }, ensure_ascii=False)

    html = _MAPA_HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    out_path.write_text(html, encoding="utf-8")

    console.print(f"[green]✓[/green] Mapa saved → [bold]{out_path}[/bold]")
    console.print(
        f"[dim]{n_pts} memories · {method_name}[/dim]"
        + (" [dim]· animation enabled[/dim]" if animate and frames_data else "")
    )

    if open_browser:
        _wb.open(out_path.as_uri())


_MAPA_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>El Mapa — memo</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f172a; color: #e2e8f0; font-family: ui-monospace, 'Cascadia Code', monospace; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  #header { padding: 12px 20px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #1e293b; flex-shrink: 0; }
  #header h1 { font-size: 1rem; font-weight: 600; color: #f1f5f9; letter-spacing: 0.05em; }
  #header .meta { font-size: 0.75rem; color: #64748b; }
  #search-box { margin-left: auto; background: #1e293b; border: 1px solid #334155; color: #e2e8f0; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; outline: none; width: 220px; }
  #search-box:focus { border-color: #4f8ef7; }
  #plot { flex: 1; width: 100%; }
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
  <h1>El Mapa</h1>
  <span class="meta" id="meta-label"></span>
  <input id="search-box" type="search" placeholder="Filtrar memorias…" />
</div>
<div id="plot"></div>
<div id="sidebar">
  <span id="sidebar-close" onclick="closeSidebar()">✕</span>
  <h2>Memoria</h2>
  <div id="sidebar-title"></div>
  <div id="sidebar-meta"></div>
  <div id="sidebar-id" onclick="copyId()" title="Click para copiar ID"></div>
  <div id="sidebar-tags"></div>
</div>
<div id="legend"><h3>Tipos</h3><div id="legend-items"></div></div>
<div id="toast" id="toast">ID copiado</div>

<script>
const DATA = __DATA_JSON__;
let currentId = null;

// Build legend
const li = document.getElementById('legend-items');
Object.entries(DATA.type_colors).forEach(([type, color]) => {
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = `<div class="legend-dot" style="background:${color}"></div><span>${type}</span>`;
  li.appendChild(item);
});
document.getElementById('meta-label').textContent =
  `${DATA.n} memorias · ${DATA.method}`;

// Point colours from type
const colors = DATA.types.map(t => DATA.type_colors[t] || '#94a3b8');

const hovertext = DATA.ids.map((id, i) =>
  `<b>${DATA.titles[i]}</b><br><span style="color:#94a3b8">${DATA.types[i]}</span>`
  + (DATA.tags[i] ? `<br><span style="color:#64748b">${DATA.tags[i]}</span>` : '')
  + `<br><span style="color:#475569">${DATA.created[i]}</span>`
);

const trace = {
  x: DATA.xs, y: DATA.ys,
  mode: 'markers',
  type: 'scatter',
  marker: {
    size: 8, color: colors, opacity: 0.85,
    line: { width: 0.5, color: '#1e293b' }
  },
  text: hovertext,
  hovertemplate: '%{text}<extra></extra>',
  customdata: DATA.ids,
};

const layout = {
  paper_bgcolor: '#0f172a',
  plot_bgcolor: '#0f172a',
  xaxis: { visible: false, zeroline: false },
  yaxis: { visible: false, zeroline: false },
  margin: { t: 10, l: 10, r: 10, b: 10 },
  hovermode: 'closest',
  hoverlabel: {
    bgcolor: '#1e293b', bordercolor: '#334155',
    font: { family: 'ui-monospace', size: 12, color: '#e2e8f0' }
  },
};

let frames = [];
let sliders = [];
if (DATA.frames && DATA.frames.length > 1) {
  frames = DATA.frames.map(f => ({
    name: f.name,
    data: [{
      x: f.x, y: f.y,
      text: f.ids.map((id, i) =>
        `<b>${f.titles[i]}</b><br><span style="color:#94a3b8">${f.types[i]}</span>`
      ),
      marker: { color: f.types.map(t => DATA.type_colors[t] || '#94a3b8') },
      customdata: f.ids,
    }]
  }));
  sliders = [{
    active: frames.length - 1,
    steps: DATA.frames.map((f, i) => ({
      label: f.name, method: 'animate',
      args: [[f.name], { mode: 'immediate', frame: { duration: 0 }, transition: { duration: 0 } }],
    })),
    x: 0.05, y: 0, xanchor: 'left', yanchor: 'top',
    len: 0.9,
    bgcolor: '#1e293b', bordercolor: '#334155',
    font: { color: '#64748b', size: 10 },
    currentvalue: { prefix: 'Hasta: ', font: { color: '#94a3b8', size: 11 }, xanchor: 'center' },
  }];
  layout.sliders = sliders;
}

Plotly.newPlot('plot', [trace], layout, { responsive: true, displayModeBar: false })
  .then(gd => {
    if (frames.length > 1) Plotly.addFrames(gd, frames);
  });

document.getElementById('plot').on('plotly_click', function(data) {
  const pt = data.points[0];
  const idx = pt.pointIndex;
  currentId = DATA.ids[idx];
  document.getElementById('sidebar-title').textContent = DATA.titles[idx];
  document.getElementById('sidebar-meta').textContent =
    `${DATA.types[idx]} · creado ${DATA.created[idx]} · actualizado ${DATA.updated[idx]}`;
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
});

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
}

function copyId() {
  if (!currentId) return;
  navigator.clipboard.writeText(currentId).then(() => {
    const t = document.getElementById('toast');
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 1800);
  });
}

// Search filter
document.getElementById('search-box').addEventListener('input', function() {
  const q = this.value.toLowerCase().trim();
  if (!q) {
    Plotly.restyle('plot', { 'marker.opacity': [0.85] });
    return;
  }
  const opacities = DATA.titles.map((title, i) =>
    title.toLowerCase().includes(q) || DATA.tags[i].toLowerCase().includes(q) ||
    DATA.types[i].toLowerCase().includes(q) ? 0.9 : 0.08
  );
  Plotly.restyle('plot', { 'marker.opacity': [opacities] });
});
</script>
</body>
</html>"""
