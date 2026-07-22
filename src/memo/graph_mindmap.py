"""Render a graph neighborhood as a self-contained interactive HTML mindmap.

Viz output only — no MCP tool, no cognition verb (respects the
no-cognition-on-MCP invariant). Consumes existing `GraphNavigator` exports;
adds no graph computation.

The generated HTML mirrors memo's Health-Dashboard security pattern
(`web_build.py` + `html_security.py`): inlined CSS/JS, CSP-nonce, no external
`<script src>`/CDN. The renderer is a small memo-owned vanilla-SVG tree drawer
(pan/zoom/fold) — no vendored third-party JS — so the offline invariant holds
by construction under `default-src 'none'; script-src 'nonce-...'`.
"""

from __future__ import annotations

import html
from collections import deque
from typing import Any

from memo.html_security import content_security_policy, html_safe_json, new_csp_nonce

MindmapNode = dict[str, Any]

DEFAULT_NODE_CAP = 300


def build_mindmap_tree(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    center: str,
    depth: int,
    node_cap: int = DEFAULT_NODE_CAP,
) -> MindmapNode:
    """Convert a graph subgraph into a nested markmap ``{content, children}`` tree.

    Breadth-first from ``center`` out to ``depth`` hops, each node placed under
    its first-visited parent (so cycles never recurse and each entity appears
    once). Total node count is capped at ``node_cap`` to keep the HTML small.
    """
    label = {n["id"]: n.get("label", n["id"]) for n in nodes}
    adj: dict[str, list[str]] = {}
    for edge in edges:
        src, dst = edge["source"], edge["target"]
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, []).append(src)
    # Deterministic neighbor order → stable output for golden/diff tests.
    for name in adj:
        adj[name] = sorted(set(adj[name]))

    root: MindmapNode = {"content": label.get(center, center), "children": []}
    node_of: dict[str, MindmapNode] = {center: root}
    used = {center}
    frontier: deque[str] = deque([center])

    for _ in range(max(depth, 0)):
        nxt: deque[str] = deque()
        for name in frontier:
            for neighbor in adj.get(name, []):
                if neighbor in used or len(used) >= node_cap:
                    continue
                used.add(neighbor)
                child: MindmapNode = {"content": label.get(neighbor, neighbor), "children": []}
                node_of[name]["children"].append(child)
                node_of[neighbor] = child
                nxt.append(neighbor)
            if len(used) >= node_cap:
                break
        if len(used) >= node_cap:
            break
        frontier = nxt

    return root


def mindmap_filename(center: str) -> str:
    """Filesystem-safe basename for a default output path (no slashes/spaces)."""
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in center).strip("-")
    return f"memo-mindmap-{safe or 'graph'}.html"


def render_mindmap_html(tree: MindmapNode, title: str = "memo graph") -> str:
    """Render a complete, self-contained, offline-clean interactive HTML document."""
    nonce = new_csp_nonce()
    csp = content_security_policy(nonce, allow_local_fetch=False)
    data = html_safe_json(tree, ensure_ascii=False)
    return (
        _TEMPLATE.replace("__CSP_POLICY__", csp)
        .replace("__TITLE__", html.escape(title))
        .replace("__CSP_NONCE__", nonce)
        .replace("__DATA_JSON__", data)
        .replace("__MINDMAP_JS__", _MINDMAP_JS)
    )


# ── inlined renderer (memo-owned vanilla SVG; no third-party JS) ────────────

_MINDMAP_JS = r"""
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var COL = 210, ROW = 26;
  var svg = document.getElementById("mm");
  var gRoot = document.createElementNS(NS, "g");
  svg.appendChild(gRoot);

  var root = JSON.parse(document.getElementById("payload").textContent);
  (function prep(n) {
    n._collapsed = false;
    (n.children || []).forEach(prep);
  })(root);

  var tx = 0, ty = 0, scale = 1, fitted = false;
  function applyTransform() {
    gRoot.setAttribute("transform", "translate(" + tx + "," + ty + ") scale(" + scale + ")");
  }

  function layout() {
    var placed = [], links = [], y = 0;
    (function walk(n, depth) {
      var x = depth * COL, myY;
      var kids = n._collapsed ? [] : (n.children || []);
      if (kids.length === 0) {
        myY = y * ROW; y += 1;
      } else {
        var cys = kids.map(function (k) { return walk(k, depth + 1); });
        myY = (cys[0] + cys[cys.length - 1]) / 2;
        kids.forEach(function (k, i) { links.push([x, myY, x + COL, cys[i]]); });
      }
      placed.push({ n: n, x: x, y: myY });
      return myY;
    })(root, 0);
    return { placed: placed, links: links };
  }

  function render() {
    while (gRoot.firstChild) gRoot.removeChild(gRoot.firstChild);
    var lay = layout();
    lay.links.forEach(function (l) {
      var mx = (l[0] + l[2]) / 2;
      var p = document.createElementNS(NS, "path");
      p.setAttribute("class", "edge");
      p.setAttribute("d", "M" + l[0] + "," + l[1] + " C" + mx + "," + l[1] +
        " " + mx + "," + l[3] + " " + l[2] + "," + l[3]);
      gRoot.appendChild(p);
    });
    lay.placed.forEach(function (item) {
      var g = document.createElementNS(NS, "g");
      g.setAttribute("transform", "translate(" + item.x + "," + item.y + ")");
      var hasKids = (item.n.children || []).length > 0;
      if (hasKids) {
        var c = document.createElementNS(NS, "circle");
        c.setAttribute("class", "knob" + (item.n._collapsed ? " collapsed" : ""));
        c.setAttribute("cx", "-7"); c.setAttribute("cy", "0"); c.setAttribute("r", "4");
        c.addEventListener("mousedown", function (e) { e.stopPropagation(); });
        c.addEventListener("click", function (e) {
          e.stopPropagation();
          item.n._collapsed = !item.n._collapsed;
          render();
        });
        g.appendChild(c);
      }
      var t = document.createElementNS(NS, "text");
      t.setAttribute("class", "label"); t.setAttribute("x", "2"); t.setAttribute("y", "4");
      t.textContent = item.n.content;
      g.appendChild(t);
      gRoot.appendChild(g);
    });
    if (!fitted) { autofit(); fitted = true; }
  }

  function autofit() {
    var b = gRoot.getBBox();
    var W = svg.clientWidth || 800, H = svg.clientHeight || 600, pad = 48;
    if (b.width === 0 || b.height === 0) { applyTransform(); return; }
    scale = Math.min((W - pad) / b.width, (H - pad) / b.height, 1.4);
    tx = pad / 2 - b.x * scale;
    ty = H / 2 - (b.y + b.height / 2) * scale;
    applyTransform();
  }

  svg.addEventListener("wheel", function (e) {
    e.preventDefault();
    var r = svg.getBoundingClientRect();
    var mx = e.clientX - r.left, my = e.clientY - r.top;
    var f = e.deltaY < 0 ? 1.1 : 0.9;
    tx = mx - (mx - tx) * f; ty = my - (my - ty) * f; scale *= f;
    applyTransform();
  }, { passive: false });

  var dragging = false, lx = 0, ly = 0;
  svg.addEventListener("mousedown", function (e) { dragging = true; lx = e.clientX; ly = e.clientY; });
  window.addEventListener("mousemove", function (e) {
    if (!dragging) return;
    tx += e.clientX - lx; ty += e.clientY - ly; lx = e.clientX; ly = e.clientY;
    applyTransform();
  });
  window.addEventListener("mouseup", function () { dragging = false; });
  window.addEventListener("resize", function () { fitted = false; render(); });

  render();
})();
"""

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<meta name="referrer" content="no-referrer" />
<meta http-equiv="Content-Security-Policy" content="__CSP_POLICY__" />
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0a0e16; --panel: #131a28; --border: #243049;
    --fg: #eef3fb; --fg-mute: #93a1b8; --edge: #3a4a6b; --knob: #5b9dff;
  }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg);
    font: 14px/1.4 system-ui, -apple-system, sans-serif; overflow: hidden; }
  header { position: fixed; top: 0; left: 0; right: 0; padding: 8px 14px;
    background: linear-gradient(var(--panel), rgba(19,26,40,0)); pointer-events: none;
    display: flex; gap: 12px; align-items: baseline; z-index: 2; }
  header h1 { margin: 0; font-size: 14px; font-weight: 600; }
  header .hint { color: var(--fg-mute); font-size: 12px; }
  #mm { width: 100vw; height: 100vh; display: block; cursor: grab; }
  #mm:active { cursor: grabbing; }
  .edge { fill: none; stroke: var(--edge); stroke-width: 1.4; }
  .label { fill: var(--fg); font-size: 13px; }
  .knob { fill: var(--bg); stroke: var(--knob); stroke-width: 1.6; cursor: pointer; }
  .knob.collapsed { fill: var(--knob); }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="hint">drag to pan · scroll to zoom · click a node's dot to fold</span>
</header>
<svg id="mm"></svg>
<script id="payload" type="application/json" nonce="__CSP_NONCE__">__DATA_JSON__</script>
<script nonce="__CSP_NONCE__">__MINDMAP_JS__</script>
</body>
</html>"""
