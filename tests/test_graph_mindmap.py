from __future__ import annotations

import json

from click.testing import CliRunner

from memo.cli import cli
from memo.graph_mindmap import build_mindmap_tree, render_mindmap_html


def _env(tmp_cfg) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_VAULT_PATH": str(tmp_cfg.vault_path),
        "MEMO_AUTO_PROJECT_TAG": "0",
    }


# ── tree builder ──────────────────────────────────────────────────────────


def test_tree_centers_and_nests_neighbors() -> None:
    nodes = [
        {"id": "memo", "label": "memo"},
        {"id": "sqlite", "label": "sqlite"},
        {"id": "mlx", "label": "mlx"},
    ]
    edges = [
        {"source": "memo", "target": "sqlite"},
        {"source": "memo", "target": "mlx"},
    ]
    tree = build_mindmap_tree(nodes, edges, center="memo", depth=1)
    assert tree["content"] == "memo"
    labels = {c["content"] for c in tree["children"]}
    assert labels == {"sqlite", "mlx"}


def test_tree_respects_node_cap() -> None:
    nodes = [{"id": f"n{i}", "label": f"n{i}"} for i in range(500)]
    edges = [{"source": "n0", "target": f"n{i}"} for i in range(1, 500)]
    tree = build_mindmap_tree(nodes, edges, center="n0", depth=1, node_cap=10)
    assert len(tree["children"]) <= 9  # center + at most node_cap-1 children


def test_tree_depth_two_nests_grandchildren() -> None:
    nodes = [{"id": x, "label": x} for x in ("a", "b", "c")]
    edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
    tree = build_mindmap_tree(nodes, edges, center="a", depth=2)
    b = tree["children"][0]
    assert b["content"] == "b"
    assert b["children"][0]["content"] == "c"


def test_tree_depth_one_does_not_descend() -> None:
    nodes = [{"id": x, "label": x} for x in ("a", "b", "c")]
    edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
    tree = build_mindmap_tree(nodes, edges, center="a", depth=1)
    b = tree["children"][0]
    assert b["children"] == []  # c is 2 hops away, excluded at depth=1


def test_tree_no_cycles_when_graph_has_cycle() -> None:
    nodes = [{"id": x, "label": x} for x in ("a", "b", "c")]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c"},
        {"source": "c", "target": "a"},
    ]
    tree = build_mindmap_tree(nodes, edges, center="a", depth=5)
    # every node visited at most once — no infinite recursion, bounded size
    seen: list[str] = []

    def walk(n: dict) -> None:
        seen.append(n["content"])
        for c in n["children"]:
            walk(c)

    walk(tree)
    assert sorted(seen) == ["a", "b", "c"]


# ── HTML renderer ─────────────────────────────────────────────────────────


def test_html_is_offline_clean_and_embeds_tree() -> None:
    tree = build_mindmap_tree(
        [{"id": "memo", "label": "memo"}, {"id": "x", "label": "x"}],
        [{"source": "memo", "target": "x"}],
        center="memo",
        depth=1,
    )
    html = render_mindmap_html(tree, title="t")
    assert html.lstrip().lower().startswith("<!doctype html>")
    # offline-clean: no external resources
    assert 'src="http' not in html and 'href="http' not in html
    assert "cdn" not in html.lower()
    # tree embedded as JSON payload
    assert '"content"' in html
    assert '"memo"' in html
    # CSP present + nonce'd inline scripts
    assert "content-security-policy" in html.lower()
    assert "nonce=" in html


def test_html_escapes_title_and_payload() -> None:
    tree = build_mindmap_tree(
        [{"id": "</script>", "label": "</script>"}], [], center="</script>", depth=1
    )
    html = render_mindmap_html(tree, title="<b>x</b>")
    # The JSON payload data must not carry a literal </script> that would break
    # out of its <script type="application/json"> node.
    after_open = html.split('type="application/json"', 1)[1]
    data = after_open.split(">", 1)[1].split("</script>", 1)[0]
    assert "</script>" not in data
    assert "\\u003c/script" in data  # the entity name survived, escaped


# ── CLI subcommand ────────────────────────────────────────────────────────


def _stub_mem(nodes, edges):
    class _Nav:
        def export_json(self, include_memories: bool = False):
            return {"nodes": nodes, "edges": edges}

    class _Mem:
        navigator = _Nav()

    return _Mem()


def test_mindmap_cli_writes_html(tmp_path, tmp_cfg, monkeypatch) -> None:
    mem = _stub_mem(
        [{"id": "memo", "label": "memo"}, {"id": "x", "label": "x"}],
        [{"source": "memo", "target": "x"}],
    )
    monkeypatch.setattr("memo.cli_graph._get_memory", lambda _cfg: mem)
    out = tmp_path / "mm.html"
    result = CliRunner().invoke(
        cli,
        ["graph", "mindmap", "memo", "--depth", "1", "--out", str(out), "--no-open"],
        env=_env(tmp_cfg),
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.read_text(encoding="utf-8").lstrip().lower().startswith("<!doctype html>")


def test_mindmap_cli_empty_graph_is_graceful(tmp_path, tmp_cfg, monkeypatch) -> None:
    monkeypatch.setattr("memo.cli_graph._get_memory", lambda _cfg: _stub_mem([], []))
    out = tmp_path / "mm.html"
    result = CliRunner().invoke(
        cli,
        ["graph", "mindmap", "--out", str(out), "--no-open"],
        env=_env(tmp_cfg),
    )
    assert result.exit_code == 0, result.output
    assert not out.exists()
    assert "empty" in result.output.lower()


def test_mindmap_cli_defaults_to_top_degree_entity(tmp_path, tmp_cfg, monkeypatch) -> None:
    # hub has degree 3, others degree 1 → hub is the default center
    nodes = [{"id": x, "label": x} for x in ("hub", "a", "b", "c")]
    edges = [
        {"source": "hub", "target": "a"},
        {"source": "hub", "target": "b"},
        {"source": "hub", "target": "c"},
    ]
    monkeypatch.setattr("memo.cli_graph._get_memory", lambda _cfg: _stub_mem(nodes, edges))
    out = tmp_path / "mm.html"
    result = CliRunner().invoke(
        cli,
        ["graph", "mindmap", "--out", str(out), "--no-open"],
        env=_env(tmp_cfg),
    )
    assert result.exit_code == 0, result.output
    html = out.read_text(encoding="utf-8")
    payload = json.loads(
        html.split('type="application/json"', 1)[1].split(">", 1)[1].split("</script>", 1)[0]
    )
    assert payload["content"] == "hub"
