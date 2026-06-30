"""Tests for articulation-bridge detection (spec 3, phase 3)."""

from __future__ import annotations

from memo.graph_bridges import find_bridges


def _two_triangles_joined() -> dict[str, dict[str, float]]:
    """Two triangles {j,a1,a2} and {j,b1,b2} sharing only the joining node ``j``."""
    edges = [
        ("j", "a1"),
        ("j", "a2"),
        ("a1", "a2"),
        ("j", "b1"),
        ("j", "b2"),
        ("b1", "b2"),
    ]
    adj: dict[str, dict[str, float]] = {}
    for a, b in edges:
        adj.setdefault(a, {})[b] = 1.0
        adj.setdefault(b, {})[a] = 1.0
    return adj


def test_joining_node_is_the_bridge():
    bridges = find_bridges(_two_triangles_joined())
    assert len(bridges) == 1
    br = bridges[0]
    assert br["bridge"] == "j"
    assert set(br["left"]) == {"a1", "a2"}
    assert set(br["right"]) == {"b1", "b2"}


def test_single_triangle_has_no_bridge():
    adj: dict[str, dict[str, float]] = {}
    for a, b in [("a", "b"), ("b", "c"), ("a", "c")]:
        adj.setdefault(a, {})[b] = 1.0
        adj.setdefault(b, {})[a] = 1.0
    assert find_bridges(adj) == []


def test_min_side_excludes_small_components():
    # Each side has only 2 entities; min_side=3 rejects the bridge.
    assert find_bridges(_two_triangles_joined(), min_side=3) == []


def test_find_bridges_is_deterministic():
    adj = _two_triangles_joined()
    assert find_bridges(adj) == find_bridges(adj)
