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


def test_giant_component_side_is_excluded():
    # 'j' joins a BIG component (>max_side) and a small triangle. The giant side
    # is not a meaningful "theme", so with it excluded there is only one bounded
    # side and 'j' is NOT emitted as a bridge (fixes the 'memo and X' degeneration).
    adj: dict[str, dict[str, float]] = {}

    def link(a: str, b: str) -> None:
        adj.setdefault(a, {})[b] = 1.0
        adj.setdefault(b, {})[a] = 1.0

    big = [f"g{i}" for i in range(45)]
    link("j", big[0])
    for i in range(len(big) - 1):
        link(big[i], big[i + 1])  # 45-node chain hanging off j (> max_side=40)
    link("j", "s1")
    link("s1", "s2")
    link("s2", "j")  # small bounded triangle j-s1-s2

    bridges = find_bridges(adj, min_side=2, max_side=40)
    assert all(b["bridge"] != "j" for b in bridges)  # j's giant side excluded


def test_articulation_points_match_known_graph():
    from memo.graph_bridges import _articulation_points, _symmetric_neighbors

    aps = _articulation_points(_symmetric_neighbors(_two_triangles_joined()))
    assert aps == {"j"}  # only the joining node is an articulation point
