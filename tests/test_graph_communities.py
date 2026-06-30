from __future__ import annotations

from memo.graph_communities import label_propagation


def test_two_dense_clusters_get_two_labels() -> None:
    # {a,b,c} dense; {x,y,z} dense; one thin bridge c-x.
    adj = {
        "a": {"b": 5.0, "c": 5.0},
        "b": {"a": 5.0, "c": 5.0},
        "c": {"a": 5.0, "b": 5.0, "x": 1.0},
        "x": {"y": 5.0, "z": 5.0, "c": 1.0},
        "y": {"x": 5.0, "z": 5.0},
        "z": {"x": 5.0, "y": 5.0},
    }
    labels = label_propagation(adj)
    groups: dict[int, set[str]] = {}
    for node, lb in labels.items():
        groups.setdefault(lb, set()).add(node)
    sizes = sorted(len(s) for s in groups.values())
    assert sizes == [3, 3]  # two clusters, the thin bridge did not fuse them


def test_deterministic() -> None:
    adj = {"a": {"b": 1.0}, "b": {"a": 1.0}}
    assert label_propagation(adj) == label_propagation(adj)


def test_degree_normalization_breaks_hub_fusion() -> None:
    from memo.graph_communities import degree_normalized

    # Two tight triangles + a strong hub `h` wired to all six nodes.
    adj: dict[str, dict[str, float]] = {}

    def link(u: str, v: str, w: float) -> None:
        adj.setdefault(u, {})[v] = w
        adj.setdefault(v, {})[u] = w

    for tri in (("a", "b", "c"), ("x", "y", "z")):
        for i in range(len(tri)):
            for j in range(i + 1, len(tri)):
                link(tri[i], tri[j], 5.0)
    for n in ("a", "b", "c", "x", "y", "z"):
        link("h", n, 10.0)

    # Raw: the strong hub fuses everything into one label.
    raw = label_propagation(adj)
    assert len(set(raw.values())) <= 2

    # Degree-normalized: the two triangles stay distinct communities.
    norm = label_propagation(degree_normalized(adj))
    groups: dict[int, set[str]] = {}
    for node, lb in norm.items():
        groups.setdefault(lb, set()).add(node)
    cores = [g & {"a", "b", "c", "x", "y", "z"} for g in groups.values()]
    assert {"a", "b", "c"} in cores
    assert {"x", "y", "z"} in cores
