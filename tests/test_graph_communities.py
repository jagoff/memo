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
