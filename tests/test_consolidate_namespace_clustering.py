"""Consolidation must propose only merges the write path will accept.

A merged record carries the UNION of its members' tags, and
`identity.namespace_for_write` refuses a union holding more than one `project:`
slug (`ambiguous_namespace`). Clustering purely by cosine happily mixes
projects, so the proposals were generated and then rejected at apply time —
measured on the live corpus, 14 of 15 clusters died that way.
"""

import pytest

from memo.identity import IdentityConflictError, cluster_scope, namespace_for_write


def _item(mid: str, tags: list[str], emb: list[float] | None = None) -> dict:
    return {
        "id": mid,
        "title": f"title {mid}",
        "type": "note",
        "tags": tags,
        "path": f"{mid}.md",
        "updated": "2026-01-01T00:00:00+00:00",
        "emb": emb or [1.0, 0.0, 0.0, 0.0],
    }


def _project_slugs_of(cluster: dict) -> set[str]:
    slugs = set()
    for member in cluster["members"]:
        for tag in member["tags"]:
            if tag.startswith("project:"):
                slugs.add(tag)
    return slugs


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ([], ""),
        (["note"], ""),
        (["project:memo"], "memo"),
        (["Project:Memo", "note"], "memo"),
        (["project:memo", "project:memflow"], None),
        (["project:"], None),
    ],
)
def test_cluster_scope_agrees_with_the_write_path(tags, expected):
    assert cluster_scope(tags) == expected
    if expected is None:
        with pytest.raises(IdentityConflictError):
            namespace_for_write(tags, auto_project=False)
    else:
        namespace_for_write(tags, auto_project=False)


def test_identical_bodies_in_different_projects_cluster_separately(mock_memory, monkeypatch):
    """Same embedding, different `project:` tag — one cosine cluster today, and
    every merge it proposes is refused. Split it by project instead."""
    mem = mock_memory
    items = [_item(f"memo{i}", ["project:memo"]) for i in range(3)]
    items += [_item(f"flow{i}", ["project:memflow"]) for i in range(2)]

    monkeypatch.setattr(mem, "_pull_embeddings", lambda **kwargs: items)
    monkeypatch.setattr(mem, "_read_body", lambda path: f"body:{path}")

    clusters = mem.consolidate(threshold=0.85, max_clusters=10, skip_llm=True)

    assert [c["size"] for c in clusters] == [3, 2]
    assert [_project_slugs_of(c) for c in clusters] == [{"project:memo"}, {"project:memflow"}]


def test_untagged_memories_never_get_absorbed_into_a_project(mock_memory, monkeypatch):
    """Untagged records are mergeable with anything as far as the write path is
    concerned, but folding them into a project cluster would silently reassign
    them (and, under MEMO_STORE_BY_PROJECT, move their file and change the
    recall project boost). Keep them in their own scope."""
    mem = mock_memory
    items = [_item(f"memo{i}", ["project:memo"]) for i in range(2)]
    items += [_item(f"loose{i}", []) for i in range(2)]

    monkeypatch.setattr(mem, "_pull_embeddings", lambda **kwargs: items)
    monkeypatch.setattr(mem, "_read_body", lambda path: f"body:{path}")

    clusters = mem.consolidate(threshold=0.85, max_clusters=10, skip_llm=True)

    assert len(clusters) == 2
    assert [_project_slugs_of(c) for c in clusters] == [{"project:memo"}, set()]


def test_a_record_that_is_ambiguous_by_itself_is_never_proposed(mock_memory, monkeypatch):
    """Two `project:` slugs on ONE record poison any cluster it joins — the union
    stays ambiguous no matter who it merges with. Leave it out."""
    mem = mock_memory
    items = [_item(f"memo{i}", ["project:memo"]) for i in range(2)]
    items.append(_item("mixed", ["project:memo", "project:memflow"]))

    monkeypatch.setattr(mem, "_pull_embeddings", lambda **kwargs: items)
    monkeypatch.setattr(mem, "_read_body", lambda path: f"body:{path}")

    clusters = mem.consolidate(threshold=0.85, max_clusters=10, skip_llm=True)

    assert len(clusters) == 1
    assert {m["id"] for m in clusters[0]["members"]} == {"memo0", "memo1"}


def test_every_proposed_cluster_survives_namespace_for_write(mock_memory, monkeypatch):
    """The invariant, stated directly: the union of a proposal's tags must be
    writable. This is what `apply_merge` hands to `Memory.save`."""
    mem = mock_memory
    items = [
        _item("a", ["project:memo", "note"]),
        _item("b", ["project:memo", "decision"]),
        _item("c", ["project:memo-spec"]),
        _item("d", ["project:memo-spec", "note"]),
        _item("e", []),
        _item("f", ["note"]),
    ]

    monkeypatch.setattr(mem, "_pull_embeddings", lambda **kwargs: items)
    monkeypatch.setattr(mem, "_read_body", lambda path: f"body:{path}")

    clusters = mem.consolidate(threshold=0.85, max_clusters=10, skip_llm=True)

    assert clusters
    for cluster in clusters:
        union: set[str] = set()
        for member in cluster["members"]:
            union.update(member["tags"])
        namespace_for_write(sorted(union), auto_project=False)
