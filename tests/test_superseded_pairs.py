"""`Memory.superseded_pairs()` (Task 11, deferred from Task 5).

Scans `cfg.memory_dir / "inactive" / *.md` for archived memories stamped
with `extra.superseded_by` (see `lifecycle.py:archive_memory`) and returns
`(stale_id, superseding_id, title)` tuples for the reliability detector.
"""

from __future__ import annotations

from datetime import UTC

import frontmatter


def test_superseded_pairs_returns_tuple_for_stamped_file(mock_memory, tmp_cfg):
    inactive = tmp_cfg.memory_dir / "inactive"
    inactive.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(
        "body text",
        id="old1",
        title="use X",
        type="fact",
        extra={"superseded_by": "new1"},
    )
    (inactive / "old1.md").write_text(frontmatter.dumps(post), encoding="utf-8")

    pairs = mock_memory.superseded_pairs()

    assert pairs == [("old1", "new1", "use X")]


def test_superseded_pairs_skips_files_without_superseded_by(mock_memory, tmp_cfg):
    inactive = tmp_cfg.memory_dir / "inactive"
    inactive.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post("body text", id="old2", title="stale, not superseded", type="fact")
    (inactive / "old2.md").write_text(frontmatter.dumps(post), encoding="utf-8")

    assert mock_memory.superseded_pairs() == []


def test_superseded_pairs_no_inactive_dir_returns_empty(mock_memory):
    assert mock_memory.superseded_pairs() == []


def test_superseded_pairs_respects_limit(mock_memory, tmp_cfg):
    inactive = tmp_cfg.memory_dir / "inactive"
    inactive.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        post = frontmatter.Post(
            "body",
            id=f"old{i}",
            title=f"fact {i}",
            type="fact",
            extra={"superseded_by": f"new{i}"},
        )
        (inactive / f"old{i}.md").write_text(frontmatter.dumps(post), encoding="utf-8")

    assert len(mock_memory.superseded_pairs(limit=2)) == 2


def test_superseded_pairs_skips_malformed_yaml_but_keeps_valid_pairs(mock_memory, tmp_cfg):
    """A malformed archive entry must not sink reliability nudges for every
    OTHER valid pair (I3 review fix: the per-file guard caught only
    `(OSError, ValueError)`, but `frontmatter.loads` raises `yaml.YAMLError`
    on bad YAML — neither superclass — so one bad file used to be fatal)."""
    inactive = tmp_cfg.memory_dir / "inactive"
    inactive.mkdir(parents=True, exist_ok=True)
    # Sorts before the valid file so it's processed first.
    (inactive / "bad.md").write_text(
        "---\nid: [unclosed\ntitle: broken\n---\nbody\n", encoding="utf-8"
    )
    post = frontmatter.Post(
        "body text",
        id="old1",
        title="use X",
        type="fact",
        extra={"superseded_by": "new1"},
    )
    (inactive / "good.md").write_text(frontmatter.dumps(post), encoding="utf-8")

    assert mock_memory.superseded_pairs() == [("old1", "new1", "use X")]


def test_superseded_pairs_skips_entries_missing_own_id(mock_memory, tmp_cfg):
    """A superseded-by-stamped file with no `id` of its own must be skipped —
    otherwise it yields `subject_id=""`, `action="memo get "` (M2 review fix)."""
    inactive = tmp_cfg.memory_dir / "inactive"
    inactive.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(
        "body text",
        title="no id",
        type="fact",
        extra={"superseded_by": "new1"},
    )
    (inactive / "idless.md").write_text(frontmatter.dumps(post), encoding="utf-8")

    assert mock_memory.superseded_pairs() == []


# Every fixture above writes files whose filename order already matches the
# order under test, and none stamps `superseded_at` — so `sorted(glob("*.md"))`
# looked correct. On the live store that sort is by hex memory id, which is
# neither recency nor relevance: 767 archived memories produced the same seven
# nudges every session (ids 03ef…, 1f28…, 2b77…, 4ae2…, 4dd7…, 550e…, 68e0…),
# forever, for supersessions resolved weeks earlier.


def _archived(inactive, *, mid, superseded_by, title, superseded_at=None):
    extra = {"superseded_by": superseded_by}
    if superseded_at is not None:
        extra["superseded_at"] = superseded_at
    post = frontmatter.Post("body", id=mid, title=title, type="fact", extra=extra)
    (inactive / f"{mid}.md").write_text(frontmatter.dumps(post), encoding="utf-8")


def test_superseded_pairs_are_ordered_by_recency_not_by_filename(mock_memory, tmp_cfg):
    from datetime import datetime, timedelta

    inactive = tmp_cfg.memory_dir / "inactive"
    inactive.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    # `aaa` sorts first by filename but was superseded longest ago.
    _archived(
        inactive,
        mid="aaa",
        superseded_by="n1",
        title="old news",
        superseded_at=(now - timedelta(days=20)).isoformat(),
    )
    _archived(
        inactive,
        mid="zzz",
        superseded_by="n2",
        title="just superseded",
        superseded_at=(now - timedelta(hours=1)).isoformat(),
    )

    pairs = mock_memory.superseded_pairs()

    assert [p[0] for p in pairs] == ["zzz", "aaa"]


def test_superseded_pairs_drops_supersessions_too_old_to_be_news(mock_memory, tmp_cfg):
    from datetime import datetime, timedelta

    inactive = tmp_cfg.memory_dir / "inactive"
    inactive.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    _archived(
        inactive,
        mid="ancient",
        superseded_by="n1",
        title="resolved months ago",
        superseded_at=(now - timedelta(days=400)).isoformat(),
    )

    assert mock_memory.superseded_pairs() == []
