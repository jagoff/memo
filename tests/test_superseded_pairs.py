"""`Memory.superseded_pairs()` (Task 11, deferred from Task 5).

Scans `cfg.memory_dir / "inactive" / *.md` for archived memories stamped
with `extra.superseded_by` (see `lifecycle.py:archive_memory`) and returns
`(stale_id, superseding_id, title)` tuples for the reliability detector.
"""

from __future__ import annotations

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
