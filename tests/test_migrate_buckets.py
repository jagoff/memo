import frontmatter

from memo.runtime.migrate import _bucket_by_project


def _write_flat(memory_dir, name, tags):
    memory_dir.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post("body", id=name, title=name, type="note", tags=tags)
    (memory_dir / f"{name}.md").write_text(frontmatter.dumps(post), encoding="utf-8")


def test_bucket_by_project_moves_tagged_and_untagged(tmp_cfg):
    md = tmp_cfg.memory_dir
    _write_flat(md, "a", ["project:memo"])
    _write_flat(md, "b", [])
    moved = _bucket_by_project(tmp_cfg)
    assert (md / "memo" / "a.md").is_file()
    assert (md / "_global" / "b.md").is_file()
    assert not (md / "a.md").exists()
    assert moved == 2


def test_bucket_by_project_is_idempotent(tmp_cfg):
    md = tmp_cfg.memory_dir
    _write_flat(md, "a", ["project:memo"])
    _bucket_by_project(tmp_cfg)
    moved_again = _bucket_by_project(tmp_cfg)  # already bucketed -> 0
    assert moved_again == 0
    assert (md / "memo" / "a.md").is_file()
