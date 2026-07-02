import pytest

from memo.errors import StorageError
from memo.memory.facade import Memory


def test_save_buckets_md_under_project_folder(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_STORE_BY_PROJECT", "1")
    mem = Memory(tmp_cfg)
    rec = mem.save(
        content="cuerpo",
        title="Hola",
        type_="note",
        tags=["project:memo"],
        auto_project=False,
        defer_embed=True,
    )
    assert rec.path.startswith("memo/")
    assert (tmp_cfg.memory_dir / rec.path).is_file()


def test_save_untagged_goes_to_global_bucket(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_STORE_BY_PROJECT", "1")
    mem = Memory(tmp_cfg)
    rec = mem.save(
        content="cuerpo", title="Hola", type_="note", auto_project=False, defer_embed=True
    )
    assert rec.path.startswith("_global/")
    assert (tmp_cfg.memory_dir / rec.path).is_file()


def test_store_by_project_off_keeps_flat_layout(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_STORE_BY_PROJECT", "0")
    mem = Memory(tmp_cfg)
    rec = mem.save(
        content="cuerpo",
        title="Hola",
        type_="note",
        tags=["project:memo"],
        auto_project=False,
        defer_embed=True,
    )
    assert "/" not in rec.path


def test_save_traversal_project_tag_stays_inside_memory_dir(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_STORE_BY_PROJECT", "1")
    mem = Memory(tmp_cfg)
    rec = mem.save(
        content="cuerpo",
        title="Hola",
        type_="note",
        tags=["project:../../evil"],
        auto_project=False,
        defer_embed=True,
    )
    abs_path = (tmp_cfg.memory_dir / rec.path).resolve()
    assert abs_path.is_relative_to(tmp_cfg.memory_dir.resolve())
    assert abs_path.is_file()
    # bucket is the sanitized slug; nothing planted outside the vault
    assert rec.path.startswith("evil/")
    assert not (tmp_cfg.memory_dir / "../../evil").resolve().exists()


def test_save_refuses_rel_path_outside_memory_dir(tmp_cfg, monkeypatch):
    mem = Memory(tmp_cfg)
    monkeypatch.setattr(mem, "_build_rel_path", lambda *a, **k: "../outside.md")
    with pytest.raises(StorageError):
        mem.save(
            content="cuerpo",
            title="Hola",
            type_="note",
            auto_project=False,
            defer_embed=True,
        )
    assert not (tmp_cfg.memory_dir / "../outside.md").resolve().exists()
