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
