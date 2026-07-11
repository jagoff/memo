def test_first_run_picker_prints_sync_tip(monkeypatch, tmp_path, capsys):
    import memo.cli as cli_mod

    class _Result:
        data_dir = tmp_path / "memorias"
        vault_path = None

    monkeypatch.setattr(cli_mod, "run_picker", lambda: _Result())
    monkeypatch.setattr(
        "memo.config_md.write_default_config",
        lambda **kw: (tmp_path / "config.toml", None)
    )

    cli_mod._run_picker_and_save()

    out = capsys.readouterr().out
    assert "memo sync setup" in out
    assert "entre Macs" in out
