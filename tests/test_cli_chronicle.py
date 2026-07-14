"""Tests for the `memo chronicle` reader command."""

from __future__ import annotations

from click.testing import CliRunner


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }


def test_chronicle_reads_latest(tmp_path):
    from memo.cli import cli

    chron = tmp_path / "data" / "_chronicle"
    chron.mkdir(parents=True)
    (chron / "2026-07-12.md").write_text("# Crónica — 2026-07-12\nviejo\n", encoding="utf-8")
    (chron / "2026-07-13.md").write_text("# Crónica — 2026-07-13\nnuevo\n", encoding="utf-8")

    result = CliRunner().invoke(cli, ["chronicle"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "nuevo" in result.output and "viejo" not in result.output


def test_chronicle_date_and_missing(tmp_path):
    from memo.cli import cli

    result = CliRunner().invoke(cli, ["chronicle", "--date", "2026-01-01"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "no hay crónica" in result.output.lower()
