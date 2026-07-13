from click.testing import CliRunner

from memo import ask_gaps as ag
from memo import interject as ij
from memo.cli import cli
from memo.cli_interject import ask_group, interject_group


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_interject_shadow_reports_counts(tmp_path):
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    ij.log_shadow(sd, ij.shadow_record("switch instead", ["a" * 32], rendered=False))
    ij.log_shadow(sd, ij.shadow_record("revert that", ["a" * 32], rendered=True))
    r = CliRunner().invoke(interject_group, ["shadow"], env=_env(tmp_path))
    assert r.exit_code == 0
    assert "2" in r.output  # 2 would-fires
    assert "a" * 8 in r.output or "a" * 32 in r.output


def test_interject_shadow_empty(tmp_path):
    r = CliRunner().invoke(interject_group, ["shadow"], env=_env(tmp_path))
    assert r.exit_code == 0
    assert "no interject" in r.output.lower()


def test_ask_shadow_reports_counts(tmp_path):
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    ag.log_shadow(sd, ag.shadow_record({"prompt": "what is mmr", "count": 3}, rendered=False))
    r = CliRunner().invoke(ask_group, ["shadow"], env=_env(tmp_path))
    assert r.exit_code == 0
    assert "1" in r.output


def test_ask_group_name_is_ask_gaps_not_ask(tmp_path):
    """Regression: the group must NOT be named ``ask`` — that collides with
    the pre-existing stable-core `memo ask` question-answering command."""
    assert ask_group.name == "ask-gaps"


def test_interject_silence(tmp_path):
    r = CliRunner().invoke(
        interject_group, ["silence"], env={**_env(tmp_path), "MEMO_SESSION_ID": "s9"}
    )
    assert r.exit_code == 0
    assert ij.should_render(tmp_path / "state", "s9", max_per_session=5) is False


def test_interject_shadow_registered_on_main_cli(tmp_path):
    """Regression: `memo interject shadow` must be reachable through the main
    `cli` group, not just the standalone Click group — the docstrings in
    flags_recall.py / interject.py promise this exact command surface."""
    r = CliRunner().invoke(cli, ["interject", "shadow"], env=_env(tmp_path))
    assert r.exit_code == 0
    assert "no interject" in r.output.lower()


def test_ask_gaps_shadow_registered_on_main_cli(tmp_path):
    """Regression: `memo ask-gaps shadow` must be reachable through the main
    `cli` group — flags_misc.py's MEMO_ASK_GAPS_ENABLED docstring promises it."""
    r = CliRunner().invoke(cli, ["ask-gaps", "shadow"], env=_env(tmp_path))
    assert r.exit_code == 0
    assert "no ask-gaps" in r.output.lower()


def test_ask_gaps_registration_does_not_shadow_core_ask_command(tmp_path):
    """Regression: registering the Phase-3 review group must not clobber the
    pre-existing stable-core `memo ask` question-answering command."""
    r = CliRunner().invoke(cli, ["ask", "--help"], env=_env(tmp_path))
    assert r.exit_code == 0
    assert "shadow" not in r.output.lower()  # this is the real ask command, not ask-gaps


def test_interject_silence_registered_on_main_cli(tmp_path):
    r = CliRunner().invoke(
        cli, ["interject", "silence"], env={**_env(tmp_path), "MEMO_SESSION_ID": "s9"}
    )
    assert r.exit_code == 0
    assert ij.should_render(tmp_path / "state", "s9", max_per_session=5) is False
