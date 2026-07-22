from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from memo.cli import cli

pytestmark = pytest.mark.resource_hygiene


def test_digest_off_by_default(tmp_path):
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "d"),
        "MEMO_STATE_DIR": str(tmp_path / "s"),
    }
    r = CliRunner().invoke(cli, ["digest"], env=env)
    assert r.exit_code == 0
    assert "disabled" in r.output.lower() or "MEMO_PROACTIVE_ENABLED" in r.output


def test_digest_empty_when_enabled_no_candidates(tmp_path):
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "d"),
        "MEMO_STATE_DIR": str(tmp_path / "s"),
        "MEMO_PROACTIVE_ENABLED": "1",
    }
    r = CliRunner().invoke(cli, ["digest"], env=env)
    assert r.exit_code == 0
    assert "nothing to surface" in r.output


def test_digest_routes_dismiss_and_snooze_feedback_and_closes_store(tmp_path, monkeypatch):
    active = [object()]
    store = MagicMock(name="proactive_store")
    store.active_candidates.return_value = active
    store_context = MagicMock(name="proactive_store_context")
    store_context.__enter__.return_value = store
    store_factory = MagicMock(return_value=store_context)
    dismiss = MagicMock()
    snooze = MagicMock()
    routed = object()
    compute = MagicMock(return_value=routed)
    render = MagicMock(return_value="digest rendered")
    monkeypatch.setattr("memo.proactive.store.ProactiveStore", store_factory)
    monkeypatch.setattr("memo.cli_proactive._record_dismiss_feedback", dismiss)
    monkeypatch.setattr("memo.cli_proactive._record_snooze_feedback", snooze)
    monkeypatch.setattr("memo.proactive.engine.compute_routed", compute)
    monkeypatch.setattr("memo.proactive.surfaces.render_digest", render)
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "d"),
        "MEMO_STATE_DIR": str(tmp_path / "s"),
        "MEMO_PROACTIVE_ENABLED": "1",
    }

    result = CliRunner().invoke(
        cli,
        ["digest", "--dismiss", "nudge-1", "--snooze", "reliability", "--days", "3"],
        env=env,
    )

    assert result.exit_code == 0, result.output
    assert "digest rendered" in result.output
    store_factory.assert_called_once_with(tmp_path / "s" / "proactive.db")
    store_context.__exit__.assert_called_once()
    store.active_candidates.assert_called_once()
    now = store.active_candidates.call_args.args[0]
    dismiss.assert_called_once_with(store, "nudge-1", active, now)
    snooze.assert_called_once_with(store, "reliability", 3, active, now)
    assert compute.call_args.args == (store,)
    assert compute.call_args.kwargs["now"] == now
    render.assert_called_once_with(routed)
