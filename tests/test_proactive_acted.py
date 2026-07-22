import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.cli_proactive import record_acted_if_matches
from memo.proactive.nudge import KIND_RELIABILITY, Nudge
from memo.proactive.store import ProactiveStore

pytestmark = pytest.mark.resource_hygiene


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_running_the_action_records_acted(tmp_path: Path):
    with ProactiveStore(tmp_path / "p.db") as store:
        store.put_candidates(
            [
                Nudge.make(
                    KIND_RELIABILITY,
                    subject_id="old1",
                    urgency=0.9,
                    value=0.8,
                    title="t",
                    evidence=("new1",),
                    action="memo review old1",
                    created_at="2026-07-21T10:00:00Z",
                )
            ]
        )
        record_acted_if_matches(
            store,
            command_line="memo review old1",
            now="2026-07-21T10:05:00Z",
            window_min=30,
        )
        multipliers = store.kind_multipliers(floor=0.2)
        assert multipliers[KIND_RELIABILITY] >= 1.0  # acted counted


def test_mismatched_command_records_nothing(tmp_path: Path):
    with ProactiveStore(tmp_path / "p.db") as store:
        store.put_candidates(
            [
                Nudge.make(
                    KIND_RELIABILITY,
                    subject_id="old1",
                    urgency=0.9,
                    value=0.8,
                    title="t",
                    evidence=("new1",),
                    action="memo review old1",
                    created_at="2026-07-21T10:00:00Z",
                )
            ]
        )
        record_acted_if_matches(
            store,
            command_line="memo review other",
            now="2026-07-21T10:05:00Z",
            window_min=30,
        )
        multipliers = store.kind_multipliers(floor=0.2)
        assert KIND_RELIABILITY not in multipliers


def test_outside_window_records_nothing(tmp_path: Path):
    with ProactiveStore(tmp_path / "p.db") as store:
        store.put_candidates(
            [
                Nudge.make(
                    KIND_RELIABILITY,
                    subject_id="old1",
                    urgency=0.9,
                    value=0.8,
                    title="t",
                    evidence=("new1",),
                    action="memo review old1",
                    created_at="2026-07-21T10:00:00Z",
                )
            ]
        )
        record_acted_if_matches(
            store,
            command_line="memo review old1",
            now="2026-07-21T11:00:00Z",
            window_min=30,
        )
        multipliers = store.kind_multipliers(floor=0.2)
        assert KIND_RELIABILITY not in multipliers


def test_memo_get_records_acted_for_matching_nudge_when_enabled(tmp_path: Path) -> None:
    """`memo get <id>` IS the reliability nudge's suggested action — running it
    must feed `ProactiveStore.kind_multipliers` (I1 review fix: production had
    no caller for `record_acted_if_matches`, so the feedback loop only ever
    decayed)."""
    r = CliRunner()
    env = _env(tmp_path)
    save = r.invoke(
        cli, ["save", "--type", "note", "--defer-embed", "--json", "throwaway"], env=env
    )
    assert save.exit_code == 0, save.output
    mid = json.loads(save.output)["id"]

    with ProactiveStore(tmp_path / "state" / "proactive.db") as store:
        now = datetime.now(UTC).isoformat()
        store.put_candidates(
            [
                Nudge.make(
                    KIND_RELIABILITY,
                    subject_id=mid,
                    urgency=0.9,
                    value=0.8,
                    title="t",
                    evidence=("new1",),
                    action=f"memo get {mid}",
                    created_at=now,
                )
            ]
        )

        out = r.invoke(cli, ["get", mid], env={**env, "MEMO_PROACTIVE_ENABLED": "1"})
        assert out.exit_code == 0, out.output

        multipliers = store.kind_multipliers(floor=0.2)
        assert multipliers[KIND_RELIABILITY] >= 1.0


def test_memo_get_does_not_record_acted_when_disabled(tmp_path: Path) -> None:
    """Dark-flag guard: `MEMO_PROACTIVE_ENABLED` unset (default) must not touch
    the feedback loop at all."""
    r = CliRunner()
    env = _env(tmp_path)
    save = r.invoke(
        cli, ["save", "--type", "note", "--defer-embed", "--json", "throwaway"], env=env
    )
    assert save.exit_code == 0, save.output
    mid = json.loads(save.output)["id"]

    with ProactiveStore(tmp_path / "state" / "proactive.db") as store:
        now = datetime.now(UTC).isoformat()
        store.put_candidates(
            [
                Nudge.make(
                    KIND_RELIABILITY,
                    subject_id=mid,
                    urgency=0.9,
                    value=0.8,
                    title="t",
                    evidence=("new1",),
                    action=f"memo get {mid}",
                    created_at=now,
                )
            ]
        )

        out = r.invoke(cli, ["get", mid], env=env)
        assert out.exit_code == 0, out.output

        multipliers = store.kind_multipliers(floor=0.2)
        assert KIND_RELIABILITY not in multipliers
