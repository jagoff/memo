"""Impact planning and explicit activation for configuration changes."""

from __future__ import annotations

from memo.tui.config.apply import PlannedChange
from memo.tui.config.impact import (
    ImpactAction,
    ImpactController,
    ImpactTarget,
    plan_impacts,
)


def _change(key: str, before: object, after: object) -> PlannedChange:
    return PlannedChange(key=key, before=before, after=after)


def test_model_change_requires_recall_daemon_restart() -> None:
    actions = plan_impacts((_change("models.model_profile", "light", "balanced"),))

    assert [action.target for action in actions] == [ImpactTarget.RECALL_DAEMON]


def test_hook_change_reports_hook_rewire() -> None:
    actions = plan_impacts((_change("update.hook_selfheal", False, True),))

    assert ImpactTarget.HOOKS in {action.target for action in actions}


def test_embedder_changes_deduplicate_restart_and_reindex() -> None:
    actions = plan_impacts(
        (
            _change("models.embedder_model", "old", "new"),
            _change("models.embedder_dims", 1024, 2560),
        )
    )

    assert [action.target for action in actions] == [
        ImpactTarget.RECALL_DAEMON,
        ImpactTarget.REINDEX,
    ]


def test_unchanged_value_has_no_impacts() -> None:
    assert plan_impacts((_change("models.model_profile", "balanced", "balanced"),)) == ()


def test_execute_uses_injected_executor_and_reports_partial_failure() -> None:
    seen: list[tuple[str, ...]] = []

    def executor(argv: tuple[str, ...]) -> tuple[int, str]:
        seen.append(argv)
        return (1, "failed")

    result = ImpactController(executor=executor).execute(
        (
            ImpactAction(
                ImpactTarget.RECALL_DAEMON,
                "Restart recall daemon",
                ("memo", "recall-daemon", "restart"),
            ),
        )
    )

    assert seen == [("memo", "recall-daemon", "restart")]
    assert result[0].success is False
    assert result[0].output == "failed"
