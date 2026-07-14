"""Explicit post-save impact planning for configuration changes."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from memo.tui.config.apply import PlannedChange
from memo.tui.config.catalog import catalog_by_key


class ImpactTarget(StrEnum):
    RECALL_DAEMON = "recall-daemon"
    WATCHER = "watcher"
    HOOKS = "hooks"
    REINDEX = "reindex"


@dataclass(frozen=True)
class ImpactAction:
    target: ImpactTarget
    label: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class ImpactResult:
    action: ImpactAction
    success: bool
    returncode: int
    output: str


_ACTIONS = {
    ImpactTarget.RECALL_DAEMON: ImpactAction(
        ImpactTarget.RECALL_DAEMON,
        "Restart recall daemon",
        ("memo", "recall-daemon", "restart"),
    ),
    ImpactTarget.WATCHER: ImpactAction(
        ImpactTarget.WATCHER,
        "Restart filesystem watcher",
        ("memo", "install-watcher"),
    ),
    ImpactTarget.HOOKS: ImpactAction(
        ImpactTarget.HOOKS,
        "Rewire memo hooks",
        ("memo", "install-recall-hook"),
    ),
    ImpactTarget.REINDEX: ImpactAction(
        ImpactTarget.REINDEX,
        "Rebuild vector index",
        ("memo", "reindex", "--rebuild"),
    ),
}
_TARGET_ORDER = tuple(ImpactTarget)


def plan_impacts(changes: tuple[PlannedChange, ...]) -> tuple[ImpactAction, ...]:
    """Describe activation work without performing any lifecycle action."""
    catalog = catalog_by_key()
    targets: set[ImpactTarget] = set()
    for change in changes:
        if not change.unset and change.before == change.after:
            continue
        try:
            spec = catalog[change.key]
        except KeyError as exc:
            raise KeyError(f"unknown config key {change.key!r}") from exc
        targets.update(ImpactTarget(target) for target in spec.restart_targets)
    return tuple(_ACTIONS[target] for target in _TARGET_ORDER if target in targets)


Executor = Callable[[tuple[str, ...]], tuple[int, str]]


def _execute(argv: tuple[str, ...]) -> tuple[int, str]:
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return completed.returncode, output


class ImpactController:
    def __init__(self, executor: Executor | None = None) -> None:
        self.executor = executor or _execute

    def execute(self, actions: tuple[ImpactAction, ...]) -> tuple[ImpactResult, ...]:
        results: list[ImpactResult] = []
        for action in actions:
            try:
                returncode, output = self.executor(action.argv)
            except OSError as exc:
                returncode, output = -1, str(exc)
            results.append(
                ImpactResult(
                    action=action,
                    success=returncode == 0,
                    returncode=returncode,
                    output=output,
                )
            )
        return tuple(results)


__all__ = [
    "ImpactAction",
    "ImpactController",
    "ImpactResult",
    "ImpactTarget",
    "plan_impacts",
]
