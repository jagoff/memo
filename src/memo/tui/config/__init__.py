"""Terminal configuration center."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memo.tui.config.app import ConfigApp


def __getattr__(name: str) -> Any:
    if name in {"ConfigApp", "run_config_tui"}:
        from memo.tui.config.app import ConfigApp, run_config_tui

        return {"ConfigApp": ConfigApp, "run_config_tui": run_config_tui}[name]
    raise AttributeError(name)

__all__ = ["ConfigApp", "run_config_tui"]
