"""Shared terminal UI constants."""

from __future__ import annotations

CONFIG_DOMAINS = (
    "Storage",
    "Models",
    "Search",
    "Recall",
    "Capture",
    "Graph",
    "Hooks",
    "Maintenance",
    "Advanced",
)

MIN_TERMINAL_WIDTH = 80
MIN_TERMINAL_HEIGHT = 24

SEMANTIC_COLORS = {
    "success": "#66c989",
    "warning": "#e3b85b",
    "error": "#ef6f72",
    "information": "#61c7d7",
    "foreground": "#e8edf2",
    "background": "#101418",
}

__all__ = [
    "CONFIG_DOMAINS",
    "MIN_TERMINAL_HEIGHT",
    "MIN_TERMINAL_WIDTH",
    "SEMANTIC_COLORS",
]
