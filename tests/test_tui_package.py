"""Compatibility checks for the consolidated terminal UI package."""

from __future__ import annotations

from pathlib import Path


def test_picker_compatibility_exports_are_identical() -> None:
    from memo.setup.picker import PickerResult as old_result
    from memo.setup.picker import run_picker as old_picker
    from memo.tui.picker import PickerResult as new_result
    from memo.tui.picker import run_picker as new_picker

    assert old_result is new_result
    assert old_picker is new_picker


def test_dashboard_compatibility_exports_are_identical() -> None:
    from memo.dashboard_tui import render as old_render
    from memo.dashboard_tui import run_tui as old_run
    from memo.tui.dashboard import render as new_render
    from memo.tui.dashboard import run_tui as new_run

    assert old_render is new_render
    assert old_run is new_run


def test_resume_compatibility_exports_are_identical() -> None:
    from memo.resume._tui import pick_resume_candidate_interactive as old_picker
    from memo.tui.resume import pick_resume_candidate_interactive as new_picker

    assert old_picker is new_picker


def test_top_level_tui_modules_are_compatibility_shims() -> None:
    import memo.dashboard_panels
    import memo.dashboard_tui
    import memo.resume._tui
    import memo.setup.picker

    for module in (
        memo.dashboard_panels,
        memo.dashboard_tui,
        memo.resume._tui,
        memo.setup.picker,
    ):
        assert module.__file__ is not None
        body = Path(module.__file__).read_text(encoding="utf-8")
        assert "from memo.tui" in body
        assert "def run_tui" not in body
        assert "def run_picker" not in body
