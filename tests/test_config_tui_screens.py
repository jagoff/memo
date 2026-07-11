"""Safe setup, review, conflict, recovery, and result screen flows."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button

from memo.tui.config.app import ConfigApp
from memo.tui.config.impact import ImpactAction, ImpactResult
from memo.tui.config.screens import (
    ApplyResultScreen,
    ConflictScreen,
    FirstRunWizard,
    RecoveryScreen,
    ReviewScreen,
)
from memo.tui.config.session import ConfigSession
from memo.tui.config.widgets import ValidationSummary


class FakeImpactController:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.executed: list[tuple[ImpactAction, ...]] = []

    def execute(self, actions: tuple[ImpactAction, ...]) -> tuple[ImpactResult, ...]:
        self.executed.append(actions)
        return tuple(
            ImpactResult(
                action=action,
                success=not self.fail,
                returncode=1 if self.fail else 0,
                output="failed" if self.fail else "ok",
            )
            for action in actions
        )


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_DIR": str(tmp_path / "memo-home"),
        "MEMO_CONFIG_FILE": str(tmp_path / "legacy.toml"),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def _session(tmp_path: Path, *, configured: bool = True) -> ConfigSession:
    if configured:
        home = tmp_path / "memo-home"
        home.mkdir(parents=True, exist_ok=True)
        (home / "memo-config.md").write_text("# Memo config\n", encoding="utf-8")
    return ConfigSession.open(_env(tmp_path))


@pytest.mark.asyncio
async def test_missing_config_opens_four_step_wizard(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path, configured=False))

    async with app.run_test(size=(120, 36)):
        wizard = app.screen

        assert isinstance(wizard, FirstRunWizard)
        assert wizard.step_count == 4


@pytest.mark.asyncio
async def test_review_blocks_invalid_draft(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path))

    async with app.run_test(size=(120, 36)):
        app.session.set_value("recall.top_k", -1)
        app.action_review()

        assert not isinstance(app.screen, ReviewScreen)
        assert app.query_one(ValidationSummary).blocking_count == 1


@pytest.mark.asyncio
async def test_apply_save_only_does_not_execute_impacts(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.set_value("models.model_profile", "quality")
    controller = FakeImpactController()
    app = ConfigApp(session, impact_controller=controller)

    async with app.run_test(size=(120, 36)) as pilot:
        app.action_review()
        await pilot.pause()
        await pilot.click("#apply-save-only")
        await pilot.pause()

        assert controller.executed == []
        assert isinstance(app.screen, ApplyResultScreen)


@pytest.mark.asyncio
async def test_wizard_cancel_creates_no_files(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    app = ConfigApp(_session(tmp_path, configured=False))

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        await pilot.click("#wizard-cancel")
        await pilot.pause()

    assert not home.exists()


@pytest.mark.asyncio
async def test_same_key_edit_opens_conflict_screen(tmp_path: Path) -> None:
    path = tmp_path / "memo-home" / "config" / "recall-config.md"
    path.parent.mkdir(parents=True)
    path.write_text("```toml\n[recall]\ntop_k = 3\n```\n", encoding="utf-8")
    session = ConfigSession.open(_env(tmp_path))
    session.set_value("recall.top_k", 5)
    app = ConfigApp(session)

    async with app.run_test(size=(120, 36)) as pilot:
        app.action_review()
        await pilot.pause()
        path.write_text("```toml\n[recall]\ntop_k = 4\n```\n", encoding="utf-8")
        await pilot.click("#apply-save-only")
        await pilot.pause()

        assert isinstance(app.screen, ConflictScreen)
        assert app.screen.disk_values == {"recall.top_k": 4}


@pytest.mark.asyncio
async def test_malformed_markdown_opens_recovery_screen(tmp_path: Path) -> None:
    path = tmp_path / "memo-home" / "config" / "recall-config.md"
    path.parent.mkdir(parents=True)
    path.write_text("```toml\n[recall\ntop_k = 3\n```\n", encoding="utf-8")
    app = ConfigApp(ConfigSession.open(_env(tmp_path)))

    async with app.run_test(size=(120, 36)):
        assert isinstance(app.screen, RecoveryScreen)


@pytest.mark.asyncio
async def test_partial_activation_failure_is_visible(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.set_value("models.model_profile", "quality")
    controller = FakeImpactController(fail=True)
    app = ConfigApp(session, impact_controller=controller)

    async with app.run_test(size=(120, 36)) as pilot:
        app.action_review()
        await pilot.pause()
        await pilot.click("#apply-with-actions")
        await pilot.pause()

        assert isinstance(app.screen, ApplyResultScreen)
        assert app.screen.failure_count == 1


@pytest.mark.asyncio
async def test_legacy_only_install_offers_migration(tmp_path: Path) -> None:
    (tmp_path / "legacy.toml").write_text(
        f'[storage]\ndata_dir = "{tmp_path / "legacy-data"}"\n',
        encoding="utf-8",
    )
    app = ConfigApp(_session(tmp_path, configured=False))

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()

        assert isinstance(app.screen, FirstRunWizard)
        assert app.screen.query_one("#wizard-migrate-legacy", Button)
