"""Textual terminal configuration center."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.widgets import Button, Input, Label, Select, Static, Switch

from memo.tui.common import CONFIG_DOMAINS, MIN_TERMINAL_HEIGHT, MIN_TERMINAL_WIDTH
from memo.tui.config.catalog import Visibility
from memo.tui.config.controls import (
    SettingChanged,
    SettingInput,
    SettingSelect,
    SettingSwitch,
)
from memo.tui.config.session import ConfigSession, SettingState
from memo.tui.config.widgets import (
    DomainNav,
    SettingRow,
    ValidationSummary,
    setting_widget_id,
)


class ConfigApp(App[int]):
    CSS_PATH = "styles.tcss"
    TITLE = "memo config"
    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+r", "review", "Review"),
        ("ctrl+d", "discard", "Discard"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, session: ConfigSession) -> None:
        super().__init__()
        self.session = session
        self.active_domain = "Recall"
        self.search_query = ""
        self.show_advanced = False
        self._ignore_search_initial = True
        self._ignore_domain_initial = True
        self._ignore_advanced_initial = True

    def _visible_states(self) -> tuple[SettingState, ...]:
        states = self.session.states()
        query = self.search_query.strip().lower()
        if query:
            terms = query.split()
            return tuple(
                state
                for state in states
                if all(
                    term
                    in " ".join(
                        (
                            state.spec.key,
                            state.spec.label,
                            state.spec.description,
                            state.spec.domain,
                            state.spec.section,
                        )
                    ).lower()
                    for term in terms
                )
            )
        return tuple(
            state
            for state in states
            if state.spec.domain == self.active_domain
            and (self.show_advanced or state.spec.visibility is Visibility.COMMON)
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="config-center"):
            with Horizontal(id="config-header"):
                yield Label("memo config", id="config-title")
                yield Static("Terminal configuration center", id="config-context")
            with Horizontal(id="config-toolbar"):
                yield Input(placeholder="Search every setting", id="setting-search", compact=True)
                yield Label("Advanced", id="advanced-label")
                yield Switch(False, animate=False, id="show-advanced")
                yield Select(
                    [(domain, domain) for domain in CONFIG_DOMAINS],
                    value=self.active_domain,
                    allow_blank=False,
                    id="domain-select",
                    compact=True,
                )
            with Horizontal(id="workspace"):
                yield DomainNav(CONFIG_DOMAINS, self.active_domain)
                yield VerticalScroll(
                    *(SettingRow(state) for state in self._visible_states()),
                    id="settings-list",
                )
            with Horizontal(id="config-footer"):
                yield ValidationSummary()
                yield Static("0 pending", id="pending-summary")
                yield Button("Discard", id="discard-draft")
                yield Button("Review", id="review-draft", variant="primary")
        yield Static(
            f"memo config requires at least {MIN_TERMINAL_WIDTH}x{MIN_TERMINAL_HEIGHT}",
            id="size-warning",
        )

    def on_mount(self) -> None:
        self._update_responsive(self.screen.size.width, self.screen.size.height)
        self._update_summary()

    def on_resize(self, event: Resize) -> None:
        self._update_responsive(event.size.width, event.size.height)

    def _update_responsive(self, width: int, height: int) -> None:
        too_small = width < MIN_TERMINAL_WIDTH or height < MIN_TERMINAL_HEIGHT
        self.screen.set_class(width < 100, "narrow")
        self.query_one("#config-center").display = not too_small
        self.query_one("#size-warning").display = too_small

    async def _refresh_settings(self) -> None:
        container = self.query_one("#settings-list", VerticalScroll)
        await container.remove_children()
        states = self._visible_states()
        if states:
            await container.mount(*(SettingRow(state) for state in states))
        else:
            await container.mount(Static("No settings match", id="empty-settings"))

    async def on_input_changed(self, event: Input.Changed) -> None:
        if isinstance(event.input, SettingInput):
            if (
                event.input._ignore_initial_event
                and event.value == event.input._memo_initial_value
            ):
                event.input._ignore_initial_event = False
                return
            if event.input._events_enabled:
                event.input.post_message(SettingChanged(event.input.setting_key, event.value))
            return
        if event.input.id != "setting-search":
            return
        if self._ignore_search_initial and event.value == "":
            self._ignore_search_initial = False
            return
        self.search_query = event.value
        await self._refresh_settings()

    async def on_select_changed(self, event: Select.Changed) -> None:
        if isinstance(event.select, SettingSelect):
            if (
                event.select._ignore_initial_event
                and event.value == event.select._memo_initial_value
            ):
                event.select._ignore_initial_event = False
                return
            if event.select._events_enabled and event.value is not Select.NULL:
                event.select.post_message(
                    SettingChanged(event.select.setting_key, str(event.value))
                )
            return
        if event.select.id != "domain-select" or event.value is Select.NULL:
            return
        if self._ignore_domain_initial and event.value == self.active_domain:
            self._ignore_domain_initial = False
            return
        self.active_domain = str(event.value)
        self.search_query = ""
        self.query_one("#setting-search", Input).value = ""
        await self._refresh_settings()

    async def on_switch_changed(self, event: Switch.Changed) -> None:
        if isinstance(event.switch, SettingSwitch):
            if (
                event.switch._ignore_initial_event
                and event.value == event.switch._memo_initial_value
            ):
                event.switch._ignore_initial_event = False
                return
            if event.switch._events_enabled:
                event.switch.post_message(
                    SettingChanged(event.switch.setting_key, event.value)
                )
            return
        if event.switch.id != "show-advanced":
            return
        if self._ignore_advanced_initial and event.value is False:
            self._ignore_advanced_initial = False
            return
        self.show_advanced = event.value
        await self._refresh_settings()

    async def on_domain_nav_domain_selected(self, event: DomainNav.DomainSelected) -> None:
        self.active_domain = event.domain
        self.search_query = ""
        self.query_one("#setting-search", Input).value = ""
        select = self.query_one("#domain-select", Select)
        select.value = event.domain
        await self._refresh_settings()

    def on_setting_changed(self, event: SettingChanged) -> None:
        self.session.set_value(event.key, event.value)
        row = self.query_one(f"#{setting_widget_id(event.key)}", SettingRow)
        row.update_state(self.session.state(event.key))
        self._update_summary()

    def _update_summary(self) -> None:
        self.query_one(ValidationSummary).set_issues(self.session.issues())
        pending = len(self.session.draft.operations)
        self.query_one("#pending-summary", Static).update(f"{pending} pending")

    async def action_discard(self) -> None:
        self.session.discard()
        await self._refresh_settings()
        self._update_summary()

    def action_review(self) -> None:
        self._update_summary()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "discard-draft":
            await self.action_discard()
        elif event.button.id == "review-draft":
            self.action_review()


def run_config_tui(env: Mapping[str, str] | None = None) -> int:
    result = ConfigApp(ConfigSession.open(env)).run()
    return result if isinstance(result, int) else 0


__all__ = ["ConfigApp", "run_config_tui"]
