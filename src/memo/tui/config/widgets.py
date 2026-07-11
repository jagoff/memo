"""Reusable widgets for the terminal configuration center."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Label, Static

from memo.tui.config.controls import control_for
from memo.tui.config.session import SettingState, ValidationIssue, ValueSource


def setting_widget_id(key: str) -> str:
    return "setting-" + key.replace(".", "-").replace("_", "-")


class DomainNav(VerticalScroll):
    class DomainSelected(Message):
        def __init__(self, domain: str) -> None:
            self.domain = domain
            super().__init__()

    def __init__(self, domains: tuple[str, ...], active: str) -> None:
        super().__init__(id="domain-nav")
        self.domains = domains
        self.active = active

    def compose(self) -> ComposeResult:
        for domain in self.domains:
            classes = "domain-active" if domain == self.active else ""
            yield Button(
                domain,
                id=f"domain-{domain.lower()}",
                classes=f"domain-button {classes}".strip(),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        domain = str(event.button.label)
        self.active = domain
        for button in self.query(".domain-button"):
            button.set_class(button is event.button, "domain-active")
        self.post_message(self.DomainSelected(domain))


class SourceBadge(Static):
    _LABELS: ClassVar[dict[ValueSource, str]] = {
        ValueSource.ENV: "ENV",
        ValueSource.MARKDOWN: "MD",
        ValueSource.OVERLAY: "TUNED",
        ValueSource.LEGACY: "LEGACY",
        ValueSource.DEFAULT: "DEFAULT",
        ValueSource.DERIVED: "DERIVED",
    }

    def __init__(self, source: ValueSource) -> None:
        self.source = source
        super().__init__(self._LABELS[source], classes=f"source-badge source-{source.value}")

    def set_source(self, source: ValueSource) -> None:
        self.source = source
        self.update(self._LABELS[source])
        for value in ValueSource:
            self.set_class(value is source, f"source-{value.value}")


class SettingRow(Vertical):
    def __init__(self, state: SettingState) -> None:
        super().__init__(id=setting_widget_id(state.spec.key), classes="setting-row")
        self.state = state
        self.setting_key = state.spec.key
        self.control = control_for(state)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="setting-line"):
            with Vertical(classes="setting-copy"):
                yield Label(self.state.spec.label, classes="setting-label")
                yield Static(self.state.spec.description, classes="setting-description")
                yield Static(self.state.spec.key, classes="setting-key")
            with Horizontal(classes="setting-meta"):
                yield SourceBadge(self.state.source)
                yield self.control
        yield Static(self._issue_text(), classes="setting-issue")

    def _issue_text(self) -> str:
        if self.state.issues:
            return self.state.issues[0].message
        if not self.state.available:
            return self.state.availability_reason
        if self.state.spec.policy.value != "persistent":
            return f"{self.state.spec.policy.value}; not stored in Markdown"
        return ""

    def update_state(self, state: SettingState) -> None:
        self.state = state
        self.query_one(SourceBadge).set_source(state.source)
        self.query_one(".setting-issue", Static).update(self._issue_text())
        self.set_class(bool(state.issues), "has-error")


class ValidationSummary(Static):
    def __init__(self) -> None:
        self.blocking_count = 0
        super().__init__("Configuration is valid", id="validation-summary")

    def set_issues(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.blocking_count = sum(issue.blocking for issue in issues)
        if self.blocking_count:
            self.update(f"{self.blocking_count} blocking issue(s)")
            self.set_class(True, "summary-error")
        else:
            self.update("Configuration is valid")
            self.set_class(False, "summary-error")


__all__ = [
    "DomainNav",
    "SettingRow",
    "SourceBadge",
    "ValidationSummary",
    "setting_widget_id",
]
