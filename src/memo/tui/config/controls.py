"""Typed Textual controls that emit configuration intents."""

from __future__ import annotations

from typing import Literal

from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, Select, Switch

from memo.tui.config.catalog import PersistencePolicy, SettingKind
from memo.tui.config.session import SettingState


def control_id(key: str) -> str:
    return "control-" + key.replace(".", "-").replace("_", "-")


def _editable_value(state: SettingState) -> object | None:
    if state.pending_unset:
        return state.effective_value
    if state.pending_value is not None:
        return state.pending_value
    return state.effective_value


class SettingChanged(Message):
    def __init__(self, key: str, value: object) -> None:
        self.key = key
        self.value = value
        super().__init__()


class SettingSwitch(Switch):
    def __init__(self, state: SettingState, *, disabled: bool) -> None:
        self.setting_key = state.spec.key
        self._events_enabled = False
        self._memo_initial_value = bool(_editable_value(state))
        self._ignore_initial_event = True
        super().__init__(
            value=self._memo_initial_value,
            animate=False,
            id=control_id(state.spec.key),
            classes="setting-control",
            disabled=disabled,
            tooltip=state.spec.description,
        )

    def on_mount(self) -> None:
        self.call_after_refresh(self._enable_events)

    def _enable_events(self) -> None:
        self._events_enabled = True

class SettingSelect(Select[str]):
    def __init__(self, state: SettingState, *, disabled: bool) -> None:
        self.setting_key = state.spec.key
        self._events_enabled = False
        self._memo_initial_value = str(_editable_value(state))
        self._ignore_initial_event = True
        options = [(choice.label, choice.value) for choice in state.spec.choices]
        super().__init__(
            options,
            allow_blank=False,
            value=self._memo_initial_value,
            id=control_id(state.spec.key),
            classes="setting-control",
            disabled=disabled,
            tooltip=state.spec.description,
            compact=True,
        )

    def on_mount(self) -> None:
        self.call_after_refresh(self._enable_events)

    def _enable_events(self) -> None:
        self._events_enabled = True

class SettingInput(Input):
    def __init__(self, state: SettingState, *, disabled: bool) -> None:
        self.setting_key = state.spec.key
        self._events_enabled = False
        input_type: Literal["integer", "number", "text"] = (
            "integer"
            if state.spec.kind is SettingKind.INT
            else "number"
            if state.spec.kind is SettingKind.FLOAT
            else "text"
        )
        value = _editable_value(state)
        self._memo_initial_value = "" if value is None else str(value)
        self._ignore_initial_event = True
        super().__init__(
            None,
            type=input_type,
            password=state.spec.kind is SettingKind.SECRET or state.spec.sensitive,
            id=control_id(state.spec.key),
            classes="setting-control",
            disabled=disabled,
            tooltip=state.spec.description,
            compact=True,
        )

    def on_mount(self) -> None:
        self.value = self._memo_initial_value
        self.call_after_refresh(self._enable_events)

    def _enable_events(self) -> None:
        self._events_enabled = True

def control_for(state: SettingState) -> Widget:
    disabled = (
        not state.available or state.spec.policy is not PersistencePolicy.PERSISTENT
    )
    if state.spec.kind is SettingKind.BOOL:
        return SettingSwitch(state, disabled=disabled)
    if state.spec.choices:
        return SettingSelect(state, disabled=disabled)
    return SettingInput(state, disabled=disabled)


__all__ = [
    "SettingChanged",
    "SettingInput",
    "SettingSelect",
    "SettingSwitch",
    "control_for",
    "control_id",
]
