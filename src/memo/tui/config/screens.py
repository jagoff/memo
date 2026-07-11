"""Setup, review, conflict, recovery, and result screens."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Protocol, cast

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, Switch

from memo.config_md import config_dir, config_home
from memo.tui.config.apply import ApplyPlan, PlannedChange, TransactionReceipt
from memo.tui.config.catalog import SettingKind, catalog_by_key, domain_file_for_key
from memo.tui.config.impact import ImpactResult, plan_impacts
from memo.tui.config.session import ConfigSession, ValidationIssue


class _ConfigApp(Protocol):
    def action_review(self) -> None: ...

    async def apply_review(self, plan: ApplyPlan, *, activate: bool) -> None: ...

    async def resolve_conflict(self, plan: ApplyPlan, *, keep_draft: bool) -> None: ...

    async def enter_read_only(self) -> None: ...

    async def restore_backup(self, manifest: Path) -> None: ...

    async def reload_session(self, *, close_screens: bool = False) -> None: ...

    async def refresh_center(self) -> None: ...


class _ConfigModal(ModalScreen[None]):
    @property
    def config_app(self) -> _ConfigApp:
        return cast("_ConfigApp", self.app)


def _masked_value(change: PlannedChange, value: object | None) -> str:
    spec = catalog_by_key()[change.key]
    if spec.sensitive or spec.kind is SettingKind.SECRET:
        return "********"
    if change.unset and value is None:
        return "<unset>"
    return str(value)


def _pending_transaction(home: Path) -> Path | None:
    root = home / ".transactions"
    if not root.is_dir():
        return None
    for path in sorted(root.glob("*/manifest.json"), reverse=True):
        try:
            state = json.loads(path.read_text(encoding="utf-8")).get("state")
        except (OSError, json.JSONDecodeError):
            continue
        if state in {"prepared", "committing", "rollback_failed"}:
            return path
    return None


def _latest_backup(home: Path) -> Path | None:
    root = home / ".transactions"
    if not root.is_dir():
        return None
    for path in sorted(root.glob("*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries = manifest.get("files", [])
        if any(
            (path.parent / str(entry.get("backup", ""))).is_file()
            for entry in entries
            if isinstance(entry, dict)
        ):
            return path
    return None


class FirstRunWizard(_ConfigModal):
    step_count = 4
    _TITLES = (
        "Storage and vault",
        "Hardware and model profile",
        "Integrations and recall",
        "Privacy, capture, and summary",
    )

    def __init__(self, session: ConfigSession) -> None:
        super().__init__()
        self.session = session
        self.step = 0
        self.values: dict[str, object] = {}
        self.legacy_available = bool(session.legacy)

    def compose(self) -> ComposeResult:
        with Vertical(classes="screen-panel wizard-panel"):
            yield Label("First-run configuration", classes="screen-title")
            yield Static(
                f"Step {self.step + 1} of {self.step_count}: {self._TITLES[self.step]}",
                id="wizard-progress",
            )
            with VerticalScroll(id="wizard-body"):
                yield from self._step_widgets()
            with Horizontal(classes="screen-actions"):
                if self.legacy_available and self.step == 0:
                    yield Button("Migrate legacy", id="wizard-migrate-legacy")
                if self.step:
                    yield Button("Back", id="wizard-back")
                yield Button("Cancel", id="wizard-cancel")
                yield Button(
                    "Review" if self.step == self.step_count - 1 else "Next",
                    id="wizard-next",
                    variant="primary",
                )

    def _step_widgets(self) -> ComposeResult:
        if self.step == 0:
            yield Label("Memory data directory")
            yield Input(
                str(self.values.get("storage.data_dir", self.session.state("storage.data_dir").effective_value)),
                id="wizard-data-dir",
            )
            yield Label("Obsidian vault (optional)")
            yield Input(
                str(self.values.get("storage.vault_path", self.session.state("storage.vault_path").effective_value or "")),
                id="wizard-vault-path",
            )
        elif self.step == 1:
            yield Label("Model profile")
            profile = str(
                self.values.get(
                    "models.model_profile",
                    self.session.state("models.model_profile").effective_value,
                )
            )
            yield Select(
                [("Light", "light"), ("Balanced", "balanced"), ("Quality", "quality")],
                value=profile,
                allow_blank=False,
                id="wizard-model-profile",
            )
            yield Static("Balanced is recommended for most installations.")
        elif self.step == 2:
            yield Label("Recall hook")
            yield Switch(
                not bool(self.values.get("recall.disable", False)),
                id="wizard-recall-enabled",
                animate=False,
            )
            yield Label("Keep hooks repaired automatically")
            yield Switch(
                bool(self.values.get("update.hook_selfheal", True)),
                id="wizard-hook-selfheal",
                animate=False,
            )
        else:
            yield Label("Automatic capture")
            yield Switch(
                not bool(self.values.get("capture.disable", False)),
                id="wizard-capture-enabled",
                animate=False,
            )
            yield Label("Redact detected secrets")
            yield Switch(
                bool(self.values.get("privacy.redact_secrets", True)),
                id="wizard-redact-secrets",
                animate=False,
            )
            yield Label("Honor private markers")
            yield Switch(
                bool(self.values.get("privacy.private_markers", True)),
                id="wizard-private-markers",
                animate=False,
            )
            yield Static(
                "Draft summary\n"
                + "\n".join(f"  {key}: {value}" for key, value in self.values.items())
                + "\nReview will show every change before any file is created.",
                classes="screen-note",
            )

    def _capture_step(self) -> None:
        if self.step == 0:
            self.values["storage.data_dir"] = self.query_one("#wizard-data-dir", Input).value
            vault = self.query_one("#wizard-vault-path", Input).value.strip()
            if vault:
                self.values["storage.vault_path"] = vault
        elif self.step == 1:
            value = self.query_one("#wizard-model-profile", Select).value
            if value is not Select.NULL:
                self.values["models.model_profile"] = str(value)
        elif self.step == 2:
            self.values["recall.disable"] = not self.query_one(
                "#wizard-recall-enabled", Switch
            ).value
            self.values["update.hook_selfheal"] = self.query_one(
                "#wizard-hook-selfheal", Switch
            ).value
        else:
            self.values["capture.disable"] = not self.query_one(
                "#wizard-capture-enabled", Switch
            ).value
            self.values["privacy.redact_secrets"] = self.query_one(
                "#wizard-redact-secrets", Switch
            ).value
            self.values["privacy.private_markers"] = self.query_one(
                "#wizard-private-markers", Switch
            ).value

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id
        if button_id == "wizard-cancel":
            self.app.exit(0)
            return
        if button_id == "wizard-back":
            self.step -= 1
            await self.recompose()
            return
        if button_id == "wizard-migrate-legacy":
            await self._migrate_legacy()
            return
        if button_id != "wizard-next":
            return
        self._capture_step()
        if self.step < self.step_count - 1:
            self.step += 1
            await self.recompose()
            return
        for key, value in self.values.items():
            self.session.set_value(key, value)
        self.dismiss()
        self.app.call_after_refresh(self.config_app.action_review)

    async def _migrate_legacy(self) -> None:
        from memo.config_md import write_default_config

        data_dir = self.session.legacy.get("data_dir")
        if not data_dir:
            return
        vault = self.session.legacy.get("vault_path")
        write_default_config(
            data_dir=Path(str(data_dir)),
            vault_path=Path(str(vault)) if vault else None,
            env=self.session.env,
        )
        self.dismiss()
        await self.config_app.reload_session()


class ReviewScreen(_ConfigModal):
    def __init__(self, session: ConfigSession, plan: ApplyPlan) -> None:
        super().__init__()
        self.session = session
        self.plan = plan
        self.impacts = plan_impacts(plan.changes)

    def compose(self) -> ComposeResult:
        with Vertical(classes="screen-panel review-panel"):
            yield Label("Review changes", classes="screen-title")
            with VerticalScroll(id="review-changes"):
                for change in self.plan.changes:
                    state = self.session.state(change.key)
                    warning = " (ENV override remains effective)" if state.env_override else ""
                    yield Static(
                        f"{change.key}\n  configured: {_masked_value(change, change.before)} -> "
                        f"{_masked_value(change, change.after)}\n"
                        f"  effective now: {_masked_value(change, state.effective_value)}{warning}",
                        classes="review-change",
                    )
                yield Static("Affected files", classes="review-section")
                affected = {
                    self.plan.snapshot.key_files.get(change.key)
                    or config_dir(self.session.env) / domain_file_for_key(change.key)
                    for change in self.plan.changes
                }
                for path in sorted(affected):
                    yield Static(f"  {path}")
                if self.impacts:
                    yield Static("Activation actions", classes="review-section")
                    for action in self.impacts:
                        yield Static(f"  {action.label}: {' '.join(action.argv)}")
            with Horizontal(classes="screen-actions"):
                yield Button("Cancel", id="review-cancel")
                yield Button("Save only", id="apply-save-only", variant="primary")
                if self.impacts:
                    yield Button("Save and activate", id="apply-with-actions", variant="warning")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "review-cancel":
            self.dismiss()
        elif event.button.id == "apply-save-only":
            await self.config_app.apply_review(self.plan, activate=False)
        elif event.button.id == "apply-with-actions":
            await self.config_app.apply_review(self.plan, activate=True)


class ConflictScreen(_ConfigModal):
    def __init__(
        self,
        plan: ApplyPlan,
        keys: tuple[str, ...],
        disk_values: dict[str, object | None],
    ) -> None:
        super().__init__()
        self.plan = plan
        self.keys = keys
        self.disk_values = disk_values

    def compose(self) -> ComposeResult:
        changes = {change.key: change for change in self.plan.changes}
        with Vertical(classes="screen-panel conflict-panel"):
            yield Label("Configuration changed on disk", classes="screen-title")
            for key in self.keys:
                change = changes[key]
                yield Static(
                    f"{key}\n  baseline: {_masked_value(change, change.before)}\n"
                    f"  disk: {_masked_value(change, self.disk_values.get(key))}\n"
                    f"  draft: {_masked_value(change, change.after)}",
                    classes="review-change",
                )
            with Horizontal(classes="screen-actions"):
                yield Button("Back", id="conflict-cancel")
                yield Button("Reload disk", id="conflict-reload")
                yield Button("Keep draft", id="conflict-keep-draft", variant="warning")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "conflict-cancel":
            self.dismiss()
        elif event.button.id == "conflict-reload":
            await self.config_app.resolve_conflict(self.plan, keep_draft=False)
        elif event.button.id == "conflict-keep-draft":
            await self.config_app.resolve_conflict(self.plan, keep_draft=True)


class RecoveryScreen(_ConfigModal):
    def __init__(
        self,
        session: ConfigSession,
        issues: tuple[ValidationIssue, ...],
    ) -> None:
        super().__init__()
        self.session = session
        self.issues = issues
        self.pending_manifest = _pending_transaction(config_home(session.env))
        self.backup_manifest = self.pending_manifest or _latest_backup(config_home(session.env))

    def compose(self) -> ComposeResult:
        with Vertical(classes="screen-panel recovery-panel"):
            yield Label("Configuration needs recovery", classes="screen-title")
            with VerticalScroll(id="recovery-issues"):
                for issue in self.issues:
                    yield Static(f"{issue.key}: {issue.message}", classes="recovery-issue")
                if self.pending_manifest:
                    yield Static(f"Interrupted transaction: {self.pending_manifest}")
            with Horizontal(classes="screen-actions"):
                yield Button("Open editor", id="recovery-editor")
                if self.backup_manifest:
                    yield Button("Restore backup", id="recovery-backup", variant="warning")
                yield Button("Read only", id="recovery-read-only")
                yield Button("Exit", id="recovery-exit")

    def _source_path(self) -> Path:
        existing = [path for path, fingerprint in self.session.snapshot.files.items() if fingerprint.exists]
        return sorted(existing)[0] if existing else config_home(self.session.env) / "memo-config.md"

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "recovery-exit":
            self.app.exit(1)
        elif event.button.id == "recovery-read-only":
            await self.config_app.enter_read_only()
        elif event.button.id == "recovery-editor":
            await self._open_editor()
        elif event.button.id == "recovery-backup" and self.backup_manifest:
            await self.config_app.restore_backup(self.backup_manifest)

    async def _open_editor(self) -> None:
        argv = (*shlex.split(os.environ.get("EDITOR", "vi")), str(self._source_path()))

        def run_editor() -> tuple[int, str]:
            completed = subprocess.run(argv, check=False, shell=False)
            return completed.returncode, ""

        await self.app.run_worker(run_editor, thread=True, exclusive=True).wait()
        self.dismiss()
        await self.config_app.reload_session()


class ApplyResultScreen(_ConfigModal):
    def __init__(
        self,
        receipt: TransactionReceipt,
        results: tuple[ImpactResult, ...],
    ) -> None:
        super().__init__()
        self.receipt = receipt
        self.results = results
        self.failure_count = sum(not result.success for result in results)

    def compose(self) -> ComposeResult:
        with Vertical(classes="screen-panel result-panel"):
            yield Label("Configuration saved", classes="screen-title")
            yield Static(f"{len(self.receipt.files)} file(s) committed")
            for result in self.results:
                status = "OK" if result.success else "FAILED"
                yield Static(f"{status}  {result.action.label}\n{result.output}")
            if self.failure_count:
                yield Static(
                    f"{self.failure_count} activation action(s) failed; saved files were kept.",
                    classes="result-error",
                )
            with Horizontal(classes="screen-actions"):
                yield Button("Close", id="result-close", variant="primary")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "result-close":
            self.dismiss()
            await self.config_app.refresh_center()


__all__ = [
    "ApplyResultScreen",
    "ConflictScreen",
    "FirstRunWizard",
    "RecoveryScreen",
    "ReviewScreen",
]
