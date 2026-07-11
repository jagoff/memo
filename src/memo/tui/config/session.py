"""Source-aware, write-free configuration drafts for the Textual UI."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from memo.config_md import ConfigProblem, ConfigValue, configured_values, validate_markdown_config
from memo.platform_detect import is_apple_silicon
from memo.tui.config.catalog import (
    PersistencePolicy,
    SettingKind,
    SettingSpec,
    build_catalog,
    catalog_by_key,
    persistence_policy_for_key,
)


class ValueSource(StrEnum):
    ENV = "env"
    MARKDOWN = "markdown"
    OVERLAY = "overlay"
    LEGACY = "legacy"
    DEFAULT = "default"
    DERIVED = "derived"


@dataclass(frozen=True)
class ValidationIssue:
    key: str
    message: str
    blocking: bool = True


@dataclass(frozen=True)
class SettingState:
    spec: SettingSpec
    configured_value: object | None
    effective_value: object | None
    source: ValueSource
    default_value: object | None
    env_override: str | None = None
    pending_value: object | None = None
    pending_unset: bool = False
    available: bool = True
    availability_reason: str = ""
    issues: tuple[ValidationIssue, ...] = ()


@dataclass(frozen=True)
class DraftOperation:
    key: str
    value: object | None = None
    unset: bool = False


@dataclass
class ConfigDraft:
    operations: dict[str, DraftOperation] = field(default_factory=dict)

    def set(self, key: str, value: object) -> None:
        self.operations[key] = DraftOperation(key=key, value=value)

    def unset(self, key: str) -> None:
        self.operations[key] = DraftOperation(key=key, unset=True)

    def clear(self) -> None:
        self.operations.clear()


@dataclass(frozen=True)
class PlannedChange:
    key: str
    before: object | None
    after: object | None
    unset: bool = False


@dataclass(frozen=True)
class ApplyPlan:
    changes: tuple[PlannedChange, ...]
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def blocked(self) -> bool:
        return any(issue.blocking for issue in self.issues)


_POLICY_ERRORS = {
    PersistencePolicy.RUNTIME_ONLY: "runtime-only setting cannot be persisted",
    PersistencePolicy.DERIVED: "derived setting cannot be persisted",
    PersistencePolicy.SECRET: "secret setting must use encrypted secret storage",
}


def _coerce_scalar(spec: SettingSpec, raw: object) -> object:
    if raw is None:
        return None
    if spec.config_field:
        from memo.config import Config

        return getattr(Config.model_validate({spec.config_field: raw}), spec.config_field)
    if spec.kind is SettingKind.BOOL:
        if isinstance(raw, bool):
            return raw
        bool_text = str(raw).strip().lower()
        if bool_text in {"1", "true", "yes", "on"}:
            return True
        if bool_text in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"expected a boolean value, got {raw!r}")
    value: object
    if spec.kind is SettingKind.INT:
        value = int(str(raw).strip())
    elif spec.kind is SettingKind.FLOAT:
        value = float(str(raw).strip())
    elif spec.kind is SettingKind.PATH:
        value = Path(os.path.expandvars(str(raw))).expanduser().resolve()
    else:
        value = str(raw)
    if spec.minimum is not None and float(str(value)) < spec.minimum:
        raise ValueError(f"must be >= {spec.minimum:g}")
    if spec.maximum is not None and float(str(value)) > spec.maximum:
        raise ValueError(f"must be <= {spec.maximum:g}")
    if spec.choices and str(value) not in {choice.value for choice in spec.choices}:
        allowed = ", ".join(choice.value for choice in spec.choices)
        raise ValueError(f"expected one of: {allowed}")
    return value


def _platform_key() -> str:
    return "darwin-arm64" if is_apple_silicon() else "unsupported"


def _legacy_storage(
    env: Mapping[str, str], *, use_default_path: bool
) -> dict[str, object]:
    from memo.setup.config_io import load_config_file

    raw_path = env.get("MEMO_CONFIG_FILE")
    if raw_path:
        path = Path(raw_path).expanduser()
    elif use_default_path:
        path = None
    else:
        return {}
    data = load_config_file(path)
    if not isinstance(data, dict) or not isinstance(data.get("storage"), dict):
        return {}
    return dict(data["storage"])


class ConfigSession:
    """An in-memory editing session over memo's effective configuration."""

    def __init__(
        self,
        env: Mapping[str, str],
        markdown: Mapping[str, ConfigValue],
        legacy: Mapping[str, object],
        overlay: Mapping[str, str],
        source_problems: tuple[ConfigProblem, ...] = (),
    ) -> None:
        self.env = dict(env)
        self.markdown = dict(markdown)
        self.legacy = dict(legacy)
        self.overlay = dict(overlay)
        self._source_issues = tuple(
            ValidationIssue(problem.key or problem.file, problem.error) for problem in source_problems
        )
        self.draft = ConfigDraft()
        self._catalog = build_catalog()
        self._by_key = catalog_by_key()
        self._base_states = {spec.key: self._resolve_state(spec) for spec in self._catalog}

    @classmethod
    def open(cls, env: Mapping[str, str] | None = None) -> ConfigSession:
        from memo.tuned_overlay import overlay_values

        source = dict(os.environ if env is None else env)
        return cls(
            source,
            configured_values(source),
            _legacy_storage(source, use_default_path=env is None),
            overlay_values(source),
            tuple(validate_markdown_config(source)),
        )

    def _resolve_state(self, spec: SettingSpec, *, include_markdown: bool = True) -> SettingState:
        configured = self.markdown.get(spec.key)
        env_override = self.env.get(spec.env_name) if spec.env_name else None
        raw: object = spec.default
        source = ValueSource.DEFAULT
        issues: tuple[ValidationIssue, ...] = ()

        if spec.config_field in self.legacy and self.legacy[spec.config_field] not in (None, ""):
            raw = self.legacy[spec.config_field]
            source = ValueSource.LEGACY
        if spec.env_name in self.overlay:
            raw = self.overlay[spec.env_name]
            source = ValueSource.OVERLAY
        if include_markdown and configured is not None:
            raw = configured.value
            source = ValueSource.MARKDOWN
        if env_override not in (None, ""):
            raw = env_override
            source = ValueSource.ENV

        if source is ValueSource.DEFAULT:
            derived = self._profile_derived_value(spec)
            if derived is not None:
                raw = derived
                source = ValueSource.DERIVED

        try:
            effective = _coerce_scalar(spec, raw)
        except (TypeError, ValueError, ValidationError) as exc:
            effective = raw
            issues = (ValidationIssue(spec.key, str(exc)),)

        available = not spec.platforms or _platform_key() in spec.platforms
        reason = "" if available else "Available only on Apple Silicon."
        return SettingState(
            spec=spec,
            configured_value=configured.value if configured else None,
            effective_value=effective,
            source=source,
            default_value=spec.default,
            env_override=env_override,
            available=available,
            availability_reason=reason,
            issues=issues,
        )

    def _profile_derived_value(self, spec: SettingSpec) -> object | None:
        from memo.config import MODEL_PROFILES

        if not spec.config_field or spec.config_field == "model_profile":
            return None
        profile_spec = self._by_key["models.model_profile"]
        profile_state = self._resolve_state(profile_spec)
        profile = str(profile_state.effective_value)
        return MODEL_PROFILES.get(profile, {}).get(spec.config_field)

    def _ensure_persistent(self, key: str) -> SettingSpec:
        try:
            spec = self._by_key[key]
        except KeyError as exc:
            raise KeyError(f"unknown config key {key!r}") from exc
        policy = persistence_policy_for_key(key)
        if policy is not PersistencePolicy.PERSISTENT:
            raise ValueError(_POLICY_ERRORS[policy])
        return spec

    def set_value(self, key: str, raw: object) -> None:
        spec = self._ensure_persistent(key)
        try:
            value = _coerce_scalar(spec, raw)
        except (TypeError, ValueError, ValidationError):
            value = raw
        self.draft.set(key, value)

    def unset_value(self, key: str) -> None:
        self._ensure_persistent(key)
        self.draft.unset(key)

    def discard(self) -> None:
        self.draft.clear()

    def state(self, key: str) -> SettingState:
        try:
            base = self._base_states[key]
        except KeyError as exc:
            raise KeyError(f"unknown config key {key!r}") from exc
        operation = self.draft.operations.get(key)
        if operation is None:
            return replace(base, issues=self._issues_for_key(key))
        if operation.unset:
            return replace(
                base,
                pending_unset=True,
                issues=self._issues_for_key(key),
            )
        return replace(
            base,
            pending_value=operation.value,
            issues=self._issues_for_key(key),
        )

    def states(self) -> tuple[SettingState, ...]:
        return tuple(self.state(spec.key) for spec in self._catalog)

    def _projected_value(self, key: str) -> object | None:
        operation = self.draft.operations.get(key)
        if operation is None:
            return self._base_states[key].effective_value
        if not operation.unset:
            return operation.value
        return self._resolve_state(self._by_key[key], include_markdown=False).effective_value

    def _draft_issues(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for key, operation in self.draft.operations.items():
            state = self._base_states[key]
            if not state.available:
                issues.append(ValidationIssue(key, state.availability_reason))
            if operation.unset:
                continue
            try:
                _coerce_scalar(state.spec, operation.value)
            except (TypeError, ValueError, ValidationError) as exc:
                issues.append(ValidationIssue(key, str(exc)))

        if self._projected_value("storage.memories_in_vault") and not self._projected_value(
            "storage.vault_path"
        ):
            issues.append(
                ValidationIssue(
                    "storage.memories_in_vault",
                    "vault_path is required when memories_in_vault is enabled",
                )
            )

        from memo.config import MODEL_PROFILES

        model = self._projected_value("models.embedder_model")
        dims = self._projected_value("models.embedder_dims")
        expected_dims = {
            str(profile["embedder_model"]): profile["embedder_dims"]
            for profile in MODEL_PROFILES.values()
        }.get(str(model))
        if expected_dims is not None and dims != expected_dims:
            issues.append(
                ValidationIssue(
                    "models.embedder_dims",
                    f"embedder model requires {expected_dims} dimensions",
                )
            )
        return issues

    def _issues_for_key(self, key: str) -> tuple[ValidationIssue, ...]:
        base = self._base_states[key]
        return base.issues + tuple(issue for issue in self._draft_issues() if issue.key == key)

    def issues(self) -> tuple[ValidationIssue, ...]:
        base = tuple(issue for state in self._base_states.values() for issue in state.issues)
        candidates = self._source_issues + base + tuple(self._draft_issues())
        return tuple(dict.fromkeys(candidates))

    def review(self) -> ApplyPlan:
        changes: list[PlannedChange] = []
        for operation in self.draft.operations.values():
            configured = self.markdown.get(operation.key)
            changes.append(
                PlannedChange(
                    key=operation.key,
                    before=configured.value if configured else None,
                    after=None if operation.unset else operation.value,
                    unset=operation.unset,
                )
            )
        return ApplyPlan(changes=tuple(changes), issues=self.issues())


__all__ = [
    "ApplyPlan",
    "ConfigDraft",
    "ConfigSession",
    "DraftOperation",
    "PlannedChange",
    "SettingState",
    "ValidationIssue",
    "ValueSource",
]
