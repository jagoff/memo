"""Markdown-backed memo configuration.

User-facing config lives in ~/.config/memo/memo-config.md and
~/.config/memo/config/*-config.md. Only fenced TOML blocks are parsed;
surrounding Markdown is for humans.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memo.flags_base import _FALSE, _TRUE, FlagSpec

_TOML_BLOCK_RE = re.compile(r"```toml\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_CONFIG_FILENAMES = {
    "storage-config.md",
    "models-config.md",
    "search-config.md",
    "recall-config.md",
    "capture-config.md",
    "graph-config.md",
    "hooks-config.md",
    "advanced-config.md",
}


@dataclass(frozen=True)
class ConfigValue:
    key: str
    value: Any
    source: str
    file: str
    env_name: str | None = None
    field_name: str | None = None


@dataclass(frozen=True)
class ConfigProblem:
    file: str
    key: str
    value: str
    error: str


_FIELD_PATHS: dict[str, str] = {
    "storage.data_dir": "data_dir",
    "storage.vault_path": "vault_path",
    "storage.memory_subdir": "memory_subdir",
    "storage.state_dir": "state_dir",
    "storage.memories_in_vault": "memories_in_vault",
    "storage.single_db": "single_db",
    "models.model_profile": "model_profile",
    "models.llm_model": "llm_model",
    "models.helper_model": "helper_model",
    "models.embedder_model": "embedder_model",
    "models.embedder_dims": "embedder_dims",
    "models.embedder_backend": "embedder_backend",
    "models.st_embedder_model": "st_embedder_model",
    "models.reranker_enabled": "reranker_enabled",
    "models.reranker_model": "reranker_model",
    "models.reranker_revision": "reranker_revision",
    "models.rerank_input_k": "rerank_input_k",
    "models.rerank_fusion_alpha": "rerank_fusion_alpha",
    "search.max_content_chars": "max_content_chars",
    "search.default_limit": "search_default_limit",
}

_cache: dict[str, tuple[tuple[tuple[str, float], ...], dict[str, ConfigValue], list[ConfigProblem]]] = {}


def config_home(env: Mapping[str, str] | None = None) -> Path:
    src = os.environ if env is None else env
    raw = src.get("MEMO_CONFIG_DIR")
    if raw:
        return Path(os.path.expandvars(raw)).expanduser().resolve()
    return (Path.home() / ".config" / "memo").resolve()


def index_path(env: Mapping[str, str] | None = None) -> Path:
    return config_home(env) / "memo-config.md"


def config_dir(env: Mapping[str, str] | None = None) -> Path:
    return config_home(env) / "config"


def _domain_files(env: Mapping[str, str] | None = None) -> list[Path]:
    root = config_dir(env)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix == ".md")


def _config_paths(env: Mapping[str, str] | None = None) -> list[Path]:
    """Return the index followed by domain files so domains retain precedence."""
    index = index_path(env)
    return ([index] if index.is_file() else []) + _domain_files(env)


def _signature(paths: list[Path]) -> tuple[tuple[str, float], ...]:
    sig: list[tuple[str, float]] = []
    for path in paths:
        try:
            sig.append((str(path), path.stat().st_mtime))
        except OSError:
            sig.append((str(path), -1.0))
    return tuple(sig)


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, inner in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(next_prefix, inner))
        return out
    return {prefix: value}


def _flag_path_map() -> dict[str, str]:
    from memo.flags import REGISTRY

    paths: dict[str, str] = {}
    for env_name, spec in REGISTRY.items():
        raw = env_name.removeprefix("MEMO_").lower()
        group_prefix = f"{spec.group}_"
        key = raw.removeprefix(group_prefix) if raw.startswith(group_prefix) else raw
        paths[f"{spec.group}.{key}"] = env_name
    return paths


def _coerce_bool_for_flag(raw: Any) -> str | None:
    if isinstance(raw, bool):
        return "on" if raw else "off"
    text = str(raw).strip().lower()
    if text in _TRUE:
        return "on"
    if text in _FALSE:
        return "off"
    return None


def _to_flag_string(env_name: str, raw: Any) -> str:
    from memo.flags import REGISTRY

    spec = REGISTRY[env_name]
    if spec.kind == "bool":
        coerced = _coerce_bool_for_flag(raw)
        return "" if coerced is None else coerced
    return str(raw)


def _coerce_flag_value(spec: FlagSpec, raw: Any) -> Any:
    """Match ``memo.flags._coerce`` for a TOML value without importing flags."""
    text = str(raw)
    if spec.kind == "bool":
        low = text.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"expected a boolean (1/0/true/false), got {text!r}")
    if spec.kind == "int":
        int_value = int(text.strip())
        if spec.min_val is not None and int_value < spec.min_val:
            raise ValueError(f"{spec.name} must be >= {spec.min_val}, got {int_value}")
        if spec.max_val is not None and int_value > spec.max_val:
            raise ValueError(f"{spec.name} must be <= {spec.max_val}, got {int_value}")
        return int_value
    if spec.kind == "float":
        float_value = float(text.strip())
        if spec.min_val is not None and float_value < spec.min_val:
            raise ValueError(f"{spec.name} must be >= {spec.min_val}, got {float_value}")
        if spec.max_val is not None and float_value > spec.max_val:
            raise ValueError(f"{spec.name} must be <= {spec.max_val}, got {float_value}")
        return float_value
    return text


def _validate_mapped_values(values: Mapping[str, ConfigValue]) -> list[ConfigProblem]:
    from pydantic import ValidationError

    from memo.config import Config
    from memo.flags import REGISTRY

    problems: list[ConfigProblem] = []
    for value in values.values():
        if value.env_name:
            try:
                _coerce_flag_value(REGISTRY[value.env_name], value.value)
            except ValueError as exc:
                problems.append(ConfigProblem(value.file, value.key, str(value.value), str(exc)))

    fields = {value.field_name: value.value for value in values.values() if value.field_name}
    try:
        Config(**fields)
    except ValidationError as exc:
        by_field = {value.field_name: value for value in values.values() if value.field_name}
        for error in exc.errors():
            field_name = error["loc"][0]
            if not isinstance(field_name, str):
                continue
            config_value = by_field.get(field_name)
            if config_value:
                problems.append(
                    ConfigProblem(
                        config_value.file,
                        config_value.key,
                        str(config_value.value),
                        f"config validation error: {error['msg']}",
                    )
                )
    return problems


def _read_uncached(paths: list[Path]) -> tuple[dict[str, ConfigValue], list[ConfigProblem]]:
    flag_paths = _flag_path_map()
    values: dict[str, ConfigValue] = {}
    problems: list[ConfigProblem] = []
    for path in paths:
        if path.name != "memo-config.md" and path.name not in _CONFIG_FILENAMES:
            if path.name.endswith("-config.md"):
                problems.append(
                    ConfigProblem(str(path), "", "", "unknown config file; register its domain first")
                )
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(ConfigProblem(str(path), "", "", f"failed to read file: {exc}"))
            continue
        for idx, block in enumerate(_TOML_BLOCK_RE.findall(text), start=1):
            try:
                parsed = tomllib.loads(block)
            except tomllib.TOMLDecodeError as exc:
                problems.append(ConfigProblem(str(path), f"block:{idx}", "", f"TOML parse error: {exc}"))
                continue
            for key, raw in _flatten("", parsed).items():
                env_name = flag_paths.get(key)
                field_name = _FIELD_PATHS.get(key)
                if env_name is None and field_name is None:
                    problems.append(ConfigProblem(str(path), key, str(raw), "unknown config key"))
                    continue
                config_value = ConfigValue(
                    key=key,
                    value=raw,
                    source="markdown",
                    file=str(path),
                    env_name=env_name,
                    field_name=field_name,
                )
                # Validate each occurrence before a later file can override it.
                problems.extend(_validate_mapped_values({key: config_value}))
                values[key] = config_value
    return values, problems


def load_values(env: Mapping[str, str] | None = None) -> dict[str, ConfigValue]:
    paths = _config_paths(env)
    sig = _signature(paths)
    cache_key = str(config_home(env))
    cached = _cache.get(cache_key)
    if cached and cached[0] == sig:
        return cached[1]
    values, problems = _read_uncached(paths)
    _cache[cache_key] = (sig, values, problems)
    return values


def validate_markdown_config(env: Mapping[str, str] | None = None) -> list[ConfigProblem]:
    paths = _config_paths(env)
    sig = _signature(paths)
    cache_key = str(config_home(env))
    cached = _cache.get(cache_key)
    if cached and cached[0] == sig:
        return cached[2]
    values, problems = _read_uncached(paths)
    _cache[cache_key] = (sig, values, problems)
    return problems


def field_values(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for value in load_values(env).values():
        if value.field_name:
            out[value.field_name] = value.value
    return out


def flag_values(env: Mapping[str, str] | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in load_values(env).values():
        if value.env_name:
            out[value.env_name] = _to_flag_string(value.env_name, value.value)
    return {key: value for key, value in out.items() if value != ""}


def active_config_values(env: Mapping[str, str] | None = None) -> dict[str, str]:
    return flag_values(env)
