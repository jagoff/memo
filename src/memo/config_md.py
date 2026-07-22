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

# Graph controls originally lived in search/entity Markdown namespaces. Read
# those paths indefinitely, but expose and write only the canonical graph keys.
LEGACY_PATH_ALIASES: dict[str, str] = {
    "search.graph_expansion_enabled": "graph.expansion_enabled",
    "search.graph_signal_enabled": "graph.signal_enabled",
    "search.graph_reason_enabled": "graph.reason_enabled",
    "search.graph_semantic_relations": "graph.semantic_relations",
    "search.graph_hub_suppression": "graph.hub_suppression",
    "search.graph_signal_budget_ms": "graph.signal_budget_ms",
    "search.graph_hub_max_doc_freq_ratio": "graph.hub_max_doc_freq_ratio",
    "search.graph_min_entity_idf": "graph.min_entity_idf",
    "search.graph_outcome_signal_enabled": "graph.outcome_signal_enabled",
    "search.graph_outcome_weight": "graph.outcome_weight",
    "entity.graph_retrieval_enabled": "graph.retrieval_enabled",
    "entity.graph_density_boost": "graph.density_boost",
    "entity.graph_fallback_min_hits": "graph.fallback_min_hits",
    "recall.graph_proximity": "graph.signal_enabled",
    "recall.graph_proximity_weight": "graph.signal_alpha",
}


def _canonical_path_key(path_key: str) -> str:
    return LEGACY_PATH_ALIASES.get(path_key, path_key)


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


_cache: dict[
    str, tuple[tuple[tuple[str, float], ...], dict[str, ConfigValue], list[ConfigProblem]]
] = {}


def invalidate_cache() -> None:
    """Discard cached Markdown reads after an external transactional write."""
    _cache.clear()


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
    if env is not None and env is not os.environ and not env.get("MEMO_CONFIG_DIR"):
        # A caller-provided mapping is a complete, hermetic environment. Do
        # not fill a missing config directory from the process user's home;
        # validation and tests passing ``env={...}`` must inspect only that
        # explicit mapping. This mirrors tuned_overlay's custom-env boundary.
        return []
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


def _catalog_bindings() -> tuple[dict[str, str], dict[str, str]]:
    from memo.flags import REGISTRY
    from memo.tui.config.catalog import path_to_env, path_to_field

    flag_paths = {key: env_name for key, env_name in path_to_env().items() if env_name in REGISTRY}
    return flag_paths, path_to_field()


def _persistence_error(path_key: str) -> str | None:
    from memo.tui.config.catalog import PersistencePolicy, persistence_policy_for_key

    policy = persistence_policy_for_key(path_key)
    return {
        PersistencePolicy.PERSISTENT: None,
        PersistencePolicy.RUNTIME_ONLY: "runtime-only setting cannot be persisted",
        PersistencePolicy.DERIVED: "derived setting cannot be persisted",
        PersistencePolicy.SECRET: "secret setting must use encrypted secret storage",
    }[policy]


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


def _coerce_flag_bool(text: str) -> bool:
    low = text.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(f"expected a boolean (1/0/true/false), got {text!r}")


def _validate_flag_range(spec: FlagSpec, value: int | float) -> int | float:
    if spec.min_val is not None and value < spec.min_val:
        raise ValueError(f"{spec.name} must be >= {spec.min_val}, got {value}")
    if spec.max_val is not None and value > spec.max_val:
        raise ValueError(f"{spec.name} must be <= {spec.max_val}, got {value}")
    return value


def _coerce_flag_choice(spec: FlagSpec, text: str) -> str:
    value = text.strip().lower()
    if value not in spec.choices:
        allowed = ", ".join(spec.choices)
        raise ValueError(f"expected one of: {allowed}, got {text!r}")
    return value


def _coerce_flag_value(spec: FlagSpec, raw: Any) -> Any:
    """Match ``memo.flags._coerce`` for a TOML value without importing flags."""
    text = str(raw)
    if spec.kind == "bool":
        return _coerce_flag_bool(text)
    if spec.kind == "int":
        return _validate_flag_range(spec, int(text.strip()))
    if spec.kind == "float":
        return _validate_flag_range(spec, float(text.strip()))
    if spec.choices:
        return _coerce_flag_choice(spec, text)
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


def _canonical_entries(
    parsed: dict[str, Any],
    existing: Mapping[str, ConfigValue],
) -> list[tuple[str, Any]]:
    """Flatten one TOML block while preserving canonical-key precedence."""
    entries: list[tuple[str, Any]] = []
    for parsed_key, raw in _flatten("", parsed).items():
        key = _canonical_path_key(parsed_key)
        if key != parsed_key and key in existing:
            continue
        entries.append((key, raw))
    return entries


def _read_uncached(
    paths: list[Path], *, validate_values: bool
) -> tuple[dict[str, ConfigValue], list[ConfigProblem]]:
    flag_paths, field_paths = _catalog_bindings()
    values: dict[str, ConfigValue] = {}
    problems: list[ConfigProblem] = []
    for path in paths:
        if path.name != "memo-config.md" and path.name not in _CONFIG_FILENAMES:
            if path.name.endswith("-config.md"):
                problems.append(
                    ConfigProblem(
                        str(path), "", "", "unknown config file; register its domain first"
                    )
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
                problems.append(
                    ConfigProblem(str(path), f"block:{idx}", "", f"TOML parse error: {exc}")
                )
                continue
            for key, raw in _canonical_entries(parsed, values):
                env_name = flag_paths.get(key)
                field_name = field_paths.get(key)
                if env_name is None and field_name is None:
                    problems.append(ConfigProblem(str(path), key, str(raw), "unknown config key"))
                    continue
                policy_error = _persistence_error(key)
                if policy_error:
                    problems.append(ConfigProblem(str(path), key, str(raw), policy_error))
                    continue
                config_value = ConfigValue(
                    key=key,
                    value=raw,
                    source="markdown",
                    file=str(path),
                    env_name=env_name,
                    field_name=field_name,
                )
                # Validation mode checks each occurrence before a later file
                # can override it. Plain load mode intentionally skips this so
                # flags can read Markdown during memo.config import without a
                # Config -> flags -> config_md -> Config cycle.
                if validate_values:
                    problems.extend(_validate_mapped_values({key: config_value}))
                values[key] = config_value
    return values, problems


def load_values(env: Mapping[str, str] | None = None) -> dict[str, ConfigValue]:
    paths = _config_paths(env)
    sig = _signature(paths)
    cache_key = f"{config_home(env)}|load"
    cached = _cache.get(cache_key)
    if cached and cached[0] == sig:
        return cached[1]
    values, problems = _read_uncached(paths, validate_values=False)
    _cache[cache_key] = (sig, values, problems)
    return values


def configured_values(env: Mapping[str, str] | None = None) -> dict[str, ConfigValue]:
    """Return explicitly persisted values with their Markdown provenance."""
    return dict(load_values(env))


def validate_markdown_config(env: Mapping[str, str] | None = None) -> list[ConfigProblem]:
    paths = _config_paths(env)
    sig = _signature(paths)
    cache_key = f"{config_home(env)}|validate"
    cached = _cache.get(cache_key)
    if cached and cached[0] == sig:
        return cached[2]
    values, problems = _read_uncached(paths, validate_values=True)
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


def _quote(value: Any) -> str:
    if isinstance(value, bool):
        return '"on"' if value else '"off"'
    if isinstance(value, (int, float)):
        return str(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _domain_for_key(path_key: str) -> str:
    from memo.tui.config.catalog import domain_file_for_key

    return domain_file_for_key(path_key)


def _parse_raw_for_existing_key(path_key: str, raw_value: str) -> Any:
    from memo.tui.config.catalog import SettingKind, setting_kind_for_key

    kind = setting_kind_for_key(path_key)
    if kind is SettingKind.BOOL:
        value = _coerce_bool_for_flag(raw_value)
        if value is None:
            raise ValueError(f"expected a boolean value, got {raw_value!r}")
        return value
    if kind is SettingKind.INT:
        return int(raw_value)
    if kind is SettingKind.FLOAT:
        return float(raw_value)
    return raw_value


def _read_domain_table(path: Path, table: str) -> dict[str, Any]:
    if not path.is_file():
        return {}
    values = load_values({"MEMO_CONFIG_DIR": str(path.parent.parent)})
    prefix = f"{table}."
    return {
        key.removeprefix(prefix): value.value
        for key, value in values.items()
        if key.startswith(prefix)
    }


def _write_domain_file(path: Path, table: str, values: dict[str, Any], heading: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {heading}", "", "```toml", f"[{table}]"]
    for key in sorted(values):
        lines.append(f"{key} = {_quote(values[key])}")
    lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    _cache.clear()


def write_default_config(
    *,
    data_dir: Path,
    vault_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    force: bool = False,
) -> list[Path]:
    home = config_home(env)
    root = config_dir(env)
    paths = [index_path(env), *(root / name for name in sorted(_CONFIG_FILENAMES))]
    if not force:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(str(existing[0]))
    home.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    index_path(env).write_text(
        "# Memo config\n\n"
        "Persistent memo configuration lives in `config/*-config.md`.\n"
        "Environment variables override these files for temporary runtime changes.\n",
        encoding="utf-8",
    )
    storage = {"data_dir": str(data_dir), "memories_in_vault": "off", "single_db": "off"}
    if vault_path is not None:
        storage["vault_path"] = str(vault_path)
    _write_domain_file(root / "storage-config.md", "storage", storage, "Storage config")
    _write_domain_file(
        root / "models-config.md",
        "models",
        {"model_profile": "balanced", "embedder_dims": 1024},
        "Models config",
    )
    _write_domain_file(
        root / "recall-config.md",
        "recall",
        {"disable": "off", "top_k": 3, "min_sim": 0.5, "dedup_collapse": "on"},
        "Recall config",
    )
    for filename, table, heading in (
        ("search-config.md", "search", "Search config"),
        ("capture-config.md", "capture", "Capture config"),
        ("graph-config.md", "graph", "Graph config"),
        ("hooks-config.md", "hooks", "Hooks config"),
        ("advanced-config.md", "advanced", "Advanced config"),
    ):
        _write_domain_file(root / filename, table, {}, heading)
    return [path for path in paths if path.exists()]


def set_value(path_key: str, raw_value: str, env: Mapping[str, str] | None = None) -> Path:
    path_key = _canonical_path_key(path_key)
    policy_error = _persistence_error(path_key)
    if policy_error:
        raise ValueError(policy_error)
    filename = _domain_for_key(path_key)
    table, key = path_key.split(".", 1)
    path = config_dir(env) / filename
    values = _read_domain_table(path, table)
    values[key] = _parse_raw_for_existing_key(path_key, raw_value)
    _write_domain_file(path, table, values, f"{table.title()} config")
    return path


def unset_value(path_key: str, env: Mapping[str, str] | None = None) -> Path:
    path_key = _canonical_path_key(path_key)
    policy_error = _persistence_error(path_key)
    if policy_error:
        raise ValueError(policy_error)
    filename = _domain_for_key(path_key)
    table, key = path_key.split(".", 1)
    path = config_dir(env) / filename
    values = _read_domain_table(path, table)
    values.pop(key, None)
    _write_domain_file(path, table, values, f"{table.title()} config")
    return path
