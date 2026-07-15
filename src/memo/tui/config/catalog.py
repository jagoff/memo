"""Typed catalog for every setting surfaced by the configuration TUI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Any


class SettingKind(StrEnum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    PATH = "path"
    SECRET = "secret"  # noqa: S105 - setting kind, not a credential
    ENUM = "enum"


class Visibility(StrEnum):
    COMMON = "common"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"


class PersistencePolicy(StrEnum):
    PERSISTENT = "persistent"
    RUNTIME_ONLY = "runtime-only"
    SECRET = "secret"  # noqa: S105 - policy label, not a credential
    DERIVED = "derived"


class RiskLevel(StrEnum):
    NORMAL = "normal"
    CAUTION = "caution"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class SettingChoice:
    label: str
    value: str
    description: str = ""


@dataclass(frozen=True)
class FieldBinding:
    key: str
    field: str
    env_name: str
    domain: str
    section: str
    kind: SettingKind
    visibility: Visibility = Visibility.ADVANCED
    choices: tuple[SettingChoice, ...] = ()
    risk: RiskLevel = RiskLevel.NORMAL
    platforms: frozenset[str] = frozenset()
    restart_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    description: str
    domain: str
    section: str
    kind: SettingKind
    default: Any
    env_name: str | None = None
    config_field: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[SettingChoice, ...] = ()
    visibility: Visibility = Visibility.ADVANCED
    policy: PersistencePolicy = PersistencePolicy.PERSISTENT
    risk: RiskLevel = RiskLevel.NORMAL
    platforms: frozenset[str] = frozenset()
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    restart_targets: tuple[str, ...] = ()
    sensitive: bool = False


_MODEL_PROFILE_CHOICES = (
    SettingChoice("Light", "light", "Lowest operational footprint."),
    SettingChoice("Balanced", "balanced", "Recommended quality and latency."),
    SettingChoice("Quality", "quality", "Larger embedder and higher memory use."),
)
_EMBEDDER_BACKEND_CHOICES = (
    SettingChoice("Automatic", "auto", "MLX on Apple Silicon, CPU elsewhere."),
    SettingChoice("MLX", "mlx", "Apple Silicon MLX runtime."),
    SettingChoice("Sentence Transformers", "st", "Portable CPU backend."),
)


FIELD_BINDINGS: tuple[FieldBinding, ...] = (
    FieldBinding(
        "storage.data_dir",
        "data_dir",
        "MEMO_DATA_DIR",
        "Storage",
        "Paths",
        SettingKind.PATH,
        Visibility.COMMON,
        restart_targets=("watcher",),
    ),
    FieldBinding(
        "storage.vault_path",
        "vault_path",
        "MEMO_VAULT_PATH",
        "Storage",
        "Paths",
        SettingKind.PATH,
        Visibility.COMMON,
        restart_targets=("watcher",),
    ),
    FieldBinding(
        "storage.memory_subdir",
        "memory_subdir",
        "MEMO_MEMORY_SUBDIR",
        "Storage",
        "Legacy",
        SettingKind.STR,
        restart_targets=("watcher",),
    ),
    FieldBinding(
        "storage.state_dir",
        "state_dir",
        "MEMO_STATE_DIR",
        "Storage",
        "Paths",
        SettingKind.PATH,
        Visibility.COMMON,
    ),
    FieldBinding(
        "storage.memories_in_vault",
        "memories_in_vault",
        "MEMO_MEMORIES_IN_VAULT",
        "Storage",
        "Layout",
        SettingKind.BOOL,
        Visibility.COMMON,
        restart_targets=("watcher",),
    ),
    FieldBinding(
        "storage.single_db",
        "single_db",
        "MEMO_SINGLE_DB",
        "Storage",
        "Layout",
        SettingKind.BOOL,
        risk=RiskLevel.CAUTION,
    ),
    FieldBinding(
        "models.model_profile",
        "model_profile",
        "MEMO_MODEL_PROFILE",
        "Models",
        "Profile",
        SettingKind.ENUM,
        Visibility.COMMON,
        _MODEL_PROFILE_CHOICES,
        restart_targets=("recall-daemon",),
    ),
    FieldBinding(
        "models.llm_model",
        "llm_model",
        "MEMO_LLM_MODEL",
        "Models",
        "Model IDs",
        SettingKind.STR,
        restart_targets=("recall-daemon",),
    ),
    FieldBinding(
        "models.helper_model",
        "helper_model",
        "MEMO_HELPER_MODEL",
        "Models",
        "Model IDs",
        SettingKind.STR,
        restart_targets=("recall-daemon",),
    ),
    FieldBinding(
        "models.embedder_model",
        "embedder_model",
        "MEMO_EMBEDDER_MODEL",
        "Models",
        "Embeddings",
        SettingKind.STR,
        risk=RiskLevel.CAUTION,
        restart_targets=("recall-daemon", "reindex"),
    ),
    FieldBinding(
        "models.embedder_dims",
        "embedder_dims",
        "MEMO_EMBEDDER_DIMS",
        "Models",
        "Embeddings",
        SettingKind.INT,
        risk=RiskLevel.CAUTION,
        restart_targets=("recall-daemon", "reindex"),
    ),
    FieldBinding(
        "models.embedder_backend",
        "embedder_backend",
        "MEMO_EMBEDDER_BACKEND",
        "Models",
        "Embeddings",
        SettingKind.ENUM,
        Visibility.COMMON,
        _EMBEDDER_BACKEND_CHOICES,
        restart_targets=("recall-daemon",),
    ),
    FieldBinding(
        "models.st_embedder_model",
        "st_embedder_model",
        "MEMO_ST_EMBEDDER_MODEL",
        "Models",
        "Embeddings",
        SettingKind.STR,
        restart_targets=("recall-daemon", "reindex"),
    ),
    FieldBinding(
        "models.st_embedder_revision",
        "st_embedder_revision",
        "MEMO_ST_EMBEDDER_REVISION",
        "Models",
        "Embeddings",
        SettingKind.STR,
        restart_targets=("recall-daemon", "reindex"),
    ),
    FieldBinding(
        "models.reranker_enabled",
        "reranker_enabled",
        "MEMO_RERANKER_ENABLED",
        "Models",
        "Reranking",
        SettingKind.BOOL,
        Visibility.COMMON,
        platforms=frozenset({"darwin-arm64"}),
        restart_targets=("recall-daemon",),
    ),
    FieldBinding(
        "models.reranker_model",
        "reranker_model",
        "MEMO_RERANKER_MODEL",
        "Models",
        "Reranking",
        SettingKind.STR,
        platforms=frozenset({"darwin-arm64"}),
        restart_targets=("recall-daemon",),
    ),
    FieldBinding(
        "models.reranker_revision",
        "reranker_revision",
        "MEMO_RERANKER_REVISION",
        "Models",
        "Reranking",
        SettingKind.STR,
        platforms=frozenset({"darwin-arm64"}),
        restart_targets=("recall-daemon",),
    ),
    FieldBinding(
        "models.rerank_input_k",
        "rerank_input_k",
        "MEMO_RERANK_INPUT_K",
        "Models",
        "Reranking",
        SettingKind.INT,
        platforms=frozenset({"darwin-arm64"}),
        restart_targets=("recall-daemon",),
    ),
    FieldBinding(
        "models.rerank_fusion_alpha",
        "rerank_fusion_alpha",
        "MEMO_RERANK_FUSION_ALPHA",
        "Models",
        "Reranking",
        SettingKind.FLOAT,
        platforms=frozenset({"darwin-arm64"}),
        restart_targets=("recall-daemon",),
    ),
    FieldBinding(
        "search.max_content_chars",
        "max_content_chars",
        "MEMO_MAX_CONTENT_CHARS",
        "Search",
        "Limits",
        SettingKind.INT,
    ),
    FieldBinding(
        "search.default_limit",
        "search_default_limit",
        "MEMO_SEARCH_DEFAULT_LIMIT",
        "Search",
        "Limits",
        SettingKind.INT,
        Visibility.COMMON,
    ),
)


GROUP_TO_DOMAIN: dict[str, str] = {
    "behavior": "Advanced",
    "bench": "Advanced",
    "briefing": "Recall",
    "cache": "Maintenance",
    "capture": "Capture",
    "cli": "Advanced",
    "dream": "Maintenance",
    "embedder": "Models",
    "entity": "Graph",
    "feedback": "Recall",
    "graph": "Graph",
    "ingest": "Capture",
    "links": "Graph",
    "maintain": "Maintenance",
    "mcp": "Hooks",
    "misc": "Advanced",
    "outcome": "Recall",
    "privacy": "Capture",
    "recall": "Recall",
    "repo": "Search",
    "retrieval": "Search",
    "roi": "Recall",
    "search": "Search",
    "secret": "Advanced",
    "session": "Recall",
    "store": "Storage",
    "synapse": "Hooks",
    "sync": "Hooks",
    "temporal": "Search",
    "update": "Maintenance",
    "whatsapp": "Capture",
}

DOMAIN_TO_FILE: dict[str, str] = {
    "Storage": "storage-config.md",
    "Models": "models-config.md",
    "Search": "search-config.md",
    "Recall": "recall-config.md",
    "Capture": "capture-config.md",
    "Graph": "graph-config.md",
    "Hooks": "hooks-config.md",
    "Maintenance": "advanced-config.md",
    "Advanced": "advanced-config.md",
}

COMMON_ENV_NAMES = frozenset(
    {
        "MEMO_AUTO_PROJECT_TAG",
        "MEMO_CAPTURE_DISABLE",
        "MEMO_EMBEDDER_VIA_DAEMON",
        "MEMO_GRAPH_SIGNAL_ENABLED",
        "MEMO_HOOK_SELFHEAL",
        "MEMO_MCP_PROFILE",
        "MEMO_RECALL_DEDUP_COLLAPSE",
        "MEMO_RECALL_DISABLE",
        "MEMO_RECALL_MIN_SIM",
        "MEMO_RECALL_TOP_K",
        "MEMO_SESSION_IDLE_CAPTURE_SECS",
        "MEMO_SYNC_AUTO",
    }
)
RUNTIME_ONLY_ENV_NAMES = frozenset({"MEMO_AGENT_TTY", "MEMO_NONINTERACTIVE"})
SENSITIVE_ENV_NAMES: frozenset[str] = frozenset()
EXPERIMENTAL_GROUPS = frozenset({"bench", "dream"})
FLAG_RESTART_TARGETS: dict[str, tuple[str, ...]] = {
    "MEMO_HOOK_SELFHEAL": ("hooks",),
}


def _label_from_key(key: str) -> str:
    return key.rsplit(".", 1)[-1].replace("_", " ").title()


def _flag_path(env_name: str, group: str) -> str:
    raw = env_name.removeprefix("MEMO_").lower()
    group_prefix = f"{group}_"
    key = raw.removeprefix(group_prefix) if raw.startswith(group_prefix) else raw
    return f"{group}.{key}"


def _field_specs() -> list[SettingSpec]:
    from pydantic_core import PydanticUndefined

    from memo.config import Config

    out: list[SettingSpec] = []
    for binding in FIELD_BINDINGS:
        field = Config.model_fields[binding.field]
        default = field.get_default(call_default_factory=True)
        if default is PydanticUndefined:
            default = None
        out.append(
            SettingSpec(
                key=binding.key,
                label=_label_from_key(binding.key),
                description=field.description or binding.key,
                domain=binding.domain,
                section=binding.section,
                kind=binding.kind,
                default=default,
                env_name=binding.env_name,
                config_field=binding.field,
                choices=binding.choices,
                visibility=binding.visibility,
                risk=binding.risk,
                platforms=binding.platforms,
                restart_targets=binding.restart_targets,
            )
        )
    return out


def _flag_specs(bound_env_names: frozenset[str]) -> list[SettingSpec]:
    from memo.flags import REGISTRY

    kind_map = {
        "bool": SettingKind.BOOL,
        "int": SettingKind.INT,
        "float": SettingKind.FLOAT,
        "str": SettingKind.STR,
    }
    out: list[SettingSpec] = []
    for env_name, flag in REGISTRY.items():
        if env_name in bound_env_names:
            continue
        try:
            domain = GROUP_TO_DOMAIN[flag.group]
        except KeyError as exc:
            raise ValueError(f"unmapped config group: {flag.group}") from exc
        policy = (
            PersistencePolicy.RUNTIME_ONLY
            if env_name in RUNTIME_ONLY_ENV_NAMES
            else PersistencePolicy.PERSISTENT
        )
        visibility = (
            Visibility.COMMON
            if env_name in COMMON_ENV_NAMES
            else Visibility.EXPERIMENTAL
            if flag.group in EXPERIMENTAL_GROUPS
            else Visibility.ADVANCED
        )
        out.append(
            SettingSpec(
                key=_flag_path(env_name, flag.group),
                label=_label_from_key(_flag_path(env_name, flag.group)),
                description=flag.help,
                domain=domain,
                section=flag.group.replace("_", " ").title(),
                kind=kind_map[flag.kind],
                default=flag.default,
                env_name=env_name,
                minimum=flag.min_val,
                maximum=flag.max_val,
                visibility=visibility,
                policy=policy,
                restart_targets=FLAG_RESTART_TARGETS.get(env_name, ()),
                sensitive=env_name in SENSITIVE_ENV_NAMES,
            )
        )
    return out


@cache
def build_catalog() -> tuple[SettingSpec, ...]:
    field_specs = _field_specs()
    bound_env_names = frozenset(spec.env_name for spec in field_specs if spec.env_name)
    catalog = tuple(field_specs + _flag_specs(bound_env_names))
    keys = [spec.key for spec in catalog]
    if len(keys) != len(set(keys)):
        duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
        raise ValueError(f"duplicate config keys: {duplicates}")
    return catalog


@cache
def catalog_by_key() -> dict[str, SettingSpec]:
    return {spec.key: spec for spec in build_catalog()}


@cache
def path_to_env() -> dict[str, str]:
    """Return Markdown path to env bindings without importing ``Config``.

    ``memo.config`` consults Markdown while its module is still importing, so
    low-level binding lookups must not construct the enriched catalog (which
    reads Pydantic field metadata from ``Config``).
    """
    from memo.flags import REGISTRY

    paths = {binding.key: binding.env_name for binding in FIELD_BINDINGS}
    paths.update(
        {_flag_path(env_name, flag.group): env_name for env_name, flag in REGISTRY.items()}
    )
    return paths


@cache
def path_to_field() -> dict[str, str]:
    return {binding.key: binding.field for binding in FIELD_BINDINGS}


def persistence_policy_for_key(key: str) -> PersistencePolicy:
    """Return the persistence policy using only cycle-safe binding metadata."""
    env_name = path_to_env().get(key)
    if env_name is None:
        raise KeyError(f"unknown config key {key!r}")
    if env_name in RUNTIME_ONLY_ENV_NAMES:
        return PersistencePolicy.RUNTIME_ONLY
    if env_name in SENSITIVE_ENV_NAMES:
        return PersistencePolicy.SECRET
    return PersistencePolicy.PERSISTENT


def setting_kind_for_key(key: str) -> SettingKind:
    """Return a setting kind without constructing the enriched catalog."""
    for binding in FIELD_BINDINGS:
        if binding.key == key:
            return binding.kind

    from memo.flags import REGISTRY

    env_name = path_to_env().get(key)
    if env_name is None:
        raise KeyError(f"unknown config key {key!r}")
    return {
        "bool": SettingKind.BOOL,
        "int": SettingKind.INT,
        "float": SettingKind.FLOAT,
        "str": SettingKind.STR,
    }[REGISTRY[env_name].kind]


def domain_file_for_key(key: str) -> str:
    for binding in FIELD_BINDINGS:
        if binding.key == key:
            return DOMAIN_TO_FILE[binding.domain]

    from memo.flags import REGISTRY

    env_name = path_to_env().get(key)
    if env_name is None:
        raise KeyError(f"unknown config key {key!r}")
    return DOMAIN_TO_FILE[GROUP_TO_DOMAIN[REGISTRY[env_name].group]]


__all__ = [
    "FieldBinding",
    "PersistencePolicy",
    "RiskLevel",
    "SettingChoice",
    "SettingKind",
    "SettingSpec",
    "Visibility",
    "build_catalog",
    "catalog_by_key",
    "domain_file_for_key",
    "path_to_env",
    "path_to_field",
    "persistence_policy_for_key",
    "setting_kind_for_key",
]
