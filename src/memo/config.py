"""Configuration — env-var-driven, file-driven, immutable.

Resolution precedence (highest first) for storage paths:

1. Explicit kwargs to `Config(...)` / `Config.from_env(...)`.
2. `MEMO_*` env vars (e.g. `MEMO_DATA_DIR`, `MEMO_VAULT_PATH`).
3. `~/.config/memo/config.toml` `[storage]` section (written by `memo init`).
4. Legacy back-compat: if `MEMO_VAULT_PATH` + `MEMO_MEMORY_SUBDIR` are set
   but `MEMO_DATA_DIR` is not, derive `data_dir = vault_path / memory_subdir`
   so existing installs keep working before they migrate.
5. Hardcoded default: `~/Documents/memo/` for `data_dir`; `vault_path` unset.

`data_dir` and `vault_path` serve **different** roles:

- `data_dir` (always set): the directory where memo's curated memories
  (the `.md` files this tool creates) live by default. Source of record.
- `vault_path` (optional): an Obsidian vault for the cross-vault
  `memo ingest` command. Non-Obsidian users leave it unset.

The `.md` files are the source of truth; sqlite is a rebuildable index
(`memo reindex --rebuild`). When `memories_in_vault` is set (and a
`vault_path` exists), `memory_dir` points INTO the vault
(`<vault>/<SYSTEM_DIR>/AI/memory`) so the human-editable Obsidian vault is
canonical. When `single_db` is set, the sidecar `*_db` path properties
collapse onto `db_path` (one sqlite file).

Pydantic v2 used for type coercion + boundary validation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from memo.errors import MemoError
from memo.model_pins import PINNED_MODEL_REVISIONS, ModelPinError, default_revision, model_spec
from memo.platform_detect import is_apple_silicon

_log = logging.getLogger(__name__)

# Default `data_dir` — visible in Finder, iCloud-syncable. The picker
# in `memo init` lets users pick a different location (Obsidian vault,
# custom path) on first run.
_DEFAULT_DATA_DIR = Path.home() / "Documents" / "memo"

# Vault system-folder layout — single source of truth. memo's own memory
# subtree and the contacts notes live under `<SYSTEM_DIR>/...` inside an
# Obsidian vault. Rename the whole subtree via MEMO_VAULT_SYSTEM_DIR or by
# editing the one default below — don't scatter the folder name as literals.
# Keep these import-time constants pure. The configured flag is resolved lazily
# by ``memory_dir``; importing flags here creates a config -> flags -> overlay ->
# config cycle before ``_DEFAULT_STATE_DIR`` exists.
SYSTEM_DIR = "Obsidian"
AI_SUBDIR = f"{SYSTEM_DIR}/AI"
CONTACTS_SUBDIR = f"{SYSTEM_DIR}/Contacts"

# Default state dir — sqlite indexes + transient state. Separate from
# `data_dir` because: (a) state is rebuildable from `.md` files via
# `memo reindex`; (b) XDG-style location signals "managed cache" so
# users don't accidentally back it up alongside their notes.
_DEFAULT_STATE_DIR = Path.home() / ".local" / "share" / "memo"

# TOML values may be hand-edited as strings ("false"), and bool("false") is True.
# Coerce truthy/falsey spellings the way the flags registry does.
_FALSEY_STRINGS = {"0", "false", "no", "off", ""}

_LLM_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
_HELPER_MODEL = "mlx-community/Qwen3-4B-4bit"
_EMBEDDER_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
_QUALITY_LLM_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510"
_QUALITY_EMBEDDER_MODEL = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
_ST_EMBEDDER_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_RERANKER_MODEL = "mku64/Qwen3-Reranker-0.6B-mlx-8Bit"

_MODEL_ROLE_FIELDS = (
    ("MEMO_LLM_MODEL", "llm_model", "MEMO_LLM_REVISION", "llm_revision"),
    ("MEMO_HELPER_MODEL", "helper_model", "MEMO_HELPER_REVISION", "helper_revision"),
    (
        "MEMO_EMBEDDER_MODEL",
        "embedder_model",
        "MEMO_EMBEDDER_REVISION",
        "embedder_revision",
    ),
    (
        "MEMO_ST_EMBEDDER_MODEL",
        "st_embedder_model",
        "MEMO_ST_EMBEDDER_REVISION",
        "st_embedder_revision",
    ),
    (
        "MEMO_RERANKER_MODEL",
        "reranker_model",
        "MEMO_RERANKER_REVISION",
        "reranker_revision",
    ),
)

_ENV_TO_FIELD = {
    "MEMO_DATA_DIR": "data_dir",
    "MEMO_VAULT_PATH": "vault_path",
    "MEMO_MEMORY_SUBDIR": "memory_subdir",
    "MEMO_STATE_DIR": "state_dir",
    "MEMO_MODEL_PROFILE": "model_profile",
    "MEMO_LLM_MODEL": "llm_model",
    "MEMO_LLM_REVISION": "llm_revision",
    "MEMO_HELPER_MODEL": "helper_model",
    "MEMO_HELPER_REVISION": "helper_revision",
    "MEMO_EMBEDDER_MODEL": "embedder_model",
    "MEMO_EMBEDDER_REVISION": "embedder_revision",
    "MEMO_EMBEDDER_DIMS": "embedder_dims",
    "MEMO_EMBEDDER_BACKEND": "embedder_backend",
    "MEMO_ST_EMBEDDER_MODEL": "st_embedder_model",
    "MEMO_ST_EMBEDDER_REVISION": "st_embedder_revision",
    "MEMO_MAX_CONTENT_CHARS": "max_content_chars",
    "MEMO_SEARCH_DEFAULT_LIMIT": "search_default_limit",
    "MEMO_RERANKER_ENABLED": "reranker_enabled",
    "MEMO_RERANKER_MODEL": "reranker_model",
    "MEMO_RERANKER_REVISION": "reranker_revision",
    "MEMO_RERANK_INPUT_K": "rerank_input_k",
    "MEMO_RERANK_FUSION_ALPHA": "rerank_fusion_alpha",
}

_STORAGE_FIELDS = {
    "data_dir",
    "vault_path",
    "memory_subdir",
    "state_dir",
    "memories_in_vault",
    "single_db",
}

_EMBEDDER_PIN_FIELDS = {
    "embedder_model",
    "embedder_revision",
    "embedder_dims",
    "model_profile",
}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in _FALSEY_STRINGS


MODEL_PROFILES: dict[str, dict[str, object]] = {
    # Lowest operational footprint: semantic search still works, but hybrid
    # search skips the cross-encoder so recall-hook latency stays predictable.
    "light": {
        "llm_model": _LLM_MODEL,
        "llm_revision": PINNED_MODEL_REVISIONS[_LLM_MODEL],
        "helper_model": _HELPER_MODEL,
        "helper_revision": PINNED_MODEL_REVISIONS[_HELPER_MODEL],
        "embedder_model": _EMBEDDER_MODEL,
        "embedder_revision": PINNED_MODEL_REVISIONS[_EMBEDDER_MODEL],
        "embedder_dims": 1024,
        "reranker_enabled": False,
    },
    # Default: small embedder + reranker. Good quality/latency balance for a
    # local MCP memory store whose hooks may run on every prompt.
    "balanced": {
        "llm_model": _LLM_MODEL,
        "llm_revision": PINNED_MODEL_REVISIONS[_LLM_MODEL],
        "helper_model": _HELPER_MODEL,
        "helper_revision": PINNED_MODEL_REVISIONS[_HELPER_MODEL],
        "embedder_model": _EMBEDDER_MODEL,
        "embedder_revision": PINNED_MODEL_REVISIONS[_EMBEDDER_MODEL],
        "embedder_dims": 1024,
        "reranker_enabled": True,
        "reranker_model": _RERANKER_MODEL,
        "reranker_revision": PINNED_MODEL_REVISIONS[_RERANKER_MODEL],
    },
    # Higher retrieval quality. Requires a full reindex because the 4B
    # embedder emits 2560-dim vectors instead of 1024.
    "quality": {
        "llm_model": _QUALITY_LLM_MODEL,
        "llm_revision": PINNED_MODEL_REVISIONS[_QUALITY_LLM_MODEL],
        "helper_model": _HELPER_MODEL,
        "helper_revision": PINNED_MODEL_REVISIONS[_HELPER_MODEL],
        "embedder_model": _QUALITY_EMBEDDER_MODEL,
        "embedder_revision": PINNED_MODEL_REVISIONS[_QUALITY_EMBEDDER_MODEL],
        "embedder_dims": 2560,
        "reranker_enabled": True,
        "reranker_model": _RERANKER_MODEL,
        "reranker_revision": PINNED_MODEL_REVISIONS[_RERANKER_MODEL],
    },
}


def _index_embedder_profile(db_path: Path) -> tuple[str, int] | None:
    """Read the (embedder_model, dims) the existing index was built with.

    The vector index is self-describing: `schema_meta` records the builder's
    `embedder_model` + `embedder_dims` (0.8.0+), and the vec0 table encodes its
    dimensionality in `FLOAT[N]`. When the model is absent (pre-`schema_meta`
    index) it is derived from the dims via `MODEL_PROFILES` (unambiguous:
    1024→0.6B, 2560→4B). Returns None when the DB is absent/empty or nothing
    usable is recorded.

    Best-effort and read-only: any error → None, so it never blocks Config
    construction. A short-lived RO connection is WAL-safe alongside the store.
    """
    import re
    import sqlite3

    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        conn.row_factory = sqlite3.Row
        model = ""
        dims = 0
        try:
            rows = conn.execute(
                "SELECT key, value FROM schema_meta "
                "WHERE key IN ('embedder_model', 'embedder_dims')"
            ).fetchall()
            kv = {str(r["key"]): str(r["value"]) for r in rows}
            model = kv.get("embedder_model", "") or ""
            dims = int(kv.get("embedder_dims", "0") or "0")
        except (sqlite3.Error, ValueError):
            model, dims = "", 0
        if dims <= 0:
            # Fallback: read dimensionality straight off the vec0 table DDL.
            try:
                row = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec'"
                ).fetchone()
            except sqlite3.Error:
                row = None
            if row and row["sql"]:
                m = re.search(r"FLOAT\[(\d+)\]", str(row["sql"]))
                if m:
                    dims = int(m.group(1))
        if dims <= 0:
            return None
        if not model or "stub" in model.lower():
            # Derive the model from dims when unrecorded or a test stub.
            model = next(
                (
                    str(p["embedder_model"])
                    for p in MODEL_PROFILES.values()
                    if p.get("embedder_dims") == dims
                ),
                "",
            )
        return (model, dims) if model else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _apply_selected_values(kwargs: dict[str, Any], values: dict[str, Any]) -> None:
    """Apply non-empty values and pair overridden models with known pins."""
    for field, value in values.items():
        if value is not None and value != "":
            kwargs[field] = value
    selected = values.keys()
    for _model_env, model_field, _revision_env, revision_field in _MODEL_ROLE_FIELDS:
        if model_field in selected and revision_field not in selected:
            kwargs[revision_field] = default_revision(str(kwargs[model_field]))


def _apply_explicit_overrides(kwargs: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply explicit kwargs verbatim, including intentional empty values."""
    kwargs.update(overrides)
    for _model_env, model_field, _revision_env, revision_field in _MODEL_ROLE_FIELDS:
        if model_field in overrides and revision_field not in overrides:
            kwargs[revision_field] = default_revision(str(kwargs[model_field]))


def _file_config_values(file_data: dict[str, Any]) -> dict[str, Any]:
    """Return the recognized legacy TOML values with booleans normalized."""
    storage = file_data.get("storage") or {}
    values = {
        field: storage[field]
        for field in ("data_dir", "vault_path", "memory_subdir", "state_dir")
        if storage.get(field)
    }
    for field in ("memories_in_vault", "single_db"):
        if storage.get(field) is not None:
            values[field] = _coerce_bool(storage[field])
    return values


def _resolve_model_profile(
    kwargs: dict[str, Any], overrides: dict[str, Any], md_values: dict[str, Any]
) -> None:
    """Install the selected profile baseline, then restore Markdown choices."""
    profile = (
        str(
            overrides.get("model_profile")
            or os.environ.get("MEMO_MODEL_PROFILE")
            or kwargs.get("model_profile")
            or "balanced"
        )
        .strip()
        .lower()
    )
    if profile not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown MEMO_MODEL_PROFILE={profile!r}; valid: {sorted(MODEL_PROFILES)}",
        )
    kwargs.update(MODEL_PROFILES[profile])
    kwargs["model_profile"] = profile
    _apply_selected_values(kwargs, md_values)


def _reject_relative_store_env() -> None:
    """Refuse a relative store path supplied through the environment.

    An env-supplied store path is meant to name one exact directory, but a
    relative one resolves against whatever cwd the process happens to inherit
    — and the consequences diverge silently. A stale
    ``MEMO_DATA_DIR="sweep/store_cli_2/data"`` in an MCP client's env resolved
    under the home directory, so memo created a NEW EMPTY store there and every
    MCP tool answered from an empty corpus while the CLI kept reading the real
    one; the same string in a launchd plist (cwd ``/``) hit a read-only
    filesystem and crash-looped with a readable error. Same misconfiguration,
    and only the inherited cwd decided whether it was loud or invisible.

    Scoped to the environment on purpose: ``_apply_repo_and_legacy_paths``
    still sets relative ``data_dir``/``state_dir`` defaults for the zero-config
    in-repo case, where the cwd IS the directory being described.
    """
    for env_key in ("MEMO_DATA_DIR", "MEMO_STATE_DIR"):
        value = os.environ.get(env_key)
        if not value:
            continue
        if not Path(value).expanduser().is_absolute():
            raise MemoError(
                f"{env_key}={value!r} is a relative path. It resolves against the "
                "current working directory, which differs per client (a launchd "
                "agent runs at '/'), so it can silently point at a different — or "
                "empty — store. Use an absolute path (or a leading '~')."
            )


def _apply_environment_values(kwargs: dict[str, Any], md_values: dict[str, Any]) -> None:
    """Apply registered environment overrides and platform-aware toggles."""
    _reject_relative_store_env()
    env_values = {
        field: value
        for env_key, field in _ENV_TO_FIELD.items()
        if (value := os.environ.get(env_key)) is not None and value != ""
    }
    _apply_selected_values(kwargs, env_values)

    # Hardware guard: the cross-encoder reranker needs MLX (Apple Silicon), so
    # default it OFF elsewhere. Honor an explicit opt-in from EITHER the env var
    # OR persistent Markdown config (`models.reranker_enabled`, carried in
    # md_values) — only force it off when the operator asserted nothing. Env
    # presence alone was checked before, which silently clobbered a
    # markdown-persisted `reranker_enabled = true` on non-Apple-Silicon hosts.
    reranker_opt_in = "MEMO_RERANKER_ENABLED" in os.environ or "reranker_enabled" in md_values
    if not is_apple_silicon() and not reranker_opt_in:
        kwargs["reranker_enabled"] = False

    from memo.flags import flag_bool

    # Resolve through the 4-tier flag resolver (env > Markdown > overlay >
    # default) so the tuned overlay is honored too — the old raw-env gate
    # consulted flag_bool ONLY when the env was truthy, so an overlay value was
    # ignored when the env was unset. Markdown field values are already applied
    # via md_values above, and the legacy config.toml tier (which flag_bool does
    # not read) is preserved by only overwriting when a flag tier actually
    # asserts a value — an unset flag resolves to its False default and must not
    # clobber a True supplied by config.toml.
    for env_name, field in (
        ("MEMO_MEMORIES_IN_VAULT", "memories_in_vault"),
        ("MEMO_SINGLE_DB", "single_db"),
    ):
        resolved = flag_bool(env_name)
        if resolved or env_name in os.environ or field in md_values:
            kwargs[field] = resolved


def _apply_repo_and_legacy_paths(
    kwargs: dict[str, Any], file_data: dict[str, Any], md_values: dict[str, Any]
) -> None:
    """Apply zero-config repo paths, then the legacy vault-derived data dir."""
    cwd_is_repo = (Path.cwd() / "src" / "memo" / "__init__.py").is_file()
    has_data_env = "MEMO_DATA_DIR" in os.environ
    has_state_env = "MEMO_STATE_DIR" in os.environ
    has_legacy = "MEMO_VAULT_PATH" in os.environ or os.environ.get("MEMO_MEMORY_SUBDIR")
    has_storage_config = bool(file_data.get("storage")) or bool(_STORAGE_FIELDS & md_values.keys())
    if cwd_is_repo and not has_data_env and not has_legacy and not has_storage_config:
        kwargs["data_dir"] = "memorias" if (Path.cwd() / "memorias").is_dir() else "memories"
    if cwd_is_repo and not has_state_env and not has_legacy and not has_storage_config:
        kwargs["state_dir"] = ".memo-state"

    if "data_dir" not in kwargs:
        vault_path = kwargs.get("vault_path")
        memory_subdir = kwargs.get("memory_subdir")
        if vault_path and memory_subdir:
            kwargs["data_dir"] = str(Path(vault_path).expanduser() / memory_subdir)


def _validate_model_pins(cfg: Any) -> None:
    """Reject mutable remote model inputs at the production boundary."""
    for model_env, model_field, _revision_env, revision_field in _MODEL_ROLE_FIELDS:
        model_value = str(getattr(cfg, model_field))
        if "stub" in model_value.lower():
            continue
        try:
            model_spec(model_value, getattr(cfg, revision_field))
        except ModelPinError as exc:
            raise MemoError(f"{model_env} is not immutable: {exc}") from exc


def _embedder_was_pinned(md_values: dict[str, Any], overrides: dict[str, Any]) -> bool:
    return bool(
        os.environ.get("MEMO_EMBEDDER_MODEL")
        or os.environ.get("MEMO_EMBEDDER_REVISION")
        or os.environ.get("MEMO_EMBEDDER_DIMS")
        or os.environ.get("MEMO_MODEL_PROFILE")
        or _EMBEDDER_PIN_FIELDS & md_values.keys()
        or _EMBEDDER_PIN_FIELDS & overrides.keys()
    )


def _reconcile_index_embedder(cfg: Config, *, pinned: bool) -> Config:
    """Adopt an unpinned index profile, or warn about explicit dim drift."""
    adopted = _index_embedder_profile(cfg.db_path)
    if adopted is None:
        return cfg
    model, dims = adopted
    if pinned:
        if dims and dims != cfg.embedder_dims:
            _log.warning(
                "embedder dimension mismatch: config expects %dd but index was built with %dd. "
                "This usually happens after switching model profiles without reindexing. "
                "Run 'memo reindex --rebuild' to rebuild the index with the new dimensions.",
                cfg.embedder_dims,
                dims,
            )
        return cfg
    if not model or (model == cfg.embedder_model and dims == cfg.embedder_dims):
        return cfg
    _log.info(
        "adopting index embedder profile %s (%dd) over default "
        "%s (%dd) — set MEMO_EMBEDDER_MODEL/DIMS to override",
        model,
        dims,
        cfg.embedder_model,
        cfg.embedder_dims,
    )
    revision = default_revision(model)
    if revision is None:
        raise MemoError(
            "The existing index records an unpinned remote embedder "
            f"{model!r}. Set MEMO_EMBEDDER_REVISION to its exact "
            "40-character commit SHA, then run 'memo reindex --rebuild'."
        )
    return cfg.model_copy(
        update={
            "embedder_model": model,
            "embedder_revision": revision,
            "embedder_dims": dims,
        }
    )


class Config(BaseModel):
    """Process-wide configuration.

    Construct via `Config.from_env()` in production. Tests can build a
    `Config(...)` with explicit overrides.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def _pair_explicit_models_with_known_revisions(cls, value: Any) -> Any:
        """Do not attach one role's default SHA to an overridden model.

        Direct ``Config(model=...)`` construction is common in tests and
        integrations. Known shipped models get their audited pin; custom/local
        models stay unset until an exact revision is supplied (or the path is
        loaded locally).
        """
        if not isinstance(value, dict):
            return value
        paired = dict(value)
        for _model_env, model_field, _revision_env, revision_field in _MODEL_ROLE_FIELDS:
            if model_field in paired and revision_field not in paired:
                paired[revision_field] = default_revision(str(paired[model_field]))
        return paired

    # ── Storage ──────────────────────────────────────────────────────────
    data_dir: Path = Field(
        default=_DEFAULT_DATA_DIR,
        description=(
            "Directory where memo's curated memory `.md` files live. "
            "Source of record. Defaults to `~/Documents/memo/`. "
            "Override via `MEMO_DATA_DIR` or `~/.config/memo/config.toml`."
        ),
    )
    vault_path: Path | None = Field(
        default=None,
        description=(
            "Optional Obsidian vault root, used ONLY by `memo ingest` for "
            "cross-vault corpus indexing. Non-Obsidian users leave this unset. "
            "Set via `MEMO_VAULT_PATH` or by picking an Obsidian vault in "
            "`memo init`."
        ),
    )
    memory_subdir: str = Field(
        default="",
        description=(
            "DEPRECATED. Kept for legacy back-compat: if both `MEMO_VAULT_PATH` "
            "and `MEMO_MEMORY_SUBDIR` are set, `data_dir` is derived from "
            "`vault_path / memory_subdir`. New installs use `data_dir` directly."
        ),
    )
    state_dir: Path = Field(
        default=_DEFAULT_STATE_DIR,
        description="Where sqlite-vec DB + transient state live. Created if missing.",
    )
    memories_in_vault: bool = Field(
        default=False,
        description=(
            "When True AND `vault_path` is set, curated memory `.md` files live "
            "INSIDE the Obsidian vault (`<vault>/<SYSTEM_DIR>/AI/memory`) rather "
            "than in `data_dir` — the vault becomes the human-editable source of "
            "truth and sqlite a rebuildable index. Toggled via "
            "`MEMO_MEMORIES_IN_VAULT`. Ignored when `vault_path` is unset."
        ),
    )
    single_db: bool = Field(
        default=False,
        description=(
            "When True, the sidecar stores (history/graph/contradictions/"
            "crossref) live in the single main DB file (`memvec.db`) instead of "
            "separate `*.db` files. The `*_db` path properties collapse onto "
            "`db_path`. Toggled via `MEMO_SINGLE_DB`; run `memo migrate "
            "--consolidate-db` to merge any existing sidecar files first."
        ),
    )

    # ── MLX models ───────────────────────────────────────────────────────
    model_profile: str = Field(
        default="balanced",
        description=(
            "Named model bundle. `light` lowers latency, `balanced` is the "
            "default, `quality` uses a larger embedder and requires reindex."
        ),
    )
    llm_model: str = Field(
        default=_LLM_MODEL,
        description="HF id of the chat model used for synthesis / consolidation.",
    )
    llm_revision: str | None = Field(
        default=PINNED_MODEL_REVISIONS[_LLM_MODEL],
        description="Exact Hugging Face commit SHA for the chat model.",
    )
    helper_model: str = Field(
        default=_HELPER_MODEL,
        description=(
            "Smaller model used for deterministic helper tasks (tag suggestion, "
            "title extraction, dedup). `temperature=0`, `seed=42` enforced."
        ),
    )
    helper_revision: str | None = Field(
        default=PINNED_MODEL_REVISIONS[_HELPER_MODEL],
        description="Exact Hugging Face commit SHA for the helper model.",
    )
    embedder_model: str = Field(
        default=_EMBEDDER_MODEL,
        description="HF id of the embedder. Must yield 1024-dim L2-normalised vectors.",
    )
    embedder_revision: str | None = Field(
        default=PINNED_MODEL_REVISIONS[_EMBEDDER_MODEL],
        description="Exact Hugging Face commit SHA for the MLX embedder.",
    )
    embedder_dims: int = Field(
        default=1024,
        ge=2,
        description=(
            "Embedding dimensionality. Asserted at runtime against the loaded "
            "model. Tests may use very low values (e.g. 4) — that's why the "
            "lower bound is permissive; production swap to a different real "
            "model is gated by the runtime dim check in `MLXEmbedder.embed`."
        ),
    )
    embedder_backend: str = Field(
        default="auto",
        description=(
            "Embedder backend: 'auto' (MLX on Apple Silicon, else CPU "
            "sentence-transformers), 'mlx', or 'st'. See `embedder_select`."
        ),
    )
    st_embedder_model: str = Field(
        default=_ST_EMBEDDER_MODEL,
        description=(
            "HF id for the CPU (sentence-transformers) backend. Same family/dims "
            "(1024) as the default MLX quant, so the vec0 schema is unchanged. "
            "Used on Linux/Ubuntu."
        ),
    )
    st_embedder_revision: str | None = Field(
        default=PINNED_MODEL_REVISIONS[_ST_EMBEDDER_MODEL],
        description=(
            "Exact Hugging Face commit SHA for the CPU sentence-transformers "
            "embedder. Set via MEMO_ST_EMBEDDER_REVISION when overriding its model."
        ),
    )

    # ── Reranker ─────────────────────────────────────────────────────────
    # Cross-encoder applied AFTER hybrid retrieval when enabled. Lifts
    # MRR materially on diffuse queries at the cost of one forward pass
    # per candidate. Same Qwen3 family as the embedder so no extra
    # tokenizer/architecture surface is added.
    reranker_enabled: bool = Field(
        default_factory=is_apple_silicon,
        description=(
            "When True, hybrid-mode searches fetch a wider candidate "
            "set and rerank via the cross-encoder. Defaults ON on Apple "
            "Silicon and OFF elsewhere (the reranker is MLX-only); "
            "set MEMO_RERANKER_ENABLED to override."
        ),
    )
    reranker_model: str = Field(
        default=_RERANKER_MODEL,
        description=(
            "HF id of the MLX-quantised reranker. Apache 2.0 by default "
            "(Qwen3-Reranker family). For higher recall at higher cost: "
            "`vserifsaglam/Qwen3-Reranker-4B-4bit-MLX`."
        ),
    )
    reranker_revision: str | None = Field(
        default=PINNED_MODEL_REVISIONS[_RERANKER_MODEL],
        description=(
            "Exact Hugging Face commit SHA for the reranker. Set via "
            "MEMO_RERANKER_REVISION when overriding its model."
        ),
    )
    rerank_input_k: int = Field(
        default=30,
        ge=1,
        le=200,
        description=(
            "How many hybrid-fusion candidates to feed the reranker. A "
            "cross-encoder can only PROMOTE what it is handed — at 5 a correct "
            "memory buried at fusion rank 6+ is never seen. 30 lets the warm "
            "0.6B reranker (~20ms/pair, ~0.6s for 30) rescue buried hits while "
            "staying well inside the 5s recall budget; the adaptive pool "
            "(MEMO_RERANK_ADAPTIVE_POOL) widens further on diffuse queries. "
            "Override with MEMO_RERANK_INPUT_K; lower it for cold one-shot CLI "
            "on the heavier 4B reranker."
        ),
    )
    rerank_fusion_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Weight given to the reranker score in the final ranking. "
            "Final score = alpha * rerank + (1-alpha) * rrf_position_bonus. "
            "1.0 = pure rerank (vulnerable to cross-encoder false "
            "positives that the bi-encoder rejected). 0.0 = pure RRF "
            "(skips rerank signal entirely). 0.7 lets the reranker "
            "lead while keeping RRF as a structural sanity check."
        ),
    )

    # ── Limits ───────────────────────────────────────────────────────────
    max_content_chars: int = Field(
        default=64_000,
        ge=1,
        description=(
            "Truncate raw content above this CHARACTER count before embed + "
            "disk write (prevents OOM on huge dumps). Note: chars, not bytes — "
            "in UTF-8 with multi-byte glyphs (Spanish accents, emoji), the "
            "actual byte size will be larger. 64K chars ≈ 96KB in real "
            "Spanish text. Tests use small caps (e.g. 100) to verify the "
            "truncation path."
        ),
    )
    search_default_limit: int = Field(default=10, ge=1, le=100)

    # ── Pydantic validators ──────────────────────────────────────────────

    @field_validator("data_dir", "state_dir", "vault_path", mode="before")
    @classmethod
    def _expand(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return v
        return Path(os.path.expandvars(str(v))).expanduser().resolve()

    @field_validator("model_profile")
    @classmethod
    def _valid_profile(cls, v: str) -> str:
        profile = (v or "balanced").strip().lower()
        if profile not in MODEL_PROFILES:
            raise ValueError(
                f"Unknown MEMO_MODEL_PROFILE={v!r}; valid: {sorted(MODEL_PROFILES)}",
            )
        return profile

    # ── Derived paths ────────────────────────────────────────────────────

    @property
    def device_id(self) -> str:
        """Unique ID for this device, persisted in state_dir."""
        id_path = self.state_dir / ".device_id"
        if id_path.is_file():
            try:
                existing = id_path.read_text(encoding="utf-8").strip()
                if existing:
                    return existing
            except Exception as exc:
                _log.debug("config: failed to read device_id from %s: %s", id_path, exc)

        import time
        import uuid

        new_id = str(uuid.uuid4()).replace("-", "")[:12]
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            # Atomic exclusive publish: write a unique tmp file, then hard-link
            # it to the final path. A plain write_text was check-then-write —
            # concurrent first runs minted divergent ids (last write wins) and
            # readers in the truncate window saw ''. os.link fails with
            # FileExistsError when another process won the race; adopt its id.
            tmp_path = id_path.with_name(f".device_id.{os.getpid()}-{new_id}.tmp")
            tmp_path.write_text(new_id, encoding="utf-8")
            try:
                os.link(tmp_path, id_path)
            except FileExistsError:
                # Lost the mint race — the other process's id is canonical.
                for _ in range(2):
                    existing = id_path.read_text(encoding="utf-8").strip()
                    if existing:
                        return existing
                    time.sleep(0.05)
                raise  # unreadable/empty winner — fall through to transient
            finally:
                tmp_path.unlink(missing_ok=True)
        except Exception:
            # If we can't write, return a transient ID
            device_id = f"transient-{new_id}"
            _log.warning(
                "state_dir %s not writable — device_id is transient (%s). "
                "Sync and federation will not work correctly.",
                self.state_dir,
                device_id,
            )
            return device_id
        return new_id

    @property
    def memory_dir(self) -> Path:
        """Absolute path of the directory holding memory `.md` files.

        Default: equal to `data_dir`. When `memories_in_vault` is on AND a
        `vault_path` is configured, memories instead live under
        `<vault>/<SYSTEM_DIR>/AI/memory` so the Obsidian vault is the
        human-editable source of truth (git/iCloud synced). The property
        exists so callers don't track which field is authoritative.

        Note: `AI_SUBDIR` already includes `SYSTEM_DIR` (e.g. `Obsidian/AI`),
        so this resolves to `<vault>/Obsidian/AI/memory`. `memo ingest`
        excludes both `AI/` and any `id:`-frontmatter file, so memories
        placed here are never re-ingested as reference-tier rows.
        """
        if self.memories_in_vault and self.vault_path is not None:
            # Read MEMO_VAULT_SYSTEM_DIR at call time (not the frozen module-level
            # AI_SUBDIR) so a flag set after import is honored. Same default as
            # the module constant.
            from memo.flags import flag_str

            system_dir = flag_str("MEMO_VAULT_SYSTEM_DIR") or "Obsidian"
            return self.vault_path / system_dir / "AI" / "memory"
        return self.data_dir

    @property
    def db_path(self) -> Path:
        """sqlite-vec DB file path. Single file, single writer."""
        return self.state_dir / "memvec.db"

    @property
    def history_db(self) -> Path:
        """History/audit DB. Separate file by default (vec writes don't share
        its WAL); collapses onto `db_path` when `single_db` is set."""
        return self.db_path if self.single_db else self.state_dir / "history.db"

    @property
    def graph_db(self) -> Path:
        """Knowledge-graph DB (entities + entity_memory edges).
        Collapses onto `db_path` when `single_db` is set."""
        return self.db_path if self.single_db else self.state_dir / "graph.db"

    @property
    def crossref_db(self) -> Path:
        """Cross-reference DB (wikilinks + backlinks).
        Collapses onto `db_path` when `single_db` is set."""
        return self.db_path if self.single_db else self.state_dir / "crossref.db"

    @property
    def contradictions_db(self) -> Path:
        """Sidecar DB for persisted contradiction pairs + triage status.
        Collapses onto `db_path` when `single_db` is set."""
        return self.db_path if self.single_db else self.state_dir / "contradictions.db"

    @property
    def episode_db(self) -> Path:
        """Derived semantic index over work sessions (episodic memory).
        Rebuildable from transcripts; collapses onto `db_path` under `single_db`."""
        return self.db_path if self.single_db else self.state_dir / "episodes.db"

    @property
    def fact_edges_db(self) -> Path:
        """Temporal fact-edge sidecar DB.
        Rebuildable from markdown-derived extraction; collapses onto `db_path`."""
        return self.db_path if self.single_db else self.state_dir / "fact_edges.db"

    @property
    def verbatim_db(self) -> Path:
        """Lexical turn-level FTS5 sidecar. Rebuildable from transcripts; collapses onto db_path under single_db."""
        return self.db_path if self.single_db else self.state_dir / "verbatim.db"

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, **overrides: Any) -> Config:
        """Build a `Config` from env vars + config file, with optional kwargs.

        Resolution order (highest first):
          1. Explicit kwargs.
          2. `MEMO_*` env vars.
          3. `~/.config/memo/config.toml` `[storage]` section.
          4. Legacy back-compat for `data_dir` (see module docstring).
          5. Hardcoded defaults.
        """
        # Step 1: gather TOML config file values (lowest priority of the
        # three explicit sources).
        from memo.setup.config_io import load_config_file

        file_data = load_config_file() or {}
        file_data = file_data if isinstance(file_data, dict) else {}
        kwargs = _file_config_values(file_data)

        from memo.config_md import field_values as _markdown_field_values

        md_values = _markdown_field_values(os.environ)
        # Markdown config is the persistent source of truth. It wins over
        # legacy config.toml and loses to MEMO_* env vars applied below.
        _apply_selected_values(kwargs, md_values)

        # Step 2: model profile defaults. Individual env vars below
        # intentionally override profile choices.
        _resolve_model_profile(kwargs, overrides, md_values)
        # Profile defaults are a bundle baseline. Re-apply Markdown after the
        # bundle so persistent per-field choices (for example a custom
        # embedder under the quality profile) are not lost. Env vars below
        # still win over both Markdown and profile defaults.
        # Step 3: env-var overrides.
        _apply_environment_values(kwargs, md_values)

        # Step 4: zero-config repo mode — detect if we're running from
        # a cloned memo repo (cwd contains memo's source). In that case,
        # override any prior config and use paths relative to cwd so the
        # clone works out-of-the-box. Skip if user explicitly set
        # MEMO_DATA_DIR or has a legacy vault_path setup OR has an existing
        # config file (opt-in to repo mode is implied by not setting these).
        _apply_repo_and_legacy_paths(kwargs, file_data, md_values)

        # Step 6: explicit overrides win over everything.
        _apply_explicit_overrides(kwargs, overrides)
        cfg = cls(**kwargs)

        # Remote weights are executable supply-chain inputs. Validate only at
        # this production construction boundary so low-level tests can still
        # use clearly synthetic model names without network access.
        _validate_model_pins(cfg)

        # Self-describing index: adopt the embedder the existing index was built
        # with UNLESS the operator explicitly pinned one. MCP clients don't
        # inherit the shell env (~/.zshenv), so a bare `MEMO_NONINTERACTIVE=1`
        # launch falls back to the default profile (0.6B/1024) and hard-crashes
        # against a non-default-dims index — surfacing in the client as an opaque
        # "connection closed" during the MCP handshake (hit on both Claude Code
        # and Codex configs). The index records its builder; trust it when the
        # operator didn't say otherwise.
        cfg = _reconcile_index_embedder(
            cfg,
            pinned=_embedder_was_pinned(md_values, overrides),
        )
        # NB: do NOT validate the reranker model here. `Config.from_env()` must
        # stay hermetic (no network) — it's called in every test and CLI start.
        # MLXReranker._ensure_loaded() validates via snapshot_download on first
        # use, which is the right place for the network round-trip.
        return cfg

    def ensure_dirs(self) -> None:
        """Create state + data dirs if missing.

        Does NOT validate `vault_path` — that's only used by `memo ingest`
        and the ingest command does its own check. Non-Obsidian users
        never need a `vault_path`.

        Raises `RuntimeError` with a human-readable message if any directory
        cannot be created (permission denied, read-only FS, etc.) so callers
        see "data_dir is not writable" rather than a raw OSError from deep in
        pathlib.
        """
        for attr, path in (
            ("state_dir", self.state_dir),
            ("data_dir", self.data_dir),
            ("memory_dir", self.memory_dir),
        ):
            try:
                path.mkdir(parents=True, exist_ok=True)
                if attr == "state_dir":
                    # The SQLite index contains full memory bodies (FTS),
                    # operational logs, and optionally encrypted secret rows.
                    # A normal 022 umask would otherwise leave it traversable
                    # by other local users and expose daemon sockets as well.
                    path.chmod(0o700)
            except OSError as exc:
                raise RuntimeError(
                    f"memo: cannot create {attr} at {path}: {exc}\n"
                    f"Check filesystem permissions or set MEMO_DATA_DIR / MEMO_STATE_DIR "
                    f"to a writable location."
                ) from exc
