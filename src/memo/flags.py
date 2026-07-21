"""Central registry for `MEMO_*` feature/tuning environment flags.

One documented source of truth for every behavioral env var memo reads.
Storage/model config lives in `config.py` (typed `Config` model); this
module covers the ~60 *behavioral* flags that were historically read inline
via scattered `os.environ.get("MEMO_...")` calls with per-call-site defaults.

Each flag is a `FlagSpec` (kind + default + group + help). Use the typed
accessors — `flag_bool`, `flag_int`, `flag_float`, `flag_str` — or the
generic `flag(name)` which coerces by the registered kind. `validate()`
parses every *set* flag and reports misconfiguration; `active_flags()`
lists which are currently set in the environment. The `memo config`
command group surfaces both.

Flag specs are split across domain modules to keep each file under 800 lines:
  flags_recall.py   — recall hook / daemon flags
  flags_search.py   — search ranking flags
  flags_behavior.py — entity, session, capture, maintenance, synthesis
  flags_ingest.py   — transcript, briefing, repo indexing
  flags_misc.py     — embedder, feedback, MCP, synapse, cache, misc, ROI, WhatsApp, schema
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from memo.errors import ValidationError as MemoValidationError

# Re-export base types so existing `from memo.flags import FlagSpec` imports still work.
from memo.flags_base import _FALSE, _TRUE, FlagKind, FlagSpec, _spec  # noqa: F401
from memo.flags_behavior import SPECS as _behavior_specs
from memo.flags_capture import SPECS as _capture_specs
from memo.flags_ingest import SPECS as _ingest_specs
from memo.flags_misc import SPECS as _misc_specs
from memo.flags_recall import SPECS as _recall_specs
from memo.flags_search import SPECS as _search_specs

_log = logging.getLogger(__name__)

# ── Registry ────────────────────────────────────────────────────────────────
_SPECS: tuple[FlagSpec, ...] = (
    _recall_specs + _search_specs + _behavior_specs + _capture_specs + _ingest_specs + _misc_specs
)

REGISTRY: dict[str, FlagSpec] = {s.name: s for s in _SPECS}
_MISSING = object()


def _coerce_bool(raw: str) -> bool:
    low = raw.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(f"expected a boolean (1/0/true/false), got {raw!r}")


def _validate_range(spec: FlagSpec, value: int | float) -> int | float:
    if spec.min_val is not None and value < spec.min_val:
        raise ValueError(f"{spec.name} must be >= {spec.min_val}, got {value}")
    if spec.max_val is not None and value > spec.max_val:
        raise ValueError(f"{spec.name} must be <= {spec.max_val}, got {value}")
    return value


def _coerce_choice(spec: FlagSpec, raw: str) -> str:
    value = raw.strip().lower()
    if value not in spec.choices:
        allowed = ", ".join(spec.choices)
        raise ValueError(f"expected one of: {allowed}, got {raw!r}")
    return value


def _coerce(spec: FlagSpec, raw: str) -> Any:
    """Parse `raw` per the spec's kind. Raises ValueError on bad input."""
    if spec.kind == "bool":
        return _coerce_bool(raw)
    if spec.kind == "int":
        return _validate_range(spec, int(raw.strip()))
    if spec.kind == "float":
        return _validate_range(spec, float(raw.strip()))
    if spec.choices:
        return _coerce_choice(spec, raw)
    return raw  # str


def _coerce_configured_value(
    spec: FlagSpec,
    raw: str,
    *,
    strict: bool,
    strict_message: str,
    warning_message: str,
    warning_args: tuple[Any, ...],
) -> Any:
    try:
        return _coerce(spec, raw)
    except ValueError as exc:
        if strict:
            raise MemoValidationError(strict_message.format(raw=raw, exc=exc)) from exc
        _log.warning(warning_message, *warning_args)
        return _MISSING


def _flag_without_env(
    spec: FlagSpec,
    name: str,
    src: Mapping[str, str],
    *,
    strict: bool,
) -> Any:
    from memo.config_md import flag_values as _markdown_flag_values
    from memo.tuned_overlay import overlay_values

    md = _markdown_flag_values(src)
    if name in md:
        value = _coerce_configured_value(
            spec,
            md[name],
            strict=strict,
            strict_message=f"invalid value for {name} in Markdown config: {{raw!r}} ({{exc}})",
            warning_message="invalid markdown config value for %s: %r - checking overlay/default",
            warning_args=(name, md[name]),
        )
        if value is not _MISSING:
            return value

    ov = overlay_values(src)
    if name in ov:
        value = _coerce_configured_value(
            spec,
            ov[name],
            strict=strict,
            strict_message=f"invalid value for {name} in tuned overlay: {{raw!r}} ({{exc}})",
            warning_message="invalid overlay value for %s: %r — using default %r",
            warning_args=(name, ov[name], spec.default),
        )
        if value is not _MISSING:
            return value
    return spec.default


def _coerce_env_value(spec: FlagSpec, name: str, raw: str, *, strict: bool) -> Any:
    try:
        return _coerce(spec, raw)
    except ValueError as exc:
        if strict:
            raise MemoValidationError(f"invalid value for {name}: {raw!r} ({exc})") from exc
        _log.warning(
            "invalid value for %s: %r — using default %r (%s)",
            name,
            raw,
            spec.default,
            exc,
        )
        return spec.default


def flag(name: str, *, env: dict[str, str] | None = None, strict: bool = False) -> Any:
    """Return the typed, parsed value for `name`, or its default if unset.

    Unknown flags raise KeyError — every flag must be registered above.
    """
    spec = REGISTRY[name]
    src = os.environ if env is None else env
    raw = src.get(name)
    if raw is None:
        return _flag_without_env(spec, name, src, strict=strict)
    if raw == "":
        # Empty string counts as unset except for str flags whose default is also "".
        # "Unset" means the full fallback chain (markdown config > overlay >
        # default), not a shortcut straight to the built-in default — an
        # `MEMO_X=` export must not silently mask `memo config set` values.
        if spec.kind == "str" and spec.default == "":
            return ""
        return _flag_without_env(spec, name, src, strict=strict)
    return _coerce_env_value(spec, name, raw, strict=strict)


def flag_bool(
    name: str,
    *,
    env: dict[str, str] | None = None,
    strict: bool = False,
) -> bool:
    return bool(flag(name, env=env, strict=strict))


def flag_int(
    name: str,
    *,
    env: dict[str, str] | None = None,
    strict: bool = False,
) -> int | None:
    v = flag(name, env=env, strict=strict)
    return None if v is None else int(v)


def flag_float(
    name: str,
    *,
    env: dict[str, str] | None = None,
    strict: bool = False,
) -> float | None:
    v = flag(name, env=env, strict=strict)
    return None if v is None else float(v)


def flag_str(
    name: str,
    *,
    env: dict[str, str] | None = None,
    strict: bool = False,
) -> str:
    v = flag(name, env=env, strict=strict)
    return "" if v is None else str(v)


def active_flags(env: dict[str, str] | None = None) -> dict[str, str]:
    """Registered flags currently set (non-empty) in the environment."""
    src = os.environ if env is None else env
    return {n: src[n] for n in REGISTRY if src.get(n)}


def active_config_values(env: dict[str, str] | None = None) -> dict[str, str]:
    """Registered Markdown config values currently set.

    Unlike active_flags(), this is not environment state. It is explicit
    persistent config loaded from memo's Markdown config directory.
    """
    from memo.config_md import active_config_values as _active_config_values

    return _active_config_values(os.environ if env is None else env)


def unknown_memo_vars(env: dict[str, str] | None = None) -> list[str]:
    """`MEMO_*` env vars set but NOT in the registry (possible typos).

    Excludes storage/model vars owned by config.py.
    """
    src = os.environ if env is None else env
    owned = {
        "MEMO_DATA_DIR",
        "MEMO_STATE_DIR",
        "MEMO_VAULT_PATH",
        "MEMO_MEMORY_SUBDIR",
        "MEMO_EMBEDDER_MODEL",
        "MEMO_EMBEDDER_REVISION",
        "MEMO_EMBEDDER_DIMS",
        "MEMO_EMBEDDER_BACKEND",
        "MEMO_ST_EMBEDDER_MODEL",
        "MEMO_ST_EMBEDDER_REVISION",
        "MEMO_LLM_MODEL",
        "MEMO_LLM_REVISION",
        "MEMO_HELPER_MODEL",
        "MEMO_HELPER_REVISION",
        "MEMO_RERANKER_MODEL",
        "MEMO_RERANKER_ENABLED",
        "MEMO_RERANKER_REVISION",
        "MEMO_RERANK_FUSION_ALPHA",
        "MEMO_RERANK_INPUT_K",
        "MEMO_MAX_CONTENT_CHARS",
        "MEMO_SEARCH_DEFAULT_LIMIT",
        "MEMO_CONFIG_DIR",
        "MEMO_CONFIG_FILE",
        # Credential consumed by http_auth.py. Keep it out of the behavioral
        # registry so `memo config show` / active_flags never expose its value.
        "MEMO_HTTP_API_TOKEN",
        # Runtime shim/control vars. These are exported between wrapper processes
        # for IPC/idempotency, not user-configurable MEMO_* knobs.
        "MEMO_AGENT_TTY",
        "MEMO_CODEX_BADGE_SHOWN",
        "MEMO_STARTUP_BANNER_SHOWN",
    }
    return sorted(k for k in src if k.startswith("MEMO_") and k not in REGISTRY and k not in owned)


# Typed vars owned by config.py (in the `owned` set above, NOT in REGISTRY)
# whose raw env strings Config.from_env() feeds straight into pydantic. These
# validation-only specs mirror the Config field types/bounds so `memo config
# validate` rejects a garbage value (e.g. MEMO_EMBEDDER_DIMS=10z4) instead of
# reporting green and then hard-crashing every command in Config.from_env().
# Defaults/groups here are unused — config.py owns the real defaults.
_CONFIG_OWNED_TYPED_SPECS: tuple[FlagSpec, ...] = (
    _spec("MEMO_EMBEDDER_DIMS", "int", 1024, "config", "config.py-owned", min_val=2),
    _spec("MEMO_MAX_CONTENT_CHARS", "int", 64_000, "config", "config.py-owned", min_val=1),
    _spec(
        "MEMO_SEARCH_DEFAULT_LIMIT", "int", 10, "config", "config.py-owned", min_val=1, max_val=100
    ),
    _spec("MEMO_RERANK_INPUT_K", "int", 30, "config", "config.py-owned", min_val=1, max_val=200),
    _spec(
        "MEMO_RERANK_FUSION_ALPHA",
        "float",
        0.7,
        "config",
        "config.py-owned",
        min_val=0.0,
        max_val=1.0,
    ),
    _spec("MEMO_RERANKER_ENABLED", "bool", True, "config", "config.py-owned"),
)


def validate(env: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Parse every set flag; return a list of problems (empty = all good).

    Each problem is {flag, value, error}.
    """
    src = os.environ if env is None else env
    problems: list[dict[str, str]] = []
    for name, spec in REGISTRY.items():
        raw = src.get(name)
        if raw is None or raw == "":
            continue
        try:
            _coerce(spec, raw)
        except ValueError as exc:
            problems.append({"flag": name, "value": raw, "error": str(exc)})
    # config.py-owned typed vars are excluded from REGISTRY, but a garbage
    # value still hard-crashes Config.from_env() — parse them here too so
    # `memo config validate` doesn't give a false green.
    for spec in _CONFIG_OWNED_TYPED_SPECS:
        raw = src.get(spec.name)
        if raw is None or raw == "":
            continue
        try:
            _coerce(spec, raw)
        except ValueError as exc:
            problems.append({"flag": spec.name, "value": raw, "error": str(exc)})
    # MEMO_MODEL_PROFILE is a plain str spec, so _coerce can't reject an invalid
    # profile — but Config.from_env() raises on one. Check the choices here so
    # `memo config validate` doesn't give a false green that later hard-crashes.
    profile = src.get("MEMO_MODEL_PROFILE")
    if profile and profile.strip().lower() not in {"light", "balanced", "quality"}:
        problems.append(
            {
                "flag": "MEMO_MODEL_PROFILE",
                "value": profile,
                "error": "must be one of: light, balanced, quality (or empty)",
            }
        )
    from memo.config_md import validate_markdown_config

    for problem in validate_markdown_config(src):
        problems.append(
            {
                "flag": problem.key or problem.file,
                "value": problem.value,
                "error": problem.error,
            }
        )
    for var in unknown_memo_vars(env):
        problems.append(
            {"flag": var, "value": src[var], "error": "unknown MEMO_* var (typo? not in registry)"}
        )
    return problems
