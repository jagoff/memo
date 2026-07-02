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
from typing import Any

# Re-export base types so existing `from memo.flags import FlagSpec` imports still work.
from memo.flags_base import _FALSE, _TRUE, FlagKind, FlagSpec, _spec  # noqa: F401
from memo.flags_behavior import SPECS as _behavior_specs
from memo.flags_ingest import SPECS as _ingest_specs
from memo.flags_misc import SPECS as _misc_specs
from memo.flags_recall import SPECS as _recall_specs
from memo.flags_search import SPECS as _search_specs

_log = logging.getLogger(__name__)

# ── Registry ────────────────────────────────────────────────────────────────
_SPECS: tuple[FlagSpec, ...] = (
    _recall_specs + _search_specs + _behavior_specs + _ingest_specs + _misc_specs
)

REGISTRY: dict[str, FlagSpec] = {s.name: s for s in _SPECS}


def _coerce(spec: FlagSpec, raw: str) -> Any:
    """Parse `raw` per the spec's kind. Raises ValueError on bad input."""
    if spec.kind == "bool":
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"expected a boolean (1/0/true/false), got {raw!r}")
    if spec.kind == "int":
        vi = int(raw.strip())
        if spec.min_val is not None and vi < spec.min_val:
            raise ValueError(f"{spec.name} must be >= {spec.min_val}, got {vi}")
        if spec.max_val is not None and vi > spec.max_val:
            raise ValueError(f"{spec.name} must be <= {spec.max_val}, got {vi}")
        return vi
    if spec.kind == "float":
        vf = float(raw.strip())
        if spec.min_val is not None and vf < spec.min_val:
            raise ValueError(f"{spec.name} must be >= {spec.min_val}, got {vf}")
        if spec.max_val is not None and vf > spec.max_val:
            raise ValueError(f"{spec.name} must be <= {spec.max_val}, got {vf}")
        return vf
    return raw  # str


def flag(name: str, *, env: dict[str, str] | None = None) -> Any:
    """Return the typed, parsed value for `name`, or its default if unset.

    Unknown flags raise KeyError — every flag must be registered above.
    """
    spec = REGISTRY[name]
    src = os.environ if env is None else env
    raw = src.get(name)
    if raw is None or raw == "":
        # empty string counts as unset except for str flags whose default is also ""
        if raw == "" and spec.kind == "str" and spec.default == "":
            return ""
        # env unset → consult the auto-tuned overlay (env > overlay > default)
        if raw is None:
            from memo.tuned_overlay import overlay_values

            ov = overlay_values(src)
            if name in ov:
                try:
                    return _coerce(spec, ov[name])
                except ValueError:
                    _log.warning(
                        "invalid overlay value for %s: %r — using default %r",
                        name,
                        ov[name],
                        spec.default,
                    )
        return spec.default
    try:
        return _coerce(spec, raw)
    except ValueError as exc:
        _log.warning(
            "invalid value for %s: %r — using default %r (%s)",
            name,
            raw,
            spec.default,
            exc,
        )
        return spec.default


def flag_bool(name: str, *, env: dict[str, str] | None = None) -> bool:
    return bool(flag(name, env=env))


def flag_int(name: str, *, env: dict[str, str] | None = None) -> int | None:
    v = flag(name, env=env)
    return None if v is None else int(v)


def flag_float(name: str, *, env: dict[str, str] | None = None) -> float | None:
    v = flag(name, env=env)
    return None if v is None else float(v)


def flag_str(name: str, *, env: dict[str, str] | None = None) -> str:
    v = flag(name, env=env)
    return "" if v is None else str(v)


def active_flags(env: dict[str, str] | None = None) -> dict[str, str]:
    """Registered flags currently set (non-empty) in the environment."""
    src = os.environ if env is None else env
    return {n: src[n] for n in REGISTRY if src.get(n)}


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
        "MEMO_EMBEDDER_DIMS",
        "MEMO_EMBEDDER_BACKEND",
        "MEMO_ST_EMBEDDER_MODEL",
        "MEMO_LLM_MODEL",
        "MEMO_HELPER_MODEL",
        "MEMO_RERANKER_MODEL",
        "MEMO_RERANKER_ENABLED",
        "MEMO_RERANKER_REVISION",
        "MEMO_RERANK_FUSION_ALPHA",
        "MEMO_RERANK_INPUT_K",
        "MEMO_MAX_CONTENT_CHARS",
        "MEMO_SEARCH_DEFAULT_LIMIT",
        "MEMO_CONFIG_FILE",
        # Runtime shim/control vars. These are exported between wrapper processes
        # for IPC/idempotency, not user-configurable MEMO_* knobs.
        "MEMO_AGENT_TTY",
        "MEMO_CODEX_BADGE_SHOWN",
        "MEMO_STARTUP_BANNER_SHOWN",
    }
    return sorted(k for k in src if k.startswith("MEMO_") and k not in REGISTRY and k not in owned)


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
    for var in unknown_memo_vars(env):
        problems.append(
            {"flag": var, "value": src[var], "error": "unknown MEMO_* var (typo? not in registry)"}
        )
    return problems
