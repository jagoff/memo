"""Auto-tuned MEMO_* params overlay — written by `memo dream tune`, read by
`flags.flag()` with precedence env > overlay > default.

Machine-local, never committed. Deleting the file restores pure defaults. The
overlay holds scalar tuning params (float ranking knobs like
``MEMO_RECALL_MIN_SIM`` plus atomic curated-signal settings such as
``MEMO_GRAPH_SIGNAL_ENABLED`` / ``MEMO_GRAPH_SIGNAL_ALPHA``);
`_meta` carries provenance + the previous values for one-step rollback.
Every value round-trips through ``flag()``'s per-kind coercion, so it is stored
as its native JSON scalar and surfaced to ``flag()`` as a coercible string.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_FILENAME = "tuned_params.json"

# path -> (mtime, param->str-value) — keeps `flag()` off disk on the hot recall path.
_cache: dict[str, tuple[float, dict[str, str]]] = {}

# JSON scalar types the overlay is allowed to carry (bool is a subclass of int,
# so it is covered, but we branch on it first when stringifying for `flag()`).
_SCALAR = (bool, int, float, str)


def overlay_path(state_dir: Path) -> Path:
    return Path(state_dir) / _FILENAME


def read_overlay(state_dir: Path) -> dict[str, Any]:
    """Full overlay document (incl. `_meta`); {} when missing or corrupt."""
    try:
        doc = json.loads(overlay_path(state_dir).read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def params_version(state_dir: Path) -> str:
    """Stable short hash of the active scalar overlay params. Returns ``"base"``
    when the overlay is empty/missing (the identity config). Order-independent,
    so re-serialising the same params never changes the version."""
    params = _scalar_params(read_overlay(Path(state_dir)))
    if not params:
        return "base"
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _scalar_params(doc: dict[str, Any]) -> dict[str, Any]:
    """Every non-`_meta` scalar param, native type preserved. This is the full
    set the overlay carries (floats + bools + strings)."""
    return {k: v for k, v in doc.items() if k != "_meta" and isinstance(v, _SCALAR)}


def _params_only(doc: dict[str, Any]) -> dict[str, float]:
    """Numeric-only params (the float ranking knobs). Kept for callers that do
    float math on the overlay and must ignore boolean/string levers."""
    return {
        k: float(v)
        for k, v in doc.items()
        if k != "_meta" and isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _to_flag_str(v: Any) -> str:
    """Stringify a scalar for `flag()` per-kind coercion. Booleans map to the
    canonical ``1``/``0`` that the bool coercer accepts (``str(True)`` == "True"
    would be rejected)."""
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


# (explicit MEMO_STATE_DIR, cwd) -> resolved state dir. Process-lifetime cache:
# the fallback chain below touches the markdown config + disk, and `flag()` is
# on the recall hot path.
_state_dir_cache: dict[tuple[str, str], str] = {}


def _resolve_state_dir(src: Mapping[str, str]) -> str:
    """State dir the overlay lives in. Mirrors ``Config.from_env``'s fallback
    chain (env > markdown storage config > repo-cwd dev default > XDG default)
    WITHOUT building a Config: ``flag()`` sits underneath ``Config.from_env``,
    so calling it here would recurse. Without this chain the overlay was only
    ever applied when ``MEMO_STATE_DIR`` was explicitly exported — daemons and
    plain CLI runs silently ignored every tuner/graduation result."""
    explicit = src.get("MEMO_STATE_DIR")
    if explicit:
        return explicit
    if src is not os.environ:
        # A custom env mapping (tests, callers pinning a hermetic env) opts
        # out of the machine-level fallback chain: without an explicit
        # MEMO_STATE_DIR the overlay is treated as absent.
        return ""
    key = ("", str(Path.cwd()))
    cached = _state_dir_cache.get(key)
    if cached is not None:
        return cached
    resolved = _resolve_state_dir_uncached(src, None)
    _state_dir_cache[key] = resolved
    return resolved


def _resolve_state_dir_uncached(src: Mapping[str, str], explicit: str | None) -> str:
    if explicit:
        return explicit
    storage_md: dict[str, Any] = {}
    try:
        from memo.config_md import load_values

        storage_md = {k: v.value for k, v in load_values(src).items() if k.startswith("storage.")}
    except (ImportError, OSError):  # unavailable/unreadable config — use defaults
        storage_md = {}
    md_sd = storage_md.get("storage.state_dir")
    if md_sd:
        return str(md_sd)
    # Legacy TOML `[storage]` section. Config.from_env folds it into the same
    # has_storage_config gate (via `_file_config_values`) and honors its
    # state_dir. Mirror both — otherwise a repo checkout carrying a legacy
    # config.toml would wrongly fall to ./.memo-state and read the overlay from
    # a dir Config never uses.
    toml_storage: dict[str, Any] = {}
    try:
        from memo.setup.config_io import load_config_file

        file_data = load_config_file()
        if isinstance(file_data, dict):
            raw = file_data.get("storage")
            toml_storage = raw if isinstance(raw, dict) else {}
    except (ImportError, OSError):  # unavailable/unreadable config — use defaults
        toml_storage = {}
    toml_sd = toml_storage.get("state_dir")
    if toml_sd:
        return str(toml_sd)
    # Config.from_env: `has_legacy = "MEMO_VAULT_PATH" in os.environ or
    # os.environ.get("MEMO_MEMORY_SUBDIR")` — an EXPORTED-but-empty
    # MEMO_VAULT_PATH still counts (key presence, not truthiness). `src` is
    # `os.environ` here (the hermeticity gate in `_resolve_state_dir` already
    # rejected custom mappings), so mirror it exactly.
    has_legacy = "MEMO_VAULT_PATH" in src or bool(src.get("MEMO_MEMORY_SUBDIR"))
    has_storage_config = bool(storage_md) or bool(toml_storage)
    if (
        not has_legacy
        and not has_storage_config
        and (Path.cwd() / "src" / "memo" / "__init__.py").is_file()
    ):
        return str(Path.cwd() / ".memo-state")
    try:
        from memo.config import _DEFAULT_STATE_DIR
    except ImportError:
        # flag() invoked while memo.config is still importing (circular) —
        # fall back to the same XDG default the constant holds.
        return str(Path.home() / ".local" / "share" / "memo")
    return str(_DEFAULT_STATE_DIR)


def overlay_values(src: Mapping[str, str]) -> dict[str, str]:
    """param-name -> string value, for `flag()` resolution. Resolved from
    ``src["MEMO_STATE_DIR"]`` with the ``Config.from_env`` fallback chain,
    mtime-cached. {} when the overlay file is missing/corrupt."""
    sd = _resolve_state_dir(src)
    if not sd:
        return {}
    p = overlay_path(Path(sd))
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    cached = _cache.get(str(p))
    if cached and cached[0] == mtime:
        return cached[1]
    vals = {k: _to_flag_str(v) for k, v in _scalar_params(read_overlay(Path(sd))).items()}
    _cache[str(p)] = (mtime, vals)
    return vals


def write_overlay(state_dir: Path, params: dict[str, Any], meta: dict[str, Any]) -> None:
    """Write the overlay, stashing the current params under ``_meta.prev``.

    ``params`` may hold any JSON scalar (float knobs and/or boolean/string
    levers); the previous scalar set is preserved for one-step rollback.
    """
    sd = Path(state_dir)
    sd.mkdir(parents=True, exist_ok=True)
    prev = _scalar_params(read_overlay(sd))
    doc: dict[str, Any] = dict(params)
    doc["_meta"] = {**meta, "prev": prev}
    overlay_path(sd).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    _cache.pop(str(overlay_path(sd)), None)


def rollback_overlay(state_dir: Path) -> dict[str, Any] | None:
    """Restore ``_meta.prev``; returns the restored params, or None if none.
    Native scalar types are preserved (a rolled-back boolean stays a boolean)."""
    sd = Path(state_dir)
    prev = (read_overlay(sd).get("_meta") or {}).get("prev")
    if not isinstance(prev, dict) or not prev:
        return None
    restored = dict(_scalar_params(prev))
    write_overlay(sd, restored, {"set_by": "rollback"})
    return restored


def pin_prev_to_current(state_dir: Path) -> None:
    """Point ``_meta.prev`` at the CURRENT scalar params, making a subsequent
    one-step :func:`rollback_overlay` a safe no-op. Used after an online
    proof-loop revert so a later offline rollback-guard cannot resurrect the
    config the online loop just reverted away."""
    sd = Path(state_dir)
    cur = _scalar_params(read_overlay(sd))
    write_overlay(sd, cur, {"set_by": "pin-prev"})
