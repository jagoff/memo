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

- `data_dir` (always set): the directory where memo's curated memorias
  (the `.md` files this tool creates) live. Source of record.
- `vault_path` (optional): an Obsidian vault for the cross-vault
  `memo ingest` command. Non-Obsidian users leave it unset.

Pydantic v2 used for type coercion + boundary validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Default `data_dir` — visible in Finder, iCloud-syncable. The picker
# in `memo init` lets users pick a different location (Obsidian vault,
# custom path) on first run.
_DEFAULT_DATA_DIR = Path.home() / "Documents" / "memo"

# Vault system-folder layout — single source of truth. memo's own memory
# subtree and the contacts notes live under `<SYSTEM_DIR>/...` inside an
# Obsidian vault. Rename the whole subtree via MEMO_VAULT_SYSTEM_DIR or by
# editing the one default below — don't scatter the folder name as literals.
SYSTEM_DIR = os.environ.get("MEMO_VAULT_SYSTEM_DIR", "Obsidian")
AI_SUBDIR = f"{SYSTEM_DIR}/AI"
CONTACTS_SUBDIR = f"{SYSTEM_DIR}/Contacts"

# Default state dir — sqlite indexes + transient state. Separate from
# `data_dir` because: (a) state is rebuildable from `.md` files via
# `memo reindex`; (b) XDG-style location signals "managed cache" so
# users don't accidentally back it up alongside their notes.
_DEFAULT_STATE_DIR = Path.home() / ".local" / "share" / "memo"

MODEL_PROFILES: dict[str, dict[str, object]] = {
    # Lowest operational footprint: semantic search still works, but hybrid
    # search skips the cross-encoder so recall-hook latency stays predictable.
    "light": {
        "llm_model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "helper_model": "mlx-community/Qwen3-4B-4bit",
        "embedder_model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        "embedder_dims": 1024,
        "reranker_enabled": False,
    },
    # Default: small embedder + reranker. Good quality/latency balance for a
    # local MCP memory store whose hooks may run on every prompt.
    "balanced": {
        "llm_model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "helper_model": "mlx-community/Qwen3-4B-4bit",
        "embedder_model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        "embedder_dims": 1024,
        "reranker_enabled": True,
        "reranker_model": "mku64/Qwen3-Reranker-0.6B-mlx-8Bit",
    },
    # Higher retrieval quality. Requires a full reindex because the 4B
    # embedder emits 2560-dim vectors instead of 1024.
    "quality": {
        "llm_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510",
        "helper_model": "mlx-community/Qwen3-4B-4bit",
        "embedder_model": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
        "embedder_dims": 2560,
        "reranker_enabled": True,
        "reranker_model": "mku64/Qwen3-Reranker-0.6B-mlx-8Bit",
    },
}


class Config(BaseModel):
    """Process-wide configuration.

    Construct via `Config.from_env()` in production. Tests can build a
    `Config(...)` with explicit overrides.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    # ── Storage ──────────────────────────────────────────────────────────
    data_dir: Path = Field(
        default=_DEFAULT_DATA_DIR,
        description=(
            "Directory where memo's curated memoria `.md` files live. "
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

    # ── MLX models ───────────────────────────────────────────────────────
    model_profile: str = Field(
        default="balanced",
        description=(
            "Named model bundle. `light` lowers latency, `balanced` is the "
            "default, `quality` uses a larger embedder and requires reindex."
        ),
    )
    llm_model: str = Field(
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        description="HF id of the chat model used for synthesis / consolidation.",
    )
    helper_model: str = Field(
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        description=(
            "Smaller model used for deterministic helper tasks (tag suggestion, "
            "title extraction, dedup). `temperature=0`, `seed=42` enforced."
        ),
    )
    embedder_model: str = Field(
        default="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        description="HF id of the embedder. Must yield 1024-dim L2-normalised vectors.",
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

    # ── Reranker ─────────────────────────────────────────────────────────
    # Cross-encoder applied AFTER hybrid retrieval when enabled. Lifts
    # MRR materially on diffuse queries at the cost of one forward pass
    # per candidate. Same Qwen3 family as the embedder so no extra
    # tokenizer/architecture surface is added.
    reranker_enabled: bool = Field(
        default=True,
        description=(
            "When True, hybrid-mode searches fetch a wider candidate "
            "set and rerank via the cross-encoder. Disable to skip the "
            "extra forward pass — useful on Linux/CI or for benchmarks."
        ),
    )
    reranker_model: str = Field(
        default="mku64/Qwen3-Reranker-0.6B-mlx-8Bit",
        description=(
            "HF id of the MLX-quantised reranker. Apache 2.0 by default "
            "(Qwen3-Reranker family). For higher recall at higher cost: "
            "`vserifsaglam/Qwen3-Reranker-4B-4bit-MLX`."
        ),
    )
    reranker_revision: str | None = Field(
        default=None,
        description=(
            "Optional Hugging Face commit hash or revision for the reranker. "
            "Set via MEMO_RERANKER_REVISION to pin user-hosted model repos."
        ),
    )
    rerank_input_k: int = Field(
        default=30,
        ge=1,
        le=200,
        description=(
            "How many hybrid-fusion candidates to feed the reranker. "
            "Larger = better recall but linearly more inference time. "
            "30 fits the 5s recall-hook budget on M3 with the 0.6B "
            "reranker (~600ms warm)."
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
    def memory_dir(self) -> Path:
        """Absolute path of the directory holding memoria `.md` files.

        Always equal to `data_dir`. The property exists so callers don't
        need to track which field is authoritative as the schema evolves.
        """
        return self.data_dir

    @property
    def db_path(self) -> Path:
        """sqlite-vec DB file path. Single file, single writer."""
        return self.state_dir / "memvec.db"

    @property
    def history_db(self) -> Path:
        """History/audit DB (separate file so vec writes don't share WAL)."""
        return self.state_dir / "history.db"

    @property
    def graph_db(self) -> Path:
        """Knowledge-graph DB (entities + entity_memoria edges)."""
        return self.state_dir / "graph.db"

    @property
    def crossref_db(self) -> Path:
        """Cross-reference DB (wikilinks + backlinks)."""
        return self.state_dir / "crossref.db"

    @property
    def contradictions_db(self) -> Path:
        """Sidecar DB for persisted contradiction pairs + triage status."""
        return self.state_dir / "contradictions.db"

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
        storage = file_data.get("storage") or {} if isinstance(file_data, dict) else {}

        kwargs: dict[str, Any] = {}
        # File-level fields. Only keys we recognise; ignore unknown for forward-compat.
        for fkey in ("data_dir", "vault_path", "memory_subdir", "state_dir"):
            if storage.get(fkey):
                kwargs[fkey] = storage[fkey]

        # Step 2: model profile defaults. Individual env vars below
        # intentionally override profile choices.
        profile = str(
            overrides.get("model_profile")
            or os.environ.get("MEMO_MODEL_PROFILE")
            or kwargs.get("model_profile")
            or "balanced"
        ).strip().lower()
        if profile not in MODEL_PROFILES:
            raise ValueError(
                f"Unknown MEMO_MODEL_PROFILE={profile!r}; valid: {sorted(MODEL_PROFILES)}",
            )
        kwargs.update(MODEL_PROFILES[profile])
        kwargs["model_profile"] = profile

        # Step 3: env-var overrides.
        env_to_field = {
            "MEMO_DATA_DIR": "data_dir",
            "MEMO_VAULT_PATH": "vault_path",
            "MEMO_MEMORY_SUBDIR": "memory_subdir",
            "MEMO_STATE_DIR": "state_dir",
            "MEMO_MODEL_PROFILE": "model_profile",
            "MEMO_LLM_MODEL": "llm_model",
            "MEMO_HELPER_MODEL": "helper_model",
            "MEMO_EMBEDDER_MODEL": "embedder_model",
            "MEMO_EMBEDDER_DIMS": "embedder_dims",
            "MEMO_MAX_CONTENT_CHARS": "max_content_chars",
            "MEMO_SEARCH_DEFAULT_LIMIT": "search_default_limit",
            "MEMO_RERANKER_ENABLED": "reranker_enabled",
            "MEMO_RERANKER_MODEL": "reranker_model",
            "MEMO_RERANKER_REVISION": "reranker_revision",
            "MEMO_RERANK_INPUT_K": "rerank_input_k",
            "MEMO_RERANK_FUSION_ALPHA": "rerank_fusion_alpha",
        }
        for env_key, field in env_to_field.items():
            val = os.environ.get(env_key)
            if val is None or val == "":
                continue
            kwargs[field] = val

        # Step 4: legacy back-compat — if data_dir is still unset BUT the
        # legacy pair (vault_path + memory_subdir) is set, derive it.
        # This keeps pre-`memo init` installs working unchanged.
        if "data_dir" not in kwargs:
            vp = kwargs.get("vault_path")
            sd = kwargs.get("memory_subdir")
            if vp and sd:
                kwargs["data_dir"] = str(Path(vp).expanduser() / sd)

        # Step 5: explicit overrides win over everything.
        kwargs.update(overrides)
        cfg = cls(**kwargs)
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
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
