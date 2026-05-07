"""Configuration — env-var-driven, immutable, validated.

All paths are absolute after construction (resolves `~`, normalises
trailing slashes). Pydantic v2 used for type coercion + boundary
validation; runtime code passes `Config` around as an opaque value
object.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

# Default vault path matches the obsidian-rag convention. Override via
# `MEMO_VAULT_PATH`.
_DEFAULT_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "Notes"
)

# Memory artefacts live under the system-wide `99-AI/memory/` umbrella so
# Obsidian doesn't surface them in normal search and the user's PARA
# folders stay clean. Same convention as obsidian-rag's mem-vault.
DEFAULT_MEMORY_SUBDIR = "04-Archive/99-obsidian-system/99-AI/memory"

# Default state dir mirrors obsidian-rag — keep both projects co-located
# under a single `obsidian-*` namespace in `~/.local/share/`.
_DEFAULT_STATE_DIR = Path.home() / ".local" / "share" / "memo"


class Config(BaseModel):
    """Process-wide configuration.

    Construct via `Config.from_env()` in production. Tests can build a
    `Config(...)` with explicit overrides.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    # ── Storage ──────────────────────────────────────────────────────────
    vault_path: Path = Field(
        default=_DEFAULT_VAULT,
        description="Root of the Obsidian vault. All `MemoryRecord.path` are vault-relative.",
    )
    memory_subdir: str = Field(
        default=DEFAULT_MEMORY_SUBDIR,
        description="Path within the vault (POSIX-style) where memory `.md` files live.",
    )
    state_dir: Path = Field(
        default=_DEFAULT_STATE_DIR,
        description="Where sqlite-vec DB + transient state live. Created if missing.",
    )

    # ── MLX models ───────────────────────────────────────────────────────
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

    @field_validator("vault_path", "state_dir", mode="before")
    @classmethod
    def _expand(cls, v):
        if v is None:
            return v
        return Path(os.path.expandvars(str(v))).expanduser().resolve()

    # ── Derived paths ────────────────────────────────────────────────────

    @property
    def memory_dir(self) -> Path:
        """Absolute path of the memory folder inside the vault."""
        return self.vault_path / self.memory_subdir

    @property
    def db_path(self) -> Path:
        """sqlite-vec DB file path. Single file, single writer."""
        return self.state_dir / "memvec.db"

    @property
    def history_db(self) -> Path:
        """History/audit DB (separate file so vec writes don't share WAL)."""
        return self.state_dir / "history.db"

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, **overrides) -> Config:
        """Build a `Config` from `MEMO_*` env vars, with optional explicit overrides."""
        env_to_field = {
            "MEMO_VAULT_PATH": "vault_path",
            "MEMO_MEMORY_SUBDIR": "memory_subdir",
            "MEMO_STATE_DIR": "state_dir",
            "MEMO_LLM_MODEL": "llm_model",
            "MEMO_HELPER_MODEL": "helper_model",
            "MEMO_EMBEDDER_MODEL": "embedder_model",
            "MEMO_EMBEDDER_DIMS": "embedder_dims",
            "MEMO_MAX_CONTENT_CHARS": "max_content_chars",
            "MEMO_SEARCH_DEFAULT_LIMIT": "search_default_limit",
        }
        kwargs: dict = {}
        for env_key, field in env_to_field.items():
            val = os.environ.get(env_key)
            if val is None or val == "":
                continue
            kwargs[field] = val
        kwargs.update(overrides)
        return cls(**kwargs)

    def ensure_dirs(self) -> None:
        """Create state + memory dirs if missing. Vault root must already exist."""
        if not self.vault_path.is_dir():
            raise RuntimeError(
                f"Vault path does not exist: {self.vault_path}. "
                f"Set MEMO_VAULT_PATH or pass `vault_path=...`."
            )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
