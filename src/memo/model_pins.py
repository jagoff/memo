"""Immutable Hugging Face model references used by memo.

Remote model repositories are executable supply-chain inputs.  A repository
name alone is mutable, so every shipped model resolves to an exact 40-character
commit SHA before a framework loader sees it.  Local paths remain local and do
not involve Hugging Face.

This module is deliberately stdlib-only: MLX's foundation modules import it on
Linux too, where neither ``mlx`` nor ``mlx_lm`` is importable.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

PINNED_MODEL_REVISIONS: dict[str, str] = {
    "mlx-community/Qwen2.5-7B-Instruct-4bit": ("c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed"),
    "mlx-community/Qwen3-4B-4bit": "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25",
    "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ": ("6c3ae70858513f1a78e9cdca3cae330d9075cd2a"),
    "mku64/Qwen3-Reranker-0.6B-mlx-8Bit": ("ba80418a47fa1c4368a6c2287b0e449904063576"),
    "mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510": (
        "c073725c8ac051eabad9d64f4dcd3019d1072559"
    ),
    "mlx-community/Qwen3-Embedding-4B-4bit-DWQ": ("b5d88f1fe49b50d2ac01b4692ca2d387f14f9c72"),
    "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ": (
        "53bfb233acb2e50f6060c3c5709f23fac547827f"
    ),
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "vserifsaglam/Qwen3-Reranker-4B-4bit-MLX": ("9655b27c01d2ff1c49f7e672a04b70d630161b46"),
}


class ModelPinError(ValueError):
    """A remote model reference is mutable or internally inconsistent."""


@dataclass(frozen=True)
class ModelSpec:
    source: str
    revision: str | None
    is_local: bool

    @property
    def identity(self) -> str:
        if self.is_local or self.revision is None:
            return self.source
        return f"{self.source}@{self.revision}"


def _is_local_path(value: str) -> bool:
    expanded = Path(value).expanduser()
    return (
        expanded.is_absolute()
        or value.startswith(("./", "../", "~/"))
        or expanded.exists()
        or bool(re.match(r"^[A-Za-z]:[\\/]", value))
    )


def default_revision(model: str) -> str | None:
    """Return memo's audited commit for a shipped repository, if known."""
    _source, separator, suffix = model.strip().rpartition("@")
    if separator and _COMMIT_RE.fullmatch(suffix):
        return suffix
    return PINNED_MODEL_REVISIONS.get(model.strip())


def model_spec(model: str, revision: str | None = None) -> ModelSpec:
    """Parse and validate a local path or immutable Hugging Face reference.

    Remote custom models must use either ``repo@<40hex>`` or an explicit exact
    ``revision``.  Shipped repositories inherit their audited revision from
    :data:`PINNED_MODEL_REVISIONS`.
    """
    raw = model.strip()
    if not raw:
        raise ModelPinError("model reference must not be empty")

    if _is_local_path(raw):
        return ModelSpec(str(Path(raw).expanduser()), None, True)

    explicit = revision.strip().lower() if revision else None
    source, separator, inline = raw.rpartition("@")
    if separator:
        if not _COMMIT_RE.fullmatch(inline):
            raise ModelPinError(
                f"remote model {raw!r} must use an exact 40-character commit SHA; "
                "branches, tags, and short hashes are mutable"
            )
        inline = inline.lower()
        if explicit and explicit != inline:
            raise ModelPinError(
                f"remote model {source!r} has conflicting inline and explicit revisions"
            )
        raw = source
        explicit = inline

    selected = explicit or PINNED_MODEL_REVISIONS.get(raw)
    if selected is None or not _COMMIT_RE.fullmatch(selected):
        raise ModelPinError(
            f"remote model {raw!r} requires an exact 40-character commit SHA "
            f"(use {raw}@<sha> or its explicit revision setting)"
        )
    return ModelSpec(raw, selected, False)


def model_identity(model: str, revision: str | None = None) -> str:
    """Return the exact owner identity used by vector/cache metadata."""
    return model_spec(model, revision).identity


def hf_hub_cache_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the Hugging Face hub cache using its documented env precedence."""
    source = os.environ if env is None else env
    if explicit := source.get("HF_HUB_CACHE"):
        return Path(explicit).expanduser()
    if hf_home := source.get("HF_HOME"):
        return Path(hf_home).expanduser() / "hub"
    if xdg_cache := source.get("XDG_CACHE_HOME"):
        return Path(xdg_cache).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_model_snapshot(model: str, revision: str | None = None) -> str:
    """Resolve a remote commit to a local snapshot; pass local paths through."""
    spec = model_spec(model, revision)
    if spec.is_local:
        return spec.source
    from huggingface_hub import snapshot_download

    assert spec.revision is not None
    # `model_spec` accepts only an exact 40-hex commit here. Bandit's call-site
    # heuristic cannot follow that validation, so this is an audited exception.
    return snapshot_download(repo_id=spec.source, revision=spec.revision)  # nosec B615


__all__ = [
    "PINNED_MODEL_REVISIONS",
    "ModelPinError",
    "ModelSpec",
    "default_revision",
    "hf_hub_cache_dir",
    "model_identity",
    "model_spec",
    "resolve_model_snapshot",
]
