from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawMemoEnvRead:
    path: Path
    relpath: str
    line: int
    name: str


@dataclass(frozen=True)
class BroadExceptionSite:
    path: Path
    relpath: str
    line: int


# Allowed raw env reads are grouped by exact file and env var. Reasons live in
# docs/engineering/exception-policy.md and in local source comments.
RAW_MEMO_ENV_ALLOWED: set[tuple[str, str]] = {
    ("config.py", "MEMO_MODEL_PROFILE"),
    ("config.py", "MEMO_RERANKER_ENABLED"),
    ("config.py", "MEMO_MEMORIES_IN_VAULT"),
    ("config.py", "MEMO_SINGLE_DB"),
    ("config.py", "MEMO_DATA_DIR"),
    ("config.py", "MEMO_STATE_DIR"),
    ("config.py", "MEMO_VAULT_PATH"),
    ("config.py", "MEMO_MEMORY_SUBDIR"),
    ("config.py", "MEMO_EMBEDDER_MODEL"),
    ("config.py", "MEMO_EMBEDDER_DIMS"),
    ("store/schema.py", "MEMO_SKIP_MODEL_VERSION_CHECK"),
    ("memory/facade.py", "MEMO_EMBEDDER_VIA_DAEMON"),
    ("mlx_gpu.py", "MEMO_GPU_LOCK_PATH"),
    ("mlx_gpu.py", "MEMO_GPU_XPROC_LOCK"),
    ("setup/config_io.py", "MEMO_CONFIG_FILE"),
    ("embed_protocol.py", "MEMO_STATE_DIR"),
    ("runtime/autoupdate.py", "MEMO_AUTO_UPDATE"),
    ("embedder.py", "MEMO_QUERY_CACHE_SIZE"),
    ("cli.py", "MEMO_DATA_DIR"),
    ("cli.py", "MEMO_VAULT_PATH"),
    ("cli.py", "MEMO_MEMORY_SUBDIR"),
}


# First sprint only classifies high-risk target files. These lines are a
# baseline inventory, not blanket approval for future sites.
BROAD_EXCEPTION_ALLOWED: set[tuple[str, int]] = {
    ("cli_recall_hook.py", 85),
    ("cli_recall_hook.py", 110),
    ("cli_recall_hook.py", 140),
    ("cli_recall_hook.py", 154),
    ("cli_recall_hook.py", 190),
    ("cli_recall_hook.py", 214),
    ("cli_recall_hook.py", 250),
    ("cli_recall_hook.py", 253),
    ("cli_recall_hook.py", 264),
    ("cli_recall_hook.py", 320),
    ("cli_recall_hook.py", 342),
    ("cli_recall_hook.py", 355),
    ("cli_recall_hook.py", 384),
    ("cli_recall_hook.py", 435),
    ("cli_recall_hook.py", 453),
    ("cli_recall_hook.py", 487),
    ("cli_recall_hook.py", 502),
    ("cli_recall_hook.py", 536),
    # MEMO_HIT_DOSSIER batched contradict-pairs lookup: fail-open, degrades to
    # an empty disputed_by map on any store/read error (never blocks recall).
    ("cli_recall_hook.py", 550),
    ("cli_recall_hook.py", 593),
    ("cli_recall_hook.py", 615),
    ("cli_recall_hook.py", 633),
    ("cli_recall_hook.py", 652),
    ("cli_recall_hook.py", 659),
    ("memory/write_ops.py", 119),
    ("memory/write_ops.py", 159),
    ("memory/write_ops.py", 179),
    ("memory/write_ops.py", 209),
    ("memory/write_ops.py", 391),
    ("memory/write_ops.py", 469),
    ("memory/write_ops.py", 490),
    ("memory/write_ops.py", 508),
    ("memory/write_ops.py", 541),
    ("memory/write_ops.py", 694),
    ("memory/write_ops.py", 742),
    ("memory/write_ops.py", 753),
    ("memory/write_ops.py", 941),
    ("memory/write_ops.py", 997),
    ("memory/write_ops.py", 1073),
    ("recall_logic.py", 499),
    ("recall_logic.py", 608),
    ("recall_logic.py", 641),
    ("recall_logic.py", 653),
    ("recall_logic.py", 816),
    ("recall_logic.py", 937),
    ("recall_logic.py", 1070),
    ("recall_logic.py", 1098),
    ("recall_logic.py", 1143),
    # MEMO_HIT_DOSSIER batched contradict-pairs lookup: fail-open, degrades to
    # an empty disputed_by map on any store/read error (never blocks recall).
    ("recall_logic.py", 1202),
    ("recall_logic.py", 1259),
    ("recall_logic.py", 1282),
    ("recall_logic.py", 1295),
    ("recall_logic.py", 1310),
    ("store/queries.py", 160),
    ("store/queries.py", 256),
    ("store/queries.py", 405),
    ("store/queries.py", 800),
    ("store/queries.py", 811),
    ("store/queries.py", 826),
    ("store/queries.py", 849),
    ("store/queries.py", 871),
    ("store/queries.py", 894),
}


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _constant_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_os_environ_get(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "get"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "environ"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id in {"os", "_os"}
    )


def find_raw_memo_env_reads(root: Path) -> list[RawMemoEnvRead]:
    out: list[RawMemoEnvRead] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relpath = _rel(path, root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_os_environ_get(node) or not node.args:
                continue
            name = _constant_str(node.args[0])
            if name and name.startswith("MEMO_"):
                out.append(RawMemoEnvRead(path=path, relpath=relpath, line=node.lineno, name=name))
    return out


def find_broad_exception_sites(root: Path) -> list[BroadExceptionSite]:
    out: list[BroadExceptionSite] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relpath = _rel(path, root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                out.append(BroadExceptionSite(path=path, relpath=relpath, line=node.lineno))
    return out
