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
    ("cli_recall_hook.py", 421),
    ("cli_recall_hook.py", 439),
    ("cli_recall_hook.py", 473),
    ("cli_recall_hook.py", 488),
    ("cli_recall_hook.py", 522),
    ("cli_recall_hook.py", 562),
    ("cli_recall_hook.py", 579),
    ("cli_recall_hook.py", 590),
    ("cli_recall_hook.py", 597),
    ("memory/write_ops.py", 111),
    ("memory/write_ops.py", 151),
    ("memory/write_ops.py", 171),
    ("memory/write_ops.py", 201),
    ("memory/write_ops.py", 383),
    ("memory/write_ops.py", 453),
    ("memory/write_ops.py", 474),
    ("memory/write_ops.py", 492),
    ("memory/write_ops.py", 525),
    ("memory/write_ops.py", 678),
    ("memory/write_ops.py", 726),
    ("memory/write_ops.py", 737),
    ("memory/write_ops.py", 925),
    ("memory/write_ops.py", 981),
    ("memory/write_ops.py", 1057),
    ("recall_logic.py", 457),
    ("recall_logic.py", 566),
    ("recall_logic.py", 599),
    ("recall_logic.py", 611),
    ("recall_logic.py", 774),
    ("recall_logic.py", 895),
    ("recall_logic.py", 1028),
    ("recall_logic.py", 1056),
    ("recall_logic.py", 1079),
    ("recall_logic.py", 1180),
    ("recall_logic.py", 1198),
    ("recall_logic.py", 1206),
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
