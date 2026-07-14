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
    ("embedder.py", "MEMO_QUERY_CACHE_SIZE"),
    ("cli.py", "MEMO_DATA_DIR"),
    ("cli.py", "MEMO_VAULT_PATH"),
    ("cli.py", "MEMO_MEMORY_SUBDIR"),
}


# First sprint only classifies high-risk target files. These lines are a
# baseline inventory, not blanket approval for future sites.
BROAD_EXCEPTION_ALLOWED: set[tuple[str, int]] = {
    ("cli_recall_hook.py", 107),
    ("cli_recall_hook.py", 132),
    ("cli_recall_hook.py", 162),
    ("cli_recall_hook.py", 176),
    ("cli_recall_hook.py", 212),
    ("cli_recall_hook.py", 236),
    ("cli_recall_hook.py", 276),
    ("cli_recall_hook.py", 279),
    ("cli_recall_hook.py", 290),
    ("cli_recall_hook.py", 335),
    ("cli_recall_hook.py", 357),
    ("cli_recall_hook.py", 370),
    ("cli_recall_hook.py", 399),
    ("cli_recall_hook.py", 462),
    ("cli_recall_hook.py", 480),
    ("cli_recall_hook.py", 514),
    ("cli_recall_hook.py", 529),
    ("cli_recall_hook.py", 563),
    # MEMO_HIT_DOSSIER batched contradict-pairs lookup: fail-open, degrades to
    # an empty disputed_by map on any store/read error (never blocks recall).
    ("cli_recall_hook.py", 577),
    ("cli_recall_hook.py", 625),
    ("cli_recall_hook.py", 649),
    ("cli_recall_hook.py", 667),
    ("cli_recall_hook.py", 686),
    ("cli_recall_hook.py", 693),
    ("memory/write_ops.py", 119),
    ("memory/write_ops.py", 159),
    ("memory/write_ops.py", 179),
    ("memory/write_ops.py", 211),
    ("memory/write_ops.py", 393),
    ("memory/write_ops.py", 471),
    ("memory/write_ops.py", 492),
    ("memory/write_ops.py", 510),
    ("memory/write_ops.py", 543),
    ("memory/write_ops.py", 696),
    ("memory/write_ops.py", 744),
    ("memory/write_ops.py", 755),
    ("memory/write_ops.py", 943),
    ("memory/write_ops.py", 999),
    ("memory/write_ops.py", 1075),
    ("recall_logic.py", 578),
    ("recall_logic.py", 688),
    ("recall_logic.py", 722),
    ("recall_logic.py", 734),
    ("recall_logic.py", 906),
    ("recall_logic.py", 1064),
    ("recall_logic.py", 1212),
    ("recall_logic.py", 1241),
    # MEMO_HIT_DOSSIER batched contradict-pairs lookup: fail-open, degrades to
    # an empty disputed_by map on any store/read error (never blocks recall).
    ("recall_logic.py", 1298),
    ("recall_logic.py", 1357),
    ("recall_logic.py", 1415),
    ("recall_logic.py", 1440),
    ("recall_logic.py", 1453),
    ("recall_logic.py", 1468),
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
