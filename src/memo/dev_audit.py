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
    ("cli_recall_hook.py", 407),
    ("cli_recall_hook.py", 471),
    ("cli_recall_hook.py", 492),
    ("cli_recall_hook.py", 526),
    ("cli_recall_hook.py", 565),
    ("cli_recall_hook.py", 599),
    ("cli_recall_hook.py", 613),
    ("cli_recall_hook.py", 661),
    ("cli_recall_hook.py", 685),
    ("cli_recall_hook.py", 703),
    ("cli_recall_hook.py", 722),
    ("cli_recall_hook.py", 729),
    # MEMO_HIT_DOSSIER batched contradict-pairs lookup: fail-open, degrades to
    # an empty disputed_by map on any store/read error (never blocks recall).
    ("memory/write_ops.py", 121),
    ("memory/write_ops.py", 208),
    ("memory/write_ops.py", 228),
    ("memory/write_ops.py", 260),
    ("memory/write_ops.py", 442),
    ("memory/write_ops.py", 520),
    ("memory/write_ops.py", 542),
    ("memory/write_ops.py", 560),
    ("memory/write_ops.py", 596),
    ("memory/write_ops.py", 677),
    ("memory/write_ops.py", 858),
    ("memory/write_ops.py", 915),
    ("memory/write_ops.py", 926),
    ("memory/write_ops.py", 1151),
    ("memory/write_ops.py", 1207),
    ("memory/write_ops.py", 1313),
    ("recall_logic.py", 603),
    ("recall_logic.py", 713),
    ("recall_logic.py", 747),
    ("recall_logic.py", 759),
    ("recall_logic.py", 931),
    ("recall_logic.py", 1089),
    ("recall_logic.py", 1237),
    ("recall_logic.py", 1268),
    ("recall_logic.py", 1327),
    ("recall_logic.py", 1392),
    ("recall_logic.py", 1450),
    ("recall_logic.py", 1475),
    ("recall_logic.py", 1488),
    ("recall_logic.py", 1503),
    # MEMO_HIT_DOSSIER batched contradict-pairs lookup: fail-open, degrades to
    # an empty disputed_by map on any store/read error (never blocks recall).
    ("store/queries.py", 194),
    ("store/queries.py", 277),
    ("store/queries.py", 421),
    ("store/queries.py", 539),
    ("store/queries.py", 722),
    ("store/queries.py", 1119),
    ("store/queries.py", 1130),
    ("store/queries.py", 1145),
    ("store/queries.py", 1168),
    ("store/queries.py", 1190),
    ("store/queries.py", 1213),
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
