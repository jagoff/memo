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
    scope: str
    ordinal: int


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
    ("config.py", "MEMO_EMBEDDER_REVISION"),
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


# First sprint only classifies high-risk target files. These stable lexical
# identifiers are a baseline inventory, not blanket approval for future sites.
BROAD_EXCEPTION_ALLOWED: set[tuple[str, str, int]] = {
    ("cli_recall_hook.py", "recall_hook", 1),
    ("cli_recall_hook.py", "recall_hook._bail", 1),
    ("cli_recall_hook.py", "recall_hook", 2),
    ("cli_recall_hook.py", "recall_hook", 3),
    ("cli_recall_hook.py", "recall_hook", 4),
    ("cli_recall_hook.py", "recall_hook", 5),
    ("cli_recall_hook.py", "recall_hook", 6),
    ("cli_recall_hook.py", "recall_hook", 7),
    ("cli_recall_hook.py", "recall_hook", 8),
    ("cli_recall_hook.py", "recall_hook", 9),
    ("cli_recall_hook.py", "recall_hook", 10),
    ("cli_recall_hook.py", "recall_hook", 11),
    ("cli_recall_hook.py", "recall_hook._rank", 1),
    ("cli_recall_hook.py", "recall_hook._stamp_metrics", 1),
    ("cli_recall_hook.py", "recall_hook", 12),
    ("cli_recall_hook.py", "recall_hook", 13),
    ("cli_recall_hook.py", "recall_hook", 14),
    ("cli_recall_hook.py", "recall_hook", 15),
    ("cli_recall_hook.py", "recall_hook", 16),
    ("cli_recall_hook.py", "recall_hook", 17),
    ("cli_recall_hook.py", "recall_hook", 18),
    ("cli_recall_hook.py", "recall_hook", 19),
    ("cli_recall_hook.py", "recall_hook", 20),
    ("cli_recall_hook.py", "recall_hook", 21),
    # Proactive urgent nudge folded into the recall systemMessage: fail-open,
    # a nudge is a nice-to-have and must never block or delay recall.
    ("cli_recall_hook.py", "recall_hook", 22),
    # MEMO_HIT_DOSSIER batched contradict-pairs lookup: fail-open, degrades to
    # an empty disputed_by map on any store/read error (never blocks recall).
    ("memory/write_ops.py", "_upsert_declared_fact_edges_best_effort", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._presence_bump_save", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._record_graph_entities_from_extra", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._derive_metadata", 1),
    ("memory/write_ops.py", "_WriteOpsMixin.save", 1),
    ("memory/write_ops.py", "_WriteOpsMixin.save", 2),
    ("memory/write_ops.py", "_WriteOpsMixin.save", 3),
    ("memory/write_ops.py", "_WriteOpsMixin.save", 4),
    ("memory/write_ops.py", "_WriteOpsMixin.save", 5),
    ("memory/write_ops.py", "_WriteOpsMixin.save", 6),
    ("memory/write_ops.py", "_WriteOpsMixin.save", 7),
    ("memory/write_ops.py", "_WriteOpsMixin.save", 8),
    ("memory/write_ops.py", "_WriteOpsMixin.save", 9),
    # defer_embed save without topic_key: index failure routes through
    # _save_index_pending (stamps _memo_embed_pending + text-only index) so the
    # .md never silently vanishes — recovery, not a swallow.
    ("memory/write_ops.py", "_WriteOpsMixin.save", 10),
    ("memory/write_ops.py", "_WriteOpsMixin._apply_write_policy", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._absorb_into_existing", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._read_body", 1),
    ("recall_logic.py", "_session_context", 1),
    ("recall_logic.py", "knobs_from_flags", 1),
    ("recall_logic.py", "make_vec_cosine._cos", 1),
    ("recall_logic.py", "make_vec_cosine._cos", 2),
    ("recall_logic.py", "fetch_recency_band", 1),
    # Hook hot path: a corrupt/unreadable telemetry log must not block recall;
    # failure keeps the configured per-turn token budget unchanged.
    ("recall_logic.py", "_session_scaled_token_budget", 1),
    ("recall_logic.py", "_recall_logic", 1),
    ("recall_logic.py", "_recall_logic", 2),
    ("recall_logic.py", "_recall_logic", 3),
    ("recall_logic.py", "_recall_logic", 4),
    ("recall_logic.py", "_recall_logic", 5),
    ("recall_logic.py", "_recall_logic._log", 1),
    ("recall_logic.py", "_recall_logic", 6),
    ("recall_logic.py", "_recall_logic", 7),
    ("recall_logic.py", "_recall_logic", 8),
    # MEMO_HIT_DOSSIER batched contradict-pairs lookup: fail-open, degrades to
    # an empty disputed_by map on any store/read error (never blocks recall).
    ("store/queries.py", "_QueriesMixin.upsert", 1),
    ("store/queries.py", "_QueriesMixin.upsert_replacing_path_owner", 1),
    ("store/queries.py", "_QueriesMixin.replace_memory_index", 1),
    ("store/queries.py", "_QueriesMixin.upsert_text_only", 1),
    ("store/queries.py", "_QueriesMixin.update_meta", 1),
    ("store/queries.py", "_QueriesMixin.clear_memory_index", 1),
    ("store/queries.py", "_QueriesMixin.clear_memory_index", 2),
    ("store/queries.py", "_QueriesMixin.delete", 1),
    ("store/queries.py", "_QueriesMixin.delete", 2),
    ("store/queries.py", "_QueriesMixin.delete", 3),
    ("store/queries.py", "_QueriesMixin.hard_delete", 1),
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

        class _BroadExceptionVisitor(ast.NodeVisitor):
            def __init__(self, site_path: Path, site_relpath: str) -> None:
                self._path = site_path
                self._relpath = site_relpath
                self._scope: list[str] = []
                self._ordinals: dict[str, int] = {}

            def _visit_scope(
                self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ) -> None:
                self._scope.append(node.name)
                self.generic_visit(node)
                self._scope.pop()

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self._visit_scope(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_scope(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_scope(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
                if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    scope = ".".join(self._scope) or "<module>"
                    ordinal = self._ordinals.get(scope, 0) + 1
                    self._ordinals[scope] = ordinal
                    out.append(
                        BroadExceptionSite(
                            path=self._path,
                            relpath=self._relpath,
                            line=node.lineno,
                            scope=scope,
                            ordinal=ordinal,
                        )
                    )
                self.generic_visit(node)

        _BroadExceptionVisitor(path, relpath).visit(tree)
    return out
