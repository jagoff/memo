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
    ("codegraph_loader.py", "MEMO_CODEGRAPH_DISCOVERY"),
    ("codegraph_loader.py", "MEMO_CODEGRAPH_MAX_EDGES"),
    ("codegraph_loader.py", "MEMO_CODEGRAPH_DB"),
    ("setup/config_io.py", "MEMO_CONFIG_FILE"),
    ("embed_protocol.py", "MEMO_STATE_DIR"),
    ("embedder.py", "MEMO_QUERY_CACHE_SIZE"),
    ("cli.py", "MEMO_DATA_DIR"),
    ("cli.py", "MEMO_VAULT_PATH"),
    ("cli.py", "MEMO_MEMORY_SUBDIR"),
}


# The files whose broad catches are classified individually (see
# BROAD_EXCEPTION_ALLOWED). For these, lexical classification is the gate;
# scripts/quality_gate.py's per-file integer budget deliberately counts zero,
# so classifying a new fail-open site is ONE edit, not two.
BROAD_EXCEPTION_TARGET_FILES: frozenset[str] = frozenset(
    {
        "recall_logic.py",
        "memory/write_ops.py",
        "cli_recall_hook.py",
        "store/queries.py",
    }
)


# First sprint only classifies high-risk target files. These stable lexical
# identifiers are a baseline inventory, not blanket approval for future sites.
BROAD_EXCEPTION_ALLOWED: set[tuple[str, str, int]] = {
    # Negative Recall ⛔ pass: the anti-memory retrieval + block render are
    # fail-open — any store/embed/format error degrades to no ⛔ block and must
    # never break the recall payload or blow the 5s hook budget.
    ("recall_logic.py", "_negative_recall_hits", 1),
    ("recall_logic.py", "_negative_recall_block", 1),
    # Code-citation lines (MEMO_RECALL_CODE_REFS_ENABLED): verification is
    # fail-open — a codegraph open/lookup error degrades the ref line to
    # '(no verificado)' and must never break the recall render or the 5s hook
    # budget. (_code_ref_status now delegates to code_intel.ref_status, which
    # catches concrete sqlite errors itself — no broad except left there.)
    ("recall_logic.py", "_code_ref_lines", 1),
    # Graph-cluster recall compaction (MEMO_RECALL_GRAPH_COMPACT): optional
    # token-budget work on the hook hot path. Any projection/store failure
    # degrades to the uncompacted relevant/nudge lists and must never break
    # the recall payload or blow the 5s hook budget.
    ("recall_logic.py", "_apply_graph_compact", 1),
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
    # Proactive urgent rendering is optional hook-hot-path work. Store reads,
    # timestamp parsing, or rendering failures must degrade to no urgent line
    # and must never block the recall payload.
    ("cli_recall_hook.py", "_proactive_urgent_line", 1),
    # MEMO_HIT_DOSSIER batched contradict-pairs lookup: fail-open, degrades to
    # an empty disputed_by map on any store/read error (never blocks recall).
    ("memory/write_ops.py", "_upsert_declared_fact_edges_best_effort", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._presence_bump_save", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._record_graph_entities_from_extra", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._derive_metadata", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._save_core", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._save_core", 2),
    ("memory/write_ops.py", "_WriteOpsMixin._save_core", 3),
    ("memory/write_ops.py", "_WriteOpsMixin._save_core", 4),
    ("memory/write_ops.py", "_WriteOpsMixin._save_core", 5),
    ("memory/write_ops.py", "_WriteOpsMixin._save_core", 6),
    ("memory/write_ops.py", "_WriteOpsMixin._save_core", 7),
    ("memory/write_ops.py", "_WriteOpsMixin._save_core", 8),
    ("memory/write_ops.py", "_WriteOpsMixin._save_core", 9),
    # defer_embed save without topic_key: index failure routes through
    # _save_index_pending (stamps _memo_embed_pending + text-only index) so the
    # .md never silently vanishes — recovery, not a swallow.
    ("memory/write_ops.py", "_WriteOpsMixin._save_core", 10),
    # Post-commit cache capacity enforcement is optional. The durable Markdown
    # and sqlite write already succeeded, so cache backend failure must not
    # turn a committed save into a caller-visible failure.
    ("memory/write_ops.py", "_WriteOpsMixin._save_core", 11),
    # Receipts are explicitly post-commit observability. Failure is logged and
    # cannot invalidate a durable save.
    ("memory/write_ops.py", "_WriteOpsMixin._emit_save_receipt", 1),
    # Topic attachment restores the previous Markdown bytes for every store
    # failure and then re-raises; the broad catch is a rollback boundary, not a
    # swallowed error.
    ("memory/write_ops.py", "_WriteOpsMixin._attach_topic_identity_locked", 1),
    # Recovery translates any sqlite/index implementation failure into the
    # stable StorageError domain contract, preserving the original cause.
    ("memory/write_ops.py", "_WriteOpsMixin._recover_topic_reservation_locked", 1),
    # Cache eviction is post-commit housekeeping. A backend failure is logged
    # but cannot invalidate the Markdown/sqlite write that already succeeded.
    ("memory/write_ops.py", "_WriteOpsMixin._apply_cache_write_policy", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._absorb_into_existing", 1),
    ("memory/write_ops.py", "_WriteOpsMixin._read_body", 1),
    ("recall_logic.py", "_session_context", 1),
    ("recall_logic.py", "knobs_from_flags", 1),
    ("recall_logic.py", "make_vec_cosine._cos", 1),
    ("recall_logic.py", "make_vec_cosine._cos", 2),
    ("recall_logic.py", "fetch_recency_band", 1),
    # World-model projection is optional hook-hot-path enrichment: any
    # kernel/state/projection failure degrades to no kernel section and must
    # never break the recall payload or blow the 5s hook budget.
    ("recall_logic.py", "render_by_format", 1),
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
}


# Post-baseline broad catches may bypass the raw per-file ratchet only through
# this exact lexical inventory. Each site is fail-open by product contract:
#
# - compact briefing: an optional proactive nudge must not break SessionStart;
# - recall urgent line: optional sqlite/routing/rendering work must not block
#   the latency-sensitive UserPromptSubmit hook;
# - mandate sync: a nightly optional pass reports its error in the receipt and
#   must not abort the rest of dream maintenance.
# - repo-search evaluation: each strategy is an isolated trial; one provider
#   failure is recorded in the report and must not suppress the paired trial.
#
# tests/test_dev_audit.py asserts that this inventory is exact and every key
# still resolves to a real ``except Exception`` site, preventing stale or broad
# file-level exclusions.
BROAD_EXCEPTION_RATCHET_EXEMPTIONS: set[tuple[str, str, int]] = {
    ("briefing.py", "proactive_compact_line", 1),
    ("cli_recall_hook.py", "_proactive_urgent_line", 1),
    ("constitution.py", "run_mandate_sync_pass", 1),
    ("repo_eval.py", "evaluate_repo_search", 1),
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
