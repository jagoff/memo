from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from memo.cli import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "memo"

# Foundation layer: building blocks that everything else composes. They must
# not import any *other* memo module, which keeps them free of import cycles
# and safe to reuse anywhere (incl. the god-modules memory.py / cli.py).
# (config.py is foundation too but lazily imports memo.setup for file IO, so
# it's intentionally excluded from this strict leaf set.)
FOUNDATION_MODULES = ["util", "store", "embedder", "graph"]

# Pure-stdlib leaves at the very bottom of the stack. Foundation modules may
# depend on these (they import nothing from memo, so no cycle is possible).
PURE_LEAF_MODULES = {"util", "mlx_gpu"}


def _memo_imports(module_file: Path) -> set[str]:
    """Top-level memo submodules imported by `module_file` (via AST)."""
    tree = ast.parse(module_file.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("memo"):
            parts = node.module.split(".")
            if len(parts) >= 2:
                found.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "memo" and len(parts) >= 2:
                    found.add(parts[1])
    return found


def _module_imports(module: str) -> set[str]:
    """memo submodules imported by `module`, whether it's a single-file
    module (`SRC / f"{module}.py"`) or a package directory (`SRC / module`).
    For a package, unions `_memo_imports` over every `*.py` inside it."""
    single = SRC / f"{module}.py"
    if single.exists():
        return _memo_imports(single)
    pkg = SRC / module
    if pkg.is_dir():
        found: set[str] = set()
        for py in sorted(pkg.glob("*.py")):
            found |= _memo_imports(py)
        return found
    return _memo_imports(single)


def test_brain_like_cli_groups_are_not_public() -> None:
    """Memo exposes corpus primitives; Synapse owns orchestration surfaces."""
    forbidden = {"agent", "cognitive", "federation", "lifecycle", "suggest"}

    assert forbidden.isdisjoint(cli.commands)


def test_brain_like_mcp_tools_are_not_registered() -> None:
    # MCP tools live across server.py + server_*.py (registered via @server.tool()
    # def memo_*) and mcp_tools.py (ToolSpec name="memo_*"). Scan all of them so the
    # guard keeps biting after the monolith split + the memory_*->memo_* rename.
    memo_dir = REPO_ROOT / "src" / "memo"
    sources = [memo_dir / "server.py", memo_dir / "mcp_tools.py", *sorted(memo_dir.glob("server_*.py"))]
    forbidden = ("agent", "cognitive", "federation", "lifecycle", "suggest")

    for path in sources:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for prefix in forbidden:
            esc = re.escape(prefix)
            assert re.search(rf"\bdef memo_{esc}", text) is None, (
                f"brain-like MCP tool memo_{prefix}* defined in {path.name}"
            )
            assert re.search(rf'name="memo_{esc}', text) is None, (
                f"brain-like MCP tool memo_{prefix}* registered in {path.name}"
            )


def test_repo_index_does_not_write_memflow_receipts() -> None:
    source = (REPO_ROOT / "src" / "memo" / "cli.py").read_text(encoding="utf-8")

    assert "memflow_receipt" not in source
    assert "MEMO_MEMFLOW" not in source
    assert "--no-memflow-receipt" not in source


@pytest.mark.parametrize("module", FOUNDATION_MODULES)
def test_foundation_modules_import_no_other_memo_module(module: str) -> None:
    """Foundation modules stay leaf-level — no memo->memo imports, no cycles."""
    imports = _module_imports(module)
    imports.discard(module)
    imports -= PURE_LEAF_MODULES  # depending on a pure-stdlib leaf can't cycle
    assert imports == set(), (
        f"{module}.py must not import other memo modules, found: {sorted(imports)}"
    )


def test_store_never_imports_memory() -> None:
    """The store is below the Memory API; importing it back would be a cycle."""
    assert "memory" not in _module_imports("store")


def test_memory_write_ops_uses_store_public_api() -> None:
    """The save path must not reach through VecStore's private connection."""
    source = (SRC / "memory" / "write_ops.py").read_text(encoding="utf-8")

    assert "store._conn" not in source


def test_util_is_pure_stdlib_leaf() -> None:
    """memo.util is the bottom of the stack — it depends on nothing in memo."""
    assert _memo_imports(SRC / "util.py") == set()


def test_mlx_gpu_is_pure_stdlib_leaf() -> None:
    """memo.mlx_gpu is a bottom-of-stack GPU-serialization leaf (threading
    only) — it must import nothing from memo so every MLX caller can depend
    on it without risking a cycle."""
    assert _memo_imports(SRC / "mlx_gpu.py") == set()


# Optional Memory subsystems that MUST stay behind lazy @property accessors —
# constructing Memory() must not eagerly build any of them (cold-start cost +
# import-cycle risk). Guards the god-object's 3b decomposition contract.
LAZY_SUBSYSTEMS = [
    "temporal", "consolidator", "contradict_store", "contradict_scanner",
    "navigator", "contextual", "crossref", "link_suggester", "lifecycle",
    "versioning", "query_composer", "backup",
    "sync", "analytics", "dashboard",
    "import_export", "multimodal", "collaborative",
]


@pytest.mark.parametrize("name", LAZY_SUBSYSTEMS)
def test_subsystem_is_a_property_not_eager_attr(name: str) -> None:
    """Each optional subsystem is a property descriptor on Memory — so it is
    only built on access, never as a plain __init__ attribute."""
    from memo.memory import Memory

    attr = getattr(Memory, name, None)
    assert isinstance(attr, property), f"Memory.{name} must stay a lazy @property"


def test_constructing_memory_builds_no_subsystem(mock_memory) -> None:
    """A freshly built Memory has every cached subsystem backing field unset —
    construction is cheap and triggers no subsystem cold-start."""
    for name in LAZY_SUBSYSTEMS:
        backing = f"_{name}"
        if hasattr(mock_memory, backing):
            assert getattr(mock_memory, backing) is None, (
                f"Memory.__init__ eagerly built {name} (backing {backing} not None)"
            )


def test_lazy_property_caches_after_access(mock_memory) -> None:
    """Accessing a cached lazy property builds it once and memoizes."""
    assert mock_memory._temporal is None
    first = mock_memory.temporal
    assert mock_memory._temporal is first
    assert mock_memory.temporal is first


def test_optional_memory_capabilities_live_in_registry(mock_memory) -> None:
    """Experimental/advanced Memory subsystems are declared in one registry."""
    from memo.memory.capabilities import OPTIONAL_CAPABILITIES

    expected = {
        "analytics",
        "backup",
        "collaborative",
        "dashboard",
        "import_export",
        "lifecycle",
        "multimodal",
        "query_composer",
        "sync",
        "versioning",
    }

    assert expected.issubset(OPTIONAL_CAPABILITIES)
    assert mock_memory.capability("analytics") is mock_memory.analytics
    assert mock_memory.capability("lifecycle") is mock_memory.lifecycle


def test_behavior_flags_are_not_read_directly_from_environ() -> None:
    """Behavioral MEMO_* reads go through memo.flags, not ad-hoc env parsing."""
    allowed_modules = {
        SRC / "config.py",
        SRC / "flags.py",
        SRC / "flags_base.py",
        SRC / "flags_behavior.py",
        SRC / "flags_ingest.py",
        SRC / "flags_misc.py",
        SRC / "flags_recall.py",
        SRC / "flags_search.py",
    }
    config_owned = {
        "MEMO_CONFIG_FILE",
        "MEMO_DATA_DIR",
        "MEMO_EMBEDDER_DIMS",
        "MEMO_EMBEDDER_MODEL",
        "MEMO_HELPER_MODEL",
        "MEMO_LLM_MODEL",
        "MEMO_MAX_CONTENT_CHARS",
        "MEMO_MEMORY_SUBDIR",
        "MEMO_MODEL_PROFILE",
        "MEMO_RERANKER_ENABLED",
        "MEMO_RERANKER_MODEL",
        "MEMO_RERANKER_REVISION",
        "MEMO_RERANK_FUSION_ALPHA",
        "MEMO_RERANK_INPUT_K",
        "MEMO_SEARCH_DEFAULT_LIMIT",
        "MEMO_SINGLE_DB",
        "MEMO_STATE_DIR",
        "MEMO_VAULT_PATH",
    }
    pure_leaf_owned = {
        SRC / "embedder.py": {"MEMO_QUERY_CACHE_SIZE"},
        SRC / "mlx_gpu.py": {"MEMO_GPU_LOCK_PATH", "MEMO_GPU_XPROC_LOCK"},
        SRC / "store" / "schema.py": {"MEMO_SKIP_MODEL_VERSION_CHECK"},
        # store/queries.py is a foundation module. Tantivy is kept here as an
        # operational storage switch; soft-delete is registered and read through
        # memo.flags.
        SRC / "store" / "queries.py": {"MEMO_TANTIVY_ENABLED"},
        # MEMO_AGENT_TTY is set by the shim, not user-configurable; read here for IPC.
        SRC / "cli_session.py": {"MEMO_AGENT_TTY"},
        # autoupdate reads directly for the setdefault pattern (env check before flag default)
        SRC / "runtime" / "autoupdate.py": {"MEMO_AUTO_UPDATE"},
        # Three-way check (explicit 1 / explicit 0 / auto-detect) cannot use flag_bool
        # because it collapses unset → False; raw env read is the only way to distinguish
        # "not set" from "explicitly off".
        SRC / "memory" / "facade.py": {"MEMO_EMBEDDER_VIA_DAEMON"},
    }
    violations: list[str] = []

    for path in sorted(SRC.rglob("*.py")):
        if path in allowed_modules:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_env_get = (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id in {"os", "_os", "_os_min"}
            )
            is_getenv = (
                isinstance(func, ast.Attribute)
                and func.attr == "getenv"
                and isinstance(func.value, ast.Name)
                and func.value.id in {"os", "_os", "_os_min"}
            )
            if (
                (is_env_get or is_getenv)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.startswith("MEMO_")
                and node.args[0].value not in config_owned
                and node.args[0].value not in pure_leaf_owned.get(path, set())
            ):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.args[0].value}")

    assert violations == []
